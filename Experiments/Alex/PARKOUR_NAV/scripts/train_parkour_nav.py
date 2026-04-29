"""Training entrypoint for the unified parkour-nav policy.

Mirrors SIM_TO_REAL/scripts/train_expert.py structure but with:
  - No --expert_type flag (only one env here)
  - --phase {teacher,student} for two-stage training
  - asymmetric critic routing auto-enabled when cfg has a `critic` ObsGroup

Usage (H100 teacher, Phase 1):
    python scripts/train_parkour_nav.py --phase teacher --headless \
        --num_envs 4096 --max_iterations 8000 --save_interval 100 \
        --max_noise_std 0.5

Usage (H100 student distill, Phase 2):
    python scripts/train_parkour_nav.py --phase student \
        --teacher_ckpt ~/PARKOUR_NAV/logs/rsl_rl/spot_parkour_nav/.../model_8000.pt \
        --headless --num_envs 4096 --max_iterations 6000

Usage (local 32-env smoke):
    python scripts/train_parkour_nav.py --phase teacher --num_envs 32 \
        --max_iterations 5 --no_wandb --headless
"""

# -- 0. Parse args BEFORE any Isaac imports --------------------------------
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Parkour-Nav unified training")
parser.add_argument("--phase", type=str, default="teacher",
                    choices=["teacher", "student"],
                    help="teacher = privileged critic PPO. "
                         "student = distill to proprio-only policy.")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=8000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--no_wandb", action="store_true", default=False)

# LR + noise (same as SIM_TO_REAL)
parser.add_argument("--lr_max", type=float, default=1e-3)
parser.add_argument("--lr_min", type=float, default=1e-5)
parser.add_argument("--warmup_iters", type=int, default=50)
parser.add_argument("--min_noise_std", type=float, default=0.3)
parser.add_argument("--max_noise_std", type=float, default=0.5,
                    help="ALWAYS pass explicitly (Bug #28d)")

# S2R wrapper
parser.add_argument("--action_delay_steps", type=int, default=2,
                    help="Action delay ring buffer steps (2 @ 50Hz = 40ms)")
parser.add_argument("--obs_delay_steps", type=int, default=1,
                    help="Observation delay steps (1 @ 50Hz = 20ms)")
parser.add_argument("--sensor_dropout_rate", type=float, default=0.08,
                    help="Height scan ray dropout rate (parkour-paper 8%)")
parser.add_argument("--sensor_drift_rate", type=float, default=0.002,
                    help="IMU OU-process drift rate")
parser.add_argument("--obs_history_length", type=int, default=10,
                    help="History length N for obs stacking (Cheng 2024)")

# Resume / actor-only resume (from hybrid_nocoach_19999.pt)
parser.add_argument("--resume_path", type=str, default=None,
                    help="Direct path to checkpoint for full resume")
parser.add_argument("--actor_only_resume", type=str, default=None,
                    help="Path to checkpoint for actor-only resume (fresh critic)")
parser.add_argument("--critic_warmup_iters", type=int, default=0,
                    help="Freeze actor for first N iters so fresh 485-input "
                         "critic can fit value function before PPO updates the "
                         "pre-trained actor. Only applies with --actor_only_resume.")

# Phase 2 — student distill
parser.add_argument("--teacher_ckpt", type=str, default=None,
                    help="Required in --phase student")
parser.add_argument("--distill_mode", type=str, default="bc+ppo",
                    choices=["bc", "bc+ppo"],
                    help="bc: pure behavior cloning. bc+ppo: BC warmup then "
                         "switch to on-policy PPO with frozen teacher as "
                         "intermittent expert (DAGGER-style).")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- 1. Post-launch imports -------------------------------------------------
import os

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401

# Add multi_robot_training + SIM_TO_REAL to path so progressive_s2r wrapper
# and quadruped_locomotion utils import cleanly.
_ALEX_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in [
    os.path.join(_ALEX_ROOT, "multi_robot_training", "source", "quadruped_locomotion"),
    os.path.join(_ALEX_ROOT, "multi_robot_training", "multi_robot_training",
                 "source", "quadruped_locomotion"),
    os.path.expanduser("~/multi_robot_training_new/source/quadruped_locomotion"),
]:
    _p = os.path.abspath(_p)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import quadruped_locomotion  # noqa: F401

_S2R_ROOT = os.path.abspath(os.path.join(_ALEX_ROOT, "SIM_TO_REAL"))
if _S2R_ROOT not in sys.path:
    sys.path.insert(0, _S2R_ROOT)

_PARKOUR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PARKOUR_ROOT not in sys.path:
    sys.path.insert(0, _PARKOUR_ROOT)

from quadruped_locomotion.utils.training_utils import (
    clamp_noise_std,
    configure_tf32,
    register_std_safety_clamp,
)
from quadruped_locomotion.utils.lr_schedule import (
    cosine_annealing_lr,
    set_learning_rate,
)
from wrappers.progressive_s2r import ProgressiveS2RWrapper

from pn_cfg.parkour_nav_env_cfg import ParkourNavEnvCfg
from pn_cfg.parkour_nav_agent_cfg import ParkourNavPPORunnerCfg


# -- 2. Env factory ---------------------------------------------------------

PARKOUR_NAV_ENV_ID = "Isaac-Velocity-ParkourNav-Spot-v0"


def build_env(args):
    """Construct the parkour-nav env, wrap for RSL-RL + Progressive S2R.

    Mirrors SIM_TO_REAL/scripts/train_expert.py::main env construction:
      1. Instantiate ParkourNavEnvCfg and set num_envs.
      2. gym.register a unique env id (idempotent — re-registration is fine).
      3. gym.make -> ManagerBasedRLEnv.
      4. Wrap in RslRlVecEnvWrapper (handles policy/critic obs group routing).
      5. Wrap in ProgressiveS2RWrapper (action+obs delay, sensor noise,
         history stacking via the obs_delay ring buffer).
    """
    env_cfg = ParkourNavEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    print(
        f"[ENV] Creating {PARKOUR_NAV_ENV_ID} with {env_cfg.scene.num_envs} envs "
        f"(phase={args.phase})...",
        flush=True,
    )

    if PARKOUR_NAV_ENV_ID not in gym.registry:
        gym.register(
            id=PARKOUR_NAV_ENV_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={"cfg": env_cfg},
            disable_env_checker=True,
        )

    env = gym.make(PARKOUR_NAV_ENV_ID, cfg=env_cfg)

    try:
        env = RslRlVecEnvWrapper(env, clip_actions=True)
    except (TypeError, ValueError):
        env = RslRlVecEnvWrapper(env)

    s2r = ProgressiveS2RWrapper(
        env,
        max_action_delay_steps=args.action_delay_steps,
        max_obs_delay_steps=args.obs_delay_steps,
        max_dropout_rate=args.sensor_dropout_rate,
        max_drift_rate=args.sensor_drift_rate,
        s2r_start_terrain=0.2,
        s2r_full_terrain=0.6,
    )
    print(
        f"[S2R] delay 0->{args.action_delay_steps} steps, "
        f"dropout 0->{args.sensor_dropout_rate:.0%}, "
        f"history_length={args.obs_history_length}, ramp 0.2->0.6",
        flush=True,
    )
    return s2r


# -- 3. Phase loops ---------------------------------------------------------

def run_teacher_phase(args):
    """Phase 1: PPO with privileged critic observations.

    Actor sees noisy proprio + raycast (policy ObsGroup).
    Critic sees clean obs + privileged terms (critic ObsGroup).
    Isaac Lab's RslRlVecEnvWrapper routes the groups automatically when
    both `policy` and `critic` groups exist on the observations cfg.
    """
    configure_tf32()
    env = build_env(args)

    agent_cfg = ParkourNavPPORunnerCfg()
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.save_interval = args.save_interval
    agent_cfg.seed = args.seed
    if args.no_wandb:
        agent_cfg.logger = "tensorboard"

    # Bug #25 (RAWDOG §1028) defense: disable RSL-RL's adaptive LR scheduler
    # so it doesn't fight our explicit cosine_annealing_lr + value-loss
    # watchdog. Phase-FW-Plus / phase_final_b ran with adaptive AND no
    # watchdog → vf_loss oscillation → NaN cascade.
    if hasattr(agent_cfg, "algorithm") and hasattr(agent_cfg.algorithm, "schedule"):
        agent_cfg.algorithm.schedule = "fixed"
        print(f"[SCHED] Forced agent_cfg.algorithm.schedule = 'fixed'", flush=True)

    log_root = os.path.abspath(os.path.join(
        _PARKOUR_ROOT, "logs", "rsl_rl", agent_cfg.experiment_name,
    ))
    os.makedirs(log_root, exist_ok=True)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_root, device="cuda:0")
    register_std_safety_clamp(runner.alg.policy, args.min_noise_std, args.max_noise_std)

    # NaN/Inf guard on network inputs — observed failure: critic value_loss was
    # converging finite for 2 iters then jumped to NaN at iter 3 despite frozen
    # actor+std and zero-init critic head. Cause fits the HOW_TO_TRAIN_YOUR_RAWDOG
    # doc §36 pattern: raycast returns inf for missed rays → privileged obs
    # carries inf → critic forward propagates → grad NaN → weights NaN. Mirrors
    # the `torch.nan_to_num` pattern used in SIM_TO_REAL/rewards/*.py. Hooks on
    # both the raw normalizer and the MLP defend against normalizer-induced
    # NaN too (running var -> 0 on near-constant privileged terms).
    def _sanitize_input_hook(module, inputs):
        if not inputs:
            return inputs
        first = inputs[0]
        if isinstance(first, torch.Tensor):
            fixed = torch.nan_to_num(first, nan=0.0, posinf=10.0, neginf=-10.0)
            return (fixed,) + inputs[1:]
        return inputs

    def _sanitize_output_hook(module, inputs, output):
        # Sanitize network OUTPUTS too. Without this, a single exploded forward
        # value (e.g. inf V_pred) produces an inf loss → inf grad → and
        # `clip_grad_norm_` propagates the non-finite through ALL parameters
        # (total_norm becomes nan, clip_coef=nan, every param goes nan).
        # Observed at iter 106, 6 steps after actor unfreeze in option 5.
        if isinstance(output, torch.Tensor):
            return torch.nan_to_num(output, nan=0.0, posinf=10.0, neginf=-10.0)
        return output

    _policy = runner.alg.policy
    _pre_hooked = 0
    _post_hooked = 0
    for _name in ("actor", "critic", "actor_obs_normalizer", "critic_obs_normalizer"):
        _mod = getattr(_policy, _name, None)
        if isinstance(_mod, torch.nn.Module):
            _mod.register_forward_pre_hook(_sanitize_input_hook)
            _pre_hooked += 1
    for _name in ("actor", "critic"):
        _mod = getattr(_policy, _name, None)
        if isinstance(_mod, torch.nn.Module):
            _mod.register_forward_hook(_sanitize_output_hook)
            _post_hooked += 1
    print(f"[GUARD] registered nan_to_num pre-hooks on {_pre_hooked} modules, "
          f"post-hooks on {_post_hooked} modules", flush=True)

    # ---------------------------------------------------------------------
    # Value-loss watchdog (Bug #25 fix from
    # `multi_robot_training/docs/HOW_TO_TRAIN_YOUR_RAWDOG.md` §1028).
    #
    # Phase-FW-Plus + phase_final_b both died on the slow-bleed pattern:
    # value loss oscillates (10 → 56 → 973 → 28 → ... ) during what looks
    # like a healthy run, accumulates damage over ~1000 iters, then explodes
    # to 1e20+ → NaN. This wrapper layers two defenses on top of
    # runner.alg.update():
    #   1. Explicit cosine LR annealing (we forced schedule="fixed" above
    #      so RSL-RL doesn't fight us).
    #   2. Reactive watchdog — if value_function_loss > 100 for any iter,
    #      halve the LR for 50 subsequent iters. Breaks the spike →
    #      amplification → NaN cascade.
    # Plus per-iter clamp_noise_std so the std param can never escape the
    # [min_noise_std, max_noise_std] bounds between mini-batches.
    # ---------------------------------------------------------------------
    _VL_THRESHOLD = 100.0       # vf_loss above this triggers the guard
    _VL_COOLDOWN_ITERS = 50     # iters to keep the reduced LR
    _vl_penalty = [1.0]         # current LR multiplier (1.0 = normal)
    _vl_cooldown = [0]          # remaining cooldown iters
    _vl_spike_count = [0]       # total spike count for telemetry
    _iteration_counter = [0]    # absolute iter (across warmup + main learn)

    _original_update = runner.alg.update
    _max_iters_total = args.max_iterations

    def _update_with_schedule(*update_args, **update_kwargs):
        it = _iteration_counter[0]

        lr = cosine_annealing_lr(
            it, _max_iters_total,
            args.lr_max, args.lr_min, args.warmup_iters,
        )
        if _vl_cooldown[0] > 0:
            lr *= _vl_penalty[0]
            _vl_cooldown[0] -= 1
            if _vl_cooldown[0] == 0:
                _vl_penalty[0] = 1.0
                print(f"[GUARD] Value-loss cooldown expired at iter {it}; "
                      f"restoring scheduled LR.", flush=True)
        set_learning_rate(runner, lr)

        result = _original_update(*update_args, **update_kwargs)

        vl = result.get("value_function", 0.0) if isinstance(result, dict) else 0.0
        if vl > _VL_THRESHOLD:
            # COMPOUND the penalty on consecutive spikes — fixes runaway
            # explosion case (parkour_hailmary2 iter 101-105 saw vf_loss go
            # 5e5 → 1e14 → 4e22 → 6e30 with the original "set to 0.5" logic
            # because halving once doesn't keep up with 10^8/iter growth).
            # Floor at 0.001 = LR×1e-3 of nominal, enough to fully stop
            # divergent updates without bottoming out forever.
            _vl_penalty[0] = max(_vl_penalty[0] * 0.5, 0.001)
            _vl_cooldown[0] = _VL_COOLDOWN_ITERS
            _vl_spike_count[0] += 1
            print(
                f"[GUARD] Value-loss spike: {vl:.1f} > {_VL_THRESHOLD} at iter {it} "
                f"(spike #{_vl_spike_count[0]}). LR multiplier now ×{_vl_penalty[0]:.4f}, "
                f"cooldown reset to {_VL_COOLDOWN_ITERS} iters.",
                flush=True,
            )

        clamp_noise_std(runner.alg.policy, args.min_noise_std, args.max_noise_std)

        if it % 100 == 0:
            guard_str = ""
            if _vl_cooldown[0] > 0:
                guard_str = f"  [GUARD active: LR×{_vl_penalty[0]}, {_vl_cooldown[0]} iters left]"
            print(f"[TRAIN] iter={it:5d}/{_max_iters_total}  lr={lr:.2e}"
                  f"  vf_loss={vl:.3f}  spikes={_vl_spike_count[0]}{guard_str}",
                  flush=True)

        _iteration_counter[0] += 1
        return result

    runner.alg.update = _update_with_schedule
    print(f"[GUARD] Value-loss watchdog armed (threshold={_VL_THRESHOLD}, "
          f"cooldown={_VL_COOLDOWN_ITERS} iters, LR×0.5)", flush=True)
    print(f"[SCHED] Cosine LR: lr_max={args.lr_max:.1e} → "
          f"lr_min={args.lr_min:.1e} over {_max_iters_total} iters "
          f"(warmup {args.warmup_iters} iters)", flush=True)

    # Actor-only resume from hybrid_nocoach_19999.pt (H-100 Hail Mary plan)
    if args.actor_only_resume is not None:
        ckpt_path = os.path.expanduser(args.actor_only_resume)
        print(f"[RESUME] actor-only load from {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location="cuda:0", weights_only=False)
        actor_state = {
            k.replace("actor.", ""): v
            for k, v in ckpt["model_state_dict"].items()
            if k.startswith("actor.")
        }
        runner.alg.policy.actor.load_state_dict(actor_state, strict=False)
    elif args.resume_path is not None:
        ckpt_path = os.path.expanduser(args.resume_path)
        print(f"[RESUME] full load from {ckpt_path}", flush=True)
        runner.load(ckpt_path)

    print(f"[TRAIN] teacher phase: {args.max_iterations} iters on {args.num_envs} envs",
          flush=True)

    # Option-4 critic warmup: option-3 went NaN at iter 2 because
    # (a) `policy.std` is a separate nn.Parameter — NOT under actor.parameters() —
    #     so freezing actor.parameters() left std trainable, and PPO drove it to nan;
    # (b) fresh 485-input critic's random init produced large V_pred variance →
    #     value_loss overflow → grad nan → std nan.
    # Fix: also freeze std/log_std, AND zero-init critic's final layer so V starts
    # at 0 and value loss is bounded by max_return^2 during warmup.
    warmup = args.critic_warmup_iters if args.actor_only_resume else 0
    if warmup > 0:
        policy = runner.alg.policy
        print(f"[WARMUP] freezing actor + std for {warmup} iters, zero-init critic head",
              flush=True)
        for p in policy.actor.parameters():
            p.requires_grad = False
        if hasattr(policy, "std") and isinstance(policy.std, torch.nn.Parameter):
            policy.std.requires_grad = False
        if hasattr(policy, "log_std") and isinstance(policy.log_std, torch.nn.Parameter):
            policy.log_std.requires_grad = False
        # Zero-init critic output layer — find last nn.Linear in critic
        last_linear = None
        for m in policy.critic.modules():
            if isinstance(m, torch.nn.Linear):
                last_linear = m
        if last_linear is not None:
            torch.nn.init.zeros_(last_linear.weight)
            if last_linear.bias is not None:
                torch.nn.init.zeros_(last_linear.bias)
            print(f"[WARMUP] zeroed critic final Linear ({last_linear})", flush=True)
        runner.learn(num_learning_iterations=warmup, init_at_random_ep_len=True)
        print(f"[WARMUP] unfreezing actor + std, resuming full PPO", flush=True)
        for p in policy.actor.parameters():
            p.requires_grad = True
        if hasattr(policy, "std") and isinstance(policy.std, torch.nn.Parameter):
            policy.std.requires_grad = True
        if hasattr(policy, "log_std") and isinstance(policy.log_std, torch.nn.Parameter):
            policy.log_std.requires_grad = True
        remaining = max(args.max_iterations - warmup, 0)
        if remaining > 0:
            runner.learn(num_learning_iterations=remaining,
                         init_at_random_ep_len=True)
    else:
        runner.learn(num_learning_iterations=args.max_iterations,
                     init_at_random_ep_len=True)


def run_student_phase(args):
    """Phase 2: distill teacher into proprio-only GRU policy.

    Student obs: policy group only (no critic).
    Training: BC on teacher rollouts OR BC warmup + DAGGER.
    Reference: Miki 2022 distillation loss.
    """
    if args.teacher_ckpt is None:
        raise ValueError("--teacher_ckpt required in --phase student")
    raise NotImplementedError(
        "Student distillation loop is P1.9 — implement after teacher "
        "converges. Pattern: load teacher frozen, spin student runner with "
        "KL-to-teacher aux loss, rollout DAGGER batches."
    )


def main():
    print(f"\n{'='*70}", flush=True)
    print(f"  PARKOUR_NAV TRAINING — phase={args_cli.phase}", flush=True)
    print(f"{'='*70}\n", flush=True)

    if args_cli.phase == "teacher":
        run_teacher_phase(args_cli)
    else:
        run_student_phase(args_cli)


if __name__ == "__main__":
    main()
    # Do NOT call simulation_app.close() — causes CUDA deadlock.
    os._exit(0)
