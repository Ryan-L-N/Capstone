"""6-Expert Distillation Training for S2R Generalist Student.

Distills 6 terrain-specialist experts into a single generalist student via
learned attention routing + DAgger-style PPO.

The student trains at 20 Hz (real Spot rate) on balanced all-terrain with
all S2R wrappers active. Each step:
  1. PPO update (standard reward maximization)
  2. Post-hoc distillation gradient (MSE + KL on router-blended expert actions)
  3. Alpha annealing: expert-heavy (0.8) -> PPO-heavy (0.2)

Usage (H100):
    python scripts/train_distill_s2r.py \
        --expert_friction checkpoints/expert_friction/best.pt \
        --expert_stairs_up checkpoints/expert_stairs_up/best.pt \
        --expert_stairs_down checkpoints/expert_stairs_down/best.pt \
        --expert_boulders checkpoints/expert_boulders/best.pt \
        --expert_slopes checkpoints/expert_slopes/best.pt \
        --expert_mixed_rough checkpoints/expert_mixed_rough/best.pt \
        --headless --no_wandb --num_envs 4096 --max_iterations 8000

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

# -- 0. Parse args BEFORE Isaac imports ------------------------------------
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="6-expert S2R distillation")
parser.add_argument("--expert_friction", type=str, required=True)
parser.add_argument("--expert_stairs_up", type=str, required=True)
parser.add_argument("--expert_stairs_down", type=str, required=True)
parser.add_argument("--expert_boulders", type=str, required=True)
parser.add_argument("--expert_slopes", type=str, required=True)
parser.add_argument("--expert_mixed_rough", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=8000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--no_wandb", action="store_true", default=False)

# Distillation params
parser.add_argument("--alpha_start", type=float, default=0.8)
parser.add_argument("--alpha_end", type=float, default=0.2)
parser.add_argument("--kl_weight", type=float, default=0.1)
parser.add_argument("--distill_batch_size", type=int, default=8192)
parser.add_argument("--router_lr", type=float, default=1e-4)

# Safety
parser.add_argument("--min_noise_std", type=float, default=0.3)
parser.add_argument("--max_noise_std", type=float, default=0.5)
parser.add_argument("--critic_warmup_iters", type=int, default=300)

# S2R wrappers
parser.add_argument("--action_delay_steps", type=int, default=2)
parser.add_argument("--obs_delay_steps", type=int, default=1)
parser.add_argument("--sensor_dropout_rate", type=float, default=0.03)
parser.add_argument("--sensor_drift_rate", type=float, default=0.001)

# Resume
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--load_checkpoint", type=str, default="model_.*\\.pt")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- 1. Post-launch imports ------------------------------------------------
import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401

# Path setup for the reorganized layout: this Loco_Policy_3 root +
# Loco_Policy_2_ARL_Hybrid/configs (parent of SpotARLHybridEnvCfg) +
# Loco_Shared (the quadruped_locomotion shared package).
_LOCO3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ALEX_ROOT = os.path.abspath(os.path.join(_LOCO3_ROOT, ".."))
for _p in (
    _LOCO3_ROOT,
    os.path.join(_ALEX_ROOT, "Loco_Policy_2_ARL_Hybrid", "configs"),
    os.path.join(_ALEX_ROOT, "Loco_Shared"),
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import quadruped_locomotion  # noqa: F401

from quadruped_locomotion.utils.lr_schedule import cosine_annealing_lr, set_learning_rate
from quadruped_locomotion.utils.training_utils import configure_tf32, clamp_noise_std, register_std_safety_clamp

from wrappers import ActionDelayWrapper, ObservationDelayWrapper, SensorNoiseWrapper
from distillation.multi_expert_router import MultiExpertRouter
from distillation.distillation_loss import DistillationLoss


def main():
    print(f"\n{'='*70}", flush=True)
    print(f"  S2R 6-EXPERT DISTILLATION", flush=True)
    print(f"{'='*70}\n", flush=True)

    configure_tf32()

    # -- Load distillation env config (20 Hz, balanced terrain) -------------
    from configs.distillation_env_cfg import SpotS2RDistillEnvCfg
    from configs.agent_cfg import SpotS2RDistillPPORunnerCfg

    env_cfg = SpotS2RDistillEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = SpotS2RDistillPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.save_interval = args_cli.save_interval
    agent_cfg.seed = args_cli.seed
    if args_cli.no_wandb:
        agent_cfg.logger = "tensorboard"

    steps_per_iter = env_cfg.scene.num_envs * agent_cfg.num_steps_per_env

    # -- Create environment -------------------------------------------------
    gym_env_id = "S2R-Distill-Spot-v0"
    gym.register(
        id=gym_env_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"cfg": env_cfg},
        disable_env_checker=True,
    )

    env = gym.make(gym_env_id, cfg=env_cfg)
    try:
        env = RslRlVecEnvWrapper(env, clip_actions=True)
    except (TypeError, ValueError):
        env = RslRlVecEnvWrapper(env)

    # Apply S2R wrappers
    env = ObservationDelayWrapper(env, delay_steps=args_cli.obs_delay_steps)
    env = ActionDelayWrapper(env, delay_steps=args_cli.action_delay_steps)
    env = SensorNoiseWrapper(
        env,
        dropout_rate=args_cli.sensor_dropout_rate,
        drift_rate=args_cli.sensor_drift_rate,
    )

    # -- Load 6 frozen experts + create router ------------------------------
    expert_paths = {
        "friction": args_cli.expert_friction,
        "stairs_up": args_cli.expert_stairs_up,
        "stairs_down": args_cli.expert_stairs_down,
        "boulders": args_cli.expert_boulders,
        "slopes": args_cli.expert_slopes,
        "mixed_rough": args_cli.expert_mixed_rough,
    }

    router = MultiExpertRouter.load_all_experts(expert_paths, device="cuda:0")
    distill_loss_fn = DistillationLoss(kl_weight=args_cli.kl_weight)

    # -- Create runner (student) --------------------------------------------
    log_root_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "logs", "rsl_rl", "spot_s2r_distill"
    ))

    os.makedirs(log_root_path, exist_ok=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_root_path, device="cuda:0")
    log_dir = runner.log_dir if hasattr(runner, 'log_dir') else log_root_path

    register_std_safety_clamp(runner.alg.policy, args_cli.min_noise_std, args_cli.max_noise_std)

    # -- Combined optimizer (student + router gate_net) ---------------------
    student_params = list(runner.alg.policy.parameters())
    router_params = list(router.gate_net.parameters())

    # Use separate param groups with different LRs
    combined_optimizer = torch.optim.Adam([
        {"params": student_params, "lr": 1e-3},
        {"params": router_params, "lr": args_cli.router_lr},
    ])

    # -- Critic warmup (freeze actor for N iters) ---------------------------
    actor_frozen = [False]
    warmup_remaining = [args_cli.critic_warmup_iters]

    if args_cli.critic_warmup_iters > 0:
        for name, param in runner.alg.policy.named_parameters():
            if name.startswith("actor.") or name in ("std", "log_std"):
                param.requires_grad = False
        actor_frozen[0] = True
        print(f"[WARMUP] Actor frozen for {args_cli.critic_warmup_iters} iters", flush=True)

    # -- Save config --------------------------------------------------------
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # -- Training loop with distillation -----------------------------------
    print(f"[TRAIN] Starting distillation at 20 Hz (decimation=25)...", flush=True)
    start_time = time.time()

    original_update = runner.alg.update
    _iteration = [0]
    _VL_THRESHOLD = 100.0
    _VL_COOLDOWN_ITERS = 50
    _vl_penalty = [1.0]
    _vl_cooldown = [0]

    def update_with_distillation(*args, **kwargs):
        it = _iteration[0]
        max_it = max(args_cli.max_iterations, 1)

        # Alpha annealing: expert-heavy -> PPO-heavy
        frac = min(it / max_it, 1.0)
        alpha = args_cli.alpha_start + (args_cli.alpha_end - args_cli.alpha_start) * frac

        # Cosine LR
        lr = cosine_annealing_lr(it, max_it, 1e-3, 1e-5, 50)
        if _vl_cooldown[0] > 0:
            lr *= _vl_penalty[0]
            _vl_cooldown[0] -= 1
        set_learning_rate(runner, lr)

        # Standard PPO update
        result = original_update(*args, **kwargs)

        # Critic warmup countdown
        if actor_frozen[0] and warmup_remaining[0] > 0:
            warmup_remaining[0] -= 1
            if warmup_remaining[0] == 0:
                for name, param in runner.alg.policy.named_parameters():
                    if name.startswith("actor.") or name in ("std", "log_std"):
                        param.requires_grad = True
                actor_frozen[0] = False
                print(f"[WARMUP] Critic warmup complete at iter {it}. Actor unfrozen.", flush=True)

        # Post-hoc distillation (only when actor is not frozen)
        if not actor_frozen[0]:
            # Sample observation batch from rollout storage
            try:
                storage = runner.alg.storage
                if hasattr(storage, 'observations'):
                    all_obs = storage.observations.reshape(-1, 235)
                else:
                    all_obs = storage["obs"].reshape(-1, 235)
            except Exception:
                all_obs = None

            if all_obs is not None and all_obs.shape[0] >= args_cli.distill_batch_size:
                idx = torch.randperm(all_obs.shape[0])[:args_cli.distill_batch_size]
                obs_batch = all_obs[idx].detach()

                # Query router (gate_net has grad, experts frozen)
                blended_mean, blended_std, weights = router(obs_batch)

                # Student forward
                student_mean = runner.alg.policy.actor(obs_batch)
                student_std = runner.alg.policy.std.expand_as(student_mean)

                # Distillation loss
                d_loss, mse_val, kl_val = distill_loss_fn(
                    student_mean, student_std,
                    blended_mean.detach(), blended_std.detach(),
                )

                # Gradient step
                combined_optimizer.zero_grad()
                (alpha * d_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    student_params + router_params, 1.0
                )
                combined_optimizer.step()

        # Value loss watchdog
        vl = result.get("value_function", 0.0) if isinstance(result, dict) else 0.0
        if vl > _VL_THRESHOLD:
            _vl_penalty[0] = 0.5
            _vl_cooldown[0] = _VL_COOLDOWN_ITERS

        # Noise clamp
        clamp_noise_std(runner.alg.policy, args_cli.min_noise_std, args_cli.max_noise_std)

        # Logging
        if it % 100 == 0:
            elapsed = time.time() - start_time
            noise = runner.alg.policy.std.mean().item() if hasattr(runner.alg.policy, 'std') else 0
            mean_gate = ""
            if not actor_frozen[0] and 'weights' in dir():
                gate_avg = weights.mean(dim=0).cpu().tolist()
                gate_str = ", ".join(f"{MultiExpertRouter.EXPERT_NAMES[i]}={g:.2f}" for i, g in enumerate(gate_avg))
                mean_gate = f"  gate=[{gate_str}]"

            print(
                f"[DISTILL] iter={it:5d}/{max_it}  alpha={alpha:.2f}  "
                f"lr={lr:.2e}  noise={noise:.3f}  "
                f"elapsed={elapsed/3600:.1f}h{mean_gate}",
                flush=True,
            )

        _iteration[0] += 1
        return result

    runner.alg.update = update_with_distillation

    runner.learn(
        num_learning_iterations=args_cli.max_iterations,
        init_at_random_ep_len=True,
    )

    elapsed = time.time() - start_time
    print(f"\n{'='*70}", flush=True)
    print(f"  DISTILLATION COMPLETE", flush=True)
    print(f"  Time: {elapsed/3600:.1f}h  Logs: {log_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)
