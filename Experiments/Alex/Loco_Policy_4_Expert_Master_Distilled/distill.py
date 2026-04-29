"""Multi-Expert Distillation Training for Spot.

Distills two specialist experts (friction/grass + boulders/stairs) into a
single generalist student policy via terrain-routed DAgger with PPO.

The student acts in the environment (DAgger-style), then both frozen experts
are queried for what THEY would have done. A soft gate based on height scan
roughness blends the expert actions. The student learns to match this blend
via MSE + KL loss, combined with standard PPO reward maximization.

Architecture:
    Student acts in env -> Collects rollout -> Expert Router queries both
    experts on same observations -> Distillation Loss (MSE + KL) blended
    with PPO loss -> Student updates

Both experts and student use [512, 256, 128] Mason hybrid architecture
with 235-dim observations (187 height scan + 48 proprioceptive).

Usage (H100):
    cd ~/multi_expert_distillation
    python distill.py --headless \\
        --friction_expert /path/to/mason_hybrid_best_33200.pt \\
        --obstacle_expert /path/to/obstacle_best.pt \\
        --num_envs 4096 --max_iterations 5000 --no_wandb

Usage (local debug):
    python distill.py --headless \\
        --friction_expert checkpoints/mason_hybrid_best_33200.pt \\
        --obstacle_expert checkpoints/obstacle_best.pt \\
        --num_envs 64 --max_iterations 10 --no_wandb

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

# -- 0. Parse args BEFORE any Isaac imports ----------------------------------
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Multi-expert distillation for Spot")
parser.add_argument("--friction_expert", type=str, required=True,
                    help="Path to friction/grass expert checkpoint")
parser.add_argument("--obstacle_expert", type=str, required=True,
                    help="Path to boulder/stairs expert checkpoint")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--no_wandb", action="store_true", default=False)

# Distillation params
parser.add_argument("--alpha_start", type=float, default=0.8,
                    help="Initial distillation weight (high = trust experts)")
parser.add_argument("--alpha_end", type=float, default=0.2,
                    help="Final distillation weight (low = trust PPO)")
parser.add_argument("--kl_weight", type=float, default=0.1,
                    help="KL divergence weight in distillation loss")
parser.add_argument("--roughness_threshold", type=float, default=0.005,
                    help="Height scan variance threshold for expert routing")
parser.add_argument("--routing_temperature", type=float, default=0.005,
                    help="Sigmoid temperature for routing (lower = sharper)")
parser.add_argument("--distill_batch_size", type=int, default=8192,
                    help="Samples per distillation gradient step")

# Student init
parser.add_argument("--init_from", type=str, default="friction",
                    choices=["friction", "obstacle", "scratch"],
                    help="Initialize student from which expert")
parser.add_argument("--critic_warmup_iters", type=int, default=300,
                    help="Freeze actor for N iters so critic calibrates")

# Safety
parser.add_argument("--min_noise_std", type=float, default=0.3)
parser.add_argument("--max_noise_std", type=float, default=0.5)

# Resume
parser.add_argument("--resume", action="store_true", default=False,
                    help="Resume from a checkpoint")
parser.add_argument("--load_run", type=str, default=None,
                    help="Run directory to resume from")
parser.add_argument("--load_checkpoint", type=str, default=None,
                    help="Checkpoint file to resume from")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -- 1. Imports (AFTER SimulationApp) ----------------------------------------
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
import torch.nn.functional as F
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401

# Path setup for the reorganized layout: Loco_Shared (quadruped_locomotion
# package) + Loco_Policy_2_ARL_Hybrid/configs (parent SpotARLHybridEnvCfg)
# + this Loco_Policy_4 root (for expert_router + distillation_loss).
_LOCO4_ROOT = os.path.dirname(os.path.abspath(__file__))
_ALEX_ROOT = os.path.abspath(os.path.join(_LOCO4_ROOT, ".."))
for _p in (
    _LOCO4_ROOT,
    os.path.join(_ALEX_ROOT, "Loco_Policy_2_ARL_Hybrid", "configs"),
    os.path.join(_ALEX_ROOT, "Loco_Policy_2_ARL_Hybrid", "configs", "agents"),
    os.path.join(_ALEX_ROOT, "Loco_Shared"),
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import quadruped_locomotion  # noqa: F401

from expert_router import ExpertRouter
from distillation_loss import DistillationLoss

# TF32 for H100 performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# -- 2. Main -----------------------------------------------------------------

def main():
    # -- Load configs (path setup done at module load) --
    from arl_hybrid_env_cfg import SpotARLHybridEnvCfg
    from rsl_rl_arl_hybrid_cfg import SpotARLHybridPPORunnerCfg

    env_cfg = SpotARLHybridEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    # Bug #22: disable body_height_tracking on rough terrain
    env_cfg.rewards.body_height_tracking.weight = 0.0

    agent_cfg = SpotARLHybridPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed
    agent_cfg.save_interval = args_cli.save_interval
    agent_cfg.experiment_name = "spot_distill"
    agent_cfg.logger = "tensorboard"

    # Log directory
    log_root = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root = os.path.abspath(log_root)
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # -- Banner --
    print(f"\n{'='*70}")
    print(f"  MULTI-EXPERT DISTILLATION — SPOT")
    print(f"  {'='*66}")
    print(f"  Friction expert:  {args_cli.friction_expert}")
    print(f"  Obstacle expert:  {args_cli.obstacle_expert}")
    print(f"  Init from:        {args_cli.init_from}")
    print(f"  Alpha:            {args_cli.alpha_start} -> {args_cli.alpha_end}")
    print(f"  KL weight:        {args_cli.kl_weight}")
    print(f"  Roughness thr:    {args_cli.roughness_threshold}")
    print(f"  Routing temp:     {args_cli.routing_temperature}")
    print(f"  Envs:             {args_cli.num_envs}")
    print(f"  Max iterations:   {args_cli.max_iterations}")
    print(f"  Critic warmup:    {args_cli.critic_warmup_iters}")
    print(f"  Log dir:          {log_dir}")
    print(f"{'='*70}\n", flush=True)

    # -- Create environment --
    env = gym.make("Locomotion-ARLHybrid-Spot-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # -- Create student runner --
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )

    # Register std safety clamp (Bug #24)
    from quadruped_locomotion.utils.training_utils import register_std_safety_clamp
    register_std_safety_clamp(runner.alg.policy, args_cli.min_noise_std, args_cli.max_noise_std)

    # -- Resume or initialize student --
    if args_cli.resume and args_cli.load_run and args_cli.load_checkpoint:
        resume_path = os.path.join(log_root, args_cli.load_run, args_cli.load_checkpoint)
        print(f"[RESUME] Loading from {resume_path}")
        runner.load(resume_path)
        # Skip critic warmup on resume
        args_cli.critic_warmup_iters = 0
    elif args_cli.init_from != "scratch":
        init_path = (args_cli.friction_expert if args_cli.init_from == "friction"
                     else args_cli.obstacle_expert)
        init_ckpt = torch.load(init_path, map_location=agent_cfg.device, weights_only=False)
        full_state = init_ckpt.get("model_state_dict", init_ckpt)

        # Actor-only resume: load actor weights + std, leave critic fresh
        actor_keys = {k: v for k, v in full_state.items()
                      if k.startswith("actor.") or k in ("std", "log_std")}
        runner.alg.policy.load_state_dict(actor_keys, strict=False)
        print(f"[INIT] Loaded {len(actor_keys)} actor keys from {args_cli.init_from} expert")

    # -- Load frozen experts --
    device = agent_cfg.device
    friction_expert = ExpertRouter.load_expert(
        args_cli.friction_expert, num_obs=235, num_actions=12, device=device)
    obstacle_expert = ExpertRouter.load_expert(
        args_cli.obstacle_expert, num_obs=235, num_actions=12, device=device)
    print("[EXPERTS] Both experts loaded and frozen")

    # -- Create router and loss --
    router = ExpertRouter(
        friction_expert, obstacle_expert,
        height_scan_dims=187,
        threshold=args_cli.roughness_threshold,
        temperature=args_cli.routing_temperature,
    )
    distill_loss_fn = DistillationLoss(kl_weight=args_cli.kl_weight)

    # -- Save config --
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # -- Critic warmup: freeze actor --
    warmup_remaining = [args_cli.critic_warmup_iters]
    actor_frozen = [False]
    if args_cli.critic_warmup_iters > 0 and args_cli.init_from != "scratch":
        for name, param in runner.alg.policy.named_parameters():
            if name.startswith("actor.") or name in ("std", "log_std"):
                param.requires_grad = False
        actor_frozen[0] = True
        print(f"[WARMUP] Actor frozen for {args_cli.critic_warmup_iters} iters")

    # -- Monkey-patch PPO update to add distillation loss --
    original_update = runner.alg.update
    iter_counter = [0]
    start_time = time.time()

    def update_with_distillation():
        it = iter_counter[0]

        # Compute alpha (distillation weight) — linear anneal
        frac = min(it / max(args_cli.max_iterations, 1), 1.0)
        alpha = args_cli.alpha_start + (args_cli.alpha_end - args_cli.alpha_start) * frac

        # Standard PPO update first
        result = original_update()

        # Post-hoc distillation step: separate gradient pass on rollout obs
        if not actor_frozen[0]:
            storage = runner.alg.storage
            num_steps = agent_cfg.num_steps_per_env
            # RSL-RL stores observations as TensorDict or plain tensor
            obs_data = storage.observations
            if hasattr(obs_data, 'keys'):
                all_obs = obs_data["policy"][:num_steps].reshape(-1, 235)
            else:
                all_obs = obs_data[:num_steps].reshape(-1, 235)

            # Sample a mini-batch
            n = min(args_cli.distill_batch_size, all_obs.shape[0])
            idx = torch.randperm(all_obs.shape[0], device=all_obs.device)[:n]
            obs_batch = all_obs[idx]

            # Query experts (frozen, no grad)
            expert_mean, expert_std, gate, _, _ = router.get_expert_actions(obs_batch)

            # Student forward (with grad)
            student_mean = runner.alg.policy.actor(obs_batch)
            student_std = runner.alg.policy.std.expand_as(student_mean)

            # Compute and apply distillation loss
            d_loss, mse_val, kl_val = distill_loss_fn(
                student_mean, student_std, expert_mean, expert_std)

            runner.alg.optimizer.zero_grad()
            (alpha * d_loss).backward()
            torch.nn.utils.clip_grad_norm_(runner.alg.policy.parameters(), 1.0)
            runner.alg.optimizer.step()

        # Critic warmup countdown
        if actor_frozen[0] and warmup_remaining[0] > 0:
            warmup_remaining[0] -= 1
            if warmup_remaining[0] == 0:
                for name, param in runner.alg.policy.named_parameters():
                    if name.startswith("actor.") or name in ("std", "log_std"):
                        param.requires_grad = True
                actor_frozen[0] = False
                print(f"[WARMUP] Critic warmup complete at iter {it}. Actor unfrozen.")

        # Clamp noise std (Bug #24)
        from quadruped_locomotion.utils.training_utils import clamp_noise_std
        clamp_noise_std(runner.alg.policy, args_cli.min_noise_std, args_cli.max_noise_std)

        # Logging
        if it % 100 == 0 and not actor_frozen[0]:
            elapsed_h = (time.time() - start_time) / 3600
            noise = runner.alg.policy.std.mean().item()
            # Re-query for logging stats
            with torch.no_grad():
                obs_log = runner.alg.storage.observations
                if hasattr(obs_log, 'keys'):
                    sample_obs = obs_log["policy"][0][:1000]
                else:
                    sample_obs = obs_log[0].reshape(-1, 235)[:1000]
                _, _, gate_sample, _, _ = router.get_expert_actions(sample_obs)
                mean_gate = gate_sample.mean().item()
                s_mean = runner.alg.policy.actor(sample_obs)
                s_std = runner.alg.policy.std.expand_as(s_mean)
                e_mean, e_std, _, _, _ = router.get_expert_actions(sample_obs)
                _, mse_log, kl_log = distill_loss_fn(s_mean, s_std, e_mean, e_std)

            print(
                f"[DISTILL] iter={it:5d}/{args_cli.max_iterations}  "
                f"alpha={alpha:.3f}  mse={mse_log:.4f}  kl={kl_log:.4f}  "
                f"gate={mean_gate:.3f}  noise={noise:.3f}  "
                f"elapsed={elapsed_h:.1f}h",
                flush=True)

            # TensorBoard
            w = getattr(runner, 'writer', None)
            if w is not None:
                w.add_scalar("Distill/alpha", alpha, it)
                w.add_scalar("Distill/mse_loss", mse_log, it)
                w.add_scalar("Distill/kl_loss", kl_log, it)
                w.add_scalar("Distill/mean_gate", mean_gate, it)

        iter_counter[0] += 1
        return result

    runner.alg.update = update_with_distillation

    # -- Train --
    print("\n[TRAIN] Starting multi-expert distillation...", flush=True)
    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)

    elapsed = (time.time() - start_time) / 3600
    print(f"\n{'='*70}")
    print(f"  DISTILLATION COMPLETE — {elapsed:.1f} hours")
    print(f"  Checkpoints: {log_dir}")
    print(f"{'='*70}\n", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)  # CUDA deadlock fix (Bug: never call simulation_app.close())
