"""Phase 1: Unified PPO Training Script for Spot and Vision60.

Single script parameterized by --robot spot|vision60:
  - Loads robot-specific EnvCfg and PPOCfg
  - Cosine LR annealing with linear warmup
  - Progressive domain randomization (Vision60) or fixed DR (Spot)
  - Weights & Biases integration for experiment tracking
  - Custom per-reward-term W&B logging

Usage (H100 — Spot):
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/multi_robot_training/train_ppo.py --headless \\
        --robot spot --num_envs 20480

Usage (H100 — Vision60):
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/multi_robot_training/train_ppo.py --headless \\
        --robot vision60 --num_envs 20480

Usage (local debug):
    isaaclab.bat -p /path/to/train_ppo.py --headless \\
        --robot spot --num_envs 64 --max_iterations 10 --no_wandb

Created for AI2C Tech Capstone — MS for Autonomy, February 2026
"""

# ── 0. Parse args BEFORE any Isaac imports ──────────────────────────────
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Multi-robot PPO training (Phase 1)")
parser.add_argument("--robot", type=str, required=True, choices=["spot", "vision60"],
                    help="Robot to train: spot or vision60")
parser.add_argument("--num_envs", type=int, default=20480,
                    help="Number of parallel environments (default 20480 for H100)")
parser.add_argument("--max_iterations", type=int, default=60000,
                    help="Max training iterations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--resume", action="store_true", default=False,
                    help="Resume from latest checkpoint")
parser.add_argument("--load_run", type=str, default=None,
                    help="Run directory to resume from")
parser.add_argument("--load_checkpoint", type=str, default="model_.*\\.pt",
                    help="Checkpoint regex to load")
parser.add_argument("--lr_max", type=float, default=1e-3,
                    help="Max learning rate for cosine annealing")
parser.add_argument("--lr_min", type=float, default=1e-5,
                    help="Min learning rate for cosine annealing")
parser.add_argument("--warmup_iters", type=int, default=500,
                    help="LR warmup iterations")
parser.add_argument("--dr_expansion_iters", type=int, default=15000,
                    help="Iterations over which DR expands (Vision60 only)")
parser.add_argument("--no_wandb", action="store_true", default=False,
                    help="Disable Weights & Biases logging")
parser.add_argument("--min_noise_std", type=float, default=0.3,
                    help="Minimum noise std floor")
parser.add_argument("--max_noise_std", type=float, default=2.0,
                    help="Maximum noise std ceiling")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 1. Imports (AFTER SimulationApp) ────────────────────────────────────
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Trigger standard gym registrations
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path

# Add our project to path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from shared.lr_schedule import cosine_annealing_lr, set_learning_rate
from shared.dr_schedule import update_dr_params, DR_SCHEDULE
from shared.training_utils import configure_tf32, clamp_noise_std

# Enable TF32
configure_tf32()

# ── 2. Robot-Specific Config Loading ───────────────────────────────────

def load_robot_configs(robot: str):
    """Load env and agent configs for the specified robot.

    Returns:
        (env_cfg, agent_cfg, env_id) tuple.
    """
    if robot == "spot":
        from configs.spot_ppo_env_cfg import SpotPPOEnvCfg
        from configs.spot_ppo_cfg import SpotPPORunnerCfg
        return SpotPPOEnvCfg(), SpotPPORunnerCfg(), "Isaac-Velocity-Robust-Spot-PPO-v0"
    else:
        from configs.vision60_ppo_env_cfg import Vision60PPOEnvCfg
        from configs.vision60_ppo_cfg import Vision60PPORunnerCfg
        return Vision60PPOEnvCfg(), Vision60PPORunnerCfg(), "Isaac-Velocity-Robust-Vision60-PPO-v0"


# ── 3. Main ─────────────────────────────────────────────────────────────

def main():
    robot = args_cli.robot
    env_cfg, agent_cfg, env_id = load_robot_configs(robot)

    # Apply CLI overrides
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed

    # W&B toggle
    if args_cli.no_wandb:
        agent_cfg.logger = "tensorboard"

    # Logging
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    steps_per_iter = env_cfg.scene.num_envs * agent_cfg.num_steps_per_env

    print(f"\n{'='*70}", flush=True)
    print(f"  MULTI-ROBOT PPO TRAINING — PHASE 1", flush=True)
    print(f"  {'='*66}", flush=True)
    print(f"  Robot:            {robot.upper()}", flush=True)
    print(f"  Envs:             {env_cfg.scene.num_envs}", flush=True)
    print(f"  Max iterations:   {agent_cfg.max_iterations}", flush=True)
    print(f"  Save interval:    {agent_cfg.save_interval}", flush=True)
    print(f"  LR schedule:      cosine {args_cli.lr_max} -> {args_cli.lr_min}", flush=True)
    print(f"  Warmup iters:     {args_cli.warmup_iters}", flush=True)
    print(f"  Entropy coef:     {agent_cfg.algorithm.entropy_coef}", flush=True)
    print(f"  Mini-batches:     {agent_cfg.algorithm.num_mini_batches}", flush=True)
    print(f"  Steps per env:    {agent_cfg.num_steps_per_env}", flush=True)
    print(f"  Network:          {agent_cfg.policy.actor_hidden_dims}", flush=True)
    print(f"  Episode length:   {env_cfg.episode_length_s}s", flush=True)
    print(f"  Logger:           {agent_cfg.logger}", flush=True)
    print(f"  Log dir:          {log_dir}", flush=True)
    print(f"  Steps/iteration:  {steps_per_iter:,}", flush=True)
    print(f"  Est. total steps: {steps_per_iter * agent_cfg.max_iterations / 1e9:.1f}B", flush=True)

    if robot == "vision60":
        print(f"  DR expansion:     {args_cli.dr_expansion_iters} iterations", flush=True)
        print(f"  Progressive DR Schedule:", flush=True)
        print(f"    Friction:  [{DR_SCHEDULE['static_friction_min'][0]}, "
              f"{DR_SCHEDULE['static_friction_max'][0]}] -> "
              f"[{DR_SCHEDULE['static_friction_min'][1]}, "
              f"{DR_SCHEDULE['static_friction_max'][1]}]", flush=True)

    print(f"{'='*70}\n", flush=True)

    # ── Create environment ──────────────────────────────────────────────
    EnvCfgClass = type(env_cfg)
    gym.register(
        id=env_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{EnvCfgClass.__module__}:{EnvCfgClass.__name__}",
        },
    )
    env = gym.make(env_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── Create runner ───────────────────────────────────────────────────
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )

    # ── Resume if requested ─────────────────────────────────────────────
    start_iteration = 0
    if args_cli.resume:
        resume_path = get_checkpoint_path(
            log_root_path, args_cli.load_run, args_cli.load_checkpoint
        )
        print(f"[INFO] Resuming from: {resume_path}", flush=True)
        runner.load(resume_path)
        ckpt_name = os.path.basename(resume_path)
        try:
            start_iteration = int(ckpt_name.split("_")[1].split(".")[0]) + 1
            print(f"[INFO] Resuming from iteration {start_iteration}", flush=True)
        except (IndexError, ValueError):
            print("[WARN] Could not parse iteration from checkpoint name", flush=True)

    # ── Save config ─────────────────────────────────────────────────────
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    with open(os.path.join(log_dir, "params", "training_params.txt"), "w") as f:
        f.write(f"robot: {robot}\n")
        f.write(f"num_envs: {env_cfg.scene.num_envs}\n")
        f.write(f"max_iterations: {agent_cfg.max_iterations}\n")
        f.write(f"lr_max: {args_cli.lr_max}\n")
        f.write(f"lr_min: {args_cli.lr_min}\n")
        f.write(f"warmup_iters: {args_cli.warmup_iters}\n")
        f.write(f"seed: {args_cli.seed}\n")
        f.write(f"steps_per_iter: {steps_per_iter}\n")
        f.write(f"network: {agent_cfg.policy.actor_hidden_dims}\n")
        f.write(f"logger: {agent_cfg.logger}\n")

    # ── W&B custom logging setup ────────────────────────────────────────
    wandb_run = None
    if agent_cfg.logger == "wandb":
        try:
            import wandb
            # RSL-RL OnPolicyRunner initializes W&B internally,
            # but we grab the active run for custom metric logging
            wandb_run = wandb.run
            if wandb_run is not None:
                print(f"[W&B] Active run: {wandb_run.name} ({wandb_run.url})", flush=True)
        except ImportError:
            print("[WARN] wandb not installed — falling back to TensorBoard", flush=True)

    # ── Training loop ───────────────────────────────────────────────────
    print(f"\n[TRAIN] Starting {robot.upper()} training with cosine annealing LR...",
          flush=True)
    start_time = time.time()

    # Set initial learning rate
    initial_lr = cosine_annealing_lr(
        start_iteration, agent_cfg.max_iterations,
        args_cli.lr_max, args_cli.lr_min, args_cli.warmup_iters
    )
    set_learning_rate(runner, initial_lr)
    print(f"[TRAIN] Initial LR: {initial_lr:.2e}", flush=True)

    # Monkey-patch the PPO update with LR schedule + DR
    original_update = runner.alg.update
    _lr_log_interval = 1000
    _iteration_counter = [start_iteration]

    def update_with_schedule(*args, **kwargs):
        """Wrapper: cosine LR + progressive DR (if Vision60) + noise clamp."""
        it = _iteration_counter[0]

        # Cosine LR annealing
        lr = cosine_annealing_lr(
            it, agent_cfg.max_iterations,
            args_cli.lr_max, args_cli.lr_min, args_cli.warmup_iters
        )
        set_learning_rate(runner, lr)

        # Progressive DR for Vision60
        dr_info = None
        if robot == "vision60":
            dr_info = update_dr_params(env, it, args_cli.dr_expansion_iters)

        # Run the actual PPO update
        result = original_update(*args, **kwargs)

        # Safety: clamp noise std
        clamp_noise_std(runner.alg.policy, args_cli.min_noise_std, args_cli.max_noise_std)

        # Custom W&B logging
        if wandb_run is not None and it % 100 == 0:
            import wandb
            log_data = {"lr": lr}
            if dr_info is not None:
                log_data["dr/fraction"] = dr_info["dr_fraction"]
                log_data["dr/push_vel"] = dr_info["push_vel"]
                log_data["dr/ext_force"] = dr_info["ext_force"]
            wandb.log(log_data, step=it)

        # Periodic console logging
        if it % _lr_log_interval == 0:
            elapsed = time.time() - start_time
            hours = elapsed / 3600
            fps = (it - start_iteration) * steps_per_iter / max(elapsed, 1) if it > start_iteration else 0
            noise = runner.alg.policy.std.mean().item() if hasattr(runner.alg.policy, 'std') else 0

            dr_str = ""
            if dr_info is not None:
                dr_str = f"  dr={dr_info['dr_fraction']:.1%}"

            print(
                f"[TRAIN] iter={it:6d}/{agent_cfg.max_iterations}  "
                f"lr={lr:.2e}{dr_str}  "
                f"noise={noise:.3f}  "
                f"elapsed={hours:.1f}h  fps={fps:.0f}",
                flush=True,
            )

        _iteration_counter[0] += 1
        return result

    runner.alg.update = update_with_schedule

    # Run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations - start_iteration,
        init_at_random_ep_len=True,
    )

    # ── Training complete ───────────────────────────────────────────────
    elapsed = time.time() - start_time
    hours = elapsed / 3600
    total_steps = (agent_cfg.max_iterations - start_iteration) * steps_per_iter

    print(f"\n{'='*70}", flush=True)
    print(f"  {robot.upper()} TRAINING COMPLETE", flush=True)
    print(f"  {'='*66}", flush=True)
    print(f"  Total time:       {hours:.1f} hours ({elapsed:.0f} seconds)", flush=True)
    print(f"  Total steps:      {total_steps / 1e9:.1f}B", flush=True)
    print(f"  Avg throughput:   {total_steps / elapsed:.0f} steps/sec", flush=True)
    print(f"  Checkpoints:      {log_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    # Use os._exit(0) to avoid CUDA deadlock on close
    import os as _os
    _os._exit(0)
