"""
100-Hour Spot Multi-Terrain Robust Locomotion Training — H100 NVL 96GB
======================================================================

Standalone training script for RSL-RL PPO with:
  - ROBUST_TERRAINS_CFG (12 terrain types, 400 patches)
  - Massively expanded domain randomization (friction 0.05–1.5)
  - Enhanced MLP [1024, 512, 256] — ~2M parameters
  - Cosine annealing LR: 1e-3 → 1e-5 over 60,000 iterations
  - 20,480 parallel environments on H100 NVL 96GB

Usage (on H100 server):
    cd ~/IsaacLab
    ./isaaclab.sh -p /path/to/train_100hr.py --headless --num_envs 20480

Usage (local debug — RTX 2000 Ada):
    cd C:\\IsaacLab
    isaaclab.bat -p /path/to/train_100hr.py --headless --num_envs 64 --max_iterations 100

Created for AI2C Tech Capstone — MS for Autonomy, February 2026
"""

# ── 0. Parse args BEFORE any Isaac imports ──────────────────────────────
import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="100hr Spot multi-terrain robust training")
parser.add_argument("--num_envs", type=int, default=20480,
                    help="Number of parallel environments (default 20480 for H100 NVL)")
parser.add_argument("--max_iterations", type=int, default=60000,
                    help="Max training iterations (default 60000 for ~100hr)")
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
parser.add_argument("--warmup_iters", type=int, default=3000,
                    help="LR warmup iterations (linear ramp from lr_min to lr_max)")
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

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils import configclass

# Trigger standard gym registrations
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path

# Add our project to path for custom configs
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from configs.env_cfg import Spot100hrEnvCfg
from configs.ppo_cfg import Spot100hrPPORunnerCfg

# Enable TF32 for faster matmul on H100
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# ── 2. Cosine Annealing LR Schedule ────────────────────────────────────

def cosine_annealing_lr(iteration: int, max_iterations: int,
                        lr_max: float, lr_min: float,
                        warmup_iters: int) -> float:
    """Compute learning rate with linear warmup + cosine annealing.

    Args:
        iteration: Current training iteration.
        max_iterations: Total training iterations.
        lr_max: Peak learning rate (after warmup).
        lr_min: Minimum learning rate (end of cosine decay).
        warmup_iters: Number of warmup iterations (linear ramp).

    Returns:
        Learning rate for this iteration.
    """
    if iteration < warmup_iters:
        # Linear warmup: lr_min → lr_max over warmup_iters
        return lr_min + (lr_max - lr_min) * (iteration / warmup_iters)
    else:
        # Cosine annealing: lr_max → lr_min over remaining iterations
        progress = (iteration - warmup_iters) / max(1, max_iterations - warmup_iters)
        return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def set_learning_rate(runner: OnPolicyRunner, lr: float):
    """Override the learning rate in the PPO optimizer."""
    for param_group in runner.alg.optimizer.param_groups:
        param_group["lr"] = lr


# ── 3. Main ─────────────────────────────────────────────────────────────

def main():
    # --- Environment config ---
    env_cfg = Spot100hrEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # --- Agent config ---
    agent_cfg = Spot100hrPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed

    # --- Logging ---
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    print(f"\n{'='*70}", flush=True)
    print(f"  100-HOUR SPOT MULTI-TERRAIN ROBUST TRAINING", flush=True)
    print(f"  {'='*66}", flush=True)
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
    print(f"  Friction range:   {env_cfg.events.physics_material.params['static_friction_range']}", flush=True)
    print(f"  Log dir:          {log_dir}", flush=True)
    steps_per_iter = env_cfg.scene.num_envs * agent_cfg.num_steps_per_env
    print(f"  Steps/iteration:  {steps_per_iter:,}", flush=True)
    print(f"  Est. total steps: {steps_per_iter * agent_cfg.max_iterations / 1e9:.1f}B", flush=True)
    print(f"{'='*70}\n", flush=True)

    # --- Create environment ---
    # We register our custom env inline since it's not in the IsaacLab task registry
    gym.register(
        id="Isaac-Velocity-Robust-Spot-100hr-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{Spot100hrEnvCfg.__module__}:{Spot100hrEnvCfg.__name__}",
        },
    )
    env = gym.make("Isaac-Velocity-Robust-Spot-100hr-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # --- Create runner ---
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )

    # --- Resume if requested ---
    start_iteration = 0
    if args_cli.resume:
        resume_path = get_checkpoint_path(
            log_root_path, args_cli.load_run, args_cli.load_checkpoint
        )
        print(f"[INFO] Resuming from: {resume_path}", flush=True)
        runner.load(resume_path)
        # Extract iteration number from checkpoint filename
        ckpt_name = os.path.basename(resume_path)
        try:
            start_iteration = int(ckpt_name.split("_")[1].split(".")[0]) + 1
            print(f"[INFO] Resuming from iteration {start_iteration}", flush=True)
        except (IndexError, ValueError):
            print("[WARN] Could not parse iteration from checkpoint name", flush=True)

    # --- Save config ---
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # --- Save training parameters for reproducibility ---
    with open(os.path.join(log_dir, "params", "training_params.txt"), "w") as f:
        f.write(f"num_envs: {env_cfg.scene.num_envs}\n")
        f.write(f"max_iterations: {agent_cfg.max_iterations}\n")
        f.write(f"lr_max: {args_cli.lr_max}\n")
        f.write(f"lr_min: {args_cli.lr_min}\n")
        f.write(f"warmup_iters: {args_cli.warmup_iters}\n")
        f.write(f"seed: {args_cli.seed}\n")
        f.write(f"steps_per_iter: {steps_per_iter}\n")
        f.write(f"network: {agent_cfg.policy.actor_hidden_dims}\n")
        f.write(f"episode_length_s: {env_cfg.episode_length_s}\n")
        f.write(f"friction_range: {env_cfg.events.physics_material.params['static_friction_range']}\n")

    # --- Custom training loop with cosine annealing ---
    print("\n[TRAIN] Starting training with cosine annealing LR schedule...", flush=True)
    start_time = time.time()

    # Set initial learning rate
    initial_lr = cosine_annealing_lr(
        start_iteration, agent_cfg.max_iterations,
        args_cli.lr_max, args_cli.lr_min, args_cli.warmup_iters
    )
    set_learning_rate(runner, initial_lr)
    print(f"[TRAIN] Initial LR: {initial_lr:.2e}", flush=True)

    # Use RSL-RL's built-in learn() but hook into its iteration callback
    # to update the LR. We override by monkey-patching the alg's lr.
    #
    # RSL-RL's OnPolicyRunner.learn() calls self.alg.update() each iteration,
    # which reads lr from the optimizer. We update it before each iteration.

    # Store the original update method
    original_update = runner.alg.update

    _lr_log_interval = 1000
    _iteration_counter = [start_iteration]  # mutable for closure

    def update_with_lr_schedule(*args, **kwargs):
        """Wrapper that applies cosine LR before each PPO update."""
        it = _iteration_counter[0]
        lr = cosine_annealing_lr(
            it, agent_cfg.max_iterations,
            args_cli.lr_max, args_cli.lr_min, args_cli.warmup_iters
        )
        set_learning_rate(runner, lr)

        if it % _lr_log_interval == 0:
            elapsed = time.time() - start_time
            hours = elapsed / 3600
            fps = (it - start_iteration) * steps_per_iter / max(elapsed, 1)
            print(
                f"[TRAIN] iter={it:6d}/{agent_cfg.max_iterations}  "
                f"lr={lr:.2e}  elapsed={hours:.1f}h  fps={fps:.0f}",
                flush=True,
            )

        _iteration_counter[0] += 1
        return original_update(*args, **kwargs)

    runner.alg.update = update_with_lr_schedule

    # Run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations - start_iteration,
        init_at_random_ep_len=True,
    )

    # --- Training complete ---
    elapsed = time.time() - start_time
    hours = elapsed / 3600
    total_steps = (agent_cfg.max_iterations - start_iteration) * steps_per_iter

    print(f"\n{'='*70}", flush=True)
    print(f"  TRAINING COMPLETE", flush=True)
    print(f"  {'='*66}", flush=True)
    print(f"  Total time:       {hours:.1f} hours ({elapsed:.0f} seconds)", flush=True)
    print(f"  Total steps:      {total_steps / 1e9:.1f}B", flush=True)
    print(f"  Avg throughput:   {total_steps / elapsed:.0f} steps/sec", flush=True)
    print(f"  Checkpoints:      {log_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
