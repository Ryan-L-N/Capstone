#!/usr/bin/env python3
"""
48-Hour Optimized PPO Training: ANYmal-C Rough Terrain
======================================================
Capstone Project - Immersive Modeling & Simulation for Autonomy

Focus Areas:
  1. LOCOMOTION  - Enhanced velocity tracking + gait quality rewards
  2. PROPRIOCEPTION - Leveraging joint state observations with stable base
  3. JOINT EFFICIENCY - Heavy torque/energy penalties for smooth, efficient motion

Hardware Target: NVIDIA H100 NVL (96GB VRAM)
  - Environments: 8,192 (sustained safe thermal: 65% GPU, 49C, 188W)
  - Estimated iterations: ~17,000 in 48 hours
  - Total timesteps: ~3.3 billion

PPO Optimizations (vs default):
  - Deeper networks: [1024, 512, 256] for complex proprioceptive reasoning
  - Lower LR (3e-4) with adaptive schedule for stable long-term training
  - Higher GAE lambda (0.97) for long-horizon credit assignment
  - 8 learning epochs x 8 mini-batches per update
  - Tighter KL constraint (0.008) for conservative policy updates

Reward Engineering (vs default):
  - Joint torques penalty: 20x increase -> forces efficient actuator usage
  - Joint acceleration penalty: 10x increase -> smooth joint trajectories
  - Action rate penalty: 2.5x increase -> continuous, smooth control signals
  - Gait quality (feet air time): 2x increase -> proper walking patterns
  - Base stability: 2x increase -> leverages proprioceptive sensing
  - Flat orientation: ENABLED -> maintains upright posture
  - Joint limits: ENABLED -> respects mechanical constraints

Usage:
  python train_48h_anymal_c.py [--num_envs 8192] [--max_iterations 17000]
  python train_48h_anymal_c.py --resume --checkpoint /path/to/model.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

# ─── CLI Arguments ───────────────────────────────────────────────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="48h Optimized ANYmal-C Training")
parser.add_argument("--num_envs", type=int, default=8192,
                    help="Number of parallel environments (default: 8192)")
parser.add_argument("--max_iterations", type=int, default=17000,
                    help="Maximum training iterations (default: 17000 ~ 48h)")
parser.add_argument("--resume", action="store_true", default=False,
                    help="Resume from last checkpoint")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint .pt file for resuming")
parser.add_argument("--run_name", type=str, default=None,
                    help="Custom run name for logging directory")
# Add AppLauncher args (handles --headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True  # Force headless for H100 server

# ─── Launch Simulator ────────────────────────────────────────────────────────
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─── Post-Launch Imports (after Omniverse is initialized) ────────────────────
import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401 - registers all Isaac Lab tasks
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# ─── Constants ───────────────────────────────────────────────────────────────
TASK_NAME = "Isaac-Velocity-Rough-Anymal-C-v0"

# ─── Logging Setup ───────────────────────────────────────────────────────────
if args_cli.run_name:
    run_name = args_cli.run_name
else:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"48h_optimized_{timestamp}"

log_dir = f"/home/t2user/IsaacLab/logs/rsl_rl/anymal_c_48h/{run_name}"
os.makedirs(log_dir, exist_ok=True)

# ─── Banner ──────────────────────────────────────────────────────────────────
print("=" * 72)
print("  48-HOUR OPTIMIZED PPO TRAINING")
print("  ANYmal-C Rough Terrain - Locomotion / Proprioception / Efficiency")
print("=" * 72)
print(f"  Task:            {TASK_NAME}")
print(f"  Environments:    {args_cli.num_envs:,}")
print(f"  Max iterations:  {args_cli.max_iterations:,}")
print(f"  Log directory:   {log_dir}")
print(f"  Resume:          {args_cli.resume}")
print(f"  Device:          cuda:0")

steps_per_iter = args_cli.num_envs * 24
total_steps = args_cli.max_iterations * steps_per_iter
est_hours = (args_cli.max_iterations * 10) / 3600  # ~10s per iter estimate
print(f"  Steps/iteration: {steps_per_iter:,}")
print(f"  Total timesteps: {total_steps:,.0f} ({total_steps/1e9:.2f} billion)")
print(f"  Est. duration:   ~{est_hours:.1f} hours")
print("=" * 72)


# =============================================================================
# STEP 1: ENVIRONMENT CONFIGURATION
# =============================================================================
print("\n[1/4] Loading environment configuration...")
env_cfg = parse_env_cfg(TASK_NAME, device="cuda:0", num_envs=args_cli.num_envs)

# ─── Reward Weight Modifications ─────────────────────────────────────────────
# Focus: Locomotion quality + Proprioceptive awareness + Joint efficiency
print("[2/4] Applying optimized reward weights...\n")

# Store original weights for comparison logging
reward_changes = {}


def modify_reward(reward_term, new_weight, description):
    """Modify a reward weight and log the change."""
    old_weight = reward_term.weight
    reward_term.weight = new_weight
    status = "ENABLED" if old_weight == 0.0 else f"{new_weight/old_weight:.0f}x" if old_weight != 0 else "NEW"
    reward_changes[description] = (old_weight, new_weight, status)


# LOCOMOTION: Stronger velocity tracking drives
modify_reward(env_cfg.rewards.track_lin_vel_xy_exp, 1.5,
              "Linear velocity tracking")
modify_reward(env_cfg.rewards.track_ang_vel_z_exp, 0.75,
              "Angular velocity tracking")

# PROPRIOCEPTION: Base stability forces policy to rely on proprioceptive obs
modify_reward(env_cfg.rewards.lin_vel_z_l2, -2.0,
              "Vertical velocity penalty")
modify_reward(env_cfg.rewards.ang_vel_xy_l2, -0.1,
              "Roll/pitch angular vel penalty")
modify_reward(env_cfg.rewards.flat_orientation_l2, -1.0,
              "Flat orientation (PROPRIOCEPTION)")

# JOINT EFFICIENCY: Heavy torque and smoothness penalties
modify_reward(env_cfg.rewards.dof_torques_l2, -0.0002,
              "Joint torques L2 (EFFICIENCY)")
modify_reward(env_cfg.rewards.dof_acc_l2, -2.5e-6,
              "Joint acceleration L2 (EFFICIENCY)")
modify_reward(env_cfg.rewards.action_rate_l2, -0.025,
              "Action rate L2 (SMOOTHNESS)")
modify_reward(env_cfg.rewards.dof_pos_limits, -5.0,
              "Joint position limits (SAFETY)")

# GAIT QUALITY: Better walking patterns
modify_reward(env_cfg.rewards.feet_air_time, 0.25,
              "Feet air time (GAIT)")
modify_reward(env_cfg.rewards.undesired_contacts, -1.0,
              "Undesired contacts penalty")

# Print reward modification table
print("  ┌─────────────────────────────────────┬──────────┬──────────┬─────────┐")
print("  │ Reward Term                         │ Default  │ Modified │ Change  │")
print("  ├─────────────────────────────────────┼──────────┼──────────┼─────────┤")
for desc, (old, new, status) in reward_changes.items():
    print(f"  │ {desc:<37s} │ {old:>8.5f} │ {new:>8.5f} │ {status:>7s} │")
print("  └─────────────────────────────────────┴──────────┴──────────┴─────────┘")

# Save reward config to file for reference
reward_log_path = os.path.join(log_dir, "reward_config.txt")
with open(reward_log_path, "w") as f:
    f.write("48-Hour Training - Reward Configuration\n")
    f.write("=" * 60 + "\n\n")
    for desc, (old, new, status) in reward_changes.items():
        f.write(f"{desc}: {old} -> {new} ({status})\n")
print(f"\n  Reward config saved to: {reward_log_path}")


# =============================================================================
# STEP 2: CREATE ENVIRONMENT
# =============================================================================
print("\n[3/4] Creating environment...")
env = gym.make(TASK_NAME, cfg=env_cfg)
env = RslRlVecEnvWrapper(env)
print(f"  Observation dim: {env.num_obs}")
print(f"  Action dim:      {env.num_actions}")
print(f"  Num envs:        {env.num_envs}")


# =============================================================================
# STEP 3: OPTIMIZED PPO AGENT CONFIGURATION
# =============================================================================
print("\n[4/4] Configuring optimized PPO agent...")

agent_cfg = {
    "seed": 42,
    "device": "cuda:0",
    "num_steps_per_env": 24,
    "max_iterations": args_cli.max_iterations,
    "empirical_normalization": False,
    "policy": {
        "class_name": "ActorCritic",
        "init_noise_std": 0.8,                      # default: 1.0
        "actor_hidden_dims": [1024, 512, 256],       # default: [512, 256, 128]
        "critic_hidden_dims": [1024, 512, 256],      # default: [512, 256, 128]
        "activation": "elu",
    },
    "algorithm": {
        "class_name": "PPO",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.005,
        "num_learning_epochs": 8,                    # default: 5
        "num_mini_batches": 8,                       # default: 4
        "learning_rate": 3e-4,                       # default: 1e-3
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.97,                                # default: 0.95
        "desired_kl": 0.008,                         # default: 0.01
        "max_grad_norm": 1.0,
    },
    "save_interval": 500,                            # default: 50
    "logger": "tensorboard",
}

# Save agent config to file for reference
agent_log_path = os.path.join(log_dir, "agent_config.txt")
with open(agent_log_path, "w") as f:
    f.write("48-Hour Training - PPO Agent Configuration\n")
    f.write("=" * 60 + "\n\n")
    import json
    f.write(json.dumps(agent_cfg, indent=2))
print(f"  Agent config saved to: {agent_log_path}")

print("\n  PPO Optimizations:")
print("  ├─ Networks:      [1024, 512, 256] (deeper for proprioception)")
print("  ├─ Learning rate:  3e-4 adaptive (stable for 48h)")
print("  ├─ GAE lambda:     0.97 (long-horizon credit assignment)")
print("  ├─ Epochs/batches: 8 x 8 (sample efficient)")
print("  ├─ Desired KL:     0.008 (conservative updates)")
print("  ├─ Init noise:     0.8 (focused exploration)")
print("  └─ Save interval:  every 500 iterations")


# =============================================================================
# STEP 4: CREATE RUNNER AND TRAIN
# =============================================================================
runner = OnPolicyRunner(env, agent_cfg, log_dir=log_dir, device="cuda:0")

# Resume from checkpoint if specified
if args_cli.resume:
    if args_cli.checkpoint:
        checkpoint_path = args_cli.checkpoint
    else:
        # Find latest checkpoint in log dir
        checkpoint_path = get_checkpoint_path(log_dir)
    print(f"\n  Resuming from checkpoint: {checkpoint_path}")
    runner.load(checkpoint_path)


# =============================================================================
# BEGIN TRAINING
# =============================================================================
print("\n" + "=" * 72)
print("  TRAINING STARTED")
print(f"  Start time:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Target:          {args_cli.max_iterations:,} iterations")
print(f"  Save interval:   every 500 iterations")
print(f"  Checkpoints at:  {log_dir}")
print("=" * 72 + "\n")

start_time = time.time()

runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)

elapsed = time.time() - start_time
hours = elapsed / 3600

# =============================================================================
# TRAINING COMPLETE
# =============================================================================
print("\n" + "=" * 72)
print("  TRAINING COMPLETE")
print(f"  End time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Total duration:  {hours:.2f} hours ({elapsed:.0f} seconds)")
print(f"  Iterations:      {args_cli.max_iterations:,}")
print(f"  Total timesteps: {total_steps:,.0f}")
print(f"  Avg iter time:   {elapsed/args_cli.max_iterations:.2f} seconds")
print(f"  Logs saved to:   {log_dir}")
print("=" * 72)

# Cleanup
env.close()
simulation_app.close()
