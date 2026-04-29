#!/usr/bin/env python3
"""
48-Hour Optimized PPO Training: Boston Dynamics Spot - Flat Terrain
===================================================================
Capstone Project - Immersive Modeling & Simulation for Autonomy

Focus Areas:
  1. LOCOMOTION  - Enhanced velocity tracking + gait coordination rewards
  2. PROPRIOCEPTION - Body state awareness, orientation control, foot clearance
  3. JOINT EFFICIENCY - Torque penalties, action smoothness, joint velocity limits

Hardware Target: NVIDIA H100 NVL (96GB VRAM)
  - Environments: 8,192 (sustained safe thermal: 65% GPU, 49C, 188W)
  - Estimated iterations: ~17,000 in 48 hours
  - Total timesteps: ~3.3 billion

Task: Isaac-Velocity-Flat-Spot-v0 (built-in Boston Dynamics Spot)

Built-in Spot Reward Structure (modified for our focus):
  TASK REWARDS:
    - air_time: Proper foot swing timing
    - base_angular_velocity: Turn tracking
    - base_linear_velocity: Speed tracking with ramp
    - foot_clearance: Lift feet properly (proprioception!)
    - gait: Trot gait coordination (FL-HR, FR-HL synced)
  PENALTY REWARDS:
    - action_smoothness: Smooth actuator commands
    - air_time_variance: Consistent foot timing
    - base_motion: Minimize unnecessary body motion
    - base_orientation: Stay upright (proprioception!)
    - foot_slip: Don't slide feet
    - joint_acc: Smooth joint trajectories (efficiency!)
    - joint_pos: Stay near default pose when standing
    - joint_torques: Minimize energy usage (efficiency!)
    - joint_vel: Limit joint speeds (efficiency!)

Usage:
  python train_48h_spot.py [--num_envs 8192] [--max_iterations 17000]
  python train_48h_spot.py --resume --checkpoint /path/to/model.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

# ─── CLI Arguments ───────────────────────────────────────────────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="48h Optimized Spot Training")
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
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

# ─── Launch Simulator ────────────────────────────────────────────────────────
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─── Post-Launch Imports ─────────────────────────────────────────────────────
import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# ─── Constants ───────────────────────────────────────────────────────────────
TASK_NAME = "Isaac-Velocity-Flat-Spot-v0"

# ─── Logging Setup ───────────────────────────────────────────────────────────
if args_cli.run_name:
    run_name = args_cli.run_name
else:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"48h_spot_optimized_{timestamp}"

log_dir = f"/home/t2user/IsaacLab/logs/rsl_rl/spot_48h/{run_name}"
os.makedirs(log_dir, exist_ok=True)

# ─── Banner ──────────────────────────────────────────────────────────────────
print("=" * 72)
print("  48-HOUR OPTIMIZED PPO TRAINING")
print("  Boston Dynamics Spot - Locomotion / Proprioception / Efficiency")
print("=" * 72)
print(f"  Task:            {TASK_NAME}")
print(f"  Environments:    {args_cli.num_envs:,}")
print(f"  Max iterations:  {args_cli.max_iterations:,}")
print(f"  Log directory:   {log_dir}")
print(f"  Resume:          {args_cli.resume}")
print(f"  Device:          cuda:0")

steps_per_iter = args_cli.num_envs * 24
total_steps = args_cli.max_iterations * steps_per_iter
est_hours = (args_cli.max_iterations * 10) / 3600
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
print("[2/4] Applying optimized reward weights for Spot...\n")

reward_changes = {}


def modify_reward(reward_term, new_weight, description):
    """Modify a reward weight and log the change."""
    old_weight = reward_term.weight
    reward_term.weight = new_weight
    if old_weight == 0.0:
        status = "ENABLED"
    elif old_weight != 0:
        ratio = new_weight / old_weight
        status = f"{ratio:.1f}x"
    else:
        status = "NEW"
    reward_changes[description] = (old_weight, new_weight, status)


# ═══════════════════════════════════════════════════════════════════════════
# LOCOMOTION REWARDS (enhanced tracking + gait quality)
# ═══════════════════════════════════════════════════════════════════════════
modify_reward(env_cfg.rewards.base_linear_velocity, 7.5,
              "Linear velocity tracking")        # 5.0 -> 7.5 (1.5x)
modify_reward(env_cfg.rewards.base_angular_velocity, 7.5,
              "Angular velocity tracking")       # 5.0 -> 7.5 (1.5x)
modify_reward(env_cfg.rewards.gait, 15.0,
              "Trot gait coordination")          # 10.0 -> 15.0 (1.5x)
modify_reward(env_cfg.rewards.air_time, 7.5,
              "Foot air time (swing phase)")     # 5.0 -> 7.5 (1.5x)

# ═══════════════════════════════════════════════════════════════════════════
# PROPRIOCEPTION REWARDS (body awareness + stability)
# ═══════════════════════════════════════════════════════════════════════════
modify_reward(env_cfg.rewards.base_orientation, -5.0,
              "Base orientation (PROPRIOCEPTION)")  # -3.0 -> -5.0 (1.7x)
modify_reward(env_cfg.rewards.base_motion, -3.0,
              "Base motion penalty (STABILITY)")    # -2.0 -> -3.0 (1.5x)
modify_reward(env_cfg.rewards.foot_clearance, 1.0,
              "Foot clearance (PROPRIOCEPTION)")    # 0.5 -> 1.0 (2.0x)
modify_reward(env_cfg.rewards.foot_slip, -1.0,
              "Foot slip penalty (STABILITY)")      # -0.5 -> -1.0 (2.0x)

# ═══════════════════════════════════════════════════════════════════════════
# JOINT EFFICIENCY REWARDS (energy + smoothness)
# ═══════════════════════════════════════════════════════════════════════════
modify_reward(env_cfg.rewards.joint_torques, -0.002,
              "Joint torques (EFFICIENCY)")         # -5e-4 -> -0.002 (4x)
modify_reward(env_cfg.rewards.joint_acc, -0.0004,
              "Joint acceleration (EFFICIENCY)")    # -1e-4 -> -4e-4 (4x)
modify_reward(env_cfg.rewards.joint_vel, -0.02,
              "Joint velocity (EFFICIENCY)")        # -1e-2 -> -0.02 (2x)
modify_reward(env_cfg.rewards.action_smoothness, -2.0,
              "Action smoothness (EFFICIENCY)")     # -1.0 -> -2.0 (2x)
modify_reward(env_cfg.rewards.air_time_variance, -2.0,
              "Air time variance (GAIT QUALITY)")   # -1.0 -> -2.0 (2x)
modify_reward(env_cfg.rewards.joint_pos, -1.0,
              "Joint position penalty")             # -0.7 -> -1.0 (1.4x)

# Print reward modification table
print("  ┌──────────────────────────────────────┬──────────┬──────────┬─────────┐")
print("  │ Reward Term                          │ Default  │ Modified │ Change  │")
print("  ├──────────────────────────────────────┼──────────┼──────────┼─────────┤")
for desc, (old, new, status) in reward_changes.items():
    print(f"  │ {desc:<38s} │ {old:>8.4f} │ {new:>8.4f} │ {status:>7s} │")
print("  └──────────────────────────────────────┴──────────┴──────────┴─────────┘")

# Save reward config to file
reward_log_path = os.path.join(log_dir, "reward_config.txt")
with open(reward_log_path, "w") as f:
    f.write("48-Hour Spot Training - Reward Configuration\n")
    f.write("=" * 60 + "\n\n")
    for desc, (old, new, status) in reward_changes.items():
        f.write(f"{desc}: {old} -> {new} ({status})\n")
print(f"\n  Reward config saved to: {reward_log_path}")


# =============================================================================
# STEP 2: CREATE ENVIRONMENT
# =============================================================================
print("\n[3/4] Creating Spot environment...")
env = gym.make(TASK_NAME, cfg=env_cfg)
env = RslRlVecEnvWrapper(env)
print(f"  Num envs:        {env.num_envs}")


# =============================================================================
# STEP 3: OPTIMIZED PPO AGENT CONFIGURATION
# =============================================================================
print("\n[4/4] Configuring optimized PPO agent for Spot...")

agent_cfg = {
    "seed": 42,
    "device": "cuda:0",
    "num_steps_per_env": 24,
    "max_iterations": args_cli.max_iterations,
    "empirical_normalization": False,
    "obs_groups": {
        "policy": ["policy"],
        "critic": ["policy"],
    },
    "policy": {
        "class_name": "ActorCritic",
        "init_noise_std": 0.8,                      # default: 1.0
        "actor_obs_normalization": False,
        "critic_obs_normalization": False,
        "actor_hidden_dims": [1024, 512, 256],       # default: [512, 256, 128]
        "critic_hidden_dims": [1024, 512, 256],      # default: [512, 256, 128]
        "activation": "elu",
    },
    "algorithm": {
        "class_name": "PPO",
        "value_loss_coef": 0.5,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.0025,                      # same as Spot default
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

# Save agent config
agent_log_path = os.path.join(log_dir, "agent_config.txt")
with open(agent_log_path, "w") as f:
    f.write("48-Hour Spot Training - PPO Agent Configuration\n")
    f.write("=" * 60 + "\n\n")
    import json
    f.write(json.dumps(agent_cfg, indent=2))
print(f"  Agent config saved to: {agent_log_path}")

print("\n  PPO Optimizations for Spot:")
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

if args_cli.resume:
    if args_cli.checkpoint:
        checkpoint_path = args_cli.checkpoint
    else:
        checkpoint_path = get_checkpoint_path(log_dir)
    print(f"\n  Resuming from checkpoint: {checkpoint_path}")
    runner.load(checkpoint_path)

# =============================================================================
# BEGIN TRAINING
# =============================================================================
print("\n" + "=" * 72)
print("  TRAINING STARTED - BOSTON DYNAMICS SPOT")
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
print("  TRAINING COMPLETE - BOSTON DYNAMICS SPOT")
print(f"  End time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Total duration:  {hours:.2f} hours ({elapsed:.0f} seconds)")
print(f"  Iterations:      {args_cli.max_iterations:,}")
print(f"  Total timesteps: {total_steps:,.0f}")
print(f"  Avg iter time:   {elapsed/args_cli.max_iterations:.2f} seconds")
print(f"  Logs saved to:   {log_dir}")
print("=" * 72)

env.close()
simulation_app.close()
