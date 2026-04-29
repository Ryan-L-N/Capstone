"""
48-Hour Spot Rough Terrain Training — H100 NVL
===============================================

Standalone training script for RSL-RL PPO on Spot rough terrain.
Imports existing Isaac Lab Spot configs and applies reward / PPO overrides
for improved proprioception, joint efficiency, and locomotion quality.

Usage (on H100 server):
    cd ~/IsaacLab
    ./isaaclab.sh -p /path/to/spot_rough_48h_cfg.py --headless --num_envs 8192

Created for AI2C Tech Capstone — MS for Autonomy, February 2026
"""

# ── 0. Parse args BEFORE any Isaac imports ──────────────────────────────
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="48h Spot rough terrain training")
parser.add_argument("--num_envs", type=int, default=8192,
                    help="Number of parallel environments (default 8192 for H100)")
parser.add_argument("--max_iterations", type=int, default=30000,
                    help="Max training iterations (default 30000 for ~48h)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--resume", action="store_true", default=False,
                    help="Resume from latest checkpoint")
parser.add_argument("--load_run", type=str, default=None,
                    help="Run directory to resume from")
parser.add_argument("--load_checkpoint", type=str, default="model_.*\\.pt",
                    help="Checkpoint regex to load")
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

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)
from isaaclab.utils import configclass

# Trigger gym registrations
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path

# Import existing Spot rough terrain configs
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.rough_env_cfg import (
    SpotRoughEnvCfg,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# ── 2. Custom PPO Config ────────────────────────────────────────────────

@configclass
class SpotRough48hPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config tuned for 48-hour training on H100.

    Changes from SpotRoughPPORunnerCfg:
    - 30,000 iterations (up from 5,000)
    - Lower LR: 3e-4 (from 1e-3) for stability over long training
    - Higher entropy: 0.008 (from 0.005) for more exploration
    - 8 mini-batches (from 4) to match 8,192 envs
    - save_interval: 500 (60 checkpoints over full run)
    - init_noise_std: 0.8 (from 1.0) for less initial noise
    """
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "spot_rough"
    run_name = "48h_proprioception"
    store_code_state = False
    seed = 42
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# ── 3. Apply Reward Overrides ───────────────────────────────────────────

def apply_reward_overrides(env_cfg: SpotRoughEnvCfg):
    """Apply 48h reward weight overrides for proprioception + efficiency.

    These override the weights defined in SpotRoughRewardsCfg.
    The reward FUNCTIONS remain identical — only weights change.
    """
    r = env_cfg.rewards

    # ── Positive rewards (task) ──
    # base_linear_velocity: +5.0 → +7.0 (stronger velocity tracking)
    r.base_linear_velocity.weight = 7.0
    # foot_clearance: +2.0 → +2.5 (higher stepping for stairs)
    r.foot_clearance.weight = 2.5
    # gait, air_time, base_angular_velocity: keep as-is

    # ── Penalties (efficiency + proprioception) ──
    # action_smoothness: -1.0 → -2.0 (smoother actuation)
    r.action_smoothness.weight = -2.0
    # base_motion: -2.0 → -3.0 (less vertical bouncing)
    r.base_motion.weight = -3.0
    # base_orientation: -3.0 → -5.0 (stronger upright incentive)
    r.base_orientation.weight = -5.0
    # foot_slip: -0.5 → -1.0 (less foot sliding)
    r.foot_slip.weight = -1.0
    # joint_acc: -1e-4 → -5e-4 (smoother joint trajectories)
    r.joint_acc.weight = -5.0e-4
    # joint_pos: -0.7 → -1.0 (stay closer to default stance)
    r.joint_pos.weight = -1.0
    # joint_torques: -5e-4 → -2e-3 (energy efficiency)
    r.joint_torques.weight = -2.0e-3
    # joint_vel: -1e-2 → -2e-2 (slower, controlled movements)
    r.joint_vel.weight = -2.0e-2

    print("\n[48h] Reward weight overrides applied:")
    print(f"  base_linear_velocity : +7.0  (was +5.0)")
    print(f"  foot_clearance       : +2.5  (was +2.0)")
    print(f"  action_smoothness    : -2.0  (was -1.0)")
    print(f"  base_motion          : -3.0  (was -2.0)")
    print(f"  base_orientation     : -5.0  (was -3.0)")
    print(f"  foot_slip            : -1.0  (was -0.5)")
    print(f"  joint_acc            : -5e-4 (was -1e-4)")
    print(f"  joint_pos            : -1.0  (was -0.7)")
    print(f"  joint_torques        : -2e-3 (was -5e-4)")
    print(f"  joint_vel            : -2e-2 (was -1e-2)")


def apply_event_overrides(env_cfg: SpotRoughEnvCfg):
    """Enable external force perturbation (was zeroed out)."""
    evt = env_cfg.events.base_external_force_torque
    if evt is not None:
        evt.params["force_range"] = (-3.0, 3.0)
        evt.params["torque_range"] = (-1.0, 1.0)
        print("[48h] External force perturbation enabled: ±3.0 N, ±1.0 Nm")


# ── 4. Main ─────────────────────────────────────────────────────────────

def main():
    # --- Environment config ---
    env_cfg = SpotRoughEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # Apply our reward + event overrides
    apply_reward_overrides(env_cfg)
    apply_event_overrides(env_cfg)

    # --- Agent config ---
    agent_cfg = SpotRough48hPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed

    # --- Logging ---
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    print(f"\n{'='*70}")
    print(f"  48-HOUR SPOT ROUGH TERRAIN TRAINING")
    print(f"  {'='*66}")
    print(f"  Envs:           {env_cfg.scene.num_envs}")
    print(f"  Max iterations: {agent_cfg.max_iterations}")
    print(f"  Save interval:  {agent_cfg.save_interval}")
    print(f"  Learning rate:  {agent_cfg.algorithm.learning_rate}")
    print(f"  Entropy coef:   {agent_cfg.algorithm.entropy_coef}")
    print(f"  Mini-batches:   {agent_cfg.algorithm.num_mini_batches}")
    print(f"  Network:        {agent_cfg.policy.actor_hidden_dims}")
    print(f"  Log dir:        {log_dir}")
    print(f"{'='*70}\n")

    # --- Create environment ---
    env = gym.make("Isaac-Velocity-Rough-Spot-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # --- Create runner ---
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )

    # --- Resume if requested ---
    if args_cli.resume:
        resume_path = get_checkpoint_path(
            log_root_path, args_cli.load_run, args_cli.load_checkpoint
        )
        print(f"[INFO] Resuming from: {resume_path}")
        runner.load(resume_path)

    # --- Save config ---
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # --- Train ---
    start_time = time.time()
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )
    elapsed = time.time() - start_time

    hours = elapsed / 3600
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"  Total time: {hours:.1f} hours ({elapsed:.0f} seconds)")
    print(f"  Checkpoints: {log_dir}")
    print(f"{'='*70}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
