"""Inference / Visualization Script for Trained Policies.

Loads a trained checkpoint and runs the policy in a reduced-size environment
for visual inspection.

Usage:
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/multi_robot_training/play.py \\
        --robot spot --checkpoint /path/to/model_XXXXX.pt --num_envs 50

Created for AI2C Tech Capstone — MS for Autonomy, February 2026
"""

# ── 0. Parse args BEFORE any Isaac imports ──────────────────────────────
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play trained multi-robot policy")
parser.add_argument("--robot", type=str, required=True, choices=["spot", "vision60"],
                    help="Robot: spot or vision60")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to trained checkpoint")
parser.add_argument("--num_envs", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 1. Imports (AFTER SimulationApp) ────────────────────────────────────
import os

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# ── 2. Main ─────────────────────────────────────────────────────────────

def main():
    robot = args_cli.robot

    # Load PLAY variant env configs (reduced size, no DR)
    if robot == "spot":
        from configs.spot_ppo_env_cfg import SpotPPOEnvCfg_PLAY
        from configs.spot_ppo_cfg import SpotPPORunnerCfg
        env_cfg = SpotPPOEnvCfg_PLAY()
        agent_cfg = SpotPPORunnerCfg()
        env_id = "Isaac-Velocity-Play-Spot-v0"
        EnvCfgClass = SpotPPOEnvCfg_PLAY
    else:
        from configs.vision60_ppo_env_cfg import Vision60PPOEnvCfg
        from configs.vision60_ppo_cfg import Vision60PPORunnerCfg
        env_cfg = Vision60PPOEnvCfg()
        agent_cfg = Vision60PPORunnerCfg()
        env_id = "Isaac-Velocity-Play-Vision60-v0"
        EnvCfgClass = Vision60PPOEnvCfg

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # Reduce for visualization
    env_cfg.scene.env_spacing = 2.5
    if hasattr(env_cfg.scene, 'terrain') and hasattr(env_cfg.scene.terrain, 'terrain_generator'):
        if env_cfg.scene.terrain.terrain_generator is not None:
            env_cfg.scene.terrain.terrain_generator.num_rows = 5
            env_cfg.scene.terrain.terrain_generator.num_cols = 5
            env_cfg.scene.terrain.terrain_generator.curriculum = False
    env_cfg.scene.terrain.max_init_terrain_level = None

    # Disable noise for clean visualization
    env_cfg.observations.policy.enable_corruption = False

    # Disable DR
    if hasattr(env_cfg.events, 'base_external_force_torque'):
        env_cfg.events.base_external_force_torque = None
    if hasattr(env_cfg.events, 'push_robot'):
        env_cfg.events.push_robot = None

    agent_cfg.logger = "tensorboard"

    print(f"\n{'='*70}", flush=True)
    print(f"  PLAY — {robot.upper()}", flush=True)
    print(f"  {'='*66}", flush=True)
    print(f"  Checkpoint:  {args_cli.checkpoint}", flush=True)
    print(f"  Envs:        {env_cfg.scene.num_envs}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Create environment
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

    # Create runner and load checkpoint
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    runner.load(args_cli.checkpoint)
    print(f"[INFO] Loaded checkpoint: {args_cli.checkpoint}", flush=True)

    # Run inference
    policy = runner.get_inference_policy(device=agent_cfg.device)
    obs, _ = env.get_observations()

    while simulation_app.is_running():
        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _, _ = env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)
