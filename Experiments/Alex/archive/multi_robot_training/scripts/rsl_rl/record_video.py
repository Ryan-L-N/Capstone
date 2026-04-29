"""Record a video of a trained policy running in Isaac Lab (headless).

Usage:
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/multi_robot_training/record_video.py --headless --enable_cameras \
        --robot spot --checkpoint /path/to/model_498.pt --num_envs 16 --steps 500

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

# ── 0. Parse args BEFORE any Isaac imports ──────────────────────────────
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record video of trained policy")
parser.add_argument("--robot", type=str, required=True, choices=["spot", "vision60"])
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=500, help="Number of sim steps to record")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default="/home/t2user/videos")
parser.add_argument("--terrain", type=str, default="flat", choices=["flat", "robust"])
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

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import quadruped_locomotion  # noqa: F401

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg


# ── 2. Main ─────────────────────────────────────────────────────────────

def main():
    robot = args_cli.robot

    if robot == "spot":
        from quadruped_locomotion.tasks.locomotion.config.spot.env_cfg import SpotLocomotionEnvCfg
        from quadruped_locomotion.tasks.locomotion.config.spot.agents.rsl_rl_ppo_cfg import SpotPPORunnerCfg
        env_cfg = SpotLocomotionEnvCfg()
        agent_cfg = SpotPPORunnerCfg()
        env_id = "Isaac-Velocity-Record-Spot-v0"
        EnvCfgClass = SpotLocomotionEnvCfg
    else:
        from quadruped_locomotion.tasks.locomotion.config.vision60.env_cfg import Vision60LocomotionEnvCfg
        from quadruped_locomotion.tasks.locomotion.config.vision60.agents.rsl_rl_ppo_cfg import Vision60PPORunnerCfg
        env_cfg = Vision60LocomotionEnvCfg()
        agent_cfg = Vision60PPORunnerCfg()
        env_id = "Isaac-Velocity-Record-Vision60-v0"
        EnvCfgClass = Vision60LocomotionEnvCfg

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # Flat terrain for clean video
    if args_cli.terrain == "flat":
        env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=5,
            num_cols=5,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=False,
            sub_terrains={
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
            },
        )

    # Reduce terrain for visualization
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
    print(f"  RECORD VIDEO — {robot.upper()}", flush=True)
    print(f"  {'='*66}", flush=True)
    print(f"  Checkpoint:  {args_cli.checkpoint}", flush=True)
    print(f"  Steps:       {args_cli.steps}", flush=True)
    print(f"  Envs:        {args_cli.num_envs}", flush=True)
    print(f"  Output:      {args_cli.output_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Create environment with video recording
    gym.register(
        id=env_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{EnvCfgClass.__module__}:{EnvCfgClass.__name__}",
        },
    )

    env = gym.make(env_id, cfg=env_cfg, render_mode="rgb_array")

    # Wrap with video recorder
    os.makedirs(args_cli.output_dir, exist_ok=True)
    video_kwargs = {
        "video_folder": args_cli.output_dir,
        "step_trigger": lambda step: step == 0,  # Record from start
        "video_length": args_cli.steps,
        "disable_logger": True,
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Create runner and load checkpoint
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    runner.load(args_cli.checkpoint)
    print(f"[INFO] Loaded: {args_cli.checkpoint}", flush=True)

    # Run inference and record
    policy = runner.get_inference_policy(device=agent_cfg.device)
    obs, _ = env.get_observations()

    print(f"[RECORD] Running {args_cli.steps} steps...", flush=True)
    for step in range(args_cli.steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _, _ = env.step(actions)
        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}/{args_cli.steps}", flush=True)

    print(f"[DONE] Video saved to {args_cli.output_dir}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)
