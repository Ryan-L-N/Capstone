"""Pedipulation Visualization Script — Play trained pedipulation policy.

Loads a 240-dim pedipulation checkpoint and runs inference with optional
command overrides for active leg and foot target position.

Usage:
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/pedipulation/scripts/play_pedi.py \\
        --checkpoint /path/to/model_XXXX.pt

    # Override commands for debugging:
    ./isaaclab.sh -p ~/pedipulation/scripts/play_pedi.py \\
        --checkpoint /path/to/model_XXXX.pt \\
        --active_leg left --target_x 0.4 --target_y 0.1 --target_z -0.1

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

# ── 0. Parse args BEFORE any Isaac imports ──────────────────────────────
import argparse
import os
import sys

# Add pedipulation root to sys.path
PEDI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PEDI_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play pedipulation policy")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to trained 240-dim pedipulation checkpoint")
parser.add_argument("--num_envs", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--active_leg", type=str, default=None,
                    choices=["left", "right", "none"],
                    help="Override leg selection (left/right/none)")
parser.add_argument("--target_x", type=float, default=None,
                    help="Override foot target X (body frame, meters)")
parser.add_argument("--target_y", type=float, default=None,
                    help="Override foot target Y (body frame, meters)")
parser.add_argument("--target_z", type=float, default=None,
                    help="Override foot target Z (body frame, meters)")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 1. Imports (AFTER SimulationApp) ────────────────────────────────────
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import quadruped_locomotion  # noqa: F401
import configs  # noqa: F401

from configs.pedi_env_cfg import PedipulationSpotEnvCfg_PLAY
from configs.pedi_ppo_cfg import PedipulationPPORunnerCfg


# ── 2. Main ─────────────────────────────────────────────────────────────

def main():
    env_cfg = PedipulationSpotEnvCfg_PLAY()
    agent_cfg = PedipulationPPORunnerCfg()

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.logger = "tensorboard"

    # Build override tensors
    override_flags = None
    override_target = None

    if args_cli.active_leg is not None:
        if args_cli.active_leg == "left":
            override_flags = torch.tensor([1.0, 0.0])
        elif args_cli.active_leg == "right":
            override_flags = torch.tensor([0.0, 1.0])
        else:
            override_flags = torch.tensor([0.0, 0.0])

    if args_cli.target_x is not None:
        override_target = torch.tensor([
            args_cli.target_x,
            args_cli.target_y if args_cli.target_y is not None else 0.0,
            args_cli.target_z if args_cli.target_z is not None else -0.1,
        ])

    print(f"\n{'='*70}", flush=True)
    print(f"  PLAY PEDIPULATION — SPOT", flush=True)
    print(f"  {'='*66}", flush=True)
    print(f"  Checkpoint:  {args_cli.checkpoint}", flush=True)
    print(f"  Envs:        {env_cfg.scene.num_envs}", flush=True)
    print(f"  Active leg:  {args_cli.active_leg or 'auto (from command)'}", flush=True)
    if override_target is not None:
        print(f"  Target:      [{override_target[0]:.2f}, {override_target[1]:.2f}, {override_target[2]:.2f}]", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Create environment
    env = gym.make("Pedipulation-Spot-Play-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Create runner and load checkpoint
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    runner.load(args_cli.checkpoint)
    print(f"[INFO] Loaded checkpoint: {args_cli.checkpoint}", flush=True)

    # Get inference policy
    policy = runner.get_inference_policy(device=agent_cfg.device)
    obs = env.get_observations()
    device = obs.device

    # Move overrides to device
    if override_flags is not None:
        override_flags = override_flags.to(device)
    if override_target is not None:
        override_target = override_target.to(device)

    # Observation layout:
    # [0:48]   proprio
    # [48:51]  foot_target (3)
    # [51:53]  leg_flags (2)
    # [53:240] height_scan (187)
    FOOT_TARGET_START = 48
    FOOT_TARGET_END = 51
    LEG_FLAGS_START = 51
    LEG_FLAGS_END = 53

    step = 0
    while simulation_app.is_running():
        # Override pedipulation commands in observation tensor
        if override_flags is not None:
            obs[:, LEG_FLAGS_START:LEG_FLAGS_END] = override_flags.unsqueeze(0)
        if override_target is not None:
            obs[:, FOOT_TARGET_START:FOOT_TARGET_END] = override_target.unsqueeze(0)

        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)

        step += 1
        if step % 500 == 0:
            print(f"[PLAY] Step {step}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)
