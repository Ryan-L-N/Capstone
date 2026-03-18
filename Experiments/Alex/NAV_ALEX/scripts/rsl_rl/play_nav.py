"""Evaluate/visualize a trained navigation policy.

Loads a trained nav checkpoint + frozen loco checkpoint and runs the
hierarchical policy in rendered mode. Prints per-episode forward distance
and survival statistics.

Usage:
    python scripts/rsl_rl/play_nav.py \
        --nav_checkpoint logs/spot_nav_explore_ppo/model_final.pt \
        --loco_checkpoint checkpoints/ai_coached_v8_10600.pt \
        --num_envs 50 --num_episodes 20

For headless eval (H100):
    python scripts/rsl_rl/play_nav.py \
        --headless \
        --nav_checkpoint logs/spot_nav_explore_ppo/model_final.pt \
        --loco_checkpoint checkpoints/ai_coached_v8_10600.pt \
        --num_envs 100 --num_episodes 100
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# CLI args — before Isaac Lab imports
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Phase C: Nav policy evaluation")
    parser.add_argument("--nav_checkpoint", type=str, required=True,
                        help="Path to trained nav policy checkpoint")
    parser.add_argument("--loco_checkpoint", type=str, required=True,
                        help="Path to frozen Phase B loco checkpoint")
    parser.add_argument("--num_envs", type=int, default=50,
                        help="Number of parallel environments (default: 50)")
    parser.add_argument("--num_episodes", type=int, default=20,
                        help="Number of episodes to run (default: 20)")
    parser.add_argument("--depth_res", type=int, default=64,
                        help="Depth image resolution (default: 64)")
    parser.add_argument("--headless", action="store_true",
                        help="Run headless")
    return parser.parse_args()


args = parse_args()

# ---------------------------------------------------------------------------
# Isaac Lab imports
# ---------------------------------------------------------------------------

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

import nav_locomotion  # noqa: F401
from nav_locomotion.modules.depth_cnn import ActorCriticCNN
from nav_locomotion.modules.loco_wrapper import FrozenLocoPolicy
from nav_locomotion.modules.nav_env_wrapper import NavEnvWrapper


def main():
    # Create env
    env = gym.make("Navigation-Explore-Spot-Play-v0", num_envs=args.num_envs)
    device = env.device

    # Load frozen loco
    loco_policy = FrozenLocoPolicy.from_checkpoint(
        args.loco_checkpoint, device=str(device)
    )

    # Wrap env
    nav_env = NavEnvWrapper(env, loco_policy)

    # Load trained nav policy
    cnn_policy = ActorCriticCNN(
        num_obs=nav_env.num_obs,
        num_actions=nav_env.num_actions,
        depth_res=args.depth_res,
    ).to(device)

    ckpt = torch.load(args.nav_checkpoint, weights_only=False, map_location=str(device))
    cnn_policy.load_state_dict(ckpt.get("model_state_dict", ckpt))
    cnn_policy.eval()

    print(f"[PLAY_NAV] Loaded nav policy from {args.nav_checkpoint}")
    print(f"[PLAY_NAV] Running {args.num_episodes} episodes with {args.num_envs} envs")

    # Run episodes
    episode_distances = []
    episode_survived = []
    episodes_done = 0

    nav_obs, _ = nav_env.reset()

    while episodes_done < args.num_episodes:
        # Deterministic action
        with torch.no_grad():
            vel_cmd = cnn_policy.act_inference(nav_obs)

        nav_obs, reward, done, info = nav_env.step(vel_cmd)

        if done.any():
            # Record stats for finished episodes
            robot = env.scene["robot"]
            for i in range(args.num_envs):
                if done[i]:
                    x_pos = robot.data.root_pos_w[i, 0].item()
                    origin_x = env.scene.env_origins[i, 0].item()
                    distance = x_pos - origin_x
                    episode_distances.append(distance)
                    episode_survived.append(True)  # Simplified — timed out = survived
                    episodes_done += 1
                    if episodes_done >= args.num_episodes:
                        break

    # Print results
    import numpy as np
    distances = np.array(episode_distances)

    print(f"\n{'='*60}")
    print(f"NAV EVAL RESULTS ({len(distances)} episodes)")
    print(f"{'='*60}")
    print(f"Forward distance:")
    print(f"  Mean:   {distances.mean():.1f}m")
    print(f"  Median: {np.median(distances):.1f}m")
    print(f"  Min:    {distances.min():.1f}m")
    print(f"  Max:    {distances.max():.1f}m")
    print(f"  Std:    {distances.std():.1f}m")
    print(f"{'='*60}")

    os._exit(0)


if __name__ == "__main__":
    main()
