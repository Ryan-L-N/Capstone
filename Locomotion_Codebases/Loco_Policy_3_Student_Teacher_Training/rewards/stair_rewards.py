"""V5b Stair Rewards — 3 targeted functions for stair climbing.

1. stair_tread_placement: Reward feet landing on center of detected stair treads
2. flying_gait_penalty:   Penalize all 4 feet off ground (anti-bounce from literature)
3. dont_wait_penalty:     Penalize standing still when commanded to move (from ANYmal Parkour)

Minimal, surgical additions to the V3 adaptive reward base.
Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

from rewards.adaptive_rewards import _compute_scales


# =====================================================================
# 1. STAIR TREAD PLACEMENT — reward feet landing on center of flat regions
# =====================================================================

def stair_tread_placement(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    flatness_threshold: float = 0.005,
    center_reward_std: float = 0.08,
) -> torch.Tensor:
    """Reward feet in contact that land on center of detected stair treads.

    Uses height scanner to detect flat regions (stair treads) via local
    Z-variance. Flat regions have low variance between neighboring rays.
    Feet in contact near the center of flat regions get higher reward.

    Only active on rough terrain (terrain_scale > 0.3).
    """
    terrain_scale, _ = _compute_scales(env)
    asset = env.scene[asset_cfg.name]

    # Only active on rough terrain
    active = (terrain_scale > 0.3).float()

    # Height scanner data
    scanner = env.scene.sensors["height_scanner"]
    ray_hits = scanner.data.ray_hits_w  # (N, n_rays, 3)
    ray_z = ray_hits[:, :, 2]
    ray_z = torch.nan_to_num(ray_z, nan=0.0, posinf=0.0, neginf=0.0)
    ray_xy = ray_hits[:, :, :2]  # (N, n_rays, 2)

    N = ray_z.shape[0]
    n_rays = ray_z.shape[1]

    # Compute local Z-variance using pairwise differences with neighbors
    # Instead of reshaping to grid (ray count varies), use k-nearest approach:
    # For each ray, compute variance with its immediate neighbors in the flat array
    # Rays are ordered in a grid pattern, so neighbors are at ±1 and ±sqrt(n_rays)
    cols = int(round(n_rays ** 0.5))
    if cols * cols != n_rays:
        # Non-square grid — use approximate grid
        cols = max(1, int(n_rays ** 0.5))
    rows = max(1, n_rays // cols)
    usable = rows * cols

    grid_z = ray_z[:, :usable].view(N, rows, cols)  # (N, rows, cols)

    # Pad and compute local variance via 3x3 window
    padded = F.pad(grid_z, (1, 1, 1, 1), mode='replicate')  # (N, rows+2, cols+2)
    local_var = torch.zeros_like(grid_z)
    for di in range(3):
        for dj in range(3):
            diff = padded[:, di:di+rows, dj:dj+cols] - grid_z
            local_var = local_var + diff * diff
    local_var = local_var / 9.0

    # Flat mask: low local variance = stair tread
    is_flat = (local_var < flatness_threshold).float()  # (N, rows, cols)
    is_flat_1d = is_flat.view(N, usable)  # (N, usable)

    # Contact sensor for foot contact detection
    contact_sensor = env.scene.sensors["contact_forces"]
    all_forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]  # (N, n_bodies)
    all_forces = torch.nan_to_num(all_forces, nan=0.0)

    # Get foot positions and contact state
    foot_ids = asset_cfg.body_ids
    n_feet = len(foot_ids) if hasattr(foot_ids, '__len__') else 4
    foot_xy = asset.data.body_pos_w[:, foot_ids, :2]  # (N, n_feet, 2)
    foot_contact = all_forces[:, foot_ids].abs() > 1.0  # (N, n_feet)

    # Ray XY positions (usable subset)
    ray_xy_usable = ray_xy[:, :usable, :]  # (N, usable, 2)

    reward = torch.zeros(N, device=env.device)

    for f in range(n_feet):
        in_contact = foot_contact[:, f].float()  # (N,)

        # Distance from this foot to each ray hit
        foot_pos = foot_xy[:, f:f+1, :]  # (N, 1, 2)
        dist = torch.norm(ray_xy_usable - foot_pos, dim=2)  # (N, usable)

        # Mask non-flat cells with large distance
        masked_dist = dist + (1.0 - is_flat_1d) * 100.0  # (N, usable)

        # Min distance to any flat cell center
        min_dist = masked_dist.min(dim=1).values  # (N,)
        min_dist = torch.clamp(min_dist, 0.0, 2.0)

        # Gaussian reward: peak when foot is on center of flat region
        foot_reward = torch.exp(-min_dist * min_dist / (center_reward_std * center_reward_std))
        reward = reward + foot_reward * in_contact

    reward = reward / max(n_feet, 1)  # Average across feet
    return torch.clamp(torch.nan_to_num(reward, nan=0.0) * active, 0.0, 1.0)


# =====================================================================
# 2. FLYING GAIT PENALTY — penalize all 4 feet off ground (anti-bounce)
# =====================================================================

def flying_gait_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize when all 4 feet leave the ground simultaneously.

    From Huang et al. 2026 (arxiv 2602.03087) — simple and effective
    anti-bounce penalty. Returns 1.0 when all feet airborne, 0.0 otherwise.

    Directly prevents the bouncing-in-place exploit that killed V4.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # Vertical contact forces for all bodies
    forces_z = contact_sensor.data.net_forces_w_history[:, 0, :, 2]  # (N, n_bodies)
    forces_z = torch.nan_to_num(forces_z, nan=0.0)

    # Check foot bodies (use sensor_cfg.body_ids if available)
    if hasattr(sensor_cfg, 'body_ids') and sensor_cfg.body_ids is not None:
        foot_forces = forces_z[:, sensor_cfg.body_ids]  # (N, n_feet)
    else:
        # Fallback: use all bodies
        foot_forces = forces_z

    # Each foot: in contact if |force_z| > threshold
    in_contact = foot_forces.abs() > force_threshold  # (N, n_feet)

    # Count feet in contact
    n_in_contact = in_contact.float().sum(dim=1)  # (N,)

    # Penalty: 1.0 when NO feet in contact, 0.0 otherwise
    all_airborne = (n_in_contact == 0).float()

    return all_airborne


# =====================================================================
# 3. DON'T WAIT PENALTY — penalize standing still when commanded to move
# =====================================================================

def dont_wait_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    vel_threshold: float = 0.2,
) -> torch.Tensor:
    """Penalize low velocity when command says move. From ANYmal Parkour (ETH Zurich).

    When the velocity command is nonzero but the robot's actual body velocity
    is below threshold, apply a penalty. This prevents the standing-still
    exploit where the robot learns that not moving minimizes penalties.

    Returns 1.0 when commanded to move but standing still, 0.0 otherwise.
    """
    _, command_scale = _compute_scales(env)
    asset = env.scene[asset_cfg.name]

    # Body velocity magnitude (XY plane, body frame)
    vel_xy = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)  # (N,)

    # Command says move (command_scale > 0.3 means meaningful velocity command)
    moving_cmd = (command_scale > 0.3).float()

    # Robot is too slow
    too_slow = (vel_xy < vel_threshold).float()

    # Penalty: commanded to move but standing still
    return moving_cmd * too_slow
