"""Directional Progress Reward — direction-aware climbing/descending reward.

Detects whether the robot is on ascending or descending terrain using the
height scan, then rewards appropriate movement:
  - Ascending:  reward upward body velocity + forward walking
  - Descending: reward downward body velocity + forward walking (controlled descent)
  - Flat:       reward forward walking only

Created for Stair V6 bidirectional training.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

from rewards.adaptive_rewards import _compute_scales


def directional_progress_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    vz_scale: float = 8.0,
    vx_scale: float = 2.0,
    slope_threshold: float = 0.02,
) -> torch.Tensor:
    """Direction-aware progress reward for bidirectional stair training.

    Uses height scan to detect terrain slope ahead vs behind, then rewards:
      - Ascending (slope > threshold): positive body vz + forward vx
      - Descending (slope < -threshold): controlled negative vz + forward vx
      - Flat (|slope| < threshold): forward vx only

    Args:
        env: The RL environment.
        asset_cfg: Robot asset config.
        sensor_cfg: Height scanner sensor config.
        vz_scale: Scale factor for vertical velocity reward.
        vx_scale: Scale factor for forward velocity reward.
        slope_threshold: Min slope magnitude to classify as ascending/descending.
    """
    _, command_scale = _compute_scales(env)
    asset = env.scene[asset_cfg.name]

    # -- Detect terrain direction from height scan --
    scanner = env.scene.sensors[sensor_cfg.name]
    height_data = scanner.data.ray_hits_w[:, :, 2]  # (N, n_rays)
    height_data = torch.nan_to_num(height_data, nan=0.0, posinf=0.0, neginf=0.0)
    height_data = torch.clamp(height_data, -10.0, 10.0)

    n_rays = height_data.shape[1]
    # Forward rays (front 25%) vs rear rays (back 25%)
    quarter = max(1, n_rays // 4)
    forward_height = torch.median(height_data[:, -quarter:], dim=1).values  # (N,)
    rear_height = torch.median(height_data[:, :quarter], dim=1).values      # (N,)

    slope = forward_height - rear_height  # positive = ascending, negative = descending

    # Classification masks (soft boundaries via tanh for gradient flow)
    ascending = torch.tanh(torch.relu(slope - slope_threshold) * 20.0)   # 0-1
    descending = torch.tanh(torch.relu(-slope - slope_threshold) * 20.0) # 0-1
    flat = 1.0 - ascending - descending  # remainder
    flat = torch.clamp(flat, 0.0, 1.0)

    # -- Body velocities --
    body_vz = asset.data.root_lin_vel_w[:, 2]   # world-frame vertical
    body_vx = asset.data.root_lin_vel_b[:, 0]   # body-frame forward
    body_vz = torch.nan_to_num(body_vz, nan=0.0)
    body_vx = torch.nan_to_num(body_vx, nan=0.0)

    # Forward gate: vz reward ONLY when robot is also walking forward.
    # Without this, robot learns to "stand up tall" on two legs for free vz reward.
    forward_gate = torch.clamp(body_vx * 2.0, 0.0, 1.0)  # ramps 0->1 over 0->0.5 m/s

    # -- Ascending reward: vz gated on forward progress + forward vx --
    up_reward = torch.relu(body_vz) * vz_scale * forward_gate + torch.relu(body_vx) * vx_scale

    # -- Descending reward: controlled negative vz gated on forward + forward vx --
    down_reward = torch.relu(-body_vz) * vz_scale * forward_gate + torch.relu(body_vx) * vx_scale

    # -- Flat reward: forward velocity only --
    flat_reward = torch.relu(body_vx) * vx_scale

    # -- Combine based on detected direction --
    reward = ascending * up_reward + descending * down_reward + flat * flat_reward

    # Scale by command magnitude (zero when standing)
    reward = reward * command_scale

    return torch.clamp(torch.nan_to_num(reward, nan=0.0), 0.0, 5.0)
