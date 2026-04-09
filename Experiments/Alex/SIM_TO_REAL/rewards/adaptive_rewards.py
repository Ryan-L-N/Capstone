"""V3 Adaptive Reward System -- 7 terrain + command-aware rewards.

Two input signals drive all adaptations:
  1. terrain_scale (0=flat, 1=rough) from height-scan variance
  2. command_scale (0=standing, 1=walking) from velocity command magnitude

Rewards adapt automatically:
  1. Clearance:  4cm flat -> 25cm rough, 0 when standing
  2. Velocity:   100% flat -> 40% rough
  3. Smoothness: -1.5 flat -> -0.3 rough
  4. Body Height: 42cm flat -> 30cm rough
  5. Gait:       full flat -> 33% rough, 0 when standing
  6. Foot Slip:  -1.2 flat -> -0.2 rough
  7. Standing:   30x joint penalty when command is zero

Created for AI2C Tech Capstone -- MS for Autonomy, March 2026
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


# -- Terrain thresholds (height-scan variance) --
_VAR_FLAT = 0.001
_VAR_ROUGH = 0.02

# -- Cache for per-step scale computation --
_cached_step = {"count": -1, "terrain_scale": None, "command_scale": None}


def _compute_scales(env):
    """Compute terrain and command scales once per step, cached."""
    step = env.episode_length_buf.sum().item()
    if _cached_step["count"] == step:
        return _cached_step["terrain_scale"], _cached_step["command_scale"]

    from isaaclab.sensors import RayCaster

    # -- Terrain scale from height scan variance --
    scanner = env.scene.sensors["height_scanner"]
    height_data = scanner.data.ray_hits_w[:, :, 2]
    height_data = torch.nan_to_num(height_data, nan=0.0, posinf=0.0, neginf=0.0)
    height_data = torch.clamp(height_data, -10.0, 10.0)
    variance = torch.var(height_data, dim=1)
    variance = torch.nan_to_num(variance, nan=0.0)
    terrain_scale = torch.clamp(
        (variance - _VAR_FLAT) / (_VAR_ROUGH - _VAR_FLAT + 1e-8), 0.0, 1.0
    )

    # -- Command scale from velocity magnitude --
    cmd = env.command_manager.get_command("base_velocity")
    cmd_mag = torch.norm(cmd[:, :2], dim=1)
    command_scale = torch.clamp(cmd_mag / 0.5, 0.0, 1.0)

    _cached_step["count"] = step
    _cached_step["terrain_scale"] = terrain_scale
    _cached_step["command_scale"] = command_scale

    return terrain_scale, command_scale


def _lerp(a, b, t):
    """Linear interpolation: a when t=0, b when t=1."""
    return a + t * (b - a)


# =====================================================================
# 1. ADAPTIVE CLEARANCE
# =====================================================================

def adaptive_clearance_reward(
    env, asset_cfg, sensor_cfg,
    std=0.05, tanh_mult=2.0,
    target_flat=0.04, target_rough=0.25,
):
    """Foot clearance with terrain-adaptive target. Zero lift when standing."""
    from isaaclab.assets import RigidObject

    terrain_scale, command_scale = _compute_scales(env)
    asset = env.scene[asset_cfg.name]

    target = _lerp(target_flat, target_rough, terrain_scale) * command_scale

    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    foot_z = torch.nan_to_num(foot_z, nan=0.0)
    foot_z_error = torch.square(foot_z - target.unsqueeze(1))
    foot_z_error = torch.clamp(foot_z_error, 0.0, 1.0)

    foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    foot_vel_xy = torch.nan_to_num(foot_vel_xy, nan=0.0)
    foot_vel_mag = torch.tanh(tanh_mult * torch.norm(foot_vel_xy, dim=2))

    reward = foot_z_error * foot_vel_mag
    result = torch.exp(-torch.sum(reward, dim=1) / std)
    return torch.nan_to_num(result, nan=0.0)


# =====================================================================
# 1b. REAR CLEARANCE BONUS — break the 22m boulder wall
# =====================================================================

def rear_clearance_bonus(
    env, asset_cfg, sensor_cfg,
    target_flat=0.06, target_rough=0.35,
    tanh_mult=2.0,
):
    """Bonus for rear feet lifting higher during swing.

    The 22m boulder wall: front legs clear 25cm+ boulders, rear legs wedge.
    Root cause: adaptive_clearance treats all 4 feet equally but rear legs
    naturally drag. This reward gives extra incentive for rear feet (hl, hr)
    to lift high when swinging (not in contact with ground).

    Only active on rough terrain + when moving (scales to zero on flat/standing).
    Uses foot z-velocity > 0 (upward) as the swing gate instead of xy-velocity,
    rewarding the *upward arc* of the rear leg swing — like a dog/cat pulling
    its hind legs up and over an obstacle.
    """
    terrain_scale, command_scale = _compute_scales(env)
    asset = env.scene[asset_cfg.name]

    # Adaptive target: higher on rough terrain, zero when standing
    target = _lerp(target_flat, target_rough, terrain_scale) * command_scale

    # Rear foot z-positions (hl_foot, hr_foot only)
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    foot_z = torch.nan_to_num(foot_z, nan=0.0)

    # Reward: how close to target height (Gaussian kernel)
    height_error = torch.square(foot_z - target.unsqueeze(1))
    height_error = torch.clamp(height_error, 0.0, 1.0)
    height_reward = torch.exp(-height_error / 0.02)  # Tighter std = stronger gradient

    # Swing gate: only reward when foot is moving upward (z-velocity > 0)
    foot_vz = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    foot_vz = torch.nan_to_num(foot_vz, nan=0.0)
    swing_up = torch.tanh(tanh_mult * torch.clamp(foot_vz, min=0.0))

    # Combined: height accuracy during upward swing
    reward = (height_reward * swing_up).mean(dim=1)
    return torch.nan_to_num(reward, nan=0.0)


# =====================================================================
# 2. ADAPTIVE VELOCITY
# =====================================================================

def adaptive_velocity_reward(
    env, asset_cfg,
    std=1.0, speed_flat=1.0, speed_rough=0.4,
):
    """Velocity tracking scaled by terrain. Full speed on flat, 40% on rough."""
    terrain_scale, _ = _compute_scales(env)
    asset = env.scene[asset_cfg.name]

    speed_scale = _lerp(speed_flat, speed_rough, terrain_scale)

    cmd = env.command_manager.get_command("base_velocity")
    target_vel = cmd[:, :2] * speed_scale.unsqueeze(1)
    actual_vel = asset.data.root_lin_vel_b[:, :2]

    error = torch.sum(torch.square(target_vel - actual_vel), dim=1)
    return torch.exp(-error / (std ** 2))


# =====================================================================
# 3. ADAPTIVE SMOOTHNESS
# =====================================================================

def adaptive_smoothness_penalty(
    env,
    scale_flat=1.5, scale_rough=0.3,
):
    """Action smoothness penalty scaled by terrain. Strict on flat, loose on rough."""
    terrain_scale, _ = _compute_scales(env)
    scale = _lerp(scale_flat, scale_rough, terrain_scale)

    raw = torch.linalg.norm(
        env.action_manager.action - env.action_manager.prev_action, dim=1
    )
    raw = torch.clamp(raw, 0.0, 10.0)
    return raw * scale


# =====================================================================
# 4. ADAPTIVE BODY HEIGHT
# =====================================================================

def adaptive_height_penalty(
    env, sensor_cfg,
    height_flat=0.42, height_rough=0.30,
):
    """Body height penalty with terrain-adaptive target. Tall on flat, crouch on rough."""
    terrain_scale, _ = _compute_scales(env)

    scanner = env.scene.sensors[sensor_cfg.name]
    height_data = scanner.data.ray_hits_w[:, :, 2]
    height_data = torch.nan_to_num(height_data, nan=0.0, posinf=0.0, neginf=0.0)
    ground_z = torch.median(height_data, dim=1).values

    body_z = env.scene["robot"].data.root_pos_w[:, 2]
    actual_height = body_z - ground_z

    target = _lerp(height_flat, height_rough, terrain_scale)
    error = torch.square(actual_height - target)
    return torch.clamp(error, 0.0, 1.0)


# =====================================================================
# 5. ADAPTIVE GAIT
# =====================================================================

def adaptive_gait_reward(
    env, sensor_cfg, asset_cfg,
    std=0.1, max_err=0.2, velocity_threshold=0.5,
    synced_feet_pair_names=(("fl_foot", "hr_foot"), ("fr_foot", "hl_foot")),
    gait_scale_flat=1.0, gait_scale_rough=0.33,
):
    """Diagonal trot reward scaled by terrain + zeroed when standing."""
    terrain_scale, command_scale = _compute_scales(env)

    contact_sensor = env.scene.sensors[sensor_cfg.name]

    cmd = env.command_manager.get_command("base_velocity")
    cmd_mag = torch.norm(cmd[:, :2], dim=1)
    moving = cmd_mag > velocity_threshold

    contact_forces = contact_sensor.data.net_forces_w_history
    current_contact = contact_forces[:, 0, :, 2].abs() > 1.0

    all_body_names = contact_sensor.body_names
    reward = torch.zeros(env.num_envs, device=env.device)

    for pair in synced_feet_pair_names:
        idx_a = all_body_names.index(pair[0])
        idx_b = all_body_names.index(pair[1])
        sync_error = (current_contact[:, idx_a].float() - current_contact[:, idx_b].float()).abs()
        reward += torch.exp(-sync_error / std)

    reward = reward / len(synced_feet_pair_names)

    gait_scale = _lerp(gait_scale_flat, gait_scale_rough, terrain_scale)
    reward = reward * gait_scale * command_scale * moving.float()

    return reward


# =====================================================================
# 6. ADAPTIVE FOOT SLIP
# =====================================================================

def adaptive_slip_penalty(
    env, asset_cfg, sensor_cfg,
    threshold=1.0,
    scale_flat=1.2, scale_rough=0.2,
):
    """Foot slip penalty scaled by terrain. Strict on flat (ice), loose on rough."""
    terrain_scale, _ = _compute_scales(env)
    scale = _lerp(scale_flat, scale_rough, terrain_scale)

    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    foot_vel = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)

    contact_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, 2]
    in_contact = contact_forces.abs() > threshold

    slip = torch.sum(foot_vel * in_contact.float(), dim=1)
    slip = torch.clamp(slip, 0.0, 10.0)

    return slip * scale


# =====================================================================
# 7. ADAPTIVE STANDING
# =====================================================================

def adaptive_standing_penalty(
    env, asset_cfg,
    stand_scale=30.0, cmd_threshold=0.1,
):
    """Massive joint deviation + velocity penalty when command is near zero."""
    _, command_scale = _compute_scales(env)
    asset = env.scene[asset_cfg.name]

    standing = (command_scale < 0.2).float()

    joint_pos = asset.data.joint_pos
    default_pos = asset.data.default_joint_pos
    deviation = torch.sum(torch.square(joint_pos - default_pos), dim=1)
    deviation = torch.clamp(deviation, 0.0, 10.0)

    body_vel = torch.norm(asset.data.root_lin_vel_b[:, :3], dim=1)
    body_vel = torch.clamp(body_vel, 0.0, 5.0)

    return (deviation * stand_scale + body_vel * 5.0) * standing
