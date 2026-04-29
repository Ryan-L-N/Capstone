"""Split orientation penalties — separate pitch (allow for stairs) from roll (prevent samba).

The stock base_orientation_penalty uses norm(projected_gravity_b[:, :2]) which penalizes
pitch and roll equally. On stairs, the robot NEEDS to pitch forward/back but should NOT
roll side-to-side.

projected_gravity_b in body frame:
  - [:, 0] = x-component = pitch (forward/back tilt)
  - [:, 1] = y-component = roll (side-to-side tilt)

When robot is level: projected_gravity_b = [0, 0, -1]
When pitched forward: projected_gravity_b = [-sin(pitch), 0, -cos(pitch)]
When rolled right:    projected_gravity_b = [0, sin(roll), -cos(roll)]

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg


def base_pitch_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize forward/backward body tilt (pitch).

    Low weight = robot can lean into stairs.
    Uses abs(gravity_x) in body frame. Clamped to [0, 1] for NaN safety.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    pitch = torch.abs(asset.data.projected_gravity_b[:, 0])
    return torch.clamp(pitch, 0.0, 1.0)


def base_roll_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize side-to-side body tilt (roll).

    High weight = prevents samba/lateral instability.
    Uses abs(gravity_y) in body frame. Clamped to [0, 1] for NaN safety.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    roll = torch.abs(asset.data.projected_gravity_b[:, 1])
    return torch.clamp(roll, 0.0, 1.0)
