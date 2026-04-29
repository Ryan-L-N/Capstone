"""Motor power penalty — penalizes mechanical power consumption.

Computes sum(|torque_i * joint_vel_i|) across all joints. This encourages
energy-efficient gaits and discourages wasteful joint oscillation.

Normal range for Spot trotting: 10-200 W. Clamped to [0, 500] for gradient
safety (Bug #29 pattern from quadruped_locomotion/mdp/rewards.py).

Addresses Risk R7 (no energy/power reward) from sim-to-real evaluation.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def motor_power_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize mechanical power: sum(|torque_i * joint_vel_i|).

    Clamped to [0, 500] for gradient safety. NaN/Inf sanitized.

    Args:
        env: The RL environment.
        asset_cfg: Configuration for the robot asset.

    Returns:
        (num_envs,) tensor of per-environment power penalty values.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque       # (num_envs, 12)
    joint_vel = asset.data.joint_vel         # (num_envs, 12)

    # Mechanical power = |torque * angular_velocity| per joint, summed
    power = torch.abs(torque * joint_vel).sum(dim=1)  # (num_envs,)

    # Clamp to prevent gradient explosion (Bug #29 pattern)
    power = torch.clamp(power, 0.0, 500.0)

    # NaN/Inf sanitization (Bug #24 pattern)
    return torch.where(torch.isfinite(power), power, torch.zeros_like(power))
