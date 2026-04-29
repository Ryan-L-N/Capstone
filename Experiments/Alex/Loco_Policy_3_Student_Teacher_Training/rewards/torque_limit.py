"""Per-joint torque limit penalty — penalizes exceeding real motor limits.

Computes sum(ReLU(|torque_i| - limit_i)) across all joints. This teaches
the policy to respect real Spot motor torque capacity:
  - Hip motors (hx, hy): 45 Nm max
  - Knee motors (kn): 100 Nm max (conservative; real range 30-100 Nm angle-dependent)

Clamped to [0, 200] for gradient safety (Bug #29 pattern).

Addresses Risk R6 (no motor torque limits during training) from sim-to-real evaluation.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def torque_limit_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    hip_limit: float = 45.0,
    knee_limit: float = 100.0,
) -> torch.Tensor:
    """Penalize joint torques exceeding real motor limits.

    Uses soft violation: ReLU(|torque| - limit) per joint, summed.
    Only penalizes torques ABOVE the limit — in-range torques contribute zero.

    Spot DOF ordering (type-grouped, 12 joints):
      [fl_hx, fr_hx, hl_hx, hr_hx,   # 4 abduction (hip)  → hip_limit
       fl_hy, fr_hy, hl_hy, hr_hy,   # 4 hip flexion       → hip_limit
       fl_kn, fr_kn, hl_kn, hr_kn]   # 4 knee flexion      → knee_limit

    Args:
        env: The RL environment.
        asset_cfg: Configuration for the robot asset.
        hip_limit: Maximum torque for hip motors (Nm). Default 45.0.
        knee_limit: Maximum torque for knee motors (Nm). Default 100.0.

    Returns:
        (num_envs,) tensor of per-environment torque violation penalty values.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torque = torch.abs(asset.data.applied_torque)  # (num_envs, 12)

    # Build per-joint limit tensor matching Spot DOF ordering
    # 4 hx (hip) + 4 hy (hip) + 4 kn (knee) = 12
    limits = torch.tensor(
        [hip_limit] * 4 + [hip_limit] * 4 + [knee_limit] * 4,
        device=torque.device,
        dtype=torque.dtype,
    )

    # Soft violation: only penalize torques exceeding their limit
    violation = torch.relu(torque - limits.unsqueeze(0))  # (num_envs, 12)
    result = violation.sum(dim=1)  # (num_envs,)

    # Clamp to prevent gradient explosion (Bug #29 pattern)
    result = torch.clamp(result, 0.0, 200.0)

    # NaN/Inf sanitization (Bug #24 pattern)
    return torch.where(torch.isfinite(result), result, torch.zeros_like(result))
