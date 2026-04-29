"""Pedipulation reward functions — foot tracking, stability, and locomotion.

7 new reward terms for pedipulation + 5 reused from locomotion.
All rewards are conditioned on leg selection flags:
  - foot_tracking, standing_stability, body_stillness, passive_legs,
    foot_smoothness → active only when a leg is selected (flag=1)
  - walking → active only in walking mode (flags=[0,0])
  - Others → always active

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


# =============================================================================
# Constants — Spot joint indices (type-grouped DOF ordering)
# =============================================================================

FL_JOINT_IDS = [0, 4, 8]   # fl_hx, fl_hy, fl_kn
FR_JOINT_IDS = [1, 5, 9]   # fr_hx, fr_hy, fr_kn
HL_JOINT_IDS = [2, 6, 10]  # hl_hx, hl_hy, hl_kn
HR_JOINT_IDS = [3, 7, 11]  # hr_hx, hr_hy, hr_kn
HIND_JOINT_IDS = [2, 3, 6, 7, 10, 11]  # all hind leg joints


# =============================================================================
# Helpers
# =============================================================================

def _get_leg_flags(env: ManagerBasedRLEnv):
    """Get current leg selection flags from command manager.

    Returns:
        left_active: (N,) bool — left front leg is the manipulator
        right_active: (N,) bool — right front leg is the manipulator
        any_active: (N,) bool — either leg is active (manipulation mode)
        walking: (N,) bool — no leg active (walking mode)
    """
    leg_cmd = env.command_manager.get_command("leg_selection")  # (N, 2)
    left_active = leg_cmd[:, 0] > 0.5
    right_active = leg_cmd[:, 1] > 0.5
    any_active = left_active | right_active
    walking = ~any_active
    return left_active, right_active, any_active, walking


def _foot_pos_body_frame(asset: Articulation, body_id: int) -> torch.Tensor:
    """Get a specific foot's position in body frame.

    Args:
        asset: Robot articulation.
        body_id: Body index of the foot in the asset's body list.

    Returns:
        (N, 3) foot position in body frame.
    """
    foot_pos_w = asset.data.body_pos_w[:, body_id, :]  # (N, 3)
    base_pos_w = asset.data.root_pos_w                 # (N, 3)
    base_quat = asset.data.root_quat_w                 # (N, 4) [w,x,y,z]

    rel_pos_w = foot_pos_w - base_pos_w
    return quat_rotate_inverse(base_quat, rel_pos_w)   # (N, 3)


# =============================================================================
# Pedipulation-specific rewards
# =============================================================================

class FootTrackingReward(ManagerTermBase):
    """Reward for moving the active front foot toward the target position.

    Uses exp kernel: reward = exp(-distance / sigma).
    Only active when a leg is selected (flags != [0,0]).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene["robot"]
        self.sigma = cfg.params.get("sigma", 0.1)

        # Resolve foot body IDs
        self.fl_foot_id = cfg.params["fl_cfg"].body_ids[0]
        self.fr_foot_id = cfg.params["fr_cfg"].body_ids[0]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sigma: float = 0.1,
        fl_cfg: SceneEntityCfg = None,
        fr_cfg: SceneEntityCfg = None,
    ) -> torch.Tensor:
        left_active, right_active, any_active, _ = _get_leg_flags(env)

        if not any_active.any():
            return torch.zeros(env.num_envs, device=env.device)

        # Get target from command
        target = env.command_manager.get_command("foot_target")  # (N, 3)

        # Get both foot positions in body frame
        fl_pos = _foot_pos_body_frame(self.asset, self.fl_foot_id)
        fr_pos = _foot_pos_body_frame(self.asset, self.fr_foot_id)

        # Select active foot per environment
        active_pos = torch.where(left_active.unsqueeze(1), fl_pos, fr_pos)

        # Distance to target
        dist = torch.linalg.norm(active_pos - target, dim=1)
        dist = torch.clamp(dist, 0.0, 2.0)

        reward = torch.exp(-dist / self.sigma)
        reward = reward * any_active.float()

        return torch.where(torch.isfinite(reward), reward, torch.zeros_like(reward))


class StandingStabilityReward(ManagerTermBase):
    """Reward for maintaining stability during manipulation.

    Combines:
    - Body level (low roll/pitch via projected gravity)
    - Three non-active feet in ground contact

    Only active when a leg is selected.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene["robot"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.contact_body_ids = cfg.params["sensor_cfg"].body_ids

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg = None,
    ) -> torch.Tensor:
        left_active, right_active, any_active, _ = _get_leg_flags(env)

        if not any_active.any():
            return torch.zeros(env.num_envs, device=env.device)

        # 1. Body orientation: projected_gravity_b ideal = [0, 0, -1]
        gravity = self.asset.data.projected_gravity_b  # (N, 3)
        tilt = torch.linalg.norm(gravity[:, :2], dim=1)
        orientation_score = torch.exp(-tilt * 5.0)

        # 2. Three-foot contact: non-active feet should be on ground
        # contact_body_ids order: [fl, fr, hl, hr]
        forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.contact_body_ids]
        force_mags = torch.norm(forces, dim=-1)  # (N, 4)
        in_contact = force_mags > 1.0

        # Mask out the active foot (we want it lifted, not grounded)
        active_mask = torch.ones_like(in_contact)
        active_mask[left_active, 0] = 0   # exclude FL when left active
        active_mask[right_active, 1] = 0  # exclude FR when right active

        # Count non-active feet in contact (target: 3 out of 3)
        passive_contact = (in_contact * active_mask).sum(dim=1).float()
        contact_score = torch.clamp(passive_contact / 3.0, 0.0, 1.0)

        reward = orientation_score * contact_score * any_active.float()
        return torch.where(torch.isfinite(reward), reward, torch.zeros_like(reward))


def walking_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float = 1.0,
) -> torch.Tensor:
    """Velocity tracking reward — active only during walking mode (flags=[0,0]).

    Prevents catastrophic forgetting of locomotion during pedipulation training.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    _, _, _, walking = _get_leg_flags(env)

    cmd_vel = env.command_manager.get_command("base_velocity")[:, :2]
    actual_vel = asset.data.root_lin_vel_b[:, :2]

    vel_error = torch.linalg.norm(cmd_vel - actual_vel, dim=1)
    reward = torch.exp(-vel_error / std)
    reward = reward * walking.float()

    return torch.where(torch.isfinite(reward), reward, torch.zeros_like(reward))


def body_stillness_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    lin_std: float = 0.5,
    ang_std: float = 1.0,
) -> torch.Tensor:
    """Reward for keeping the body still during manipulation.

    Low body velocity → high reward. Only active when a leg is selected.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    _, _, any_active, _ = _get_leg_flags(env)

    lin_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    ang_vel = torch.linalg.norm(asset.data.root_ang_vel_b, dim=1)

    lin_score = torch.exp(-lin_vel / lin_std)
    ang_score = torch.exp(-ang_vel / ang_std)

    reward = lin_score * ang_score * any_active.float()
    return torch.where(torch.isfinite(reward), reward, torch.zeros_like(reward))


def passive_legs_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize non-active legs deviating from default joint positions.

    During manipulation, support legs (hind legs + non-active front leg)
    should stay planted near their default positions.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    left_active, right_active, any_active, _ = _get_leg_flags(env)

    if not any_active.any():
        return torch.zeros(env.num_envs, device=env.device)

    joint_pos = asset.data.joint_pos        # (N, 12)
    default_pos = asset.data.default_joint_pos  # (N, 12)
    pos_error = joint_pos - default_pos

    # Always penalize hind legs during manipulation
    hind_error = pos_error[:, HIND_JOINT_IDS].pow(2).sum(dim=1)

    # Penalize the non-active front leg
    fl_error = pos_error[:, FL_JOINT_IDS].pow(2).sum(dim=1)
    fr_error = pos_error[:, FR_JOINT_IDS].pow(2).sum(dim=1)

    # Left active → penalize FR (right front); Right active → penalize FL
    passive_front_error = torch.where(left_active, fr_error, fl_error)

    penalty = hind_error + passive_front_error
    penalty = torch.clamp(penalty, 0.0, 10.0)
    penalty = penalty * any_active.float()

    return torch.where(torch.isfinite(penalty), penalty, torch.zeros_like(penalty))


def foot_smoothness_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize jerky active foot movements via joint velocity magnitude.

    Only active when a leg is selected.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    left_active, right_active, any_active, _ = _get_leg_flags(env)

    if not any_active.any():
        return torch.zeros(env.num_envs, device=env.device)

    joint_vel = asset.data.joint_vel  # (N, 12)

    fl_vel_sq = joint_vel[:, FL_JOINT_IDS].pow(2).sum(dim=1)
    fr_vel_sq = joint_vel[:, FR_JOINT_IDS].pow(2).sum(dim=1)

    active_vel_sq = torch.where(left_active, fl_vel_sq, fr_vel_sq)
    active_vel_sq = torch.clamp(active_vel_sq, 0.0, 50.0)
    active_vel_sq = active_vel_sq * any_active.float()

    return torch.where(torch.isfinite(active_vel_sq), active_vel_sq, torch.zeros_like(active_vel_sq))
