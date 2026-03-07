"""Navigation reward functions for Phase C nav policy training.

Goal-reaching, obstacle avoidance, and path efficiency rewards.
All penalties clamped per Bug #29 convention.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_progress_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward distance reduction toward goal each step.

    Positive when robot moves closer to goal, negative when moving away.
    Uses body-frame relative goal position to avoid world-frame Z issues (Bug #22).
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # Goal position in body frame (set by nav command manager)
    goal_rel = env.command_manager.get_command("nav_goal")[:, :2]  # (N, 2) — dx, dy in body frame
    current_dist = torch.linalg.norm(goal_rel, dim=1)

    # Previous distance stored in env extras (set at end of step)
    prev_dist = env.extras.get("nav_prev_goal_dist", current_dist.clone())
    progress = prev_dist - current_dist  # positive = getting closer

    # Store for next step
    env.extras["nav_prev_goal_dist"] = current_dist.clone()

    return progress


def goal_reached_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.5,
) -> torch.Tensor:
    """One-time bonus when robot reaches within threshold of goal.

    Returns 1.0 on the step the robot enters the goal zone, 0.0 otherwise.
    """
    goal_rel = env.command_manager.get_command("nav_goal")[:, :2]
    dist = torch.linalg.norm(goal_rel, dim=1)
    reached = (dist < threshold).float()
    return reached


def collision_penalty(
    env: ManagerBasedRLEnv,
    lidar_cfg: SceneEntityCfg,
    min_range: float = 0.3,
) -> torch.Tensor:
    """Penalize when LiDAR detects obstacles within min_range.

    Clamped to [0, 1] per Bug #29 convention.
    """
    lidar: RayCaster = env.scene.sensors[lidar_cfg.name]
    # Ray distances normalized to [0, 1] — 0 = hit at sensor, 1 = max range
    ray_distances = lidar.data.ray_hits_w[..., :2]  # XY hit positions
    # Compute distances from robot
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2].unsqueeze(1)  # (N, 1, 2)
    hit_dists = torch.linalg.norm(ray_distances - robot_pos, dim=-1)  # (N, num_rays)

    # Count rays hitting within danger zone
    danger = (hit_dists < min_range).float().sum(dim=1)  # (N,)
    result = torch.clamp(danger / 10.0, 0.0, 1.0)  # Normalize: 10 close hits = max penalty
    return torch.where(torch.isfinite(result), result, torch.zeros_like(result))


def path_efficiency_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward straight-line path efficiency (actual distance / straight-line).

    Tracks cumulative distance traveled vs straight-line goal distance.
    Returns 0-1 where 1 = perfectly straight path.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    current_pos = asset.data.root_pos_w[:, :2]

    prev_pos = env.extras.get("nav_prev_pos", current_pos.clone())
    step_dist = torch.linalg.norm(current_pos - prev_pos, dim=1)

    cumulative = env.extras.get("nav_cumulative_dist", torch.zeros_like(step_dist))
    cumulative = cumulative + step_dist
    env.extras["nav_cumulative_dist"] = cumulative
    env.extras["nav_prev_pos"] = current_pos.clone()

    # Straight-line distance from start
    start_pos = env.extras.get("nav_start_pos", current_pos.clone())
    straight_line = torch.linalg.norm(current_pos - start_pos, dim=1)

    # Efficiency = straight / cumulative (1.0 = perfect, capped at 1.0)
    efficiency = straight_line / (cumulative + 1e-6)
    efficiency = torch.clamp(efficiency, 0.0, 1.0)
    return efficiency


def command_smoothness_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize jerky velocity commands from nav policy.

    Clamped to [0, 5] per Bug #29 convention.
    """
    current_cmd = env.action_manager.action[:, :3]  # vx, vy, wz
    prev_cmd = env.extras.get("nav_prev_cmd", current_cmd.clone())

    diff = torch.linalg.norm(current_cmd - prev_cmd, dim=1)
    env.extras["nav_prev_cmd"] = current_cmd.clone()

    result = torch.clamp(diff, 0.0, 5.0)
    return torch.where(torch.isfinite(result), result, torch.zeros_like(result))


def speed_bonus_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    max_speed: float = 2.0,
) -> torch.Tensor:
    """Reward forward speed toward goal (not just any direction).

    Projects velocity onto goal direction for directional speed bonus.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    goal_rel = env.command_manager.get_command("nav_goal")[:, :2]

    # Goal direction (unit vector in body frame)
    goal_dist = torch.linalg.norm(goal_rel, dim=1, keepdim=True)
    goal_dir = goal_rel / (goal_dist + 1e-6)  # (N, 2)

    # Body-frame velocity
    body_vel = asset.data.root_lin_vel_b[:, :2]  # (N, 2)

    # Project velocity onto goal direction
    toward_goal_speed = (body_vel * goal_dir).sum(dim=1)  # (N,)

    # Only reward positive (toward goal), normalize by max_speed
    reward = torch.clamp(toward_goal_speed / max_speed, 0.0, 1.0)
    return reward
