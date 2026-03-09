"""Navigation MDP components — rewards and terrains."""

from .rewards import (
    collision_penalty,
    command_smoothness_penalty,
    goal_progress_reward,
    goal_reached_reward,
    path_efficiency_reward,
    speed_bonus_reward,
)
from .terrains import NavArenaCfg, ObstacleCfg, generate_obstacle_positions, sample_arena_type, sample_goal_position

__all__ = [
    # Rewards
    "goal_progress_reward",
    "goal_reached_reward",
    "collision_penalty",
    "path_efficiency_reward",
    "command_smoothness_penalty",
    "speed_bonus_reward",
    # Terrains
    "NavArenaCfg",
    "ObstacleCfg",
    "generate_obstacle_positions",
    "sample_arena_type",
    "sample_goal_position",
]
