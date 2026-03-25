"""Locomotion MDP components — rewards, commands, and terrains."""

from .rewards import (
    VegetationDragReward,
    body_height_tracking_penalty,
    body_scraping_penalty,
    clamped_action_smoothness_penalty,
    clamped_joint_acceleration_penalty,
    clamped_joint_torques_penalty,
    clamped_joint_velocity_penalty,
    contact_force_smoothness_penalty,
    stumble_penalty,
    terrain_relative_height_penalty,
    velocity_modulation_reward,
)
from .commands import TerrainScaledVelocityCommand, TerrainScaledVelocityCommandCfg
from .terrains import ROBUST_TERRAINS_CFG

__all__ = [
    # Rewards
    "VegetationDragReward",
    "body_height_tracking_penalty",
    "body_scraping_penalty",
    "clamped_action_smoothness_penalty",
    "clamped_joint_acceleration_penalty",
    "clamped_joint_torques_penalty",
    "clamped_joint_velocity_penalty",
    "contact_force_smoothness_penalty",
    "stumble_penalty",
    "terrain_relative_height_penalty",
    "velocity_modulation_reward",
    # Commands
    "TerrainScaledVelocityCommand",
    "TerrainScaledVelocityCommandCfg",
    # Terrains
    "ROBUST_TERRAINS_CFG",
]
