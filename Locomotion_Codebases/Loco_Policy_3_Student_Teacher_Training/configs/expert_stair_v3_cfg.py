"""Stair Master V3 -- Adaptive reward system.
All 7 rewards terrain + command aware. Let the terrain teach.
"""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from .base_s2r_env_cfg import SpotS2RBaseEnvCfg
from .terrain_cfgs import OBSTACLE_PARKOUR_TERRAINS_CFG
from rewards.adaptive_rewards import (
    adaptive_clearance_reward, adaptive_velocity_reward,
    adaptive_smoothness_penalty, adaptive_height_penalty,
    adaptive_gait_reward, adaptive_slip_penalty, adaptive_standing_penalty,
)


@configclass
class SpotStairV3EnvCfg(SpotS2RBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = OBSTACLE_PARKOUR_TERRAINS_CFG

        # Replace 6 fixed rewards with 7 adaptive ones
        self.rewards.foot_clearance = RewardTermCfg(
            func=adaptive_clearance_reward, weight=1.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                    "sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.rewards.base_linear_velocity = RewardTermCfg(
            func=adaptive_velocity_reward, weight=5.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.action_smoothness = RewardTermCfg(
            func=adaptive_smoothness_penalty, weight=-1.0,
            params={},
        )
        self.rewards.terrain_relative_height = RewardTermCfg(
            func=adaptive_height_penalty, weight=-2.0,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.rewards.gait = RewardTermCfg(
            func=adaptive_gait_reward, weight=10.0,
            params={"asset_cfg": SceneEntityCfg("robot"),
                    "sensor_cfg": SceneEntityCfg("contact_forces")},
        )
        self.rewards.foot_slip = RewardTermCfg(
            func=adaptive_slip_penalty, weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )
        self.rewards.adaptive_standing = RewardTermCfg(
            func=adaptive_standing_penalty, weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # Loosen fixed penalties for climbing
        self.rewards.joint_pos.weight = -0.4
        self.rewards.base_pitch.weight = -0.2
        self.rewards.base_orientation.weight = -0.5

        # Standing envs
        self.commands.base_velocity.rel_standing_envs = 0.20
