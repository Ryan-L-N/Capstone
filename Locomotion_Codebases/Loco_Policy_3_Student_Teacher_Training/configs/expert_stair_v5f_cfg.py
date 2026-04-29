"""Stair Master V5f — THE FIX. Inverted pyramid stairs (spawn at BOTTOM, climb UP).

V4 through V5e all used MeshPyramidStairsTerrainCfg which spawns at the TOP.
The robot learned descent perfectly but never practiced ascending.
V5f uses MeshInvertedPyramidStairsTerrainCfg — spawn at BOTTOM, must climb UP.

SURGE rewards + mastery curriculum + correct terrain = should finally learn ascent.
"""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from .base_s2r_env_cfg import SpotS2RBaseEnvCfg
from .terrain_cfgs import ASCENDING_STAIRS_TERRAINS_CFG
from rewards.adaptive_rewards import (
    adaptive_clearance_reward,
    adaptive_velocity_reward,
    adaptive_smoothness_penalty,
    adaptive_height_penalty,
    adaptive_gait_reward,
    adaptive_slip_penalty,
)
from rewards.stair_climbing_rewards import height_gain_reward
from rewards.stair_rewards import (
    stair_tread_placement,
    flying_gait_penalty,
    dont_wait_penalty,
)


@configclass
class SpotStairV5fEnvCfg(SpotS2RBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ASCENDING_STAIRS_TERRAINS_CFG

        # ── ADAPTIVE REWARDS — SURGE VALUES ───────────────────────────
        self.rewards.foot_clearance = RewardTermCfg(
            func=adaptive_clearance_reward, weight=5.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
        self.rewards.base_linear_velocity = RewardTermCfg(
            func=adaptive_velocity_reward, weight=10.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.action_smoothness = RewardTermCfg(
            func=adaptive_smoothness_penalty, weight=-0.3,
            params={},
        )
        self.rewards.terrain_relative_height = RewardTermCfg(
            func=adaptive_height_penalty, weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.rewards.gait = RewardTermCfg(
            func=adaptive_gait_reward, weight=8.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.foot_slip = RewardTermCfg(
            func=adaptive_slip_penalty, weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        )

        # ── KINEMATIC CHAIN — WIDE OPEN ───────────────────────────────
        self.rewards.joint_pos.weight = -0.1
        self.rewards.base_pitch.weight = -0.05
        self.rewards.base_orientation.weight = -0.3
        self.rewards.base_motion.weight = -0.8
        self.rewards.air_time.weight = 2.0

        # ── EXPLOIT BLOCKERS ──────────────────────────────────────────
        self.rewards.base_roll.weight = -7.0
        self.rewards.motor_power.weight = -0.0001
        self.rewards.torque_limit.weight = -0.05
        self.rewards.undesired_contacts.weight = -0.5

        # ── 3 STAIR REWARDS ──────────────────────────────────────────
        self.rewards.stair_tread_placement = RewardTermCfg(
            func=stair_tread_placement, weight=2.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "flatness_threshold": 0.005,
                "center_reward_std": 0.08,
            },
        )
        self.rewards.flying_gait = RewardTermCfg(
            func=flying_gait_penalty, weight=-3.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        )
        self.rewards.dont_wait = RewardTermCfg(
            func=dont_wait_penalty, weight=-5.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # ── HEIGHT GAIN (critical for inverted terrain — reward climbing UP)
        self.rewards.height_gain = RewardTermCfg(
            func=height_gain_reward, weight=3.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # ── MINIMAL STANDING ─────────────────────────────────────────
        self.commands.base_velocity.rel_standing_envs = 0.05
