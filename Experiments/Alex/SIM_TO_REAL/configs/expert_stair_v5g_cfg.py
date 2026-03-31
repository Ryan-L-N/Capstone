"""Stair Master V5g — Tighter gait + height gain on ascending terrain.

Lessons from V5f:
- Inverted pyramid terrain is correct (spawn at BOTTOM, climb UP)
- SURGE loose penalties gave messy gait → model 900 peaked at 21.7m then regressed
- Gait=12.0 boost mid-run helped but too late
- Need tighter gait from the START, not loose-then-tighten

V5g = V5f terrain + tighter penalties + height_gain + gait=12 from start
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
class SpotStairV5gEnvCfg(SpotS2RBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ASCENDING_STAIRS_TERRAINS_CFG

        # ── ADAPTIVE REWARDS — TIGHTER THAN SURGE ─────────────────────
        self.rewards.foot_clearance = RewardTermCfg(
            func=adaptive_clearance_reward, weight=4.0,    # V5f:5.0 → tighter
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
        self.rewards.base_linear_velocity = RewardTermCfg(
            func=adaptive_velocity_reward, weight=10.0,    # Strong forward drive
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.action_smoothness = RewardTermCfg(
            func=adaptive_smoothness_penalty, weight=-0.5,  # V5f:-0.3 → tighter for cleaner gait
            params={},
        )
        self.rewards.terrain_relative_height = RewardTermCfg(
            func=adaptive_height_penalty, weight=-1.5,      # V5f:-1.0 → slightly tighter
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.rewards.gait = RewardTermCfg(
            func=adaptive_gait_reward, weight=12.0,          # Strong from start (V5f needed mid-run boost)
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.foot_slip = RewardTermCfg(
            func=adaptive_slip_penalty, weight=-0.8,         # V5f:-0.5 → slightly tighter
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        )

        # ── KINEMATIC CHAIN — MODERATE (between SURGE and conservative) ──
        self.rewards.joint_pos.weight = -0.2         # V5f:-0.1 → tighter
        self.rewards.base_pitch.weight = -0.1        # V5f:-0.05 → tighter
        self.rewards.base_orientation.weight = -0.3
        self.rewards.base_motion.weight = -1.0       # V5f:-0.8 → tighter
        self.rewards.air_time.weight = 2.0

        # ── EXPLOIT BLOCKERS ──────────────────────────────────────────
        self.rewards.base_roll.weight = -8.0
        self.rewards.motor_power.weight = -0.0002    # V5f:-0.0001 → slightly tighter
        self.rewards.torque_limit.weight = -0.05
        self.rewards.undesired_contacts.weight = -0.5

        # ── 4 STAIR REWARDS (added height_gain for ascending) ────────
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
        self.rewards.height_gain = RewardTermCfg(
            func=height_gain_reward, weight=2.0,      # Reward climbing UP (critical for inverted terrain)
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # ── MINIMAL STANDING ─────────────────────────────────────────
        self.commands.base_velocity.rel_standing_envs = 0.05
