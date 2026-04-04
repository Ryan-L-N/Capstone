"""Stair Master V5b — Research-backed stair climbing on V3 adaptive base.

3 new rewards:
  1. stair_tread_placement: Reward feet landing on center of detected stair treads
  2. flying_gait_penalty:   Penalize all 4 feet off ground (anti-bounce, Huang 2026)
  3. dont_wait_penalty:     Penalize standing still when commanded to move (ANYmal Parkour)

V5 fix: dont_wait prevents standing-still exploit. Reduced adaptive_standing,
motor_power, standing_envs. Stronger forward velocity drive.
"""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from .base_s2r_env_cfg import SpotS2RBaseEnvCfg
from .terrain_cfgs import STAIR_CLIMB_TERRAINS_CFG
from rewards.adaptive_rewards import (
    adaptive_clearance_reward,
    adaptive_velocity_reward,
    adaptive_smoothness_penalty,
    adaptive_height_penalty,
    adaptive_gait_reward,
    adaptive_slip_penalty,
    adaptive_standing_penalty,
)
from rewards.stair_rewards import (
    stair_tread_placement,
    flying_gait_penalty,
    dont_wait_penalty,
)


@configclass
class SpotStairV5EnvCfg(SpotS2RBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = STAIR_CLIMB_TERRAINS_CFG

        # ── V3 ADAPTIVE REWARDS (the working base) ────────────────────
        self.rewards.foot_clearance = RewardTermCfg(
            func=adaptive_clearance_reward, weight=2.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
        self.rewards.base_linear_velocity = RewardTermCfg(
            func=adaptive_velocity_reward, weight=7.0,   # V5: 5.0 → V5b: 7.0 stronger fwd
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.action_smoothness = RewardTermCfg(
            func=adaptive_smoothness_penalty, weight=-0.7,
            params={},
        )
        self.rewards.terrain_relative_height = RewardTermCfg(
            func=adaptive_height_penalty, weight=-2.0,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.rewards.gait = RewardTermCfg(
            func=adaptive_gait_reward, weight=10.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.foot_slip = RewardTermCfg(
            func=adaptive_slip_penalty, weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        )
        self.rewards.adaptive_standing = RewardTermCfg(
            func=adaptive_standing_penalty, weight=-0.3,  # V5: -1.0 → V5b: -0.3 (was too strong)
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # ── KINEMATIC CHAIN UNLOCK (from Trial 12b) ───────────────────
        self.rewards.joint_pos.weight = -0.3        # Allow deeper knee bend
        self.rewards.base_pitch.weight = -0.15       # Allow forward lean
        self.rewards.base_roll.weight = -5.0         # TIGHTEN (anti-flip)
        self.rewards.base_orientation.weight = -0.5
        self.rewards.air_time.weight = 3.0           # Less airborne reward

        # ── V5b FIXES ─────────────────────────────────────────────────
        self.rewards.motor_power.weight = -0.001     # V5: -0.005 → V5b: -0.001 (was 5x too punishing)
        self.rewards.undesired_contacts.weight = -0.8  # V5: -1.5 → V5b: -0.8 (ANYmal: reduce for climbing)

        # ── 3 STAIR-SPECIFIC REWARDS ──────────────────────────────────
        self.rewards.stair_tread_placement = RewardTermCfg(
            func=stair_tread_placement, weight=1.5,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "flatness_threshold": 0.005,
                "center_reward_std": 0.08,
            },
        )
        self.rewards.flying_gait = RewardTermCfg(
            func=flying_gait_penalty, weight=-2.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        )
        self.rewards.dont_wait = RewardTermCfg(
            func=dont_wait_penalty, weight=-1.0,      # ANYmal Parkour: penalize standing when cmd says move
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # ── STANDING ENVS (reduced from V5's 20%) ────────────────────
        self.commands.base_velocity.rel_standing_envs = 0.10
