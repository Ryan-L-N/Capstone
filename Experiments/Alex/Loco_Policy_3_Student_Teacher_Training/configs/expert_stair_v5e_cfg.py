"""Stair Master V5e — SURGE + Promote-Only + Mastery Ratchet.

Three innovations working together:
1. SURGE rewards: loose penalties, high noise, explore aggressively
2. Promote-only curriculum: after full actor release (iter 700+), no demotion
3. Mastery ratchet: reward scales by current_level / highest_reached

The robot's ONLY viable strategy is to charge forward up harder stairs.
Every escape route blocked: cant bounce (flying_gait), cant stand (dont_wait),
cant retreat (promote-only), cant farm easy terrain (mastery ratchet).

Same reward weights as V5d SURGE. The curriculum + ratchet are the new additions.
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
)
from rewards.stair_rewards import (
    stair_tread_placement,
    flying_gait_penalty,
    dont_wait_penalty,
)


@configclass
class SpotStairV5eEnvCfg(SpotS2RBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = STAIR_CLIMB_TERRAINS_CFG

        # ── ADAPTIVE REWARDS — SURGE VALUES (same as V5d) ────────────
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

        # ── KINEMATIC CHAIN — WIDE OPEN (same as V5d) ────────────────
        self.rewards.joint_pos.weight = -0.1
        self.rewards.base_pitch.weight = -0.05
        self.rewards.base_orientation.weight = -0.3
        self.rewards.base_motion.weight = -0.8
        self.rewards.air_time.weight = 2.0

        # ── EXPLOIT BLOCKERS — TIGHT ─────────────────────────────────
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

        # ── MINIMAL STANDING ─────────────────────────────────────────
        self.commands.base_velocity.rel_standing_envs = 0.05
