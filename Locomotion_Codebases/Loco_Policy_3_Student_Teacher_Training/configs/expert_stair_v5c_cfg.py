"""Stair Master V5c — Fresh start from distilled_6899 (proven walker).

All lessons baked in from the start (no mid-run tuning needed):
- dont_wait=-5.0 (standing still is very expensive)
- adaptive_standing=0.0 (no standing reward at all)
- standing_envs=5% (minimal)
- motor_power=-0.0002 (don't punish climbing effort)
- torque_limit=-0.1 (let motors push near limits)
- base_motion=-1.5 (allow body surge)
- base_roll=-6.0 (anti-flip)
- lr_max=1e-4 (prevent value loss cascade that killed V5 and V5b)
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
class SpotStairV5cEnvCfg(SpotS2RBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = STAIR_CLIMB_TERRAINS_CFG

        # ── ADAPTIVE REWARDS (proven base) ────────────────────────────
        self.rewards.foot_clearance = RewardTermCfg(
            func=adaptive_clearance_reward, weight=2.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
        self.rewards.base_linear_velocity = RewardTermCfg(
            func=adaptive_velocity_reward, weight=7.0,
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

        # ── NO STANDING REWARD (lesson: it teaches standing-still exploit)
        # adaptive_standing NOT included. Weight 0 by omission.

        # ── KINEMATIC CHAIN (all baked in) ────────────────────────────
        self.rewards.joint_pos.weight = -0.3
        self.rewards.base_pitch.weight = -0.15
        self.rewards.base_roll.weight = -6.0         # Anti-flip (tightened from -5.0)
        self.rewards.base_orientation.weight = -0.5
        self.rewards.base_motion.weight = -1.5        # Allow body surge for push-up
        self.rewards.air_time.weight = 3.0

        # ── MOTOR PUSH (baked in, not mid-run) ────────────────────────
        self.rewards.motor_power.weight = -0.0002     # Let motors push hard for climbing
        self.rewards.torque_limit.weight = -0.1        # Let motors approach limits
        self.rewards.undesired_contacts.weight = -0.8  # Reduced for climbing (ANYmal approach)

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
            func=dont_wait_penalty, weight=-5.0,       # STRONG — standing still is very expensive
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # ── MINIMAL STANDING (5% — just enough for zero-cmd stability)
        self.commands.base_velocity.rel_standing_envs = 0.05
