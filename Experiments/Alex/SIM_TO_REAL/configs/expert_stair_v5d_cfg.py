"""Stair Master V5d — SURGE approach. Start loose, let PPO discover climbing.

Philosophy: PPO explores via noise. If penalties suppress climbing behavior
during exploration, it NEVER discovers "push hard through steps". So we
start with very loose penalties + high rewards to let PPO find the climbing
solution, then tighten via dashboard later.

Exploit blockers stay tight: dont_wait, flying_gait, base_roll.
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
class SpotStairV5dEnvCfg(SpotS2RBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = STAIR_CLIMB_TERRAINS_CFG

        # ── ADAPTIVE REWARDS — SURGE VALUES ───────────────────────────
        self.rewards.foot_clearance = RewardTermCfg(
            func=adaptive_clearance_reward, weight=5.0,   # SURGE: 2.0 → 5.0 (force high stepping)
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
        self.rewards.base_linear_velocity = RewardTermCfg(
            func=adaptive_velocity_reward, weight=10.0,   # SURGE: 7.0 → 10.0 (CHARGE up steps)
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.action_smoothness = RewardTermCfg(
            func=adaptive_smoothness_penalty, weight=-0.3, # SURGE: -0.7 → -0.3 (allow explosive motion)
            params={},
        )
        self.rewards.terrain_relative_height = RewardTermCfg(
            func=adaptive_height_penalty, weight=-1.0,     # SURGE: -2.0 → -1.0 (allow crouching for push)
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.rewards.gait = RewardTermCfg(
            func=adaptive_gait_reward, weight=8.0,         # Slightly reduced from 10 — allow gait variation
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.foot_slip = RewardTermCfg(
            func=adaptive_slip_penalty, weight=-0.5,        # SURGE: -1.0 → -0.5 (allow grip-seeking)
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        )

        # ── KINEMATIC CHAIN — WIDE OPEN ───────────────────────────────
        self.rewards.joint_pos.weight = -0.1          # SURGE: -0.3 → -0.1 (full ROM for climbing)
        self.rewards.base_pitch.weight = -0.05         # SURGE: -0.15 → -0.05 (lean hard into steps)
        self.rewards.base_orientation.weight = -0.3    # Slightly loosened
        self.rewards.base_motion.weight = -0.8         # SURGE: -1.5 → -0.8 (big body surges allowed)
        self.rewards.air_time.weight = 2.0             # Reduced — don't reward airborne

        # ── EXPLOIT BLOCKERS — STAY TIGHT ─────────────────────────────
        self.rewards.base_roll.weight = -7.0           # TIGHT — no flipping
        self.rewards.motor_power.weight = -0.0001      # Almost free — let motors push MAX
        self.rewards.torque_limit.weight = -0.05        # Almost free — push near limits
        self.rewards.undesired_contacts.weight = -0.5   # Loose — some body contact OK for climbing

        # ── 3 STAIR REWARDS ───────────────────────────────────────────
        self.rewards.stair_tread_placement = RewardTermCfg(
            func=stair_tread_placement, weight=2.0,     # Slightly boosted
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "flatness_threshold": 0.005,
                "center_reward_std": 0.08,
            },
        )
        self.rewards.flying_gait = RewardTermCfg(
            func=flying_gait_penalty, weight=-3.0,       # TIGHT — no bouncing
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        )
        self.rewards.dont_wait = RewardTermCfg(
            func=dont_wait_penalty, weight=-5.0,          # TIGHT — no standing still
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # ── MINIMAL STANDING ──────────────────────────────────────────
        self.commands.base_velocity.rel_standing_envs = 0.05
