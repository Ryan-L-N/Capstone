"""Stair Master V5i — Linear stairs + push recovery + stronger forward/climb.

Fixes from V5h:
- Enable push forces (robot learns to recover, not freeze)
- Narrow reset perturbation (less falling off step edges at spawn)
- Raise friction floor 0.3→0.5 (stairs are high friction in eval)
- Boost forward velocity 12→15 (CHARGE up stairs)
- Boost height_gain 2→4 (strong climbing reward)
"""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, EventTermCfg as EventTerm

import isaaclab.envs.mdp as mdp

from .base_s2r_env_cfg import SpotS2RBaseEnvCfg
from .terrain_cfgs import LINEAR_STAIRS_TERRAINS_CFG
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
class SpotStairV5iEnvCfg(SpotS2RBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = LINEAR_STAIRS_TERRAINS_CFG

        # ── DR FIXES ─────────────────────────────────────────────────
        # Enable push forces — learn to recover from perturbations
        self.events.base_external_force_torque.params["force_range"] = (-1.0, 1.0)
        self.events.base_external_force_torque.params["torque_range"] = (-0.5, 0.5)

        # Raise friction floor — stairs are high friction (1.0 in eval)
        self.events.physics_material.params["static_friction_range"] = (0.5, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 0.8)

        # Narrow reset perturbation — less spawning-already-falling on step edges
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.3, 0.3),
            "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.5, 0.5),
        }

        # ── ADAPTIVE REWARDS ──────────────────────────────────────────
        self.rewards.foot_clearance = RewardTermCfg(
            func=adaptive_clearance_reward, weight=4.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
        self.rewards.base_linear_velocity = RewardTermCfg(
            func=adaptive_velocity_reward, weight=15.0,    # BOOSTED: 12→15
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.action_smoothness = RewardTermCfg(
            func=adaptive_smoothness_penalty, weight=-0.5,
            params={},
        )
        self.rewards.terrain_relative_height = RewardTermCfg(
            func=adaptive_height_penalty, weight=-1.5,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.rewards.gait = RewardTermCfg(
            func=adaptive_gait_reward, weight=12.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.foot_slip = RewardTermCfg(
            func=adaptive_slip_penalty, weight=-0.8,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        )

        # ── KINEMATIC CHAIN ───────────────────────────────────────────
        self.rewards.joint_pos.weight = -0.2
        self.rewards.base_pitch.weight = -0.1
        self.rewards.base_orientation.weight = -0.3
        self.rewards.base_motion.weight = -1.0
        self.rewards.air_time.weight = 2.0

        # ── EXPLOIT BLOCKERS ──────────────────────────────────────────
        self.rewards.base_roll.weight = -8.0
        self.rewards.motor_power.weight = -0.0002
        self.rewards.torque_limit.weight = -0.05
        self.rewards.undesired_contacts.weight = -0.5

        # ── 4 STAIR REWARDS ──────────────────────────────────────────
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
            func=height_gain_reward, weight=4.0,           # BOOSTED: 2→4
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # ── MINIMAL STANDING ─────────────────────────────────────────
        self.commands.base_velocity.rel_standing_envs = 0.05
