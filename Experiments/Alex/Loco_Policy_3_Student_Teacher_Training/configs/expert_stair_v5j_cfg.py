"""Stair Master V5j — Linear stairs + HEIGHT-BASED curriculum.

THE FIX for the sideways-skirting exploit. Previous curriculum rewarded
XY distance (4m) — robot walked sideways along step edges without climbing.
V5j uses terrain_levels_height: promotes based on Z height gained.

Same rewards as V5i (DR fixes + boosted velocity/height_gain).
"""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

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
class StairHeightCurriculumCfg:
    """Height-based curriculum — promotes based on Z gained, not XY distance."""
    terrain_levels = CurrTerm(func=mdp.terrain_levels_height)


@configclass
class SpotStairV5jEnvCfg(SpotS2RBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = LINEAR_STAIRS_TERRAINS_CFG

        # ── HEIGHT-BASED CURRICULUM (the fix) ─────────────────────────
        self.curriculum = StairHeightCurriculumCfg()

        # ── DR FIXES (from V5i) ──────────────────────────────────────
        self.events.base_external_force_torque.params["force_range"] = (-1.0, 1.0)
        self.events.base_external_force_torque.params["torque_range"] = (-0.5, 0.5)
        self.events.physics_material.params["static_friction_range"] = (0.5, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 0.8)
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.3, 0.3),
            "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.5, 0.5),
        }

        # ── ADAPTIVE REWARDS (same as V5i) ────────────────────────────
        self.rewards.foot_clearance = RewardTermCfg(
            func=adaptive_clearance_reward, weight=4.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
        self.rewards.base_linear_velocity = RewardTermCfg(
            func=adaptive_velocity_reward, weight=15.0,
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
            func=height_gain_reward, weight=4.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # ── MINIMAL STANDING ─────────────────────────────────────────
        self.commands.base_velocity.rel_standing_envs = 0.05
