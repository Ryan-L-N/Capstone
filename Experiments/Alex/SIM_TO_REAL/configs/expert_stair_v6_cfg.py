"""Stair Master V6 — Bidirectional stairs training.

35% ascending + 35% descending + 15% flat + 15% rough.
15% standing envs (zero velocity commands on any terrain).

Key change: directional_progress_reward replaces height_gain_reward.
Detects terrain slope from height scan and rewards:
  - Ascending: upward body velocity + forward walking
  - Descending: controlled downward velocity + forward walking
  - Flat: forward walking only

Resume from boulder_v6_4500.pt (22.9m on stairs without stair training).
"""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .base_s2r_env_cfg import SpotS2RBaseEnvCfg
from .terrain_cfgs import BIDIRECTIONAL_STAIRS_TERRAINS_CFG
from rewards.adaptive_rewards import (
    adaptive_clearance_reward,
    adaptive_velocity_reward,
    adaptive_smoothness_penalty,
    adaptive_height_penalty,
    adaptive_gait_reward,
    adaptive_slip_penalty,
)
from rewards.directional_progress_reward import directional_progress_reward
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
class SpotStairV6EnvCfg(SpotS2RBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = BIDIRECTIONAL_STAIRS_TERRAINS_CFG

        # ── HEIGHT-BASED CURRICULUM ──────────────────────────────────
        self.curriculum = StairHeightCurriculumCfg()

        # ── DR FIXES ─────────────────────────────────────────────────
        self.events.base_external_force_torque.params["force_range"] = (-1.0, 1.0)
        self.events.base_external_force_torque.params["torque_range"] = (-0.5, 0.5)
        self.events.physics_material.params["static_friction_range"] = (0.5, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 0.8)
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.3, 0.3),
            "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.5, 0.5),
        }

        # ── ADAPTIVE REWARDS ─────────────────────────────────────────
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

        # ── KINEMATIC CHAIN ──────────────────────────────────────────
        self.rewards.joint_pos.weight = -0.2
        self.rewards.base_pitch.weight = -0.05
        self.rewards.base_orientation.weight = -0.3
        self.rewards.base_motion.weight = -1.0
        self.rewards.air_time.weight = 2.0

        # ── EXPLOIT BLOCKERS ─────────────────────────────────────────
        self.rewards.base_roll.weight = -8.0
        self.rewards.motor_power.weight = -0.0002
        self.rewards.torque_limit.weight = -0.05
        self.rewards.undesired_contacts.weight = -0.5

        # ── STAIR REWARDS ────────────────────────────────────────────
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

        # ── NEW V6: DIRECTIONAL PROGRESS (replaces height_gain) ─────
        self.rewards.directional_progress = RewardTermCfg(
            func=directional_progress_reward, weight=4.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )

        # ── 15% STANDING ENVS ────────────────────────────────────────
        self.commands.base_velocity.rel_standing_envs = 0.15
