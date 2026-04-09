"""Stair Master V7 — Conservative bidirectional + mixed terrain.

Fixes from V6c: too fast curriculum (56% flip), pure stairs (no diversity).

V7 changes:
  - 1m wide terrain (no sidestepping, OOB termination on lateral drift)
  - Slow curriculum (2x promotion thresholds, target <10% flip)
  - Mixed terrain: 30% up + 30% down + 10% rough + 10% boulders + 20% flat
  - 20% standing envs (stability on stairs)
  - rear_clearance_bonus (hind foot lift — from boulder V7)
  - Tighter OOB: distance_buffer=0.3 (1m wide terrain)

Resume from boulder_v6_4500.pt.
"""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
import math

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .base_s2r_env_cfg import SpotS2RBaseEnvCfg
from .terrain_cfgs import STAIR_V7_TERRAINS_CFG
from rewards.adaptive_rewards import (
    adaptive_clearance_reward,
    adaptive_velocity_reward,
    adaptive_smoothness_penalty,
    adaptive_height_penalty,
    adaptive_gait_reward,
    adaptive_slip_penalty,
    rear_clearance_bonus,
)
from rewards.directional_progress_reward import directional_progress_reward
from rewards.stair_rewards import (
    stair_tread_placement,
    flying_gait_penalty,
    dont_wait_penalty,
)
from rewards.stair_climbing_rewards import (
    front_foot_step_clearance_reward,
    riser_collision_penalty,
)


@configclass
class StairSlowCurriculumCfg:
    """Slow abs-height curriculum — 2x promotion thresholds for <10% flip."""
    terrain_levels = CurrTerm(func=mdp.terrain_levels_abs_height_slow)


@configclass
class SpotStairV7EnvCfg(SpotS2RBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = STAIR_V7_TERRAINS_CFG

        # ── SLOW CURRICULUM ──────────────────────────────────────────
        self.curriculum = StairSlowCurriculumCfg()

        # ── TIGHT OOB — 1m wide terrain, terminate on lateral drift ──
        self.terminations.terrain_out_of_bounds = DoneTerm(
            func=mdp.terrain_out_of_bounds,
            params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 0.3},
            time_out=True,
        )

        # ── DR FIXES ─────────────────────────────────────────────────
        self.events.base_external_force_torque.params["force_range"] = (-1.0, 1.0)
        self.events.base_external_force_torque.params["torque_range"] = (-0.5, 0.5)
        self.events.physics_material.params["static_friction_range"] = (0.5, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 0.8)
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.5, 0.5), "y": (-0.3, 0.3), "z": (-0.3, 0.3),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.3, 0.3),
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

        # ── DIRECTIONAL PROGRESS ─────────────────────────────────────
        self.rewards.directional_progress = RewardTermCfg(
            func=directional_progress_reward, weight=4.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )

        # ── REAR CLEARANCE BONUS (from boulder V7) ───────────────────
        self.rewards.rear_clearance = RewardTermCfg(
            func=rear_clearance_bonus, weight=3.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["hl_foot", "hr_foot"]),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )

        # ── FRONT FOOT STEP CLEARANCE (clear the riser, don't hit it) ─
        self.rewards.front_clearance = RewardTermCfg(
            func=front_foot_step_clearance_reward, weight=3.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["fl_foot", "fr_foot"]),
            },
        )

        # ── RISER COLLISION PENALTY (penalize hitting step edges) ─────
        self.rewards.riser_collision = RewardTermCfg(
            func=riser_collision_penalty, weight=-2.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["fl_foot", "fr_foot"]),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["fl_foot", "fr_foot"]),
            },
        )

        # ── 20% STANDING ENVS ────────────────────────────────────────
        self.commands.base_velocity.rel_standing_envs = 0.20
