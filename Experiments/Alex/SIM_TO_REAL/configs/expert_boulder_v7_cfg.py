"""Boulder Master V7 — Rear leg clearance bonus to break the 22m wall.

V6 result: 26.6m zone 3 (rear legs wedge on 25cm+ boulders).
Root cause: adaptive_clearance treats all 4 feet equally, but rear legs
naturally drag while front legs lift. This config adds a rear-specific
clearance bonus (hl_foot, hr_foot only) that rewards upward arc motion
during swing phase — like a dog pulling its hind legs up and over.

Changes from V6:
  - NEW: rear_clearance reward (weight=3.0) — rear feet lift bonus
  - NEW: front_clearance reward (weight=3.0) — front feet clear next obstacle surface
  - NEW: riser_collision penalty (weight=-2.0) — penalize front feet hitting obstacle faces
  - Resume from boulder_v6_expert_4500.pt (best V6 checkpoint)
"""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from .base_s2r_env_cfg import SpotS2RBaseEnvCfg
from rewards.adaptive_rewards import (
    adaptive_clearance_reward,
    adaptive_velocity_reward,
    adaptive_smoothness_penalty,
    adaptive_height_penalty,
    adaptive_gait_reward,
    adaptive_slip_penalty,
    rear_clearance_bonus,
)
from rewards.stair_climbing_rewards import height_gain_reward
from rewards.stair_rewards import (
    flying_gait_penalty,
    dont_wait_penalty,
)
from rewards.stair_climbing_rewards import (
    front_foot_step_clearance_reward,
    riser_collision_penalty,
)

_COMMON = dict(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
)

# Same terrain as V6 — dense unavoidable boulders
DENSE_BOULDER_TERRAINS_CFG = TerrainGeneratorCfg(
    **_COMMON,
    sub_terrains={
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.40,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.10, 0.80),
            obstacle_height_range=(0.03, 0.60),
            num_obstacles=80,
            platform_width=1.5,
            border_width=0.25,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.25,
            grid_width=0.45,
            grid_height_range=(0.05, 0.35),
            platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(0.03, 0.15),
            noise_step=0.02,
            border_width=0.25,
        ),
        "repeated_boxes": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            proportion=0.10,
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=30,
                height=0.05,
                size=(0.2, 0.2),
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=15,
                height=0.50,
                size=(0.6, 0.6),
            ),
            platform_width=2.0,
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.05,
        ),
    },
)


@configclass
class SpotBoulderV7EnvCfg(SpotS2RBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = DENSE_BOULDER_TERRAINS_CFG

        # ── DR FIXES (from V5i) ──────────────────────────────────────
        self.events.base_external_force_torque.params["force_range"] = (-1.0, 1.0)
        self.events.base_external_force_torque.params["torque_range"] = (-0.5, 0.5)
        self.events.physics_material.params["static_friction_range"] = (0.5, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 0.8)
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.3, 0.3),
            "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.5, 0.5),
        }

        # ── ADAPTIVE REWARDS (same as V6) ─────────────────────────────
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
            func=adaptive_smoothness_penalty, weight=-0.4,
            params={},
        )
        self.rewards.terrain_relative_height = RewardTermCfg(
            func=adaptive_height_penalty, weight=-1.0,
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
            func=adaptive_slip_penalty, weight=-0.8,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        )

        # ── KINEMATIC CHAIN — MODERATE (same as V6) ──────────────────
        self.rewards.joint_pos.weight = -0.15
        self.rewards.base_pitch.weight = -0.05
        self.rewards.base_orientation.weight = -0.2
        self.rewards.base_motion.weight = -0.8
        self.rewards.air_time.weight = 4.0

        # ── EXPLOIT BLOCKERS (same as V6) ─────────────────────────────
        self.rewards.base_roll.weight = -7.0
        self.rewards.motor_power.weight = -0.0002
        self.rewards.torque_limit.weight = -0.05
        self.rewards.undesired_contacts.weight = -0.5

        # ── BOULDER REWARDS (same as V6) ──────────────────────────────
        self.rewards.flying_gait = RewardTermCfg(
            func=flying_gait_penalty, weight=-0.5,
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
            func=height_gain_reward, weight=1.5,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # ── NEW V7: REAR CLEARANCE BONUS ─────────────────────────────
        # Break the 22m wall — reward hind feet lifting during swing.
        # Only hl_foot and hr_foot, not front feet.
        self.rewards.rear_clearance = RewardTermCfg(
            func=rear_clearance_bonus, weight=3.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["hl_foot", "hr_foot"]),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )

        # ── FRONT FOOT OBSTACLE CLEARANCE ─────────────────────────────
        self.rewards.front_clearance = RewardTermCfg(
            func=front_foot_step_clearance_reward, weight=3.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["fl_foot", "fr_foot"]),
            },
        )

        # ── OBSTACLE COLLISION PENALTY (front feet hitting boulder faces) ─
        self.rewards.riser_collision = RewardTermCfg(
            func=riser_collision_penalty, weight=-2.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["fl_foot", "fr_foot"]),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["fl_foot", "fr_foot"]),
            },
        )

        # ── MINIMAL STANDING ─────────────────────────────────────────
        self.commands.base_velocity.rel_standing_envs = 0.20
