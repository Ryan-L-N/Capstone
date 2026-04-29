"""Boulder Master V6 — Dense unavoidable obstacles + DR fixes.

Start from boulder_v5_3400 (22.1m zone 3 record — got foot stuck, not a gait problem).
Dense obstacle terrain: robot MUST engage boulders to cover distance.
DR fixes from V5i: push forces, higher friction, narrower reset.
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
)
from rewards.stair_climbing_rewards import height_gain_reward
from rewards.stair_rewards import (
    flying_gait_penalty,
    dont_wait_penalty,
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

DENSE_BOULDER_TERRAINS_CFG = TerrainGeneratorCfg(
    **_COMMON,
    sub_terrains={
        # Dense discrete obstacles (40%) — unavoidable, must step over/through
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.40,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.10, 0.80),
            obstacle_height_range=(0.03, 0.60),
            num_obstacles=80,
            platform_width=1.5,
            border_width=0.25,
        ),
        # Random grid boxes (25%) — flat tops for stepping practice
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.25,
            grid_width=0.45,
            grid_height_range=(0.05, 0.35),
            platform_width=2.0,
        ),
        # Random rough ground (20%) — uneven footing between obstacles
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(0.03, 0.15),
            noise_step=0.02,
            border_width=0.25,
        ),
        # Repeated boxes (10%) — progressive size scaling
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
        # Minimal flat (5%) — just enough for gait maintenance
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.05,
        ),
    },
)


@configclass
class SpotBoulderV6EnvCfg(SpotS2RBaseEnvCfg):
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

        # ── ADAPTIVE REWARDS ──────────────────────────────────────────
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

        # ── KINEMATIC CHAIN — MODERATE ────────────────────────────────
        self.rewards.joint_pos.weight = -0.15
        self.rewards.base_pitch.weight = -0.05
        self.rewards.base_orientation.weight = -0.2
        self.rewards.base_motion.weight = -0.8
        self.rewards.air_time.weight = 4.0           # Higher — hop-enabling for boulders

        # ── EXPLOIT BLOCKERS ──────────────────────────────────────────
        self.rewards.base_roll.weight = -7.0
        self.rewards.motor_power.weight = -0.0002
        self.rewards.torque_limit.weight = -0.05
        self.rewards.undesired_contacts.weight = -0.5

        # ── BOULDER REWARDS ───────────────────────────────────────────
        self.rewards.flying_gait = RewardTermCfg(
            func=flying_gait_penalty, weight=-0.5,     # Loose — allow hopping over boulders
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

        # ── MINIMAL STANDING ─────────────────────────────────────────
        self.commands.base_velocity.rel_standing_envs = 0.20
