"""Spot Phase 1 PPO Environment Configuration.

Uses shared ROBUST_TERRAINS_CFG (12 types) and all 19 reward terms.
235-dim observations (48 proprioceptive + 187 height scan).

Template: 100hr_env_run/configs/env_cfg.py
Created for AI2C Tech Capstone — MS for Autonomy, February 2026
"""

import isaaclab.sim as sim_utils
from isaaclab.envs import ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from isaaclab_assets.robots.spot import SPOT_CFG  # isort: skip

import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from shared.terrain_cfg import ROBUST_TERRAINS_CFG
from shared.reward_terms import (
    VegetationDragReward,
    body_height_tracking_penalty,
    contact_force_smoothness_penalty,
    stumble_penalty,
    velocity_modulation_reward,
)


# =============================================================================
# Observations — 235-dim (48 proprioceptive + 187 height scan)
# =============================================================================

@configclass
class SpotPPOObservationsCfg:
    """Observation specifications with increased noise for robustness."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.15, n_max=0.15),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.15, n_max=0.15),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        actions = ObsTerm(func=mdp.last_action)

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.15, n_max=0.15),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# =============================================================================
# Actions
# =============================================================================

@configclass
class SpotPPOActionsCfg:
    """12-dim joint position actions."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


# =============================================================================
# Commands
# =============================================================================

@configclass
class SpotPPOCommandsCfg:
    """Velocity command ranges."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 3.0), lin_vel_y=(-1.5, 1.5), ang_vel_z=(-2.0, 2.0)
        ),
    )


# =============================================================================
# Events — Aggressive domain randomization
# =============================================================================

@configclass
class SpotPPOEventCfg:
    """Massively expanded domain randomization for Spot."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.05, 1.5),
            "dynamic_friction_range": (0.02, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-8.0, 8.0),
            "operation": "add",
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "force_range": (-8.0, 8.0),
            "torque_range": (-3.0, 3.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-1.5, 1.5), "y": (-1.0, 1.0), "z": (-0.5, 0.5),
                "roll": (-0.7, 0.7), "pitch": (-0.7, 0.7), "yaw": (-1.0, 1.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=spot_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-3.0, 3.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 12.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)},
        },
    )


# =============================================================================
# Rewards — All 19 terms with Spot body names
# =============================================================================

@configclass
class SpotPPORewardsCfg:
    """Full 19 reward terms for Spot robust training."""

    # -- Task rewards (positive) --

    air_time = RewardTermCfg(
        func=spot_mdp.air_time_reward,
        weight=3.0,
        params={
            "mode_time": 0.3,
            "velocity_threshold": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    base_angular_velocity = RewardTermCfg(
        func=spot_mdp.base_angular_velocity_reward,
        weight=5.0,
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_linear_velocity = RewardTermCfg(
        func=spot_mdp.base_linear_velocity_reward,
        weight=7.0,
        params={"std": 1.0, "ramp_rate": 0.5, "ramp_at_vel": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    foot_clearance = RewardTermCfg(
        func=spot_mdp.foot_clearance_reward,
        weight=3.5,
        params={
            "std": 0.05, "tanh_mult": 2.0, "target_height": 0.10,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    gait = RewardTermCfg(
        func=spot_mdp.GaitReward,
        weight=10.0,
        params={
            "std": 0.1, "max_err": 0.2, "velocity_threshold": 0.5,
            "synced_feet_pair_names": (("fl_foot", "hr_foot"), ("fr_foot", "hl_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )
    velocity_modulation = RewardTermCfg(
        func=velocity_modulation_reward,
        weight=2.0,
        params={
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    vegetation_drag = RewardTermCfg(
        func=VegetationDragReward,
        weight=-0.001,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "drag_max": 20.0,
            "contact_threshold": 1.0,
            "vegetation_terrain_name": "vegetation_plane",
            "friction_terrain_name": "friction_plane",
        },
    )

    # -- Penalties (negative) --

    action_smoothness = RewardTermCfg(func=spot_mdp.action_smoothness_penalty, weight=-2.0)
    air_time_variance = RewardTermCfg(
        func=spot_mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    base_motion = RewardTermCfg(
        func=spot_mdp.base_motion_penalty,
        weight=-4.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_orientation = RewardTermCfg(
        func=spot_mdp.base_orientation_penalty,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    foot_slip = RewardTermCfg(
        func=spot_mdp.foot_slip_penalty,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )
    joint_acc = RewardTermCfg(
        func=spot_mdp.joint_acceleration_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
    )
    joint_pos = RewardTermCfg(
        func=spot_mdp.joint_position_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0, "velocity_threshold": 0.5,
        },
    )
    joint_torques = RewardTermCfg(
        func=spot_mdp.joint_torques_penalty,
        weight=-2.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_vel = RewardTermCfg(
        func=spot_mdp.joint_velocity_penalty,
        weight=-5.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
    )
    body_height_tracking = RewardTermCfg(
        func=body_height_tracking_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.42},
    )
    contact_force_smoothness = RewardTermCfg(
        func=contact_force_smoothness_penalty,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    stumble = RewardTermCfg(
        func=stumble_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "knee_height": 0.15, "force_threshold": 5.0,
        },
    )


# =============================================================================
# Terminations
# =============================================================================

@configclass
class SpotPPOTerminationsCfg:
    """Termination terms for Spot."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*leg"]), "threshold": 1.0},
    )
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


# =============================================================================
# Curriculum
# =============================================================================

@configclass
class SpotPPOCurriculumCfg:
    """Terrain difficulty progression."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


# =============================================================================
# Main Environment Config
# =============================================================================

@configclass
class SpotPPOEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Spot Phase 1 PPO Environment — 12 terrains, 19 rewards, full DR.

    235-dim observations (48 proprioceptive + 187 height scan).
    Target: H100 NVL 96GB with 20,480 parallel environments.
    """

    observations: SpotPPOObservationsCfg = SpotPPOObservationsCfg()
    actions: SpotPPOActionsCfg = SpotPPOActionsCfg()
    commands: SpotPPOCommandsCfg = SpotPPOCommandsCfg()
    rewards: SpotPPORewardsCfg = SpotPPORewardsCfg()
    terminations: SpotPPOTerminationsCfg = SpotPPOTerminationsCfg()
    events: SpotPPOEventCfg = SpotPPOEventCfg()
    curriculum: SpotPPOCurriculumCfg = SpotPPOCurriculumCfg()

    viewer = ViewerCfg(eye=(10.5, 10.5, 3.0), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        super().__post_init__()

        # Physics — 500 Hz with decimation=10 → 50 Hz control
        self.decimation = 10
        self.episode_length_s = 30.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"

        # GPU PhysX buffers
        self.sim.physx.gpu_collision_stack_size = 2**31
        self.sim.physx.gpu_max_rigid_contact_count = 2**24
        self.sim.physx.gpu_max_rigid_patch_count = 2**24

        self.scene.contact_forces.update_period = self.sim.dt

        # Spot robot
        self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Height scanner — 17x11 grid = 187 dims
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # ROBUST terrain — 12 types, 400 patches
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=ROBUST_TERRAINS_CFG,
            max_init_terrain_level=5,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )


@configclass
class SpotPPOEnvCfg_PLAY(SpotPPOEnvCfg):
    """Reduced-size variant for visual testing."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
