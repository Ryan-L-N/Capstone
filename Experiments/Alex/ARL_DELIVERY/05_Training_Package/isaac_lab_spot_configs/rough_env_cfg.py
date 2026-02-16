# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Spot Rough Terrain Environment Configuration
# =============================================
# Extends the base LocomotionVelocityRoughEnvCfg with Spot-specific
# rewards, events, observations, and physics settings.
#
# Key differences from SpotFlatEnvCfg:
#   - Height scanner ENABLED (160-dim terrain perception)
#   - ROUGH_TERRAINS_CFG (stairs, boxes, slopes) instead of cobblestone
#   - Observation corruption enabled (domain randomization on obs)
#   - Terrain curriculum enabled (progressive difficulty)
#   - Wider friction/mass randomization for robustness
#   - Increased foot_clearance reward weight for stair climbing
#
# Created for AI2C Tech Capstone - MS for Autonomy, February 2026

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

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab_assets.robots.spot import SPOT_CFG  # isort: skip


# =============================================================================
# Observations -- add height_scan to Spot's proprioceptive obs
# =============================================================================

@configclass
class SpotRoughObservationsCfg:
    """Observation specifications for Spot rough terrain.

    Same proprioceptive terms as SpotFlatEnvCfg plus 160-dim height scan.
    Observation corruption is ENABLED for domain randomization.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Proprioceptive observations (same as flat, Spot-specific noise values)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
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

        # Height scan -- THE KEY ADDITION for rough terrain
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True  # Domain randomization on obs
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


# =============================================================================
# Actions -- slightly higher scale than flat for rough terrain agility
# =============================================================================

@configclass
class SpotRoughActionsCfg:
    """Action specifications for Spot rough terrain."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


# =============================================================================
# Commands -- same as flat config
# =============================================================================

@configclass
class SpotRoughCommandsCfg:
    """Command specifications for Spot rough terrain."""

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
# Events -- wider randomization than flat for robustness
# =============================================================================

@configclass
class SpotRoughEventCfg:
    """Domain randomization for Spot rough terrain training."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-1.5, 1.5),
                "y": (-1.0, 1.0),
                "z": (-0.5, 0.5),
                "roll": (-0.7, 0.7),
                "pitch": (-0.7, 0.7),
                "yaw": (-1.0, 1.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=spot_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-2.5, 2.5),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


# =============================================================================
# Rewards -- Spot-specific rewards with boosted foot_clearance for stairs
# =============================================================================

@configclass
class SpotRoughRewardsCfg:
    """Reward terms for Spot rough terrain.

    Same structure as SpotFlatEnvCfg rewards but with increased
    foot_clearance weight to encourage higher stepping for stairs.
    """

    # -- task
    air_time = RewardTermCfg(
        func=spot_mdp.air_time_reward,
        weight=5.0,
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
        weight=5.0,
        params={"std": 1.0, "ramp_rate": 0.5, "ramp_at_vel": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    foot_clearance = RewardTermCfg(
        func=spot_mdp.foot_clearance_reward,
        weight=2.0,  # Increased from 0.5 to encourage high stepping on stairs
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.12,  # Slightly higher target for stairs (was 0.1)
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    gait = RewardTermCfg(
        func=spot_mdp.GaitReward,
        weight=10.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "synced_feet_pair_names": (("fl_foot", "hr_foot"), ("fr_foot", "hl_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    # -- penalties (same as flat)
    action_smoothness = RewardTermCfg(func=spot_mdp.action_smoothness_penalty, weight=-1.0)
    air_time_variance = RewardTermCfg(
        func=spot_mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    base_motion = RewardTermCfg(
        func=spot_mdp.base_motion_penalty, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    base_orientation = RewardTermCfg(
        func=spot_mdp.base_orientation_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    foot_slip = RewardTermCfg(
        func=spot_mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )
    joint_acc = RewardTermCfg(
        func=spot_mdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
    )
    joint_pos = RewardTermCfg(
        func=spot_mdp.joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
        },
    )
    joint_torques = RewardTermCfg(
        func=spot_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_vel = RewardTermCfg(
        func=spot_mdp.joint_velocity_penalty,
        weight=-1.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
    )


# =============================================================================
# Terminations -- same as flat
# =============================================================================

@configclass
class SpotRoughTerminationsCfg:
    """Termination terms for Spot rough terrain."""

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
# Curriculum -- enable terrain difficulty progression
# =============================================================================

@configclass
class SpotRoughCurriculumCfg:
    """Curriculum for progressive terrain difficulty."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


# =============================================================================
# Main Environment Config
# =============================================================================

@configclass
class SpotRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Spot rough terrain locomotion environment.

    Inherits from LocomotionVelocityRoughEnvCfg (which provides the rough
    terrain scene with height scanner, contact forces, etc.) and overrides
    with Spot-specific observations, actions, rewards, events, and physics.

    Key features:
    - 208-dim observation (48 proprioceptive + 160 height scan)
    - ROUGH_TERRAINS_CFG with stairs, boxes, slopes (6 terrain types)
    - Spot-specific gait reward (trot enforcer)
    - Terrain curriculum (progressive difficulty)
    - 500 Hz physics with decimation=10 (50 Hz control)
    """

    # Override MDP components with Spot-specific versions
    observations: SpotRoughObservationsCfg = SpotRoughObservationsCfg()
    actions: SpotRoughActionsCfg = SpotRoughActionsCfg()
    commands: SpotRoughCommandsCfg = SpotRoughCommandsCfg()
    rewards: SpotRoughRewardsCfg = SpotRoughRewardsCfg()
    terminations: SpotRoughTerminationsCfg = SpotRoughTerminationsCfg()
    events: SpotRoughEventCfg = SpotRoughEventCfg()
    curriculum: SpotRoughCurriculumCfg = SpotRoughCurriculumCfg()

    # Viewer
    viewer = ViewerCfg(eye=(10.5, 10.5, 3.0), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        # Post init of parent (sets up rough terrain scene, sensors, etc.)
        super().__post_init__()

        # Spot-specific physics (500 Hz like flat config)
        self.decimation = 10  # 50 Hz control
        self.episode_length_s = 20.0
        self.sim.dt = 0.002  # 500 Hz physics
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"

        # Update sensor periods for 500 Hz physics
        self.scene.contact_forces.update_period = self.sim.dt

        # Swap in the Spot robot
        self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Height scanner -- CRITICAL: Spot's base body is "body", not "base"
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # Rough terrain with full terrain types
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG,
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
class SpotRoughEnvCfg_PLAY(SpotRoughEnvCfg):
    """Reduced-size evaluation variant for visual testing."""

    def __post_init__(self):
        super().__post_init__()

        # Smaller scene for evaluation
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None

        # Reduce terrain patches to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Disable randomization for clean evaluation
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
