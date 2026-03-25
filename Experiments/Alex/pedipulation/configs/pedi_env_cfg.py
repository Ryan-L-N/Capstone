"""Pedipulation environment configuration for Spot.

240-dim observations: 48 proprio + 3 foot_target + 2 leg_flags + 187 height_scan.
Transfer-learning from hybrid_nocoach_19999.pt (MH-2a, [512,256,128]).

Based on: "Clearing Clutter on Staircases via Quadrupedal Pedipulation"
          (Sriganesh et al., CMU, arXiv:2509.20516)

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

import math

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

from quadruped_locomotion.tasks.locomotion.mdp.rewards import (
    clamped_action_smoothness_penalty,
    clamped_joint_torques_penalty,
    terrain_relative_height_penalty,
)

from mdp.commands import FootTargetCommandCfg, LegSelectionCommandCfg
from mdp.rewards import (
    FootTrackingReward,
    StandingStabilityReward,
    walking_reward,
    body_stillness_reward,
    passive_legs_penalty,
    foot_smoothness_penalty,
)
from mdp.terrains import PEDI_STAIRCASE_TERRAINS_CFG


# =============================================================================
# Observations — 240-dim (48 proprio + 5 pedipulation + 187 height scan)
# =============================================================================

@configclass
class PediObservationsCfg:
    """Observation specs: standard proprio + pedipulation commands + height scan."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group — 240 dims total."""

        # ---- Proprioceptive (48 dims) ----
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

        # ---- Pedipulation commands (5 dims) ----
        foot_target = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "foot_target"},
        )
        leg_flags = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "leg_selection"},
        )

        # ---- Height scan (187 dims) ----
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
# Actions — same 12-dim joint positions as locomotion
# =============================================================================

@configclass
class PediActionsCfg:
    """12-dim joint position actions."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.2, use_default_offset=True
    )


# =============================================================================
# Commands — velocity + foot target + leg selection
# =============================================================================

@configclass
class PediCommandsCfg:
    """Velocity commands for walking + pedipulation commands."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 2.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.5, 1.5)
        ),
    )

    # Leg selection MUST be declared before foot_target so it's initialized first
    # (FootTargetCommand reads leg_selection during resampling)
    leg_selection = LegSelectionCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 15.0),
        standing_fraction=0.6,
        debug_vis=False,
    )

    foot_target = FootTargetCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        x_range=(0.20, 0.55),
        y_range=(-0.20, 0.20),
        z_range=(-0.35, 0.10),
        debug_vis=False,
    )


# =============================================================================
# Events — domain randomization (matching Spot locomotion baseline)
# =============================================================================

@configclass
class PediEventCfg:
    """DR events — same as SpotEventCfg from locomotion training."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.5),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
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

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-1.0, 1.0), "y": (-0.5, 0.5), "z": (-0.3, 0.3),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.7, 0.7),
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
# Rewards — 12 terms (7 pedipulation + 5 shared locomotion)
# =============================================================================

@configclass
class PediRewardsCfg:
    """Pedipulation reward terms.

    Key design: rewards are conditioned on leg_selection flags.
    Walking rewards active when flags=[0,0], manipulation rewards when flag=1.
    """

    # -- Pedipulation task rewards (positive) --

    foot_tracking = RewardTermCfg(
        func=FootTrackingReward,
        weight=10.0,
        params={
            "sigma": 0.1,
            "fl_cfg": SceneEntityCfg("robot", body_names=["fl_foot"]),
            "fr_cfg": SceneEntityCfg("robot", body_names=["fr_foot"]),
        },
    )
    walking = RewardTermCfg(
        func=walking_reward,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 1.0},
    )
    standing_stability = RewardTermCfg(
        func=StandingStabilityReward,
        weight=5.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["fl_foot", "fr_foot", "hl_foot", "hr_foot"],
            ),
        },
    )
    body_stillness = RewardTermCfg(
        func=body_stillness_reward,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "lin_std": 0.5, "ang_std": 1.0},
    )

    # -- Pedipulation penalties --

    passive_legs = RewardTermCfg(
        func=passive_legs_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    foot_smoothness = RewardTermCfg(
        func=foot_smoothness_penalty,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # -- Shared locomotion penalties (always active) --

    action_smoothness = RewardTermCfg(
        func=clamped_action_smoothness_penalty,
        weight=-1.0,
    )
    joint_torques = RewardTermCfg(
        func=clamped_joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    base_orientation = RewardTermCfg(
        func=spot_mdp.base_orientation_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    terrain_relative_height = RewardTermCfg(
        func=terrain_relative_height_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "terrain_scaled": True,
            "height_easy": 0.42,
            "height_hard": 0.35,
            "variance_flat": 0.001,
            "variance_rough": 0.02,
        },
    )
    dof_pos_limits = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body"]),
            "threshold": 1.0,
        },
    )


# =============================================================================
# Terminations
# =============================================================================

@configclass
class PediTerminationsCfg:
    """Same termination conditions as locomotion — flip-over + timeout."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_flip_over = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": math.radians(150.0)},
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
class PediCurriculumCfg:
    """Terrain difficulty progression."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


# =============================================================================
# Main Environment Config
# =============================================================================

@configclass
class PedipulationSpotEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Pedipulation environment for Spot.

    240-dim observations: 48 proprio + 3 foot_target + 2 leg_flags + 187 height_scan.
    Transfer-learned from hybrid_nocoach_19999.pt via weight surgery.
    """

    observations: PediObservationsCfg = PediObservationsCfg()
    actions: PediActionsCfg = PediActionsCfg()
    commands: PediCommandsCfg = PediCommandsCfg()
    rewards: PediRewardsCfg = PediRewardsCfg()
    terminations: PediTerminationsCfg = PediTerminationsCfg()
    events: PediEventCfg = PediEventCfg()
    curriculum: PediCurriculumCfg = PediCurriculumCfg()

    viewer = ViewerCfg(
        eye=(10.5, 10.5, 3.0), origin_type="world", env_index=0, asset_name="robot"
    )

    def __post_init__(self):
        super().__post_init__()

        # Physics — 500 Hz sim, decimation=10 → 50 Hz control
        self.decimation = 10
        self.episode_length_s = 30.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
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

        # Default terrain: staircase-heavy (overridden by --phase in train_pedi.py)
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=PEDI_STAIRCASE_TERRAINS_CFG,
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
class PedipulationSpotEnvCfg_PLAY(PedipulationSpotEnvCfg):
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
