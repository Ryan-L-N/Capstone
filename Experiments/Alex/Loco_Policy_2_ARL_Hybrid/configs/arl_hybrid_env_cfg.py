"""Mason Hybrid Spot Environment — Mason's proven rewards + our terrain + AI Coach.

Uses Mason's clean 11-reward structure with his proven weights, plus 3 surgical
additions from our system (terrain_relative_height, dof_pos_limits, clamped
action_smoothness). Trained on our ROBUST_TERRAINS_CFG (12 types, 400 patches).

The AI Coach activates after a configurable silent period to push through
terrain plateaus.

Key differences from our SpotLocomotionEnvCfg:
  - Mason's reward weights (gait=10, vel=5, joint_pos=-0.7, orientation=-3.0)
  - Mason's [512, 256, 128] network (not [1024, 512, 256])
  - Mason's lighter DR (mass ±2.5, friction 0.3-1.0)
  - Mason's observation noise disabled (enable_corruption=False)
  - Mason's body_contact termination (hard kill, not soft penalty)
  - Mason's velocity_threshold=0.5, mode_time=0.3
  - Mason's hip-only joint_acc/joint_vel penalties
  - Dropped: velocity_modulation, vegetation_drag, undesired_contacts,
             body_scraping, contact_force_smoothness
  - 14 total reward terms (Mason's 11 + terrain_relative_height + dof_pos_limits
    + body_height_tracking/stumble frozen at 0)

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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg

import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from isaaclab_assets.robots.spot import SPOT_CFG  # isort: skip

from quadruped_locomotion.tasks.locomotion.mdp.terrains import ROBUST_TERRAINS_CFG
from quadruped_locomotion.tasks.locomotion.mdp.rewards import (
    body_height_tracking_penalty,
    clamped_action_smoothness_penalty,
    clamped_joint_acceleration_penalty,
    clamped_joint_torques_penalty,
    clamped_joint_velocity_penalty,
    stumble_penalty,
    terrain_relative_height_penalty,
)


# =============================================================================
# Observations — Mason's config (no corruption noise)
# =============================================================================

@configclass
class HybridObservationsCfg:
    """Mason's observation config — corruption disabled for cleaner learning."""

    @configclass
    class PolicyCfg(ObsGroup):

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
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

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# =============================================================================
# Actions
# =============================================================================

@configclass
class HybridActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.2, use_default_offset=True
    )


# =============================================================================
# Commands — Mason's velocity ranges
# =============================================================================

@configclass
class HybridCommandsCfg:
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
# Events — Mason's lighter domain randomization
# =============================================================================

@configclass
class HybridEventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-2.5, 2.5),
            "operation": "add",
        },
    )

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
# Rewards — Mason's 11 terms + 3 surgical additions (14 total)
# =============================================================================

@configclass
class HybridRewardsCfg:
    """Mason's proven weights + terrain_relative_height + dof_pos_limits."""

    # -- Task rewards (Mason's exact values) --

    air_time = RewardTermCfg(
        func=spot_mdp.air_time_reward,
        weight=5.0,
        params={
            "mode_time": 0.3,            # Mason's (ours was 0.2)
            "velocity_threshold": 0.5,   # Mason's (ours was 0.25)
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    base_angular_velocity = RewardTermCfg(
        func=spot_mdp.base_angular_velocity_reward,
        weight=5.0,  # Mason's (ours drifted to 11.09)
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_linear_velocity = RewardTermCfg(
        func=spot_mdp.base_linear_velocity_reward,
        weight=5.0,  # Mason's (ours drifted to 14.26)
        params={"std": 1.0, "ramp_rate": 0.5, "ramp_at_vel": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    foot_clearance = RewardTermCfg(
        func=spot_mdp.foot_clearance_reward,
        weight=0.5,  # Mason's (ours was 2.0)
        params={
            "std": 0.05, "tanh_mult": 2.0, "target_height": 0.1,  # Mason's target 0.1 (ours 0.125)
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    gait = RewardTermCfg(
        func=spot_mdp.GaitReward,
        weight=10.0,  # Mason's (ours was 8.5)
        params={
            "std": 0.1, "max_err": 0.2,
            "velocity_threshold": 0.5,   # Mason's (ours was 0.25)
            "synced_feet_pair_names": (("fl_foot", "hr_foot"), ("fr_foot", "hl_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    # -- Penalties (Mason's exact values, clamped action_smoothness) --

    action_smoothness = RewardTermCfg(
        func=clamped_action_smoothness_penalty,  # Our clamped version (Bug #29 safety)
        weight=-1.0,  # Mason's
    )
    air_time_variance = RewardTermCfg(
        func=spot_mdp.air_time_variance_penalty,
        weight=-1.0,  # Mason's
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    base_motion = RewardTermCfg(
        func=spot_mdp.base_motion_penalty,
        weight=-2.0,  # Mason's (ours drifted to -2.88)
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_orientation = RewardTermCfg(
        func=spot_mdp.base_orientation_penalty,
        weight=-3.0,  # Mason's (ours drifted to -2.4)
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    foot_slip = RewardTermCfg(
        func=spot_mdp.foot_slip_penalty,
        weight=-0.5,  # Same in both
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )
    joint_acc = RewardTermCfg(
        func=clamped_joint_acceleration_penalty,  # Clamped wrapper (Bug #29 safety)
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},  # Mason's: hip only
    )
    joint_pos = RewardTermCfg(
        func=spot_mdp.joint_position_penalty,
        weight=-0.7,  # Mason's (ours was -0.3 — too loose)
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,  # Mason's (ours was 0.25)
        },
    )
    joint_torques = RewardTermCfg(
        func=clamped_joint_torques_penalty,  # Clamped wrapper (Bug #29 safety)
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_vel = RewardTermCfg(
        func=clamped_joint_velocity_penalty,  # Clamped wrapper (Bug #29 safety)
        weight=-1.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},  # Mason's: hip only
    )

    # -- Our additions (proven necessary, not in Mason's config) --

    terrain_relative_height = RewardTermCfg(
        func=terrain_relative_height_penalty,
        weight=-2.0,  # Prevents belly-crawl exploit (Bug #27)
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "terrain_scaled": False,
            "target_height": 0.37,  # Fixed standing height — robot MUST stand up
        },
    )
    dof_pos_limits = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-3.0,  # Prevents knee locking at URDF limits
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    # Frozen at 0 — present so coach can see them but guardrails prevent changes
    body_height_tracking = RewardTermCfg(
        func=body_height_tracking_penalty,
        weight=0.0,  # Bug #22: world-frame Z
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.42},
    )
    stumble = RewardTermCfg(
        func=stumble_penalty,
        weight=0.0,  # Bug #28b: world-frame Z
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "knee_height": 0.15, "force_threshold": 5.0,
        },
    )


# =============================================================================
# Terminations — Mason's body_contact + our flip_over
# =============================================================================

@configclass
class HybridTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*leg"]),
            "threshold": 1.0,
        },
    )
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
class HybridCurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


# =============================================================================
# Scene — must define terrain/robot/sensors as class attributes so they exist
# before LocomotionVelocityRoughEnvCfg.__post_init__() runs
# =============================================================================

@configclass
class HybridSceneCfg(InteractiveSceneCfg):
    # terrain
    terrain = TerrainImporterCfg(
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
        debug_vis=True,
    )

    # robot
    robot: ArticulationCfg = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# =============================================================================
# Main Environment Config
# =============================================================================

@configclass
class SpotARLHybridEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Mason's proven rewards + our robust terrain + AI Coach ready.

    Network: [512, 256, 128] (Mason's — 3x smaller, trains faster)
    Terrain: ROBUST_TERRAINS_CFG (our 12-type, 400-patch config)
    Rewards: 14 terms (Mason's 11 + terrain_relative_height + dof_pos_limits + 2 frozen)
    DR: Mason's lighter randomization
    """

    scene: HybridSceneCfg = HybridSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: HybridObservationsCfg = HybridObservationsCfg()
    actions: HybridActionsCfg = HybridActionsCfg()
    commands: HybridCommandsCfg = HybridCommandsCfg()
    rewards: HybridRewardsCfg = HybridRewardsCfg()
    terminations: HybridTerminationsCfg = HybridTerminationsCfg()
    events: HybridEventCfg = HybridEventCfg()
    curriculum: HybridCurriculumCfg = HybridCurriculumCfg()

    viewer = ViewerCfg(eye=(10.5, 10.5, 3.0), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        super().__post_init__()

        # Physics — 500 Hz sim, 50 Hz control (Mason's decimation=10)
        self.decimation = 10
        self.episode_length_s = 20.0  # Mason's (ours was 30)
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"

        # GPU PhysX buffers — prevent collision stack overflow on dense terrain
        self.sim.physx.gpu_collision_stack_size = 2**31
        self.sim.physx.gpu_max_rigid_contact_count = 2**24
        self.sim.physx.gpu_max_rigid_patch_count = 2**24

        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class SpotARLHybridEnvCfg_PLAY(SpotARLHybridEnvCfg):
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
