"""Vision60 Phase 1 PPO Environment Configuration.

UPGRADED from vision60_training/ baseline:
  - 12 terrain types (was 7) via shared ROBUST_TERRAINS_CFG
  - All 19 reward terms ENABLED (5 were zeroed in the original)
  - Vision60 body names from V60_PARAMS

235-dim observations (48 proprioceptive + 187 height scan).

Template: vision60_training/configs/vision60_env_cfg.py
Created for AI2C Tech Capstone — MS for Autonomy, February 2026
"""

import os
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

import sys

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

# Vision60 asset — loaded from URDF
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg

_URDF_PATH = os.path.join(os.path.expanduser("~"), "vision60_training", "urdf", "vision60_v5.urdf")

VISION60_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_URDF_PATH,
        fix_base=False,
        merge_fixed_joints=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=80.0,
                damping=2.0,
            ),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            "joint_0": 0.9, "joint_2": 0.9, "joint_4": 0.9, "joint_6": 0.9,
            "joint_1": 1.67, "joint_3": 1.67, "joint_5": 1.67, "joint_7": 1.67,
            "joint_8": 0.03, "joint_9": 0.03, "joint_10": -0.03, "joint_11": -0.03,
        },
    ),
    actuators={
        "vision60_all": ImplicitActuatorCfg(
            joint_names_expr=["joint_.*"],
            effort_limit=87.5,
            stiffness=80.0,
            damping=2.0,
        ),
    },
)


# =============================================================================
# Observations — 235-dim (same structure as Spot, robot-agnostic)
# =============================================================================

@configclass
class Vision60PPOObservationsCfg:
    """235-dim observation space (48 proprioceptive + 187 height scan)."""

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
class Vision60PPOActionsCfg:
    """12-dim joint position actions."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


# =============================================================================
# Commands
# =============================================================================

@configclass
class Vision60PPOCommandsCfg:
    """Velocity commands — same ranges as Spot."""

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
# Events — Progressive DR (starts mild, expands via dr_schedule.py)
# =============================================================================

@configclass
class Vision60PPOEventCfg:
    """Domain randomization for Vision60 — starts mild, expanded progressively."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.3),
            "dynamic_friction_range": (0.25, 1.1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
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
            "force_range": (-3.0, 3.0),
            "torque_range": (-1.0, 1.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.3, 0.3),
                "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-1.5, 1.5),
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
# Rewards — All 19 terms ENABLED with Vision60 body names
# UPGRADED: 5 terms that were zeroed in vision60_training now have active weights
# =============================================================================

@configclass
class Vision60PPORewardsCfg:
    """Full 19 reward terms adapted for Vision60."""

    # -- Task rewards (positive) --

    air_time = RewardTermCfg(
        func=spot_mdp.air_time_reward,
        weight=2.0,  # Reduced from Spot's 3.0 — heavier robot
        params={
            "mode_time": 0.3, "velocity_threshold": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="lower.*"),
        },
    )
    base_angular_velocity = RewardTermCfg(
        func=spot_mdp.base_angular_velocity_reward,
        weight=5.0,
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_linear_velocity = RewardTermCfg(
        func=spot_mdp.base_linear_velocity_reward,
        weight=12.0,
        params={"std": 1.0, "ramp_rate": 0.5, "ramp_at_vel": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    foot_clearance = RewardTermCfg(
        func=spot_mdp.foot_clearance_reward,
        weight=3.5,
        params={
            "std": 0.05, "tanh_mult": 2.0, "target_height": 0.08,
            "asset_cfg": SceneEntityCfg("robot", body_names="lower.*"),
        },
    )
    gait = RewardTermCfg(
        func=spot_mdp.GaitReward,
        weight=15.0,
        params={
            "std": 0.1, "max_err": 0.2, "velocity_threshold": 0.5,
            "synced_feet_pair_names": (("lower0", "lower3"), ("lower2", "lower1")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    # UPGRADED: Was weight=0.0 in vision60_training — now enabled
    velocity_modulation = RewardTermCfg(
        func=velocity_modulation_reward,
        weight=2.0,
        params={
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="lower.*"),
        },
    )

    # UPGRADED: Was weight=0.0 in vision60_training — now enabled
    vegetation_drag = RewardTermCfg(
        func=VegetationDragReward,
        weight=-0.001,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="lower.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="lower.*"),
            "drag_max": 20.0,
            "contact_threshold": 1.0,
            "vegetation_terrain_name": "vegetation_plane",
            "friction_terrain_name": "friction_plane",
        },
    )

    # -- Penalties (negative) --

    action_smoothness = RewardTermCfg(func=spot_mdp.action_smoothness_penalty, weight=-0.5)
    air_time_variance = RewardTermCfg(
        func=spot_mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="lower.*")},
    )
    base_motion = RewardTermCfg(
        func=spot_mdp.base_motion_penalty,
        weight=-3.0,  # Reduced from Spot's -4.0
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_orientation = RewardTermCfg(
        func=spot_mdp.base_orientation_penalty,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    foot_slip = RewardTermCfg(
        func=spot_mdp.foot_slip_penalty,
        weight=-2.0,  # Reduced from Spot's -3.0 — more weight = more traction
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="lower.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="lower.*"),
            "threshold": 1.0,
        },
    )
    joint_acc = RewardTermCfg(
        func=spot_mdp.joint_acceleration_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="joint_.*")},
    )
    joint_pos = RewardTermCfg(
        func=spot_mdp.joint_position_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0, "velocity_threshold": 0.5,
        },
    )
    # Penalize joints approaching URDF limits — prevents leg folding
    dof_pos_limits = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="joint_.*")},
    )
    joint_torques = RewardTermCfg(
        func=spot_mdp.joint_torques_penalty,
        weight=-2.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_vel = RewardTermCfg(
        func=spot_mdp.joint_velocity_penalty,
        weight=-3.0e-2,  # Reduced from Spot's -5e-2 — more inertia
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="joint_.*")},
    )

    # UPGRADED: Was weight=0.0 in vision60_training — now enabled
    body_height_tracking = RewardTermCfg(
        func=body_height_tracking_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.55},
    )

    # UPGRADED: Was weight=0.0 in vision60_training — now enabled
    contact_force_smoothness = RewardTermCfg(
        func=contact_force_smoothness_penalty,
        weight=-0.02,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="lower.*")},
    )

    # UPGRADED: Was weight=0.0 in vision60_training — now enabled
    stumble = RewardTermCfg(
        func=stumble_penalty,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="lower.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="lower.*"),
            "knee_height": 0.20, "force_threshold": 5.0,
        },
    )


# =============================================================================
# Terminations — body-only (relaxed for Vision60)
# =============================================================================

@configclass
class Vision60PPOTerminationsCfg:
    """Relaxed termination — only torso contact kills the episode."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body"]), "threshold": 1.0},
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
class Vision60PPOCurriculumCfg:
    """Terrain difficulty progression."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


# =============================================================================
# Main Environment Config
# =============================================================================

@configclass
class Vision60PPOEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Vision60 Phase 1 PPO Environment — 12 terrains, 19 rewards.

    UPGRADED from vision60_training/:
      - 12 terrain types (was 7)
      - All 19 reward terms enabled (was 14 active + 5 zeroed)
      - Same 235-dim obs, progressive DR, relaxed termination
    """

    observations: Vision60PPOObservationsCfg = Vision60PPOObservationsCfg()
    actions: Vision60PPOActionsCfg = Vision60PPOActionsCfg()
    commands: Vision60PPOCommandsCfg = Vision60PPOCommandsCfg()
    rewards: Vision60PPORewardsCfg = Vision60PPORewardsCfg()
    terminations: Vision60PPOTerminationsCfg = Vision60PPOTerminationsCfg()
    events: Vision60PPOEventCfg = Vision60PPOEventCfg()
    curriculum: Vision60PPOCurriculumCfg = Vision60PPOCurriculumCfg()

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

        # Vision60 robot
        self.scene.robot = VISION60_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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

        # ROBUST terrain — 12 types, 400 patches (UPGRADED from 7-type scratch)
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
