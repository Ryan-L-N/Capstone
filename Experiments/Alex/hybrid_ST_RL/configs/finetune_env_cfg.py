"""Stage 1: Progressive Fine-Tuning Environment Configuration.

Identical to Spot100hrEnvCfg EXCEPT domain randomization starts at 48hr-like
values and the training script progressively expands them over 15K iterations.

Key differences from 100hr env_cfg.py:
  - physics_material: mode="reset" (not "startup") so friction re-randomizes
    as the progressive schedule expands the range
  - add_base_mass: mode="reset" (not "startup") for same reason
  - DR starts at 48hr-compatible values (friction [0.3, 1.3], push +/-0.5, etc.)
  - Everything else identical: 12 terrain types, 18 reward terms, 30s episodes

Created for AI2C Tech Capstone — hybrid_ST_RL, February 2026
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

##
# Pre-defined configs
##
from isaaclab_assets.robots.spot import SPOT_CFG  # isort: skip

# Our custom terrain and reward configs
import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from configs.terrain_cfg import ROBUST_TERRAINS_CFG
from rewards.reward_terms import (
    VegetationDragReward,
    body_height_tracking_penalty,
    contact_force_smoothness_penalty,
    stumble_penalty,
    velocity_modulation_reward,
)


# =============================================================================
# Observations — same 235-dim space as 48hr and 100hr configs
# =============================================================================

@configclass
class SpotFinetuneObservationsCfg:
    """235-dim observation space (48 proprioceptive + 187 height scan).

    Noise levels match the 100hr config (increased ~50% from 48hr).
    """

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
# Actions — same as 48hr config
# =============================================================================

@configclass
class SpotFinetuneActionsCfg:
    """12-dim joint position actions, scale=0.25."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


# =============================================================================
# Commands — same velocity ranges as 48hr config
# =============================================================================

@configclass
class SpotFinetuneCommandsCfg:
    """Velocity commands — same as 48hr and 100hr configs."""

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
# Events — Progressive DR (starts at 48hr-like values)
# =============================================================================

@configclass
class SpotFinetuneEventCfg:
    """Domain randomization starting at 48hr-compatible values.

    CRITICAL: physics_material and add_base_mass use mode="reset" (not "startup")
    so the training script can progressively expand the ranges and have them
    take effect at each episode reset.

    The training script (train_finetune.py) linearly interpolates these params
    from the start values below to the target values over 15K iterations.
    """

    # --- Reset events (re-randomized each episode) ---

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",  # CHANGED from "startup" — enables progressive expansion
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.3),     # Start: 48hr-like
            "dynamic_friction_range": (0.25, 1.1),    # Start: 48hr-like
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",  # CHANGED from "startup" — enables progressive expansion
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-5.0, 5.0),  # Start: same as 48hr
            "operation": "add",
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "force_range": (-3.0, 3.0),    # Start: same as 48hr
            "torque_range": (-1.0, 1.0),   # Start: same as 48hr
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
            "velocity_range": (-2.5, 2.5),   # Start: same as 48hr
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # --- Interval events ---

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),  # Start: same as 48hr
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},  # Start: same as 48hr
        },
    )


# =============================================================================
# Rewards — Same 18 terms as 100hr config (unchanged)
# =============================================================================

@configclass
class SpotFinetuneRewardsCfg:
    """Reward terms — identical to Spot100hrRewardsCfg.

    All 18 terms including 4 custom (velocity_modulation, vegetation_drag,
    body_height_tracking, contact_force_smoothness, stumble).
    """

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
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.10,
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
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
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
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": 0.42,
        },
    )

    contact_force_smoothness = RewardTermCfg(
        func=contact_force_smoothness_penalty,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )

    stumble = RewardTermCfg(
        func=stumble_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "knee_height": 0.15,
            "force_threshold": 5.0,
        },
    )


# =============================================================================
# Terminations — same as 100hr / 48hr
# =============================================================================

@configclass
class SpotFinetuneTerminationsCfg:
    """Termination terms — identical to 48hr and 100hr configs."""

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
# Curriculum — terrain difficulty progression
# =============================================================================

@configclass
class SpotFinetuneCurriculumCfg:
    """Automatic terrain curriculum (promotion/demotion based on velocity)."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


# =============================================================================
# Main Environment Config
# =============================================================================

@configclass
class SpotFinetuneEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Stage 1 Progressive Fine-Tuning Environment.

    Identical to Spot100hrEnvCfg except:
    - DR events use mode="reset" for progressive expansion
    - DR starts at 48hr-compatible values
    - GPU buffers sized for 8K envs (not 65K)
    - 30s episodes, 12 terrain types, 18 reward terms
    """

    observations: SpotFinetuneObservationsCfg = SpotFinetuneObservationsCfg()
    actions: SpotFinetuneActionsCfg = SpotFinetuneActionsCfg()
    commands: SpotFinetuneCommandsCfg = SpotFinetuneCommandsCfg()
    rewards: SpotFinetuneRewardsCfg = SpotFinetuneRewardsCfg()
    terminations: SpotFinetuneTerminationsCfg = SpotFinetuneTerminationsCfg()
    events: SpotFinetuneEventCfg = SpotFinetuneEventCfg()
    curriculum: SpotFinetuneCurriculumCfg = SpotFinetuneCurriculumCfg()

    viewer = ViewerCfg(eye=(10.5, 10.5, 3.0), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        super().__post_init__()

        # Physics — 500 Hz with decimation=10 -> 50 Hz control
        self.decimation = 10
        self.episode_length_s = 30.0  # 50% longer than 48hr's 20s
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"

        # GPU PhysX buffers — sized for 8K envs with 12 terrain types
        self.sim.physx.gpu_collision_stack_size = 2**30  # 1 GB (enough for 8K)
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23

        # Sensors
        self.scene.contact_forces.update_period = self.sim.dt

        # Spot robot
        self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Height scanner — 17x11 grid = 187 dims, 0.1m resolution
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # ROBUST terrain — 12 types, 400 patches, 10x40 grid
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
