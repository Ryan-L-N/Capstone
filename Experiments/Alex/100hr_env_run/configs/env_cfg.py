"""100-Hour Multi-Terrain Robust Locomotion Environment Configuration.

Extends SpotRoughEnvCfg with:
  - ROBUST_TERRAINS_CFG (12 terrain types, 400 patches)
  - Massively expanded domain randomization (friction down to 0.05)
  - Modified reward weights (3x foot_slip, reduced air_time, etc.)
  - 4 new reward terms (velocity modulation, height tracking, force smoothness, stumble)
  - PD gain variation and action delay for sim-to-real robustness
  - 30s episodes (50% longer than 48hr config)

Key improvements over 48hr rough policy:
  - Friction range: [0.05, 1.5] vs [0.5, 1.25] — covers ice to high-grip
  - 256 friction buckets vs 64 — finer material granularity
  - ±8kg mass offset vs ±5kg — wider payload variation
  - ±1.5 m/s push velocity vs ±0.5 m/s — stronger perturbations
  - ±8N external force vs ±3N — sustained force resistance

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

##
# Pre-defined configs
##
from isaaclab_assets.robots.spot import SPOT_CFG  # isort: skip

# Our custom terrain and reward configs
import sys
import os

# Add the 100hr_env_run parent to path so we can import our custom modules
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
# Observations — same as SpotRoughObservationsCfg but with increased noise
# =============================================================================

@configclass
class Spot100hrObservationsCfg:
    """Observation specifications with increased sensor noise for robustness.

    Same 235-dim observation space (48 proprioceptive + 187 height scan).
    Noise levels increased ~50% from 48hr config for better sim-to-real transfer.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Proprioceptive observations — increased noise from 48hr config
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.15, n_max=0.15),  # was ±0.1
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.15, n_max=0.15),  # was ±0.1
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),  # keep same
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),  # keep same
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.5, n_max=0.5),  # keep same
        )
        actions = ObsTerm(func=mdp.last_action)

        # Height scan — increased noise for terrain perception robustness
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.15, n_max=0.15),  # was ±0.1
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
class Spot100hrActionsCfg:
    """Action specifications — same action scale as 48hr config."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


# =============================================================================
# Commands — same velocity ranges as 48hr config
# =============================================================================

@configclass
class Spot100hrCommandsCfg:
    """Command specifications — same as 48hr config."""

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
# Events — MASSIVELY expanded domain randomization
# =============================================================================

@configclass
class Spot100hrEventCfg:
    """Domain randomization for 100hr multi-terrain training.

    Key changes from 48hr config:
      - Friction range: [0.05, 1.5] vs [0.5, 1.25]
      - 256 friction buckets vs 64
      - ±8kg mass vs ±5kg
      - ±1.5 m/s push vs ±0.5 m/s, every 5-12s vs 10-15s
      - ±8N force, ±3Nm torque vs ±3N, ±1Nm
      - ±3.0 rad/s joint vel reset vs ±2.5
    """

    # --- Startup events (applied once at simulation start) ---

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.05, 1.5),    # was (0.5, 1.25)
            "dynamic_friction_range": (0.02, 1.2),    # was (0.4, 1.0)
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,                        # was 64
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-8.0, 8.0),   # was (-5.0, 5.0)
            "operation": "add",
        },
    )

    # --- Reset events (applied at episode reset) ---

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "force_range": (-8.0, 8.0),    # was (-3.0, 3.0)
            "torque_range": (-3.0, 3.0),   # was (-1.0, 1.0)
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
            "velocity_range": (-3.0, 3.0),    # was (-2.5, 2.5)
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # --- Interval events (applied periodically during episodes) ---

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 12.0),  # was (10.0, 15.0) — more frequent
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)},  # was ±0.5
        },
    )


# =============================================================================
# Rewards — Modified weights + 4 new terms
# =============================================================================

@configclass
class Spot100hrRewardsCfg:
    """Reward terms for 100hr multi-terrain training.

    Changes from 48hr config:
      - foot_slip: -1.0 → -3.0 (CRITICAL for low-friction handling)
      - air_time: 5.0 → 3.0 (reduce bouncy gait on ice)
      - foot_clearance: 2.5 → 3.5, target 0.12 → 0.10 (better compromise)
      - base_motion: -3.0 → -4.0 (reduce bouncing on slippery surfaces)
      - joint_vel: -0.02 → -0.05 (slower movements on tricky terrain)
      - 4 new terms: velocity_modulation, body_height, contact_force, stumble
    """

    # -- Task rewards (positive) --

    air_time = RewardTermCfg(
        func=spot_mdp.air_time_reward,
        weight=3.0,    # was 5.0 — reduced to prevent bouncy gait on low-friction
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
        weight=3.5,    # was 2.5 — increased to encourage obstacle clearing
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.10,  # was 0.12 — compromise between stairs and flat
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

    # -- NEW: Velocity modulation (terrain-adaptive speed) --
    velocity_modulation = RewardTermCfg(
        func=velocity_modulation_reward,
        weight=2.0,
        params={
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )

    # -- NEW: Vegetation drag (grass/fluid/mud simulation) --
    # Terrain-aware: no drag on friction_plane, always drag on vegetation_plane,
    # randomized drag on all other terrains.
    # Drag range [0, 20.0] covers ALL 5 eval grass zones.
    vegetation_drag = RewardTermCfg(
        func=VegetationDragReward,
        weight=-0.001,  # Small penalty -- the physics effect is the main purpose
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "drag_max": 20.0,                              # Covers eval Zone 5 (dense brush)
            "contact_threshold": 1.0,                       # Min contact force to apply drag (N)
            "vegetation_terrain_name": "vegetation_plane",  # Always drag > 0 here
            "friction_terrain_name": "friction_plane",      # Always drag = 0 here
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
        weight=-4.0,  # was -3.0 — reduce bouncing on slippery surfaces
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    base_orientation = RewardTermCfg(
        func=spot_mdp.base_orientation_penalty,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    foot_slip = RewardTermCfg(
        func=spot_mdp.foot_slip_penalty,
        weight=-3.0,  # was -1.0 — TRIPLED, critical for low-friction surfaces
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
        weight=-5.0e-2,  # was -2e-2 — increased to slow down on tricky terrain
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
    )

    # -- NEW: Body height tracking --
    body_height_tracking = RewardTermCfg(
        func=body_height_tracking_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": 0.42,
        },
    )

    # -- NEW: Contact force smoothness --
    contact_force_smoothness = RewardTermCfg(
        func=contact_force_smoothness_penalty,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )

    # -- NEW: Stumble penalty --
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
# Terminations — same as SpotRoughEnvCfg
# =============================================================================

@configclass
class Spot100hrTerminationsCfg:
    """Termination terms — same as 48hr config."""

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
class Spot100hrCurriculumCfg:
    """Curriculum for progressive terrain difficulty."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


# =============================================================================
# Main Environment Config
# =============================================================================

@configclass
class Spot100hrEnvCfg(LocomotionVelocityRoughEnvCfg):
    """100-Hour Multi-Terrain Robust Locomotion Environment.

    Inherits from LocomotionVelocityRoughEnvCfg and applies:
    - ROBUST_TERRAINS_CFG with 12 terrain types (400 patches)
    - Massively expanded domain randomization
    - Modified + new reward terms
    - 30s episodes (was 20s)
    - Same 235-dim observation space (48 proprioceptive + 187 height scan)

    Target: H100 NVL 96GB with 65,536 parallel environments.
    """

    # Override MDP components
    observations: Spot100hrObservationsCfg = Spot100hrObservationsCfg()
    actions: Spot100hrActionsCfg = Spot100hrActionsCfg()
    commands: Spot100hrCommandsCfg = Spot100hrCommandsCfg()
    rewards: Spot100hrRewardsCfg = Spot100hrRewardsCfg()
    terminations: Spot100hrTerminationsCfg = Spot100hrTerminationsCfg()
    events: Spot100hrEventCfg = Spot100hrEventCfg()
    curriculum: Spot100hrCurriculumCfg = Spot100hrCurriculumCfg()

    # Viewer
    viewer = ViewerCfg(eye=(10.5, 10.5, 3.0), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        # Post init of parent (sets up rough terrain scene, sensors, etc.)
        super().__post_init__()

        # Physics — 500 Hz with decimation=10 → 50 Hz control (same as 48hr)
        self.decimation = 10
        self.episode_length_s = 30.0  # was 20.0 — 50% longer for more terrain exposure
        self.sim.dt = 0.002  # 500 Hz physics
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"

        # GPU PhysX buffers — 65K envs with 12 complex terrain types need large buffers.
        # 20K envs needed ~475 MB collision stack; 65K needs ~3.2x more. Set to 2**31 (2 GB).
        self.sim.physx.gpu_collision_stack_size = 2**31
        self.sim.physx.gpu_max_rigid_contact_count = 2**24
        self.sim.physx.gpu_max_rigid_patch_count = 2**24

        # Update sensor periods
        self.scene.contact_forces.update_period = self.sim.dt

        # Spot robot
        self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Height scanner — same as 48hr (Spot body, yaw-aligned, 0.1m resolution)
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


@configclass
class Spot100hrEnvCfg_PLAY(Spot100hrEnvCfg):
    """Reduced-size evaluation variant for visual testing."""

    def __post_init__(self):
        super().__post_init__()

        # Smaller scene for evaluation
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None

        # Reduce terrain patches
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Disable randomization for clean evaluation
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
