"""Spot navigation environment configuration for Phase C exploration.

Key design decisions:
    - RayCasterCamera (not TiledCamera) for headless H100 compatibility — uses PhysX
      Warp mesh raycasting, no Vulkan needed.
    - 64x64 depth at 90deg FOV with 30m range for long-range pathfinding.
    - Height scanner (187 rays) for terrain-relative height penalty (Bug #22 fix).
    - Exploration mode: no waypoints, maximize forward distance while surviving.
    - Decimation=50 for 10 Hz nav policy (loco runs 5x per nav step inside wrapper).
    - 9 reward terms with Bug #29 clamping convention.
    - 6-level curriculum terrain with 10 sub-terrain types (incl. friction + vegetation).

Observation layout (4108-dim total):
    [0:4096]     — Flattened 64x64 depth image (normalized [0, 1], 30m range)
    [4096:4099]  — Body linear velocity (3)
    [4099:4102]  — Body angular velocity (3)
    [4102:4105]  — Projected gravity (3)
    [4105:4108]  — Previous action / velocity command (3)

Actions (3-dim):
    [0] vx  — forward/back  [-1.0, 3.0] m/s
    [1] vy  — left/right    [-1.5, 1.5] m/s
    [2] wz  — turn          [-2.0, 2.0] rad/s
"""

from __future__ import annotations

import math

from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import RayCasterCfg, RayCasterCameraCfg, ContactSensorCfg
from isaaclab.sensors.ray_caster import patterns

# Isaac Lab asset configs
try:
    from isaaclab_assets.robots.boston_dynamics import SPOT_CFG
except (ImportError, ModuleNotFoundError):
    from isaaclab_assets.robots.spot import SPOT_CFG

# Local terrain, reward, and observation imports
from nav_locomotion.tasks.navigation.mdp.terrains import NAV_TERRAIN_IMPORTER_CFG
import nav_locomotion.tasks.navigation.mdp.rewards as nav_rewards
from nav_locomotion.tasks.navigation.mdp.observations import depth_image_obs, nav_prev_action


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@configclass
class SpotNavSceneCfg(InteractiveSceneCfg):
    """Scene with Spot robot, depth camera (RayCaster), and height scanner."""

    # Terrain
    terrain = NAV_TERRAIN_IMPORTER_CFG

    # Robot
    robot: ArticulationCfg = SPOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.65),
            joint_pos={
                ".*_hx": 0.0,    # hip abduction
                "f.*_hy": 0.9,   # front hip pitch
                "h.*_hy": 1.1,   # hind hip pitch
                ".*_kn": -1.5,   # knee
            },
            joint_vel={".*": 0.0},
        ),
    )

    # Depth camera via RayCasterCamera (PhysX raycasting, no Vulkan needed)
    # Front-facing, mounted on body: +0.3m X (forward), +0.3m Z (above body center)
    # 10-degree downward tilt to see ground ahead
    depth_camera: RayCasterCameraCfg = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
        update_period=0.1,  # 10 Hz (matches nav policy rate)
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.3, 0.0, 0.3),
            rot=(
                math.cos(math.radians(-5)),   # w (half-angle for 10-deg downward tilt)
                0.0,
                math.sin(math.radians(-5)),   # pitch
                0.0,
            ),
        ),
        data_types=["distance_to_image_plane"],
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,     # mm (standard lens)
            horizontal_aperture=20.955,  # ~90 deg FOV at 64px
            height=64,
            width=64,
        ),
        max_distance=30.0,  # 30m range for long-range pathfinding
        drift_range=(0.0, 0.0),  # No drift
        debug_vis=False,
    )

    # Contact force sensor for vegetation drag (foot contact detection)
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=1,
        track_air_time=False,
    )

    # Height scanner for terrain-relative height penalty (same as Phase B)
    # 17x11 grid = 187 rays, 0.1m resolution
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
        update_period=0.0,  # Every sim step
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,
            size=(1.6, 1.0),  # 17 x 11 = 187 rays
        ),
        mesh_prim_paths=["/World/ground"],
        max_distance=100.0,
        drift_range=(0.0, 0.0),
        debug_vis=False,
    )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

@configclass
class NavObservationsCfg:
    """Navigation observations: depth image + proprioception."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Policy observation group — fed to the ActorCriticCNN.

        Layout: [depth_flat (4096) | lin_vel (3) | ang_vel (3) | gravity (3) | prev_action (3)]
        Total: 4108 dimensions
        """
        concatenate_terms = True

        # Depth image (64x64 = 4096 flattened, normalized [0,1])
        depth_image = ObservationTermCfg(
            func=depth_image_obs,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "max_distance": 30.0,
            },
        )

        # Proprioception (12 dims total)
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity)
        last_action = ObservationTermCfg(func=nav_prev_action)

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Actions — 3-dim velocity commands
# ---------------------------------------------------------------------------

@configclass
class NavActionsCfg:
    """Navigation actions: velocity commands [vx, vy, wz].

    These are NOT joint actions — the NavEnvWrapper translates these into
    12-dim joint actions via the frozen loco policy.
    """
    # Using joint_position as placeholder — the wrapper overrides step()
    # to route through the frozen loco policy instead.
    velocity_command = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
    )


# ---------------------------------------------------------------------------
# Events (Domain Randomization)
# ---------------------------------------------------------------------------

@configclass
class NavEventsCfg:
    """Domain randomization events for navigation training."""

    # Physics material randomization
    physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.5),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Add random mass to base body (±5 kg)
    add_base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # Reset robot position at episode start
    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.1, 0.1)},
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
        },
    )

    # Reset joint states
    reset_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-2.5, 2.5),
        },
    )

    # Random pushes during training
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


# ---------------------------------------------------------------------------
# Rewards — 8 terms
# ---------------------------------------------------------------------------

@configclass
class NavRewardsCfg:
    """Navigation reward configuration — 9 terms with Bug #29 clamping."""

    # Positive rewards
    forward_velocity = RewardTermCfg(
        func=nav_rewards.forward_velocity_reward,
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    survival = RewardTermCfg(
        func=nav_rewards.survival_reward,
        weight=1.0,
    )
    terrain_traversal = RewardTermCfg(
        func=nav_rewards.terrain_traversal_reward,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "max_distance": 50.0},
    )

    # Penalties
    terrain_relative_height = RewardTermCfg(
        func=nav_rewards.terrain_relative_height_penalty,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg("robot"),
            "height_easy": 0.42,
            "height_hard": 0.35,
            "terrain_scaled": True,
        },
    )
    drag_penalty = RewardTermCfg(
        func=nav_rewards.drag_penalty,
        weight=-1.5,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg("robot"),
            "height_threshold": 0.25,
        },
    )
    cmd_smoothness = RewardTermCfg(
        func=nav_rewards.cmd_smoothness_penalty,
        weight=-1.0,
    )
    lateral_velocity = RewardTermCfg(
        func=nav_rewards.lateral_velocity_penalty,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    angular_velocity = RewardTermCfg(
        func=nav_rewards.angular_velocity_penalty,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Vegetation drag — applies real drag forces on feet + small penalty
    vegetation_drag = RewardTermCfg(
        func=nav_rewards.VegetationDragReward,
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


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------

@configclass
class NavTerminationsCfg:
    """Episode termination conditions."""

    # 30-second episodes
    time_out = TerminationTermCfg(
        func=mdp.time_out,
        time_out=True,
    )

    # Flip detection (150-degree tilt)
    body_flip = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.1, "asset_cfg": SceneEntityCfg("robot")},
    )

    # Bad orientation (fallen over — gravity Z component < -0.5 means >120 deg tilt)
    bad_orientation = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.5, "asset_cfg": SceneEntityCfg("robot")},
    )


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

@configclass
class NavCurriculumCfg:
    """Terrain difficulty curriculum — placeholder for terrain-based promotion.

    Note: terrain_levels_vel is not available in this Isaac Lab version.
    Curriculum terrain promotion is handled internally by TerrainGeneratorCfg
    when curriculum=True. Robots are promoted/demoted based on episode length.
    """
    pass


# ---------------------------------------------------------------------------
# Main Environment Config
# ---------------------------------------------------------------------------

@configclass
class SpotNavExploreCfg(ManagerBasedRLEnvCfg):
    """Full Spot navigation exploration environment configuration.

    2048 parallel envs, 10 Hz nav policy, 6-level terrain curriculum.
    Designed for H100 headless training with RayCasterCamera depth.
    """

    # Scene
    scene: SpotNavSceneCfg = SpotNavSceneCfg(
        num_envs=2048,
        env_spacing=10.0,
    )

    # Observations (4108 dims)
    observations: NavObservationsCfg = NavObservationsCfg()

    # Actions (3 dims — velocity commands)
    actions: NavActionsCfg = NavActionsCfg()

    # Events (domain randomization)
    events: NavEventsCfg = NavEventsCfg()

    # Rewards (9 terms)
    rewards: NavRewardsCfg = NavRewardsCfg()

    # Terminations
    terminations: NavTerminationsCfg = NavTerminationsCfg()

    # Curriculum
    curriculum: NavCurriculumCfg = NavCurriculumCfg()

    # Simulation parameters
    sim = SimulationCfg(
        dt=0.002,  # 500 Hz physics
        render_interval=50,  # Render at nav rate (10 Hz)
    )

    # Episode length: 30 seconds at 10 Hz = 300 steps
    episode_length_s = 30.0

    # Decimation: 50 physics steps per nav step = 10 Hz nav policy
    # (500 Hz / 50 = 10 Hz)
    decimation = 50


@configclass
class SpotNavExploreCfg_PLAY(SpotNavExploreCfg):
    """Reduced-env variant for visualization and evaluation."""

    scene: SpotNavSceneCfg = SpotNavSceneCfg(
        num_envs=50,
        env_spacing=10.0,
    )
