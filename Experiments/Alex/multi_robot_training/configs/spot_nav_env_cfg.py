"""Spot Phase C Navigation Environment Configuration.

Hierarchical architecture: Nav policy (10 Hz) outputs velocity commands,
frozen loco policy (50 Hz) converts to joint positions.

Observations (255-dim):
  - Goal: relative position (2) + distance (1) + orientation sin/cos (2) = 5
  - Robot: velocity (3) = 3
  - LiDAR: 180 rays normalized [0,1] = 180
  - Depth: CNN-encoded features = 64
  - Previous velocity command (3) = 3
  Total: 255

Actions (3-dim): [vx, vy, wz] velocity commands to loco policy.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

import math
import os
import sys

import isaaclab.sim as sim_utils
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from isaaclab_assets.robots.spot import SPOT_CFG  # isort: skip

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from shared.nav_rewards import (
    collision_penalty,
    command_smoothness_penalty,
    goal_progress_reward,
    goal_reached_reward,
    path_efficiency_reward,
    speed_bonus_reward,
)


# =============================================================================
# Nav Policy Observations — 255-dim
# =============================================================================

@configclass
class SpotNavObservationsCfg:
    """Navigation observations: goal + robot state + lidar + depth features."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Nav policy observations."""

        # Goal info (5-dim): relative XY (2) + distance (1) + heading sin/cos (2)
        # These are provided by the nav command manager
        nav_goal = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "nav_goal"},
        )

        # Robot velocity in body frame (3-dim)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        # LiDAR scan (180-dim): front 180 degrees, normalized [0,1]
        lidar_scan = ObsTerm(
            func=mdp.height_scan,  # Reuses ray-cast infrastructure
            params={"sensor_cfg": SceneEntityCfg("lidar_scanner")},
            noise=Unoise(n_min=-0.02, n_max=0.02),
            clip=(0.0, 1.0),
        )

        # Depth camera features (64-dim): CNN-encoded
        # Note: requires separate CNN encoder module during training
        depth_features = ObsTerm(
            func=mdp.height_scan,  # Placeholder — replaced by CNN encoder at runtime
            params={"sensor_cfg": SceneEntityCfg("depth_camera")},
            clip=(0.0, 1.0),
        )

        # Previous velocity command (3-dim)
        prev_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# =============================================================================
# Nav Actions — 3-dim velocity commands
# =============================================================================

@configclass
class SpotNavActionsCfg:
    """3-dim velocity command actions: [vx, vy, wz].

    Output is clipped to safe ranges and passed to frozen loco policy.
    """
    # Action space: continuous velocity commands
    # vx: [-1, 3] m/s, vy: [-1.5, 1.5] m/s, wz: [-2, 2] rad/s
    vel_command = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],  # Placeholder — actual action is velocity, not joint pos
        scale=1.0,
        use_default_offset=False,
    )


# =============================================================================
# Nav Rewards
# =============================================================================

@configclass
class SpotNavRewardsCfg:
    """Navigation reward terms."""

    goal_progress = RewardTermCfg(
        func=goal_progress_reward,
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    goal_reached = RewardTermCfg(
        func=goal_reached_reward,
        weight=100.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.5},
    )

    collision = RewardTermCfg(
        func=collision_penalty,
        weight=-5.0,
        params={"lidar_cfg": SceneEntityCfg("lidar_scanner"), "min_range": 0.3},
    )

    path_efficiency = RewardTermCfg(
        func=path_efficiency_reward,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    cmd_smoothness = RewardTermCfg(
        func=command_smoothness_penalty,
        weight=-1.0,
    )

    speed_bonus = RewardTermCfg(
        func=speed_bonus_reward,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "max_speed": 2.0},
    )


# =============================================================================
# Terminations
# =============================================================================

@configclass
class SpotNavTerminationsCfg:
    """Nav episode terminations."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    body_flip_over = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": math.radians(150.0)},
    )


# =============================================================================
# Main Nav Environment Config
# =============================================================================

@configclass
class SpotNavEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Spot Phase C Navigation Environment.

    Key differences from Phase B loco env:
    - Nav policy runs at 10 Hz (every 5 loco steps)
    - Observations include LiDAR (180 rays) and depth camera (64 features)
    - Actions are velocity commands [vx, vy, wz], not joint positions
    - Frozen loco policy handles low-level control
    - Procedural arenas with random obstacles
    """

    observations: SpotNavObservationsCfg = SpotNavObservationsCfg()
    rewards: SpotNavRewardsCfg = SpotNavRewardsCfg()
    terminations: SpotNavTerminationsCfg = SpotNavTerminationsCfg()

    viewer = ViewerCfg(eye=(20.0, 20.0, 10.0), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        super().__post_init__()

        # Nav policy at 10 Hz (loco at 50 Hz, so decimation=50 relative to sim)
        # Sim at 500 Hz, nav step = every 50 sim steps = 10 Hz
        self.decimation = 50
        self.episode_length_s = 60.0  # Longer episodes for navigation
        self.sim.dt = 0.002
        self.sim.render_interval = 10  # Render at loco rate for camera

        # Fewer envs due to camera rendering overhead
        self.scene.num_envs = 512

        # Spot robot
        self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # LiDAR sensor — 180 rays, front 180 degrees, 10m range
        self.scene.lidar_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body",
            offset=RayCasterCfg.OffsetCfg(pos=(0.3, 0.0, 0.3)),  # Front of body, slightly elevated
            ray_alignment="yaw",
            pattern_cfg=patterns.PolarPatternCfg(
                num_rays=180,
                start_angle=-math.pi / 2,  # -90 degrees
                end_angle=math.pi / 2,     # +90 degrees
            ),
            max_distance=10.0,
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        # Height scanner (for frozen loco policy observations)
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        # Depth camera — 64x64, 90 degree FOV
        self.scene.depth_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body/depth_camera",
            offset=CameraCfg.OffsetCfg(
                pos=(0.35, 0.0, 0.2),  # Front of body
                rot=(1.0, 0.0, 0.0, 0.0),  # Forward-facing
            ),
            width=64,
            height=64,
            data_types=["distance_to_camera"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
            ),
            update_period=0.1,  # 10 Hz — matches nav policy rate
        )

        # Flat arena terrain (obstacles added procedurally at runtime)
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
