"""Attempt 6: From-Scratch Environment Configuration (v2).

Changes from Attempt 5 scratch_env_cfg.py:
  1. Relaxed termination: body-only (no leg segments) — gives robot more
     timesteps to learn instead of dying on every shin scrape.
  2. Gentler spawn perturbations: reduced initial velocities and angular rates
     so the robot starts from a survivable state.
  3. Simplified rewards: disabled 5 niche terms (vegetation_drag,
     velocity_modulation, body_height_tracking, contact_force_smoothness,
     stumble) to give the critic a cleaner 14-term gradient signal.

Everything else is identical to Attempt 5: same 235-dim obs, same terrain
curriculum (7 types, flat start), same physics, same progressive DR.

Created for AI2C Tech Capstone — hybrid_ST_RL Attempt 6, February 2026
"""

import isaaclab.sim as sim_utils
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.spot import SPOT_CFG  # isort: skip

# Reuse configs from finetune env
import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from configs.finetune_env_cfg import (
    SpotFinetuneObservationsCfg,
    SpotFinetuneActionsCfg,
    SpotFinetuneCommandsCfg,
    SpotFinetuneRewardsCfg,
    SpotFinetuneEventCfg,
    SpotFinetuneCurriculumCfg,
)
from configs.scratch_terrain_cfg import SCRATCH_TERRAINS_CFG
from rewards.reward_terms import (
    VegetationDragReward,
    body_height_tracking_penalty,
    contact_force_smoothness_penalty,
    stumble_penalty,
    velocity_modulation_reward,
)


# =============================================================================
# Terminations — RELAXED: body-only (no leg segments)
# =============================================================================

@configclass
class SpotScratchTerminationsCfg:
    """Relaxed termination — only torso contact kills the episode.

    Attempt 5 terminated on ["body", ".*leg"] which killed episodes when
    any leg segment scraped the terrain. For a from-scratch policy, shin
    contact during early stumbling is inevitable and informative. Leg
    contact is still penalized via rewards but doesn't end the episode.

    This matches Kumar et al. (2023) which only terminates on base collision.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body"]),
            "threshold": 1.0,
        },
    )
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


# =============================================================================
# Events — GENTLER spawn perturbations for from-scratch learning
# =============================================================================

@configclass
class SpotScratchEventCfg(SpotFinetuneEventCfg):
    """Gentler reset conditions for a robot that can't stand yet.

    Attempt 5 spawned robots with ±1.5 m/s velocity and ±0.7 rad/s roll/pitch,
    which is great for robustness testing but terrible for learning from scratch.
    Reduced to modest perturbations so the robot starts from a survivable state.

    Push robot and DR settings are unchanged — progressive DR still ramps up
    over 15K iterations via the training script's update_dr_params().
    """

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),       # Was ±1.5 — too aggressive for random init
                "y": (-0.5, 0.5),       # Was ±1.0
                "z": (-0.3, 0.3),       # Was ±0.5
                "roll": (-0.3, 0.3),    # Was ±0.7 — robot spawned mid-tumble
                "pitch": (-0.3, 0.3),   # Was ±0.7
                "yaw": (-0.5, 0.5),     # Was ±1.0
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=spot_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),    # Was ±0.2 — tighter around default
            "velocity_range": (-1.5, 1.5),    # Was ±2.5 — less initial joint motion
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


# =============================================================================
# Rewards — SIMPLIFIED: disable 5 niche terms, keep 14 core terms
# =============================================================================

@configclass
class SpotScratchRewardsCfg(SpotFinetuneRewardsCfg):
    """14 core reward terms — niche terms zeroed out for cleaner gradients.

    The 5 disabled terms (vegetation_drag, velocity_modulation,
    body_height_tracking, contact_force_smoothness, stumble) are irrelevant
    when the robot can't stand. They add noise to the critic's value estimates
    and create conflicting gradient signals during the critical first 1K iters.

    These can be re-enabled later once the robot learns to walk.
    """

    # Zero out niche terms — keep the RewardTermCfg but set weight=0
    vegetation_drag = RewardTermCfg(
        func=VegetationDragReward,
        weight=0.0,  # Was -0.001 — near-zero noise, disabled
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "drag_max": 20.0,
            "contact_threshold": 1.0,
            "vegetation_terrain_name": "vegetation_plane",
            "friction_terrain_name": "friction_plane",
        },
    )

    velocity_modulation = RewardTermCfg(
        func=velocity_modulation_reward,
        weight=0.0,  # Was 2.0 — requires robot to be moving, useless while falling
        params={
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )

    body_height_tracking = RewardTermCfg(
        func=body_height_tracking_penalty,
        weight=0.0,  # Was -2.0 — irrelevant while robot is falling over
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": 0.42,
        },
    )

    contact_force_smoothness = RewardTermCfg(
        func=contact_force_smoothness_penalty,
        weight=0.0,  # Was -0.5 — gentle foot placement doesn't matter while falling
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )

    stumble = RewardTermCfg(
        func=stumble_penalty,
        weight=0.0,  # Was -2.0 — redundant now that leg termination is removed
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "knee_height": 0.15,
            "force_threshold": 5.0,
        },
    )


# =============================================================================
# Main Environment Config
# =============================================================================

@configclass
class SpotScratchEnvCfgV2(LocomotionVelocityRoughEnvCfg):
    """Attempt 6: From-Scratch Training Environment (v2).

    Same 235-dim observations and terrain curriculum as Attempt 5, but with:
    - Relaxed termination (body-only, not legs)
    - Gentler spawn perturbations
    - Simplified 14-term reward signal
    """

    observations: SpotFinetuneObservationsCfg = SpotFinetuneObservationsCfg()
    actions: SpotFinetuneActionsCfg = SpotFinetuneActionsCfg()
    commands: SpotFinetuneCommandsCfg = SpotFinetuneCommandsCfg()
    rewards: SpotScratchRewardsCfg = SpotScratchRewardsCfg()
    terminations: SpotScratchTerminationsCfg = SpotScratchTerminationsCfg()
    events: SpotScratchEventCfg = SpotScratchEventCfg()
    curriculum: SpotFinetuneCurriculumCfg = SpotFinetuneCurriculumCfg()

    viewer = ViewerCfg(eye=(10.5, 10.5, 3.0), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        super().__post_init__()

        # Physics — 500 Hz with decimation=10 -> 50 Hz control
        self.decimation = 10
        self.episode_length_s = 30.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"

        # GPU PhysX buffers
        self.sim.physx.gpu_collision_stack_size = 2**30
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

        # SCRATCH terrain — 7 types, 300 patches, 10x30 grid
        # max_init_terrain_level=0 ensures ALL robots start on flat/easiest terrain
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=SCRATCH_TERRAINS_CFG,
            max_init_terrain_level=0,  # START FLAT — curriculum promotes as robot improves
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
