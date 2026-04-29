"""Attempt 5: From-Scratch Environment Configuration.

Reuses observations, actions, commands, rewards, terminations, and curriculum
from the finetune env config (they're identical — same 235-dim obs, 19 rewards).

Key differences from finetune env:
  - Uses SCRATCH_TERRAINS_CFG (7 types, flat-start curriculum)
    instead of ROBUST_TERRAINS_CFG (12 types, mid-start)
  - max_init_terrain_level=0 — ALL robots start on easiest terrain (flat)
  - No changes to physics, DR, or rewards

Created for AI2C Tech Capstone — hybrid_ST_RL Attempt 5, February 2026
"""

import isaaclab.sim as sim_utils
from isaaclab.envs import ViewerCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.spot import SPOT_CFG  # isort: skip

# Reuse all manager configs from finetune env (obs, actions, commands, rewards, etc.)
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
    SpotFinetuneTerminationsCfg,
    SpotFinetuneEventCfg,
    SpotFinetuneCurriculumCfg,
)
from configs.scratch_terrain_cfg import SCRATCH_TERRAINS_CFG


@configclass
class SpotScratchEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Attempt 5: From-Scratch Training Environment.

    Same 235-dim observations, 19 reward terms, and physics as the finetune env.
    Uses terrain curriculum starting from flat (max_init_terrain_level=0).
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
