"""Expert 2: STAIRS UP MASTER — 80% ascending stairs + 20% flat.

Specializes in climbing stairs from 3cm to 25cm step height.
Kinematic chain unlocked: foot_clearance up, joint_pos loosened, orientation loosened.
This follows the Trial 12b obstacle tuning lesson.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from isaaclab.utils import configclass

from .base_s2r_env_cfg import SpotS2RBaseEnvCfg
from .terrain_cfgs import STAIRS_UP_TERRAINS_CFG


@configclass
class SpotStairsUpExpertEnvCfg(SpotS2RBaseEnvCfg):
    """Stairs-up specialist: ascending stairs of all heights."""

    def __post_init__(self):
        super().__post_init__()

        # Terrain: 40% pyramid_stairs_up + 40% hf_stairs_up + 20% flat
        self.scene.terrain.terrain_generator = STAIRS_UP_TERRAINS_CFG

        # Reward overrides: unlock kinematic chain for step-up motion
        self.rewards.foot_clearance.weight = 2.0     # Base: 0.5 -> lift feet higher
        self.rewards.base_orientation.weight = -2.0   # Base: -3.0 -> allow body tilt
        self.rewards.joint_pos.weight = -0.3          # Base: -0.7 -> allow extreme ROM
