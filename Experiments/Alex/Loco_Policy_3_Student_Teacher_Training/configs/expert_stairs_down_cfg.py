"""Expert 3: STAIRS DOWN MASTER — 80% descending stairs + 20% flat.

Specializes in descending stairs safely with controlled foot placement.
Same kinematic chain tuning as stairs-up expert.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from isaaclab.utils import configclass

from .base_s2r_env_cfg import SpotS2RBaseEnvCfg
from .terrain_cfgs import STAIRS_DOWN_TERRAINS_CFG


@configclass
class SpotStairsDownExpertEnvCfg(SpotS2RBaseEnvCfg):
    """Stairs-down specialist: controlled descent of all step heights."""

    def __post_init__(self):
        super().__post_init__()

        # Terrain: 80% pyramid_stairs_down + 20% flat
        self.scene.terrain.terrain_generator = STAIRS_DOWN_TERRAINS_CFG

        # Reward overrides: unlock kinematic chain for step-down motion
        self.rewards.foot_clearance.weight = 2.0     # Base: 0.5 -> controlled placement
        self.rewards.base_orientation.weight = -2.0   # Base: -3.0 -> allow body tilt
        self.rewards.joint_pos.weight = -0.3          # Base: -0.7 -> allow extreme ROM
