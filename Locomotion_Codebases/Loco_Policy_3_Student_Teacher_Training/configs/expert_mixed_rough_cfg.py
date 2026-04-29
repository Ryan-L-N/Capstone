"""Expert 6: MIXED ROUGH MASTER — 80% rough/stepping stones + 20% flat.

Specializes in precise foot placement on irregular terrain.
Tighter gait enforcement for precision, moderate foot clearance.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from isaaclab.utils import configclass

from .base_s2r_env_cfg import SpotS2RBaseEnvCfg
from .terrain_cfgs import MIXED_ROUGH_TERRAINS_CFG


@configclass
class SpotMixedRoughExpertEnvCfg(SpotS2RBaseEnvCfg):
    """Mixed rough specialist: precise stepping on uneven ground."""

    def __post_init__(self):
        super().__post_init__()

        # Terrain: 40% random_rough + 40% stepping_stones + 20% flat
        self.scene.terrain.terrain_generator = MIXED_ROUGH_TERRAINS_CFG

        # Reward overrides: precision focus
        self.rewards.foot_clearance.weight = 1.5     # Base: 0.5 -> moderate clearance
        self.rewards.gait.weight = 12.0              # Base: 10.0 -> tighter gait
        self.rewards.joint_pos.weight = -0.5         # Base: -0.7 -> slight ROM increase
