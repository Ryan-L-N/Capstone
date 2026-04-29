"""Expert 4: BOULDER MASTER — 80% boxes/obstacles + 20% flat.

Specializes in navigating over and through boulder fields.
Most aggressive kinematic chain unlocking: foot_clearance 2.5 for high step-over.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from isaaclab.utils import configclass

from .base_s2r_env_cfg import SpotS2RBaseEnvCfg
from .terrain_cfgs import BOULDER_TERRAINS_CFG


@configclass
class SpotBouldersExpertEnvCfg(SpotS2RBaseEnvCfg):
    """Boulder specialist: step over/around obstacles up to 30cm."""

    def __post_init__(self):
        super().__post_init__()

        # Terrain: 35% boxes + 25% discrete + 20% repeated + 20% flat
        self.scene.terrain.terrain_generator = BOULDER_TERRAINS_CFG

        # Reward overrides: maximize step-up/over capability
        self.rewards.foot_clearance.weight = 2.5     # Base: 0.5 -> high step-over
        self.rewards.base_orientation.weight = -2.0   # Base: -3.0 -> allow body tilt
        self.rewards.joint_pos.weight = -0.3          # Base: -0.7 -> extreme ROM
