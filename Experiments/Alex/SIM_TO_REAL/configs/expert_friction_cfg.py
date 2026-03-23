"""Expert 1: FRICTION MASTER — 80% friction planes + 20% flat.

Specializes in walking on surfaces from ice (mu=0.05) to high-grip rubber (mu=1.5).
Increased foot_slip penalty for traction focus.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from isaaclab.utils import configclass

from .base_s2r_env_cfg import SpotS2RBaseEnvCfg
from .terrain_cfgs import FRICTION_TERRAINS_CFG


@configclass
class SpotFrictionExpertEnvCfg(SpotS2RBaseEnvCfg):
    """Friction specialist: traction management on slippery to grippy surfaces."""

    def __post_init__(self):
        super().__post_init__()

        # Terrain: 80% friction planes + 20% flat
        self.scene.terrain.terrain_generator = FRICTION_TERRAINS_CFG

        # Reward overrides: traction focus
        self.rewards.foot_slip.weight = -1.5    # Base: -0.5 -> traction focus
