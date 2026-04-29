"""Expert 5: SLOPES MASTER — 80% slopes/waves + 20% flat.

Specializes in traversing inclines 0-30 degrees up and down.
Increased foot_slip penalty for traction on slopes.
Slightly loosened orientation for natural slope traversal posture.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from isaaclab.utils import configclass

from .base_s2r_env_cfg import SpotS2RBaseEnvCfg
from .terrain_cfgs import SLOPES_TERRAINS_CFG


@configclass
class SpotSlopesExpertEnvCfg(SpotS2RBaseEnvCfg):
    """Slopes specialist: uphill/downhill traversal up to 30 degrees."""

    def __post_init__(self):
        super().__post_init__()

        # Terrain: 35% slope_up + 35% slope_down + 10% wave + 20% flat
        self.scene.terrain.terrain_generator = SLOPES_TERRAINS_CFG

        # Reward overrides: traction + slight tilt tolerance
        self.rewards.foot_slip.weight = -1.5          # Base: -0.5 -> traction focus
        self.rewards.base_orientation.weight = -2.5   # Base: -3.0 -> slight tilt ok
