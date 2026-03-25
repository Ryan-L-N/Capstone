"""Distillation environment config — all terrains at 20 Hz with full S2R hardening.

The distilled student trains on a balanced mix of ALL terrain types at the real
Spot deployment rate (20 Hz, decimation=25). All S2R wrappers are also active
(action delay, sensor noise) — applied in train_distill_s2r.py.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from isaaclab.utils import configclass

from .base_s2r_env_cfg import SpotS2RBaseEnvCfg
from .terrain_cfgs import DISTILLATION_TERRAINS_CFG


@configclass
class SpotS2RDistillEnvCfg(SpotS2RBaseEnvCfg):
    """Distillation env: balanced all-terrain at 20 Hz control rate."""

    def __post_init__(self):
        super().__post_init__()

        # Terrain: balanced mix of all 11 terrain types
        self.scene.terrain.terrain_generator = DISTILLATION_TERRAINS_CFG

        # 20 Hz control rate (500 Hz physics / 25 = 20 Hz)
        # This matches the real Spot SDK command rate
        self.decimation = 25
        self.sim.render_interval = self.decimation

        # Moderate reward adjustments for generalist student
        self.rewards.action_smoothness.weight = -1.5  # Between base (-1.0) and energy (-2.0)
        self.rewards.motor_power.weight = -0.005      # Same as base
