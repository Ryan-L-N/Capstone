"""Expert 7: OBSTACLE PARKOUR — 45% stairs + 45% boulders + 10% flat.

Combined stairs+boulders specialist with aggressive kinematic chain unlocking.
Loosened joints AND motion-quality penalties for parkour-style step-up/over.
Split pitch/roll orientation: allow stair angling, prevent lateral samba.

v3 (2026-03-25): Split base_orientation into pitch (loose) and roll (tight).
Reward weights from live tuning session via control panel.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from isaaclab.utils import configclass

from .base_s2r_env_cfg import SpotS2RBaseEnvCfg
from .terrain_cfgs import OBSTACLE_PARKOUR_TERRAINS_CFG


@configclass
class SpotObstacleParkourExpertEnvCfg(SpotS2RBaseEnvCfg):
    """Obstacle parkour: stairs + boulders with split pitch/roll orientation."""

    def __post_init__(self):
        super().__post_init__()

        # Terrain: 45% stairs + 45% boulders + 10% flat
        self.scene.terrain.terrain_generator = OBSTACLE_PARKOUR_TERRAINS_CFG

        # === Reward weights from live tuning session ===
        # Task rewards (boosted for aggressive obstacle traversal)
        self.rewards.air_time.weight = 9.0
        self.rewards.base_angular_velocity.weight = 7.0
        self.rewards.base_linear_velocity.weight = 9.0
        self.rewards.foot_clearance.weight = 6.5
        self.rewards.gait.weight = 8.0

        # Split orientation: pitch loose (stairs), roll tight (stability)
        self.rewards.base_orientation.weight = 0.0   # Disabled — replaced by pitch/roll
        self.rewards.base_pitch.weight = -0.5         # Allow stair angling
        self.rewards.base_roll.weight = -3.0          # Prevent samba/lateral tipping

        # Motion quality penalties (tuned for obstacle traversal)
        self.rewards.action_smoothness.weight = -0.675
        self.rewards.air_time_variance.weight = -2.0  # Anti-goose-stepping
        self.rewards.base_motion.weight = -1.5
        self.rewards.joint_pos.weight = -0.2          # Very loose ROM for parkour

        # Power penalties (loosened — obstacles need force)
        self.rewards.motor_power.weight = -0.0005
        self.rewards.joint_torques.weight = -0.00075
        self.rewards.torque_limit.weight = -0.225

        # Boulder stability
        self.rewards.undesired_contacts.weight = -2.5

        # Stronger standing behavior (per Pras + teleop testing)
        self.rewards.joint_pos.params["stand_still_scale"] = 10.0

        # More standing practice for controllability
        self.commands.base_velocity.rel_standing_envs = 0.15
