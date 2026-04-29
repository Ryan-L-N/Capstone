"""Boulder Master V5 — Custom boulder field matching eval + all innovations.

Training terrain matches eval progression:
- Low difficulty: dense small obstacles (3-15cm, step through/over)
- Mid difficulty: mixed sizes (15-50cm, step over + navigate around)
- High difficulty: sparse large boulders (50-120cm, navigate around)

Uses polyhedra (D8/D12/D20) matching eval's boulder shapes.
"""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from .base_s2r_env_cfg import SpotS2RBaseEnvCfg
from .terrain_cfgs import BOULDER_FIELD_TERRAINS_CFG
from rewards.adaptive_rewards import (
    adaptive_clearance_reward,
    adaptive_velocity_reward,
    adaptive_smoothness_penalty,
    adaptive_height_penalty,
    adaptive_gait_reward,
    adaptive_slip_penalty,
)
from rewards.stair_climbing_rewards import height_gain_reward
from rewards.stair_rewards import (
    flying_gait_penalty,
    dont_wait_penalty,
)


@configclass
class SpotBoulderV5EnvCfg(SpotS2RBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = BOULDER_FIELD_TERRAINS_CFG

        # ── ADAPTIVE REWARDS — MODERATE SURGE ─────────────────────────
        self.rewards.foot_clearance = RewardTermCfg(
            func=adaptive_clearance_reward, weight=5.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
        self.rewards.base_linear_velocity = RewardTermCfg(
            func=adaptive_velocity_reward, weight=10.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.action_smoothness = RewardTermCfg(
            func=adaptive_smoothness_penalty, weight=-0.4,
            params={},
        )
        self.rewards.terrain_relative_height = RewardTermCfg(
            func=adaptive_height_penalty, weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.rewards.gait = RewardTermCfg(
            func=adaptive_gait_reward, weight=10.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.foot_slip = RewardTermCfg(
            func=adaptive_slip_penalty, weight=-0.8,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        )

        # ── KINEMATIC CHAIN — MODERATE ────────────────────────────────
        self.rewards.joint_pos.weight = -0.15
        self.rewards.base_pitch.weight = -0.05
        self.rewards.base_orientation.weight = -0.2   # Looser — allow turning to navigate
        self.rewards.base_motion.weight = -0.8
        self.rewards.air_time.weight = 2.0

        # ── EXPLOIT BLOCKERS ──────────────────────────────────────────
        self.rewards.base_roll.weight = -7.0
        self.rewards.motor_power.weight = -0.0002
        self.rewards.torque_limit.weight = -0.05
        self.rewards.undesired_contacts.weight = -0.5

        # ── 3 BOULDER REWARDS ─────────────────────────────────────────
        self.rewards.flying_gait = RewardTermCfg(
            func=flying_gait_penalty, weight=-3.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        )
        self.rewards.dont_wait = RewardTermCfg(
            func=dont_wait_penalty, weight=-5.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.height_gain = RewardTermCfg(
            func=height_gain_reward, weight=1.5,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # ── MINIMAL STANDING ─────────────────────────────────────────
        self.commands.base_velocity.rel_standing_envs = 0.05
