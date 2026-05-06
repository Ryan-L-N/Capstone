"""Base sim-to-real hardened environment config for all 6 experts.

Extends SpotARLHybridEnvCfg with:
  - Observation corruption enabled (Mason had False)
  - Increased observation noise (closer to real sensor profiles)
  - External push forces enabled (Mason had 0.0)
  - Wider domain randomization (mass, friction)
  - Motor power penalty (energy efficiency, Risk R7)
  - Torque limit penalty (motor limits, Risk R6)
  - Increased joint_torques penalty weight

Each expert subclasses this and overrides terrain + reward weights.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

import math
import sys
import os

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

# Import the parent ARL Hybrid config (Loco_Policy_2) and shared utilities
# (Loco_Shared). Both live as siblings of this Loco_Policy_3 directory under
# Experiments/Alex/, so the relative-path computation is robust to either a
# local checkout or an H100 deployment that mirrors the same layout.
_LOCO3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_LOCO_CODEBASES_ROOT = os.path.abspath(os.path.join(_LOCO3_ROOT, ".."))
_PATHS = [
    os.path.join(_LOCO_CODEBASES_ROOT, "Loco_Policy_2_ARL_Hybrid", "configs"),
    os.path.join(_LOCO_CODEBASES_ROOT, "Loco_Shared"),
]
for _p in _PATHS:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from arl_hybrid_env_cfg import (
    SpotARLHybridEnvCfg,
    HybridActionsCfg,
    HybridCommandsCfg,
    HybridCurriculumCfg,
    HybridSceneCfg,
)
from quadruped_locomotion.tasks.locomotion.mdp.rewards import (
    body_height_tracking_penalty,
    clamped_action_smoothness_penalty,
    clamped_joint_acceleration_penalty,
    clamped_joint_torques_penalty,
    clamped_joint_velocity_penalty,
    stumble_penalty,
    terrain_relative_height_penalty,
)

# Import our new S2R reward terms
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rewards.motor_power import motor_power_penalty
from rewards.torque_limit import torque_limit_penalty
from rewards.orientation_split import base_pitch_penalty, base_roll_penalty


# =============================================================================
# Observations — Enable corruption + increased noise
# =============================================================================

@configclass
class S2RObservationsCfg:
    """Mason's observation config with corruption enabled.

    Noise ranges stay at Mason's values. The ProgressiveS2RWrapper adds
    additional noise (dropout, drift, spikes) that scales with terrain level.
    """

    @configclass
    class PolicyCfg(ObsGroup):

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),    # Mason's values
            clip=(-1.0, 1.0),
        )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),    # Mason's values
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),    # Mason's values
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),  # Mason's values
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),  # Mason's values
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.5, n_max=0.5),    # Mason's values
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True   # Enable so Mason noise is applied
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# =============================================================================
# Events — S2R hardened domain randomization
# =============================================================================

@configclass
class S2REventCfg:
    """Mason's safe DR values — S2R hardening comes from ProgressiveS2RWrapper.

    Physics DR stays at Mason's proven values to avoid instant falls on
    low-friction terrain. The ProgressiveS2RWrapper adds sensor noise,
    dropout, delay, and drift that scale with terrain curriculum level.

    Previous attempt with wider DR (friction 0.15, mass ±5, pushes ±3N)
    caused 100% body_contact termination — policy fell immediately.
    """

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),      # Mason's safe values
            "dynamic_friction_range": (0.3, 0.8),      # Mason's safe values
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-2.5, 2.5),   # Mason's safe values
            "operation": "add",
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "force_range": (0.0, 0.0),                 # Mason's (disabled)
            "torque_range": (0.0, 0.0),                # Mason's (disabled)
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-1.5, 1.5), "y": (-1.0, 1.0), "z": (-0.5, 0.5),
                "roll": (-0.7, 0.7), "pitch": (-0.7, 0.7), "yaw": (-1.0, 1.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=spot_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-2.5, 2.5),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),                 # Mason's safe values
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


# =============================================================================
# Terminations — body_contact REMOVED, replaced by soft penalty
# =============================================================================

@configclass
class S2RTerminationsCfg:
    """Soft termination config — no hard body_contact kill.

    Mason's HybridTerminationsCfg has body_contact as a hard termination which
    kills episodes instantly on any body touch. This prevents exploration when
    reward weights change (the actor tries new motions, touches ground once,
    episode ends, no learning signal).

    Our fix: remove body_contact termination entirely. Add undesired_contacts
    as a soft reward penalty (-1.5) so the policy is discouraged from body
    contact but can still learn from the rest of the episode.
    """
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # body_contact REMOVED — replaced by undesired_contacts soft penalty in rewards
    body_flip_over = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": math.radians(150.0)},
    )
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        # Phase-FW-Plus-2-rev2: kept at 3.0. The Phase-FW-Plus-2 attempt to
        # tighten 3.0 -> 1.5 collapsed training in <400 iters (combined with
        # tightened riser range it created an "everywhere is hard" regime
        # the curriculum couldn't escape — body_flip 13% -> 81%, terrain
        # demoted to 0.001). rev2 keeps 3.0 to preserve the proven curriculum
        # behavior; the FW-stair-bypass problem is addressed by curriculum
        # rebalance only.
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


# =============================================================================
# Rewards — Mason's 14 + 2 S2R + 1 undesired_contacts = 17 terms
# =============================================================================

@configclass
class S2RRewardsCfg:
    """Mason's proven weights + motor_power + torque_limit.

    This is the BASE reward config. Expert configs override specific weights.
    """

    # -- Task rewards (Mason's exact values) --

    air_time = RewardTermCfg(
        func=spot_mdp.air_time_reward,
        weight=5.0,
        params={
            "mode_time": 0.3, "velocity_threshold": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    base_angular_velocity = RewardTermCfg(
        func=spot_mdp.base_angular_velocity_reward,
        weight=5.0,
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_linear_velocity = RewardTermCfg(
        func=spot_mdp.base_linear_velocity_reward,
        weight=5.0,
        params={"std": 1.0, "ramp_rate": 0.5, "ramp_at_vel": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    foot_clearance = RewardTermCfg(
        func=spot_mdp.foot_clearance_reward,
        weight=0.5,  # OVERRIDDEN per expert
        params={
            "std": 0.05, "tanh_mult": 2.0, "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    gait = RewardTermCfg(
        func=spot_mdp.GaitReward,
        weight=10.0,  # OVERRIDDEN per expert
        params={
            "std": 0.1, "max_err": 0.2, "velocity_threshold": 0.5,
            "synced_feet_pair_names": (("fl_foot", "hr_foot"), ("fr_foot", "hl_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    # -- Penalties (Mason's exact values, with clamped wrappers) --

    action_smoothness = RewardTermCfg(
        func=clamped_action_smoothness_penalty,
        weight=-1.0,
    )
    air_time_variance = RewardTermCfg(
        func=spot_mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    base_motion = RewardTermCfg(
        func=spot_mdp.base_motion_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_orientation = RewardTermCfg(
        func=spot_mdp.base_orientation_penalty,
        weight=0.0,  # REPLACED by split pitch/roll below
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_pitch = RewardTermCfg(
        func=base_pitch_penalty,
        weight=-0.5,  # Light — allow stair angling. OVERRIDDEN per expert
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_roll = RewardTermCfg(
        func=base_roll_penalty,
        weight=-3.0,  # Heavy — prevent samba/lateral tipping. OVERRIDDEN per expert
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    foot_slip = RewardTermCfg(
        func=spot_mdp.foot_slip_penalty,
        weight=-0.5,  # OVERRIDDEN per expert
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )
    joint_acc = RewardTermCfg(
        func=clamped_joint_acceleration_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
    )
    joint_pos = RewardTermCfg(
        func=spot_mdp.joint_position_penalty,
        weight=-0.7,  # OVERRIDDEN per expert
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0, "velocity_threshold": 0.5,
        },
    )
    joint_torques = RewardTermCfg(
        func=clamped_joint_torques_penalty,
        weight=-1.0e-3,  # Mason: -5e-4 -> S2R: -1e-3 (torque awareness)
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_vel = RewardTermCfg(
        func=clamped_joint_velocity_penalty,
        weight=-1.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
    )

    # -- Proven additions --

    terrain_relative_height = RewardTermCfg(
        func=terrain_relative_height_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "terrain_scaled": False,
            "target_height": 0.37,
        },
    )
    dof_pos_limits = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    # -- Soft body contact penalty (replaces Mason's hard termination) --

    undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*leg"]),
            "threshold": 1.0,
        },
    )

    # -- NEW S2R additions --

    motor_power = RewardTermCfg(
        func=motor_power_penalty,
        weight=-0.005,  # Light: don't compromise terrain performance
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    torque_limit = RewardTermCfg(
        func=torque_limit_penalty,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "hip_limit": 45.0,
            "knee_limit": 100.0,
        },
    )

    # -- Frozen (weight=0.0, present for config compatibility) --

    body_height_tracking = RewardTermCfg(
        func=body_height_tracking_penalty,
        weight=0.0,  # Bug #22: world-frame Z
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.42},
    )
    stumble = RewardTermCfg(
        func=stumble_penalty,
        weight=0.0,  # Bug #28b: world-frame Z
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "knee_height": 0.15, "force_threshold": 5.0,
        },
    )


# =============================================================================
# Main S2R Base Environment Config
# =============================================================================

@configclass
class SpotS2RBaseEnvCfg(SpotARLHybridEnvCfg):
    """S2R-hardened base environment for all 6 terrain experts.

    Extends Mason hybrid with:
      - Observation corruption enabled + wider noise
      - External forces, wider mass/friction DR, more frequent pushes
      - Motor power + torque limit rewards
      - Increased joint_torques weight

    Expert configs subclass this and override terrain + reward weights.
    """

    observations: S2RObservationsCfg = S2RObservationsCfg()
    rewards: S2RRewardsCfg = S2RRewardsCfg()
    events: S2REventCfg = S2REventCfg()

    # Keep Mason's actions, commands, terminations, curriculum, scene
    actions: HybridActionsCfg = HybridActionsCfg()
    commands: HybridCommandsCfg = HybridCommandsCfg()
    terminations: S2RTerminationsCfg = S2RTerminationsCfg()
    curriculum: HybridCurriculumCfg = HybridCurriculumCfg()
    scene: HybridSceneCfg = HybridSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        # Physics stays identical to Mason: 500 Hz sim, 50 Hz control
        # Expert configs may override decimation (e.g., low_freq expert)
