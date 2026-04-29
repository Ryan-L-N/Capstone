"""Parkour-Nav unified environment config.

Inherits SIM_TO_REAL's hardened base and applies parkour-calibrated overrides
cribbed from Cheng 2024 (Extreme Parkour) + legged_gym defaults:

  - Wider DR (friction 0.6-2.0, mass 0-3kg, motor ±20%, push 8s @ 0.5 m/s)
  - 3D command randomization (vx, vy, ωz) with command curriculum
  - Asymmetric observations: policy group (proprio + noisy scan)
                             critic group (+ privileged terrain/dynamics)
  - action_scale bumped to 0.3 (V18 ceiling was 0.2-limited stride)
  - 10-step observation history via S2R wrapper's obs_delay / ring buffer
  - Cole-style obstacle scatter on every terrain patch
  - Rewards: pure velocity tracking + smoothness + survival.
    Explicitly NO altitude_reward, NO directional_progress_reward.

Status: SCAFFOLD — TODO markers flag the pieces still to implement.
"""

import os
import sys

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.envs.mdp as core_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

_SIM2REAL_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "SIM_TO_REAL")
)
if _SIM2REAL_ROOT not in sys.path:
    sys.path.insert(0, _SIM2REAL_ROOT)

_PARKOUR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PARKOUR_ROOT not in sys.path:
    sys.path.insert(0, _PARKOUR_ROOT)

from configs.base_s2r_env_cfg import (
    SpotS2RBaseEnvCfg,
    S2RObservationsCfg,
    S2REventCfg,
    S2RRewardsCfg,
)

from modules import privileged_obs


# =============================================================================
# Observations — asymmetric groups (policy vs critic)
# =============================================================================

@configclass
class ParkourNavObservationsCfg(S2RObservationsCfg):
    """Extends base obs with a privileged critic group.

    policy group = Mason-235 (187-ray scan + proprio + actions) with noise
    critic group = policy group + (true friction, mass, terrain height field,
                                   contact forces, body lin/ang vel without noise)

    Isaac Lab's RslRlVecEnvWrapper routes `policy` -> actor, `critic` -> value.
    """

    # `policy` group inherited from S2RObservationsCfg.PolicyCfg

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations — clean values, never shown to actor."""

        # Clean versions of actor obs (no noise)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        actions = ObsTerm(func=mdp.last_action)

        # Privileged terms — actor never sees these.
        # Clean height_scan above already gives the critic ground-truth
        # terrain elevation (the parkour teacher's most important signal).
        friction_coefficient = ObsTerm(
            func=privileged_obs.friction_coefficient,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        added_mass = ObsTerm(
            func=privileged_obs.added_mass_base,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["body"])},
        )
        foot_contact_forces = ObsTerm(
            func=privileged_obs.foot_contact_forces,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )

        def __post_init__(self):
            self.enable_corruption = False   # clean
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


# =============================================================================
# Events — parkour-calibrated DR (from Cheng 2024 Extreme Parkour defaults)
# =============================================================================

@configclass
class ParkourNavEventCfg(S2REventCfg):
    """Overrides base DR with parkour-paper ranges."""

    def __post_init__(self):
        # Phase-4 (Apr 24 PM): friction floor dropped further to close the
        # zone-5 gap. Phase-3 widened to (0.6, 2.0) static / (0.4, 1.8) dynamic
        # but iter 7400 still FLIP/FELL at friction zone 5 (~46-48m, 1.9m
        # short of COMPLETE). Eval zone 5 friction must be below 0.6 — drop
        # static floor to 0.3, dynamic to 0.2. Matches parkour-paper "ice"
        # regime. Mass / push ranges kept at Phase-3 values.
        self.physics_material.params["static_friction_range"] = (0.3, 2.0)
        self.physics_material.params["dynamic_friction_range"] = (0.2, 1.8)

        self.add_base_mass.params["mass_distribution_params"] = (0.0, 3.0)

        self.push_robot.interval_range_s = (6.0, 10.0)
        # Hail Mary revert: action_scale back to 0.3 so push range returns to
        # Phase-FW-Plus baseline (±0.6 m/s). Phase-Final's ±0.8 was tuned for
        # action_scale=0.5; mismatched here.
        self.push_robot.params["velocity_range"] = {
            "x": (-0.6, 0.6), "y": (-0.6, 0.6),
        }

    # Motor strength randomization — scale Kp/Kd by [0.8, 1.2] at startup.
    # Cheng 2024 Extreme Parkour uses this to model real-motor torque drift.
    # Applied once per env on reset-to-default (mode="reset").
    randomize_motor_gains = EventTerm(
        func=core_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


# =============================================================================
# Rewards — velocity tracking + smoothness + survival.
# Explicitly NO altitude / directional_progress.
# =============================================================================

@configclass
class ParkourNavRewardsCfg(S2RRewardsCfg):
    """Stripped-down reward stack cribbed from legged_gym + extreme_parkour.

    The V19 post-mortem proved that altitude-style rewards destabilize
    flat-ground gait under fine-tune. Pure velocity tracking + terrain
    curriculum is what the parkour papers use.
    """

    def __post_init__(self):
        # Kill anything altitude-shaped. V19 taught us.
        if hasattr(self, "directional_progress"):
            self.directional_progress.weight = 0.0

        # Hail Mary revert: action_scale 0.5 → 0.3, so action_smoothness
        # weight returns to the Phase-9/FW-Plus value (-1.5). Phase-Final's
        # -2.0 was a compensation for the bigger 0.5 action delta; not needed
        # at 0.3.
        self.action_smoothness.weight = -1.5

        # Hail Mary revert: joint_torques weight back to S2R baseline (-1e-3).
        # Phase-Final tightened to -1.5e-3 to compensate for action_scale=0.5
        # + cmd_vx 4.0 — both reverted, so penalty returns to baseline.
        self.joint_torques.weight = -1.0e-3

        # Keep orientation penalty mild — stairs require body pitch.
        self.base_pitch.weight = -0.25
        self.base_roll.weight = -3.0

        # Close the Option 6 reward-hacking exploit. With base S2R weights,
        # gait(10) + air_time(5) + ang_vel(5) + foot_clearance(0.5) = 20.5
        # of *unconditional* positive reward — a policy can earn +200 by
        # standing still with a pretty cadence (observed: vel_xy_err=3,
        # terrain_levels=0.0, reward=+228). Rebalance so forward velocity
        # dominates the positive stack.
        self.base_linear_velocity.weight = 10.0   # was 5.0 — dominate stack
        self.gait.weight                 =  3.0   # was 10 — neuter jiggle bonus
        self.air_time.weight             =  2.0   # was 5  — same


# =============================================================================
# Commands — 3D, parkour-style with curriculum (TODO wire curriculum)
# =============================================================================

# Base Mason HybridCommandsCfg already provides vx/vy/ωz resampling.
# We override ranges; command curriculum to be added via a custom curriculum term.
# TODO: implement ParkourCommandCurriculumTerm that widens ranges as
#       mean terrain_level crosses thresholds:
#         level < 2:  vx [0.2, 0.8],  vy [-0.3, 0.3],  ωz [-0.5, 0.5]
#         level 2-5:  vx [0.3, 1.2],  vy [-0.6, 0.6],  ωz [-1.0, 1.0]
#         level > 5:  vx [0.5, 1.5],  vy [-0.8, 0.8],  ωz [-1.5, 1.5]


# =============================================================================
# Top-level env cfg
# =============================================================================

@configclass
class ParkourNavEnvCfg(SpotS2RBaseEnvCfg):
    """Unified parkour + nav environment.

    Inherits SIM_TO_REAL base (hardened DR, Mason obs, soft termination).
    Overrides: wider DR, asymmetric critic, parkour rewards, obstacle scatter.
    """

    observations: ParkourNavObservationsCfg = ParkourNavObservationsCfg()
    events: ParkourNavEventCfg = ParkourNavEventCfg()
    rewards: ParkourNavRewardsCfg = ParkourNavRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # --- Hail Mary: action_scale = 0.3 (Phase-9/FW-Plus baseline) ---
        # Phase-Final tried 0.5 (Cheng 2024 default for ANYmal); the from-
        # scratch run collapsed under Bug #25 slow bleed. The 22100 ckpt
        # (project speed record + zone-5 stair survival) was trained at 0.3,
        # and 0.3 is also the eval-script default + Cole-compatible setting.
        # Drop-in compatible with the 22100 / Phase-9 spec sheet.
        if hasattr(self.actions, "joint_pos") and hasattr(self.actions.joint_pos, "scale"):
            self.actions.joint_pos.scale = 0.3

        # --- Swap terrain to unified parkour curriculum ---
        from pn_cfg.parkour_nav_terrain_cfg import PARKOUR_NAV_TERRAINS_CFG
        self.scene.terrain.terrain_generator = PARKOUR_NAV_TERRAINS_CFG

        # --- Tighten command ranges to parkour-realistic values ---
        # Mason defaults (-2 to 3 m/s vx, ±1.5 m/s vy, ±2 rad/s ωz) are fine
        # for pure velocity tracking but produce aggressive targets on rough
        # terrain. Cheng 2024 uses vx <= 1.5, vy <= 0.8, ωz <= 1.5.
        # Dynamic curriculum widening by terrain level is deferred — with
        # game-curriculum-driven terrain promotion, fixed tight ranges behave
        # equivalently at Phase 1 scope.
        # Tight command ranges for from-scratch first run. terrain_levels_vel
        # promotes when distance > ~0.5 * (cmd_vel * ep_len); with wide ranges
        # the promote bar is ~15m/ep — unreachable for a baby policy, so the
        # curriculum stalls at level 0 (observed in Option 6). Tight ranges
        # cut the bar to ~6m/ep. Widen in fine-tune resume once curriculum
        # is climbing.
        # Phase-8 (Apr 26): widen ang_vel_z further (1.5→2.0) to cover Cole
        # APF's spawn-time turn commands. Phase-7 Cole eval crashed at t=2.8s
        # because navigator commanded wz=1.91 rad/s on spawn — out of Phase-7
        # training range (1.5 max), and combined with high vx = unstable on
        # flat ground (made worse by harder-terrain gait drift). Widening to
        # 2.0 covers that edge case with margin.
        # Phase-6 backward (-1.0 vx floor) and lateral (0.8 vy) kept.
        # Hail Mary revert: lin_vel_x max 4.0 → 1.5 m/s. Phase-Final's 4.0
        # was the Cheng 2024 ceiling, and combined with action_scale=0.5 +
        # the 50% stair-relevant terrain mix produced the slow-bleed
        # collapse. Phase-9 / FW-Plus's max-1.5 m/s produced the project-
        # record 22100 policy. terrain_levels_vel still self-regulates the
        # promote bar at the operating range.
        if hasattr(self.commands, "base_velocity"):
            self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.5)
            self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
            self.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0)
            self.commands.base_velocity.resampling_time_range = (4.0, 6.0)

        # --- Scene env spacing (num_envs set by train script) ---
        self.scene.env_spacing = 2.5
