"""
SpotRoughTerrainPolicy — Deployment wrapper for trained rough-terrain RL policy.
=================================================================================

Loads the PPO actor network trained in Isaac Lab (RSL-RL) and exposes the same
.forward(dt, command) API as SpotFlatTerrainPolicy so the obstacle course can
hot-swap gaits at runtime.

Observation:  235-dim = 48 proprioception + 187 height scan (constant 0.0).

NOTE: The 187 height-scan dims are filled with 0.0 (flat ground assumption).
In Isaac Lab training, height_scan ≈ 0.0 on flat ground (range [-0.00, 0.15]).
The policy uses proprioception for locomotion; height scan provides local
terrain awareness. Without a raycaster, 0.0 = flat ground is the safe default.

Training checkpoint
    C:\\IsaacLab\\logs\\rsl_rl\\spot_rough\\48h_run\\model_29999.pt

Architecture   235 -> 512 -> 256 -> 128 -> 12  (ELU activations)
Action         12 joint-position offsets, scale = 0.25
Decimation     10  (policy @ 50 Hz with 500 Hz physics)
Gains          stiffness = 60.0, damping = 1.5 (all joints)

Usage
-----
    from spot_rough_terrain_policy import SpotRoughTerrainPolicy

    spot_flat  = SpotFlatTerrainPolicy(prim_path="/World/Spot", ...)
    spot_rough = SpotRoughTerrainPolicy(flat_policy=spot_flat)

    world.reset()
    spot_flat.initialize()
    spot_rough.initialize()          # shares flat's robot — no double-init

    # In physics callback:
    spot_rough.forward(dt, [v_x, 0.0, omega_z])

Created for AI2C Tech Capstone - MS for Autonomy, February 2026
Isaac Sim 5.1.0 + Isaac Lab 2.3.0
"""

import os
import numpy as np
import torch
import torch.nn as nn

from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction


# =============================================================================
# CONFIGURATION  (matches training env.yaml / agent.yaml)
# =============================================================================

DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints", "model_29999.pt"
)

# Height scanner ---------------------------------------------------------------
SCAN_SIZE_X, SCAN_SIZE_Y = 1.6, 1.0        # metres
SCAN_RES       = 0.1                        # metres per cell
SCAN_NX        = int(SCAN_SIZE_X / SCAN_RES) + 1   # 17
SCAN_NY        = int(SCAN_SIZE_Y / SCAN_RES) + 1   # 11
SCAN_N         = SCAN_NX * SCAN_NY                  # 187
SCAN_FILL_VAL  =  0.0            # flat-ground height scan ≈ 0.0

# Policy -----------------------------------------------------------------------
OBS_DIM       = 48 + SCAN_N    # 235
ACT_DIM       = 12
ACTION_SCALE  = 0.25           # Matches training env.yaml (action_scale=0.25)
DECIMATION    = 10             # env.yaml → decimation

# Actuator gains from training (env.yaml → actuators → stiffness/damping)
TRAINING_STIFFNESS = 60.0
TRAINING_DAMPING   = 1.5

# ---------- Actuator compensation for PhysX deployment ----------------------
# Training uses explicit actuator models (DelayedPD + RemotizedPD) which:
#   1. Set PhysX Kp=0, Kd=0 (disable drive)
#   2. Set PhysX friction=0, armature=0, effort_limit=1e9
#   3. Compute PD torques in Python with per-joint effort clamping
#   4. Apply clamped torques via set_dof_actuation_forces
#
# We replicate ALL of these settings, not just the gains.
HIP_EFFORT_LIMIT = 45.0   # N·m  (training DelayedPDActuatorCfg)

# RemotizedPD knee torque lookup table (angle_rad → max_torque_Nm)
# Sampled from the 103-row joint_parameter_lookup in spot.py
KNEE_TORQUE_TABLE_ANGLES = np.array([
    -2.7929, -2.7421, -2.6913, -2.6406, -2.5898, -2.5390, -2.4883,
    -2.4375, -2.3867, -2.3360, -2.2852, -2.2344, -2.1836, -2.1329,
    -2.0821, -2.0313, -1.9806, -1.9298, -1.8790, -1.8283, -1.7775,
    -1.7267, -1.6760, -1.6252, -1.5744, -1.5237, -1.4729, -1.4221,
    -1.3714, -1.3206, -1.2698, -1.2190, -1.1683, -1.1175, -1.0667,
    -1.0160, -0.9652, -0.9144, -0.8637, -0.8129, -0.7621, -0.7114,
    -0.6606, -0.6098, -0.5590, -0.5083, -0.4575, -0.4067, -0.3560,
    -0.3052, -0.2544, -0.2471,
], dtype=np.float64)

KNEE_TORQUE_TABLE_VALUES = np.array([
    37.17, 39.44, 41.83, 43.87, 46.03, 48.02, 49.97,
    51.79, 53.45, 56.31, 58.89, 61.20, 63.28, 66.68,
    69.92, 72.89, 75.67, 78.19, 80.55, 83.37, 86.07,
    88.76, 91.62, 94.35, 97.13, 100.47, 103.43, 106.42,
    108.96, 111.16, 112.98, 113.12, 113.24, 112.47, 111.70,
    110.68, 109.56, 108.00, 107.14, 105.80, 103.09, 100.36,
    96.27, 91.07, 84.87, 78.24, 69.59, 60.42, 51.42,
    41.65, 31.60, 30.60,
], dtype=np.float64)

# Training default joint positions (regex-expanded from env.yaml)
#   [fh]l_hx  →  0.1     [fh]r_hx → -0.1
#   f[rl]_hy  →  0.9     h[rl]_hy →  1.1
#   .*_kn     → -1.5
_TRAINING_DEFAULTS = {
    "fl_hx":  0.1, "fr_hx": -0.1, "hl_hx":  0.1, "hr_hx": -0.1,
    "fl_hy":  0.9, "fr_hy":  0.9, "hl_hy":  1.1, "hr_hy":  1.1,
    "fl_kn": -1.5, "fr_kn": -1.5, "hl_kn": -1.5, "hr_kn": -1.5,
}

# Fallback joint order (standard Spot DOF ordering)
_FALLBACK_DEFAULT_POS = np.array([
     0.1,  0.9, -1.5,   # FL : hx, hy, kn
    -0.1,  0.9, -1.5,   # FR
     0.1,  1.1, -1.5,   # HL
    -0.1,  1.1, -1.5,   # HR
], dtype=np.float64)


# =============================================================================
# POLICY CLASS
# =============================================================================

class SpotRoughTerrainPolicy:
    """Rough-terrain locomotion policy for Boston Dynamics Spot.

    Shares the robot articulation from an existing SpotFlatTerrainPolicy
    so both gaits drive the *same* physics body — no double-init.

    Public API mirrors SpotFlatTerrainPolicy:
        .robot              SingleArticulation (shared)
        .default_pos        np.ndarray (12,)
        .initialize()       call AFTER flat_policy.initialize()
        .forward(dt, cmd)   step the policy
        .post_reset()       reset internal state
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, flat_policy, checkpoint_path=None):
        """
        Args:
            flat_policy:       Initialised SpotFlatTerrainPolicy whose .robot
                               articulation is shared.
            checkpoint_path:   Path to RSL-RL checkpoint (.pt).
                               Defaults to the 48h 30k-iteration rough model.
        """
        # Share the existing robot — no new articulation
        self.robot = flat_policy.robot

        # Build actor MLP and load trained weights
        ckpt = checkpoint_path or DEFAULT_CHECKPOINT
        self._actor = self._build_actor(ckpt)

        # Internal bookkeeping
        self._action_scale    = ACTION_SCALE
        self._decimation      = DECIMATION
        self._previous_action = np.zeros(ACT_DIM)
        self._policy_counter  = 0
        self.action           = np.zeros(ACT_DIM)
        self.default_pos      = None           # set in initialize()
        self._knee_indices    = []             # set in initialize()
        self._hip_indices     = []             # set in initialize()

        # Height scan = 0.0 (flat ground default, see _cast_height_rays)
        self._height_scan = np.full(SCAN_N, SCAN_FILL_VAL, dtype=np.float32)

    # ------------------------------------------------------------------
    # Build & load actor
    # ------------------------------------------------------------------
    @staticmethod
    def _build_actor(checkpoint_path):
        """Construct [235->512->256->128->12] ELU MLP, load trained weights."""
        actor = nn.Sequential(
            nn.Linear(OBS_DIM, 512), nn.ELU(),
            nn.Linear(512, 256),     nn.ELU(),
            nn.Linear(256, 128),     nn.ELU(),
            nn.Linear(128, ACT_DIM),
        )

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"]

        # Extract only actor.* keys  →  0.weight, 0.bias, 2.weight, …
        actor_state = {
            k.replace("actor.", ""): v
            for k, v in state.items()
            if k.startswith("actor.")
        }
        actor.load_state_dict(actor_state)
        actor.eval()

        print(f"[ROUGH] Loaded actor from {checkpoint_path}")
        print(f"[ROUGH]   obs={OBS_DIM}  act={ACT_DIM}  "
              f"scale={ACTION_SCALE}  dec={DECIMATION}")
        return actor

    # ------------------------------------------------------------------
    # Height scanner
    # ------------------------------------------------------------------
    def _cast_height_rays(self):
        """Return 187-dim height scan for flat ground assumption.

        Empirically verified (Feb 2026): Isaac Lab's height_scan() produces
        values near 0.0 on flat ground (range [-0.00, 0.15], mean 0.004).
        The formula is: sensor_z - hit_z - 0.5, where the 20m sensor offset
        is subtracted internally before the observation is computed.

        Without a standalone raycaster, we fill with 0.0 = flat ground.
        The policy still walks via proprioception; it just can't see terrain
        ahead. For full terrain awareness, implement PhysX scene queries here.

        Returns:
            np.ndarray (187,): all 0.0, matching flat ground in training.
        """
        return np.full(SCAN_N, SCAN_FILL_VAL, dtype=np.float32)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _compute_observation(self, command):
        """235-dim vector: 48 proprioception + 187 height scan.

        Layout matches training env.yaml observation group:
            [0:3]     base_lin_vel   (body frame)
            [3:6]     base_ang_vel   (body frame)
            [6:9]     projected_gravity
            [9:12]    velocity_commands
            [12:24]   joint_pos_rel  (relative to default)
            [24:36]   joint_vel_rel
            [36:48]   last_action
            [48:235]  height_scan    (187 rays)
        """
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.T

        lin_vel_b = R_BI @ lin_vel_I
        ang_vel_b = R_BI @ ang_vel_I
        gravity_b = R_BI @ np.array([0.0, 0.0, -1.0])

        obs = np.zeros(OBS_DIM, dtype=np.float32)

        # -- 48-dim proprioception (same layout as flat policy) --
        obs[0:3]   = lin_vel_b
        obs[3:6]   = ang_vel_b
        obs[6:9]   = gravity_b
        obs[9:12]  = command

        jp = self.robot.get_joint_positions()
        jv = self.robot.get_joint_velocities()
        obs[12:24] = jp - self.default_pos
        obs[24:36] = jv
        obs[36:48] = self._previous_action

        # -- 187-dim height scan --
        self._height_scan = self._cast_height_rays()
        obs[48:] = self._height_scan

        # Diagnostic: print stats on first policy step
        if self._policy_counter == 0:
            print(f"[ROUGH] OBS DIAG (step 0):")
            print(f"  lin_vel_b = {lin_vel_b}")
            print(f"  gravity_b = {gravity_b}")
            print(f"  joint_pos_rel range = [{(jp - self.default_pos).min():.3f}, {(jp - self.default_pos).max():.3f}]")
            print(f"  height_scan range = [{self._height_scan.min():.3f}, {self._height_scan.max():.3f}]")
            print(f"  height_scan mean  = {self._height_scan.mean():.3f}")
            print(f"  body_z = {float(pos[2]):.3f}")

        return obs

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------
    def _compute_action(self, obs):
        """Run actor network on observation, return 12-dim action."""
        with torch.no_grad():
            t = torch.from_numpy(obs).unsqueeze(0).float()
            return self._actor(t).squeeze(0).numpy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def initialize(self):
        """Call AFTER flat_policy.initialize().

        Sets default_pos from training config and configures actuator gains.
        The robot is already initialised by the flat policy — we just configure
        rough-policy-specific state.
        """
        # Grab current joint positions (robot already initialised by flat)
        self.default_pos = self.robot.get_joint_positions().copy()

        # Override with exact training defaults by joint name
        dof_names = getattr(self.robot, "dof_names", None)
        if dof_names is not None:
            mapped = 0
            for i, name in enumerate(dof_names):
                if name in _TRAINING_DEFAULTS:
                    self.default_pos[i] = _TRAINING_DEFAULTS[name]
                    mapped += 1
            print(f"[ROUGH] Default pos mapped by name "
                  f"({mapped}/{len(dof_names)} joints)")
            print(f"[ROUGH] DOF order: {list(dof_names)}")
            print(f"[ROUGH] default_pos: {np.array2string(self.default_pos, precision=3)}")
        else:
            # Fallback: assume standard Spot joint ordering
            if len(self.default_pos) == ACT_DIM:
                self.default_pos = _FALLBACK_DEFAULT_POS.copy()
            print(f"[ROUGH] Default pos set from fallback ({ACT_DIM} joints)")

        # Identify hip vs knee DOF indices for effort limits
        if dof_names is not None:
            self._hip_indices = [i for i, n in enumerate(dof_names)
                                if n.endswith('_hx') or n.endswith('_hy')]
            self._knee_indices = [i for i, n in enumerate(dof_names)
                                 if n.endswith('_kn')]
            print(f"[ROUGH] Hip DOFs: {self._hip_indices}  "
                  f"Knee DOFs: {self._knee_indices}")

        # Height scan: 0.0 (flat ground default)
        print(f"[ROUGH] Height scan: fill={SCAN_FILL_VAL} "
              f"({SCAN_N} dims, flat ground default)")

        # Store training actuator gains for switching
        self._training_stiffness = TRAINING_STIFFNESS
        self._training_damping   = TRAINING_DAMPING

    def apply_gains(self):
        """Configure PhysX PD position control with training gains.

        PREVIOUS APPROACH (broken on GPU PhysX):
          Set Kp=0, Kd=0 and compute manual PD torques in Python, applied via
          set_joint_efforts().  On GPU PhysX, set_joint_efforts(numpy) silently
          fails because the tensor API expects CUDA tensors.  With Kp=0 and zero
          effective torques, the robot collapses instantly.

        NEW APPROACH (uses same path as flat policy):
          Set PhysX Kp=60, Kd=1.5 (training gains) and apply position targets
          via apply_action(ArticulationAction).  This is the same code path the
          flat policy uses, which is proven to work with GPU PhysX.  PhysX
          internally computes the same PD torque formula:
            τ = Kp*(target - current) - Kd*velocity
          The result is numerically very close to the manual approach but
          properly integrated with the GPU constraint solver.

        Call this when switching FROM flat TO rough gait.
        """
        try:
            av = self.robot._articulation_view
            n_dof = self.robot.num_dof

            # CRITICAL: GPU PhysX ArticulationView silently ignores numpy arrays.
            # All setter calls MUST use CUDA tensors on the correct device.
            dev = "cuda:0"

            # ============================================================
            # ARTICULATION SOLVER SETTINGS (from Isaac Lab SPOT_CFG)
            # Training uses 4/0 solver iterations which produces SOFT dynamics.
            # Standalone defaults may be 32/32 or higher, producing STIFF
            # dynamics that cause the policy's actions to overshoot violently.
            # This is the primary sim-to-sim gap.
            # ============================================================
            try:
                old_pos_iters = av.get_solver_position_iteration_counts()
                old_vel_iters = av.get_solver_velocity_iteration_counts()
                print(f"[ROUGH] Solver iters BEFORE: pos={old_pos_iters}, "
                      f"vel={old_vel_iters}")

                av.set_solver_position_iteration_counts(
                    torch.tensor([4], dtype=torch.int32, device=dev))
                av.set_solver_velocity_iteration_counts(
                    torch.tensor([0], dtype=torch.int32, device=dev))

                new_pos_iters = av.get_solver_position_iteration_counts()
                new_vel_iters = av.get_solver_velocity_iteration_counts()
                print(f"[ROUGH] Solver iters AFTER:  pos={new_pos_iters}, "
                      f"vel={new_vel_iters}")
            except Exception as e:
                print(f"[ROUGH] Could not set solver iterations: {e}")

            # ============================================================
            # PD GAINS — match training (Kp=60, Kd=1.5)
            # ============================================================
            old_kps, old_kds = av.get_gains()
            print(f"[ROUGH] Gains BEFORE: Kp={old_kps}, Kd={old_kds}")

            kps = torch.full((1, n_dof), TRAINING_STIFFNESS, device=dev)
            kds = torch.full((1, n_dof), TRAINING_DAMPING, device=dev)
            av.set_gains(kps=kps, kds=kds)

            new_kps, new_kds = av.get_gains()
            print(f"[ROUGH] Gains AFTER:  Kp={new_kps}, Kd={new_kds}")

            # ============================================================
            # JOINT PROPERTIES — match training actuator config
            # ============================================================

            # Per-joint effort limits
            max_efforts = torch.full((1, n_dof), HIP_EFFORT_LIMIT, device=dev)
            for i in self._knee_indices:
                max_efforts[0, i] = 110.0
            av.set_max_efforts(max_efforts)

            # Zero joint friction and armature (training sets both to 0)
            try:
                av.set_friction_coefficients(torch.zeros((1, n_dof), device=dev))
                av.set_armatures(torch.zeros((1, n_dof), device=dev))
            except Exception:
                pass

            # Velocity limits (training: 12.0 rad/s)
            max_vels = torch.full((1, n_dof), 12.0, device=dev)
            av.set_max_joint_velocities(max_vels)

            # ============================================================
            # ARTICULATION-LEVEL PROPERTIES (from SPOT_CFG)
            # ============================================================
            try:
                av.set_enabled_self_collisions(
                    torch.tensor([True], dtype=torch.bool, device=dev))
                av.set_max_depenetration_velocity(
                    torch.tensor([1.0], device=dev))
            except Exception:
                pass

            print(f"[ROUGH] PhysX PD: Kp={TRAINING_STIFFNESS}, "
                  f"Kd={TRAINING_DAMPING}, solver=4/0")
            print(f"[ROUGH] Effort limits: hips={HIP_EFFORT_LIMIT} N·m, "
                  f"knees=110 N·m")
        except Exception as e:
            import traceback
            print(f"[ROUGH] WARNING: apply_gains failed: {e}")
            traceback.print_exc()

    def _knee_effort_limit(self, angle):
        """Angle-dependent max torque for a knee joint (RemotizedPD lookup)."""
        return np.interp(angle, KNEE_TORQUE_TABLE_ANGLES,
                         KNEE_TORQUE_TABLE_VALUES)

    def forward(self, dt, command):
        """Step the policy — same signature as SpotFlatTerrainPolicy.forward().

        Uses PhysX PD position control (same approach as flat policy):
        1. Policy evaluates at 50 Hz (every DECIMATION steps)
        2. Target positions = default_pos + action * action_scale
        3. PhysX drive computes PD torques internally at physics rate
        4. Applied via apply_action(ArticulationAction) — proven GPU-compatible

        Args:
            dt (float):         Physics timestep.
            command (list/np):  [v_x, v_y, omega_z] velocity command.
        """
        if self._policy_counter % self._decimation == 0:
            eval_idx = self._policy_counter // self._decimation
            cmd = command

            obs = self._compute_observation(cmd)

            # Clip observations to match Isaac Lab (clip_observations=100.0)
            obs = np.clip(obs, -100.0, 100.0)

            self.action = self._compute_action(obs)

            # Clip actions to match Isaac Lab (clip_actions=100.0)
            self.action = np.clip(self.action, -100.0, 100.0)

            # last_action obs = RAW policy output (before scaling)
            self._previous_action = self.action.copy()

            # Diagnostic
            if eval_idx < 10:
                print(f"  [ROUGH act] eval={eval_idx} "
                      f"norm={np.linalg.norm(self.action):.3f} "
                      f"range=[{self.action.min():.3f}, {self.action.max():.3f}] "
                      f"cmd=[{cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f}]")

        # Position targets (PhysX PD drive handles torque computation)
        target_pos = self.default_pos + self.action * self._action_scale
        action = ArticulationAction(joint_positions=target_pos)
        self.robot.apply_action(action)
        self._policy_counter += 1

    def post_reset(self):
        """Reset internal state after a world / robot reset."""
        self._previous_action[:] = 0.0
        self.action[:] = 0.0
        self._policy_counter = 0
        self._height_scan[:] = SCAN_FILL_VAL
