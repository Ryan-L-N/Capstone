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

# PhysX scene query for raycasting (used when no analytical ground_height_fn)
from omni.physx import get_physx_scene_query_interface


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
SCAN_FILL_VAL  =  0.0            # fallback if raycast misses
SCAN_OFFSET    =  0.5            # Isaac Lab default height offset
SCAN_RAY_HEIGHT = 20.0           # ray origin height above body (matches training)
SCAN_MAX_DIST  = 100.0           # max raycast distance

# Policy -----------------------------------------------------------------------
OBS_DIM       = 48 + SCAN_N    # 235
ACT_DIM       = 12
ACTION_SCALE  = 0.2            # Both standard and Mason configs use scale=0.2
ACTION_SCALE_MASON = 0.2      # Mason baseline/hybrid uses scale=0.2
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

# BUG FIX (Apr 23): Isaac Sim's Spot USD exposes type-grouped DOF ordering
# [hx×4, hy×4, kn×4], NOT leg-grouped. Previous leg-grouped array planted
# knee values into hx slots at episode reset → robot always started in a
# deformed pose. Verified with `spot.robot.get_joint_positions()` after
# stabilization — hx indices (0..3) settled to ~0, hy (4..7) to ~0.4-1.5,
# kn (8..11) to ~-1.5 to -1.8.
_FALLBACK_DEFAULT_POS = np.array([
     0.1, -0.1,  0.1, -0.1,   # hx: fl, fr, hl, hr
     0.9,  0.9,  1.1,  1.1,   # hy: fl, fr, hl, hr
    -1.5, -1.5, -1.5, -1.5,   # kn: fl, fr, hl, hr
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
    def __init__(self, flat_policy, checkpoint_path=None, ground_height_fn=None,
                 arl_baseline=False, action_scale=None, stiffness=None, damping=None,
                 robot_prim_path="/World/Spot", heightscan_ignore_obstacles=True,
                 heightscan_clip=(-0.2, 0.3)):
        """
        Args:
            flat_policy:       Initialised SpotFlatTerrainPolicy whose .robot
                               articulation is shared.
            checkpoint_path:   Path to RSL-RL checkpoint (.pt).
                               Defaults to the 48h 30k-iteration rough model.
            ground_height_fn:  Optional callable(x_pos) -> ground_z.
                               When provided, enables analytical height scanning
                               so the policy can "see" terrain (e.g. stairs).
                               When None, height scan is filled with 0.0 (flat).
            arl_baseline:    If True, use Mason's obs order (height_scan first)
                               and action_scale=0.2.
            action_scale:      Override action scale (e.g., 0.4 for stair v5 policy).
                               If None, uses the default for mason/standard.
        """
        self._arl_baseline = arl_baseline

        # Share the existing robot — no new articulation
        self.robot = flat_policy.robot

        # Build actor MLP and load trained weights
        ckpt = checkpoint_path or DEFAULT_CHECKPOINT
        self._actor = self._build_actor(ckpt)

        # Internal bookkeeping
        if action_scale is not None:
            self._action_scale = action_scale
        else:
            self._action_scale = ACTION_SCALE_MASON if arl_baseline else ACTION_SCALE

        # Store PD gain overrides for initialize()
        self._stiffness_override = stiffness
        self._damping_override = damping
        self._decimation      = DECIMATION
        self._previous_action = np.zeros(ACT_DIM)
        self._policy_counter  = 0
        self.action           = np.zeros(ACT_DIM)
        self.default_pos      = None           # set in initialize()
        self._knee_indices    = []             # set in initialize()
        self._hip_indices     = []             # set in initialize()

        # Ground height function for analytical height scanning
        self._ground_height_fn = ground_height_fn

        # Prim-path prefix used to detect self-hits during PhysX raycast.
        # BUG FIX (Apr 18): was hard-coded to "/World/Robot" but robot is created
        # at "/World/Spot" in cole_arena_skillnav_lite.py. All 187 rays were
        # falling through the self-hit guard and reading body hits as terrain,
        # giving heights in [-1, -0.5] while training saw [0.00, 0.15]. Policy
        # then tried to "climb out of a pit" → 10× command amplification.
        self._robot_prim_path = robot_prim_path

        # Heightscan OOD fix (Apr 20): at training time the raycaster saw only
        # terrain heightfield (no added obstacle boxes). At deploy time the
        # above-ray hits obstacle tops → height = base_z - 0.5 - 0.5 ≈ -0.48,
        # far below training range [0, 0.15] → V6 crouches and falls.
        # - ignore_obstacles=True: primary ray starts at base_z-0.45 (below
        #   obstacle tops, above ground) so it passes under boxes and hits
        #   ground; above-ray becomes fallback when below-ray misses (slopes).
        # - clip: tighter than [-1,1] to guarantee in-distribution values.
        self._heightscan_ignore_obstacles = heightscan_ignore_obstacles
        self._heightscan_clip = heightscan_clip

        # Pre-compute scan grid offsets (body-frame, rotated by yaw at runtime)
        # Training config: GridPatternCfg(resolution=0.1, size=[1.6, 1.0])
        # Grid is 17 (X) × 11 (Y) = 187 points centered on body
        xs = np.linspace(-SCAN_SIZE_X / 2, SCAN_SIZE_X / 2, SCAN_NX)
        ys = np.linspace(-SCAN_SIZE_Y / 2, SCAN_SIZE_Y / 2, SCAN_NY)
        gx, gy = np.meshgrid(xs, ys, indexing='ij')  # (17, 11)
        self._scan_offsets_x = gx.ravel()  # (187,)
        self._scan_offsets_y = gy.ravel()  # (187,)

        # Height scan buffer
        self._height_scan = np.full(SCAN_N, SCAN_FILL_VAL, dtype=np.float32)

    # ------------------------------------------------------------------
    # Build & load actor
    # ------------------------------------------------------------------
    @staticmethod
    def _build_actor(checkpoint_path):
        """Auto-detect hidden sizes from checkpoint and build actor MLP."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"]

        # Extract only actor.* keys  →  0.weight, 0.bias, 2.weight, …
        actor_state = {
            k.replace("actor.", ""): v
            for k, v in state.items()
            if k.startswith("actor.")
        }

        # Auto-detect architecture from weight shapes
        # Keys are like 0.weight, 2.weight, 4.weight, 6.weight
        weight_keys = sorted(k for k in actor_state if k.endswith(".weight"))
        layers = []
        for wk in weight_keys:
            out_dim, in_dim = actor_state[wk].shape
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != ACT_DIM:  # don't add activation after final layer
                layers.append(nn.ELU())
        actor = nn.Sequential(*layers)

        hidden_sizes = [actor_state[wk].shape[0] for wk in weight_keys[:-1]]
        print(f"[ROUGH] Auto-detected architecture: {OBS_DIM} -> {hidden_sizes} -> {ACT_DIM}")

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
        """Cast 187 rays downward in a 17x11 grid, matching Isaac Lab training.

        Pipeline (matches Isaac Lab's RayCaster + height_scan observation):
        1. Grid offsets in body frame: 1.6m x 1.0m, 0.1m resolution
        2. Rotate grid by body YAW only (pitch/roll ignored, matching training)
        3. Cast rays using PhysX scene queries OR analytical ground_height_fn
        4. Compute: height = body_z - hit_z - 0.5
        5. Clip to [-1.0, 1.0]

        When ground_height_fn is provided (e.g. stairs), uses analytical queries.
        When None, uses PhysX raycasting to detect actual terrain geometry
        (boulders, obstacles, etc.).

        Returns:
            np.ndarray (187,): height scan values, clipped to [-1, 1].
        """
        # Get robot world pose
        pos, quat = self.robot.get_world_pose()
        base_x = float(pos[0])
        base_y = float(pos[1])
        base_z = float(pos[2])

        # Extract yaw for rotating the scan grid (ray_alignment="yaw")
        w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny, cosy)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # Rotate scan grid offsets by yaw and add robot position
        world_x = base_x + cos_yaw * self._scan_offsets_x - sin_yaw * self._scan_offsets_y
        world_y = base_y + sin_yaw * self._scan_offsets_x + cos_yaw * self._scan_offsets_y

        # --- Analytical mode (stairs — fast, exact) ---
        if self._ground_height_fn is not None:
            scan = np.empty(SCAN_N, dtype=np.float32)
            for i in range(SCAN_N):
                ground_z = self._ground_height_fn(float(world_x[i]))
                scan[i] = base_z - ground_z - SCAN_OFFSET
            np.clip(scan, -1.0, 1.0, out=scan)
            return scan

        # --- PhysX raycast mode (boulders, any terrain) ---
        scene_query = get_physx_scene_query_interface()
        heights = np.full(SCAN_N, SCAN_FILL_VAL, dtype=np.float32)
        # Start BELOW the robot body so the first hit is never the robot itself.
        # Robot belly sits at ~0.55 m; lower knee paws at ~0.0-0.1 m. Starting at
        # (base_z - 0.45) is below the body collision mesh but above the ground,
        # so the first downward hit is terrain (or an obstacle top if one is under
        # the paw). Prim-path self-hit check kept as belt-and-braces.
        ray_origin_z_above = base_z + SCAN_RAY_HEIGHT
        ray_origin_z_below = base_z - 0.45

        robot_prefix = self._robot_prim_path

        def _cast(x, y, z):
            return scene_query.raycast_closest(
                (float(x), float(y), float(z)), (0.0, 0.0, -1.0), SCAN_MAX_DIST)

        def _is_self(hit_dict):
            return robot_prefix in hit_dict.get("rigidBody", "")

        if self._heightscan_ignore_obstacles:
            # Below-first: ray starts below obstacle tops, sees ground only.
            # Matches Isaac Lab training where RayCaster observed only the
            # terrain heightfield (no discrete obstacle meshes in the scan).
            for i in range(SCAN_N):
                hit = _cast(world_x[i], world_y[i], ray_origin_z_below)
                if hit["hit"] and not _is_self(hit):
                    heights[i] = base_z - hit["position"][2] - SCAN_OFFSET
                    continue
                # Fallback above-ray for cases where below-ray misses (sloped
                # spawn, robot elevated above scan point, etc.)
                hit2 = _cast(world_x[i], world_y[i], ray_origin_z_above)
                if hit2["hit"] and not _is_self(hit2):
                    heights[i] = base_z - hit2["position"][2] - SCAN_OFFSET
        else:
            # Legacy above-first behavior (pre-Apr 20): picks up obstacle tops.
            for i in range(SCAN_N):
                hit = _cast(world_x[i], world_y[i], ray_origin_z_above)
                if hit["hit"] and not _is_self(hit):
                    heights[i] = base_z - hit["position"][2] - SCAN_OFFSET
                    continue
                hit2 = _cast(world_x[i], world_y[i], ray_origin_z_below)
                if hit2["hit"] and not _is_self(hit2):
                    heights[i] = base_z - hit2["position"][2] - SCAN_OFFSET

        np.clip(heights, self._heightscan_clip[0], self._heightscan_clip[1], out=heights)

        return heights

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

        jp = self.robot.get_joint_positions()
        jv = self.robot.get_joint_velocities()

        # -- 187-dim height scan --
        self._height_scan = self._cast_height_rays()

        obs = np.zeros(OBS_DIM, dtype=np.float32)

        if self._arl_baseline:
            # Mason: height_scan(187) first, then proprioception(48)
            obs[0:SCAN_N]          = self._height_scan
            o = SCAN_N  # 187
            obs[o:o+3]            = lin_vel_b
            obs[o+3:o+6]         = ang_vel_b
            obs[o+6:o+9]         = gravity_b
            obs[o+9:o+12]        = command
            obs[o+12:o+24]       = jp - self.default_pos
            obs[o+24:o+36]       = jv
            obs[o+36:o+48]       = self._previous_action
        else:
            # Our training: proprioception(48) first, height_scan(187) last
            obs[0:3]   = lin_vel_b
            obs[3:6]   = ang_vel_b
            obs[6:9]   = gravity_b
            obs[9:12]  = command
            obs[12:24] = jp - self.default_pos
            obs[24:36] = jv
            obs[36:48] = self._previous_action
            obs[48:]   = self._height_scan

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

        # Height scan mode
        if self._ground_height_fn is not None:
            print(f"[ROUGH] Height scan: ANALYTICAL ({SCAN_N} dims, "
                  f"terrain-aware via ground_height_fn)")
        else:
            print(f"[ROUGH] Height scan: PHYSX RAYCAST ({SCAN_N} dims, "
                  f"real terrain geometry via scene queries)")

        # Store training actuator gains for switching (overridable for stair policies)
        self._training_stiffness = self._stiffness_override if self._stiffness_override is not None else TRAINING_STIFFNESS
        self._training_damping   = self._damping_override if self._damping_override is not None else TRAINING_DAMPING

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

                # Match training SPOT_CFG: solver_position_iteration_count=4,
                # solver_velocity_iteration_count=0. Was 8/2 which is stiffer;
                # that produced larger contact impulses per step and a ~5-10×
                # command amplification on Cole rich cubes.
                av.set_solver_position_iteration_counts(
                    torch.tensor([4], dtype=torch.int32, device="cpu"))
                av.set_solver_velocity_iteration_counts(
                    torch.tensor([0], dtype=torch.int32, device="cpu"))

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

            kps = torch.full((1, n_dof), TRAINING_STIFFNESS, device="cpu")
            kds = torch.full((1, n_dof), TRAINING_DAMPING, device="cpu")
            av.set_gains(kps=kps, kds=kds)

            new_kps, new_kds = av.get_gains()
            print(f"[ROUGH] Gains AFTER:  Kp={new_kps}, Kd={new_kds}")

            # ============================================================
            # JOINT PROPERTIES — match training actuator config
            # ============================================================

            # Per-joint effort limits
            max_efforts = torch.full((1, n_dof), HIP_EFFORT_LIMIT, device="cpu")
            for i in self._knee_indices:
                max_efforts[0, i] = 110.0
            av.set_max_efforts(max_efforts)

            # Zero joint friction and armature (training sets both to 0)
            try:
                av.set_friction_coefficients(torch.zeros((1, n_dof), device="cpu"))
                av.set_armatures(torch.zeros((1, n_dof), device="cpu"))
            except Exception:
                pass

            # Velocity limits (training: 12.0 rad/s)
            max_vels = torch.full((1, n_dof), 12.0, device="cpu")
            av.set_max_joint_velocities(max_vels)

            # ============================================================
            # ARTICULATION-LEVEL PROPERTIES (from SPOT_CFG)
            # ============================================================
            try:
                av.set_enabled_self_collisions(
                    torch.tensor([True], dtype=torch.bool, device="cpu"))
                av.set_max_depenetration_velocity(
                    torch.tensor([10.0], device="cpu"))
            except Exception:
                pass

            print(f"[ROUGH] PhysX PD: Kp={TRAINING_STIFFNESS}, "
                  f"Kd={TRAINING_DAMPING}, solver=4/0 (matches SPOT_CFG training)")
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

        # Clamp to physical joint limits to prevent leg penetration
        # Spot joint limits (from URDF): hx [-0.785, 0.785], hy [-0.898, 2.295], kn [-2.793, -0.254]
        joint_lower = np.array([-0.785, -0.785, -0.785, -0.785,
                                -0.898, -0.898, -0.898, -0.898,
                                -2.793, -2.793, -2.793, -2.793])
        joint_upper = np.array([ 0.785,  0.785,  0.785,  0.785,
                                 2.295,  2.295,  2.295,  2.295,
                                -0.254, -0.254, -0.254, -0.254])
        target_pos = np.clip(target_pos, joint_lower, joint_upper)

        action = ArticulationAction(joint_positions=target_pos)
        self.robot.apply_action(action)
        self._policy_counter += 1

    def post_reset(self):
        """Reset internal state after a world / robot reset."""
        self._previous_action[:] = 0.0
        self.action[:] = 0.0
        self._policy_counter = 0
        self._height_scan[:] = SCAN_FILL_VAL
