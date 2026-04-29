"""Vision60RoughTerrainPolicy — Deployment wrapper for trained Vision60 RL policy.

Mirror of spot_rough_terrain_policy.py adapted for Vision60:
  - PD gains: Kp=80, Kd=2.0 (vs Spot's 60/1.5)
  - Joint ordering: numeric (joint_0 through joint_11)
  - Effort limit: 87.5 Nm (uniform, from URDF)
  - Default joints: hip=0.9, knee=1.67, abduction=+-0.03

Observation: 235-dim = 48 proprioception + 187 height scan (constant 0.0).

Architecture: 235 -> 1024 -> 512 -> 256 -> 12 (ELU activations)
Action:       12 joint-position offsets, scale = 0.25
Decimation:   10 (policy @ 50 Hz with 500 Hz physics)
Gains:        stiffness = 80.0, damping = 2.0 (all joints)

Created for AI2C Tech Capstone — MS for Autonomy, February 2026
"""

import os
import numpy as np
import torch
import torch.nn as nn

from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "checkpoints", "vision60_best.pt"
)

# Height scanner
SCAN_SIZE_X, SCAN_SIZE_Y = 1.6, 1.0
SCAN_RES = 0.1
SCAN_NX = int(SCAN_SIZE_X / SCAN_RES) + 1   # 17
SCAN_NY = int(SCAN_SIZE_Y / SCAN_RES) + 1   # 11
SCAN_N = SCAN_NX * SCAN_NY                  # 187
SCAN_FILL_VAL = 0.0

# Policy
OBS_DIM = 48 + SCAN_N    # 235
ACT_DIM = 12
ACTION_SCALE = 0.25
DECIMATION = 10

# Vision60-specific actuator parameters
TRAINING_STIFFNESS = 80.0
TRAINING_DAMPING = 2.0
EFFORT_LIMIT = 87.5  # Uniform for all joints (from URDF)

# Vision60 default joint positions
_TRAINING_DEFAULTS = {
    "joint_0": 0.9,    # FL hip
    "joint_2": 0.9,    # RL hip
    "joint_4": 0.9,    # FR hip
    "joint_6": 0.9,    # RR hip
    "joint_1": 1.67,   # FL knee
    "joint_3": 1.67,   # RL knee
    "joint_5": 1.67,   # FR knee
    "joint_7": 1.67,   # RR knee
    "joint_8": 0.03,   # FL abduction
    "joint_9": 0.03,   # RL abduction
    "joint_10": -0.03, # FR abduction
    "joint_11": -0.03, # RR abduction
}

_FALLBACK_DEFAULT_POS = np.array([
    0.9, 1.67,          # FL: hip, knee
    0.9, 1.67,          # RL: hip, knee
    0.9, 1.67,          # FR: hip, knee
    0.9, 1.67,          # RR: hip, knee
    0.03, 0.03, -0.03, -0.03,  # abduction: FL, RL, FR, RR
], dtype=np.float64)


# =============================================================================
# POLICY CLASS
# =============================================================================

class Vision60RoughTerrainPolicy:
    """Rough-terrain locomotion policy for Ghost Robotics Vision60.

    Mirrors SpotRoughTerrainPolicy API but with Vision60-specific parameters.
    """

    def __init__(self, flat_policy, checkpoint_path=None, ground_height_fn=None):
        """
        Args:
            flat_policy:       Initialised Vision60 flat policy whose .robot
                               articulation is shared.
            checkpoint_path:   Path to trained checkpoint (.pt).
            ground_height_fn:  Optional callable(x_pos) -> ground_z for height scan.
        """
        self.robot = flat_policy.robot

        ckpt = checkpoint_path or DEFAULT_CHECKPOINT
        self._actor = self._build_actor(ckpt)

        self._action_scale = ACTION_SCALE
        self._decimation = DECIMATION
        self._previous_action = np.zeros(ACT_DIM)
        self._policy_counter = 0
        self.action = np.zeros(ACT_DIM)
        self.default_pos = None
        self._ground_height_fn = ground_height_fn

        # Pre-compute scan grid
        xs = np.linspace(-SCAN_SIZE_X / 2, SCAN_SIZE_X / 2, SCAN_NX)
        ys = np.linspace(-SCAN_SIZE_Y / 2, SCAN_SIZE_Y / 2, SCAN_NY)
        gx, gy = np.meshgrid(xs, ys, indexing='ij')
        self._scan_offsets_x = gx.ravel()
        self._scan_offsets_y = gy.ravel()
        self._height_scan = np.full(SCAN_N, SCAN_FILL_VAL, dtype=np.float32)

    @staticmethod
    def _build_actor(checkpoint_path):
        """Construct [235->1024->512->256->12] ELU MLP, load trained weights."""
        actor = nn.Sequential(
            nn.Linear(OBS_DIM, 1024), nn.ELU(),
            nn.Linear(1024, 512),     nn.ELU(),
            nn.Linear(512, 256),      nn.ELU(),
            nn.Linear(256, ACT_DIM),
        )

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"]

        actor_state = {
            k.replace("actor.", ""): v
            for k, v in state.items()
            if k.startswith("actor.")
        }
        actor.load_state_dict(actor_state)
        actor.eval()

        print(f"[V60-ROUGH] Loaded actor from {checkpoint_path}")
        print(f"[V60-ROUGH]   obs={OBS_DIM}  act={ACT_DIM}  "
              f"scale={ACTION_SCALE}  dec={DECIMATION}")
        return actor

    def _cast_height_rays(self):
        """Compute 187-dim height scan."""
        if self._ground_height_fn is None:
            return np.full(SCAN_N, SCAN_FILL_VAL, dtype=np.float32)

        pos, quat = self.robot.get_world_pose()
        base_x, base_y, base_z = float(pos[0]), float(pos[1]), float(pos[2])

        w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny, cosy)
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

        world_x = base_x + cos_yaw * self._scan_offsets_x - sin_yaw * self._scan_offsets_y

        scan = np.empty(SCAN_N, dtype=np.float32)
        for i in range(SCAN_N):
            ground_z = self._ground_height_fn(float(world_x[i]))
            scan[i] = base_z - ground_z - 0.5

        np.clip(scan, -1.0, 1.0, out=scan)
        return scan

    def _compute_observation(self, command):
        """235-dim vector: 48 proprioception + 187 height scan."""
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.T

        lin_vel_b = R_BI @ lin_vel_I
        ang_vel_b = R_BI @ ang_vel_I
        gravity_b = R_BI @ np.array([0.0, 0.0, -1.0])

        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0:3] = lin_vel_b
        obs[3:6] = ang_vel_b
        obs[6:9] = gravity_b
        obs[9:12] = command

        jp = self.robot.get_joint_positions()
        jv = self.robot.get_joint_velocities()
        obs[12:24] = jp - self.default_pos
        obs[24:36] = jv
        obs[36:48] = self._previous_action

        self._height_scan = self._cast_height_rays()
        obs[48:] = self._height_scan

        return obs

    def _compute_action(self, obs):
        """Run actor network on observation."""
        with torch.no_grad():
            t = torch.from_numpy(obs).unsqueeze(0).float()
            return self._actor(t).squeeze(0).numpy()

    def initialize(self):
        """Call AFTER flat_policy.initialize()."""
        self.default_pos = self.robot.get_joint_positions().copy()

        dof_names = getattr(self.robot, "dof_names", None)
        if dof_names is not None:
            mapped = 0
            for i, name in enumerate(dof_names):
                if name in _TRAINING_DEFAULTS:
                    self.default_pos[i] = _TRAINING_DEFAULTS[name]
                    mapped += 1
            print(f"[V60-ROUGH] Default pos mapped ({mapped}/{len(dof_names)} joints)")
        else:
            if len(self.default_pos) == ACT_DIM:
                self.default_pos = _FALLBACK_DEFAULT_POS.copy()
            print(f"[V60-ROUGH] Default pos from fallback ({ACT_DIM} joints)")

        if self._ground_height_fn is not None:
            print(f"[V60-ROUGH] Height scan: ANALYTICAL ({SCAN_N} dims)")
        else:
            print(f"[V60-ROUGH] Height scan: fill={SCAN_FILL_VAL} ({SCAN_N} dims)")

    def apply_gains(self):
        """Configure PhysX PD position control with Vision60 training gains."""
        try:
            av = self.robot._articulation_view
            n_dof = self.robot.num_dof
            dev = "cuda:0"

            # Solver iterations (matching training)
            try:
                av.set_solver_position_iteration_counts(
                    torch.tensor([4], dtype=torch.int32, device=dev))
                av.set_solver_velocity_iteration_counts(
                    torch.tensor([4], dtype=torch.int32, device=dev))
            except Exception as e:
                print(f"[V60-ROUGH] Could not set solver iterations: {e}")

            # PD gains — Kp=80, Kd=2.0
            kps = torch.full((1, n_dof), TRAINING_STIFFNESS, device=dev)
            kds = torch.full((1, n_dof), TRAINING_DAMPING, device=dev)
            av.set_gains(kps=kps, kds=kds)

            # Uniform effort limits
            max_efforts = torch.full((1, n_dof), EFFORT_LIMIT, device=dev)
            av.set_max_efforts(max_efforts)

            # Zero joint friction and armature
            try:
                av.set_friction_coefficients(torch.zeros((1, n_dof), device=dev))
                av.set_armatures(torch.zeros((1, n_dof), device=dev))
            except Exception:
                pass

            print(f"[V60-ROUGH] PhysX PD: Kp={TRAINING_STIFFNESS}, Kd={TRAINING_DAMPING}")
            print(f"[V60-ROUGH] Effort limit: {EFFORT_LIMIT} N*m (all joints)")
        except Exception as e:
            import traceback
            print(f"[V60-ROUGH] WARNING: apply_gains failed: {e}")
            traceback.print_exc()

    def forward(self, dt, command):
        """Step the policy."""
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            obs = np.clip(obs, -100.0, 100.0)
            self.action = self._compute_action(obs)
            self.action = np.clip(self.action, -100.0, 100.0)
            self._previous_action = self.action.copy()

        target_pos = self.default_pos + self.action * self._action_scale
        action = ArticulationAction(joint_positions=target_pos)
        self.robot.apply_action(action)
        self._policy_counter += 1

    def post_reset(self):
        """Reset internal state."""
        self._previous_action[:] = 0.0
        self.action[:] = 0.0
        self._policy_counter = 0
        self._height_scan[:] = SCAN_FILL_VAL
