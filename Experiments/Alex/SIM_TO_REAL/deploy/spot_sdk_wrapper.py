"""Spot SDK wrapper — bridges policy I/O with Boston Dynamics Spot SDK.

Maps the 235-dim policy observation from Spot SDK state feedback and
translates 12-dim policy actions into Spot JointCommand messages.

Runs at 20 Hz to match the distillation training control rate.

Observation layout (235-dim):
  [0:187]   = height_scan (from depth camera via height_scan_builder)
  [187:190] = base_lin_vel (from Spot state estimator)
  [190:193] = base_ang_vel (from Spot IMU)
  [193:196] = projected_gravity (from Spot IMU accelerometer)
  [196:199] = velocity_commands (from joystick/planner)
  [199:211] = joint_pos relative to default (from Spot joint encoders)
  [211:223] = joint_vel (from Spot joint encoders)
  [223:235] = last_action (previous policy output)

Action: 12-dim joint position offsets, scaled by ACTION_SCALE=0.2,
        added to default joint positions, sent as position targets
        with Kp=60, Kd=1.5.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


# Spot default joint positions (same as training)
SPOT_DEFAULT_POSITIONS = {
    "fl_hx": 0.1, "fr_hx": -0.1,
    "fl_hy": 0.9, "fr_hy": 0.9,
    "fl_kn": -1.5, "fr_kn": -1.5,
    "hl_hx": 0.1, "hr_hx": -0.1,
    "hl_hy": 1.1, "hr_hy": 1.1,
    "hl_kn": -1.5, "hr_kn": -1.5,
}

# DOF ordering (type-grouped, matching training)
DOF_ORDER = [
    "fl_hx", "fr_hx", "hl_hx", "hr_hx",  # abduction
    "fl_hy", "fr_hy", "hl_hy", "hr_hy",  # hip flexion
    "fl_kn", "fr_kn", "hl_kn", "hr_kn",  # knee
]

ACTION_SCALE = 0.2
CONTROL_RATE_HZ = 20.0
KP = 60.0
KD = 1.5


@dataclass
class SpotPolicyConfig:
    """Configuration for Spot policy deployment."""
    checkpoint_path: str = ""
    hidden_dims: list = None
    action_scale: float = ACTION_SCALE
    control_rate_hz: float = CONTROL_RATE_HZ
    kp: float = KP
    kd: float = KD
    device: str = "cpu"


class SpotPolicyRunner:
    """Runs trained policy on real Boston Dynamics Spot.

    Usage:
        runner = SpotPolicyRunner(config)
        runner.setup(spot_client)

        while running:
            obs = runner.build_observation(spot_state, velocity_cmd, height_scan)
            action = runner.get_action(obs)
            joint_cmd = runner.build_joint_command(action)
            spot_client.send_command(joint_cmd)
            time.sleep(1.0 / CONTROL_RATE_HZ)
    """

    def __init__(self, config: SpotPolicyConfig):
        self.config = config
        self.hidden_dims = config.hidden_dims or [512, 256, 128]
        self.device = config.device
        self.actor = None
        self.last_action = np.zeros(12, dtype=np.float32)

        # Default positions as array (DOF_ORDER)
        self.default_pos = np.array(
            [SPOT_DEFAULT_POSITIONS[name] for name in DOF_ORDER],
            dtype=np.float32,
        )

    def setup(self, checkpoint_path: str = None):
        """Load the trained policy checkpoint.

        Args:
            checkpoint_path: Path to model_XXXX.pt. Uses config path if None.
        """
        path = checkpoint_path or self.config.checkpoint_path

        # Build actor network
        layers = []
        input_dim = 235
        for dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ELU())
            input_dim = dim
        layers.append(nn.Linear(input_dim, 12))
        self.actor = nn.Sequential(*layers)

        # Load weights
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        actor_state = {
            k.replace("actor.", ""): v
            for k, v in state.items()
            if k.startswith("actor.")
        }
        self.actor.load_state_dict(actor_state)
        self.actor.eval()

        print(f"[SPOT] Policy loaded from {path} ({sum(p.numel() for p in self.actor.parameters())} params)")

    def build_observation(
        self,
        base_lin_vel: np.ndarray,       # (3,) m/s
        base_ang_vel: np.ndarray,       # (3,) rad/s
        projected_gravity: np.ndarray,  # (3,) unit vector
        velocity_cmd: np.ndarray,       # (3,) [vx, vy, yaw_rate]
        joint_pos: np.ndarray,          # (12,) radians
        joint_vel: np.ndarray,          # (12,) rad/s
        height_scan: np.ndarray,        # (187,) meters
    ) -> np.ndarray:
        """Build 235-dim policy observation from Spot state.

        Args:
            All arrays should match the training observation layout.

        Returns:
            (235,) observation array.
        """
        # Joint position relative to default
        joint_pos_rel = joint_pos - self.default_pos

        obs = np.concatenate([
            height_scan,           # 0:187
            base_lin_vel,          # 187:190
            base_ang_vel,          # 190:193
            projected_gravity,     # 193:196
            velocity_cmd,          # 196:199
            joint_pos_rel,         # 199:211
            joint_vel,             # 211:223
            self.last_action,      # 223:235
        ]).astype(np.float32)

        assert obs.shape == (235,), f"Obs shape mismatch: {obs.shape}"
        return obs

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Run policy inference.

        Args:
            obs: (235,) observation array.

        Returns:
            (12,) raw action (before scaling).
        """
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        action_tensor = self.actor(obs_tensor)
        action = action_tensor.squeeze(0).cpu().numpy()
        self.last_action = action.copy()
        return action

    def build_joint_command(self, action: np.ndarray) -> dict:
        """Convert policy action to Spot SDK joint command format.

        Args:
            action: (12,) raw action from get_action().

        Returns:
            Dict with joint names -> (position, kp, kd) for Spot SDK.
        """
        scaled_action = action * self.config.action_scale
        target_positions = self.default_pos + scaled_action

        command = {}
        for i, name in enumerate(DOF_ORDER):
            command[name] = {
                "position": float(target_positions[i]),
                "kp": self.config.kp,
                "kd": self.config.kd,
            }
        return command

    def stop(self):
        """Send zero-action command (return to default pose)."""
        self.last_action = np.zeros(12, dtype=np.float32)
        return self.build_joint_command(np.zeros(12))
