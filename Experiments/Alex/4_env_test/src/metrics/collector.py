"""Per-episode metrics collector — accumulates data during simulation.

Tracks per-step robot state and computes per-episode summary metrics.
Exports results as JSONL (one JSON object per line per episode).
"""

import json
import math
import os

import numpy as np


# Fall threshold in meters
FALL_THRESHOLD = 0.15
# Completion threshold in meters (x-axis)
COMPLETION_X = 49.0
# Zone length in meters
ZONE_LENGTH = 10.0


def quat_to_euler(quat):
    """Convert quaternion [w, x, y, z] to (roll, pitch, yaw) in radians.

    Uses scalar-first convention matching Isaac Sim.
    """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]

    # Roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    # Yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def _get_zone(x_pos):
    """Get zone index (0-4) from x position."""
    return max(0, min(4, int(x_pos / ZONE_LENGTH)))


class MetricsCollector:
    """Accumulates per-step data and exports per-episode metrics to JSONL."""

    def __init__(self, output_dir, env_name, policy_name):
        """
        Args:
            output_dir: Directory for JSONL output files
            env_name: Environment name (friction, grass, boulder, stairs)
            policy_name: Policy name (flat, rough)
        """
        self.output_dir = output_dir
        self.env_name = env_name
        self.policy_name = policy_name
        self.episodes = []

        # Per-episode buffers (reset each episode)
        self._reset_buffers()
        self._episode_id = None
        self._episode_active = False

    def _reset_buffers(self):
        """Clear step data buffers for a new episode."""
        self._positions = []      # (x, y, z) per step
        self._rolls = []
        self._pitches = []
        self._heights = []
        self._ang_vels = []       # ||angular_velocity|| per step
        self._fwd_vels = []       # forward velocity per step
        self._energies = []       # |torque * joint_vel| sum per step
        self._sim_times = []
        self._fall_detected = False
        self._fall_location = None
        self._fall_zone = None
        self._max_x = 0.0
        self._completion_time = None
        self._step_count = 0

    def start_episode(self, episode_id):
        """Begin recording a new episode."""
        self._reset_buffers()
        self._episode_id = episode_id
        self._episode_active = True

    def step(self, root_pos, root_quat, root_lin_vel, root_ang_vel,
             joint_torques=None, joint_vel=None, sim_time=0.0):
        """Record one control step of data.

        Args:
            root_pos: (3,) numpy — world position [x, y, z]
            root_quat: (4,) numpy — quaternion [w, x, y, z]
            root_lin_vel: (3,) numpy — linear velocity [vx, vy, vz]
            root_ang_vel: (3,) numpy — angular velocity [wx, wy, wz]
            joint_torques: (12,) numpy or None — applied torques
            joint_vel: (12,) numpy or None — joint velocities
            sim_time: float — current simulation time in seconds
        """
        if not self._episode_active:
            return

        self._step_count += 1
        x, y, z = float(root_pos[0]), float(root_pos[1]), float(root_pos[2])
        self._positions.append((x, y, z))
        self._heights.append(z)

        # Extract euler angles
        roll, pitch, yaw = quat_to_euler(root_quat)
        self._rolls.append(roll)
        self._pitches.append(pitch)

        # Angular velocity magnitude
        ang_vel_mag = float(np.linalg.norm(root_ang_vel))
        self._ang_vels.append(ang_vel_mag)

        # Forward velocity (x component)
        self._fwd_vels.append(float(root_lin_vel[0]))

        # Energy: sum(|torque * joint_vel|)
        if joint_torques is not None and joint_vel is not None:
            energy = float(np.sum(np.abs(joint_torques * joint_vel)))
            self._energies.append(energy)

        self._sim_times.append(sim_time)

        # Track max progress
        if x > self._max_x:
            self._max_x = x

        # Check completion
        if self._completion_time is None and x >= COMPLETION_X:
            self._completion_time = sim_time

        # Check fall
        if not self._fall_detected and z < FALL_THRESHOLD:
            self._fall_detected = True
            self._fall_location = x
            self._fall_zone = _get_zone(x) + 1  # 1-indexed

    def episode_done(self):
        """Check if the current episode should terminate.

        Returns True if robot has fallen.
        Timeout is managed by the outer loop via step count.
        """
        return self._fall_detected

    @property
    def last_progress(self):
        """Max x-position achieved in current/last episode."""
        return self._max_x

    def end_episode(self):
        """Compute final metrics for the current episode and store them."""
        if not self._episode_active:
            return {}

        self._episode_active = False

        rolls = np.array(self._rolls) if self._rolls else np.array([0.0])
        pitches = np.array(self._pitches) if self._pitches else np.array([0.0])
        heights = np.array(self._heights) if self._heights else np.array([0.0])
        ang_vels = np.array(self._ang_vels) if self._ang_vels else np.array([0.0])
        fwd_vels = np.array(self._fwd_vels) if self._fwd_vels else np.array([0.0])

        mean_roll = float(np.mean(np.abs(rolls)))
        mean_pitch = float(np.mean(np.abs(pitches)))
        height_var = float(np.var(heights))
        mean_ang_vel = float(np.mean(ang_vels))

        # Stability score: lower = more stable
        stability = (
            1.0 * mean_roll +
            1.0 * mean_pitch +
            10.0 * height_var +
            0.5 * mean_ang_vel
        )

        # Episode duration
        if self._sim_times:
            episode_length = self._sim_times[-1] - self._sim_times[0]
        else:
            episode_length = 0.0

        metrics = {
            "episode_id": self._episode_id,
            "policy": self.policy_name,
            "environment": self.env_name,
            "completion": self._max_x >= COMPLETION_X,
            "progress": round(self._max_x, 3),
            "zone_reached": min(5, _get_zone(self._max_x) + 1),
            "time_to_complete": round(self._completion_time, 3) if self._completion_time else None,
            "stability_score": round(stability, 6),
            "mean_roll": round(mean_roll, 6),
            "mean_pitch": round(mean_pitch, 6),
            "height_variance": round(height_var, 6),
            "mean_ang_vel": round(mean_ang_vel, 6),
            "fall_detected": self._fall_detected,
            "fall_location": round(self._fall_location, 3) if self._fall_location is not None else None,
            "fall_zone": self._fall_zone,
            "mean_velocity": round(float(np.mean(fwd_vels)), 4),
            "total_energy": round(sum(self._energies), 2) if self._energies else 0.0,
            "episode_length": round(episode_length, 3),
        }

        self.episodes.append(metrics)
        return metrics

    def save(self, output_dir=None):
        """Append all collected episodes to JSONL file.

        File: {output_dir}/{env_name}_{policy_name}_episodes.jsonl
        """
        out = output_dir or self.output_dir
        os.makedirs(out, exist_ok=True)

        filename = f"{self.env_name}_{self.policy_name}_episodes.jsonl"
        filepath = os.path.join(out, filename)

        with open(filepath, "a") as f:
            for ep in self.episodes:
                f.write(json.dumps(ep) + "\n")

        count = len(self.episodes)
        self.episodes.clear()
        print(f"Saved {count} episodes to {filepath}")
        return filepath
