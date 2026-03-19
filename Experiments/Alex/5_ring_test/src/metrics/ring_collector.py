"""Per-ring metrics collector and composite scoring for the 5-ring gauntlet.

Tracks per-step robot state and computes per-ring and per-episode summary
metrics. Exports results as JSONL (one JSON object per line per episode).

Composite score: sum(waypoints_in_ring / 10 * ring_weight)
  Weights = [10, 20, 30, 40, 50], max = 150
"""

import json
import math
import os

import numpy as np

from configs.ring_params import (
    NUM_RINGS, WAYPOINTS_PER_RING, RING_WEIGHTS, MAX_SCORE,
    get_ring_for_radius,
)


# Fall threshold in meters
FALL_THRESHOLD = 0.15


def quat_to_euler(quat):
    """Convert quaternion [w, x, y, z] to (roll, pitch, yaw) in radians."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class RingCollector:
    """Accumulates per-step data and exports per-episode ring metrics to JSONL."""

    def __init__(self, output_dir, policy_name):
        self.output_dir = output_dir
        self.policy_name = policy_name
        self.episodes = []

        self._reset_buffers()
        self._episode_id = None
        self._episode_active = False

    def _reset_buffers(self):
        """Clear step data buffers for a new episode."""
        self._positions = []
        self._rolls = []
        self._pitches = []
        self._heights = []
        self._ang_vels = []
        self._fwd_vels = []
        self._sim_times = []

        # Per-ring tracking
        self._ring_times = {r: [] for r in range(1, NUM_RINGS + 1)}
        self._ring_rolls = {r: [] for r in range(1, NUM_RINGS + 1)}
        self._ring_pitches = {r: [] for r in range(1, NUM_RINGS + 1)}
        self._ring_heights = {r: [] for r in range(1, NUM_RINGS + 1)}
        self._ring_ang_vels = {r: [] for r in range(1, NUM_RINGS + 1)}

        self._fall_detected = False
        self._fall_location = None
        self._fall_ring = None
        self._fall_radius = None
        self._max_radius = 0.0
        self._total_distance = 0.0
        self._prev_pos = None
        self._step_count = 0

    def start_episode(self, episode_id):
        """Begin recording a new episode."""
        self._reset_buffers()
        self._episode_id = episode_id
        self._episode_active = True

    def step(self, root_pos, root_quat, root_lin_vel, root_ang_vel,
             sim_time=0.0):
        """Record one control step of data.

        Args:
            root_pos: (3,) numpy — world position [x, y, z]
            root_quat: (4,) numpy — quaternion [w, x, y, z]
            root_lin_vel: (3,) numpy — linear velocity [vx, vy, vz]
            root_ang_vel: (3,) numpy — angular velocity [wx, wy, wz]
            sim_time: float — current simulation time in seconds
        """
        if not self._episode_active:
            return

        self._step_count += 1
        x, y, z = float(root_pos[0]), float(root_pos[1]), float(root_pos[2])
        self._positions.append((x, y, z))
        self._heights.append(z)

        # Distance from center
        r = np.sqrt(x * x + y * y)
        if r > self._max_radius:
            self._max_radius = r

        # Track total path distance
        if self._prev_pos is not None:
            dx = x - self._prev_pos[0]
            dy = y - self._prev_pos[1]
            self._total_distance += np.sqrt(dx * dx + dy * dy)
        self._prev_pos = (x, y, z)

        # Euler angles
        roll, pitch, yaw = quat_to_euler(root_quat)
        self._rolls.append(roll)
        self._pitches.append(pitch)

        # Angular velocity magnitude
        ang_vel_mag = float(np.linalg.norm(root_ang_vel))
        self._ang_vels.append(ang_vel_mag)

        # Forward velocity
        self._fwd_vels.append(float(root_lin_vel[0]))
        self._sim_times.append(sim_time)

        # Per-ring data
        ring_num = get_ring_for_radius(r)
        if 1 <= ring_num <= NUM_RINGS:
            self._ring_times[ring_num].append(sim_time)
            self._ring_rolls[ring_num].append(roll)
            self._ring_pitches[ring_num].append(pitch)
            self._ring_heights[ring_num].append(z)
            self._ring_ang_vels[ring_num].append(ang_vel_mag)

        # Fall detection (height relative to ground, ground is flat at z=0)
        if not self._fall_detected and z < FALL_THRESHOLD:
            self._fall_detected = True
            self._fall_location = (x, y)
            self._fall_radius = r
            self._fall_ring = ring_num

    def episode_done(self):
        """Check if the current episode should terminate (fall detected)."""
        return self._fall_detected

    def end_episode(self, ring_follower=None):
        """Compute final metrics for the current episode.

        Args:
            ring_follower: Optional RingFollower instance for waypoint counts.

        Returns:
            dict: Episode metrics
        """
        if not self._episode_active:
            return {}

        self._episode_active = False

        # Waypoints per ring
        ring_wps = {}
        rings_completed = 0
        total_wps = 0

        for ring_num in range(1, NUM_RINGS + 1):
            wps = 0
            if ring_follower is not None:
                wps = ring_follower.waypoints_in_ring(ring_num)
            ring_wps[ring_num] = wps
            total_wps += wps
            if wps >= WAYPOINTS_PER_RING:
                rings_completed += 1

        # Composite score
        composite_score = 0.0
        for ring_num in range(1, NUM_RINGS + 1):
            frac = ring_wps[ring_num] / WAYPOINTS_PER_RING
            composite_score += frac * RING_WEIGHTS[ring_num - 1]

        # Per-ring stability scores
        ring_scores = {}
        for ring_num in range(1, NUM_RINGS + 1):
            rolls = np.array(self._ring_rolls[ring_num]) if self._ring_rolls[ring_num] else np.array([0.0])
            pitches = np.array(self._ring_pitches[ring_num]) if self._ring_pitches[ring_num] else np.array([0.0])
            heights = np.array(self._ring_heights[ring_num]) if self._ring_heights[ring_num] else np.array([0.0])
            ang_vels = np.array(self._ring_ang_vels[ring_num]) if self._ring_ang_vels[ring_num] else np.array([0.0])
            times = self._ring_times[ring_num]

            stability = (
                1.0 * float(np.mean(np.abs(rolls))) +
                1.0 * float(np.mean(np.abs(pitches))) +
                10.0 * float(np.var(heights)) +
                0.5 * float(np.mean(ang_vels))
            )

            time_in_ring = 0.0
            if len(times) >= 2:
                time_in_ring = times[-1] - times[0]

            ring_scores[f"ring_{ring_num}"] = {
                "waypoints": ring_wps[ring_num],
                "time_s": round(time_in_ring, 2),
                "stability": round(stability, 4),
                "label": ["Flat", "Low Friction", "Vegetation",
                          "Boulder Field", "Extreme Mixed"][ring_num - 1],
            }

        # Episode duration
        episode_length = 0.0
        if self._sim_times:
            episode_length = self._sim_times[-1] - self._sim_times[0]

        # Path efficiency: ideal distance / actual distance
        # Ideal = sum of straight-line distances between consecutive waypoints
        # Approximate with total_wps * avg inter-waypoint distance
        path_efficiency = 0.0
        if self._total_distance > 0 and total_wps > 0:
            # Rough ideal: circumference contributions per ring visited
            ideal_dist = 0.0
            for ring_num in range(1, NUM_RINGS + 1):
                if ring_wps[ring_num] > 0:
                    r = ring_num * 10.0 - 5.0  # midpoint radius
                    # Arc between waypoints = 2*pi*r / 10
                    arc = 2.0 * np.pi * r / WAYPOINTS_PER_RING
                    ideal_dist += arc * ring_wps[ring_num]
            # Add transition distances (~10m each)
            ideal_dist += max(0, rings_completed - 1) * 10.0
            if ideal_dist > 0:
                path_efficiency = min(1.0, ideal_dist / self._total_distance)

        metrics = {
            "episode_id": self._episode_id,
            "policy": self.policy_name,
            "total_waypoints_reached": total_wps,
            "rings_completed": rings_completed,
            "ring_scores": ring_scores,
            "composite_score": round(composite_score, 2),
            "max_score": MAX_SCORE,
            "fall_detected": self._fall_detected,
            "fall_ring": self._fall_ring,
            "fall_radius": round(self._fall_radius, 2) if self._fall_radius is not None else None,
            "path_efficiency": round(path_efficiency, 4),
            "total_distance_m": round(self._total_distance, 2),
            "episode_length_s": round(episode_length, 2),
            "max_radius_reached": round(self._max_radius, 2),
        }

        self.episodes.append(metrics)
        return metrics

    def save(self, output_dir=None):
        """Append all collected episodes to JSONL file."""
        out = output_dir or self.output_dir
        os.makedirs(out, exist_ok=True)

        filename = f"ring_{self.policy_name}_episodes.jsonl"
        filepath = os.path.join(out, filename)

        with open(filepath, "a") as f:
            for ep in self.episodes:
                f.write(json.dumps(ep) + "\n")

        count = len(self.episodes)
        self.episodes.clear()
        print(f"Saved {count} episodes to {filepath}")
        return filepath
