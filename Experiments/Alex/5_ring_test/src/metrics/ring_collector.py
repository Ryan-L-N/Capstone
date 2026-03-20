"""Per-quadrant metrics collector and composite scoring for the 4-quadrant gauntlet.

Tracks per-step robot state and computes per-quadrant, per-level, and per-episode
summary metrics. Exports results as JSONL.

Composite score: per quadrant = sum(wps_in_level / 2 * level_weight)
  Level weights = [10, 20, 30, 40, 50], max per quadrant = 150, max total = 600
"""

import json
import math
import os

import numpy as np

from configs.ring_params import (
    NUM_QUADRANTS, NUM_LEVELS, WPS_PER_LEVEL, LEVEL_WEIGHTS,
    MAX_SCORE_PER_QUADRANT, MAX_SCORE, QUADRANT_DEFS, LEVEL_WIDTH,
    get_quadrant_and_level,
)


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


class QuadrantCollector:
    """Accumulates per-step data and exports per-episode quadrant metrics to JSONL."""

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

        # Per-quadrant, per-level tracking
        self._quad_level_rolls = {}
        self._quad_level_pitches = {}
        self._quad_level_heights = {}
        self._quad_level_ang_vels = {}
        self._quad_level_times = {}
        for q in range(NUM_QUADRANTS):
            for lvl in range(1, NUM_LEVELS + 1):
                key = (q, lvl)
                self._quad_level_rolls[key] = []
                self._quad_level_pitches[key] = []
                self._quad_level_heights[key] = []
                self._quad_level_ang_vels[key] = []
                self._quad_level_times[key] = []

        self._fall_detected = False
        self._fall_location = None
        self._fall_quadrant = None
        self._fall_level = None
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
             sim_time=0.0, ground_z=0.0):
        """Record one control step of data.

        Args:
            root_pos: (3,) numpy — world position [x, y, z]
            root_quat: (4,) numpy — quaternion [w, x, y, z]
            root_lin_vel: (3,) numpy — linear velocity
            root_ang_vel: (3,) numpy — angular velocity
            sim_time: float — simulation time in seconds
            ground_z: float — ground surface height at robot position (for stairs)
        """
        if not self._episode_active:
            return

        self._step_count += 1
        x, y, z = float(root_pos[0]), float(root_pos[1]), float(root_pos[2])
        self._positions.append((x, y, z))

        height_above_ground = z - ground_z
        self._heights.append(height_above_ground)

        r = np.sqrt(x * x + y * y)
        if r > self._max_radius:
            self._max_radius = r

        if self._prev_pos is not None:
            dx = x - self._prev_pos[0]
            dy = y - self._prev_pos[1]
            self._total_distance += np.sqrt(dx * dx + dy * dy)
        self._prev_pos = (x, y, z)

        roll, pitch, yaw = quat_to_euler(root_quat)
        self._rolls.append(roll)
        self._pitches.append(pitch)

        ang_vel_mag = float(np.linalg.norm(root_ang_vel))
        self._ang_vels.append(ang_vel_mag)
        self._fwd_vels.append(float(root_lin_vel[0]))
        self._sim_times.append(sim_time)

        # Per-quadrant-level data
        quad_idx, level_num = get_quadrant_and_level(x, y)
        key = (quad_idx, level_num)
        if key in self._quad_level_rolls:
            self._quad_level_rolls[key].append(roll)
            self._quad_level_pitches[key].append(pitch)
            self._quad_level_heights[key].append(height_above_ground)
            self._quad_level_ang_vels[key].append(ang_vel_mag)
            self._quad_level_times[key].append(sim_time)

        # Fall detection
        if not self._fall_detected and height_above_ground < FALL_THRESHOLD:
            self._fall_detected = True
            self._fall_location = (x, y)
            self._fall_radius = r
            self._fall_quadrant = quad_idx
            self._fall_level = level_num

    def episode_done(self):
        """Check if the current episode should terminate (fall detected)."""
        return self._fall_detected

    def end_episode(self, follower=None):
        """Compute final metrics for the current episode.

        Args:
            follower: Optional QuadrantFollower instance for waypoint counts.

        Returns:
            dict: Episode metrics
        """
        if not self._episode_active:
            return {}

        self._episode_active = False

        # Per-quadrant scoring
        quadrant_scores = {}
        total_wps = 0
        total_score = 0.0

        for quad_idx in range(NUM_QUADRANTS):
            qname = QUADRANT_DEFS[quad_idx]["name"]
            qlabel = QUADRANT_DEFS[quad_idx]["label"]

            quad_wps = 0
            quad_score = 0.0
            level_details = {}

            for lvl in range(1, NUM_LEVELS + 1):
                key = (quad_idx, lvl)
                wps = 0
                if follower is not None:
                    wps = follower.waypoints_in_quadrant_level(quad_idx, lvl)

                # Stability for this level
                rolls = np.array(self._quad_level_rolls[key]) if self._quad_level_rolls[key] else np.array([0.0])
                pitches = np.array(self._quad_level_pitches[key]) if self._quad_level_pitches[key] else np.array([0.0])
                heights = np.array(self._quad_level_heights[key]) if self._quad_level_heights[key] else np.array([0.0])
                ang_vels = np.array(self._quad_level_ang_vels[key]) if self._quad_level_ang_vels[key] else np.array([0.0])
                times = self._quad_level_times[key]

                stability = (
                    1.0 * float(np.mean(np.abs(rolls))) +
                    1.0 * float(np.mean(np.abs(pitches))) +
                    10.0 * float(np.var(heights)) +
                    0.5 * float(np.mean(ang_vels))
                )

                time_in_level = 0.0
                if len(times) >= 2:
                    time_in_level = times[-1] - times[0]

                level_score = (wps / WPS_PER_LEVEL) * LEVEL_WEIGHTS[lvl - 1]
                quad_score += level_score
                quad_wps += wps

                level_details[f"L{lvl}"] = {
                    "waypoints": wps,
                    "time_s": round(time_in_level, 2),
                    "stability": round(stability, 4),
                    "score": round(level_score, 2),
                }

            total_wps += quad_wps
            total_score += quad_score

            quadrant_scores[qname] = {
                "label": qlabel,
                "waypoints": quad_wps,
                "score": round(quad_score, 2),
                "max_score": MAX_SCORE_PER_QUADRANT,
                "levels": level_details,
            }

        # Episode duration
        episode_length = 0.0
        if self._sim_times:
            episode_length = self._sim_times[-1] - self._sim_times[0]

        # Path efficiency
        path_efficiency = 0.0
        if self._total_distance > 0 and total_wps > 0:
            # Rough ideal: straight lines between WPs
            ideal_dist = total_wps * 8.0  # ~8m average inter-WP distance
            path_efficiency = min(1.0, ideal_dist / self._total_distance)

        fall_quad_name = None
        if self._fall_quadrant is not None:
            fall_quad_name = QUADRANT_DEFS[self._fall_quadrant]["name"]

        metrics = {
            "episode_id": self._episode_id,
            "policy": self.policy_name,
            "total_waypoints_reached": total_wps,
            "quadrant_scores": quadrant_scores,
            "composite_score": round(total_score, 2),
            "max_score": MAX_SCORE,
            "fall_detected": self._fall_detected,
            "fall_quadrant": fall_quad_name,
            "fall_level": self._fall_level,
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

        filename = f"quadrant_{self.policy_name}_episodes.jsonl"
        filepath = os.path.join(out, filename)

        with open(filepath, "a") as f:
            for ep in self.episodes:
                f.write(json.dumps(ep) + "\n")

        count = len(self.episodes)
        self.episodes.clear()
        print(f"Saved {count} episodes to {filepath}")
        return filepath
