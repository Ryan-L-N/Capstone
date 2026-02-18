"""Waypoint follower — generates velocity commands to guide robot through zones.

6 waypoints along the arena centerline (y=15m):
  WP0 (spawn): (0.0, 15.0)
  WP1:         (10.0, 15.0) — Zone 1/2 boundary
  WP2:         (20.0, 15.0) — Zone 2/3 boundary
  WP3:         (30.0, 15.0) — Zone 3/4 boundary
  WP4:         (40.0, 15.0) — Zone 4/5 boundary
  WP5 (goal):  (50.0, 15.0)

Heading controller:
  theta_err = atan2(wp_y - robot_y, wp_x - robot_x) - robot_yaw
  vx = 1.0 m/s (constant forward), vy = 0.0
  omega_z = Kp * theta_err (Kp=2.0)

Commands clamped to training ranges: vx[-2, 3], vy[-1.5, 1.5], wz[-2, 2]
"""

import numpy as np

# Waypoints along arena centerline
WAYPOINTS = np.array([
    [0.0, 15.0],
    [10.0, 15.0],
    [20.0, 15.0],
    [30.0, 15.0],
    [40.0, 15.0],
    [50.0, 15.0],
], dtype=np.float64)

KP_YAW = 2.0
TARGET_VX = 1.0
WAYPOINT_THRESHOLD = 0.5


class WaypointFollower:
    """Generates velocity commands to follow waypoints through the arena.

    Designed for standalone mode (single robot, numpy arrays).
    """

    def __init__(self):
        self._current_wp = 0
        self._waypoints = WAYPOINTS.copy()

    def reset(self):
        """Reset to first waypoint."""
        self._current_wp = 0

    @property
    def is_done(self):
        """True if robot has reached or passed the final waypoint."""
        return self._current_wp >= len(self._waypoints) - 1

    @property
    def current_waypoint_index(self):
        return self._current_wp

    def compute_commands(self, root_pos, root_yaw):
        """Compute velocity command to steer toward current waypoint.

        Args:
            root_pos: (3,) numpy array — robot world position [x, y, z]
            root_yaw: float — robot yaw angle in radians

        Returns:
            (3,) numpy array — [vx, vy, omega_z] velocity command
        """
        wp_idx = min(self._current_wp, len(self._waypoints) - 1)
        target = self._waypoints[wp_idx]

        # Heading to waypoint
        dx = target[0] - root_pos[0]
        dy = target[1] - root_pos[1]
        desired_yaw = np.arctan2(dy, dx)

        # Heading error wrapped to [-pi, pi]
        yaw_err = desired_yaw - root_yaw
        yaw_err = np.arctan2(np.sin(yaw_err), np.cos(yaw_err))

        # Velocity commands
        vx = TARGET_VX
        vy = 0.0
        omega_z = KP_YAW * yaw_err

        # Clamp to training ranges
        vx = np.clip(vx, -2.0, 3.0)
        omega_z = np.clip(omega_z, -2.0, 2.0)

        # Advance waypoint when robot passes waypoint x-position
        if root_pos[0] >= target[0] - WAYPOINT_THRESHOLD:
            if self._current_wp < len(self._waypoints) - 1:
                self._current_wp += 1

        return np.array([vx, vy, omega_z])
