"""Circular waypoint follower for the 5-ring gauntlet.

Generates 50 ring waypoints (10 per ring at 36-deg spacing, clockwise)
plus 4 transition waypoints between rings. Total: 54 waypoints.

Waypoint order:
  Ring 1: 10 waypoints at r=5m,  angles 0, 36, 72, ..., 324 deg
  Transition 1→2: 1 waypoint at r=10m, angle=0
  Ring 2: 10 waypoints at r=15m, angles 0, 36, ..., 324 deg
  Transition 2→3: ...
  ...
  Ring 5: 10 waypoints at r=45m, angles 0, 36, ..., 324 deg

Heading controller:
  Point-to-point heading with distance-based speed modulation.
  vx = 1.0 (>3m), 0.7 (1-3m), 0.3 (<1m)
  omega_z = clip(2.0 * yaw_error, -2.0, 2.0)
"""

import numpy as np

from configs.ring_params import (
    NUM_RINGS, WAYPOINTS_PER_RING, RING_WIDTH,
    WAYPOINT_THRESHOLD, ring_midpoint_radius,
)


KP_YAW = 2.0


def _generate_waypoints():
    """Generate all 54 waypoints in order.

    Returns:
        list of (x, y) tuples, list of ring assignments (1-5, 0 for transition)
    """
    waypoints = []
    ring_assignments = []

    for ring_idx in range(NUM_RINGS):
        ring_num = ring_idx + 1
        r = ring_midpoint_radius(ring_idx)

        # 10 waypoints evenly spaced clockwise (negative angle direction)
        for wp in range(WAYPOINTS_PER_RING):
            angle = -2.0 * np.pi * wp / WAYPOINTS_PER_RING  # clockwise
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            waypoints.append((x, y))
            ring_assignments.append(ring_num)

        # Transition waypoint at ring boundary (angle=0, at outer edge)
        if ring_idx < NUM_RINGS - 1:
            trans_r = (ring_idx + 1) * RING_WIDTH  # boundary radius
            waypoints.append((trans_r, 0.0))
            ring_assignments.append(0)  # transition

    return waypoints, ring_assignments


class RingFollower:
    """Ring-aware circular waypoint follower.

    Generates velocity commands to guide the robot through 54 waypoints
    across 5 concentric rings of increasing difficulty.
    """

    def __init__(self):
        self._waypoints, self._ring_assignments = _generate_waypoints()
        self._num_waypoints = len(self._waypoints)
        self._current_wp = 0

        # Pre-compute ring waypoint ranges for progress tracking
        self._ring_wp_ranges = {}  # ring_num -> (start_idx, end_idx)
        for ring_num in range(1, NUM_RINGS + 1):
            indices = [i for i, r in enumerate(self._ring_assignments)
                       if r == ring_num]
            if indices:
                self._ring_wp_ranges[ring_num] = (indices[0], indices[-1])

    def reset(self):
        """Reset to first waypoint."""
        self._current_wp = 0

    def compute_commands(self, root_pos, root_yaw):
        """Compute velocity command to steer toward current waypoint.

        Args:
            root_pos: (3,) numpy array — robot world position [x, y, z]
            root_yaw: float — robot yaw angle in radians

        Returns:
            (3,) numpy array — [vx, vy, omega_z] velocity command
        """
        if self.is_done:
            return np.array([0.0, 0.0, 0.0])

        wp_idx = min(self._current_wp, self._num_waypoints - 1)
        tx, ty = self._waypoints[wp_idx]

        # Distance and heading to waypoint
        dx = tx - root_pos[0]
        dy = ty - root_pos[1]
        dist = np.sqrt(dx * dx + dy * dy)
        desired_yaw = np.arctan2(dy, dx)

        # Heading error wrapped to [-pi, pi]
        yaw_err = desired_yaw - root_yaw
        yaw_err = np.arctan2(np.sin(yaw_err), np.cos(yaw_err))

        # Distance-based speed modulation
        if dist > 3.0:
            vx = 1.0
        elif dist > 1.0:
            vx = 0.7
        else:
            vx = 0.3

        vy = 0.0
        omega_z = KP_YAW * yaw_err

        # Clamp to training ranges
        vx = np.clip(vx, -2.0, 3.0)
        omega_z = np.clip(omega_z, -2.0, 2.0)

        # Advance waypoint when close enough
        if dist < WAYPOINT_THRESHOLD:
            if self._current_wp < self._num_waypoints:
                self._current_wp += 1

        return np.array([vx, vy, omega_z])

    @property
    def is_done(self):
        """True if all 54 waypoints have been reached."""
        return self._current_wp > self._num_waypoints - 1

    @property
    def current_ring(self):
        """Current ring number (1-5), or 0 if on a transition waypoint."""
        if self.is_done:
            return NUM_RINGS
        wp_idx = min(self._current_wp, self._num_waypoints - 1)
        return self._ring_assignments[wp_idx]

    @property
    def waypoints_completed(self):
        """Number of waypoints reached so far (0-54)."""
        return self._current_wp

    def ring_progress(self):
        """Get (ring_number, waypoints_completed_in_ring) tuple.

        Returns:
            (int, int): Current ring (1-5) and waypoints done in that ring.
        """
        ring_num = self.current_ring
        if ring_num == 0:
            # On transition — count as previous ring complete
            ring_num = max(1, self._ring_assignments[
                max(0, self._current_wp - 1)])

        # Count completed waypoints in this ring
        completed_in_ring = 0
        for i in range(self._current_wp):
            if i < self._num_waypoints and self._ring_assignments[i] == ring_num:
                completed_in_ring += 1

        return ring_num, completed_in_ring

    def waypoints_in_ring(self, ring_num):
        """Count how many waypoints were completed in a specific ring."""
        completed = 0
        for i in range(min(self._current_wp, self._num_waypoints)):
            if self._ring_assignments[i] == ring_num:
                completed += 1
        return completed

    @property
    def all_waypoints(self):
        """Return list of all (x, y) waypoints for visualization."""
        return list(self._waypoints)

    @property
    def current_target(self):
        """Return current target waypoint (x, y) or None if done."""
        if self.is_done:
            return None
        wp_idx = min(self._current_wp, self._num_waypoints - 1)
        return self._waypoints[wp_idx]
