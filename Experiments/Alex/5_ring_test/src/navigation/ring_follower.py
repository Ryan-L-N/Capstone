"""Quadrant waypoint follower for the 4-quadrant gauntlet.

Generates waypoints across 4 terrain quadrants, each with 5 difficulty levels.
2 waypoints per level per quadrant = 10/quadrant, 40 total + 4 transitions = 44.

Navigation order: LEVEL BY LEVEL — at each difficulty ring, visit all 4 quadrants
before moving outward:
  L1: Friction(2) → Grass(2) → Boulders(2) → Stairs(2) → transition out
  L2: Friction(2) → Grass(2) → Boulders(2) → Stairs(2) → transition out
  ...
  L5: Friction(2) → Grass(2) → Boulders(2) → Stairs(2)

This ensures the robot circles the arena at each difficulty before progressing.
Stairs waypoints are at pyramid summit positions (z > 0).

Heading controller:
  Point-to-point heading with distance-based speed modulation.
  vx = 1.0 (>3m), 0.7 (1-3m), 0.3 (<1m)
  omega_z = clip(2.0 * yaw_error, -2.0, 2.0)
"""

import numpy as np

from configs.ring_params import (
    NUM_QUADRANTS, NUM_LEVELS, WPS_PER_LEVEL, WPS_PER_QUADRANT,
    LEVEL_WIDTH, WAYPOINT_THRESHOLD, QUADRANT_DEFS, STAIRS_LEVELS,
    level_midpoint_radius, pyramid_summit_height,
)


KP_YAW = 2.0


def _generate_waypoints():
    """Generate all 44 waypoints in level-first order.

    Outer loop: levels 1-5 (inner to outer)
    Inner loop: quadrants 0-3 (friction → grass → boulders → stairs)
    Per combo: 2 waypoints spread across the quadrant arc.
    Transition WP after each level (except the last).

    Returns:
        list of (x, y, z) tuples
        list of (quadrant_index, level_number) assignments
            quadrant_index: 0-3, level_number: 1-5 (0 for transition)
    """
    waypoints = []
    assignments = []

    for lvl_idx in range(NUM_LEVELS):
        r = level_midpoint_radius(lvl_idx)
        level_num = lvl_idx + 1

        for quad_idx in range(NUM_QUADRANTS):
            qdef = QUADRANT_DEFS[quad_idx]
            a_start = qdef["angle_start"]
            a_end = qdef["angle_end"]
            a_span = a_end - a_start

            for wp_i in range(WPS_PER_LEVEL):
                frac = (wp_i + 1) / (WPS_PER_LEVEL + 1)
                angle = a_start + a_span * frac

                x = r * np.cos(angle)
                y = r * np.sin(angle)

                if qdef["name"] == "stairs":
                    z = pyramid_summit_height(STAIRS_LEVELS[lvl_idx])
                else:
                    z = 0.0

                waypoints.append((x, y, z))
                assignments.append((quad_idx, level_num))

        # Transition waypoint after this level ring (except last)
        if lvl_idx < NUM_LEVELS - 1:
            # Place at angle=0 (friction start), next level's midpoint radius
            next_r = level_midpoint_radius(lvl_idx + 1)
            waypoints.append((next_r, 0.0, 0.0))
            assignments.append((-1, 0))  # -1 = transition between levels

    return waypoints, assignments


class QuadrantFollower:
    """Quadrant-aware waypoint follower for the 4-terrain gauntlet."""

    def __init__(self):
        self._waypoints, self._assignments = _generate_waypoints()
        self._num_waypoints = len(self._waypoints)
        self._current_wp = 0

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
        tx, ty, tz = self._waypoints[wp_idx]

        # 2D distance (ignore z for approach)
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

        # Advance waypoint when close enough (2D distance)
        if dist < WAYPOINT_THRESHOLD:
            if self._current_wp < self._num_waypoints:
                self._current_wp += 1

        return np.array([vx, vy, omega_z])

    @property
    def is_done(self):
        """True if all waypoints have been reached."""
        return self._current_wp > self._num_waypoints - 1

    @property
    def current_quadrant(self):
        """Current quadrant index (0-3), or -1 if on a level transition."""
        if self.is_done:
            return NUM_QUADRANTS - 1
        wp_idx = min(self._current_wp, self._num_waypoints - 1)
        return self._assignments[wp_idx][0]

    @property
    def current_quadrant_name(self):
        """Current quadrant name string, or 'transition' for level transitions."""
        qi = self.current_quadrant
        if qi < 0:
            return "transition"
        return QUADRANT_DEFS[qi]["name"]

    @property
    def current_level(self):
        """Current level number (1-5), or 0 if on transition."""
        if self.is_done:
            return NUM_LEVELS
        wp_idx = min(self._current_wp, self._num_waypoints - 1)
        return self._assignments[wp_idx][1]

    @property
    def waypoints_completed(self):
        """Number of waypoints reached so far."""
        return self._current_wp

    def waypoints_in_quadrant(self, quad_idx):
        """Count completed waypoints in a specific quadrant (excl. transitions)."""
        completed = 0
        for i in range(min(self._current_wp, self._num_waypoints)):
            q, lvl = self._assignments[i]
            if q == quad_idx and lvl > 0:
                completed += 1
        return completed

    def waypoints_in_quadrant_level(self, quad_idx, level_num):
        """Count completed waypoints in a specific quadrant and level."""
        completed = 0
        for i in range(min(self._current_wp, self._num_waypoints)):
            q, lvl = self._assignments[i]
            if q == quad_idx and lvl == level_num:
                completed += 1
        return completed

    @property
    def all_waypoints(self):
        """Return list of all (x, y, z) waypoints for visualization."""
        return list(self._waypoints)

    @property
    def current_target(self):
        """Return current target waypoint (x, y, z) or None if done."""
        if self.is_done:
            return None
        wp_idx = min(self._current_wp, self._num_waypoints - 1)
        return self._waypoints[wp_idx]

    def stairs_waypoint_positions(self):
        """Return just the (x, y) positions for stairs quadrant waypoints.

        Used by the arena builder to know where to place pyramids.
        """
        stairs_quad_idx = 3  # Q4 = stairs
        positions = []
        for i, (wp, (q, lvl)) in enumerate(zip(self._waypoints, self._assignments)):
            if q == stairs_quad_idx and lvl > 0:
                positions.append((wp[0], wp[1]))
        return positions
