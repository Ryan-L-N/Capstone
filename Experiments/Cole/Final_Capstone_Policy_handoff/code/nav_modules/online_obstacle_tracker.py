"""Online obstacle tracker — simulates limited-range sensing on a known scene.

The robot starts with zero obstacle knowledge. On each sense step, any ground-truth
obstacle whose center is within `sense_radius` of the robot is added to a
persistent "known" set. A* replans over only the discovered obstacles.

This is a stand-in for LIDAR/raycast SLAM: the knowledge grows as the robot
explores, same as a real unknown-arena demo, but skips the physics-sim overhead
of projecting per-ray hits into an occupancy grid.
"""

import math


class OnlineObstacleTracker:
    def __init__(self, all_obstacles, sense_radius=3.5):
        """
        all_obstacles: list of (cx, cy, size) — the full ground-truth set (hidden
                       from the navigator; this class reveals them incrementally).
        sense_radius: meters; an obstacle is discovered when the robot passes
                      within this distance of its center.
        """
        self._all = list(all_obstacles) if all_obstacles else []
        self.sense_radius = float(sense_radius)
        self._known_idx = set()

    def sense(self, robot_xy):
        """Reveal any ground-truth obstacles inside sense_radius. Returns True
        if the known-set grew on this step (caller can trigger a replan)."""
        rx, ry = float(robot_xy[0]), float(robot_xy[1])
        R2 = self.sense_radius * self.sense_radius
        grew = False
        for idx, (cx, cy, size) in enumerate(self._all):
            if idx in self._known_idx:
                continue
            half = size * 0.5
            # Use nearest-point-on-square to center — handles large obstacles
            dx = max(abs(rx - cx) - half, 0.0)
            dy = max(abs(ry - cy) - half, 0.0)
            if dx * dx + dy * dy <= R2:
                self._known_idx.add(idx)
                grew = True
        return grew

    def known_obstacles(self):
        return [self._all[i] for i in self._known_idx]

    def reset(self):
        self._known_idx.clear()

    @property
    def n_known(self):
        return len(self._known_idx)

    @property
    def n_total(self):
        return len(self._all)
