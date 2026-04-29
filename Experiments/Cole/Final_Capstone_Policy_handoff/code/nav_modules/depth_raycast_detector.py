"""Forward-facing PhysX raycast 'depth sensor' for obstacle detection.

Replaces the ground-truth reveal in OnlineObstacleTracker with real sensing:
casts a fan of rays from the robot's head, projects hits into a persistent
2D occupancy grid, and exports clustered obstacle bounding boxes in the
`(cx, cy, size)` format that `grid_astar_planner.py` consumes.

This is the honest version of "put the robot in an unknown arena" — obstacles
are detected by physics, not by distance-gating cheat data.
"""

import math
import numpy as np


class DepthRaycastObstacleDetector:
    def __init__(
        self,
        robot_prim_path="/World/Spot",
        sensor_height=0.55,
        fov_horiz_deg=90.0,
        fov_vert_deg=30.0,
        n_rays_h=64,
        n_rays_v=16,
        max_distance=8.0,
        grid_res=0.4,
        grid_bounds=(-25.0, 25.0),
        min_hits_per_cell=2,
        max_obstacle_height=2.0,
        min_obstacle_height=0.05,
    ):
        """
        robot_prim_path: prefix used to filter self-hits from raycast results.
        sensor_height: z-offset of the sensor origin above the robot base (m).
        fov_horiz_deg: horizontal field of view (degrees).
        fov_vert_deg:  vertical field of view, symmetric about horizontal (deg).
        n_rays_h:      horizontal ray count.
        n_rays_v:      vertical ray count.
        max_distance:  max raycast distance (m).
        grid_res:      occupancy grid cell size (m).
        grid_bounds:   square arena extent (lo, hi) in world frame.
        min_hits_per_cell: hits needed before a cell becomes an obstacle.
        max/min_obstacle_height: ignore hits outside this z-range (floor/ceiling).
        """
        self.robot_prefix = robot_prim_path
        self.sensor_height = float(sensor_height)
        self.max_distance = float(max_distance)
        self.min_hits_per_cell = int(min_hits_per_cell)
        self.max_obstacle_height = float(max_obstacle_height)
        self.min_obstacle_height = float(min_obstacle_height)

        lo, hi = grid_bounds
        self.grid_lo = float(lo)
        self.grid_res = float(grid_res)
        self.grid_n = int(math.ceil((hi - lo) / grid_res))
        self._hits = np.zeros((self.grid_n, self.grid_n), dtype=np.int32)

        h_half = math.radians(fov_horiz_deg * 0.5)
        v_half = math.radians(fov_vert_deg * 0.5)
        yaws = np.linspace(-h_half, h_half, n_rays_h)
        pitches = np.linspace(-v_half, v_half, n_rays_v)
        self._ray_dirs_body = np.stack(
            [
                (math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p))
                for p in pitches
                for y in yaws
            ],
            axis=0,
        ).astype(np.float32)  # shape (n_rays_h * n_rays_v, 3)

        self._cached_obstacles = []
        self._dirty = True

    def reset(self):
        self._hits.fill(0)
        self._cached_obstacles = []
        self._dirty = True

    def _world_from_body(self, yaw, v):
        c, s = math.cos(yaw), math.sin(yaw)
        x = c * v[0] - s * v[1]
        y = s * v[0] + c * v[1]
        return x, y, v[2]

    def _cell_of(self, x, y):
        i = int((x - self.grid_lo) / self.grid_res)
        j = int((y - self.grid_lo) / self.grid_res)
        if 0 <= i < self.grid_n and 0 <= j < self.grid_n:
            return i, j
        return None

    def sense(self, robot_xy, robot_yaw, robot_z=None):
        """Cast the ray fan from the robot's head. Update the occupancy grid.

        Returns True if any new cell crossed the min-hits threshold (caller may
        trigger a replan). The robot_z arg is optional — if omitted, uses
        sensor_height relative to z=0 (works when terrain is flat).
        """
        from omni.physx import get_physx_scene_query_interface
        sq = get_physx_scene_query_interface()
        bz = (float(robot_z) if robot_z is not None else 0.0) + self.sensor_height
        ox, oy, oz = float(robot_xy[0]), float(robot_xy[1]), bz

        grew = False
        hits = self._hits
        threshold = self.min_hits_per_cell
        zmin = self.min_obstacle_height
        zmax = self.max_obstacle_height

        for dir_body in self._ray_dirs_body:
            wx, wy, wz = self._world_from_body(robot_yaw, dir_body)
            norm = math.sqrt(wx * wx + wy * wy + wz * wz)
            if norm < 1e-6:
                continue
            wx, wy, wz = wx / norm, wy / norm, wz / norm
            hit = sq.raycast_closest((ox, oy, oz), (wx, wy, wz), self.max_distance)
            if not hit["hit"]:
                continue
            if self.robot_prefix in hit.get("rigidBody", ""):
                continue
            pos = hit["position"]
            hz = float(pos[2])
            if hz < zmin or hz > zmax:
                continue
            cell = self._cell_of(float(pos[0]), float(pos[1]))
            if cell is None:
                continue
            i, j = cell
            prev = hits[i, j]
            hits[i, j] = prev + 1
            if prev < threshold <= prev + 1:
                grew = True

        if grew:
            self._dirty = True
        return grew

    def known_obstacles(self):
        """Export clustered occupancy as a list of (cx, cy, size) tuples.

        A cell crosses into "known" once its hit count ≥ min_hits_per_cell.
        Each qualifying cell becomes a single obstacle of size `grid_res`.
        The grid A* planner's own inflation pads them with safety margin.
        """
        if not self._dirty:
            return self._cached_obstacles
        occupied_ij = np.argwhere(self._hits >= self.min_hits_per_cell)
        out = []
        half_res = self.grid_res * 0.5
        for (i, j) in occupied_ij:
            cx = self.grid_lo + (i + 0.5) * self.grid_res
            cy = self.grid_lo + (j + 0.5) * self.grid_res
            out.append((float(cx), float(cy), self.grid_res))
        self._cached_obstacles = out
        self._dirty = False
        return out

    @property
    def n_known(self):
        return int(np.count_nonzero(self._hits >= self.min_hits_per_cell))

    @property
    def n_total_cells(self):
        return int(self.grid_n * self.grid_n)
