"""Height scan builder — converts real depth sensor data to 187-dim observation.

Transforms depth camera point cloud or LiDAR data into the same 17x11 grid
format used during training (1.6m x 1.0m coverage, 0.1m resolution).

The height scan measures terrain elevation relative to the robot body,
using yaw-only rotation (matching spot_rough_terrain_policy.py convention).

Missing/invalid data filled with 0.0 (matching training convention for flat ground).

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class HeightScanBuilder:
    """Converts real sensor data to 187-dim height scan observation.

    Grid: 17 points forward/back x 11 points left/right = 187 points
    Coverage: 1.6m (forward/back) x 1.0m (left/right)
    Resolution: 0.1m between grid points
    Origin: Robot body center
    """

    def __init__(
        self,
        grid_rows: int = 17,
        grid_cols: int = 11,
        grid_size_x: float = 1.6,   # Forward/back coverage (m)
        grid_size_y: float = 1.0,   # Left/right coverage (m)
        fill_value: float = 0.0,    # Value for missing data
        clip_range: float = 1.0,    # Clip heights to [-1.0, 1.0] m
    ):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.fill_value = fill_value
        self.clip_range = clip_range

        # Pre-compute grid positions in body frame
        # x: forward (+) to backward (-), y: left (+) to right (-)
        x_range = np.linspace(-grid_size_x / 2, grid_size_x / 2, grid_rows)
        y_range = np.linspace(-grid_size_y / 2, grid_size_y / 2, grid_cols)
        self.grid_x, self.grid_y = np.meshgrid(x_range, y_range, indexing='ij')
        self.grid_x = self.grid_x.flatten()  # (187,)
        self.grid_y = self.grid_y.flatten()  # (187,)

    def from_point_cloud(
        self,
        points: np.ndarray,
        robot_pos: np.ndarray,
        robot_yaw: float,
        robot_height: float,
    ) -> np.ndarray:
        """Convert 3D point cloud to 187-dim height scan.

        Args:
            points: (N, 3) point cloud in world frame [x, y, z].
            robot_pos: (3,) robot position in world frame.
            robot_yaw: Robot yaw angle (radians).
            robot_height: Robot body height above ground (m).

        Returns:
            (187,) height scan observation (heights relative to robot body).
        """
        # Transform points to robot body frame (yaw-only rotation)
        cos_yaw = np.cos(-robot_yaw)
        sin_yaw = np.sin(-robot_yaw)

        # Translate to robot origin
        local = points - robot_pos

        # Rotate to body frame (yaw only — matching training convention)
        body_x = local[:, 0] * cos_yaw - local[:, 1] * sin_yaw
        body_y = local[:, 0] * sin_yaw + local[:, 1] * cos_yaw
        body_z = local[:, 2]

        # Initialize height scan with fill value
        height_scan = np.full(self.grid_rows * self.grid_cols, self.fill_value, dtype=np.float32)

        # For each grid cell, find the closest point and use its height
        resolution = self.grid_size_x / (self.grid_rows - 1)
        half_cell = resolution / 2

        for i in range(len(self.grid_x)):
            # Find points within this grid cell
            x_mask = np.abs(body_x - self.grid_x[i]) < half_cell
            y_mask = np.abs(body_y - self.grid_y[i]) < half_cell
            cell_mask = x_mask & y_mask

            if cell_mask.any():
                # Use maximum height in cell (conservative — detects obstacles)
                cell_height = np.max(body_z[cell_mask])
                # Height relative to robot body
                height_scan[i] = cell_height - robot_pos[2]

        # Subtract robot height to get terrain height relative to body
        height_scan -= robot_height

        # Clip to training range
        height_scan = np.clip(height_scan, -self.clip_range, self.clip_range)

        return height_scan

    def from_depth_image(
        self,
        depth_image: np.ndarray,
        camera_intrinsics: dict,
        camera_extrinsics: np.ndarray,
        robot_pos: np.ndarray,
        robot_yaw: float,
        robot_height: float,
    ) -> np.ndarray:
        """Convert depth image to 187-dim height scan via point cloud.

        Args:
            depth_image: (H, W) depth in meters.
            camera_intrinsics: Dict with 'fx', 'fy', 'cx', 'cy'.
            camera_extrinsics: (4, 4) camera-to-world transform.
            robot_pos: (3,) robot position in world frame.
            robot_yaw: Robot yaw angle (radians).
            robot_height: Robot body height above ground (m).

        Returns:
            (187,) height scan observation.
        """
        H, W = depth_image.shape
        fx, fy = camera_intrinsics['fx'], camera_intrinsics['fy']
        cx, cy = camera_intrinsics['cx'], camera_intrinsics['cy']

        # Create pixel grid
        u, v = np.meshgrid(np.arange(W), np.arange(H))

        # Valid depth mask
        valid = (depth_image > 0.1) & (depth_image < 5.0)

        # Back-project to 3D (camera frame)
        z = depth_image[valid]
        x = (u[valid] - cx) * z / fx
        y = (v[valid] - cy) * z / fy

        # Camera frame points
        cam_points = np.stack([x, y, z], axis=-1)  # (N, 3)

        # Transform to world frame
        ones = np.ones((cam_points.shape[0], 1))
        cam_homogeneous = np.hstack([cam_points, ones])  # (N, 4)
        world_points = (camera_extrinsics @ cam_homogeneous.T).T[:, :3]  # (N, 3)

        return self.from_point_cloud(world_points, robot_pos, robot_yaw, robot_height)
