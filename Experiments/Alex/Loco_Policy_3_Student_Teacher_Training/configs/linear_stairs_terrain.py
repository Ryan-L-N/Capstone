"""Linear Staircase Terrain — One-directional ascending + descending stairs.

Option B: Steps go UP in +X for the first half, then DOWN for the second half.
Robot spawns at the bottom (X=0), climbs to the peak, then descends.

This directly matches the eval staircase format (linear, not pyramid/radial).
Eliminates the sideways climbing behavior from inverted pyramid terrain.

Usage in terrain_cfgs.py:
    from .linear_stairs_terrain import linear_staircase_terrain, LinearStaircaseTerrainCfg

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""
from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPyramidStairsTerrainCfg
from isaaclab.utils import configclass


@configclass
class LinearStaircaseTerrainCfg(MeshPyramidStairsTerrainCfg):
    """Configuration for a linear staircase terrain (up then down).

    Inherits step_height_range, step_width, platform_width from parent.
    The terrain goes UP for the first half of X, then DOWN for the second half.
    Robot spawns at X=0 (bottom). Forward = climbing.
    """
    function = None  # Will be set below


def linear_staircase_terrain(
    difficulty: float, cfg: LinearStaircaseTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a linear staircase terrain: UP in +X, then DOWN in +X.

    The terrain is a series of rectangular steps spanning the full Y-width,
    ascending from X=0 to X=size[0]/2 (the peak), then descending from
    X=size[0]/2 to X=size[0].

    Args:
        difficulty: 0.0 to 1.0, controls step height.
        cfg: Terrain configuration.

    Returns:
        (meshes, origin) where origin is at the bottom (X=0, Y=center, Z=0).
    """
    # Resolve step height from difficulty
    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    step_width = cfg.step_width
    size_x, size_y = cfg.size

    # Number of steps for ascending half
    half_x = size_x / 2.0
    num_steps_up = int(half_x / step_width)
    if num_steps_up < 1:
        num_steps_up = 1

    meshes = []

    # ── ASCENDING HALF (X = 0 to size_x/2) ───────────────────────────
    for i in range(num_steps_up):
        # Each step is a box spanning full Y width
        x_start = i * step_width
        x_center = x_start + step_width / 2.0
        y_center = size_y / 2.0
        z_top = (i + 1) * step_height
        box_height = z_top  # Box goes from ground (Z=0) to step top

        if box_height <= 0:
            continue

        box_dims = (step_width, size_y, box_height)
        box_pos = (x_center, y_center, box_height / 2.0)
        box = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        meshes.append(box)

    peak_height = num_steps_up * step_height

    # ── PLATFORM AT PEAK ──────────────────────────────────────────────
    platform_width = min(cfg.platform_width, step_width * 2)
    platform_x = half_x + platform_width / 2.0
    platform_dims = (platform_width, size_y, peak_height)
    platform_pos = (platform_x, size_y / 2.0, peak_height / 2.0)
    platform = trimesh.creation.box(
        platform_dims, trimesh.transformations.translation_matrix(platform_pos)
    )
    meshes.append(platform)

    # ── DESCENDING HALF (X = size_x/2 + platform to size_x) ──────────
    descent_start = half_x + platform_width
    num_steps_down = int((size_x - descent_start) / step_width)
    if num_steps_down < 1:
        num_steps_down = 1

    for i in range(num_steps_down):
        x_start = descent_start + i * step_width
        x_center = x_start + step_width / 2.0
        y_center = size_y / 2.0
        # Descending: height decreases from peak
        z_top = peak_height - (i + 1) * step_height
        if z_top <= 0:
            break

        box_dims = (step_width, size_y, z_top)
        box_pos = (x_center, y_center, z_top / 2.0)
        box = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        meshes.append(box)

    # ── GROUND PLANE (base) ───────────────────────────────────────────
    ground_dims = (size_x, size_y, 0.02)
    ground_pos = (size_x / 2.0, size_y / 2.0, -0.01)
    ground = trimesh.creation.box(
        ground_dims, trimesh.transformations.translation_matrix(ground_pos)
    )
    meshes.append(ground)

    # ── ORIGIN: spawn at bottom-left, ground level ────────────────────
    # Robot spawns at X=0, Y=center, Z=step_height (on the first step)
    origin = np.array([step_width / 2.0, size_y / 2.0, step_height])

    return meshes, origin


# Set the function on the config class
LinearStaircaseTerrainCfg.function = linear_staircase_terrain
