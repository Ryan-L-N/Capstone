"""Linear Staircase Heightfields — Ascending and Descending stairs.

Returns 2D numpy height arrays (like all HfTerrainBaseCfg functions).
  - linear_stairs_terrain:      Steps go UP in +X. Spawn at X=0 (ground).
  - linear_stairs_down_terrain: Steps go DOWN in +X. Spawn at X=0 (elevated).

Difficulty controls step height (3cm to 22cm).
"""
from __future__ import annotations

import numpy as np

from isaaclab.terrains.height_field import hf_terrains_cfg


def linear_stairs_terrain(difficulty: float, cfg: hf_terrains_cfg.HfTerrainBaseCfg) -> np.ndarray:
    """Generate a one-directional ascending staircase heightfield.

    Steps go UP in +X direction. The robot spawns at X=0 (ground level)
    and must walk forward to climb. No descent, no flat top, no pyramid.

    Args:
        difficulty: 0.0 to 1.0, controls step height.
        cfg: HfTerrainBaseCfg with size, horizontal_scale, vertical_scale.

    Returns:
        2D numpy height array, shape (num_x, num_y), discretized heights.
    """
    # Terrain grid dimensions
    width = int(cfg.size[0] / cfg.horizontal_scale)
    length = int(cfg.size[1] / cfg.horizontal_scale)

    # Step parameters
    step_height_min = 0.03  # 3cm (zone 1)
    step_height_max = 0.22  # 22cm (zone 5)
    step_height = step_height_min + difficulty * (step_height_max - step_height_min)
    step_width_m = 0.30  # 30cm tread depth (matching eval)
    step_width_cells = max(1, int(step_width_m / cfg.horizontal_scale))

    # Platform at start (spawn area)
    platform_width_m = getattr(cfg, 'platform_width', 2.0)
    platform_cells = max(1, int(platform_width_m / cfg.horizontal_scale))

    # Discretize height
    height_discrete = step_height / cfg.vertical_scale

    # Build height array
    hf = np.zeros((width, length), dtype=np.int16)

    # Fill with ascending steps after the platform
    current_height = 0
    x = platform_cells
    step_num = 0

    while x < width:
        x_end = min(x + step_width_cells, width)
        step_num += 1
        current_height = int(step_num * height_discrete)
        hf[x:x_end, :] = current_height
        x = x_end

    return hf


def linear_stairs_down_terrain(difficulty: float, cfg: hf_terrains_cfg.HfTerrainBaseCfg) -> np.ndarray:
    """Generate a one-directional descending staircase heightfield.

    Steps go DOWN in +X direction. The robot spawns at X=0 (elevated)
    and must walk forward to descend. Mirror of linear_stairs_terrain.

    Args:
        difficulty: 0.0 to 1.0, controls step height.
        cfg: HfTerrainBaseCfg with size, horizontal_scale, vertical_scale.

    Returns:
        2D numpy height array, shape (num_x, num_y), discretized heights.
    """
    # Generate ascending stairs then flip along X axis
    hf_up = linear_stairs_terrain(difficulty, cfg)
    max_height = hf_up.max()
    # Invert: highest point becomes ground, ground becomes highest
    hf_down = max_height - hf_up
    return hf_down.astype(np.int16)
