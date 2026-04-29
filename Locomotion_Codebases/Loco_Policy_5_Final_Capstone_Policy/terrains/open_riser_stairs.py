"""Open-riser pyramid-stair sub-terrain for Phase-10.

Mirrors Isaac Lab's `MeshPyramidStairsTerrainCfg` 4-way symmetric layout but
replaces the solid stepped boxes with thin tread slabs at each riser height.
Empty space (gaps) between treads — no vertical riser face for the policy
to push against. Approximates Final World architectural staircases whose
imported USDs lack riser collision.

Reference (Isaac Lab):
- Cfg pattern:   isaaclab.terrains.trimesh.mesh_terrains_cfg.MeshPyramidStairsTerrainCfg
- Mesh fn:       isaaclab.terrains.trimesh.mesh_terrains.pyramid_stairs_terrain
                 (lines 51-148; we replace lines 101-134 — the box-ring inner
                 body — with thin slabs centered tread_thickness below each
                 step's top surface)
"""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.utils import configclass


def open_riser_stairs_terrain(
    difficulty: float, cfg: "MeshOpenRiserStairsTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a pyramid stair pattern with open risers (treads only)."""

    # interp riser height from difficulty
    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    # clamp tread thickness so there is always at least 50% gap
    tread_thickness = min(cfg.tread_thickness, 0.5 * step_height)

    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (
        2 * cfg.step_width
    ) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (
        2 * cfg.step_width
    ) + 1
    num_steps = int(min(num_steps_x, num_steps_y))

    meshes_list: list[trimesh.Trimesh] = []

    # solid border (spawn / approach lives here at z = -step_height/2)
    if cfg.border_width > 0.0:
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -step_height / 2.0]
        border_inner_size = (
            cfg.size[0] - 2 * cfg.border_width,
            cfg.size[1] - 2 * cfg.border_width,
        )
        meshes_list += make_border(cfg.size, border_inner_size, step_height, border_center)

    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (
        cfg.size[0] - 2 * cfg.border_width,
        cfg.size[1] - 2 * cfg.border_width,
    )

    # generate the stair pattern — only top tread slabs, no risers
    for k in range(num_steps):
        box_size = (
            terrain_size[0] - 2 * k * cfg.step_width,
            terrain_size[1] - 2 * k * cfg.step_width,
        )
        # tread top is at (k+1) * step_height; slab is centered tread_thickness/2 below
        tread_top_z = (k + 1) * step_height
        slab_z = tread_top_z - tread_thickness / 2.0
        box_offset = (k + 0.5) * cfg.step_width

        # top
        box_dims = (box_size[0], cfg.step_width, tread_thickness)
        box_pos = (
            terrain_center[0],
            terrain_center[1] + terrain_size[1] / 2.0 - box_offset,
            slab_z,
        )
        meshes_list.append(
            trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        )
        # bottom
        box_pos = (
            terrain_center[0],
            terrain_center[1] - terrain_size[1] / 2.0 + box_offset,
            slab_z,
        )
        meshes_list.append(
            trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        )
        # right / left — match pyramid_stairs by trimming step_width on both ends
        # so the corners are not double-counted
        box_dims = (cfg.step_width, box_size[1] - 2 * cfg.step_width, tread_thickness)
        box_pos = (
            terrain_center[0] + terrain_size[0] / 2.0 - box_offset,
            terrain_center[1],
            slab_z,
        )
        meshes_list.append(
            trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        )
        box_pos = (
            terrain_center[0] - terrain_size[0] / 2.0 + box_offset,
            terrain_center[1],
            slab_z,
        )
        meshes_list.append(
            trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        )

    # central platform — solid pillar from -step_height/2 up to top, like pyramid_stairs
    platform_dims = (
        terrain_size[0] - 2 * num_steps * cfg.step_width,
        terrain_size[1] - 2 * num_steps * cfg.step_width,
        (num_steps + 2) * step_height,
    )
    platform_pos = (
        terrain_center[0],
        terrain_center[1],
        terrain_center[2] + num_steps * step_height / 2.0,
    )
    meshes_list.append(
        trimesh.creation.box(
            platform_dims, trimesh.transformations.translation_matrix(platform_pos)
        )
    )

    # spawn / origin — same convention as pyramid_stairs (top of central platform)
    origin = np.array([terrain_center[0], terrain_center[1], (num_steps + 1) * step_height])
    return meshes_list, origin


@configclass
class MeshOpenRiserStairsTerrainCfg(SubTerrainBaseCfg):
    """Pyramid-stair-style terrain with open risers (treads only, gaps between).

    Approximates architectural staircases whose imported USDs ship with
    only horizontal tread collision — no vertical riser faces. Forces the
    policy to land precisely on tread surfaces without relying on the
    riser face for upward propulsion.
    """

    function = open_riser_stairs_terrain

    border_width: float = 0.0
    """Width of the solid border around the terrain (m). The spawn approach lives here."""

    step_height_range: tuple[float, float] = MISSING
    """[min, max] riser height (m). Same convention as MeshPyramidStairsTerrainCfg."""

    step_width: float = MISSING
    """Tread depth (m). Same convention as MeshPyramidStairsTerrainCfg."""

    platform_width: float = 1.0
    """Square central platform size (m). Solid pillar — gives the policy somewhere to stand."""

    tread_thickness: float = 0.04
    """Vertical thickness of each tread slab (m). Auto-clamped to step_height/2 so the
    gap between treads is always at least 50% of the riser height."""
