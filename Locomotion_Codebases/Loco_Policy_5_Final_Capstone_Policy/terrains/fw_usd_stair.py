"""FW USD staircase sub-terrain for Phase-v5.

Loads Colby's risered SM_Staircase_*.usd files as training terrain.
Pattern matches `open_riser_stairs.py` — a function `(difficulty, cfg)
-> (list[trimesh], origin)` that Isaac Lab's terrain generator calls
per patch.

Key facts about Colby's USDs (per
`Experiments/Colby/FW_Stairs_Riser_Project/usd_source/`):
- metersPerUnit = 0.01 (cm units in USD; values must be scaled by 0.01
  to get meters)
- up_axis = Z
- 4 USDs by topology + height:
  - SM_StaircaseHalf_02: small switchback, ~3.0m rise (easy)
  - SM_StaircaseHalf_01: half-height wide straight, ~3.0m rise
  - SM_Staircase_02: full straight, ~5.3m rise (medium)
  - SM_Staircase_01: full switchback, ~5.3m rise (hardest)
- Colby's `add_risers.py` (Apr 30) added solid riser triangles to fix
  the foot-in-gap failure mode.
"""

from __future__ import annotations

import os
from dataclasses import MISSING, field

import numpy as np
import trimesh

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.utils import configclass


# ---- Module-level cache (USD load is expensive; do it once per file) ----
_USD_TRIMESH_CACHE: dict[str, trimesh.Trimesh] = {}


def _usd_to_trimesh(usd_path: str, mpu_scale: float = 0.01) -> trimesh.Trimesh:
    """Convert a USD file's mesh geometry into a single trimesh.Trimesh.

    Walks all Mesh prims in the stage, applies their world transforms,
    triangulates faces, scales to meters via `mpu_scale` (default 0.01
    = cm-to-m).

    Imports pxr lazily because this module is imported during cfg init
    (before SimulationApp may be fully started). The function itself is
    only called inside the terrain generator, where Isaac Sim is up.
    """
    if usd_path in _USD_TRIMESH_CACHE:
        return _USD_TRIMESH_CACHE[usd_path]

    from pxr import Usd, UsdGeom, Gf  # lazy

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Could not open USD: {usd_path}")

    all_pts: list[list[float]] = []
    all_faces: list[list[int]] = []

    xform_cache = UsdGeom.XformCache()

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts_attr = mesh.GetPointsAttr().Get()
        fc_attr = mesh.GetFaceVertexCountsAttr().Get()
        fi_attr = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts_attr or not fc_attr or not fi_attr:
            continue

        # World transform for this prim
        world_xform = xform_cache.GetLocalToWorldTransform(prim)

        offset = len(all_pts)
        for p in pts_attr:
            tp = world_xform.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
            all_pts.append([float(tp[0]) * mpu_scale,
                            float(tp[1]) * mpu_scale,
                            float(tp[2]) * mpu_scale])

        cursor = 0
        for count in fc_attr:
            face = list(fi_attr[cursor:cursor + count])
            cursor += count
            if count < 3:
                continue
            elif count == 3:
                all_faces.append([face[0] + offset, face[1] + offset, face[2] + offset])
            else:
                # Fan triangulation for quads + n-gons
                for i in range(1, count - 1):
                    all_faces.append([face[0] + offset,
                                      face[i] + offset,
                                      face[i + 1] + offset])

    if not all_pts or not all_faces:
        raise RuntimeError(f"USD has no mesh data: {usd_path}")

    mesh = trimesh.Trimesh(
        vertices=np.array(all_pts, dtype=np.float64),
        faces=np.array(all_faces, dtype=np.int64),
        process=False,  # don't merge/dedupe — preserve Colby's riser triangles
    )
    _USD_TRIMESH_CACHE[usd_path] = mesh
    return mesh


def fw_usd_stair_terrain(
    difficulty: float, cfg: "FwUsdStairTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Place an FW USD staircase in the 8x8m patch.

    Difficulty selects one of the 4 USDs (easy → hard). The selected
    USD's geometry is centered in the patch with its bbox.min lifted
    to z=0 so Spot can spawn on the floor in front of the stairs.

    Returns ([staircase mesh, surrounding flat platform], origin) where
    origin is the spawn position 0.8m in front of the stair's
    front-bottom edge.
    """
    # Pick USD by difficulty bucket
    n = len(cfg.usd_paths)
    idx = min(int(difficulty * n), n - 1)
    usd_path = cfg.usd_paths[idx]
    if not os.path.isfile(usd_path):
        raise FileNotFoundError(f"FW USD not found: {usd_path}")

    stair_mesh = _usd_to_trimesh(usd_path).copy()

    # Lift the mesh so bbox min z = 0
    bbox = stair_mesh.bounds  # shape (2, 3) — [min, max]
    z_lift = -bbox[0][2]  # bring min z to 0
    stair_mesh.apply_translation([0.0, 0.0, z_lift])

    # Center horizontally in the patch
    center_x = cfg.size[0] * 0.5
    center_y = cfg.size[1] * 0.5
    bbox_after_lift = stair_mesh.bounds
    cx_curr = (bbox_after_lift[0][0] + bbox_after_lift[1][0]) * 0.5
    cy_curr = (bbox_after_lift[0][1] + bbox_after_lift[1][1]) * 0.5
    stair_mesh.apply_translation([center_x - cx_curr, center_y - cy_curr, 0.0])

    # Build a flat platform underneath that fills the patch around the stair
    # so Spot has somewhere stable to spawn.
    platform_thickness = 0.05
    platform = trimesh.creation.box(
        extents=(cfg.size[0], cfg.size[1], platform_thickness),
    )
    platform.apply_translation([center_x, center_y, -platform_thickness * 0.5])

    meshes = [stair_mesh, platform]

    # Spawn 0.8m in front of the stair's front-bottom edge.
    # Use the post-lift bbox to find front (max x for the FW USD orientation —
    # the SM_Staircase_*.usd files have bottom_surface near +x, top near -x.
    bbox_final = stair_mesh.bounds
    front_x = bbox_final[1][0]  # max x
    spawn_x = min(front_x + 0.8, cfg.size[0] - 0.5)
    origin = np.array([spawn_x, center_y, 0.55])  # 0.55m above floor (Spot height)
    return meshes, origin


@configclass
class FwUsdStairTerrainCfg(SubTerrainBaseCfg):
    """Sub-terrain that loads Colby's risered FW staircase USDs.

    The 4 USDs map to difficulty levels:
      0.00 → SM_StaircaseHalf_02 (small switchback, easiest)
      0.33 → SM_StaircaseHalf_01 (half-height wide straight)
      0.66 → SM_Staircase_02 (full straight, 5.3m rise)
      1.00 → SM_Staircase_01 (full switchback, hardest)
    """

    function = fw_usd_stair_terrain

    usd_paths: list = MISSING  # type: ignore
    """Absolute paths to the FW staircase USDs in difficulty order."""
