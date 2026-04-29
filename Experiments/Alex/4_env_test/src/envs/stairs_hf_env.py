"""Heightfield staircase environment — same 5-zone layout as stairs_env but using
heightfield mesh geometry instead of USD cubes.

Matches the training terrain collision behavior (smooth heightfield edges vs sharp
cube edges). Use this to evaluate policies trained on heightfield stairs.

Zone layout identical to stairs_env:
  Zone 1 (0-10m):  3cm steps  — Access ramp
  Zone 2 (10-20m): 8cm steps  — Low residential
  Zone 3 (20-30m): 13cm steps — Standard residential
  Zone 4 (30-40m): 18cm steps — Steep commercial
  Zone 5 (40-50m): 23cm steps — Maximum challenge
"""

import numpy as np
from configs.zone_params import ZONE_PARAMS, ARENA_WIDTH, ZONE_LENGTH, TRANSITION_STEPS


def _build_stair_heightfield(zones, horizontal_scale=0.05, vertical_scale=0.005):
    """Build a heightfield array for the entire 5-zone staircase.

    Args:
        zones: List of zone dicts from ZONE_PARAMS["stairs"]
        horizontal_scale: XY resolution in meters (0.05m = 5cm)
        vertical_scale: Z resolution in meters

    Returns:
        heights: 2D numpy array of heights
        horizontal_scale: XY scale
        vertical_scale: Z scale
    """
    total_x = ZONE_LENGTH * len(zones)  # 50m
    nx = int(total_x / horizontal_scale)
    ny = int(ARENA_WIDTH / horizontal_scale)

    heights = np.zeros((nx, ny), dtype=np.float32)

    cumulative_z = 0.0
    prev_step_height = None

    for zone in zones:
        step_height = zone["step_height"]
        step_depth = zone["step_depth"]
        num_steps = zone["num_steps"]
        x_start = zone["x_start"]

        for s in range(num_steps):
            # Transition ramp at zone boundaries
            if prev_step_height is not None and s < TRANSITION_STEPS:
                frac = (s + 1) / (TRANSITION_STEPS + 1)
                riser = prev_step_height + (step_height - prev_step_height) * frac
            else:
                riser = step_height

            cumulative_z += riser

            # Pixel range for this step
            x_px_start = int((x_start + s * step_depth) / horizontal_scale)
            x_px_end = int((x_start + (s + 1) * step_depth) / horizontal_scale)
            x_px_start = max(0, min(x_px_start, nx))
            x_px_end = max(0, min(x_px_end, nx))

            heights[x_px_start:x_px_end, :] = cumulative_z

        # Fill gap at end of zone
        gap_start = int((x_start + num_steps * step_depth) / horizontal_scale)
        gap_end = int((x_start + ZONE_LENGTH) / horizontal_scale)
        gap_start = max(0, min(gap_start, nx))
        gap_end = max(0, min(gap_end, nx))
        if gap_start < gap_end:
            heights[gap_start:gap_end, :] = cumulative_z

        prev_step_height = step_height

    return heights, horizontal_scale, vertical_scale


def _heightfield_to_trimesh(heights, horizontal_scale, vertical_scale):
    """Convert a 2D heightfield to triangle mesh vertices and triangles.

    Args:
        heights: 2D array of height values (in meters, NOT discretized)
        horizontal_scale: XY spacing in meters
        vertical_scale: unused (heights already in meters)

    Returns:
        vertices: (N, 3) float array
        triangles: (M, 3) int array of vertex indices
    """
    nx, ny = heights.shape
    vertices = np.zeros((nx * ny, 3), dtype=np.float32)

    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            vertices[idx, 0] = i * horizontal_scale
            vertices[idx, 1] = j * horizontal_scale
            vertices[idx, 2] = heights[i, j]

    # Two triangles per grid cell
    triangles = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            v0 = i * ny + j
            v1 = v0 + 1
            v2 = (i + 1) * ny + j
            v3 = v2 + 1
            triangles.append([v0, v2, v1])
            triangles.append([v1, v2, v3])

    return vertices, np.array(triangles, dtype=np.int32)


def create_stairs_hf_environment(stage, cfg=None):
    """Build the staircase environment using heightfield mesh.

    Same 5-zone layout as create_stairs_environment but using a single
    triangle mesh instead of stacked USD cubes. This matches the collision
    behavior of training terrain (heightfield-based).
    """
    from pxr import UsdGeom, UsdPhysics, Gf, Vt, UsdShade, PhysxSchema

    from envs.base_arena import disable_default_ground
    disable_default_ground(stage)

    zones = ZONE_PARAMS["stairs"]

    # Build heightfield
    print("  Building heightfield staircase...", flush=True)
    heights, h_scale, v_scale = _build_stair_heightfield(zones, horizontal_scale=0.05)

    # Convert to triangle mesh
    vertices, triangles = _heightfield_to_trimesh(heights, h_scale, v_scale)

    print(f"    Mesh: {len(vertices)} vertices, {len(triangles)} triangles", flush=True)
    print(f"    Grid: {heights.shape[0]}x{heights.shape[1]} at {h_scale}m resolution", flush=True)

    # Create USD mesh
    mesh_path = "/World/Staircase/mesh"
    UsdGeom.Xform.Define(stage, "/World/Staircase")
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)

    # Set mesh data
    points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))

    face_vertex_counts = [3] * len(triangles)
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_vertex_counts))

    face_vertex_indices = triangles.flatten().tolist()
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_vertex_indices))

    # Color — grey concrete
    mesh.GetDisplayColorAttr().Set([Gf.Vec3f(0.55, 0.55, 0.52)])

    # Physics collision
    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collision.CreateApproximationAttr().Set("meshSimplification")

    # Physics material
    mat_path = "/World/Staircase/StairMat"
    UsdShade.Material.Define(stage, mat_path)
    mat_prim = stage.GetPrimAtPath(mat_path)
    phys_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
    phys_mat.CreateStaticFrictionAttr(1.0)
    phys_mat.CreateDynamicFrictionAttr(1.0)
    phys_mat.CreateRestitutionAttr(0.01)
    physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
    physx_mat.CreateFrictionCombineModeAttr().Set("multiply")

    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(UsdShade.Material.Get(stage, mat_path))

    # Print zone info (same as stairs_env)
    print(f"  Stairs HF environment: {len(zones)} zones", flush=True)
    base = 0.0
    for z in zones:
        rise = z["step_height"] * z["num_steps"]
        base += rise
        print(f"    Zone {z['zone']}: {z['step_height']}m steps × {z['num_steps']} "
              f"= {rise:.2f}m rise — {z['label']}", flush=True)
    print(f"    Total elevation gain: {base:.2f}m", flush=True)
