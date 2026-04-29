"""Unstructured boulder field — mixed D8/D10/D12/D20 polyhedra.

Shape distribution: 25% each of:
- D8  (octahedron)       — 8 triangular faces, 6 vertices
- D10 (trapezohedron)    — 10 kite-shaped faces, 12 vertices
- D12 (dodecahedron)     — 12 pentagonal faces, 20 vertices
- D20 (icosahedron)      — 20 triangular faces, 12 vertices

Zone layout (increasing size and decreasing density):
  Zone 1 (0-10m):  3-5cm edge,   15/m², ~4500 total  — Gravel
  Zone 2 (10-20m): 10-15cm edge,  8/m², ~2400 total  — River rocks
  Zone 3 (20-30m): 25-35cm edge,  4/m², ~1200 total  — Large rocks
  Zone 4 (30-40m): 50-70cm edge,  2/m², ~600 total   — Small boulders
  Zone 5 (40-50m): 80-120cm edge, 1/m², ~300 total   — Large boulders

Implementation: USD trimesh prims (UsdGeom.Mesh) with convexHull collision,
kinematic rigid bodies (immovable). Random SO(3) rotation per obstacle.

Reuses patterns from:
- ARL_DELIVERY/02_Obstacle_Course/spot_obstacle_course.py (USD mesh creation)
- capstone_test.md:306-496 (polyhedron mesh generators)
"""

import numpy as np

from configs.zone_params import ZONE_PARAMS, ARENA_WIDTH, ZONE_LENGTH


# Golden ratio (used by D12 and D20)
PHI = (1.0 + np.sqrt(5.0)) / 2.0


def _unit_verts(verts):
    """Normalize vertices so the longest axis spans 1.0."""
    v = np.array(verts, dtype=np.float32)
    v -= v.mean(axis=0)
    v /= np.abs(v).max()
    return v


def octahedron_mesh(edge_length):
    """Regular octahedron (D8) — 8 triangular faces, 6 vertices."""
    verts = _unit_verts([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
    ]) * edge_length * 0.707

    faces = [
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [0, 3, 5], [3, 1, 5], [1, 2, 5], [2, 0, 5],
    ]
    return verts, faces


def trapezohedron_mesh(edge_length):
    """Pentagonal trapezohedron (D10) — 10 kite faces, 12 vertices."""
    top = [0, 0, 1.0]
    bot = [0, 0, -1.0]

    upper_ring, lower_ring = [], []
    for i in range(5):
        angle_u = 2 * np.pi * i / 5
        angle_l = 2 * np.pi * (i + 0.5) / 5
        upper_ring.append([np.cos(angle_u) * 0.85, np.sin(angle_u) * 0.85, 0.30])
        lower_ring.append([np.cos(angle_l) * 0.85, np.sin(angle_l) * 0.85, -0.30])

    verts = _unit_verts([top] + upper_ring + lower_ring + [bot]) * edge_length * 0.6

    faces = []
    for i in range(5):
        u0 = 1 + i
        u1 = 1 + (i + 1) % 5
        l0 = 6 + i
        l1 = 6 + (i + 1) % 5
        faces.append([0, u0, l0])
        faces.append([0, l0, u1])
        faces.append([11, l0, u0])
        faces.append([11, u1, l0])
    return verts, faces


def dodecahedron_mesh(edge_length):
    """Regular dodecahedron (D12) — 12 pentagonal faces, 20 vertices."""
    raw = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            for s3 in (-1, 1):
                raw.append([s1, s2, s3])
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            raw.append([0, s1 * PHI, s2 / PHI])
            raw.append([s1 / PHI, 0, s2 * PHI])
            raw.append([s1 * PHI, s2 / PHI, 0])

    verts = _unit_verts(raw) * edge_length * 0.75
    faces = _convex_hull_faces(verts)
    return verts, faces


def icosahedron_mesh(edge_length):
    """Regular icosahedron (D20) — 20 triangular faces, 12 vertices."""
    raw = []
    for s in (-1, 1):
        raw.append([0, s, PHI])
        raw.append([0, s, -PHI])
        raw.append([PHI, 0, s])
        raw.append([-PHI, 0, s])
        raw.append([s, PHI, 0])
        raw.append([s, -PHI, 0])

    verts = _unit_verts(raw) * edge_length * 0.525
    faces = _convex_hull_faces(verts)
    return verts, faces


def _convex_hull_faces(verts):
    """Compute triangulated convex hull faces via scipy."""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(verts)
    return hull.simplices.tolist()


SHAPE_GENERATORS = {
    "D8": octahedron_mesh,
    "D10": trapezohedron_mesh,
    "D12": dodecahedron_mesh,
    "D20": icosahedron_mesh,
}

SHAPE_NAMES = list(SHAPE_GENERATORS.keys())


def create_polyhedron(stage, path, shape_name, edge_length, position, rotation):
    """Create a polyhedral trimesh obstacle at the given position.

    Args:
        stage: USD stage
        path: Prim path for the mesh
        shape_name: One of "D8", "D10", "D12", "D20"
        edge_length: Edge length in meters
        position: (x, y, z) world position
        rotation: (rx, ry, rz) euler angles in degrees

    Returns:
        The created mesh prim
    """
    from pxr import UsdGeom, UsdPhysics, Gf

    gen_fn = SHAPE_GENERATORS[shape_name]
    verts, faces = gen_fn(edge_length)

    # Flatten face indices for USD
    face_indices = []
    face_counts = []
    for f in faces:
        face_indices.extend(f)
        face_counts.append(len(f))

    # Create USD mesh
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set(verts.tolist())
    mesh.GetFaceVertexIndicesAttr().Set(face_indices)
    mesh.GetFaceVertexCountsAttr().Set(face_counts)

    # Position and rotation
    xf = UsdGeom.Xformable(mesh.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*position))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))

    # Display color — grey-brown rock colors
    grey = 0.4 + np.random.uniform(-0.1, 0.1)
    mesh.GetDisplayColorAttr().Set([Gf.Vec3f(grey, grey * 0.9, grey * 0.8)])

    # Physics: kinematic rigid body with convex hull collision
    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collision.CreateApproximationAttr("convexHull")

    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateKinematicEnabledAttr(True)

    return mesh


def populate_boulder_zone(stage, zone_path, zone, rng):
    """Scatter boulders within a zone.

    Args:
        stage: USD stage
        zone_path: Parent prim path for this zone
        zone: Zone parameter dict
        rng: numpy RandomState for reproducibility

    Returns:
        int: Number of boulders created
    """
    from pxr import UsdGeom

    UsdGeom.Xform.Define(stage, zone_path)

    count = zone["count"]
    edge_min, edge_max = zone["edge_range"]
    x_start = zone["x_start"]
    x_end = zone["x_end"]

    for b in range(count):
        # Random shape (25% each)
        shape = SHAPE_NAMES[b % 4]

        # Random edge length within range
        edge = rng.uniform(edge_min, edge_max)

        # Random position within zone
        x = rng.uniform(x_start, x_end)
        y = rng.uniform(0, ARENA_WIDTH)
        z = edge * 0.3  # partially embedded in ground

        # Random rotation (SO(3))
        rotation = (
            rng.uniform(0, 360),
            rng.uniform(0, 360),
            rng.uniform(0, 360),
        )

        boulder_path = f"{zone_path}/boulder_{b}"
        create_polyhedron(stage, boulder_path, shape, edge, (x, y, z), rotation)

    return count


def create_boulder_environment(stage, cfg=None):
    """Build the boulder field environment.

    Creates 5 zones with mixed polyhedra of increasing size.

    Args:
        stage: USD stage
        cfg: Optional config (unused, params come from zone_params)
    """
    from pxr import UsdGeom

    root = "/World/BoulderArena"
    UsdGeom.Xform.Define(stage, root)

    zones = ZONE_PARAMS["boulder"]
    rng = np.random.RandomState(42)
    total_boulders = 0

    for zone in zones:
        zone_idx = zone["zone"]
        zone_path = f"{root}/zone_{zone_idx}"
        count = populate_boulder_zone(stage, zone_path, zone, rng)
        total_boulders += count

    print(f"  Boulder environment: {len(zones)} zones, {total_boulders} boulders")
    for z in zones:
        print(f"    Zone {z['zone']}: edge={z['edge_range']}, "
              f"count={z['count']} — {z['label']}")
