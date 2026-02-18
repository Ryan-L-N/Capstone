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
"""

# TODO: Implementation
# - Mesh generators: octahedron_mesh(), trapezohedron_mesh(), dodecahedron_mesh(), icosahedron_mesh()
# - _convex_hull_faces(verts) helper
# - create_polyhedron(stage, path, shape_name, edge_length, position, rotation)
# - populate_boulder_zone(stage, zone_idx, x_start, edge_range, density, seed)
# - create_boulder_environment(stage, cfg)
# - SHAPE_GENERATORS = {"D8": octahedron_mesh, "D10": trapezohedron_mesh, ...}


def create_boulder_environment(stage, cfg):
    """Build the boulder field environment."""
    raise NotImplementedError("TODO: Implement boulder environment builder")
