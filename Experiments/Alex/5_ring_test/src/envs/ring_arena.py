"""Circular ring arena builder — 5 concentric terrain rings.

Builds a 50m-radius circular arena with 5 rings of increasing difficulty:
  Ring 1 (0-10m)  — Flat ground, normal friction
  Ring 2 (10-20m) — Low friction surface
  Ring 3 (20-30m) — Vegetation stalks + drag
  Ring 4 (30-40m) — Boulder field
  Ring 5 (40-50m) — Extreme mixed (low friction + large boulders)

Ground construction: Each ring = 36 trapezoidal cube segments (10-deg arcs)
with per-ring physics material. Default ground plane kept underneath as
collision failsafe.

Transition corridor: 3m-wide clear path at angle=0 between rings (no obstacles).
Containment wall: 36-segment outer boundary at r=51m.
"""

import numpy as np

from configs.ring_params import (
    ARENA_RADIUS, NUM_RINGS, NUM_GROUND_SEGMENTS, RING_PARAMS,
)
from envs.boulder_meshes import create_polyhedron, SHAPE_NAMES
from envs.vegetation import create_stalks_polar


# Corridor clear zone: angle=0, half-width=1.5m
CORRIDOR_ANGLE = 0.0
CORRIDOR_HALF_WIDTH = 1.5


def _create_ring_material(stage, name, mu_s, mu_d):
    """Create a physics material with specified friction.

    Args:
        stage: USD stage
        name: Material name
        mu_s: Static friction coefficient
        mu_d: Dynamic friction coefficient

    Returns:
        str: Material prim path
    """
    from pxr import UsdShade, UsdPhysics, UsdGeom, PhysxSchema

    parent = "/World/Physics/Materials"
    if not stage.GetPrimAtPath("/World/Physics").IsValid():
        UsdGeom.Xform.Define(stage, "/World/Physics")
    if not stage.GetPrimAtPath(parent).IsValid():
        UsdGeom.Xform.Define(stage, parent)

    mat_path = f"{parent}/{name}"
    UsdShade.Material.Define(stage, mat_path)

    prim = stage.GetPrimAtPath(mat_path)
    phys_mat = UsdPhysics.MaterialAPI.Apply(prim)
    phys_mat.CreateStaticFrictionAttr(mu_s)
    phys_mat.CreateDynamicFrictionAttr(mu_d)
    phys_mat.CreateRestitutionAttr(0.01)

    physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(prim)
    physx_mat.CreateFrictionCombineModeAttr().Set("multiply")
    physx_mat.CreateRestitutionCombineModeAttr().Set("min")

    return mat_path


def _create_ring_segments(stage, ring_params, mat_path):
    """Create 36 ground cube segments for one ring.

    Each segment is a thin cube positioned and scaled to approximate a
    trapezoidal arc section of the annular ring.

    Args:
        stage: USD stage
        ring_params: Dict with ring specs (r_inner, r_outer, color, etc.)
        mat_path: Physics material prim path
    """
    from pxr import UsdGeom, UsdPhysics, UsdShade, Gf

    ring_num = ring_params["ring"]
    r_inner = ring_params["r_inner"]
    r_outer = ring_params["r_outer"]
    color = ring_params["color"]

    ring_path = f"/World/RingArena/ring_{ring_num}"
    UsdGeom.Xform.Define(stage, ring_path)

    angle_step = 2.0 * np.pi / NUM_GROUND_SEGMENTS  # 10 degrees
    r_mid = (r_inner + r_outer) / 2.0
    width = r_outer - r_inner

    # For ring 1 (r_inner=0), use a disk approximation
    if r_inner < 0.1:
        r_inner_eff = 0.0
    else:
        r_inner_eff = r_inner

    for seg in range(NUM_GROUND_SEGMENTS):
        angle_center = angle_step * (seg + 0.5)

        # Segment center position
        cx = r_mid * np.cos(angle_center)
        cy = r_mid * np.sin(angle_center)
        cz = 0.002  # slightly above ground plane failsafe

        # Segment dimensions: approximate trapezoid as a rectangle
        # Length along arc = r_mid * angle_step
        seg_length = r_mid * angle_step
        seg_width = width

        # For the inner ring (r=0 to 10), segments near center need special sizing
        if r_inner_eff < 0.1:
            # For ring 1, make the inner edge a small triangle-ish shape
            seg_length = max(seg_length, 1.0)

        seg_path = f"{ring_path}/seg_{seg}"
        cube = UsdGeom.Cube.Define(stage, seg_path)
        cube.GetSizeAttr().Set(1.0)

        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz))
        # Rotate to align with arc tangent
        rot_deg = np.degrees(angle_center)
        xf.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, rot_deg))
        # Scale: X = radial width, Y = arc length, Z = thin slab
        xf.AddScaleOp().Set(Gf.Vec3d(seg_width, seg_length, 0.004))

        cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

        # Physics collision + material binding
        prim = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(UsdShade.Material.Get(stage, mat_path))


def _scatter_boulders_polar(stage, ring_params, rng):
    """Scatter boulders in an annular region using polar coordinates.

    Args:
        stage: USD stage
        ring_params: Dict with ring specs
        rng: numpy RandomState

    Returns:
        int: Number of boulders created
    """
    from pxr import UsdGeom

    ring_num = ring_params["ring"]
    r_inner = ring_params["r_inner"]
    r_outer = ring_params["r_outer"]
    obs = ring_params["obstacles"]

    if obs is None or obs["type"] != "boulders":
        return 0

    density = obs["density"]
    edge_min, edge_max = obs["edge_range"]

    area = np.pi * (r_outer**2 - r_inner**2)
    count = int(density * area)

    boulder_path = f"/World/RingArena/boulders_{ring_num}"
    UsdGeom.Xform.Define(stage, boulder_path)

    created = 0
    for b in range(count):
        # Uniform random in annular region
        r = np.sqrt(rng.uniform(r_inner**2, r_outer**2))
        theta = rng.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Skip corridor region
        angle_diff = np.abs(np.arctan2(np.sin(theta - CORRIDOR_ANGLE),
                                       np.cos(theta - CORRIDOR_ANGLE)))
        corridor_arc = CORRIDOR_HALF_WIDTH / max(r, 0.1)
        if angle_diff < corridor_arc:
            continue

        edge = rng.uniform(edge_min, edge_max)
        z = edge * 0.3  # partially embedded

        # Random shape (25% each) and rotation
        shape = SHAPE_NAMES[created % 4]
        rotation = (
            rng.uniform(0, 360),
            rng.uniform(0, 360),
            rng.uniform(0, 360),
        )

        path = f"{boulder_path}/boulder_{created}"
        create_polyhedron(stage, path, shape, edge, (x, y, z), rotation)
        created += 1

    return created


def _create_containment_wall(stage, radius=51.0, height=2.0, thickness=0.5):
    """Create outer boundary wall as 36 segments.

    Args:
        stage: USD stage
        radius: Wall center radius
        height: Wall height in meters
        thickness: Wall thickness in meters
    """
    from pxr import UsdGeom, UsdPhysics, Gf

    wall_path = "/World/RingArena/wall"
    UsdGeom.Xform.Define(stage, wall_path)

    angle_step = 2.0 * np.pi / NUM_GROUND_SEGMENTS
    arc_length = radius * angle_step

    for seg in range(NUM_GROUND_SEGMENTS):
        angle_center = angle_step * (seg + 0.5)
        cx = radius * np.cos(angle_center)
        cy = radius * np.sin(angle_center)

        seg_path = f"{wall_path}/seg_{seg}"
        cube = UsdGeom.Cube.Define(stage, seg_path)
        cube.GetSizeAttr().Set(1.0)

        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, height / 2.0))
        rot_deg = np.degrees(angle_center)
        xf.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, rot_deg))
        xf.AddScaleOp().Set(Gf.Vec3d(thickness, arc_length, height))

        cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.3, 0.3, 0.3)])

        prim = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        rb.CreateKinematicEnabledAttr(True)


def _create_ring_markers(stage):
    """Create thin colored rings at ring boundaries for visual clarity.

    Args:
        stage: USD stage
    """
    from pxr import UsdGeom, Gf

    marker_path = "/World/RingArena/markers"
    UsdGeom.Xform.Define(stage, marker_path)

    # Ring boundary radii: 10, 20, 30, 40
    boundary_radii = [10.0, 20.0, 30.0, 40.0]
    colors = [
        (1.0, 1.0, 0.0),   # yellow
        (0.0, 1.0, 0.0),   # green
        (1.0, 0.5, 0.0),   # orange
        (1.0, 0.0, 0.0),   # red
    ]

    angle_step = 2.0 * np.pi / 72  # 5-degree segments for smooth circle

    for r_idx, (radius, color) in enumerate(zip(boundary_radii, colors)):
        ring_marker_path = f"{marker_path}/boundary_{r_idx + 1}"
        UsdGeom.Xform.Define(stage, ring_marker_path)

        arc_length = radius * angle_step
        for seg in range(72):
            angle = angle_step * (seg + 0.5)
            cx = radius * np.cos(angle)
            cy = radius * np.sin(angle)

            seg_path = f"{ring_marker_path}/seg_{seg}"
            cube = UsdGeom.Cube.Define(stage, seg_path)
            cube.GetSizeAttr().Set(1.0)

            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, 0.01))
            rot_deg = np.degrees(angle)
            xf.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, rot_deg))
            xf.AddScaleOp().Set(Gf.Vec3d(0.15, arc_length, 0.002))

            cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])


def create_ring_arena(stage):
    """Build the complete 5-ring circular arena.

    Creates ring ground segments with per-ring friction, scatters obstacles
    for rings 3-5, adds containment wall and ring boundary markers.

    Args:
        stage: USD stage

    Returns:
        dict: Summary of what was created
    """
    from pxr import UsdGeom

    root = "/World/RingArena"
    UsdGeom.Xform.Define(stage, root)

    rng = np.random.RandomState(42)
    summary = {"rings": [], "total_boulders": 0, "total_stalks": 0}

    for ring_params in RING_PARAMS:
        ring_num = ring_params["ring"]
        print(f"  Building ring {ring_num}: {ring_params['label']}...")

        # Create physics material
        mat_name = f"RingMat_{ring_num}"
        mat_path = _create_ring_material(
            stage, mat_name,
            ring_params["mu_static"],
            ring_params["mu_dynamic"],
        )

        # Create ground segments
        _create_ring_segments(stage, ring_params, mat_path)

        # Scatter obstacles
        num_boulders = 0
        num_stalks = 0

        if ring_params["obstacles"] is not None:
            obs_type = ring_params["obstacles"]["type"]

            if obs_type == "boulders":
                num_boulders = _scatter_boulders_polar(stage, ring_params, rng)
                summary["total_boulders"] += num_boulders

            elif obs_type == "stalks":
                stalk_parent = f"{root}/stalks_{ring_num}"
                num_stalks = create_stalks_polar(
                    stage, stalk_parent,
                    ring_params["r_inner"], ring_params["r_outer"],
                    ring_params["obstacles"]["density"],
                    ring_params["obstacles"]["height_range"],
                    rng,
                    corridor_angle=CORRIDOR_ANGLE,
                    corridor_half_width=CORRIDOR_HALF_WIDTH,
                )
                summary["total_stalks"] += num_stalks

        ring_info = {
            "ring": ring_num,
            "label": ring_params["label"],
            "mu_s": ring_params["mu_static"],
            "mu_d": ring_params["mu_dynamic"],
            "boulders": num_boulders,
            "stalks": num_stalks,
        }
        summary["rings"].append(ring_info)
        print(f"    mu_s={ring_params['mu_static']}, mu_d={ring_params['mu_dynamic']}, "
              f"boulders={num_boulders}, stalks={num_stalks}")

    # Containment wall
    print("  Building containment wall...")
    _create_containment_wall(stage)

    # Ring boundary markers
    print("  Adding ring boundary markers...")
    _create_ring_markers(stage)

    print(f"\n  Ring arena complete: {summary['total_boulders']} boulders, "
          f"{summary['total_stalks']} stalks")

    return summary
