"""4-quadrant arena builder — Friction, Grass, Boulders, Stairs sectors.

Builds a 50m-radius circular arena divided into 4 quadrant sectors (90 deg each):
  Q1 (0-90°):   Friction — decreasing mu outward (5 levels)
  Q2 (90-180°): Grass — increasing stalk density + drag (5 levels)
  Q3 (180-270°):Boulders — increasing size + density (5 levels)
  Q4 (270-360°):Stairs — pyramid waypoints of increasing height (5 levels)

Ground construction: Each level within each quadrant = 9 trapezoidal cube segments
(10-deg arcs) with per-level physics material.

Containment wall at r=51m, quadrant divider lines, level boundary markers.
"""

import numpy as np

from configs.ring_params import (
    ARENA_RADIUS, NUM_QUADRANTS, NUM_LEVELS, NUM_GROUND_SEGMENTS,
    LEVEL_WIDTH, QUADRANT_DEFS, QUADRANT_LEVELS,
    FRICTION_LEVELS, GRASS_LEVELS, BOULDER_LEVELS, STAIRS_LEVELS,
    level_radius_range, level_midpoint_radius, WPS_PER_LEVEL,
)
from envs.boulder_meshes import create_polyhedron, SHAPE_NAMES
from envs.vegetation import create_stalks_polar
from envs.stair_pyramids import create_stair_pyramid


def _create_material(stage, name, mu_s, mu_d):
    """Create a physics material with specified friction."""
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


def _create_sector_ground(stage, quad_idx, level_idx, mat_path, color):
    """Create ground segments for one sector (one level within one quadrant).

    Each sector = 9 cube segments spanning 10 degrees each within the
    quadrant's 90-degree arc.
    """
    from pxr import UsdGeom, UsdPhysics, UsdShade, Gf

    qdef = QUADRANT_DEFS[quad_idx]
    qname = qdef["name"]
    r_inner, r_outer = level_radius_range(level_idx)
    r_mid = (r_inner + r_outer) / 2.0
    width = r_outer - r_inner

    sector_path = f"/World/Arena/{qname}/ground_L{level_idx + 1}"
    UsdGeom.Xform.Define(stage, sector_path)

    angle_start = qdef["angle_start"]
    angle_span = (qdef["angle_end"] - qdef["angle_start"]) / NUM_GROUND_SEGMENTS

    for seg in range(NUM_GROUND_SEGMENTS):
        angle_center = angle_start + angle_span * (seg + 0.5)
        cx = r_mid * np.cos(angle_center)
        cy = r_mid * np.sin(angle_center)

        seg_length = r_mid * angle_span
        # For innermost ring segments near center
        if r_mid < 1.0:
            seg_length = max(seg_length, 0.5)

        seg_path = f"{sector_path}/seg_{seg}"
        cube = UsdGeom.Cube.Define(stage, seg_path)
        cube.GetSizeAttr().Set(1.0)

        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, 0.002))
        rot_deg = np.degrees(angle_center)
        xf.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, rot_deg))
        xf.AddScaleOp().Set(Gf.Vec3d(width, seg_length, 0.004))

        cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

        prim = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(UsdShade.Material.Get(stage, mat_path))


def _build_friction_quadrant(stage, rng):
    """Build the friction quadrant (Q1, 0-90°)."""
    from pxr import UsdGeom
    quad_idx = 0
    qdef = QUADRANT_DEFS[quad_idx]
    root = f"/World/Arena/{qdef['name']}"
    UsdGeom.Xform.Define(stage, root)

    for lvl_idx, lvl in enumerate(FRICTION_LEVELS):
        # Color varies with friction — bluer = more slippery
        blue_frac = 1.0 - (lvl["mu_static"] / 0.90)
        color = (0.5 - blue_frac * 0.3, 0.5 - blue_frac * 0.2, 0.5 + blue_frac * 0.4)

        mat_path = _create_material(stage, f"FrictionMat_L{lvl_idx+1}",
                                    lvl["mu_static"], lvl["mu_dynamic"])
        _create_sector_ground(stage, quad_idx, lvl_idx, mat_path, color)
        print(f"    L{lvl_idx+1}: mu_s={lvl['mu_static']}, mu_d={lvl['mu_dynamic']} — {lvl['label']}")


def _build_grass_quadrant(stage, rng):
    """Build the grass quadrant (Q2, 90-180°)."""
    from pxr import UsdGeom
    quad_idx = 1
    qdef = QUADRANT_DEFS[quad_idx]
    root = f"/World/Arena/{qdef['name']}"
    UsdGeom.Xform.Define(stage, root)

    total_stalks = 0
    mat_path = _create_material(stage, "GrassMat", 0.70, 0.60)

    for lvl_idx, lvl in enumerate(GRASS_LEVELS):
        r_inner, r_outer = level_radius_range(lvl_idx)
        green = 0.6 - lvl_idx * 0.08
        color = (0.15, max(0.2, green), 0.10)

        _create_sector_ground(stage, quad_idx, lvl_idx, mat_path, color)

        # Scatter stalks in this sector
        if lvl["stalk_density"] > 0:
            stalk_path = f"{root}/stalks_L{lvl_idx+1}"
            count = create_stalks_polar(
                stage, stalk_path, r_inner, r_outer,
                lvl["stalk_density"], lvl["height_range"], rng,
                # Restrict to this quadrant's angle range
                sector_angle_start=qdef["angle_start"],
                sector_angle_end=qdef["angle_end"],
            )
            total_stalks += count

        print(f"    L{lvl_idx+1}: density={lvl['stalk_density']}/m², "
              f"drag={lvl['drag_coeff']} — {lvl['label']}")

    return total_stalks


def _build_boulder_quadrant(stage, rng):
    """Build the boulder quadrant (Q3, 180-270°)."""
    from pxr import UsdGeom
    quad_idx = 2
    qdef = QUADRANT_DEFS[quad_idx]
    root = f"/World/Arena/{qdef['name']}"
    UsdGeom.Xform.Define(stage, root)

    total_boulders = 0
    mat_path = _create_material(stage, "BoulderMat", 0.75, 0.65)

    for lvl_idx, lvl in enumerate(BOULDER_LEVELS):
        r_inner, r_outer = level_radius_range(lvl_idx)
        brown = 0.5 - lvl_idx * 0.05
        color = (0.5 + lvl_idx * 0.03, brown, brown * 0.7)

        _create_sector_ground(stage, quad_idx, lvl_idx, mat_path, color)

        # Scatter boulders in this quadrant sector
        density = lvl["density"]
        edge_min, edge_max = lvl["edge_range"]
        # Annular area for this quadrant's sector (quarter of full annulus)
        area = 0.25 * np.pi * (r_outer**2 - r_inner**2)
        count = int(density * area)

        boulder_path = f"{root}/boulders_L{lvl_idx+1}"
        UsdGeom.Xform.Define(stage, boulder_path)

        created = 0
        for b in range(count):
            r = np.sqrt(rng.uniform(r_inner**2, r_outer**2))
            theta = rng.uniform(qdef["angle_start"], qdef["angle_end"])
            x = r * np.cos(theta)
            y = r * np.sin(theta)

            edge = rng.uniform(edge_min, edge_max)
            z = edge * 0.3
            shape = SHAPE_NAMES[created % 4]
            rotation = (rng.uniform(0, 360), rng.uniform(0, 360), rng.uniform(0, 360))

            path = f"{boulder_path}/b_{created}"
            create_polyhedron(stage, path, shape, edge, (x, y, z), rotation)
            created += 1

        total_boulders += created
        print(f"    L{lvl_idx+1}: edge={lvl['edge_range']}, density={density}/m², "
              f"count={created} — {lvl['label']}")

    return total_boulders


def _build_stairs_quadrant(stage, rng, waypoint_positions):
    """Build the stairs quadrant (Q4, 270-360°) with pyramid waypoints.

    Args:
        stage: USD stage
        rng: numpy RandomState
        waypoint_positions: list of (x, y) tuples where pyramids should go.
            Should be 10 positions (2 per level).

    Returns:
        dict: level_idx -> list of summit heights for each pyramid
    """
    from pxr import UsdGeom
    quad_idx = 3
    qdef = QUADRANT_DEFS[quad_idx]
    root = f"/World/Arena/{qdef['name']}"
    UsdGeom.Xform.Define(stage, root)

    mat_path = _create_material(stage, "StairMat", 0.60, 0.50)

    # Ground segments for each level
    for lvl_idx in range(NUM_LEVELS):
        color = (0.55 - lvl_idx * 0.03, 0.55 - lvl_idx * 0.03, 0.52 - lvl_idx * 0.03)
        _create_sector_ground(stage, quad_idx, lvl_idx, mat_path, color)

    # Build pyramids at waypoint positions
    summit_heights = {}  # wp_index -> summit_z
    wp_idx = 0
    for lvl_idx, lvl in enumerate(STAIRS_LEVELS):
        level_summits = []
        for wp_in_level in range(WPS_PER_LEVEL):
            if wp_idx >= len(waypoint_positions):
                break
            pos = waypoint_positions[wp_idx]

            pyramid_path = f"{root}/pyramid_L{lvl_idx+1}_{wp_in_level}"
            summit_z = create_stair_pyramid(
                stage, pyramid_path,
                lvl["step_height"], lvl["step_depth"], lvl["num_steps"],
                pos, mat_path,
                rotation_deg=rng.uniform(0, 360),
            )
            summit_heights[wp_idx] = summit_z
            level_summits.append(summit_z)
            wp_idx += 1

        print(f"    L{lvl_idx+1}: {lvl['step_height']}m × {lvl['num_steps']} steps = "
              f"{lvl['step_height'] * lvl['num_steps']:.2f}m summit — {lvl['label']}")

    return summit_heights


def _create_containment_wall(stage, radius=51.0, height=2.0, thickness=0.5):
    """Create outer boundary wall as 36 segments."""
    from pxr import UsdGeom, UsdPhysics, Gf

    wall_path = "/World/Arena/wall"
    UsdGeom.Xform.Define(stage, wall_path)

    num_segs = 36
    angle_step = 2.0 * np.pi / num_segs
    arc_length = radius * angle_step

    for seg in range(num_segs):
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


def _create_quadrant_dividers(stage):
    """Create visual divider lines between quadrants (thin colored cubes)."""
    from pxr import UsdGeom, Gf

    dividers_path = "/World/Arena/dividers"
    UsdGeom.Xform.Define(stage, dividers_path)

    for i, qdef in enumerate(QUADRANT_DEFS):
        angle = qdef["angle_start"]
        # Line from center to wall
        for seg in range(25):
            r = seg * 2.0 + 1.0
            cx = r * np.cos(angle)
            cy = r * np.sin(angle)

            seg_path = f"{dividers_path}/q{i}_seg{seg}"
            cube = UsdGeom.Cube.Define(stage, seg_path)
            cube.GetSizeAttr().Set(1.0)

            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, 0.01))
            rot_deg = np.degrees(angle)
            xf.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, rot_deg))
            xf.AddScaleOp().Set(Gf.Vec3d(0.1, 2.0, 0.002))

            cube.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 1.0, 1.0)])


def _create_level_markers(stage):
    """Create thin rings at level boundaries (10m, 20m, 30m, 40m)."""
    from pxr import UsdGeom, Gf

    marker_path = "/World/Arena/markers"
    UsdGeom.Xform.Define(stage, marker_path)

    for lvl_idx in range(1, NUM_LEVELS):
        radius = lvl_idx * LEVEL_WIDTH
        ring_path = f"{marker_path}/ring_{lvl_idx}"
        UsdGeom.Xform.Define(stage, ring_path)

        num_segs = 72
        angle_step = 2.0 * np.pi / num_segs
        arc_length = radius * angle_step

        for seg in range(num_segs):
            angle = angle_step * (seg + 0.5)
            cx = radius * np.cos(angle)
            cy = radius * np.sin(angle)

            seg_path = f"{ring_path}/seg_{seg}"
            cube = UsdGeom.Cube.Define(stage, seg_path)
            cube.GetSizeAttr().Set(1.0)

            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, 0.008))
            rot_deg = np.degrees(angle)
            xf.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, rot_deg))
            xf.AddScaleOp().Set(Gf.Vec3d(0.1, arc_length, 0.002))

            cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.8, 0.8, 0.0)])


def create_quadrant_arena(stage, stairs_wp_positions):
    """Build the complete 4-quadrant arena.

    Args:
        stage: USD stage
        stairs_wp_positions: list of (x, y) for stair pyramid placement (10 positions)

    Returns:
        dict: Arena summary including pyramid summit heights
    """
    from pxr import UsdGeom

    root = "/World/Arena"
    UsdGeom.Xform.Define(stage, root)

    rng = np.random.RandomState(42)
    summary = {"total_boulders": 0, "total_stalks": 0, "pyramid_summits": {}}

    # Q1: Friction
    print("  Building Q1: Friction...")
    _build_friction_quadrant(stage, rng)

    # Q2: Grass
    print("  Building Q2: Grass...")
    total_stalks = _build_grass_quadrant(stage, rng)
    summary["total_stalks"] = total_stalks

    # Q3: Boulders
    print("  Building Q3: Boulders...")
    total_boulders = _build_boulder_quadrant(stage, rng)
    summary["total_boulders"] = total_boulders

    # Q4: Stairs (pyramids)
    print("  Building Q4: Stairs (pyramids)...")
    summit_heights = _build_stairs_quadrant(stage, rng, stairs_wp_positions)
    summary["pyramid_summits"] = summit_heights

    # Containment wall
    print("  Building containment wall...")
    _create_containment_wall(stage)

    # Quadrant dividers + level markers
    print("  Adding quadrant dividers + level markers...")
    _create_quadrant_dividers(stage)
    _create_level_markers(stage)

    print(f"\n  Arena complete: {summary['total_boulders']} boulders, "
          f"{summary['total_stalks']} stalks, "
          f"{len(summit_heights)} pyramids")

    return summary
