"""Staircase environment — ascending steps with smooth zone transitions.

Zone layout (increasing step height):
  Zone 1 (0-10m):  3cm steps,  30cm tread, 33 steps  — Access ramp
  Zone 2 (10-20m): 8cm steps,  30cm tread, 33 steps  — Low residential
  Zone 3 (20-30m): 13cm steps, 30cm tread, 33 steps  — Standard residential
  Zone 4 (30-40m): 18cm steps, 30cm tread, 33 steps  — Steep commercial
  Zone 5 (40-50m): 23cm steps, 30cm tread, 33 steps  — Maximum challenge

Zone boundaries use 5 transition steps with linearly interpolated riser heights.
Stairs span full 30m width (Y-axis).

Reuses patterns from:
- ARL_DELIVERY/02_Obstacle_Course/spot_obstacle_course.py (create_steps)
"""

from configs.zone_params import ZONE_PARAMS, ARENA_WIDTH, ZONE_LENGTH, TRANSITION_STEPS


def _create_step_material(stage):
    """Create a physics material for stair surfaces."""
    from pxr import UsdShade, UsdPhysics, UsdGeom, PhysxSchema

    parent = "/World/Physics/Materials"
    if not stage.GetPrimAtPath(parent).IsValid():
        UsdGeom.Xform.Define(stage, "/World/Physics")
        UsdGeom.Xform.Define(stage, parent)

    mat_path = f"{parent}/StepMat"
    UsdShade.Material.Define(stage, mat_path)

    prim = stage.GetPrimAtPath(mat_path)
    phys_mat = UsdPhysics.MaterialAPI.Apply(prim)
    phys_mat.CreateStaticFrictionAttr(0.6)
    phys_mat.CreateDynamicFrictionAttr(0.5)
    phys_mat.CreateRestitutionAttr(0.01)

    physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(prim)
    physx_mat.CreateFrictionCombineModeAttr().Set("multiply")
    physx_mat.CreateRestitutionCombineModeAttr().Set("min")

    return mat_path


def create_stair_zone(stage, zone_path, step_height, step_depth, num_steps,
                      width, base_elevation, mat_path, prev_step_height=None):
    """Create ascending stairs as stacked box prims.

    Each step is a solid cube from z=0 up to cumulative height,
    so there are no hollow gaps underneath.

    If prev_step_height is given, the first TRANSITION_STEPS steps use
    linearly interpolated riser heights to smooth the zone boundary.

    Args:
        stage: USD stage
        zone_path: Parent prim path for this zone
        step_height: Height of each step in meters
        step_depth: Depth (tread) of each step in meters
        num_steps: Number of steps
        width: Width of stairs (Y-axis) in meters
        base_elevation: Starting elevation (top of previous zone)
        mat_path: Physics material prim path
        prev_step_height: Previous zone's step height (None for zone 1)

    Returns:
        float: Final elevation at top of stairs (for chaining zones)
    """
    from pxr import UsdGeom, UsdPhysics, UsdShade, Gf

    UsdGeom.Xform.Define(stage, zone_path)

    current_z = base_elevation

    for s in range(num_steps):
        # Determine this step's riser height
        if prev_step_height is not None and s < TRANSITION_STEPS:
            frac = (s + 1) / (TRANSITION_STEPS + 1)
            riser = prev_step_height + (step_height - prev_step_height) * frac
        else:
            riser = step_height

        step_path = f"{zone_path}/step_{s}"
        cube = UsdGeom.Cube.Define(stage, step_path)
        cube.GetSizeAttr().Set(1.0)

        # X position: base + step index * depth + half depth
        step_x = s * step_depth + step_depth / 2.0
        current_z += riser
        # Center of cube at half the total height from ground
        step_z = current_z / 2.0

        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(step_x, width / 2.0, step_z))
        xf.AddScaleOp().Set(Gf.Vec3d(step_depth, width, current_z))

        # Grey concrete color with slight variation
        grey = 0.55 + (s % 3) * 0.03
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(grey, grey, grey * 0.95)])

        prim = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(UsdShade.Material.Get(stage, mat_path))

    # Fill gap at end of zone (33 × 0.30 = 9.9m, zone is 10m)
    gap = ZONE_LENGTH - num_steps * step_depth
    if gap > 0.001:
        fill_path = f"{zone_path}/fill"
        fill = UsdGeom.Cube.Define(stage, fill_path)
        fill.GetSizeAttr().Set(1.0)
        fill_x = num_steps * step_depth + gap / 2.0
        fill_z = current_z / 2.0
        xf = UsdGeom.Xformable(fill.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(fill_x, width / 2.0, fill_z))
        xf.AddScaleOp().Set(Gf.Vec3d(gap, width, current_z))
        fill.GetDisplayColorAttr().Set([Gf.Vec3f(0.55, 0.55, 0.52)])
        p = fill.GetPrim()
        UsdPhysics.CollisionAPI.Apply(p)
        b = UsdShade.MaterialBindingAPI.Apply(p)
        b.Bind(UsdShade.Material.Get(stage, mat_path))

    return current_z


def create_stairs_environment(stage, cfg=None):
    """Build the staircase environment.

    Creates 5 zones of ascending stairs with smooth transition steps at boundaries.
    Each zone chains its base_elevation from the previous zone's final height.

    Args:
        stage: USD stage
        cfg: Optional config (unused, params come from zone_params)
    """
    from pxr import UsdGeom

    from envs.base_arena import disable_default_ground
    disable_default_ground(stage)

    root = "/World/Staircase"
    UsdGeom.Xform.Define(stage, root)

    zones = ZONE_PARAMS["stairs"]
    mat_path = _create_step_material(stage)

    base_elevation = 0.0
    prev_step_height = None

    for zone in zones:
        zone_idx = zone["zone"]
        step_height = zone["step_height"]
        step_depth = zone["step_depth"]
        num_steps = zone["num_steps"]

        # Create the stair zone with offset for x_start
        zone_path = f"{root}/zone_{zone_idx}"

        # We need steps positioned relative to zone x_start
        # create_stair_zone returns steps at local x coords, so we use an Xform
        # to offset the whole zone to x_start
        from pxr import Gf
        xform = UsdGeom.Xform.Define(stage, zone_path)
        UsdGeom.Xformable(xform.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(zone["x_start"], 0, 0)
        )

        # Create steps within the zone (local coordinates)
        # prev_step_height enables transition ramp at zone boundaries
        final_elevation = create_stair_zone(
            stage, zone_path, step_height, step_depth, num_steps,
            ARENA_WIDTH, base_elevation, mat_path,
            prev_step_height=prev_step_height,
        )

        prev_step_height = step_height
        base_elevation = final_elevation

    print(f"  Stairs environment: {len(zones)} zones")
    total_rise = base_elevation
    for z in zones:
        rise = z["step_height"] * z["num_steps"]
        print(f"    Zone {z['zone']}: {z['step_height']}m steps × {z['num_steps']} "
              f"= {rise:.2f}m rise — {z['label']}")
    print(f"    Total elevation gain: {total_rise:.2f}m")
