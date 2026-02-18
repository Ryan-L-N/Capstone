"""Staircase environment — ascending steps with recovery platforms.

Zone layout (increasing step height):
  Zone 1 (0-10m):  3cm steps,  30cm tread, 33 steps  — Access ramp
  Zone 2 (10-20m): 8cm steps,  30cm tread, 33 steps  — Low residential
  Zone 3 (20-30m): 13cm steps, 30cm tread, 33 steps  — Standard residential
  Zone 4 (30-40m): 18cm steps, 30cm tread, 33 steps  — Steep commercial
  Zone 5 (40-50m): 23cm steps, 30cm tread, 33 steps  — Maximum challenge

2m flat platforms between zones for stabilization/recovery.
Stairs span full 30m width (Y-axis).

Reuses patterns from:
- ARL_DELIVERY/02_Obstacle_Course/spot_obstacle_course.py (create_steps)
"""

from configs.zone_params import ZONE_PARAMS, ARENA_WIDTH


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
                      width, base_elevation, mat_path):
    """Create ascending stairs as stacked box prims.

    Each step is a solid cube from z=0 up to cumulative height,
    so there are no hollow gaps underneath.

    Args:
        stage: USD stage
        zone_path: Parent prim path for this zone
        step_height: Height of each step in meters
        step_depth: Depth (tread) of each step in meters
        num_steps: Number of steps
        width: Width of stairs (Y-axis) in meters
        base_elevation: Starting elevation (top of previous zone)
        mat_path: Physics material prim path

    Returns:
        float: Final elevation at top of stairs (for chaining zones)
    """
    from pxr import UsdGeom, UsdPhysics, UsdShade, Gf

    UsdGeom.Xform.Define(stage, zone_path)

    current_z = base_elevation

    for s in range(num_steps):
        step_path = f"{zone_path}/step_{s}"
        cube = UsdGeom.Cube.Define(stage, step_path)
        cube.GetSizeAttr().Set(1.0)

        # X position: base + step index * depth + half depth
        step_x = s * step_depth + step_depth / 2.0
        current_z += step_height
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

    return current_z


def _create_platform(stage, path, x_center, width, elevation, mat_path):
    """Create a flat recovery platform between stair zones.

    Args:
        stage: USD stage
        path: Prim path
        x_center: Center X position
        width: Y-axis width
        elevation: Top surface height
        mat_path: Physics material prim path
    """
    from pxr import UsdGeom, UsdPhysics, UsdShade, Gf

    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)

    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(x_center, width / 2.0, elevation / 2.0))
    xf.AddScaleOp().Set(Gf.Vec3d(2.0, width, elevation))

    cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.50, 0.50, 0.48)])

    prim = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(UsdShade.Material.Get(stage, mat_path))


def create_stairs_environment(stage, cfg=None):
    """Build the staircase environment.

    Creates 5 zones of ascending stairs with 2m recovery platforms between zones.
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
        final_elevation = create_stair_zone(
            stage, zone_path, step_height, step_depth, num_steps,
            ARENA_WIDTH, base_elevation, mat_path
        )

        # Create recovery platform after this zone (between zones)
        if zone_idx < len(zones):
            platform_x = zone["x_end"] - 1.0  # 2m platform centered at zone boundary
            platform_path = f"{root}/platform_{zone_idx}"
            _create_platform(
                stage, platform_path, platform_x, ARENA_WIDTH,
                final_elevation, mat_path
            )

        base_elevation = final_elevation

    print(f"  Stairs environment: {len(zones)} zones")
    total_rise = base_elevation
    for z in zones:
        rise = z["step_height"] * z["num_steps"]
        print(f"    Zone {z['zone']}: {z['step_height']}m steps × {z['num_steps']} "
              f"= {rise:.2f}m rise — {z['label']}")
    print(f"    Total elevation gain: {total_rise:.2f}m")
