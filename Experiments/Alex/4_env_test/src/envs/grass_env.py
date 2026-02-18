"""Grass / fluid resistance environment — proxy stalks + drag forces.

Zone layout (increasing resistance):
  Zone 1 (0-10m):  0 stalks/m², drag=0.5 N·s/m   — Light fluid
  Zone 2 (10-20m): 2 stalks/m², drag=2.0 N·s/m   — Thin grass
  Zone 3 (20-30m): 5 stalks/m², drag=5.0 N·s/m   — Medium lawn
  Zone 4 (30-40m): 10 stalks/m², drag=10.0 N·s/m  — Thick grass
  Zone 5 (40-50m): 20 stalks/m², drag=20.0 N·s/m  — Dense brush

Implementation:
1. Proxy stalks (kinematic cylinders) from grass_physics_config.py patterns
2. Drag approximated in eval loop by scaling velocity commands

Reuses patterns from:
- ARL_DELIVERY/04_Teleop_System/grass_physics_config.py (stalk creation)
"""

import numpy as np

from configs.zone_params import ZONE_PARAMS, ARENA_WIDTH, ZONE_LENGTH


# Maximum drag coefficient (for normalization)
MAX_DRAG = 20.0


def get_zone_drag_coeff(x_pos):
    """Get the drag coefficient for a given x position.

    Args:
        x_pos: Robot x position in meters

    Returns:
        float: Drag coefficient for the current zone
    """
    zones = ZONE_PARAMS["grass"]
    for zone in zones:
        if zone["x_start"] <= x_pos < zone["x_end"]:
            return zone["drag_coeff"]
    # Past the arena
    return zones[-1]["drag_coeff"]


def get_velocity_scale(x_pos):
    """Get velocity command scale factor for drag approximation.

    Returns a value in [0, 1] where 1.0 = no drag, lower = more resistance.
    Used to scale the vx command before sending to the policy.

    Args:
        x_pos: Robot x position in meters

    Returns:
        float: Velocity scale factor
    """
    drag = get_zone_drag_coeff(x_pos)
    return 1.0 / (1.0 + drag * 0.1)


def _create_grass_material(stage, name, friction=0.80):
    """Create a physics material for grass surfaces."""
    from pxr import UsdShade, UsdPhysics, UsdGeom, PhysxSchema

    parent = "/World/Physics/Materials"
    if not stage.GetPrimAtPath(parent).IsValid():
        UsdGeom.Xform.Define(stage, "/World/Physics")
        UsdGeom.Xform.Define(stage, parent)

    mat_path = f"{parent}/{name}"
    UsdShade.Material.Define(stage, mat_path)

    prim = stage.GetPrimAtPath(mat_path)
    phys_mat = UsdPhysics.MaterialAPI.Apply(prim)
    phys_mat.CreateStaticFrictionAttr(friction)
    phys_mat.CreateDynamicFrictionAttr(friction * 0.875)
    phys_mat.CreateRestitutionAttr(0.05)

    physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(prim)
    physx_mat.CreateFrictionCombineModeAttr().Set("average")
    physx_mat.CreateRestitutionCombineModeAttr().Set("min")

    return mat_path


def _create_stalks_for_zone(stage, zone_path, zone, rng):
    """Scatter kinematic cylinder stalks within a zone.

    Args:
        stage: USD stage
        zone_path: Parent prim path for this zone
        zone: Zone parameter dict
        rng: numpy RandomState for reproducibility

    Returns:
        int: Number of stalks created
    """
    from pxr import UsdGeom, UsdPhysics, Gf

    density = zone["stalk_density"]
    if density == 0:
        return 0

    height_range = zone["height_range"]
    x_start = zone["x_start"]
    x_end = zone["x_end"]

    # stalks/m² × zone area
    area = ZONE_LENGTH * ARENA_WIDTH
    num_stalks = int(density * area)

    # Cap stalks for performance
    num_stalks = min(num_stalks, 2000)

    for s in range(num_stalks):
        x = rng.uniform(x_start, x_end)
        y = rng.uniform(0, ARENA_WIDTH)
        height = rng.uniform(height_range[0], height_range[1])
        radius = rng.uniform(0.005, 0.015)

        stalk_path = f"{zone_path}/stalk_{s}"
        cyl = UsdGeom.Cylinder.Define(stage, stalk_path)
        cyl.GetRadiusAttr().Set(float(radius))
        cyl.GetHeightAttr().Set(float(height))
        cyl.GetDisplayColorAttr().Set([Gf.Vec3f(0.15, 0.55, 0.10)])

        # Position: sink base 2cm into ground
        cz = height / 2.0 - 0.02
        xf = UsdGeom.Xformable(cyl.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, cz))

        # Kinematic rigid body + collision
        prim = cyl.GetPrim()
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        rb.CreateKinematicEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(prim)

    return num_stalks


def create_grass_environment(stage, cfg=None):
    """Build the grass/fluid resistance environment.

    Creates zone-specific ground coloring and kinematic cylinder stalks.
    Drag is approximated in the eval loop via get_velocity_scale().

    Args:
        stage: USD stage
        cfg: Optional config (unused, params come from zone_params)
    """
    from pxr import UsdGeom, UsdPhysics, UsdShade, Gf

    root = "/World/GrassArena"
    UsdGeom.Xform.Define(stage, root)

    zones = ZONE_PARAMS["grass"]
    rng = np.random.RandomState(42)
    total_stalks = 0

    # Create ground material
    mat_path = _create_grass_material(stage, "GrassMat")

    for zone in zones:
        zone_idx = zone["zone"]
        x_start = zone["x_start"]
        x_end = zone["x_end"]

        # Ground segment with grass-colored overlay
        zone_width = x_end - x_start
        center_x = (x_start + x_end) / 2.0
        green_intensity = 0.6 - (zone_idx - 1) * 0.08

        seg_path = f"{root}/ground_{zone_idx}"
        cube = UsdGeom.Cube.Define(stage, seg_path)
        cube.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(center_x, ARENA_WIDTH / 2.0, 0.002))
        xf.AddScaleOp().Set(Gf.Vec3d(zone_width, ARENA_WIDTH, 0.001))
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.15, green_intensity, 0.10)])

        prim = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(UsdShade.Material.Get(stage, mat_path))

        # Create stalks for zones with density > 0
        zone_path = f"{root}/stalks_{zone_idx}"
        UsdGeom.Xform.Define(stage, zone_path)
        count = _create_stalks_for_zone(stage, zone_path, zone, rng)
        total_stalks += count

    print(f"  Grass environment: {len(zones)} zones, {total_stalks} stalks")
    for z in zones:
        print(f"    Zone {z['zone']}: density={z['stalk_density']}/m², "
              f"drag={z['drag_coeff']} — {z['label']}")
