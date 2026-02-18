"""Friction surface environment — 5 zones of decreasing friction.

Zone layout (high friction -> low friction):
  Zone 1 (0-10m):  mu_s=0.90, mu_d=0.80  — 60-grit sandpaper
  Zone 2 (10-20m): mu_s=0.60, mu_d=0.50  — Dry rubber on concrete
  Zone 3 (20-30m): mu_s=0.35, mu_d=0.25  — Wet concrete
  Zone 4 (30-40m): mu_s=0.15, mu_d=0.08  — Wet ice
  Zone 5 (40-50m): mu_s=0.05, mu_d=0.02  — Oil on polished steel

Implementation: 5 ground plane segments with UsdPhysics.MaterialAPI,
friction combine mode = "multiply" (matching rough_env_cfg.py:402).

Reuses patterns from:
- ARL_DELIVERY/02_Obstacle_Course/spot_obstacle_course.py (create_physics_material)
"""

from configs.zone_params import ZONE_PARAMS, ARENA_WIDTH


# Zone colors: green (high grip) → red (low grip)
ZONE_COLORS = [
    (0.2, 0.7, 0.2),   # green — sandpaper
    (0.5, 0.7, 0.2),   # yellow-green — rubber
    (0.7, 0.7, 0.2),   # yellow — wet concrete
    (0.7, 0.4, 0.2),   # orange — wet ice
    (0.7, 0.2, 0.2),   # red — oil on steel
]


def _create_friction_material(stage, name, mu_static, mu_dynamic):
    """Create a physics material with specific friction values.

    Args:
        stage: USD stage
        name: Material name (used in prim path)
        mu_static: Static friction coefficient
        mu_dynamic: Dynamic friction coefficient

    Returns:
        str: Material prim path
    """
    from pxr import UsdShade, UsdPhysics, UsdGeom, PhysxSchema

    parent = "/World/Physics/Materials"
    if not stage.GetPrimAtPath(parent).IsValid():
        UsdGeom.Xform.Define(stage, "/World/Physics")
        UsdGeom.Xform.Define(stage, parent)

    mat_path = f"{parent}/{name}"
    UsdShade.Material.Define(stage, mat_path)

    prim = stage.GetPrimAtPath(mat_path)
    phys_mat = UsdPhysics.MaterialAPI.Apply(prim)
    phys_mat.CreateStaticFrictionAttr(mu_static)
    phys_mat.CreateDynamicFrictionAttr(mu_dynamic)
    phys_mat.CreateRestitutionAttr(0.01)

    physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(prim)
    physx_mat.CreateFrictionCombineModeAttr().Set("multiply")
    physx_mat.CreateRestitutionCombineModeAttr().Set("min")

    return mat_path


def create_friction_environment(stage, cfg=None):
    """Build the friction surface environment.

    Creates 5 ground segments with decreasing friction coefficients.
    Disables the default ground plane and replaces it with zone-specific segments.

    Args:
        stage: USD stage
        cfg: Optional config (unused, params come from zone_params)
    """
    from pxr import UsdGeom, UsdPhysics, UsdShade, Gf

    from envs.base_arena import disable_default_ground
    disable_default_ground(stage)

    root = "/World/FrictionArena"
    UsdGeom.Xform.Define(stage, root)

    zones = ZONE_PARAMS["friction"]

    for i, zone in enumerate(zones):
        zone_idx = zone["zone"]
        x_start = zone["x_start"]
        x_end = zone["x_end"]
        mu_s = zone["mu_static"]
        mu_d = zone["mu_dynamic"]

        # Create physics material for this zone
        mat_path = _create_friction_material(
            stage, f"FrictionZone{zone_idx}", mu_s, mu_d
        )

        # Create ground segment as a thin cube
        zone_width = x_end - x_start
        center_x = (x_start + x_end) / 2.0
        center_y = ARENA_WIDTH / 2.0

        seg_path = f"{root}/zone_{zone_idx}"
        cube = UsdGeom.Cube.Define(stage, seg_path)
        cube.GetSizeAttr().Set(1.0)

        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(center_x, center_y, -0.005))
        xf.AddScaleOp().Set(Gf.Vec3d(zone_width, ARENA_WIDTH, 0.01))

        cube.GetDisplayColorAttr().Set([Gf.Vec3f(*ZONE_COLORS[i])])

        # Apply collision and material
        prim = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(UsdShade.Material.Get(stage, mat_path))

    print(f"  Friction environment: {len(zones)} zones created")
    for z in zones:
        print(f"    Zone {z['zone']}: mu_s={z['mu_static']}, mu_d={z['mu_dynamic']} — {z['label']}")
