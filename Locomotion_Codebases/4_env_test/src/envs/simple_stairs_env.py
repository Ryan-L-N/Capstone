"""Simple staircase — single uniform flight for sanity checks.

20 steps, 20cm rise, 30cm tread, 1m wide.
Friction: 1.0/1.0 (multiply combine) — matches S2R training range.
Total length: 20 * 0.3 = 6.0m
Total rise: 20 * 0.2 = 4.0m

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

STEP_HEIGHT = 0.20   # meters
STEP_DEPTH = 0.30    # meters
STEP_WIDTH = 10.00   # meters
NUM_STEPS = 20


def _create_step_material(stage):
    """Create physics material matching training friction."""
    from pxr import UsdShade, UsdPhysics, UsdGeom, PhysxSchema

    parent = "/World/Physics/Materials"
    if not stage.GetPrimAtPath(parent).IsValid():
        UsdGeom.Xform.Define(stage, "/World/Physics")
        UsdGeom.Xform.Define(stage, parent)

    mat_path = f"{parent}/SimpleStepMat"
    UsdShade.Material.Define(stage, mat_path)

    prim = stage.GetPrimAtPath(mat_path)
    phys_mat = UsdPhysics.MaterialAPI.Apply(prim)
    phys_mat.CreateStaticFrictionAttr(1.0)
    phys_mat.CreateDynamicFrictionAttr(1.0)
    phys_mat.CreateRestitutionAttr(0.01)

    physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(prim)
    physx_mat.CreateFrictionCombineModeAttr().Set("multiply")
    physx_mat.CreateRestitutionCombineModeAttr().Set("min")

    return mat_path


def create_simple_stairs_environment(stage, cfg=None):
    """Build a simple 20-step staircase for sanity checking.

    Stairs are centered on Y-axis, starting at x=0.
    Each step is a solid cube from ground to cumulative height.

    Args:
        stage: USD stage
        cfg: Optional config (unused)
    """
    from pxr import UsdGeom, UsdPhysics, UsdShade, Gf, PhysxSchema

    # Keep default ground plane — stairs sit on top of it
    # (don't disable like other envs that provide their own ground)

    root = "/World/SimpleStairs"
    xform = UsdGeom.Xform.Define(stage, root)
    # Shift entire staircase 1m forward so robot spawns on the approach platform
    UsdGeom.Xformable(xform.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(1.0, 0, 0))

    mat_path = _create_step_material(stage)

    # No approach platform needed — default ground plane is the floor

    # Build 20 steps (sitting on top of default ground plane)
    current_z = 0.0
    for s in range(NUM_STEPS):
        step_path = f"{root}/step_{s}"
        cube = UsdGeom.Cube.Define(stage, step_path)
        cube.GetSizeAttr().Set(1.0)

        current_z += STEP_HEIGHT
        step_x = s * STEP_DEPTH + STEP_DEPTH / 2.0
        step_z = current_z / 2.0

        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(step_x, 15.0, step_z))
        xf.AddScaleOp().Set(Gf.Vec3d(STEP_DEPTH, STEP_WIDTH, current_z))

        grey = 0.55 + (s % 3) * 0.03
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(grey, grey, grey * 0.95)])

        prim = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision_api.CreateContactOffsetAttr().Set(0.02)
        collision_api.CreateRestOffsetAttr().Set(0.01)
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(UsdShade.Material.Get(stage, mat_path))

    # No landing platform — ground plane extends beyond stairs

    total_length = NUM_STEPS * STEP_DEPTH
    total_rise = NUM_STEPS * STEP_HEIGHT
    print(f"  Simple stairs: {NUM_STEPS} steps, {STEP_HEIGHT}m rise, {STEP_DEPTH}m tread, {STEP_WIDTH}m wide")
    print(f"    Total length: {total_length:.1f}m, total rise: {total_rise:.1f}m")
    print(f"    Angle: {__import__('math').degrees(__import__('math').atan2(total_rise, total_length)):.1f} deg")
