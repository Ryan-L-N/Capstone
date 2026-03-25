"""Stair pyramid generator — stepped pyramids as waypoints in the stairs quadrant.

Each pyramid is a stack of cubes forming an ascending staircase from ground
level to a summit platform. The waypoint is at the top of the pyramid.

Pyramid shapes:
  Level 1: 5 steps × 3cm = 15cm total, ~1.5m × 1.5m footprint
  Level 2: 6 steps × 8cm = 48cm total, ~1.8m × 1.8m footprint
  Level 3: 7 steps × 13cm = 91cm total, ~2.1m × 2.1m footprint
  Level 4: 8 steps × 18cm = 144cm total, ~2.4m × 2.4m footprint
  Level 5: 10 steps × 23cm = 230cm total, ~3.0m × 3.0m footprint

Each step is a solid cube with physics collision. Steps are built as
concentric rings (like a ziggurat) so the robot can approach from any side.
The base step is widest, each step above is narrower.
"""

import numpy as np


def create_stair_pyramid(stage, path, step_height, step_depth, num_steps,
                         position, mat_path, rotation_deg=0.0):
    """Build a stepped pyramid at the given position.

    The pyramid is centered at (position.x, position.y) with base at z=0.
    Each step is a solid cube that extends from z=0 up to the step's top.

    Args:
        stage: USD stage
        path: Parent prim path for this pyramid
        step_height: Height of each step in meters
        step_depth: Tread depth of each step in meters
        num_steps: Number of steps
        position: (x, y) world position of pyramid center
        mat_path: Physics material prim path
        rotation_deg: Rotation around Z axis in degrees

    Returns:
        float: Summit height (z coordinate of pyramid top)
    """
    from pxr import UsdGeom, UsdPhysics, UsdShade, Gf

    UsdGeom.Xform.Define(stage, path)

    summit_z = 0.0

    for s in range(num_steps):
        step_num = s + 1
        # Each step shrinks inward: base is widest, top is smallest
        base_half = num_steps * step_depth / 2.0
        step_half = base_half - s * step_depth / 2.0
        step_size = max(step_half * 2.0, step_depth)  # minimum 1 tread wide

        # This step's top z
        step_top = step_height * step_num
        # Solid cube from z=0 to step_top, centered at position
        cube_z = step_top / 2.0

        step_path = f"{path}/step_{s}"
        cube = UsdGeom.Cube.Define(stage, step_path)
        cube.GetSizeAttr().Set(1.0)

        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], cube_z))
        if abs(rotation_deg) > 0.01:
            xf.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, rotation_deg))
        xf.AddScaleOp().Set(Gf.Vec3d(step_size, step_size, step_top))

        # Concrete color with slight variation
        grey = 0.55 + (s % 3) * 0.03
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(grey, grey, grey * 0.95)])

        # Physics
        prim = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        rb.CreateKinematicEnabledAttr(True)

        if mat_path:
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(UsdShade.Material.Get(stage, mat_path))

        summit_z = step_top

    return summit_z


def get_pyramid_footprint(step_depth, num_steps):
    """Get the full footprint radius of a pyramid (half the base width)."""
    return num_steps * step_depth / 2.0
