"""Vegetation stalk creation and drag logic for the grass quadrant.

Extracted from 4_env_test/src/envs/grass_env.py — stalk creation and
velocity drag scaling. Updated to support sector angle filtering.
"""

import numpy as np


def get_velocity_scale(drag_coeff):
    """Get velocity command scale factor for drag approximation.

    Returns a value in [0, 1] where 1.0 = no drag, lower = more resistance.
    Used to scale the vx command before sending to the policy.

    Args:
        drag_coeff: Drag coefficient (e.g., 5.0)

    Returns:
        float: Velocity scale factor
    """
    return 1.0 / (1.0 + drag_coeff * 0.1)


def create_stalks_polar(stage, parent_path, r_inner, r_outer, density,
                        height_range, rng, corridor_angle=0.0,
                        corridor_half_width=1.5, max_stalks=2000,
                        sector_angle_start=0.0, sector_angle_end=2*np.pi):
    """Scatter kinematic cylinder stalks in an annular sector (polar coords).

    Args:
        stage: USD stage
        parent_path: Parent prim path for stalks
        r_inner: Inner radius of annular region
        r_outer: Outer radius of annular region
        density: Stalks per square meter
        height_range: (min_height, max_height) in meters
        rng: numpy RandomState for reproducibility
        corridor_angle: Angle (radians) of clear transition corridor
        corridor_half_width: Half-width of corridor in meters
        max_stalks: Performance cap on stalk count
        sector_angle_start: Start angle of sector in radians
        sector_angle_end: End angle of sector in radians

    Returns:
        int: Number of stalks created
    """
    from pxr import UsdGeom, UsdPhysics, Gf

    # Area of the sector (fraction of full annulus)
    angle_frac = (sector_angle_end - sector_angle_start) / (2 * np.pi)
    area = angle_frac * np.pi * (r_outer**2 - r_inner**2)
    num_stalks = min(int(density * area), max_stalks)

    if num_stalks == 0:
        return 0

    UsdGeom.Xform.Define(stage, parent_path)

    created = 0
    for s in range(num_stalks):
        # Uniform random in annular sector
        r = np.sqrt(rng.uniform(r_inner**2, r_outer**2))
        theta = rng.uniform(sector_angle_start, sector_angle_end)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Skip corridor region (clear path for ring transitions)
        angle_diff = np.abs(np.arctan2(np.sin(theta - corridor_angle),
                                       np.cos(theta - corridor_angle)))
        corridor_arc = corridor_half_width / max(r, 0.1)
        if angle_diff < corridor_arc:
            continue

        height = rng.uniform(height_range[0], height_range[1])
        radius = rng.uniform(0.005, 0.015)

        stalk_path = f"{parent_path}/stalk_{created}"
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

        created += 1

    return created
