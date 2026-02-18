"""Base arena setup shared across all 4 environments.

Creates the common 30m (Y) x 50m (X) arena with:
- Ground plane
- Physics scene (GPU PhysX, 500 Hz)
- Spawn position (x=0, y=15m center)
- Lighting

IMPORTANT: SimulationApp must be created BEFORE importing this module.

Reuses patterns from:
- ARL_DELIVERY/06_Core_Library/world_factory.py
- ARL_DELIVERY/02_Obstacle_Course/spot_obstacle_course.py
"""

import numpy as np

from configs.eval_cfg import (
    PHYSICS_DT,
    RENDERING_DT,
    SPAWN_POSITION,
    STIFFNESS,
    DAMPING,
    ACTION_SCALE,
    OBS_DIM,
    HEIGHT_SCAN_FILL,
)
from configs.zone_params import ARENA_WIDTH, ARENA_LENGTH


def create_arena(headless=True):
    """Create SimulationApp, World, and stage for the evaluation arena.

    Args:
        headless: If True, run without GUI rendering.

    Returns:
        (app, world, stage) tuple.
    """
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": headless})

    # Now safe to import omni modules
    from omni.isaac.core import World
    from pxr import UsdGeom, UsdLux, Gf

    world = World(
        physics_dt=PHYSICS_DT,
        rendering_dt=RENDERING_DT,
        stage_units_in_meters=1.0,
    )

    stage = world.stage

    # Default ground plane (environments may override with custom ground)
    world.scene.add_default_ground_plane(
        z_position=0,
        name="ground",
        prim_path="/World/Ground",
        static_friction=0.8,
        dynamic_friction=0.8,
        restitution=0.01,
    )

    # Distant light for visibility
    light_path = "/World/DistantLight"
    light = UsdLux.DistantLight.Define(stage, light_path)
    light.CreateIntensityAttr(3000.0)
    UsdGeom.Xformable(light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45, 0, 0))

    return app, world, stage


def add_robot(world, stage, policy_type="flat"):
    """Load the Spot robot with the specified policy type.

    Args:
        world: omni.isaac.core.World instance
        stage: USD stage
        policy_type: "flat" or "rough"

    Returns:
        spot: Policy wrapper instance with .forward(dt, cmd) interface
    """
    import numpy as np

    if policy_type == "flat":
        from spot_flat_terrain_policy import SpotFlatTerrainPolicy
        spot = SpotFlatTerrainPolicy(
            prim_path="/World/Spot",
            name="spot",
            position=np.array(SPAWN_POSITION),
        )
    elif policy_type == "rough":
        from spot_rough_terrain_policy import SpotRoughTerrainPolicy
        spot = SpotRoughTerrainPolicy(
            prim_path="/World/Spot",
            name="spot",
            position=np.array(SPAWN_POSITION),
        )
    else:
        raise ValueError(f"Unknown policy type: {policy_type}. Use 'flat' or 'rough'.")

    return spot


def quat_to_yaw(quat):
    """Extract yaw angle from quaternion [w, x, y, z].

    Args:
        quat: (4,) array — quaternion in scalar-first convention [w, x, y, z]

    Returns:
        float — yaw angle in radians
    """
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def disable_default_ground(stage):
    """Remove/hide the default ground plane so environment-specific ground is used.

    Call this after create_arena() if the environment provides its own ground.
    """
    from pxr import UsdGeom

    ground_prim = stage.GetPrimAtPath("/World/Ground")
    if ground_prim.IsValid():
        imageable = UsdGeom.Imageable(ground_prim)
        imageable.MakeInvisible()
        # Disable collision by deactivating the prim
        ground_prim.SetActive(False)
