"""Base arena utilities shared across ring environments.

Copied from 4_env_test/src/envs/base_arena.py — provides quat_to_yaw and
disable_default_ground. Arena creation is handled by run_ring_eval.py directly.

IMPORTANT: SimulationApp must be created BEFORE importing this module.
"""

import numpy as np


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
        ground_prim.SetActive(False)
