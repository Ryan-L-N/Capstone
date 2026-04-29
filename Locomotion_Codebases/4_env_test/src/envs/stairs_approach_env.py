"""Staircase environment with a 10m flat approach before the stairs.

Identical stair geometry to stairs_env.py, but the robot spawns at x=-10m
and walks across flat ground from x=-10m to x=0m before reaching the first step.
This eliminates foot-clipping artifacts caused by spawning directly on stair geometry.

Approach area (x=-10m to x=0m):
    Flat ground at z=0, provided by the eval script's /World/GroundPlane.
    No additional geometry needed.

Stair zones (x=0m to x=50m):
    Identical to stairs_env.py — 5 zones of ascending steps.

Spawn:
    x=-10m, y=15m (arena center), z=0.6m above ground.
    Set in run_capstone_eval.py when env == "stairs_approach".
"""

from .stairs_env import create_stairs_environment


def create_stairs_approach_environment(stage, cfg=None):
    """Build the staircase environment with a flat approach area.

    The flat approach (x=-10 to x=0) is provided by the eval script's
    ground plane and requires no additional geometry here.
    The stair geometry (x=0 to x=50) is identical to stairs_env.py.

    Args:
        stage: USD stage
        cfg: Optional config (unused)
    """
    create_stairs_environment(stage, cfg)
