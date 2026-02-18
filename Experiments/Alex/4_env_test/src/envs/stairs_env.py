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
- Isaac Lab ROUGH_TERRAINS_CFG stair generation
"""

# TODO: Implementation
# - create_stair_zone(stage, zone_idx, x_start, step_height, step_depth, num_steps, width, base_elevation)
# - create_stairs_environment(stage, cfg)
# - STAIR_ZONES = [(0.03, 0.30, 33), (0.08, 0.30, 33), (0.13, 0.30, 33), (0.18, 0.30, 33), (0.23, 0.30, 33)]


def create_stairs_environment(stage, cfg):
    """Build the staircase environment."""
    raise NotImplementedError("TODO: Implement stairs environment builder")
