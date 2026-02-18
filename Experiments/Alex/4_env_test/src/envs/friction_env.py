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

# TODO: Implementation
# - create_friction_environment(stage, cfg)
# - create_friction_zone(stage, zone_idx, x_start, mu_s, mu_d)
# - FRICTION_ZONES = [(0.90, 0.80), (0.60, 0.50), (0.35, 0.25), (0.15, 0.08), (0.05, 0.02)]


def create_friction_environment(stage, cfg):
    """Build the friction surface environment."""
    raise NotImplementedError("TODO: Implement friction environment builder")
