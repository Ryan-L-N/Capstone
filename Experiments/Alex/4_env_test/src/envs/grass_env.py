"""Grass / fluid resistance environment — proxy stalks + drag forces.

Zone layout (increasing resistance):
  Zone 1 (0-10m):  0 stalks/m², drag=0.5 N·s/m   — Light fluid
  Zone 2 (10-20m): 2 stalks/m², drag=2.0 N·s/m   — Thin grass
  Zone 3 (20-30m): 5 stalks/m², drag=5.0 N·s/m   — Medium lawn
  Zone 4 (30-40m): 10 stalks/m², drag=10.0 N·s/m  — Thick grass
  Zone 5 (40-50m): 20 stalks/m², drag=20.0 N·s/m  — Dense brush

Implementation:
1. Proxy stalks (kinematic cylinders) from grass_physics_config.py patterns
2. Scripted drag force: F_drag = -c_drag * v_base (via event system)

Reuses patterns from:
- ARL_DELIVERY/04_Teleop_System/grass_physics_config.py (stalk creation)
- ARL_DELIVERY/05_Training_Package/isaac_lab_spot_configs/mdp/events.py (force application)
"""

# TODO: Implementation
# - create_grass_environment(stage, cfg)
# - create_grass_zone(stage, zone_idx, x_start, density, height_range)
# - apply_zone_drag(env, asset_cfg) — event callback for drag forces
# - GRASS_ZONES = [(0, 0.5), (2, 2.0), (5, 5.0), (10, 10.0), (20, 20.0)]


def create_grass_environment(stage, cfg):
    """Build the grass/fluid resistance environment."""
    raise NotImplementedError("TODO: Implement grass environment builder")
