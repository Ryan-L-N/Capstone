"""Base arena setup shared across all 4 environments.

Creates the common 30m (Y) x 50m (X) arena with:
- Ground plane
- Physics scene (GPU PhysX, 500 Hz)
- Spawn position (x=0, y=15m center)
- Lighting

Reuses patterns from:
- ARL_DELIVERY/06_Core_Library/world_factory.py
- ARL_DELIVERY/02_Obstacle_Course/spot_obstacle_course.py
"""

# TODO: Implementation
# - create_arena(stage, cfg) -> sets up ground plane, physics, lighting
# - ARENA_WIDTH = 30.0  (Y-axis, meters)
# - ARENA_LENGTH = 50.0 (X-axis, meters)
# - SPAWN_POSITION = (0.0, 15.0, 0.6)  (center of Y-axis)
# - ZONE_LENGTH = 10.0  (meters per zone)
# - NUM_ZONES = 5
