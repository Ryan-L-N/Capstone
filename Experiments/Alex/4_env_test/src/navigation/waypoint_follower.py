"""Waypoint follower — generates velocity commands to guide robot through zones.

6 waypoints along the arena centerline (y=15m):
  WP0 (spawn): (0.0, 15.0)
  WP1:         (10.0, 15.0) — Zone 1/2 boundary
  WP2:         (20.0, 15.0) — Zone 2/3 boundary
  WP3:         (30.0, 15.0) — Zone 3/4 boundary
  WP4:         (40.0, 15.0) — Zone 4/5 boundary
  WP5 (goal):  (50.0, 15.0)

Heading controller:
  theta_err = atan2(wp_y - robot_y, wp_x - robot_x) - robot_yaw
  vx = 1.0 m/s (constant forward)
  vy = 0.0
  omega_z = Kp * theta_err (Kp=2.0)

Commands clamped to training ranges: vx[-2, 3], vy[-1.5, 1.5], wz[-2, 2]

Feeds through UniformVelocityCommandCfg override (replacing random sampling).
"""

# TODO: Implementation
# - WaypointFollower class
#   - __init__(self, num_envs, device)
#   - reset(self, env_ids)
#   - compute_commands(self, root_pos, root_yaw) -> (N, 3) tensor [vx, vy, wz]
# - WAYPOINTS = [(0, 15), (10, 15), (20, 15), (30, 15), (40, 15), (50, 15)]
# - KP_YAW = 2.0
# - TARGET_VX = 1.0
# - WAYPOINT_THRESHOLD = 0.5


class WaypointFollower:
    """Generates velocity commands to follow waypoints through the arena."""

    def __init__(self, num_envs, device="cuda:0"):
        raise NotImplementedError("TODO: Implement WaypointFollower")
