"""A* + APF nav modules for FiveRing_Arena.

Sourced from Experiments/Cole/PARKOUR_NAV_handoff/code/nav_modules/ (Apr 19
Skill-Nav Lite recipe). Self-contained: numpy + math only.

Public entry points:
    from .skill_nav_lite import SkillNavLiteNavigator
    from .grid_astar_planner import plan_path, expand_waypoints
    from .depth_raycast_detector import DepthRaycastDetector
    from .online_obstacle_tracker import OnlineObstacleTracker
"""
