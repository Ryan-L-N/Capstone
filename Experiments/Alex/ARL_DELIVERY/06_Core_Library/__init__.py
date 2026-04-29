"""
Core shared library for MS for Autonomy experiments.
Consolidates boilerplate from 15+ standalone Isaac Sim scripts.

Usage (new scripts):
    # Step 1: Create simulation FIRST (before any other Isaac imports)
    from core.sim_app import create_simulation
    app = create_simulation(headless=False)

    # Step 2: Now import everything else
    from core import create_world, NavigationController, DataCollector
    from core import create_room_lighting, create_target_marker
"""

from .sim_app import create_simulation, get_simulation_app
from .world_factory import create_world, WorldConfig
from .lighting import create_room_lighting, LightingPreset
from .navigation import (
    NavigationController,
    quat_to_yaw,
    normalize_angle,
    distance_2d,
    calculate_path_efficiency,
)
from .data_collector import DataCollector
from .markers import create_target_marker, create_goal_marker
