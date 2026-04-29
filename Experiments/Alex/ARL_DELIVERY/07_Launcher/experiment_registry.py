"""
Registry of all experiments in the MS for Autonomy project.
Maps short IDs to script paths for the launcher.

Usage:
    from experiment_registry import EXPERIMENTS, list_experiments, get_categories
    exps = list_experiments("grass")  # filter by category
"""

import os

# Resolve paths relative to this file
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CAPSTONE_ROOT = os.path.dirname(_THIS_DIR)


def _p(*parts):
    """Build path relative to MS_for_autonomy/."""
    return os.path.join(_THIS_DIR, *parts)


def _cap(*parts):
    """Build path relative to capstone root."""
    return os.path.join(_CAPSTONE_ROOT, *parts)


EXPERIMENTS = {
    # =========================================================================
    # Grass Terrain (Spot)
    # =========================================================================
    "grass-p1": {
        "name": "Grass Phase 1: Baseline Navigation",
        "script": _p("experimental_design_grass", "code", "phase_1_baseline.py"),
        "robot": "Spot",
        "category": "grass",
    },
    "grass-p2-friction": {
        "name": "Grass Phase 2: Friction-Based Grass",
        "script": _p("experimental_design_grass", "code", "phase_2_friction_grass.py"),
        "robot": "Spot",
        "category": "grass",
    },
    "grass-p2-height": {
        "name": "Grass Phase 2: Grass Height Variation",
        "script": _p("experimental_design_grass", "code", "phase_2_grass_height.py"),
        "robot": "Spot",
        "category": "grass",
    },
    "grass-p3": {
        "name": "Grass Phase 3: Grass Density",
        "script": _p("experimental_design_grass", "code", "phase_3_grass_density.py"),
        "robot": "Spot",
        "category": "grass",
    },
    "grass-p4": {
        "name": "Grass Phase 4: Combined Obstacles",
        "script": _p("experimental_design_grass", "code", "phase_4_combined_obstacles.py"),
        "robot": "Spot",
        "category": "grass",
    },
    "grass-p5-rl": {
        "name": "Grass Phase 5: RL Training",
        "script": _p("experimental_design_grass", "code", "phase_5_rl_training.py"),
        "robot": "Spot",
        "category": "grass",
    },
    "grass-teleop": {
        "name": "Grass Teleop: WASD Keyboard Control",
        "script": _p("experimental_design_grass", "code", "spot_teleop.py"),
        "robot": "Spot",
        "category": "grass",
    },
    "grass-obstacle": {
        "name": "Grass Obstacle Course: WASD + Xbox Teleop",
        "script": _p("experimental_design_grass", "code", "spot_obstacle_course.py"),
        "robot": "Spot",
        "category": "grass",
    },

    # =========================================================================
    # Vision 60 Alpha
    # =========================================================================
    "v60-urdf": {
        "name": "Vision 60: URDF Validation",
        "script": _p("experimental_design_vision60_alpha", "code", "phase_1_urdf_validation.py"),
        "robot": "Vision60",
        "category": "vision60",
    },
    "v60-flat": {
        "name": "Vision 60: Flat Terrain Policy",
        "script": _p("experimental_design_vision60_alpha", "code", "vision60_flat_terrain_policy.py"),
        "robot": "Vision60",
        "category": "vision60",
    },
    "v60-nav": {
        "name": "Vision 60: Autonomous Navigation",
        "script": _p("Vision 60", "vision60_autonomous_nav.py"),
        "robot": "Vision60",
        "category": "vision60",
    },
    "v60-eureka": {
        "name": "Vision 60: Eureka Navigation",
        "script": _p("Vision 60", "vision60_eureka_nav.py"),
        "robot": "Vision60",
        "category": "vision60",
    },
    "v60-gait": {
        "name": "Vision 60: Gait Navigation",
        "script": _p("Vision 60", "vision60_gait_nav.py"),
        "robot": "Vision60",
        "category": "vision60",
    },
    "v60-usma": {
        "name": "Vision 60: USMA Control v5",
        "script": _p("Vision 60", "vision60_usma_control_v5.py"),
        "robot": "Vision60",
        "category": "vision60",
    },

    # =========================================================================
    # Quadcopter Drone
    # =========================================================================
    "drone-p1": {
        "name": "Drone Phase 1: Stable Flight",
        "script": _p("experimental_design_quad_drone", "code", "phase_1_stable_flight.py"),
        "robot": "Iris",
        "category": "drone",
    },
    "drone-p2": {
        "name": "Drone Phase 2: Target Detection",
        "script": _p("experimental_design_quad_drone", "code", "phase_2_target_detection.py"),
        "robot": "Iris",
        "category": "drone",
    },
    "drone-teleop": {
        "name": "Drone Teleop: RC Controller Flight",
        "script": _p("experimental_design_quad_drone", "code", "drone_teleop.py"),
        "robot": "Iris",
        "category": "drone",
    },

    # =========================================================================
    # Original Spot Training
    # =========================================================================
    "spot-p1-sync": {
        "name": "Spot Phase 1: Sync Pattern Walk",
        "script": _cap("Spot-Quadruped-Training", "Phase-1", "spot_sync_pattern.py"),
        "robot": "Spot",
        "category": "spot-training",
    },
    "spot-p1-gates": {
        "name": "Spot Phase 1: Gate Walking",
        "script": _cap("Spot-Quadruped-Training", "Phase-1", "spot_gate_working.py"),
        "robot": "Spot",
        "category": "spot-training",
    },
    "spot-p1-two-gates": {
        "name": "Spot Phase 1: Two Gates",
        "script": _cap("Spot-Quadruped-Training", "Phase-1", "spot_two_gates.py"),
        "robot": "Spot",
        "category": "spot-training",
    },
    "spot-p2-visual": {
        "name": "Spot Phase 2: Visual Maze",
        "script": _cap("Spot-Quadruped-Training", "Phase-2", "spot_visual_maze.py"),
        "robot": "Spot",
        "category": "spot-training",
    },
    "spot-p3-lidar": {
        "name": "Spot Phase 3: LiDAR Maze",
        "script": _cap("Spot-Quadruped-Training", "Phase-3", "spot_lidar_maze.py"),
        "robot": "Spot",
        "category": "spot-training",
    },
    "spot-p4-dynamic": {
        "name": "Spot Phase 4: Dynamic Gates",
        "script": _cap("Spot-Quadruped-Training", "Phase-4", "spot_dynamic_gates.py"),
        "robot": "Spot",
        "category": "spot-training",
    },
    "spot-p5-rl": {
        "name": "Spot Phase 5: RL Dynamic Gates",
        "script": _cap("Spot-Quadruped-Training", "Phase-5", "spot_dynamic_gates_rl.py"),
        "robot": "Spot",
        "category": "spot-training",
    },

    # =========================================================================
    # Test Scripts
    # =========================================================================
    "test-clutter": {
        "name": "Test: Spot Cluttered Room",
        "script": _p("test_scripts", "phase1_clutter_test.py"),
        "robot": "Spot",
        "category": "test",
    },
    "test-sensor": {
        "name": "Test: Sensor Navigation",
        "script": _p("test_scripts", "phase1_sensor_navigation.py"),
        "robot": "Spot",
        "category": "test",
    },
}


def list_experiments(category=None):
    """List experiments, optionally filtered by category."""
    if category is None:
        return dict(EXPERIMENTS)
    return {k: v for k, v in EXPERIMENTS.items() if v["category"] == category}


def get_categories():
    """Get sorted list of unique categories."""
    return sorted(set(exp["category"] for exp in EXPERIMENTS.values()))


def get_experiment(key):
    """Get a single experiment by key, or None."""
    return EXPERIMENTS.get(key)
