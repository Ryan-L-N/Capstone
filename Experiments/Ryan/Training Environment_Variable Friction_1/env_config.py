"""
Training Environment 1 — Configuration
All parameters centralized here for easy tuning.
No Isaac imports — safe to import before SimulationApp.
"""

from typing import Dict, List, Tuple, Union


class EnvConfig:
    """All environment, physics, robot, and reward parameters."""

    # --- Arena ---
    ARENA_LENGTH: float = 100.0       # X: -50 to +50
    ARENA_WIDTH:  float = 30.0        # Y: -15 to +15
    WALL_HEIGHT:  float = 2.0
    WALL_THICKNESS: float = 0.5

    # --- Robots ---
    NUM_ROBOTS: int = 5
    START_X: float = -48.0
    START_Y_POSITIONS: List[float] = [-10.0, -5.0, 0.0, 5.0, 10.0]
    START_Z: float = 0.7
    NOMINAL_HEIGHT: float = 0.7       # Expected body Z during steady-state walking

    # --- Physics ---
    PHYSICS_DT: float = 1.0 / 500.0
    RENDERING_DT: float = 10.0 / 500.0
    CONTROL_DECIMATION: int = 10      # Run RL policy every N physics steps (= 50 Hz)

    # --- Episode ---
    STABILIZE_TIME: float = 3.0       # Seconds of zero-command settling before episode begins
    MAX_EPISODE_TIME: float = 120.0
    FALL_Z: float = 0.25              # Body Z below this = fallen
    LATERAL_LIMIT: float = 13.0       # |Y| beyond this = out of bounds
    FAR_END_X: float = 48.0           # Lap reset when robot reaches here

    # --- Action limits ---
    VX_MAX: float = 2.0               # Forward velocity ceiling
    VY_LIMIT: float = 0.3             # Lateral velocity magnitude limit
    YAW_LIMIT: float = 0.3            # Yaw rate magnitude limit

    # --- Reward scales ---
    R_FORWARD:    float = 2.0
    R_ROLL:       float = 3.0
    R_PITCH:      float = 3.0
    R_ANG_RATE:   float = 1.0
    R_HEIGHT:     float = 2.0
    R_SMOOTHNESS: float = 1.0
    R_LAT_YAW:    float = 0.5
    R_ALIVE:      float = 0.2
    R_FALL:       float = -100.0


# ---------------------------------------------------------------------------
# Friction presets — static and dynamic coefficients (rubber foot on surface)
# Values from tribology literature for rubber-tipped feet.
# ---------------------------------------------------------------------------
FRICTION_CONFIG: Dict[str, Dict[str, Union[float, Tuple[float, float]]]] = {
    "asphalt_dry": {"static": 0.75, "dynamic": 0.65},
    "asphalt_wet": {"static": 0.50, "dynamic": 0.40},
    "grass_dry":   {"static": 0.40, "dynamic": 0.35},
    "grass_wet":   {"static": 0.25, "dynamic": 0.20},
    "mud":         {"static": 0.20, "dynamic": 0.15},
    "snow":        {"static": 0.15, "dynamic": 0.10},
    "ice":         {"static": 0.07, "dynamic": 0.05},
    "random":      {"static": (0.05, 0.80), "dynamic": (0.05, 0.70)},
}

# Ordered from high to low friction — used for curriculum training phases
CURRICULUM_ORDER: List[str] = [
    "asphalt_dry", "asphalt_wet", "grass_dry", "grass_wet", "mud", "snow", "ice"
]

# Module-level singleton used by all other modules
config = EnvConfig()
