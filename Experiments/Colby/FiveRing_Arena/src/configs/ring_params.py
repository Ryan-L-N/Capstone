"""Quadrant terrain specifications, waypoint generation params, and scoring weights.

4-quadrant arena layout (50m radius):
  Q1 (0-90°)   — Friction (decreasing mu outward)
  Q2 (90-180°) — Grass / Vegetation (increasing density + drag)
  Q3 (180-270°)— Boulders (increasing size)
  Q4 (270-360°)— Stairs (pyramid waypoints, increasing height)

Each quadrant has 5 difficulty levels at radii 0-10, 10-20, 20-30, 30-40, 40-50m.
2 waypoints per level per quadrant = 10 WPs/quadrant, 40 total + 3 transitions = 43.
"""

import numpy as np


# =============================================================================
# Arena geometry
# =============================================================================
ARENA_RADIUS = 50.0
ARENA_CENTER = (0.0, 0.0)
NUM_QUADRANTS = 4
NUM_LEVELS = 5
LEVEL_WIDTH = 10.0              # radial width per difficulty level
WPS_PER_LEVEL = 2               # waypoints per level per quadrant
WPS_PER_QUADRANT = NUM_LEVELS * WPS_PER_LEVEL  # 10
NUM_GROUND_SEGMENTS = 9         # per level per quadrant (10-deg arcs within 90-deg)
SPAWN_POSITION = (0.0, 0.0, 0.6)
EPISODE_TIMEOUT = 900.0         # 15 minutes
WAYPOINT_THRESHOLD = 1.5        # meters
STUCK_TIMEOUT = 30.0            # seconds

# =============================================================================
# Scoring
# =============================================================================
LEVEL_WEIGHTS = [10, 20, 30, 40, 50]  # per level (inner to outer)
MAX_SCORE_PER_QUADRANT = sum(LEVEL_WEIGHTS)  # 150
MAX_SCORE = MAX_SCORE_PER_QUADRANT * NUM_QUADRANTS  # 600

# =============================================================================
# Quadrant definitions
# =============================================================================
# Angles in radians: each quadrant spans 90 degrees
QUADRANT_DEFS = [
    {
        "name": "friction",
        "label": "Friction",
        "angle_start": 0.0,
        "angle_end": np.pi / 2,
        "color": (0.45, 0.60, 0.85),  # ice blue
    },
    {
        "name": "grass",
        "label": "Grass",
        "angle_start": np.pi / 2,
        "angle_end": np.pi,
        "color": (0.15, 0.50, 0.10),  # green
    },
    {
        "name": "boulder",
        "label": "Boulders",
        "angle_start": np.pi,
        "angle_end": 3 * np.pi / 2,
        "color": (0.65, 0.50, 0.35),  # brown
    },
    {
        "name": "stairs",
        "label": "Stairs",
        "angle_start": 3 * np.pi / 2,
        "angle_end": 2 * np.pi,
        "color": (0.55, 0.55, 0.52),  # concrete grey
    },
]

# =============================================================================
# Per-level specs for each quadrant
# =============================================================================
FRICTION_LEVELS = [
    {"level": 1, "mu_static": 0.90, "mu_dynamic": 0.80, "label": "60-grit sandpaper"},
    {"level": 2, "mu_static": 0.70, "mu_dynamic": 0.60, "label": "Dry rubber"},
    {"level": 3, "mu_static": 0.50, "mu_dynamic": 0.40, "label": "Wet concrete"},
    {"level": 4, "mu_static": 0.35, "mu_dynamic": 0.25, "label": "Wet ice"},
    {"level": 5, "mu_static": 0.30, "mu_dynamic": 0.20, "label": "Oil on steel"},
]

GRASS_LEVELS = [
    {"level": 1, "stalk_density": 2,  "drag_coeff": 2.0,  "height_range": (0.15, 0.25), "label": "Thin grass"},
    {"level": 2, "stalk_density": 5,  "drag_coeff": 5.0,  "height_range": (0.25, 0.35), "label": "Medium lawn"},
    {"level": 3, "stalk_density": 10, "drag_coeff": 10.0, "height_range": (0.30, 0.45), "label": "Thick grass"},
    {"level": 4, "stalk_density": 15, "drag_coeff": 15.0, "height_range": (0.35, 0.50), "label": "Dense brush"},
    {"level": 5, "stalk_density": 20, "drag_coeff": 20.0, "height_range": (0.40, 0.55), "label": "Jungle"},
]

BOULDER_LEVELS = [
    {"level": 1, "edge_range": (0.03, 0.05), "density": 15, "label": "Gravel"},
    {"level": 2, "edge_range": (0.10, 0.15), "density": 8,  "label": "River rocks"},
    {"level": 3, "edge_range": (0.25, 0.35), "density": 4,  "label": "Large rocks"},
    {"level": 4, "edge_range": (0.50, 0.70), "density": 2,  "label": "Small boulders"},
    {"level": 5, "edge_range": (0.80, 1.20), "density": 1,  "label": "Large boulders"},
]

STAIRS_LEVELS = [
    {"level": 1, "step_height": 0.03, "step_depth": 0.30, "num_steps": 5,  "label": "Access ramp"},
    {"level": 2, "step_height": 0.08, "step_depth": 0.30, "num_steps": 6,  "label": "Low residential"},
    {"level": 3, "step_height": 0.13, "step_depth": 0.30, "num_steps": 7,  "label": "Standard residential"},
    {"level": 4, "step_height": 0.18, "step_depth": 0.30, "num_steps": 8,  "label": "Steep commercial"},
    {"level": 5, "step_height": 0.23, "step_depth": 0.30, "num_steps": 10, "label": "Maximum challenge"},
]

# Collect all level specs by quadrant name
QUADRANT_LEVELS = {
    "friction": FRICTION_LEVELS,
    "grass": GRASS_LEVELS,
    "boulder": BOULDER_LEVELS,
    "stairs": STAIRS_LEVELS,
}


# =============================================================================
# Helper functions
# =============================================================================
def level_radius_range(level_idx):
    """Get (r_inner, r_outer) for a difficulty level (0-indexed)."""
    r_inner = level_idx * LEVEL_WIDTH
    r_outer = (level_idx + 1) * LEVEL_WIDTH
    return r_inner, r_outer


def level_midpoint_radius(level_idx):
    """Get midpoint radius for a level (0-indexed)."""
    return (level_idx + 0.5) * LEVEL_WIDTH


def quadrant_angle_center(quad_idx):
    """Get the center angle of a quadrant (0-indexed)."""
    qdef = QUADRANT_DEFS[quad_idx]
    return (qdef["angle_start"] + qdef["angle_end"]) / 2.0


def get_quadrant_for_angle(angle):
    """Get quadrant index (0-3) for a given angle in radians.

    Angle is normalized to [0, 2*pi).
    """
    angle = angle % (2 * np.pi)
    for i, qdef in enumerate(QUADRANT_DEFS):
        a_start = qdef["angle_start"]
        a_end = qdef["angle_end"]
        if a_start <= angle < a_end:
            return i
    return 0  # fallback


def get_level_for_radius(r):
    """Get level number (1-5) for a given radius."""
    level = int(r / LEVEL_WIDTH)
    return max(1, min(NUM_LEVELS, level + 1))


def get_quadrant_and_level(x, y):
    """Get (quadrant_index, level_number) for a world position."""
    r = np.sqrt(x * x + y * y)
    angle = np.arctan2(y, x) % (2 * np.pi)
    return get_quadrant_for_angle(angle), get_level_for_radius(r)


def pyramid_summit_height(stairs_level):
    """Get the height of the top of a stair pyramid for a given stairs level dict."""
    return stairs_level["step_height"] * stairs_level["num_steps"]
