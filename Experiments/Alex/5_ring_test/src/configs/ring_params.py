"""Ring terrain specifications, waypoint generation params, and scoring weights.

5-ring concentric arena layout:
  Ring 1 (0-10m)  — Flat (navigation warmup)
  Ring 2 (10-20m) — Low Friction
  Ring 3 (20-30m) — Vegetation + Drag
  Ring 4 (30-40m) — Boulder Field
  Ring 5 (40-50m) — Extreme Mixed (low friction + large boulders + step blocks)
"""

import numpy as np


# =============================================================================
# Arena geometry
# =============================================================================
ARENA_RADIUS = 50.0
ARENA_CENTER = (0.0, 0.0)
NUM_RINGS = 5
RING_WIDTH = 10.0
WAYPOINTS_PER_RING = 10
NUM_GROUND_SEGMENTS = 36       # per ring (10-deg arcs)
SPAWN_POSITION = (0.0, 0.0, 0.6)
EPISODE_TIMEOUT = 900.0        # 15 minutes
WAYPOINT_THRESHOLD = 1.5       # meters — advance when within this distance
STUCK_TIMEOUT = 30.0           # seconds — end if position unchanged

# =============================================================================
# Scoring
# =============================================================================
RING_WEIGHTS = [10, 20, 30, 40, 50]  # composite score weights per ring
MAX_SCORE = sum(RING_WEIGHTS)         # 150

# =============================================================================
# Ring terrain definitions
# =============================================================================
RING_PARAMS = [
    {
        "ring": 1,
        "r_inner": 0.0,
        "r_outer": 10.0,
        "terrain": "flat",
        "mu_static": 0.80,
        "mu_dynamic": 0.80,
        "obstacles": None,
        "drag_coeff": 0.0,
        "color": (0.55, 0.55, 0.55),  # grey
        "label": "Flat (navigation warmup)",
    },
    {
        "ring": 2,
        "r_inner": 10.0,
        "r_outer": 20.0,
        "terrain": "low_friction",
        "mu_static": 0.35,
        "mu_dynamic": 0.25,
        "obstacles": None,
        "drag_coeff": 0.0,
        "color": (0.45, 0.60, 0.85),  # ice blue
        "label": "Low Friction",
    },
    {
        "ring": 3,
        "r_inner": 20.0,
        "r_outer": 30.0,
        "terrain": "vegetation",
        "mu_static": 0.70,
        "mu_dynamic": 0.70,
        "obstacles": {
            "type": "stalks",
            "density": 5.0,          # stalks/m^2
            "height_range": (0.25, 0.35),
        },
        "drag_coeff": 5.0,
        "color": (0.15, 0.50, 0.10),  # green
        "label": "Vegetation + Drag",
    },
    {
        "ring": 4,
        "r_inner": 30.0,
        "r_outer": 40.0,
        "terrain": "boulder_field",
        "mu_static": 0.75,
        "mu_dynamic": 0.75,
        "obstacles": {
            "type": "boulders",
            "density": 2.0,           # boulders/m^2
            "edge_range": (0.15, 0.50),
        },
        "drag_coeff": 0.0,
        "color": (0.65, 0.50, 0.35),  # brown
        "label": "Boulder Field",
    },
    {
        "ring": 5,
        "r_inner": 40.0,
        "r_outer": 50.0,
        "terrain": "extreme_mixed",
        "mu_static": 0.25,
        "mu_dynamic": 0.15,
        "obstacles": {
            "type": "boulders",
            "density": 1.5,
            "edge_range": (0.40, 0.80),
        },
        "drag_coeff": 0.0,
        "color": (0.40, 0.25, 0.25),  # dark red-brown
        "label": "Extreme Mixed",
    },
]


def ring_midpoint_radius(ring_idx):
    """Get the midpoint radius for a ring (0-indexed).

    Ring 1 → 5m, Ring 2 → 15m, Ring 3 → 25m, Ring 4 → 35m, Ring 5 → 45m
    """
    params = RING_PARAMS[ring_idx]
    return (params["r_inner"] + params["r_outer"]) / 2.0


def get_ring_for_radius(r):
    """Get ring number (1-5) for a given radius. Returns 0 if inside ring 1 center."""
    for p in RING_PARAMS:
        if p["r_inner"] <= r < p["r_outer"]:
            return p["ring"]
    if r >= RING_PARAMS[-1]["r_outer"]:
        return NUM_RINGS
    return 1
