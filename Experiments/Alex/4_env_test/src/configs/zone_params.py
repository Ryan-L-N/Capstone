"""Zone specifications for all 4 environments.

Each environment has 5 zones of 10m along the X-axis (0-50m).
Arena is 30m wide along Y-axis.

Usage:
    from src.configs.zone_params import ZONE_PARAMS
    friction_zones = ZONE_PARAMS["friction"]
    # -> list of 5 dicts with zone-specific parameters
"""

ZONE_LENGTH = 10.0  # meters per zone
ARENA_WIDTH = 30.0  # meters (Y-axis)
ARENA_LENGTH = 50.0  # meters (X-axis)
NUM_ZONES = 5
TRANSITION_STEPS = 5  # gradual height ramp at start of zones 2-5

ZONE_PARAMS = {
    "friction": [
        {"zone": 1, "x_start": 0.0,  "x_end": 10.0, "mu_static": 0.90, "mu_dynamic": 0.80, "label": "60-grit sandpaper"},
        {"zone": 2, "x_start": 10.0, "x_end": 20.0, "mu_static": 0.60, "mu_dynamic": 0.50, "label": "Dry rubber on concrete"},
        {"zone": 3, "x_start": 20.0, "x_end": 30.0, "mu_static": 0.35, "mu_dynamic": 0.25, "label": "Wet concrete"},
        {"zone": 4, "x_start": 30.0, "x_end": 40.0, "mu_static": 0.15, "mu_dynamic": 0.08, "label": "Wet ice"},
        {"zone": 5, "x_start": 40.0, "x_end": 50.0, "mu_static": 0.05, "mu_dynamic": 0.02, "label": "Oil on polished steel"},
    ],
    "grass": [
        {"zone": 1, "x_start": 0.0,  "x_end": 10.0, "stalk_density": 0,  "drag_coeff": 0.5,  "height_range": None,          "label": "Light fluid"},
        {"zone": 2, "x_start": 10.0, "x_end": 20.0, "stalk_density": 2,  "drag_coeff": 2.0,  "height_range": (0.15, 0.25),  "label": "Thin grass"},
        {"zone": 3, "x_start": 20.0, "x_end": 30.0, "stalk_density": 5,  "drag_coeff": 5.0,  "height_range": (0.25, 0.35),  "label": "Medium lawn"},
        {"zone": 4, "x_start": 30.0, "x_end": 40.0, "stalk_density": 10, "drag_coeff": 10.0, "height_range": (0.30, 0.45),  "label": "Thick grass"},
        {"zone": 5, "x_start": 40.0, "x_end": 50.0, "stalk_density": 20, "drag_coeff": 20.0, "height_range": (0.35, 0.50),  "label": "Dense brush"},
    ],
    "boulder": [
        {"zone": 1, "x_start": 0.0,  "x_end": 10.0, "edge_range": (0.03, 0.05), "density": 15, "count": 4500, "label": "Gravel"},
        {"zone": 2, "x_start": 10.0, "x_end": 20.0, "edge_range": (0.10, 0.15), "density": 8,  "count": 2400, "label": "River rocks"},
        {"zone": 3, "x_start": 20.0, "x_end": 30.0, "edge_range": (0.25, 0.35), "density": 4,  "count": 1200, "label": "Large rocks"},
        {"zone": 4, "x_start": 30.0, "x_end": 40.0, "edge_range": (0.50, 0.70), "density": 2,  "count": 600,  "label": "Small boulders"},
        {"zone": 5, "x_start": 40.0, "x_end": 50.0, "edge_range": (0.80, 1.20), "density": 1,  "count": 300,  "label": "Large boulders"},
    ],
    "stairs": [
        {"zone": 1, "x_start": 0.0,  "x_end": 10.0, "step_height": 0.03, "step_depth": 0.30, "num_steps": 33, "label": "Access ramp"},
        {"zone": 2, "x_start": 10.0, "x_end": 20.0, "step_height": 0.08, "step_depth": 0.30, "num_steps": 33, "label": "Low residential"},
        {"zone": 3, "x_start": 20.0, "x_end": 30.0, "step_height": 0.13, "step_depth": 0.30, "num_steps": 33, "label": "Standard residential"},
        {"zone": 4, "x_start": 30.0, "x_end": 40.0, "step_height": 0.18, "step_depth": 0.30, "num_steps": 33, "label": "Steep commercial"},
        {"zone": 5, "x_start": 40.0, "x_end": 50.0, "step_height": 0.23, "step_depth": 0.30, "num_steps": 33, "label": "Maximum challenge"},
    ],
}

# Boulder shape distribution (25% each)
BOULDER_SHAPES = {
    "D8": {"name": "octahedron", "weight": 0.25},
    "D10": {"name": "trapezohedron", "weight": 0.25},
    "D12": {"name": "dodecahedron", "weight": 0.25},
    "D20": {"name": "icosahedron", "weight": 0.25},
}


def _transition_riser(prev_h, curr_h, step_idx):
    """Riser height for a transition step (linear interpolation).

    With TRANSITION_STEPS=5, fractions are 1/6, 2/6, ..., 5/6 of the way
    from prev_h to curr_h, so the maximum jump between any two
    consecutive steps is (curr_h - prev_h) / 6.
    """
    frac = (step_idx + 1) / (TRANSITION_STEPS + 1)
    return prev_h + (curr_h - prev_h) * frac


def get_stair_elevation(x_pos):
    """Compute expected stair surface height at a given X position.

    Walks through stair zones cumulatively — each zone's base elevation
    starts where the previous zone ended (steps chain upward).

    Zones 2-5 have 5 transition steps at their start whose riser heights
    linearly ramp from the previous zone's step_height to the current one,
    smoothing out the height jump at zone boundaries.

    Args:
        x_pos: Robot X position in meters (0-50m arena).

    Returns:
        float: Expected ground surface height (Z) in meters.
    """
    zones = ZONE_PARAMS["stairs"]
    cumulative_height = 0.0
    prev_step_height = None

    for zone in zones:
        x_start = zone["x_start"]
        x_end = zone["x_end"]
        step_height = zone["step_height"]
        step_depth = zone["step_depth"]
        num_steps = zone["num_steps"]

        if x_pos < x_start:
            return cumulative_height

        if x_pos >= x_end:
            # Past this zone — add all steps
            if prev_step_height is not None:
                for t in range(TRANSITION_STEPS):
                    cumulative_height += _transition_riser(prev_step_height, step_height, t)
                cumulative_height += step_height * (num_steps - TRANSITION_STEPS)
            else:
                cumulative_height += step_height * num_steps
            prev_step_height = step_height
            continue

        # Within this zone — compute partial elevation
        local_x = x_pos - x_start
        steps_climbed = min(int(local_x / step_depth), num_steps)

        if prev_step_height is not None and steps_climbed > 0:
            n_trans = min(steps_climbed, TRANSITION_STEPS)
            for s in range(n_trans):
                cumulative_height += _transition_riser(prev_step_height, step_height, s)
            cumulative_height += step_height * (steps_climbed - n_trans)
        else:
            cumulative_height += step_height * steps_climbed

        return cumulative_height

    return cumulative_height
