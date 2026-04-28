"""Evaluation configuration constants for the 5-ring gauntlet.

Copied from 4_env_test/src/configs/eval_cfg.py with ring-specific overrides.
"""

# --- Physics (matching rough_env_cfg.py:395-403) ---
PHYSICS_DT = 1.0 / 500.0       # 500 Hz simulation
RENDERING_DT = 10.0 / 500.0    # 50 Hz rendering
CONTROL_DT = 1.0 / 50.0        # 50 Hz control (decimation = 10)
DECIMATION = 10

# --- Episode ---
EPISODE_TIMEOUT = 900.0         # seconds per episode (15 min — longer for ring navigation)
MAX_CONTROL_STEPS = int(EPISODE_TIMEOUT / CONTROL_DT)  # 45000
FALL_THRESHOLD = 0.15           # meters — base height below this = fall
STUCK_TIMEOUT = 30.0            # seconds — end episode if position unchanged

# --- Arena ---
SPAWN_POSITION = (0.0, 0.0, 0.6)  # center of ring arena
ARENA_RADIUS = 50.0

# =============================================================================
# Spot PD gains (from LESSONS_LEARNED.md deployment checklist)
# =============================================================================
STIFFNESS = 60.0                # Kp
DAMPING = 1.5                   # Kd
ACTION_SCALE = 0.25
HIP_EFFORT_LIMIT = 45.0        # N·m
KNEE_EFFORT_LIMIT = 110.0      # N·m

# =============================================================================
# Vision60 PD gains (from vision60_asset_cfg.py / URDF)
# =============================================================================
V60_STIFFNESS = 80.0            # Kp
V60_DAMPING = 2.0               # Kd
V60_ACTION_SCALE = 0.25
V60_EFFORT_LIMIT = 87.5         # N·m (uniform, all joints)
V60_SPAWN_POSITION = (0.0, 0.0, 0.65)  # slightly higher than Spot

# --- Per-robot config dict for easy lookup ---
ROBOT_CONFIGS = {
    "spot": {
        "stiffness": STIFFNESS,
        "damping": DAMPING,
        "action_scale": ACTION_SCALE,
        "spawn_position": SPAWN_POSITION,
        "fall_threshold": FALL_THRESHOLD,
    },
    "vision60": {
        "stiffness": V60_STIFFNESS,
        "damping": V60_DAMPING,
        "action_scale": V60_ACTION_SCALE,
        "spawn_position": V60_SPAWN_POSITION,
        "fall_threshold": 0.20,
    },
}

# --- Observation space ---
OBS_DIM = 235                   # 48 proprioceptive + 187 height scan
PROPRIO_DIM = 48
HEIGHT_SCAN_DIM = 187
HEIGHT_SCAN_FILL = 0.0          # CRITICAL: 0.0 for flat ground, NOT 1.0

# --- Solver ---
SOLVER_POS_ITERS = 4
SOLVER_VEL_ITERS = 0

# --- Waypoint navigation ---
KP_YAW = 2.0
VX_RANGE = (-2.0, 3.0)
VY_RANGE = (-1.5, 1.5)
WZ_RANGE = (-2.0, 2.0)
