"""Evaluation configuration constants for the 4-environment capstone test.

All values derived from rough_env_cfg.py and LESSONS_LEARNED.md.
"""

from .zone_params import ZONE_PARAMS, ZONE_LENGTH, ARENA_WIDTH, ARENA_LENGTH, NUM_ZONES

# --- Physics (matching rough_env_cfg.py:395-403) ---
PHYSICS_DT = 1.0 / 500.0       # 500 Hz simulation
RENDERING_DT = 10.0 / 500.0    # 50 Hz rendering
CONTROL_DT = 1.0 / 50.0        # 50 Hz control (decimation = 10)
DECIMATION = 10

# --- Episode ---
EPISODE_TIMEOUT = 600.0         # seconds per episode
MAX_CONTROL_STEPS = int(EPISODE_TIMEOUT / CONTROL_DT)  # 6000
FALL_THRESHOLD = 0.15           # meters — base height below this = fall
COMPLETION_X = 49.0             # meters — x >= this = completed

# --- Arena ---
SPAWN_POSITION = (0.0, 15.0, 0.6)  # center of Y-axis, 0.6m above ground

# --- Robot PD gains (from LESSONS_LEARNED.md deployment checklist) ---
STIFFNESS = 60.0                # Kp
DAMPING = 1.5                   # Kd
ACTION_SCALE = 0.25
HIP_EFFORT_LIMIT = 45.0        # N·m
KNEE_EFFORT_LIMIT = 110.0      # N·m

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
TARGET_VX = 1.0                 # m/s forward velocity
VX_RANGE = (-2.0, 3.0)         # training command range
VY_RANGE = (-1.5, 1.5)
WZ_RANGE = (-2.0, 2.0)
WAYPOINT_THRESHOLD = 0.5       # meters — advance when within this of waypoint x
