"""
Training Environment 1 — Reward Function
Locomotion quality rewards only. No navigation or goal-reaching signals.
No Isaac imports — safe to use anywhere.
"""

from env_config import EnvConfig, config as default_config


def compute_reward(
    vel_x: float,
    ang_vel_x: float,
    ang_vel_y: float,
    roll: float,
    pitch: float,
    height: float,
    cmd_vx: float,
    cmd_vx_prev: float,
    cmd_vy: float,
    cmd_yaw: float,
    pos_z: float,
    fell: bool,
    cfg: EnvConfig = default_config,
) -> tuple:
    """
    Compute one-step locomotion reward.

    All inputs are scalars (floats). Angles are in radians.

    Returns
    -------
    total  : float              — scalar reward for this step
    parts  : dict[str, float]   — individual components for logging/debugging
    """
    parts = {}

    # Forward motion: reward positive vel_x, penalize backward motion more heavily
    if vel_x >= 0.0:
        parts["forward"] = cfg.R_FORWARD * vel_x
    else:
        parts["forward"] = -cfg.R_FORWARD * 2.0 * abs(vel_x)

    # Body stability: penalize squared tilt angles (radians)
    parts["roll"]  = -cfg.R_ROLL  * (roll  ** 2)
    parts["pitch"] = -cfg.R_PITCH * (pitch ** 2)

    # Angular rate oscillation: penalize rocking and pitching (not yaw on flat terrain)
    parts["ang_rate"] = -cfg.R_ANG_RATE * (ang_vel_x ** 2 + ang_vel_y ** 2)

    # Height maintenance: penalize deviation from nominal standing height
    # height is raw pos_z; normalized so 1.0 = NOMINAL_HEIGHT
    h_norm = height / cfg.NOMINAL_HEIGHT
    parts["height"] = -cfg.R_HEIGHT * ((h_norm - 1.0) ** 2)

    # Command smoothness: penalize jerk (sudden changes in commanded forward speed)
    parts["smoothness"] = -cfg.R_SMOOTHNESS * ((cmd_vx - cmd_vx_prev) ** 2)

    # Lateral and yaw cost: discourage unnecessary lateral/turning commands
    parts["lat_yaw"] = -cfg.R_LAT_YAW * (cmd_vy ** 2 + cmd_yaw ** 2)

    # Survival bonus: small reward each step the robot stays upright
    parts["alive"] = cfg.R_ALIVE if pos_z > cfg.FALL_Z else 0.0

    # Fall terminal penalty: one-time, large negative
    parts["fall"] = cfg.R_FALL if fell else 0.0

    total = sum(parts.values())
    return total, parts
