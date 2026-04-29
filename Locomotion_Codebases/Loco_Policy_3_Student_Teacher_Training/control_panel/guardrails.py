"""Safety guardrails for live parameter changes.

Validates reward weight changes, LR bounds, noise bounds, and S2R params
before they are applied to the running training.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

# Hard-frozen weights — cannot be changed even with --force
HARD_FROZEN = {"stumble", "body_height_tracking"}

# Sign constraints: positive = task reward, negative = penalty
POSITIVE_WEIGHTS = {"air_time", "base_angular_velocity", "base_linear_velocity",
                    "foot_clearance", "gait"}
NEGATIVE_WEIGHTS = {"action_smoothness", "air_time_variance", "base_motion",
                    "base_orientation", "base_pitch", "base_roll",
                    "foot_slip", "joint_acc", "joint_pos",
                    "joint_torques", "joint_vel", "terrain_relative_height",
                    "dof_pos_limits", "undesired_contacts", "motor_power",
                    "torque_limit"}

# Absolute bounds per weight (min, max)
WEIGHT_BOUNDS = {
    "air_time": (0.5, 15.0),
    "base_angular_velocity": (0.5, 15.0),
    "base_linear_velocity": (0.5, 15.0),
    "foot_clearance": (0.1, 10.0),
    "gait": (2.0, 20.0),
    "action_smoothness": (-3.0, -0.05),
    "air_time_variance": (-5.0, -0.1),
    "base_motion": (-5.0, -0.1),
    "base_orientation": (-5.0, 0.0),
    "base_pitch": (-5.0, -0.1),
    "base_roll": (-5.0, -0.1),
    "foot_slip": (-3.0, -0.05),
    "joint_acc": (-0.01, -1e-6),
    "joint_pos": (-2.0, -0.05),
    "joint_torques": (-0.01, -1e-5),
    "joint_vel": (-0.1, -1e-4),
    "terrain_relative_height": (-5.0, -0.1),
    "dof_pos_limits": (-10.0, -0.1),
    "undesired_contacts": (-5.0, -0.1),
    "motor_power": (-0.05, -1e-5),
    "torque_limit": (-2.0, -0.01),
}

# LR bounds
LR_ABS_MIN = 1e-7
LR_ABS_MAX = 1e-2

# Noise bounds
NOISE_ABS_MIN = 0.1
NOISE_ABS_MAX = 1.0

# S2R param bounds
S2R_BOUNDS = {
    "max_dropout_rate": (0.0, 0.3),
    "max_drift_rate": (0.0, 0.05),
    "max_spike_prob": (0.0, 0.05),
    "max_action_delay": (0, 5),
    "max_obs_delay": (0, 5),
}

# Max relative change per command (50%), overridable with force flag
MAX_RELATIVE_DELTA = 0.5


def validate_weight_change(name: str, new_value: float, current_value: float,
                           frozen: set = None, force: bool = False) -> tuple:
    """Validate a single reward weight change.

    Returns:
        (validated_value or None, list of messages)
    """
    msgs = []
    frozen = frozen or set()

    # Hard frozen
    if name in HARD_FROZEN:
        msgs.append(f"REJECTED: '{name}' is hard-frozen (Bug #22/#28b)")
        return None, msgs

    # User frozen
    if name in frozen:
        msgs.append(f"REJECTED: '{name}' is frozen by user")
        return None, msgs

    # Sign preservation
    if name in POSITIVE_WEIGHTS and new_value < 0:
        msgs.append(f"REJECTED: '{name}' is a task reward, must be positive (got {new_value})")
        return None, msgs
    if name in NEGATIVE_WEIGHTS and new_value > 0:
        msgs.append(f"REJECTED: '{name}' is a penalty, must be negative (got {new_value})")
        return None, msgs

    # Bounds check (clamp, don't reject)
    if name in WEIGHT_BOUNDS:
        lo, hi = WEIGHT_BOUNDS[name]
        if new_value < lo:
            msgs.append(f"CLAMPED: '{name}' {new_value} -> {lo} (min bound)")
            new_value = lo
        elif new_value > hi:
            msgs.append(f"CLAMPED: '{name}' {new_value} -> {hi} (max bound)")
            new_value = hi

    # Max delta check (skip if force or current is zero)
    if not force and current_value != 0:
        # For small values (abs < 1.0), use absolute delta with max 0.5
        # For larger values, use relative delta (50%)
        abs_current = abs(current_value)
        abs_delta = abs(new_value - current_value)
        if abs_current < 1.0:
            if abs_delta > 0.5:
                msgs.append(
                    f"REJECTED: '{name}' change {current_value} -> {new_value} "
                    f"(delta {abs_delta:.2f}, max 0.5 for small values). Use --force to override."
                )
                return None, msgs
        else:
            delta_pct = abs_delta / abs_current
            if delta_pct > MAX_RELATIVE_DELTA:
                msgs.append(
                    f"REJECTED: '{name}' change {current_value} -> {new_value} "
                    f"is {delta_pct:.0%} (max {MAX_RELATIVE_DELTA:.0%}). Use --force to override."
                )
                return None, msgs

    msgs.append(f"OK: '{name}' {current_value} -> {new_value}")
    return new_value, msgs


def validate_lr(lr_max: float = None, lr_min: float = None,
                current_max: float = None, current_min: float = None) -> tuple:
    """Validate LR bound changes. Returns (validated_max, validated_min, messages)."""
    msgs = []
    out_max = lr_max if lr_max is not None else current_max
    out_min = lr_min if lr_min is not None else current_min

    if out_max is not None:
        out_max = max(LR_ABS_MIN, min(LR_ABS_MAX, out_max))
    if out_min is not None:
        out_min = max(LR_ABS_MIN, min(LR_ABS_MAX, out_min))

    if out_max is not None and out_min is not None and out_min >= out_max:
        msgs.append(f"REJECTED: lr_min ({out_min}) must be < lr_max ({out_max})")
        return None, None, msgs

    msgs.append(f"OK: lr_max={out_max}, lr_min={out_min}")
    return out_max, out_min, msgs


def validate_noise(max_std: float = None, min_std: float = None) -> tuple:
    """Validate noise bound changes. Returns (validated_max, validated_min, messages)."""
    msgs = []
    if max_std is not None:
        max_std = max(NOISE_ABS_MIN, min(NOISE_ABS_MAX, max_std))
    if min_std is not None:
        min_std = max(NOISE_ABS_MIN, min(NOISE_ABS_MAX, min_std))

    if max_std is not None and min_std is not None and min_std >= max_std:
        msgs.append(f"REJECTED: min_std ({min_std}) must be < max_std ({max_std})")
        return None, None, msgs

    msgs.append(f"OK: noise max_std={max_std}, min_std={min_std}")
    return max_std, min_std, msgs


def validate_s2r_param(param: str, value) -> tuple:
    """Validate an S2R wrapper parameter change. Returns (validated_value, messages)."""
    msgs = []
    if param not in S2R_BOUNDS:
        msgs.append(f"REJECTED: unknown S2R param '{param}'. Valid: {list(S2R_BOUNDS.keys())}")
        return None, msgs

    lo, hi = S2R_BOUNDS[param]

    # Integer params
    if param in ("max_action_delay", "max_obs_delay"):
        value = int(round(value))

    value = max(lo, min(hi, value))
    msgs.append(f"OK: S2R {param} = {value}")
    return value, msgs
