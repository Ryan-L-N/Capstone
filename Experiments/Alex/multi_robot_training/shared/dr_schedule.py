"""Progressive domain randomization schedule.

Gradually expands DR parameters from easy to hard over a configurable
number of iterations. This prevents early training collapse from
overly aggressive randomization.

Extracted from vision60_training/train_vision60.py lines 86-159.
Robot-agnostic — operates on env config DR event terms.

Created for AI2C Tech Capstone — MS for Autonomy, February 2026
"""


def lerp(start: float, end: float, fraction: float) -> float:
    """Linear interpolation: start + (end - start) * clamp(fraction, 0, 1)."""
    fraction = max(0.0, min(1.0, fraction))
    return start + (end - start) * fraction


# DR schedule: (start_value, end_value) for each parameter
DR_SCHEDULE = {
    # Friction
    "static_friction_min":  (0.3,  0.1),
    "static_friction_max":  (1.3,  1.5),
    "dynamic_friction_min": (0.25, 0.08),
    "dynamic_friction_max": (1.1,  1.2),
    # Push robot
    "push_velocity":        (0.5,  1.0),
    "push_interval_min":    (10.0, 6.0),
    "push_interval_max":    (15.0, 13.0),
    # External forces
    "ext_force":            (3.0,  6.0),
    "ext_torque":           (1.0,  2.5),
    # Mass
    "mass_offset":          (5.0,  7.0),
    # Joint velocity reset
    "joint_vel_range":      (2.5,  3.0),
}


def update_dr_params(env, iteration: int, expansion_iters: int) -> dict:
    """Progressively expand domain randomization ranges.

    Args:
        env: Gymnasium environment (wrapped).
        iteration: Current training iteration.
        expansion_iters: Total iterations over which DR expands.

    Returns:
        Dict of current DR values for logging.
    """
    fraction = min(iteration / max(expansion_iters, 1), 1.0)
    cfg = env.unwrapped.cfg

    # --- Friction ---
    sf_min = lerp(*DR_SCHEDULE["static_friction_min"], fraction)
    sf_max = lerp(*DR_SCHEDULE["static_friction_max"], fraction)
    df_min = lerp(*DR_SCHEDULE["dynamic_friction_min"], fraction)
    df_max = lerp(*DR_SCHEDULE["dynamic_friction_max"], fraction)
    cfg.events.physics_material.params["static_friction_range"] = (sf_min, sf_max)
    cfg.events.physics_material.params["dynamic_friction_range"] = (df_min, df_max)

    # --- Push robot ---
    push_vel = lerp(*DR_SCHEDULE["push_velocity"], fraction)
    cfg.events.push_robot.params["velocity_range"] = {
        "x": (-push_vel, push_vel), "y": (-push_vel, push_vel)
    }
    push_min = lerp(*DR_SCHEDULE["push_interval_min"], fraction)
    push_max = lerp(*DR_SCHEDULE["push_interval_max"], fraction)
    cfg.events.push_robot.interval_range_s = (push_min, push_max)

    # --- External force / torque ---
    ext_force = lerp(*DR_SCHEDULE["ext_force"], fraction)
    ext_torque = lerp(*DR_SCHEDULE["ext_torque"], fraction)
    cfg.events.base_external_force_torque.params["force_range"] = (-ext_force, ext_force)
    cfg.events.base_external_force_torque.params["torque_range"] = (-ext_torque, ext_torque)

    # --- Mass offset ---
    mass_offset = lerp(*DR_SCHEDULE["mass_offset"], fraction)
    cfg.events.add_base_mass.params["mass_distribution_params"] = (-mass_offset, mass_offset)

    # --- Joint velocity reset range ---
    jv_range = lerp(*DR_SCHEDULE["joint_vel_range"], fraction)
    cfg.events.reset_robot_joints.params["velocity_range"] = (-jv_range, jv_range)

    return {
        "dr_fraction": fraction,
        "friction_range": f"[{sf_min:.2f}, {sf_max:.2f}]",
        "push_vel": push_vel,
        "ext_force": ext_force,
        "mass_offset": mass_offset,
    }
