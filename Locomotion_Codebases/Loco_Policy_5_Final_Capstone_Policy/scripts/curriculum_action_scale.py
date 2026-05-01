"""Phase-v6 helper: curriculum-tied action_scale + reward-weight ramp.

Reads mean(terrain_levels) from the env and updates action_scale plus
matching action_smoothness/joint_torques weights. Linear interp:
  - terrain_levels = 0  → action_scale = 0.30 (matches 22100 training)
  - terrain_levels ≥ 5  → action_scale = 0.50 (Phase-Final-equivalent authority)
And action_smoothness/joint_torques weights linearly interpolate between
Phase-9 (-1.5, -1.0e-3) at scale 0.3 and Phase-Final (-2.0, -1.5e-3) at 0.5.

Called once per iter from train_parkour_nav.py's _update_with_schedule.
Updates take effect on the NEXT rollout (action term reads cfg.scale per
step; reward manager reads cfg.weight per step).

The trained 22100 policy outputs values that × 0.3 produce a working
gait. Ramping scale only as terrain promotes means:
- At level 0 (collapsed state), scale stays at 0.3 — preserves 22100's
  converged friction/grass capability
- As curriculum advances toward stair-heavy levels, scale rises so the
  policy gets more authority where it's needed
- The critic adapts gradually because the per-iter delta is tiny
"""

from typing import Optional

# Tuning constants — tweak these here, no other code changes
_LEVEL_LO = 0.0        # action_scale = 0.30 at this level
_LEVEL_HI = 5.0        # action_scale = 0.50 at this level (and above)
_SCALE_LO = 0.30
_SCALE_HI = 0.50
_SMOOTH_LO = -1.5      # action_smoothness weight at scale 0.3
_SMOOTH_HI = -2.0      # action_smoothness weight at scale 0.5
_TORQUE_LO = -1.0e-3   # joint_torques weight at scale 0.3
_TORQUE_HI = -1.5e-3   # joint_torques weight at scale 0.5

# Track previous values to detect change + log
_prev_scale: list = [None]


def _interp(x: float, x_lo: float, x_hi: float, y_lo: float, y_hi: float) -> float:
    """Linear interp clamped to [y_lo, y_hi] — handles x_hi=x_lo gracefully."""
    if x_hi <= x_lo:
        return y_lo
    t = max(0.0, min(1.0, (x - x_lo) / (x_hi - x_lo)))
    return y_lo + t * (y_hi - y_lo)


def update_curriculum_action_scale(runner, iter_num: int) -> Optional[dict]:
    """Read terrain_levels, update action_scale + matching reward weights.

    Returns a dict of new values for telemetry, or None if env doesn't expose
    terrain_levels (e.g., during warmup before scene is ready).
    """
    try:
        env = runner.env.unwrapped
    except AttributeError:
        env = getattr(runner, "env", None)
        if env is None:
            return None

    # Read terrain_levels — per-env tensor of shape [num_envs]
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or not hasattr(terrain, "terrain_levels"):
        return None
    levels_tensor = terrain.terrain_levels
    mean_level = float(levels_tensor.float().mean().item())

    # Compute curriculum-aware values
    scale = _interp(mean_level, _LEVEL_LO, _LEVEL_HI, _SCALE_LO, _SCALE_HI)
    smooth = _interp(mean_level, _LEVEL_LO, _LEVEL_HI, _SMOOTH_LO, _SMOOTH_HI)
    torque = _interp(mean_level, _LEVEL_LO, _LEVEL_HI, _TORQUE_LO, _TORQUE_HI)

    # Apply action_scale
    try:
        action_term = env.action_manager.get_term("joint_pos")
        # Update the cfg; ManagerBasedRLEnv's action term re-reads scale each step
        action_term.cfg.scale = scale
        # If the term has a buffered `_scale` tensor, update it too
        if hasattr(action_term, "_scale"):
            try:
                action_term._scale.fill_(scale)
            except Exception:
                pass
    except Exception as e:
        if iter_num % 100 == 0:
            print(f"[CUR-SCALE] WARN: could not update action_scale: {e}", flush=True)

    # Apply reward weights
    try:
        rm = env.reward_manager
        rm.get_term_cfg("action_smoothness").weight = smooth
        rm.get_term_cfg("joint_torques").weight = torque
    except Exception as e:
        if iter_num % 100 == 0:
            print(f"[CUR-SCALE] WARN: could not update reward weights: {e}", flush=True)

    # Log every 25 iters (reduced verbosity), or whenever scale changes ≥0.01
    if (
        _prev_scale[0] is None
        or abs(scale - _prev_scale[0]) > 0.01
        or iter_num % 25 == 0
    ):
        print(
            f"[CUR-SCALE] iter={iter_num} mean_terrain={mean_level:.3f} "
            f"action_scale={scale:.3f} smoothness={smooth:.3f} torque={torque:.2e}",
            flush=True,
        )
        _prev_scale[0] = scale

    return {
        "mean_terrain_level": mean_level,
        "action_scale": scale,
        "action_smoothness_weight": smooth,
        "joint_torques_weight": torque,
    }
