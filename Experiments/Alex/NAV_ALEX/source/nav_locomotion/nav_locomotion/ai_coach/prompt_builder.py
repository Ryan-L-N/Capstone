"""Prompt builder — constructs system and user prompts for the AI coach.

Encodes navigation-specific training rules, anti-crawl enforcement,
reward weight bounds, and troubleshooting guides into structured prompts
that guide Claude's decision-making.

Adapted from multi_robot_training/ai_trainer/prompt_builder.py for
navigation reward terms and exploration-mode training.
"""

from __future__ import annotations


def build_system_prompt(
    weight_bounds: dict[str, tuple[float, float]],
    max_changes: int = 3,
    max_delta_pct: float = 0.20,
) -> str:
    """Build the system prompt for the navigation AI coach.

    Args:
        weight_bounds: Dict of {term: (min, max)} for adjustable weights.
        max_changes: Maximum simultaneous weight changes.
        max_delta_pct: Maximum percentage change per weight.

    Returns:
        System prompt string.
    """
    bounds_table = "\n".join(
        f"  {term}: [{lo:.2f}, {hi:.2f}]"
        for term, (lo, hi) in sorted(weight_bounds.items())
    )

    return f"""You are an AI training coach for a quadruped robot navigation system.

## Your Role
You monitor training metrics and recommend reward weight adjustments to help the
robot learn terrain-aware navigation. The robot uses a depth camera to see ahead
and outputs velocity commands to a frozen locomotion policy.

## PRIORITY ORDER
1. **SPEED AND HEIGHT FIRST** — The robot must move forward AND stand upright.
   A walking robot at terrain 3 is better than a crawling robot at terrain 5.
2. **Stability** — Smooth velocity commands, no spinning, no oscillation.
3. **Terrain advancement** — Higher terrain levels mean harder obstacles.

## ANTI-CRAWL RULE (CRITICAL)
If mean_body_height < 0.30m for 2+ consecutive checks:
  → You MUST tighten terrain_relative_height toward -4.0
  → You MUST NOT loosen any penalty
  → The robot is belly-crawling — this is an exploit, not progress

If drag_penalty > 1.0 sustained:
  → Flag crawling behavior
  → Recommend tightening drag_penalty weight

## TRAINING PARAMETERS
- 30,000 max iterations, 2048 envs, 128 steps/env = 262,144 steps per iteration
- Checkpoints saved every 100 iterations (~26.2M steps between saves)
- Coach consulted every N iterations (configurable, default 100)

## HARD CONSTRAINTS
- Maximum {max_changes} weight changes at a time (more causes policy collapse)
- Maximum {int(max_delta_pct * 100)}% change per weight
- Positive weights must stay positive (rewards can't become penalties)
- Negative weights must stay negative (penalties can't become rewards)
- Penalties cannot be loosened until terrain level >= 3.0

## WEIGHT BOUNDS
{bounds_table}

## TERRAIN TYPES
The curriculum includes 10 terrain types. Two are surface-property zones:
- **friction_plane** (5%): Flat ground with randomized friction. Tests low-traction locomotion.
  No drag forces. The robot must adapt speed to avoid sliding.
- **vegetation_plane** (5%): Flat ground with velocity-dependent drag forces on feet.
  Simulates grass/mud/fluid resistance. The robot must push harder to maintain speed.
The remaining 8 types are geometric obstacles (stairs, boulders, waves, etc.).

## VEGETATION DRAG
The vegetation_drag term applies REAL physics drag forces (F = -coeff * v_foot) to feet.
It is both a physics modifier AND a reward signal. Weight is very small (-0.001) because
the drag force magnitude is already large. Do NOT increase this weight beyond -0.01 or
the penalty signal will dominate other rewards.

## TROUBLESHOOTING GUIDE
| Symptom | Likely Cause | Recommended Fix |
|---------|-------------|-----------------|
| Robot not moving | forward_velocity too low | Boost forward_velocity toward 12-15 |
| Spinning in place | angular_velocity penalty too low | Tighten angular_velocity toward -1.5 |
| Crawling/belly-sliding | Height/drag penalties too weak | Tighten terrain_relative_height AND drag_penalty |
| Too cautious (stops at obstacles) | Penalties too strong | Loosen cmd_smoothness (IF terrain >= 3) |
| Jerky movement | cmd_smoothness too weak | Tighten cmd_smoothness toward -2.0 |
| Dies immediately | forward_velocity too aggressive | Reduce forward_velocity, boost survival |
| Terrain plateau (300+ iters) | Reward landscape too flat | Boost forward_velocity OR loosen one penalty |
| Slipping on friction zones | Friction DR too aggressive | This is expected — robot must learn cautious speed |
| Stalling in vegetation | Drag too strong vs forward reward | Boost forward_velocity slightly |

## WHEN TO ACT
- **no_change**: Metrics trending well, no issues. This is the safest choice.
- **adjust_weights**: Clear problem in metrics (crawling, spinning, stalling).
- **adjust_lr**: Value loss unstable (but below emergency threshold).
- **emergency_stop**: NaN detected or catastrophic collapse.

## RESPONSE FORMAT
Return ONLY a JSON object (no markdown, no explanation outside JSON):
```json
{{
    "action": "no_change" | "adjust_weights" | "adjust_lr" | "emergency_stop",
    "reasoning": "Brief explanation of your analysis and decision",
    "weight_changes": {{"term_name": new_value, ...}},
    "lr_change": null | new_lr_value,
    "confidence": 0.0 to 1.0
}}
```

If action is "no_change", weight_changes should be empty {{}}.
If action is "adjust_weights", include 1-{max_changes} weight changes.
"""


def build_user_message(
    snapshot,
    recent_history: list,
    recent_decisions: list,
    plateau_detected: bool = False,
) -> str:
    """Build the user message with current metrics for coach analysis.

    Args:
        snapshot: Current MetricsSnapshot.
        recent_history: List of recent MetricsSnapshot objects.
        recent_decisions: List of recent CoachDecision objects.
        plateau_detected: Whether terrain level has stalled.

    Returns:
        User message string.
    """
    lines = [
        "## Current Metrics",
        f"Iteration: {snapshot.iteration}",
        f"Elapsed: {snapshot.elapsed_hours:.1f} hours",
        f"Mean reward: {snapshot.mean_reward:.2f}",
        f"Survival rate: {snapshot.survival_rate:.1%}",
        f"Flip rate: {snapshot.flip_rate:.1%}",
        f"Terrain level: {snapshot.mean_terrain_level:.2f}",
        f"Value loss: {snapshot.value_loss:.4f}",
        f"Noise std: {snapshot.noise_std:.4f}",
        f"Learning rate: {snapshot.learning_rate:.2e}",
        "",
        "## Nav-Specific Metrics",
        f"Forward distance: {snapshot.mean_forward_distance:.1f}m",
        f"Body height: {snapshot.mean_body_height:.3f}m",
        f"Drag penalty: {snapshot.mean_drag_penalty:.3f}",
        f"Vegetation drag force: {getattr(snapshot, 'mean_vegetation_drag', 0.0):.3f}",
        "",
    ]

    # Reward breakdown
    if snapshot.reward_breakdown:
        lines.append("## Reward Breakdown (per episode)")
        for term, value in sorted(snapshot.reward_breakdown.items()):
            lines.append(f"  {term}: {value:.3f}")
        lines.append("")

    # Current weights
    if snapshot.current_weights:
        lines.append("## Current Weights")
        for term, weight in sorted(snapshot.current_weights.items()):
            lines.append(f"  {term}: {weight:.4f}")
        lines.append("")

    # Trends
    lines.extend([
        "## Trends (slope over recent window)",
        f"  Reward trend: {snapshot.reward_trend:+.4f}",
        f"  Terrain trend: {snapshot.terrain_trend:+.4f}",
        f"  Value loss trend: {snapshot.value_loss_trend:+.4f}",
        "",
    ])

    # Recent history
    if recent_history:
        lines.append("## Recent History (last 5 checkpoints)")
        lines.append("  Iter | Reward | Terrain | Flip% | ValueLoss")
        for h in recent_history[-5:]:
            lines.append(
                f"  {h.iteration:6d} | {h.mean_reward:7.1f} | "
                f"{h.mean_terrain_level:7.2f} | {h.flip_rate:5.1%} | "
                f"{h.value_loss:.4f}"
            )
        lines.append("")

    # Plateau alert
    if plateau_detected:
        lines.extend([
            "## ⚠ PLATEAU DETECTED",
            "Terrain level has not improved for 300+ iterations.",
            '"no_change" is NOT acceptable — the robot is stuck.',
            "Recommended: boost forward_velocity OR reduce dominant penalty.",
            "",
        ])

    # Anti-crawl alert
    if snapshot.mean_body_height > 0 and snapshot.mean_body_height < 0.30:
        lines.extend([
            "## ⚠ ANTI-CRAWL ALERT",
            f"Body height {snapshot.mean_body_height:.3f}m is BELOW 0.30m threshold.",
            "The robot may be belly-crawling. You MUST tighten height penalty.",
            "",
        ])

    # Recent decisions
    if recent_decisions:
        lines.append("## Recent Coach Decisions (last 3)")
        for d in recent_decisions[-3:]:
            changes_str = ", ".join(f"{k}={v:.3f}" for k, v in d.weight_changes.items()) if d.weight_changes else "none"
            lines.append(f"  [{d.action}] {d.reasoning[:80]}... changes: {changes_str}")
        lines.append("")

    return "\n".join(lines)
