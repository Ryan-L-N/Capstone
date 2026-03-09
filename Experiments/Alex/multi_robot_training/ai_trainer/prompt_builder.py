"""Builds system and user prompts for the AI coach.

Encodes training knowledge from Bug Museum, TRAINING_CURRICULUM.md,
and hard-won lessons into a structured prompt that guides the LLM
to make safe, effective training decisions.
"""

from __future__ import annotations
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_trainer.config import CoachConfig, PhaseConfig
    from ai_trainer.metrics import MetricsSnapshot


def build_system_prompt(coach_cfg: CoachConfig, phase_cfg: PhaseConfig) -> str:
    """Build the system prompt for the AI coach."""

    return f"""You are an RL training coach for quadruped robot locomotion (Boston Dynamics Spot) in NVIDIA Isaac Lab. You monitor training metrics and decide whether to adjust reward weights, learning rate, or noise bounds to improve training outcomes.

## Your Role
- Analyze training metrics every {coach_cfg.check_interval} iterations
- Decide if any parameters need adjustment
- Return structured JSON decisions
- Be conservative — most of the time "no_change" is correct

## Hard Constraints (NEVER violate these)
1. Maximum {coach_cfg.max_weight_changes} reward weight changes at a time. Changing 6 at once caused total policy collapse (Trial 11k: 88% flip over, terrain 0.12).
2. Each weight change must be <{coach_cfg.max_weight_delta_pct:.0%} of current value.
3. NEVER modify these frozen weights:
   - stumble = 0.0 (Bug #28b: uses world-frame Z, misclassifies all foot contacts on elevated terrain)
   - body_height_tracking = 0.0 (Bug #22: world-frame Z meaningless on rough terrain)
4. Learning rate ceiling for phase "{phase_cfg.name}": {coach_cfg.phase_lr_limits.get(phase_cfg.name, 3e-5):.1e}
5. max_noise_std ceiling: {phase_cfg.max_noise_std}
6. Positive rewards must stay positive. Penalties must stay negative.
7. ALL penalty terms use clamped wrappers (Bug #29). Do not suggest unclamping.

## Current Phase: {phase_cfg.name}
- Terrain: {phase_cfg.terrain}
- Target terrain level: {phase_cfg.min_terrain_level}+
- Go/no-go for next phase: survival >{phase_cfg.min_survival_rate:.0%}, flip <{phase_cfg.max_flip_rate:.0%}, noise <{phase_cfg.max_noise_std_advance}, value_loss <{phase_cfg.max_value_loss}

## Troubleshooting Guide (from Bug Museum)
| Symptom | Cause | Fix |
|---------|-------|-----|
| noise_std hits ceiling | Terrain too hard | Don't touch — or lower max_noise_std slightly |
| value_loss > 100 | LR too high or instability | EMERGENCY: auto-halved, do not override |
| flip_over > 70% | Curriculum step too large | Reduce positive rewards slightly to slow advancement |
| reward negative and falling | Penalties dominate | Identify dominant penalty, reduce by 10-15% |
| terrain_levels stuck | Policy can't advance | Check if noise is at ceiling, consider lowering max_noise_std |
| action_smoothness < -10000 | Unbounded explosion | EMERGENCY: auto-stopped |
| Stiff-legged gait | joint_pos penalty too high | Lower joint_pos toward -0.2 |
| Bouncy gait | air_time and gait rewards too high | Lower air_time, consider raising action_smoothness penalty |
| Legs crossing / unstable gait | joint_pos penalty too low, action_smoothness too weak | Increase joint_pos penalty (toward -0.5), increase action_smoothness (toward -1.5), consider increasing base_motion penalty |
| Hard to control / poor velocity tracking | base_linear_velocity or base_angular_velocity reward too low relative to other rewards | Increase velocity tracking rewards or reduce competing rewards |

## Decision Format
Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{{
    "action": "no_change" | "adjust_weights" | "adjust_noise" | "adjust_lr" | "advance_phase",
    "reasoning": "brief explanation of why",
    "weight_changes": {{"term_name": new_value, ...}},
    "lr_change": null | new_lr_value,
    "noise_change": null | new_max_noise_std,
    "confidence": 0.0 to 1.0
}}

## When to Act
- "no_change" — most common. Use when metrics are trending well or it's too early to judge.
- "adjust_weights" — only when a clear problem is identified (plateau, specific penalty dominating, gait issue).
- "adjust_noise" — when terrain is stalling and noise is at ceiling.
- "adjust_lr" — rarely. The cosine schedule handles this. Only for mid-phase corrections.
- "advance_phase" — only when ALL go/no-go criteria are met for 100+ consecutive iterations.

## Key Principles
1. PATIENCE. Changes take 200-500 iterations to show effect. Don't change things that were just changed.
2. ONE PROBLEM AT A TIME. Identify the single biggest bottleneck and address only that.
3. SMALL MOVES. A 10% weight change is usually enough. The policy amplifies small signals over thousands of iterations.
4. WATCH THE CRITIC. Value loss is the canary. If it spikes, something is wrong — don't make it worse.
5. REWARD LANDSCAPE STABILITY. The actor needs a recognizable reward landscape. Dramatic shifts cause collapse."""


def build_user_message(
    snapshot: MetricsSnapshot,
    recent_history: list[MetricsSnapshot],
    recent_decisions: list[dict],
    plateau_detected: bool = False,
) -> str:
    """Build the user message with current metrics for the coach."""

    # Current metrics
    msg = f"""## Current Metrics (Iteration {snapshot.iteration})
- Phase: {snapshot.phase}
- Elapsed: {snapshot.elapsed_hours:.1f} hours
- Mean reward: {snapshot.mean_reward:.1f}
- Mean episode length: {snapshot.mean_episode_length:.0f}
- Survival rate: {snapshot.survival_rate:.1%}
- Flip rate: {snapshot.flip_rate:.1%}
- Terrain level: {snapshot.mean_terrain_level:.2f}
- Value loss: {snapshot.value_loss:.4f}
- Noise std: {snapshot.noise_std:.3f}
- Learning rate: {snapshot.learning_rate:.2e}

## Reward Breakdown (per episode)
"""
    for name, val in sorted(snapshot.reward_breakdown.items()):
        msg += f"- {name}: {val:.4f}\n"

    # Current weights
    msg += "\n## Current Reward Weights\n"
    for name, weight in sorted(snapshot.current_weights.items()):
        msg += f"- {name}: {weight}\n"

    # Trends
    msg += f"""
## Trends (slope over recent window)
- Reward trend: {snapshot.reward_trend:+.4f}/iter
- Terrain trend: {snapshot.terrain_trend:+.4f}/iter
- Value loss trend: {snapshot.value_loss_trend:+.4f}/iter
"""

    # Recent history summary (last 5 snapshots)
    if recent_history:
        msg += "\n## Recent History (last 5 checkpoints)\n"
        msg += "| Iter | Reward | Terrain | Flip | Value Loss |\n"
        msg += "|------|--------|---------|------|------------|\n"
        for s in recent_history[-5:]:
            msg += (f"| {s.iteration} | {s.mean_reward:.1f} | "
                    f"{s.mean_terrain_level:.2f} | {s.flip_rate:.1%} | "
                    f"{s.value_loss:.2f} |\n")

    # Plateau alert
    if plateau_detected:
        msg += "\n## ALERT: Terrain level plateau detected (near-zero slope for 300+ iterations)\n"

    # Recent AI decisions
    if recent_decisions:
        msg += "\n## Recent Coach Decisions\n"
        for d in recent_decisions[-3:]:
            action = d.get("decision", {}).get("action", "unknown")
            reasoning = d.get("decision", {}).get("reasoning", "")
            it = d.get("iteration", "?")
            msg += f"- Iter {it}: {action} — {reasoning}\n"
            if d.get("applied_changes"):
                for name, (old, new) in d["applied_changes"].items():
                    msg += f"  - {name}: {old} → {new}\n"

    # Human observations (from human_notes.txt in run dir or home dir)
    human_notes = _read_human_notes()
    if human_notes:
        msg += f"\n## Human Observation (from visual evaluation)\n{human_notes}\n"
        msg += "NOTE: These are qualitative observations from a human watching the policy in simulation. Weight them heavily — numerical metrics cannot capture gait quality.\n"

    return msg


def _read_human_notes() -> str | None:
    """Read human observations from human_notes.txt if it exists.

    Checks two locations:
    1. ~/human_notes.txt (easiest to write via SSH)
    2. Run directory's human_notes.txt

    After reading, the file is renamed to human_notes_consumed.txt
    so the same note isn't sent to every future consultation.
    """
    search_paths = [
        os.path.expanduser("~/human_notes.txt"),
    ]
    for path in search_paths:
        if os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    content = f.read().strip()
                if content:
                    # Rename so it's only consumed once
                    consumed = path.replace(".txt", "_consumed.txt")
                    os.rename(path, consumed)
                    return content
            except Exception:
                pass
    return None
