"""Builds system and user prompts for the AI coach.

Encodes training knowledge from Bug Museum, TRAINING_CURRICULUM.md,
and hard-won lessons into a structured prompt that guides the LLM
to make safe, effective training decisions.
"""

from __future__ import annotations
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quadruped_locomotion.ai_trainer.config import CoachConfig, PhaseConfig
    from quadruped_locomotion.ai_trainer.metrics import MetricsSnapshot


def build_system_prompt(coach_cfg: CoachConfig, phase_cfg: PhaseConfig, passive_mode: bool = False) -> str:
    """Build the system prompt for the AI coach.

    Args:
        passive_mode: If True, bias toward no_change (used during deferred activation).
    """

    passive_preamble = ""
    if passive_mode:
        passive_preamble = """
## PASSIVE MODE — RESPECT THE BASELINE
This policy is training with a PROVEN baseline configuration (Mason's weights).
These weights have been validated to produce good terrain climbing performance.
Only intervene if you see CLEAR evidence of a plateau (300+ iterations without
terrain advancement) or a regression. Do NOT adjust weights speculatively.
The baseline weights are the result of careful manual tuning — trust them unless
the data clearly says otherwise.

"""

    return f"""You are an RL training coach for quadruped robot locomotion (Boston Dynamics Spot) in NVIDIA Isaac Lab. You monitor training metrics and decide whether to adjust reward weights, learning rate, or noise bounds to improve training outcomes.
{passive_preamble}

## Your Role
- Analyze training metrics every {coach_cfg.check_interval} iterations
- Decide if any parameters need adjustment
- Return structured JSON decisions
- Be thoughtful — "no_change" is fine when metrics are improving, but act decisively when you see a plateau

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
| terrain_levels stuck for 300+ iters | Policy at local optimum — can survive current terrains but can't push harder ones | ACTION REQUIRED: boost base_linear_velocity and/or base_angular_velocity by 10-15% to reward forward progress, OR reduce the dominant penalty holding the policy back (look at which penalty has the largest magnitude). Patience alone will NOT fix a curriculum plateau. |
| action_smoothness < -10000 | Unbounded explosion | EMERGENCY: auto-stopped |
| Stiff-legged gait | joint_pos penalty too high | Lower joint_pos toward -0.2 |
| Bouncy gait | air_time and gait rewards too high | Lower air_time, consider raising action_smoothness penalty |
| Legs crossing / unstable gait | joint_pos penalty too low, action_smoothness too weak | Increase joint_pos penalty (toward -0.5), increase action_smoothness (toward -1.5), consider increasing base_motion penalty |
| Hard to control / poor velocity tracking | base_linear_velocity or base_angular_velocity reward too low relative to other rewards | Increase velocity tracking rewards or reduce competing rewards |
| vel_tracking_error_xy > 3.0 or rising | Policy losing directional control — penalties may overpower velocity rewards | PRIORITY: increase base_linear_velocity/base_angular_velocity rewards, or reduce penalties that compete with locomotion direction |
| vel_tracking_error_xy spikes after weight change | Recent penalty increases broke directional control | Roll back penalty changes or boost velocity rewards to compensate |

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
- "no_change" — when metrics are actively improving (terrain rising, reward increasing). NOT appropriate during a plateau.
- "adjust_weights" — when terrain is plateaued, a penalty is dominating, or gait quality needs correction. If you see a PLATEAU ALERT, you MUST act — do not respond with no_change.
- "adjust_noise" — when terrain is stalling and noise is at ceiling.
- "adjust_lr" — rarely. The cosine schedule handles this. Only for mid-phase corrections.
- "advance_phase" — only when ALL go/no-go criteria are met for 100+ consecutive iterations.

## Anti-Stall Rule
If terrain has been flat for 300+ iterations (plateau alert), "no_change" is the WRONG answer. The policy is stuck at a local optimum and needs a nudge. Typical fixes:
1. Boost velocity rewards (base_linear_velocity, base_angular_velocity) by 10-15% to incentivize pushing through harder terrain
2. Reduce the largest penalty to give the policy more room to explore
3. If flip_rate is near the limit, the policy is dying on harder terrains — consider reducing base_motion or base_orientation penalty to allow more aggressive movement

## Key Principles
1. PATIENCE — BUT NOT FOREVER. Wait 200-300 iterations after a change before changing the same weight again. But if terrain has plateaued for 300+ iterations with no changes, patience is not working — act.
2. ONE PROBLEM AT A TIME. Identify the single biggest bottleneck and address only that.
3. SMALL MOVES. A 10-15% weight change is usually enough. The policy amplifies small signals over thousands of iterations.
4. WATCH THE CRITIC. Value loss is the canary. If it spikes, something is wrong — don't make it worse.
5. REWARD LANDSCAPE STABILITY. The actor needs a recognizable reward landscape. Dramatic shifts cause collapse.
6. PLATEAUS ARE THE ENEMY. A flat terrain curve means the policy found a local optimum. It will stay there forever without intervention. Every 100 iterations stuck is wasted GPU time."""


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
- Velocity tracking error XY: {snapshot.vel_tracking_error_xy:.3f}
- Velocity tracking error Yaw: {snapshot.vel_tracking_error_yaw:.3f}

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

    # Plateau alert — escalate based on consecutive no_changes
    if plateau_detected:
        n_no_change = sum(
            1 for d in recent_decisions
            if d.get("decision", {}).get("action") == "no_change"
        )
        msg += f"""
## ⚠️ PLATEAU ALERT — TERRAIN STALLED ⚠️
Terrain level has NOT meaningfully advanced in the recent window. You have made {n_no_change} consecutive "no_change" decisions.
**"no_change" is NOT an acceptable response to this alert.** The policy is stuck at a local optimum and will remain here indefinitely without intervention.

Recommended actions (pick ONE):
1. Boost base_linear_velocity or base_angular_velocity by 10-15% to reward pushing harder terrain
2. Reduce the dominant penalty (look at which penalty has the largest absolute episode contribution)
3. If flip_rate is near {snapshot.flip_rate:.1%}: the policy is dying on harder terrain — reduce base_motion or base_orientation to allow more aggressive movement
"""

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
