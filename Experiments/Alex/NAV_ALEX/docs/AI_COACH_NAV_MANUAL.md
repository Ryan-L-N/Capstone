# AI Coach Navigation Training Manual

## 1. Overview

The AI Coach system uses Claude Sonnet as an online training advisor for quadruped navigation policy learning. Every **250 iterations**, the coach receives a snapshot of training metrics and returns a JSON decision: either `no_change` or a set of reward weight adjustments. The coach operates in text-only mode (no VLM) because the H100 server lacks Vulkan display support.

**Cost:** Approximately $2--4 per full training run. Each API call costs roughly $0.01 (Sonnet pricing), and a typical run makes 200--400 consultations across all curriculum phases.

**Architecture:**

```
MetricsCollector (every iter)
       |
       v  (every 250 iters)
  Coach (Claude Sonnet API)
       |
       v
  Guardrails (validate/clamp)
       |
       v
  Actuator (env.reward_manager._term_cfgs[name].weight = new_val)
       |
       v
  Decision Log (JSONL audit trail)
```

The coach never touches learning rate, network architecture, or curriculum thresholds directly. It only adjusts reward weights within bounded ranges, subject to guardrail enforcement.

---

## 2. Weight Bounds Table

All 8 navigation reward terms, their defaults, allowed ranges, and coach access rules:

| Reward Term | Default Weight | Min | Max | Coach Can Adjust | Notes |
|---|---|---|---|---|---|
| `forward_velocity` | +10.0 | 3.0 | 15.0 | Yes | Primary locomotion drive |
| `survival` | +1.0 | 0.5 | 3.0 | Yes | Kept modest to avoid standing-still exploits |
| `terrain_traversal` | +2.0 | 0.5 | 5.0 | Yes | Bonus for advancing terrain curriculum |
| `terrain_relative_height` | -2.0 | -5.0 | -1.0 | Yes (tighten only until terrain >= 3) | Anti-crawl enforcement; loosening locked below terrain 3 |
| `drag_penalty` | -1.5 | -4.0 | -0.5 | Yes | Penalizes dragging limbs / belly contact |
| `cmd_smoothness` | -1.0 | -3.0 | -0.1 | Yes | Prevents jerky command oscillations |
| `lateral_velocity` | -0.3 | -1.5 | -0.05 | Yes (keep light) | Lateral dodge is useful; do not over-penalize |
| `angular_velocity` | -0.5 | -2.0 | -0.1 | Yes | Prevents excessive spinning |

**Key constraint:** `terrain_relative_height` can only be made *more negative* (tightened) when `terrain_level < 3.0`. Once terrain reaches 3.0+, the coach may loosen it within bounds.

---

## 3. Guardrail Rules

The guardrails layer (`guardrails.py`) validates every coach recommendation before it reaches the environment. All rules are hard constraints that cannot be overridden by the coach.

### 3.1 Max 3 Weight Changes at a Time

Trial 11k changed 6 weights simultaneously and the policy collapsed to 88% flip rate, terrain 0.12. The actor needs a recognizable reward landscape between consultations.

```python
if len(weight_changes) > 3:
    # Keep only the 3 changes with highest confidence
    weight_changes = sorted(weight_changes, key=lambda c: c["confidence"], reverse=True)[:3]
    guardrail_messages.append("Trimmed to top 3 changes (max-3 rule)")
```

### 3.2 Max 20% Delta Per Change

No single adjustment may exceed 20% of the current weight magnitude. This prevents shock transitions.

```python
max_delta = abs(current_weight) * 0.20
clamped_delta = max(-max_delta, min(max_delta, proposed_delta))
```

### 3.3 Sign Constraints

Positive weights stay positive. Negative weights stay negative. The coach cannot flip a reward into a penalty or vice versa.

### 3.4 Terrain-Gated Loosening

Penalty weights (negative terms) cannot be loosened (made less negative) until `terrain_level >= 3.0`. This prevents the policy from gaming easy terrain by relaxing safety constraints prematurely.

### 3.5 Near-Zero Weight Protection

For weights with absolute value below 0.5, the maximum single-step change is capped at 0.5 absolute. This prevents tiny weights from being scaled up dramatically by the percentage rule.

---

## 4. Anti-Crawl Enforcement

Belly-crawling is the single most common exploit in quadruped locomotion training (Bug #27). The robot discovers that lying flat maximizes survival reward while minimizing all penalties. The coach system encodes anti-crawl as a **hard rule** in its system prompt.

### Triggers

1. **Height check:** If `mean_body_height < 0.30m` for 2 or more consecutive coach consultations (500+ iterations), the coach **must** tighten `terrain_relative_height` penalty. This is not a suggestion; it is encoded as a mandatory action in the system prompt.

2. **Drag check:** If `drag_penalty > 1.0` sustained across consecutive checks, the coach flags a crawling behavior pattern and increases both `drag_penalty` and `terrain_relative_height` weights.

### System Prompt Encoding

```
HARD RULE: If mean_body_height < 0.30m for 2+ consecutive checks, you MUST
include terrain_relative_height tightening in your response. This is not
optional. Belly-crawling is the #1 training failure mode. Override all other
reasoning if this condition is met.
```

The guardrails layer independently verifies this rule. If the coach returns `no_change` while the crawl condition is active, the guardrails inject a forced tightening of `terrain_relative_height` by 20% and log a guardrail override.

---

## 5. Emergency Checks

Emergency checks run **every iteration** (not just at coach intervals) and take priority over all coach decisions.

| Condition | Action | Rationale |
|---|---|---|
| NaN detected in any policy parameter | Halt training immediately | Corrupted weights are unrecoverable (Bug #24) |
| `value_loss > 100` | Halve learning rate immediately | Breaks oscillation cascade before NaN (Bug #25) |

Emergency actions are logged to the decision log with `"source": "emergency"` and bypass the coach entirely.

```python
# Emergency check (runs every iteration)
if torch.isnan(params).any():
    log_emergency("nan_detected", iteration)
    sys.exit(1)

if value_loss > 100.0:
    for pg in optimizer.param_groups:
        pg["lr"] *= 0.5
    log_emergency("value_loss_halve_lr", iteration, value_loss=value_loss)
```

---

## 6. System Prompt Structure

The coach system prompt is built by `prompt_builder.py` and contains:

### Priority Order

```
1. Speed + Height (forward velocity while maintaining standing posture)
2. Stability (survival rate, flip rate, smooth commands)
3. Terrain advancement (curriculum progression)
```

### Troubleshooting Table

The prompt includes a symptom-cause-fix lookup:

| Symptom | Likely Cause | Recommended Fix |
|---|---|---|
| Terrain stalled, height dropping | Crawl exploit emerging | Tighten height penalty |
| High flip rate on new terrain | Penalties too aggressive | Slightly loosen cmd_smoothness or lateral |
| Reward oscillating wildly | Too many weight changes | Return no_change, let policy stabilize |
| Forward velocity plateaued | forward_velocity weight too low | Increase by 10--15% |
| Spinning in place | angular_velocity penalty too light | Tighten angular_velocity |

### Required Response Format

```json
{
  "action": "adjust_weights",
  "reasoning": "Terrain stalled at 2.8 for 400 iters. Survival 94% and height 0.38m are healthy. Increasing terrain_traversal to incentivize progression.",
  "weight_changes": [
    {"term": "terrain_traversal", "new_weight": 2.4, "confidence": 0.8}
  ],
  "confidence": 0.75
}
```

Or for no changes:

```json
{
  "action": "no_change",
  "reasoning": "Terrain advancing steadily (2.1 -> 2.4 over last 500 iters). All metrics healthy. No intervention needed.",
  "confidence": 0.9
}
```

---

## 7. Metrics Collected

The `MetricsCollector` aggregates metrics every iteration and provides windowed summaries to the coach.

### Standard Metrics

| Metric | Source | Description |
|---|---|---|
| `mean_reward` | Episode buffer | Mean episodic return |
| `survival_rate` | Episode buffer | Fraction of envs alive at episode end |
| `flip_rate` | Episode buffer | Fraction of envs that flipped over |
| `terrain_level` | Curriculum | Current mean terrain difficulty level |
| `value_loss` | PPO update | Critic loss (monitors stability) |
| `noise_std` | Policy | Current exploration noise |
| `learning_rate` | Optimizer | Current LR from cosine schedule |

### Navigation-Specific Metrics

| Metric | Source | Description |
|---|---|---|
| `forward_distance` | Reward term | Mean forward displacement per episode |
| `body_height` | Reward term | Mean body height above local terrain |
| `drag_penalty` | Reward term | Mean drag penalty magnitude |

### Derived Trends

- **Reward slope:** Linear regression over last 500 iterations
- **Terrain slope:** Linear regression over last 500 iterations
- **Value loss slope:** Linear regression over last 200 iterations (shorter window for faster response)
- **Plateau flag:** Set to `true` when terrain level has not increased by more than 0.1 over the last 300 iterations

---

## 8. Decision Log Format

Every coach consultation produces one JSONL entry appended to `decision_log.jsonl` in the run directory.

```json
{
  "timestamp": "2026-03-17T14:23:07Z",
  "iteration": 2500,
  "metrics": {
    "mean_reward": 142.3,
    "survival_rate": 0.91,
    "flip_rate": 0.03,
    "terrain_level": 2.8,
    "value_loss": 4.2,
    "noise_std": 0.35,
    "learning_rate": 2.1e-5,
    "forward_distance": 8.7,
    "body_height": 0.37,
    "drag_penalty": 0.42,
    "reward_slope": 0.12,
    "terrain_slope": 0.001,
    "plateau": true
  },
  "coach_decision": {
    "action": "adjust_weights",
    "reasoning": "Terrain plateaued at 2.8 for 300+ iters...",
    "weight_changes": [
      {"term": "terrain_traversal", "new_weight": 2.4, "confidence": 0.8}
    ],
    "confidence": 0.75
  },
  "guardrail_messages": [],
  "applied_changes": [
    {"term": "terrain_traversal", "old_weight": 2.0, "new_weight": 2.4}
  ],
  "api_latency_ms": 3420
}
```

Emergency entries use `"source": "emergency"` and omit the `coach_decision` field.

---

## 9. Troubleshooting Guide

### "Coach keeps recommending no_change"

1. Check if plateau detection is working: look at `plateau` field in decision log. If terrain has stalled 300+ iterations and plateau is still `false`, the metrics window may be misconfigured.
2. Verify the coach is receiving current metrics, not stale ones. Check `metrics.iteration` in the log matches the actual training iteration.
3. If all metrics are genuinely healthy and improving, `no_change` is the correct decision. The coach should not intervene when training is progressing.

### "Coach API failing"

1. Verify `ANTHROPIC_API_KEY` environment variable is set and valid.
2. Check network connectivity from the H100: `curl -s https://api.anthropic.com/v1/messages -H "x-api-key: $ANTHROPIC_API_KEY"`.
3. The system has a **3-failure fallback**: after 3 consecutive API failures, the coach is disabled for the remainder of the run and training continues with current weights. This is logged as `"source": "api_fallback"` in the decision log.
4. API latency is typically 3--5 seconds per call. If latency exceeds 30 seconds consistently, check for rate limiting.

### "Weight changes having no effect"

1. Verify the actuator is connected to the live environment's reward manager. Check that `env.reward_manager._term_cfgs[name].weight` is being written to, not a copy.
2. Confirm the reward term names in the coach response match the exact keys in `reward_manager`. A typo (e.g., `height_penalty` vs `terrain_relative_height`) will silently fail.
3. Check the decision log's `applied_changes` field. If it is empty but `coach_decision.weight_changes` is not, the guardrails blocked the change. Read `guardrail_messages` for the reason.
4. Small weight changes (under 5% delta) may not produce visible metric shifts for 200+ iterations. Allow time for the policy to adapt before diagnosing.

### "NaN during training"

1. Check which tensor went NaN first using the emergency log.
2. Most common cause: `value_loss` spiral (Bug #25). If value loss was climbing before the NaN, the LR halving may have been too late. Consider starting with a lower `lr_max`.
3. Verify all reward terms use clamped wrappers (Bug #29). Unbounded L2 norms in penalty terms are the primary NaN source.
4. Check the last saved checkpoint with `torch.isnan().any()` on every tensor before attempting to resume.
