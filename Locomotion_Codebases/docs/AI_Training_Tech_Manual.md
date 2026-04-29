# AI Training Technical Manual

### Architecture, Implementation, and Design Decisions for the AI-Guided Training System

*AI2C Tech Capstone -- MS for Autonomy, Carnegie Mellon University, March 2026*

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [The Coach (LLM Decision Engine)](#3-the-coach-llm-decision-engine)
4. [The Guardrails (Safety Layer)](#4-the-guardrails-safety-layer)
5. [The Actuator (Runtime Modification)](#5-the-actuator-runtime-modification)
6. [The Metrics Collector](#6-the-metrics-collector)
7. [The Prompt Engineering](#7-the-prompt-engineering)
8. [Phase Configuration System](#8-phase-configuration-system)
9. [TensorBoard Integration](#9-tensorboard-integration)
10. [The Decision Log](#10-the-decision-log)
11. [Emergency System](#11-emergency-system)
12. [Isaac Lab Runtime Modification (How It Actually Works)](#12-isaac-lab-runtime-modification)
13. [Flat Terrain Compatibility](#13-flat-terrain-compatibility)
14. [Known Limitations and Future Work](#14-known-limitations-and-future-work)
15. [Bug Museum Addendum (AI Trainer Bugs)](#15-bug-museum-addendum)

---

## 1. System Overview

The AI-guided training system wraps RSL-RL's `OnPolicyRunner.learn()` with a monkey-patched `update()` function that intercepts every PPO iteration. At configurable intervals, it:

1. Collects a metrics snapshot from the live environment
2. Runs emergency checks (NaN, value explosion, smoothness bomb)
3. Calls Claude via the Anthropic API with training context
4. Validates the returned decision through a guardrail system
5. Applies approved changes to the live Isaac Lab environment

The key technical insight: **Isaac Lab's reward weights are mutable at runtime.** The `RewardManager` stores term configurations as Python objects with `.weight` attributes. Changing `rm._term_cfgs["gait"].weight = 8.0` takes effect on the very next step — no environment recreation, no checkpoint save/load, no training restart.

This means an LLM can adjust the reward landscape of a running simulation with zero downtime.

---

## 2. Architecture

### Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                        train_ai.py                                │
│                                                                   │
│  ┌─────────────┐     ┌──────────────┐     ┌──────────────────┐  │
│  │ RSL-RL      │────►│ Monkey-patch │────►│ Original PPO     │  │
│  │ runner.learn│     │ update()     │     │ update()         │  │
│  └─────────────┘     └──────┬───────┘     └──────────────────┘  │
│                             │                                     │
│                  every N iterations                               │
│                             │                                     │
│                             ▼                                     │
│                  ┌──────────────────┐                             │
│                  │ MetricsCollector │                             │
│                  │ (reads env/runner)│                             │
│                  └────────┬─────────┘                             │
│                           │                                       │
│                           ▼                                       │
│                  ┌──────────────────┐                             │
│                  │ Emergency Check  │──── NaN? value>100? ──►HALT │
│                  └────────┬─────────┘                             │
│                           │ (safe)                                │
│                           ▼                                       │
│                  ┌──────────────────┐                             │
│                  │ Coach            │                             │
│                  │ (Claude API)     │                             │
│                  │ ~3s latency      │                             │
│                  └────────┬─────────┘                             │
│                           │ CoachDecision                         │
│                           ▼                                       │
│                  ┌──────────────────┐                             │
│                  │ Guardrails       │                             │
│                  │ (validate+bound) │                             │
│                  └────────┬─────────┘                             │
│                           │ approved changes                      │
│                           ▼                                       │
│              ┌────────────────────────┐                           │
│              │ Actuator               │                           │
│              │ (modify live env/runner)│                           │
│              └────────────┬───────────┘                           │
│                           │                                       │
│                           ▼                                       │
│              ┌────────────────────────┐                           │
│              │ DecisionLog + TBoard   │                           │
│              │ (JSONL + TensorBoard)  │                           │
│              └────────────────────────┘                           │
└──────────────────────────────────────────────────────────────────┘
```

### Module Dependency Graph

```
train_ai.py
    ├── ai_trainer/config.py         (PhaseConfig, CoachConfig, PHASE_CONFIGS)
    ├── ai_trainer/metrics.py        (MetricsCollector, MetricsSnapshot)
    ├── ai_trainer/coach.py          (Coach, CoachDecision)
    ├── ai_trainer/guardrails.py     (Guardrails, FROZEN_WEIGHTS)
    ├── ai_trainer/actuator.py       (Actuator)
    ├── ai_trainer/decision_log.py   (DecisionLog)
    ├── ai_trainer/prompt_builder.py (build_system_prompt, build_user_message)
    ├── shared/lr_schedule.py        (cosine_annealing_lr, set_learning_rate)
    └── shared/training_utils.py     (configure_tf32, register_std_safety_clamp)
```

### File Sizes (for context)

| File | Lines | Responsibility |
|------|-------|---------------|
| `train_ai.py` | ~600 | Orchestration, training loop, monkey-patch |
| `config.py` | ~150 | Phase definitions, coach settings, weight bounds |
| `metrics.py` | ~236 | Metric collection, trend analysis, go/no-go |
| `coach.py` | ~133 | Claude API call, JSON parsing, failure handling |
| `guardrails.py` | ~208 | Weight validation, bounds, sign constraints |
| `actuator.py` | ~106 | Runtime env/runner modification |
| `decision_log.py` | ~84 | JSONL append-only audit trail |
| `prompt_builder.py` | ~147 | System + user prompt construction |
| **Total** | **~1,664** | |

---

## 3. The Coach (LLM Decision Engine)

### `ai_trainer/coach.py`

The coach is a thin wrapper around the Anthropic Messages API with optional VLM (Vision Language Model) support.

```python
class Coach:
    def __init__(self, coach_cfg, phase_cfg, api_key, vision_enabled=False):
        self.client = anthropic.Anthropic(api_key=api_key)
        self._vision_enabled = vision_enabled
        self.system_prompt = build_system_prompt(
            coach_cfg, phase_cfg, vision_enabled=vision_enabled)
        self._consecutive_failures = 0
        self._max_failures = 3

    def get_decision(self, snapshot, recent_history, recent_decisions,
                     plateau_detected, frame_png=None):
        # Build user message with current metrics
        # If frame_png provided: multimodal content (image + text)
        # Else: plain text (backwards compatible)
        # Call Claude API
        # Parse JSON response
        # Return (CoachDecision, latency_ms)
```

### VLM Mode (ARL Hybrid v2)

When `--enable_vision` is passed to `train_ai.py`, the coach receives rendered simulation frames alongside metrics. This was added after Trial MH-1 where the coach destroyed gait quality by optimizing for terrain numbers it couldn't visually verify.

**How it works:**
1. `train_ai.py` creates the env with `render_mode="rgb_array"` (requires `--enable_cameras`)
2. Before each coach consultation, `env.render()` captures an RGB numpy array
3. PIL converts the frame to PNG bytes
4. The coach receives a multimodal message: `[{"type": "image", ...}, {"type": "text", ...}]`
5. The system prompt includes visual analysis instructions (6-point gait checklist)
6. The coach's reasoning must include a "Visual assessment:" line

**Visual override rule:** If the image shows poor gait (flopping, bouncing, dragging), the coach must NOT advance terrain or loosen penalties regardless of what the numbers say.

### Key Design Decisions

**Why Claude Sonnet, not Opus?** Latency. Sonnet responds in 2-4 seconds. At 100-iteration intervals (~15s each), a 3s API call adds ~20% overhead. Opus would be 10-15s, adding ~100% overhead. The decisions aren't complex enough to need Opus — they're pattern matching against a troubleshooting table. Sonnet also supports vision (multimodal input).

**Why not function calling / tool_use?** The decision format is simple enough that a strict JSON response works reliably. The coach returns one JSON object with 6 fields. We strip markdown fences if present and `json.loads()` the result. In testing, Claude Sonnet returns valid JSON >99% of the time with this prompt.

**Why stateless (no conversation history)?** Each API call is independent — system prompt + single user message. This avoids context window growth over 300+ consultations and eliminates the risk of the coach fixating on stale context from 1000 iterations ago. Instead, we pass the last 5 decisions as structured data in the user message, giving the coach enough recent context without conversation drift.

**Failure handling:** After 3 consecutive API errors (timeout, rate limit, malformed response), the coach marks itself unavailable (`is_available = False`) and training continues without coaching. This prevents API issues from crashing a multi-day GPU run.

### CoachDecision Dataclass

```python
@dataclass
class CoachDecision:
    action: str = "no_change"       # no_change, adjust_weights, adjust_noise,
                                     # adjust_lr, advance_phase, emergency_stop
    reasoning: str = ""
    weight_changes: dict = {}        # {term_name: new_weight}
    lr_change: float | None = None
    noise_change: float | None = None
    confidence: float = 0.5
```

---

## 4. The Guardrails (Safety Layer)

### `ai_trainer/guardrails.py`

This is the most critical module. Every decision from the LLM must pass through guardrails before touching the live training environment. The guardrails encode all Bug Museum lessons as hard constraints that the LLM cannot override.

### Weight Validation Pipeline

For each proposed weight change:

```
1. Is the weight frozen? (stumble, body_height_tracking)
   → YES: REJECT. These use world-frame Z (Bug #22, #28b).

2. Is the weight frozen for this phase?
   → YES: REJECT.

3. Does the new value violate sign constraints?
   → Positive rewards must stay positive.
   → Penalties must stay negative.
   → REJECT if violated.

4. TERRAIN-GATED PENALTY LOOSENING (v2):
   → Is this a penalty being loosened (made less negative)?
   → Is terrain < penalty_loosen_terrain (default 4.0)?
   → YES: REJECT. Penalties cannot be loosened until robot demonstrates
     clean gait at terrain >= 4.0. This prevents the coach from
     repeatedly loosening penalties to boost terrain numbers at the
     cost of gait quality (the Trial MH-1 failure mode).

5. Is the new value within absolute bounds?
   → Each term has (min, max) bounds in CoachConfig.
   → Uses tighter mason_hybrid_bounds when in mason_hybrid phase.
   → CLAMP to nearest bound.

6. Is the delta within 20% of current value?
   → CLAMP to max 20% change.
   → For weights currently at 0: max absolute delta of 0.5.

7. Are we changing too many weights at once?
   → Max 3 changes per consultation.
   → Trial 11k changed 6 → 88% flip, terrain 0.12, total collapse.
   → Keep only the first 3.
```

### Frozen Weights (Hardcoded)

```python
FROZEN_WEIGHTS = {
    "stumble": 0.0,              # Bug #28b: world-frame Z misclassifies
    "body_height_tracking": 0.0, # Bug #22: meaningless on rough terrain
}
```

These are separate from phase-specific frozen weights. The LLM's system prompt tells it these are frozen, but even if the LLM ignores the instruction, the guardrails catch it.

### Sign Constraints

```python
SIGN_POSITIVE = {"air_time", "base_angular_velocity", "base_linear_velocity",
                  "foot_clearance", "gait", "velocity_modulation"}

SIGN_NEGATIVE = {"action_smoothness", "air_time_variance", "base_motion",
                  "base_orientation", "body_scraping", "contact_force_smoothness",
                  "dof_pos_limits", "foot_slip", "joint_acc", "joint_pos",
                  "joint_torques", "joint_vel", "terrain_relative_height",
                  "undesired_contacts", "vegetation_drag"}
```

A positive reward (like `gait`) set to a negative value would punish the robot for having a good gait. A penalty (like `joint_vel`) set positive would reward jerky movements. Both are catastrophic. The guardrails make this structurally impossible.

### Weight Bounds (Per-Term)

Every modifiable weight has absolute min/max bounds:

```python
weight_bounds = {
    "air_time":                (1.0, 10.0),
    "gait":                    (0.5, 15.0),
    "base_linear_velocity":    (1.0, 15.0),
    "action_smoothness":       (-5.0, -0.05),
    "base_motion":             (-5.0, -0.1),
    "base_orientation":        (-10.0, -0.5),
    "terrain_relative_height": (-5.0, -0.5),
    # ... (full list in config.py)
}
```

These bounds represent the "safe operating envelope" — values outside this range either have no training benefit or have been proven to cause instability.

### The 20% Delta Rule

Even within bounds, changes are rate-limited to 20% per consultation. This prevents the coach from making dramatic shifts that destabilize the reward landscape.

**Example:** If `gait` is currently at 10.0, the coach can change it to anywhere in [8.0, 12.0] in one step. To reach 15.0, it would need at least 3 consultations:
- 10.0 → 12.0 (iter 100)
- 12.0 → 14.4 (iter 200)
- 14.4 → 15.0 (iter 300, clamped to max)

This gradual approach gives the policy time to adapt to each change.

### The Max-3-Changes Rule (Trial 11k Lesson)

Trial 11k changed 6 reward weights simultaneously: action_smoothness, base_motion, air_time_variance, contact_force_smoothness, gait, and air_time. Result: 88% flip rate, terrain level 0.12 (down from 5.0), total policy collapse. The critic couldn't calibrate to the new reward landscape, the actor received wildly different gradients, and the policy degenerated within 100 iterations.

**The fix:** Maximum 3 changes at a time. If the coach proposes 5, only the first 3 are applied. This limits the reward landscape shift per consultation and gives the critic a fighting chance of tracking the changes.

---

## 5. The Actuator (Runtime Modification)

### `ai_trainer/actuator.py`

The actuator is the bridge between validated decisions and the live simulation.

### Reward Weight Modification

```python
def apply_weight_changes(self, changes: dict) -> dict:
    rm = self.env.unwrapped.reward_manager
    for name, new_weight in changes.items():
        if hasattr(rm, "_term_cfgs") and name in rm._term_cfgs:
            old_weight = rm._term_cfgs[name].weight
            rm._term_cfgs[name].weight = new_weight
```

**How this works internally:** Isaac Lab's `RewardManager` evaluates rewards every step by iterating `_term_cfgs`, calling each term's function, and multiplying by `cfg.weight`. The weight is a plain Python float on a dataclass. Changing it immediately affects the next reward computation — there's no caching, no compilation step, no need to recreate the environment.

**Caveat:** This only changes the weight, not the reward function itself. The clamped wrappers (Bug #29) and other structural aspects of the reward terms are fixed at env creation time. The coach can only tune the knobs, not rewire the circuit.

### Learning Rate Modification

```python
def apply_lr_change(self, new_lr: float) -> float:
    old_lr = self.runner.alg.optimizer.param_groups[0]["lr"]
    for param_group in self.runner.alg.optimizer.param_groups:
        param_group["lr"] = new_lr
    return old_lr
```

Standard PyTorch optimizer modification. This overrides the cosine annealing schedule — the LR cooldown mechanism in the training loop holds the override for 50 iterations before cosine resumes.

### Noise Bound Modification

```python
def apply_noise_change(self, new_max_noise: float):
    if self._noise_bounds is not None:
        self._noise_bounds["max"] = new_max_noise
```

The noise bounds are stored in a mutable dict that's shared between the actuator and the `register_std_safety_clamp` hook. Changing the dict's `"max"` value immediately affects the noise clamp applied after every PPO update.

### Checkpoint Save + Verify

```python
def save_checkpoint(self, path: str) -> bool:
    self.runner.save(path)
    loaded = torch.load(path, weights_only=False, map_location="cpu")
    state = loaded.get("model_state_dict", loaded)
    for key, tensor in state.items():
        if isinstance(tensor, torch.Tensor):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                return False  # Corrupted!
    return True
```

Every checkpoint is verified after saving (Bug #24: NaN can silently corrupt checkpoints).

---

## 6. The Metrics Collector

### `ai_trainer/metrics.py`

Reads from three sources:
1. **`env.unwrapped.extras["log"]`** — RSL-RL's per-iteration logging (reward, episode length, termination rates, per-reward breakdown)
2. **`env.unwrapped.reward_manager`** — Current reward weights (for the coach to see what it's working with)
3. **`runner.alg.policy.named_parameters()`** — NaN check on policy parameters

### MetricsSnapshot

A point-in-time capture of everything relevant:

```python
@dataclass
class MetricsSnapshot:
    iteration: int
    phase: str
    elapsed_hours: float
    mean_reward: float
    mean_episode_length: float
    survival_rate: float          # fraction surviving to timeout
    flip_rate: float              # fraction terminated by flip
    mean_terrain_level: float
    value_loss: float
    policy_loss: float
    noise_std: float
    learning_rate: float
    reward_breakdown: dict        # term_name -> mean per-episode contribution
    current_weights: dict         # term_name -> current weight
    reward_trend: float           # linear regression slope
    terrain_trend: float
    value_loss_trend: float
    has_nan: bool
    value_loss_spike: bool
```

### Trend Analysis

The collector maintains a rolling deque of snapshots (default: 200). Trends are computed via `numpy.polyfit(x, values, 1)[0]` — linear regression slope over the recent window.

- **reward_trend > 0** — training is improving
- **terrain_trend = 0** — potential plateau
- **value_loss_trend > 0** — critic is destabilizing

### Plateau Detection

```python
def is_plateau(self, metric="mean_terrain_level", window=300, threshold=0.01):
    if len(self.history) < window:
        return False
    values = [getattr(s, metric) for s in list(self.history)[-window:]]
    slope = np.polyfit(np.arange(len(values)), values, 1)[0]
    return abs(slope) < threshold
```

If terrain level hasn't moved in 300 iterations (at 100-iter intervals, that's 30 consultations), the coach receives a `PLATEAU DETECTED` alert in its user message.

### Go/No-Go Checks

Phase advancement criteria are checked against the most recent N snapshots:

```python
def go_no_go(self, phase_cfg):
    recent = list(self.history)[-phase_cfg.min_consecutive_iters:]
    # All criteria must hold simultaneously for min_consecutive_iters
    avg_survival = np.mean([s.survival_rate for s in recent])
    avg_flip = np.mean([s.flip_rate for s in recent])
    avg_noise = np.mean([s.noise_std for s in recent])
    avg_vloss = np.mean([s.value_loss for s in recent])
    avg_terrain = np.mean([s.mean_terrain_level for s in recent])
    # Compare against phase thresholds...
```

---

## 7. The Prompt Engineering

### `ai_trainer/prompt_builder.py`

### System Prompt Structure

The system prompt is rebuilt whenever the phase changes or vision mode changes. It contains:

1. **Core philosophy** — "Gait quality is PRIMARY. Terrain advancement is secondary." (ARL Hybrid v2)
2. **Visual analysis instructions** (VLM mode only) — 6-point gait checklist, visual override rule
3. **Hard constraints** — Priority-ordered: gait quality protection > stability > safety
4. **Terrain-gated penalty loosening** — Penalties locked until terrain >= 4.0
5. **Phase context** — Current terrain, target metrics, go/no-go criteria
6. **Troubleshooting table** — Symptom/cause/fix from Bug Museum (includes "flopping gait" and "robot not standing up")
7. **Decision format** — Strict JSON schema
8. **Key principles** — Gait quality is king, penalties are friends, velocity ceiling 3.0-7.0

### User Message Structure

Each consultation sends:

```
## Current Metrics (Iteration 100)
- Phase: flat
- Mean reward: 45.2
- Flip rate: 35.1%
- Terrain level: 0.00
- Velocity tracking error XY: 2.350
- Velocity tracking error Yaw: 0.980
...

## Reward Breakdown (per episode)
- action_smoothness: -0.4200
- air_time: 0.0840
- gait: 0.1418
...

## Current Reward Weights
- action_smoothness: -1.0
- air_time: 3.0
- gait: 10.0
...

## Trends (slope over recent window)
- Reward trend: +0.1234/iter
- Terrain trend: +0.0000/iter
- Value loss trend: -0.0012/iter

## Recent History (last 5 checkpoints)
| Iter | Reward | Terrain | Flip | Value Loss |
|------|--------|---------|------|------------|
| 20   | 2.7    | 0.00    | 60%  | 0.45       |
| 40   | 10.1   | 0.00    | 83%  | 0.37       |
...

## Recent Coach Decisions
- Iter 20: no_change — Training just started...
- Iter 40: no_change — Still very early training...
```

### Human Visual Observation Injection

The user message can also include qualitative human feedback via two mechanisms:

1. **`~/human_notes.txt`** — `_read_human_notes()` in `prompt_builder.py` checks for this file. If found, content is appended under "## Human Observation (from visual evaluation)" with instructions to weight it heavily. File is renamed to `_consumed.txt` after reading.

2. **JSONL decision log injection** — Append a `human_observation` entry directly to `ai_coach_decisions.jsonl`. The "Recent Coach Decisions" section reads the last 3 entries, so the observation appears in the next prompt automatically. Zero downtime, no code reload needed.

Both methods enable a human-AI collaboration loop: the human watches the policy in simulation (lava arena, play.py) and provides gait quality feedback that numerical metrics can't capture. The coach integrates this with its quantitative analysis.

**Proven in Trial 11l:** At iter 2500, a human observation about legs crossing and instability led the coach to make 3 targeted weight changes that directly addressed the visual quality issues while maintaining terrain progression.

### Why This Prompt Structure Works

1. **Numbers, not narratives.** The coach sees raw data, not interpretations. This prevents us from biasing the model's analysis.

2. **History provides context.** The last 5 snapshots and 3 decisions prevent the coach from making redundant changes or contradicting itself.

3. **Troubleshooting table as knowledge base.** Instead of relying on the LLM's training data about RL, we inject our specific failure modes and fixes. The coach doesn't need to know RL theory — it needs to know that `action_smoothness < -10000` means "unbounded explosion, stop immediately."

4. **Strict output format.** JSON-only response eliminates parsing ambiguity. The system prompt says "Respond with ONLY a JSON object (no markdown, no explanation outside JSON)" and provides the exact schema.

5. **Human-in-the-loop.** Qualitative observations from visual evaluation are injected alongside quantitative metrics. The coach is told to "weight them heavily — numerical metrics cannot capture gait quality." This lets the system combine 24/7 automated monitoring with occasional human visual inspection.

---

## 8. Phase Configuration System

### `ai_trainer/config.py`

### PhaseConfig

Each phase encodes proven training parameters from TRAINING_CURRICULUM.md:

```python
PHASE_CONFIGS = {
    "flat": PhaseConfig(
        name="flat",
        terrain="flat",
        num_envs=10000,
        max_iterations=500,
        lr_max=3e-4,
        max_noise_std=1.0,   # Flat can handle more exploration
        min_survival_rate=0.95,
        max_flip_rate=0.03,
    ),
    "robust": PhaseConfig(
        name="robust",
        terrain="robust",
        num_envs=5000,
        max_iterations=30000,
        lr_max=3e-5,          # Bug #23: higher LR crashes
        max_noise_std=0.5,    # Bug #26: higher causes curriculum stall
        min_terrain_level=5.0,
        min_survival_rate=0.85,
        max_flip_rate=0.12,
    ),
}
```

### CoachConfig

Global coach settings that apply across all phases:

```python
@dataclass
class CoachConfig:
    check_interval: int = 100          # consult every N iters
    api_model: str = "claude-sonnet-4-20250514"
    max_weight_changes: int = 3        # Trial 11k lesson
    max_weight_delta_pct: float = 0.20 # max 20% change
    max_weight_delta_abs: float = 0.5  # for near-zero weights
    emergency_value_loss: float = 100.0
    emergency_smoothness: float = -10000.0
    nan_rollback: bool = True
    decision_log_path: str = "ai_coach_decisions.jsonl"
    max_stall_iters: int = 500
    history_window: int = 200
    decision_history: int = 5

    # Weight bounds — absolute limits per term
    weight_bounds: dict = {
        "air_time": (1.0, 10.0),
        "gait": (0.5, 15.0),
        "action_smoothness": (-5.0, -0.05),
        # ... (22 entries total)
    }

    # Phase-specific LR ceilings
    phase_lr_limits: dict = {
        "flat": 3e-4,
        "transition": 3e-4,
        "robust_easy": 5e-5,
        "robust": 3e-5,
    }
```

---

## 9. TensorBoard Integration

The training loop uses RSL-RL's existing `SummaryWriter` (via `runner.writer`) to log AI coach-specific metrics. The writer is lazily acquired since RSL-RL creates it after `runner.learn()` starts.

### Custom Scalar Groups

| Group | Scalars | Update Frequency |
|-------|---------|-----------------|
| `AI_Coach/` | action_code, confidence, api_latency_ms, num_changes, guardrail_blocks, reward_trend, terrain_trend, value_loss_trend, emergency | Every coach consultation |
| `Reward_Weights/` | One scalar per reward term (22 total) | Every coach consultation |
| `Reward_Contrib/` | Per-episode contribution of each term | Every coach consultation |
| `Weight_Changes/` | Delta applied per changed weight | Only when changes are made |

### Action Code Mapping

```python
action_map = {
    "no_change": 0,
    "adjust_weights": 1,
    "adjust_noise": 2,
    "adjust_lr": 3,
    "advance_phase": 4,
    "emergency_stop": 5,
}
```

In TensorBoard, `AI_Coach/action_code` shows a step plot — mostly 0 (no_change) with occasional spikes to 1 (weight adjustment). Frequent spikes to 1 suggest the coach is being too aggressive. Spikes to 5 (emergency) indicate serious instability.

### Initial Weights Logging

At iteration 0 (first update call), all 22 reward weights are logged to `Reward_Weights/` to establish a baseline. This enables visual comparison of "where we started" vs "where the coach took us."

---

## 10. The Decision Log

### `ai_trainer/decision_log.py`

Append-only JSONL file. Every coach interaction, emergency event, and phase transition gets a timestamped entry.

### Entry Types

**Coach decision:**
```json
{
  "timestamp": "2026-03-08T19:38:22.123456",
  "iteration": 100,
  "phase": "flat",
  "metrics": { /* full MetricsSnapshot */ },
  "decision": { /* full CoachDecision */ },
  "guardrail_messages": ["BOUNDED base_motion: -3.5 -> -3.6 (max 20%)"],
  "applied_changes": {"base_motion": [-4.0, -3.6]},
  "api_latency_ms": 2847.3
}
```

**Emergency:**
```json
{
  "timestamp": "2026-03-08T20:15:44.567890",
  "iteration": 500,
  "phase": "robust_easy",
  "emergency": true,
  "action": "halve_lr",
  "details": "Value loss 142.3 > 100, LR halved to 1.50e-05"
}
```

**Phase transition:**
```json
{
  "timestamp": "2026-03-08T21:00:00.000000",
  "iteration": 500,
  "phase_transition": true,
  "from_phase": "flat",
  "to_phase": "transition",
  "checkpoint": "logs/rsl_rl/.../model_500.pt"
}
```

### Post-Training Analysis

The JSONL format enables easy analysis:

```python
import json
decisions = [json.loads(line) for line in open("ai_coach_decisions.jsonl")]

# How many times did the coach actually change something?
changes = [d for d in decisions if d.get("decision", {}).get("action") == "adjust_weights"]
print(f"Weight adjustments: {len(changes)} / {len(decisions)}")

# What weights were changed most often?
from collections import Counter
weight_counts = Counter()
for d in changes:
    for name in d.get("applied_changes", {}):
        weight_counts[name] += 1
print(weight_counts.most_common(5))

# Did guardrails block anything?
blocked = [d for d in decisions if d.get("guardrail_messages")]
print(f"Guardrail interventions: {len(blocked)}")
```

---

## 11. Emergency System

Emergencies override the coach entirely. They're checked before the API call to minimize latency.

### Emergency Hierarchy

| Priority | Condition | Action | Cooldown |
|----------|-----------|--------|----------|
| 1 (critical) | `has_nan == True` | Log + halt training | Terminal |
| 2 (severe) | `value_loss > 100` | Halve LR, freeze actor for 50 iters | 50 iters |
| 3 (severe) | `action_smoothness < -10,000` | Emergency stop | Terminal |

### NaN Detection

The metrics collector checks every policy parameter:

```python
for name, param in self.runner.alg.policy.named_parameters():
    if torch.isnan(param.data).any() or torch.isinf(param.data).any():
        return True  # has_nan
```

This catches NaN before it propagates to checkpoints. Once NaN is in the policy, training is unrecoverable — the emergency system logs it and exits cleanly.

### Value Loss Halving

When `value_loss > 100`, the emergency system:
1. Halves the current LR
2. Sets a 50-iteration cooldown (LR frozen, cosine schedule paused)
3. Training continues with lower LR

This mirrors Bug #25's fix — the value loss watchdog. The emergency is logged to both the decision log and TensorBoard.

### Action Smoothness Bomb

When `action_smoothness < -10,000`, something has gone catastrophically wrong (seen in Trials 11h and 11i where it hit -1.3 trillion). The emergency system stops training immediately. This condition is unrecoverable within the current run.

---

## 12. Isaac Lab Runtime Modification (How It Actually Works)

### Reward Weight Hot-Swapping

Isaac Lab's `RewardManager` in `isaaclab/managers/reward_manager.py` works like this:

```python
class RewardManager:
    def compute(self, dt: float):
        for name, (term, cfg) in self._term_cfgs.items():
            raw = term(self._env, **cfg.params)    # Call reward function
            weighted = raw * cfg.weight             # Multiply by weight
            self._reward_buf += weighted * dt       # Accumulate
```

The `cfg.weight` is read every step. Changing it between steps changes the reward immediately. There's no compilation, JIT caching, or lazy evaluation that would prevent the change from taking effect.

**This is safe because:**
1. The reward function itself doesn't change — only its multiplier
2. The actor's gradients will naturally adapt to the new reward scale over subsequent PPO updates
3. The critic will need a few iterations to recalibrate, but 20% changes are small enough that the critic handles it without value explosion

### Optimizer LR Hot-Swapping

Standard PyTorch — `optimizer.param_groups[0]["lr"]` is read at the start of each optimizer step. Changing it between steps is the documented way to implement custom learning rate schedules.

### Noise Bound Hot-Swapping

The noise bounds are stored in a Python dict shared between the `register_std_safety_clamp` hook and the actuator:

```python
# In train_ai.py
noise_bounds = {"min": 0.3, "max": 0.5}

# register_std_safety_clamp reads noise_bounds["max"] every step
# actuator.apply_noise_change writes noise_bounds["max"]
```

This uses Python's mutable dict semantics — both the hook and the actuator reference the same object in memory. No locks needed because Isaac Lab's training loop is single-threaded.

---

## 13. Flat Terrain Compatibility

### The Problem

Isaac Lab's `TerrainImporter` in `"plane"` mode is a lightweight flat ground. It doesn't create:
- `terrain_types` (column → terrain type mapping)
- `terrain_levels` (per-env curriculum level)
- `terrain_origins` (spawn positions)

But several systems assume these exist:
1. `terrain_levels_vel` curriculum function
2. `VegetationDragReward` terrain-aware mode
3. Terrain curriculum logging

### The Fix

In `train_ai.py`'s `apply_phase_terrain()`:

```python
if terrain == "flat":
    env_cfg.scene.terrain.terrain_type = "plane"
    # Disable terrain curriculum
    if env_cfg.curriculum is not None:
        env_cfg.curriculum.terrain_levels = None
```

In `shared/reward_terms.py`'s `VegetationDragReward`:

```python
if self.terrain_aware:
    terrain = self._env_ref.scene.terrain
    if hasattr(terrain, "terrain_types"):  # Guard for flat terrain
        robot_cols = terrain.terrain_types[env_ids]
        # ... terrain-aware logic
```

---

## 14. Known Limitations and Future Work

### Current Limitations

1. **Stateless coach.** Each API call is independent. The coach can't build a mental model of training dynamics across 100+ consultations. This is by design (prevents context drift) but limits the coach's ability to detect slow trends that span many consultations.

2. **No A/B testing.** The coach can't run controlled experiments (e.g., "try gait=8 for 200 iters, then compare to gait=10"). It makes decisions based on current trends only.

3. **Phase transitions are manual.** The go/no-go check runs, but if it fails, training stops rather than the coach adjusting parameters to meet criteria. A human must decide whether to restart with different settings.

4. **Single-variable analysis only.** The coach can see all 22 reward contributions but can only change 3 at a time. It can't reason about interaction effects between rewards.

5. **No rollback.** If a weight change makes things worse, the coach can change it back at the next consultation, but it can't restore the policy to its pre-change state. The 200-500 iteration effect delay means bad changes are slow to detect and reverse.

### Future Work

1. **Conversation mode.** Maintain a sliding window of the last 10 consultations as conversation history, enabling the coach to track multi-consultation experiments.

2. **Automatic phase transitions.** Instead of stopping on go/no-go failure, let the coach adjust parameters (especially noise ceiling) to help the policy meet advancement criteria.

3. ~~**Multi-modal input.**~~ **DONE (ARL Hybrid v2).** The coach now receives rendered simulation frames via `--enable_vision`. Uses Claude Sonnet's multimodal API with base64-encoded PNG frames. The system prompt includes a 6-point visual gait checklist and a "visual override rule" that prevents terrain advancement when gait quality is visually poor.

4. **Ensemble decisions.** Call multiple Claude instances and take the majority vote. Reduces variance of individual LLM responses.

5. **Curriculum integration.** Let the coach suggest terrain configuration changes (e.g., reduce `num_rows` to cap difficulty, or increase specific terrain proportions).

---

## 15. Bug Museum Addendum (AI Trainer Bugs)

### Bug #30: terrain_types AttributeError on Flat Terrain

**When:** First local smoke test, train_ai.py with `--start_phase flat`

**What happened:** `VegetationDragReward.__init__` called `terrain.terrain_types[env_ids]` but `TerrainImporter` in plane mode doesn't have `terrain_types`.

**Root cause:** The reward term assumed curriculum terrain was always present. On flat terrain (plane mode), the `TerrainImporter` is minimal — no types, no levels, no curriculum.

**Fix:** Guard with `hasattr(terrain, "terrain_types")` before accessing.

### Bug #31: terrain_levels AttributeError on Flat Terrain

**When:** Same smoke test, after fixing Bug #30.

**What happened:** `terrain_levels_vel` curriculum function called `terrain.terrain_levels` which doesn't exist on plane terrain.

**Root cause:** The env config's curriculum was still active even though terrain was flat. The curriculum function assumes generated terrain with curriculum levels.

**Fix:** In `apply_phase_terrain()`, set `env_cfg.curriculum.terrain_levels = None` when terrain is flat. Follows Isaac Lab's convention (used by `anymal_b/flat_env_cfg.py`, `cassie/flat_env_cfg.py`).

### Bug #32: conda --no-banner Not Supported

**When:** Attempting `conda run --no-banner -n isaaclab311` on local Windows.

**What happened:** `conda-script.py: error: unrecognized arguments: --no-banner`

**Root cause:** Older conda version on this machine doesn't support the `--no-banner` flag.

**Fix:** Use `conda run -n isaaclab311` without `--no-banner`, or call the Python executable directly: `/c/miniconda3/envs/isaaclab311/python.exe`.
