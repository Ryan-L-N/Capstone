# Train Spot with AI

### A Curriculum for Teaching Spot to Walk — Guided by an LLM Coach

*AI2C Tech Capstone -- MS for Autonomy, Carnegie Mellon University, March 2026*

---

## Table of Contents

1. [What Is This?](#1-what-is-this)
2. [The 30-Second Version](#2-the-30-second-version)
3. [Prerequisites](#3-prerequisites)
4. [Quickstart: One Command to Train](#4-quickstart-one-command-to-train)
5. [The Four Phases (Flat → Transition → Robust Easy → Robust)](#5-the-four-phases)
6. [What the AI Coach Does](#6-what-the-ai-coach-does)
7. [TensorBoard: What to Watch](#7-tensorboard-what-to-watch)
8. [Decision Log: What the AI Changed and Why](#8-decision-log-what-the-ai-changed-and-why)
9. [Manual Overrides and Escape Hatches](#9-manual-overrides-and-escape-hatches)
10. [Troubleshooting](#10-troubleshooting)
11. [Run History](#11-run-history)
12. [Quick Reference: Commands and Paths](#12-quick-reference-commands-and-paths)

---

## 1. What Is This?

`train_ai.py` wraps our standard PPO training loop with an **AI coach** — a Claude LLM that monitors training metrics every N iterations and decides whether to adjust reward weights, learning rate, or noise bounds. Instead of a human staring at TensorBoard for 24 hours and making mid-run corrections (like we did for Trials 11a through 11l), the AI coach does it automatically.

The coach sees exactly what a human sees: mean reward, flip rate, terrain level, per-reward contributions, value loss trends. It returns structured JSON decisions that pass through a guardrail system before being applied to the live environment. The guardrails encode every hard-won lesson from the Bug Museum — frozen weights, sign constraints, rate limits, and emergency stops.

**Key insight:** The AI coach doesn't replace the training curriculum. It still follows the four-phase progression (flat → transition → robust_easy → robust). What it replaces is the human sitting at a terminal, watching metrics, and typing `sed` commands on the H100 to tweak reward weights mid-run.

---

## 2. The 30-Second Version

```
You have a robot (Spot).
You have 12 types of terrain.
You have 22 reward signals.
You have an AI coach (Claude) watching every 100 iterations.

The coach:
  1. Reads the training metrics
  2. Checks for emergencies (NaN, value explosion, smoothness bomb)
  3. Decides: change weights? lower noise? adjust LR? do nothing?
  4. Sends that decision through guardrails (max 3 changes, <20% delta)
  5. Applies it to the live simulation — no restart needed

You start it. You go to sleep. You wake up with a trained robot.

That's it. Everything else is details about how to keep the AI
from repeating the mistakes we already made.
```

---

## 3. Prerequisites

### Hardware
- **H100 (production):** NVIDIA H100 NVL 96GB at `172.24.254.24`
- **Local (smoke test):** Any machine with Isaac Lab + CUDA GPU

### Software
- NVIDIA Isaac Lab (Isaac Sim 5.1.0)
- Conda environment: `isaaclab311` (local) or `env_isaaclab` (H100)
- RSL-RL (PPO implementation)
- Anthropic Python SDK (`pip install anthropic`)

### API Key
You need a valid Anthropic API key. Pass it via:
- CLI: `--anthropic_api_key sk-ant-...`
- Environment variable: `export ANTHROPIC_API_KEY=sk-ant-...`

The coach uses `claude-sonnet-4-20250514` by default (~3s per API call, ~$0.01 per consultation). At 100-iteration intervals over 30,000 iterations, that's ~300 API calls = ~$3 per training run.

---

## 4. Quickstart: One Command to Train

### Full Training (H100 — Flat through Robust)

```bash
cd ~/IsaacLab
screen -dmS spot_ai bash -c '
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab &&
    export OMNI_KIT_ACCEPT_EULA=YES && export PYTHONUNBUFFERED=1 &&
    ./isaaclab.sh -p ~/multi_robot_training/train_ai.py --headless \
        --robot spot --start_phase flat --end_phase robust \
        --num_envs 5000 --max_noise_std 0.5 --min_noise_std 0.3 \
        --coach_interval 100 --save_interval 100 \
        --anthropic_api_key YOUR_KEY \
        --no_wandb 2>&1 | tee ~/ai_train_full.log
'
```

### Resume from Phase B (H100)

```bash
cd ~/IsaacLab
screen -dmS spot_ai bash -c '
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab &&
    export OMNI_KIT_ACCEPT_EULA=YES && export PYTHONUNBUFFERED=1 &&
    ./isaaclab.sh -p ~/multi_robot_training/train_ai.py --headless \
        --robot spot --start_phase robust --end_phase robust \
        --num_envs 5000 --max_noise_std 0.5 --min_noise_std 0.3 \
        --coach_interval 100 --save_interval 100 \
        --load_run 2026-03-08_19-04-32 --load_checkpoint model_1900.pt \
        --anthropic_api_key YOUR_KEY \
        --no_wandb 2>&1 | tee ~/ai_train_resume.log
'
```

### Local Smoke Test (250 envs, flat only)

```bash
/c/miniconda3/envs/isaaclab311/python.exe \
    ~/multi_robot_training/train_ai.py --headless \
    --robot spot --start_phase flat --end_phase flat \
    --num_envs 250 --max_iterations 1000 \
    --coach_interval 20 --save_interval 50 \
    --anthropic_api_key YOUR_KEY \
    --no_wandb
```

### Without AI Coach (plain training)

```bash
./isaaclab.sh -p ~/multi_robot_training/train_ai.py --headless \
    --robot spot --start_phase flat --end_phase robust \
    --num_envs 5000 --no_coach --no_wandb
```

---

## 5. The Four Phases

The training curriculum is the same four phases from `TRAINING_CURRICULUM.md`. The AI coach follows the same progression — it just automates the mid-run adjustments.

```
Phase A (flat)     →  Phase A.5 (transition)  →  Phase B-easy       →  Phase B (robust)
500 iters             1000 iters                  10000 iters           30000 iters
Learn to stand        Gentle obstacles            All 12 types, easy    Full difficulty
lr=3e-4               lr=3e-4                     lr=5e-5               lr=3e-5
```

### Phase Advancement

Between phases, the system runs a **go/no-go check**:

| Phase | Survival | Flip | Noise | Value Loss | Terrain |
|-------|----------|------|-------|------------|---------|
| flat → transition | >95% | <3% | <0.6 | <10 | — |
| transition → robust_easy | >90% | <5% | <0.6 | <10 | — |
| robust_easy → robust | >85% | <10% | <0.6 | <10 | >2.0 |
| robust (done) | >85% | <12% | — | — | >5.0 |

If the criteria aren't met, training stops and saves a state file for manual review. The AI coach will NOT force a phase transition — that's a human decision.

### Phase-Specific LR Ceilings

These are hard limits the AI coach can never exceed:

| Phase | LR Ceiling | Why |
|-------|-----------|-----|
| flat | 3e-4 | Safe for flat terrain |
| transition | 3e-4 | Same — terrain is gentle |
| robust_easy | 5e-5 | 1e-4 crashes at iter ~1134 (Bug #23) |
| robust | 3e-5 | 5e-5 crashes at iter ~4037 |

---

## 6. What the AI Coach Does

### The Loop

Every `coach_interval` iterations (default: 100, use 20 for smoke tests):

```
1. MetricsCollector reads from live env:
   - Mean reward, episode length, flip rate, survival rate
   - Terrain curriculum level
   - Value loss, policy loss, noise std
   - Per-reward breakdown (all 22 terms)
   - Linear regression trends over recent window

2. Emergency Check (overrides coach):
   - NaN in policy params → immediate rollback
   - Value loss > 100 → auto-halve LR, freeze for 50 iters
   - action_smoothness < -10,000 → emergency stop

3. AI Coach (Claude API):
   - Receives: system prompt + current metrics + recent history + past decisions
   - Returns: JSON with action, reasoning, weight_changes, lr_change, noise_change
   - Typical latency: 2-4 seconds

4. Guardrails validate the decision:
   - Max 3 weight changes at a time (Trial 11k lesson)
   - Max 20% change per weight per consultation
   - Frozen weights: stumble=0.0, body_height_tracking=0.0
   - Sign constraints: rewards stay positive, penalties stay negative
   - Absolute bounds per term (from CoachConfig)
   - Phase-specific LR ceiling

5. Actuator applies approved changes:
   - Runtime reward weight modification via reward_manager._term_cfgs
   - LR modification via optimizer.param_groups
   - Noise bound adjustment via shared mutable dict
   - No restart needed — takes effect immediately
```

### What the Coach Knows

The system prompt encodes:
- All Bug Museum entries (#22 through #29)
- Phase-specific constraints and go/no-go criteria
- Troubleshooting table (symptom → cause → fix)
- Decision format (strict JSON)
- Key principles: patience, one problem at a time, small moves, watch the critic

### What the Coach Can Change

| Parameter | Range | Rate Limit |
|-----------|-------|-----------|
| Any reward weight (except frozen) | Per-term bounds in CoachConfig | ±20% per consultation, max 3 at a time |
| Learning rate | Phase-specific floor to ceiling | One change per consultation |
| max_noise_std | 0.2 to phase max | One change per consultation |

### What the Coach Cannot Change

- `stumble` weight (frozen at 0.0 — Bug #28b)
- `body_height_tracking` weight (frozen at 0.0 — Bug #22)
- Terrain type or difficulty
- Number of environments
- Network architecture
- Phase boundaries (humans decide phase transitions)

---

## 7. TensorBoard: What to Watch

Start TensorBoard:

```bash
# Local
tensorboard --logdir logs/rsl_rl/spot_robust_ppo/ --port 6007

# H100
ssh -L 6006:localhost:6006 t2user@172.24.254.24
tensorboard --logdir ~/IsaacLab/logs/rsl_rl/spot_robust_ppo/<RUN_DIR> --port 6006
```

### Standard Training Panels

| Panel | Healthy | Warning | Danger |
|-------|---------|---------|--------|
| Mean reward | Climbing | Flat for 500+ iters | Falling |
| Flip rate | <10% | 10-30% | >50% |
| Terrain level | Climbing | Flat (plateau) | Falling |
| Value loss | <20 | 20-50 | >100 |
| Noise std | Decreasing toward 0.3 | At ceiling (0.5) | Increasing above ceiling |

### AI Coach Panels (Custom)

These appear under `AI_Coach/`, `Reward_Weights/`, `Reward_Contrib/`, and `Weight_Changes/`:

| Panel | What It Shows |
|-------|--------------|
| `AI_Coach/action_code` | 0=no_change, 1=adjust_weights, 2=adjust_noise, 3=adjust_lr |
| `AI_Coach/confidence` | Coach's self-reported confidence (0-1) |
| `AI_Coach/api_latency_ms` | Round-trip time to Claude API |
| `AI_Coach/num_changes_applied` | How many weights were actually changed |
| `AI_Coach/guardrail_blocks` | Number of guardrail rejections/modifications |
| `AI_Coach/reward_trend` | Linear slope of mean reward |
| `AI_Coach/terrain_trend` | Linear slope of terrain level |
| `Reward_Weights/*` | Current weight for each of the 22 terms |
| `Reward_Contrib/*` | Per-episode contribution of each reward term |
| `Weight_Changes/*` | Delta applied at each coach consultation |
| `AI_Coach/emergency` | Emergency events (0=none, 2=halve_lr, 3=stop) |

### What Good AI-Guided Training Looks Like

**First 200 iters:** Coach says `no_change` every time. This is correct — early training metrics are too noisy for meaningful intervention.

**Iter 200-500:** Coach might make its first adjustment — typically reducing a dominant penalty that's suppressing exploration, or nudging a reward higher.

**Iter 500+:** Coach settles into a pattern: `no_change` most consultations, occasional small adjustments when it detects a clear trend. `Reward_Weights/` lines should be mostly flat with occasional step changes.

**Red flags:** Many consecutive `adjust_weights` decisions, `guardrail_blocks` > 0 on most calls, emergency events, or `confidence` consistently below 0.5.

---

## 8. Decision Log: What the AI Changed and Why

Every coach interaction is saved to `ai_coach_decisions.jsonl` in the run's log directory. Each line is a JSON object:

```json
{
  "timestamp": "2026-03-08T19:38:22",
  "iteration": 100,
  "phase": "flat",
  "metrics": {
    "mean_reward": 45.2,
    "flip_rate": 0.35,
    "terrain_level": 0.0,
    "value_loss": 2.1,
    "reward_breakdown": {"gait": 0.85, "action_smoothness": -0.42, ...}
  },
  "decision": {
    "action": "no_change",
    "reasoning": "Metrics trending well. Reward climbing steadily...",
    "confidence": 0.9
  },
  "guardrail_messages": [],
  "applied_changes": {},
  "api_latency_ms": 2847.3
}
```

When the coach makes a change:

```json
{
  "decision": {
    "action": "adjust_weights",
    "reasoning": "base_motion penalty is dominant at -4.2/episode, suppressing exploration",
    "weight_changes": {"base_motion": -3.5},
    "confidence": 0.75
  },
  "guardrail_messages": ["BOUNDED base_motion: -3.5 -> -3.6 (max 20% change)"],
  "applied_changes": {"base_motion": [-4.0, -3.6]}
}
```

### Reviewing After Training

```bash
# Count decisions by type
cat ai_coach_decisions.jsonl | python -c "
import json, sys, collections
actions = collections.Counter()
for line in sys.stdin:
    d = json.loads(line)
    if 'decision' in d:
        actions[d['decision']['action']] += 1
for k, v in actions.most_common():
    print(f'{k}: {v}')
"

# Show all actual changes
grep '"adjust_weights"' ai_coach_decisions.jsonl | python -m json.tool
```

---

## 9. Manual Overrides and Escape Hatches

### Disable the Coach Mid-Run

If the coach is making bad decisions, kill the training process and restart with `--no_coach`. The coach's changes to reward weights are already baked into the running env, but they reset when you create a new env from the config file. The policy checkpoint retains whatever it learned.

### Override Phase Config

The `ai_trainer/config.py` file contains all phase definitions. Edit `PHASE_CONFIGS` to change:
- Go/no-go thresholds
- LR ceilings
- Noise bounds
- Iteration counts

### Add Frozen Weights

To prevent the coach from touching a specific weight, add it to:
1. `guardrails.py` → `FROZEN_WEIGHTS` (hardcoded, applies to all phases)
2. Or the phase's `PhaseConfig.frozen_weights` set (phase-specific)

### Emergency: Coach API Down

If the Anthropic API is unreachable, the coach automatically falls back to `no_change` after 3 consecutive failures. Training continues normally without coach intervention. You'll see:

```
[AI-COACH] API error (1/3): Connection timeout
[AI-COACH] API error (2/3): Connection timeout
[AI-COACH] API error (3/3): Connection timeout
[AI-COACH] Too many failures, falling back to no_change for this session
```

## 9.5. Human Visual Observation Injection (Zero-Downtime Feedback)

The AI coach excels at reading numerical metrics, but some problems — bouncy gaits, legs crossing, poor directional control — are only visible by watching the policy run in simulation. The system supports two methods for injecting human observations into the coach's decision loop **without restarting training.**

### Method 1: `~/human_notes.txt` (Built-in)

Write your observation to `~/human_notes.txt` on the H100:

```bash
ssh t2user@172.24.254.24 "cat > ~/human_notes.txt << 'EOF'
Legs are crossing during turns. Robot is unstable on flat ground.
Gait looks bouncy — too much vertical motion. Hard to control velocity.
EOF"
```

At the next coach consultation, `prompt_builder.py` reads this file, includes it in the prompt under "## Human Observation (from visual evaluation)" with a note to weight qualitative observations heavily, then renames it to `human_notes_consumed.txt` so the same note isn't sent twice.

### Method 2: JSONL Decision Log Injection (Immediate)

For immediate feedback without waiting for the next coach interval, append directly to the decision log:

```bash
ssh t2user@172.24.254.24 "cat >> ~/IsaacLab/logs/rsl_rl/spot_robust_ppo/<RUN_DIR>/ai_coach_decisions.jsonl << 'EOF'
{\"timestamp\": \"$(date -Iseconds)\", \"iteration\": \"human_eval\", \"phase\": \"robust\", \"metrics\": {}, \"decision\": {\"action\": \"human_observation\", \"reasoning\": \"HUMAN VISUAL EVALUATION: Legs crossing, unstable gait, hard to control.\"}, \"guardrail_messages\": [], \"applied_changes\": {}}
EOF"
```

The coach's "Recent Coach Decisions" section reads the last 3 entries from this file. Your observation appears in the very next prompt, and the coach factors it into its analysis alongside the numerical metrics.

### How It Worked (Trial 11l, Iter 2500)

A human pulled `model_2900.pt` into the lava arena and observed legs crossing, instability, and poor velocity tracking. The observation was injected via JSONL method. At iter 2500, the coach read it and made 3 targeted changes:
- `joint_pos`: -0.3 → -0.36 (penalize leg crossing)
- `action_smoothness`: -1.0 → -1.2 (smoother joint commands)
- `base_motion`: -2.0 → -2.4 (reduce instability)

No restart. No config edit. No kill-and-relaunch. The training continued seamlessly with the human's visual insight integrated into the AI's decision-making.

---

## 10. Troubleshooting

### Training Crashes

| Error | Cause | Fix |
|-------|-------|-----|
| `terrain_types` AttributeError | Running flat terrain with curriculum enabled | `apply_phase_terrain()` disables curriculum for flat |
| `terrain_levels` AttributeError | Same — curriculum function expects generated terrain | Fixed in train_ai.py |
| NaN at iter N | Value function instability | Emergency check auto-handles; lower LR ceiling |
| `anthropic.APIError` | API key invalid or rate limit | Check key; coach falls back to no_change |
| CUDA out of memory | Too many envs for GPU | Reduce --num_envs |

### Coach Makes Bad Decisions

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Too many changes, reward dropping | Confidence threshold too low | Coach is conservative by default; check decision log |
| Coach never changes anything | Check interval too low, or training is genuinely healthy | This is usually correct behavior |
| Guardrail blocks every change | Coach requesting out-of-bounds values | This is the guardrails working correctly |
| Coach suggests frozen weight | System prompt not loaded correctly | Check prompt_builder.py frozen weight list |

### Flat Terrain Specific

When `--start_phase flat`, the system automatically:
- Sets terrain type to "plane"
- Disables terrain curriculum (`curriculum.terrain_levels = None`)
- VegetationDragReward skips terrain-aware logic (no `terrain_types` on plane)

---

## 11. Run History

### AI-Guided Runs

| Trial | Date | Phase | Envs | Iters | Coach | Result | Notes |
|-------|------|-------|------|-------|-------|--------|-------|
| AI-01 | 2026-03-08 | flat | 250 | 1000 | every 20 | IN PROGRESS | Local smoke test. Fixed terrain_types + terrain_levels bugs |

### Key Lessons from AI Coach Runs

*(Will be populated as runs complete)*

1. **The coach is patient.** In early training (first 200 iters), the coach correctly says `no_change` every time. It recognizes that high flip rates and low rewards are normal for a random policy.

2. **API latency is ~3s.** With `claude-sonnet-4-20250514`, each consultation takes 2-4 seconds. At every 100 iterations (~15s each), this is negligible overhead.

3. **Flat terrain needs special handling.** Isaac Lab's `TerrainImporter` in plane mode doesn't have `terrain_types`, `terrain_levels`, or curriculum infrastructure. Must disable curriculum and guard terrain-aware reward functions.

---

## 12. Quick Reference: Commands and Paths

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--robot` | spot | Robot type (spot or vision60) |
| `--start_phase` | flat | Starting phase |
| `--end_phase` | robust | Ending phase |
| `--num_envs` | 5000 | Parallel environments |
| `--coach_interval` | 100 | Consult coach every N iters |
| `--coach_model` | claude-sonnet-4-20250514 | Claude model for coach |
| `--no_coach` | false | Disable AI coach |
| `--anthropic_api_key` | env var | Anthropic API key |
| `--save_interval` | 100 | Checkpoint frequency |
| `--lr_max` | per-phase | Override phase LR max |
| `--max_noise_std` | 0.5 | Noise ceiling |
| `--min_noise_std` | 0.3 | Noise floor |
| `--load_run` | none | Resume from run directory |
| `--load_checkpoint` | none | Resume from specific checkpoint |
| `--no_wandb` | false | Use TensorBoard instead of W&B |

### File Paths

```
multi_robot_training/
├── train_ai.py                    <-- Main entry point
├── ai_trainer/
│   ├── __init__.py                <-- Package exports
│   ├── config.py                  <-- Phase configs + CoachConfig
│   ├── metrics.py                 <-- MetricsCollector + MetricsSnapshot
│   ├── coach.py                   <-- Claude API interface
│   ├── guardrails.py              <-- Safety validation layer
│   ├── actuator.py                <-- Applies changes to live env
│   ├── decision_log.py            <-- JSONL audit trail
│   └── prompt_builder.py          <-- System/user prompt construction
├── configs/
│   ├── spot_ppo_env_cfg.py        <-- Spot environment config
│   └── spot_ppo_cfg.py            <-- Spot PPO runner config
├── shared/
│   ├── reward_terms.py            <-- Custom reward functions
│   ├── terrain_cfg.py             <-- 12-terrain curriculum
│   ├── lr_schedule.py             <-- Cosine annealing LR
│   └── training_utils.py          <-- TF32, noise clamping, NaN safety
└── logs/rsl_rl/spot_robust_ppo/
    └── <timestamp>/
        ├── model_*.pt             <-- Checkpoints
        ├── events.out.tfevents.*  <-- TensorBoard data
        ├── ai_coach_decisions.jsonl <-- Coach decision log
        └── params/                <-- Saved configs
```

### Monitoring

```bash
# Check training progress
tail -f ~/ai_train_full.log | grep -E "AI-COACH|AI-TRAIN|Learning iteration"

# Watch just coach decisions
tail -f ~/ai_train_full.log | grep "AI-COACH"

# TensorBoard (local)
tensorboard --logdir logs/rsl_rl/spot_robust_ppo/ --port 6007

# Review decision log
cat logs/rsl_rl/spot_robust_ppo/<run>/ai_coach_decisions.jsonl | python -m json.tool
```
