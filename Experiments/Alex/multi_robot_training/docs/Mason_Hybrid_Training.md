# Mason Hybrid Training: The Best of Both Worlds

> **TL;DR:** We took Mason's proven reward weights and smaller neural network, put them on our harder terrain, and added the AI Coach as a safety net. The goal: break through the terrain 4.83 ceiling that our custom config couldn't crack.
>
> **Status:** IN PROGRESS (launched March 10, 2026)
> **Hardware:** NVIDIA H100 NVL 96GB
> **TensorBoard:** `http://172.24.254.24:6006`
> **Run dir:** `spot_hybrid_ppo/2026-03-10_18-55-33/`

---

## What Is This?

After 11 trials and ~30 sub-iterations of tuning our custom reward config, we hit a wall at terrain level 4.83. The AI Coach kept boosting velocity rewards trying to push through, but the problem wasn't tuning — it was complexity. Our config had 22 reward terms, a 2.4M-parameter network, and so many competing penalties that the policy couldn't figure out what we actually wanted.

Meanwhile, Mason's team independently reached terrain ~6 with a *simpler* setup: 11 reward terms, an 800K-parameter network, and adaptive learning rate. Sometimes less is more.

The Mason Hybrid takes the best of both:
- **Mason's rewards** — 11 clean, proven terms with weights that work
- **Our terrain** — 12 types with friction randomization, boulders, and 10 difficulty rows
- **Our safety features** — clamped penalties (Bug #29), terrain-relative height (Bug #27), frozen broken weights (Bugs #22, #28b)
- **AI Coach** — but in deferred mode, so it watches silently while Mason's config proves itself

---

## Why This Approach?

### What Was Wrong With Our Config

| Problem | Evidence |
|---------|----------|
| Too many reward terms (22) | Coach adjustments caused cascading side effects — fix one thing, break another |
| Network too large (2.4M params) | Overfitted to easy-terrain survival rather than discovering hard-terrain strategies |
| Velocity reward drift | Coach boosted velocity 5→14.26 chasing tracking error — classic positive feedback loop |
| Overly aggressive DR | Mass ±5kg, friction 0.15-1.0 added noise the policy couldn't learn through |

### What Mason Got Right

| Feature | Mason's Value | Why It Works |
|---------|--------------|-------------|
| 11 reward terms | Fewer signals = clearer gradient | Policy knows exactly what we want |
| [512, 256, 128] network | 800K params — generalizes better | Less capacity = less overfitting to easy strategies |
| Adaptive KL schedule | LR auto-adjusts to training dynamics | No manual LR tuning, no cosine schedule decay |
| joint_pos = -0.7 | Strong joint position penalty | Prevents the wild leg movements that killed our gait quality |
| velocity_threshold = 0.5 | Higher bar for "moving" | Gait reward doesn't activate for tiny movements |
| mode_time = 0.3 | Longer gait cycle expectation | More natural trot cadence |

### What We Added (3 Surgical Fixes)

1. **`terrain_relative_height`** (weight -2.0) — Without this, the robot discovers belly-crawling as a survival strategy. Uses height-scan variance to set target: stand tall on flat ground (0.42m), crouch on rough terrain (0.35m).

2. **`dof_pos_limits`** (weight -3.0) — Penalizes joints approaching URDF limits. Without this, the policy locks knees at mechanical stops.

3. **`clamped_action_smoothness`** — Mason's config uses the raw Isaac Lab `action_smoothness` penalty, which returns unbounded L2 norms. When things go wrong, unbounded norms go to infinity → NaN. Our clamped version caps at 10.0 (Bug #29 safety).

---

## The 3-Stage Coach Activation

The AI Coach doesn't touch anything at first. Mason's config is proven — the coach shouldn't mess with it unless there's clear evidence of a problem.

```
Stage 1: SILENT (iters 0 → first plateau)
├── Collects metrics every 100 iterations
├── Logs to JSONL for post-analysis
├── NO API calls, NO changes
└── Prints status every 500 iters: "Silent mode — collecting metrics"

Stage 2: PASSIVE (after 300-iter plateau detected)
├── API calls begin (Claude Sonnet)
├── System prompt says: "RESPECT THE BASELINE"
├── Biased toward no_change
└── Only intervenes for CLEAR plateaus or regressions

Stage 3: ACTIVE (after first passive intervention)
├── Full intervention capability
├── BUT: tighter bounds than Trial 11l
│   ├── Velocity rewards: 3.0-7.0 (not 1.0-15.0)
│   ├── joint_pos: -1.0 to -0.3 (never looser than -0.3)
│   └── LR and noise changes DISABLED
└── Same max-3-changes, 20% max delta rules
```

**Why deferred?** In Trial 11l, the coach started adjusting immediately and eventually drifted velocity rewards to 14.26 — far from the proven baseline. Deferred activation means Mason's weights get a fair chance to work on their own. The coach is the safety net, not the driver.

---

## Configuration Comparison

| Parameter | Trial 11l (our config) | MH-1 (Mason Hybrid) |
|-----------|----------------------|---------------------|
| **Network** | [1024, 512, 256] — 2.4M params | [512, 256, 128] — 800K params |
| **Reward terms** | 22 | 14 (Mason's 11 + 3 additions) |
| **LR schedule** | Cosine annealing, lr_max=3e-5 | Adaptive KL, starts 1e-3 |
| **Noise** | Fixed 0.3-0.35 | Adaptive (init 1.0, KL manages) |
| **Domain randomization** | Heavy (mass ±5kg, friction 0.15-1.0) | Light (mass ±2.5kg, friction 0.3-1.0) |
| **Observation noise** | Enabled (corruption) | Disabled |
| **Gait velocity threshold** | 0.25 | 0.5 |
| **joint_pos weight** | -0.3 | -0.7 |
| **base_orientation weight** | -2.4 (drifted by coach) | -3.0 (Mason's) |
| **Episode length** | 30s | 20s |
| **Steps per env** | 32 | 24 |
| **Mini-batches** | 64 | 4 |
| **Envs** | 5000 | 4096 |
| **Terrain** | ROBUST_TERRAINS_CFG (12 types) | Same |
| **Coach mode** | Immediate | Deferred (3-stage) |

---

## Running This Training

### Launch Command

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export ANTHROPIC_API_KEY=$(cat ~/.anthropic_key)

screen -dmS hybrid_train bash -c '
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab &&
  export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 &&
  export ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) &&
  cd /home/t2user/multi_robot_training_new &&
  python scripts/rsl_rl/train_ai.py \
    --task Locomotion-MasonHybrid-Spot-v0 \
    --headless --no_wandb \
    --num_envs 4096 --save_interval 100 \
    --start_phase mason_hybrid --end_phase mason_hybrid \
    --coach_interval 100 --coach_mode deferred --activation_threshold 300 \
    --max_noise_std 1.0 \
    2>&1 | tee ~/mason_hybrid_train.log
'
```

### TensorBoard

```bash
screen -dmS tb_hybrid bash -c '
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab &&
  tensorboard --logdir ~/IsaacLab/logs/rsl_rl/spot_hybrid_ppo/2026-03-10_18-55-33/ \
    --port 6006 --bind_all
'
```

### Monitoring

```bash
# Watch training log
tail -f ~/mason_hybrid_train.log

# Check GPU usage
nvidia-smi

# Quick metrics check
grep -E 'terrain_levels|Mean reward|flip_over' ~/mason_hybrid_train.log | tail -10
```

### Resuming From a Checkpoint

```bash
python scripts/rsl_rl/train_ai.py \
  --task Locomotion-MasonHybrid-Spot-v0 \
  --headless --no_wandb \
  --num_envs 4096 --save_interval 100 \
  --start_phase mason_hybrid --end_phase mason_hybrid \
  --coach_interval 100 --coach_mode deferred --activation_threshold 300 \
  --max_noise_std 1.0 \
  --load_run 2026-03-10_18-55-33 --load_checkpoint model_XXXX.pt
```

---

## What to Watch For

| Milestone | Expected Iteration | What It Means |
|-----------|-------------------|---------------|
| Terrain > 1.0 | 300-500 | Basic locomotion works, robots surviving |
| Terrain > 3.0 | 1000-2000 | Where our old config started plateauing |
| Terrain > 5.0 | 3000-5000 | Past our ceiling — Mason's config is better |
| Coach activates | Varies | Plateau detected, coach enters passive mode |
| Terrain > 6.0 | 5000-10000 | Target zone — real rough terrain mastery |
| Terrain > 8.0 | 10000+ | Stretch goal — stairs, gaps, steep slopes |

**Red flags:**
- Flip rate > 15% sustained — terrain too hard, may need to back off
- Value loss > 15 — instability, but adaptive KL should self-correct
- Coach making more than 3 interventions in 1000 iters — too aggressive, check bounds
- Velocity rewards drifting above 7.0 — bounds should prevent this, but watch for it

---

## Key Files

| File | Purpose |
|------|---------|
| `pkg/tasks/locomotion/config/spot/mason_hybrid_env_cfg.py` | Environment config (rewards, DR, terminations, terrain) |
| `pkg/tasks/locomotion/config/spot/agents/rsl_rl_mason_hybrid_cfg.py` | PPO config (network, LR, hyperparams) |
| `pkg/tasks/locomotion/config/spot/__init__.py` | Gym registration (`Locomotion-MasonHybrid-Spot-v0`) |
| `pkg/ai_trainer/config.py` | Coach config + mason_hybrid phase + tighter bounds |
| `pkg/ai_trainer/guardrails.py` | Safety checks (LR/noise disabled, tight weight bounds) |
| `pkg/ai_trainer/coach.py` | Passive mode support |
| `pkg/ai_trainer/prompt_builder.py` | Passive mode preamble ("RESPECT THE BASELINE") |
| `scripts/rsl_rl/train_ai.py` | Training script (adaptive LR detection, deferred activation) |

---

## Lessons From Trial 11l That Shaped This Design

1. **Simpler rewards > more rewards.** 22 terms meant the policy had 22 conflicting signals. Mason's 11 terms give a clearer gradient.

2. **Smaller networks generalize better.** Our [1024,512,256] network memorized easy-terrain strategies. Mason's [512,256,128] has to learn more general locomotion.

3. **Adaptive LR > manual scheduling.** Our cosine annealing with lr_max=3e-5 was a ceiling we chose based on crash history. Mason's adaptive KL starts at 1e-3 and self-adjusts — no human guessing required.

4. **The AI Coach works, but needs guardrails.** Trial 11l proved the coach can break plateaus (gait 10→8.5 at iter 1200). It also proved unconstrained optimization drifts (velocity 5→14.26). The fix: tighter bounds, deferred activation, and LR/noise locked.

5. **Don't fight proven baselines.** Mason reached terrain ~6 without an AI coach. Our config couldn't break 4.83 with one. The simplest explanation is usually correct — the baseline was better.

---

*"The best code is no code. The best reward is no reward. And the best AI coach is one that mostly says 'no_change.'"*
