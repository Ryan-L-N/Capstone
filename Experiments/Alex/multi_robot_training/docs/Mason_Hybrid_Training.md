# Mason Hybrid Training: The Best of Both Worlds

> **TL;DR:** We took Mason's proven reward weights and smaller neural network, put them on our harder terrain, and added the AI Coach as a safety net. The goal: break through the terrain 4.83 ceiling that our custom config couldn't crack.
>
> **Status:** Both trainings COMPLETE. Hybrid no-coach: terrain 3.74 (42.6 hrs). AI-coached v8: terrain 4.83 (47 hrs).
> **Hardware:** NVIDIA H100 NVL 96GB
> **MH-1 run dir:** `spot_hybrid_ppo/2026-03-10_18-55-33/` (retired)
> **MH-2a no-coach run dir:** `spot_hybrid_ppo/2026-03-11_11-28-30/` — 20,000 iters, model_19999.pt
> **AI-coached v8 run dir:** `spot_robust_ppo/2026-03-09_12-47-39/` — 10,600 iters, model_10600.pt

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

## MH-1 Results: Coach Destroyed Gait Quality

MH-1 reached terrain 4.83 but the AI Coach repeated the same mistake as Trial 11l — it optimized for terrain numbers because it couldn't see the robot. The coach:
- Repeatedly loosened penalties to boost terrain advancement
- Boosted `base_linear_velocity` from 5.0 toward 14.26
- Produced a "flopping fish" robot that couldn't stand up (height ≈ 0.0m)

**Root causes:**
1. **No visual feedback** — coach only saw numbers, not the flopping gait
2. **Terrain-scaled height target** — `terrain_scaled=True` let the height target drop on rough terrain, so the robot learned to crawl
3. **No penalty-loosening guardrails** — coach could loosen penalties at any terrain level

## Mason Hybrid v2 (MH-2): The Fix

Three changes to prevent MH-1's failure mode:

### 1. VLM Visual Feedback (`--enable_vision`)
The coach now receives a rendered frame from the simulation at every consultation. Claude Sonnet analyzes the robot's posture and gait visually alongside the metrics. A "visual override rule" prevents terrain advancement when gait looks bad.

### 2. Fixed 0.37m Height Target
Changed `terrain_relative_height` from `terrain_scaled=True` (variable 0.35-0.42m) to `terrain_scaled=False, target_height=0.37`. The robot MUST stand at 37cm regardless of terrain difficulty.

### 3. Gait-Quality-First Prompt
Complete rewrite of the coach's system prompt:
- **Core philosophy:** "A smooth trot at terrain 4 is BETTER than a bouncy hop at terrain 6"
- **Terrain-gated penalty loosening:** Penalties locked until terrain >= 4.0
- **Velocity ceiling:** `base_linear_velocity` and `base_angular_velocity` must stay in 3.0-7.0
- **New troubleshooting entries:** "Flopping/unstable gait" and "Robot not standing up"

### MH-2a: No-Coach Baseline (COMPLETE — March 13, 2026)

After MH-1 and MH-2 VLM attempts, we decided to let Mason's config prove itself without
any AI coach interference. Pure training: Mason's rewards + our terrain + safety fixes.

**Script:** `scripts/rsl_rl/train_hybrid.py` — dedicated script, no coach code.

**Key differences from `train_ai.py`:**
- No AI coach, no VLM, no API calls
- No cosine LR — uses Mason's adaptive KL schedule (RSL-RL built-in)
- Keeps: value loss watchdog, noise clamping, std safety, checkpoints

**Final results (20,000 iterations, 42.6 hours, 2.0B steps):**
- Terrain level: 3.74 (plateaued — did not break through)
- Survival: 53%
- Flip rate: 0%
- Mean reward: ~320
- Conclusion: Mason's config alone plateaus at terrain ~3.7 on our harder terrain. The AI coach is needed to push further.

**Checkpoint:** `checkpoints/hybrid_nocoach_19999.pt` (6.6 MB)
**Run dir:** `spot_hybrid_ppo/2026-03-11_11-28-30/`

### AI-Coached v8 (Trial 11l): COMPLETE — March 13, 2026

Our custom [1024,512,256] config with AI coach (Claude Sonnet). Resumed from v7's model_3900.pt with baked reward weights from all prior coach interventions.

**Final results (~10,600 iterations, ~47 hours, ~2.0B steps):**
- Terrain level: 4.83 (our best ever, but plateaued)
- Last saved checkpoint: model_10600.pt (training continued ~2 more days but stopped saving)
- 68 checkpoints saved total
- Mean reward: 640-682

**Checkpoint:** `checkpoints/ai_coached_v8_10600.pt` (21 MB, [1024,512,256] architecture)
**Run dir:** `spot_robust_ppo/2026-03-09_12-47-39/`

**Monitor:**
```bash
tail -f ~/mason_hybrid_nocoach_train.log
grep -E 'terrain_levels|Mean reward|flip_over' ~/mason_hybrid_nocoach_train.log | tail -10
```

### 4-Environment Eval Results (2026-03-12 — 2026-03-13)

#### Mason Hybrid No-Coach (model_13000.pt) — Single Episode

| Environment | Status | Progress | Zones | Time | Notes |
|-------------|--------|----------|-------|------|-------|
| Friction | COMPLETE | 49.5m | 5/5 | 273.9s | Clean traversal, stable gait (mean_roll=0.044 rad) |
| Grass | — | — | — | — | Not yet tested |
| Boulder | FELL | 31.6m | 3/5 | 88.3s | With PhysX raycasting (+11m vs 20.6m without) |
| Stairs | FELL | 12.7m | 2/5 | 95.2s | PhysX raycasting confirmed seeing stair geometry |

#### AI-Coached v8 (model_10600.pt) — Single Episode

| Environment | Status | Progress | Zones | Time | Notes |
|-------------|--------|----------|-------|------|-------|
| Friction | COMPLETE | 49.5m | 5/5 | 50.2s | Much faster than hybrid (more aggressive gait) |
| Grass | FELL | 41.2m | 5/5 | 121.0s | Reached all zones before falling |
| Boulder | FELL | 23.0m | 3/5 | 34.8s | Similar zones to hybrid but less progress |
| Stairs | FELL | 12.7m | 2/5 | 37.1s | Same zones as hybrid |

#### Mason Baseline (model_19999.pt) — 100-Episode Sweep (Partial)

| Environment | Episodes | Mean Progress | Std | Fall Rate | Notes |
|-------------|----------|---------------|-----|-----------|-------|
| Friction | 100/100 | 33.5m | ±10.0m | 83% | High variance |
| Grass | 50/100 | 28.9m | ±6.7m | 22% | Interrupted at 50 |
| Boulder | 0/100 | — | — | — | CUDA error |
| Stairs | 0/100 | — | — | — | Not yet run |

**PhysX Raycasting (Bug E-6):** Originally the height scan was all zeros (flat ground assumption) — the policy was blind to terrain geometry. Implemented `omni.physx.get_physx_scene_query_interface()` raycasting: 187 rays, 17×11 grid, self-collision filtering. Boulder improved +11m with raycasting. Stairs confirmed hitting `/World/Staircase/zone_1/step_0,1,2`.

**Deployment notes:** Required several fixes to `4_env_test/src/` for standalone mode:
- **Decimation = 1** — standalone loop runs at 50 Hz (not 500 Hz), so policy decimation must be 1 (not 10). Without this fix, policy ran at 5 Hz → violent oscillation → immediate fall.
- **`--mason` flag** — obs order: height_scan(187) first, proprioception(48) last. Action scale 0.2.
- **Waypoint follower fix** — `is_done` triggered at 39.5m (premature). Changed `>=` to `>`.
- **Scene lighting** — dome + sun lights added (was pitch black).
- **PhysX raycasting** — height scan now uses real scene queries instead of zeros. All 4 envs use raycast.
- See `4_env_test/LESSONS_LEARNED.md` Bugs E-1 through E-6 for full details.

### MH-2b: VLM Coach (BLOCKED — needs Vulkan on H100)

The VLM pipeline works (tested locally) but the H100 has no Vulkan drivers for offscreen
rendering. `--enable_cameras` fails without `libnvidia-gl-575`. Install requires sudo:
```bash
sudo apt-get install -y libnvidia-gl-575 libvulkan1
```

Once Vulkan is available, launch with:
```bash
python scripts/rsl_rl/train_ai.py \
  --task Locomotion-MasonHybrid-Spot-v0 \
  --headless --enable_cameras --enable_vision \
  --no_wandb --save_interval 100 \
  --num_envs 4096 \
  --start_phase mason_hybrid --end_phase mason_hybrid \
  --coach_interval 100 --coach_mode deferred --activation_threshold 300 \
  --max_noise_std 1.0
```

---

## Launch Bugs (MH-1 Run History)

Three bugs hit during the first launch sequence. All fixed — documented here so they don't bite again.

### Bug MH-1: TensorBoard Points at Wrong Log Directory

**Symptom:** TensorBoard stuck at iter 96 while training was at iter 350+.

**Root cause:** When launching with `cd /home/t2user/multi_robot_training_new`, RSL-RL writes logs to `~/logs/rsl_rl/spot_hybrid_ppo/` (relative to cwd), not `~/IsaacLab/logs/rsl_rl/spot_hybrid_ppo/` where we pointed TensorBoard. The initial 96 iters came from a brief first launch attempt that *did* use the IsaacLab path.

**Fix:** Point TensorBoard at the actual log directory:
```bash
tensorboard --logdir ~/logs/rsl_rl/spot_hybrid_ppo/ --port 6006 --bind_all
```

**Lesson:** Always check where events files are actually being written (`find ~ -name 'events.out*' -mmin -30`) before assuming the log path.

### Bug MH-2: Invalid API Key — Coach Dies After 3 Failures

**Symptom:** Coach activated at iter 300 but every API call returned `401 authentication_error`. After 3 consecutive failures, the coach disabled itself permanently for the run.

**Root cause:** The `~/.anthropic_key` file on the H100 had an expired API key. The launch script reads it via `export ANTHROPIC_API_KEY=$(cat ~/.anthropic_key)`, so a bad file = a dead coach.

**Fix:** Updated `~/.anthropic_key` with a fresh key. But the failure counter (`_consecutive_failures = 3`) couldn't be reset without restarting — so we had to kill and resume from `model_400.pt`.

**Lesson:** Verify the API key works *before* launching a multi-hour training run:
```bash
curl -s https://api.anthropic.com/v1/messages \
  -H "x-api-key: $(cat ~/.anthropic_key)" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-sonnet-4-20250514","max_tokens":10,"messages":[{"role":"user","content":"ping"}]}' \
  | head -c 100
```

### Bug MH-3: Resume Can't Find Checkpoint — Log Root Mismatch

**Symptom:** `FileNotFoundError: No such file or directory: '/home/t2user/multi_robot_training_new/logs/rsl_rl/spot_hybrid_ppo'` when resuming with `--load_run`.

**Root cause:** `get_checkpoint_path()` in Isaac Lab constructs the resume path from `log_root_path`, which is based on cwd. Training ran from `~/multi_robot_training_new/`, so it looked for checkpoints in `~/multi_robot_training_new/logs/...`. But the checkpoints were in `~/logs/rsl_rl/spot_hybrid_ppo/` (where the first run wrote them).

**Fix:** Symlink the log directory:
```bash
mkdir -p ~/multi_robot_training_new/logs/rsl_rl
ln -sf ~/logs/rsl_rl/spot_hybrid_ppo ~/multi_robot_training_new/logs/rsl_rl/spot_hybrid_ppo
```

**Lesson:** RSL-RL's log path is always `{cwd}/logs/rsl_rl/{experiment_name}/`. If you change the working directory between runs, checkpoints won't be found. Symlink or always launch from the same directory.

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
| `scripts/rsl_rl/train_ai.py` | AI coach training script (VLM, deferred activation) |
| `scripts/rsl_rl/train_hybrid.py` | No-coach training script (pure Mason config + our safety) |

---

## Lessons From Trial 11l That Shaped This Design

1. **Simpler rewards > more rewards.** 22 terms meant the policy had 22 conflicting signals. Mason's 11 terms give a clearer gradient.

2. **Smaller networks generalize better.** Our [1024,512,256] network memorized easy-terrain strategies. Mason's [512,256,128] has to learn more general locomotion.

3. **Adaptive LR > manual scheduling.** Our cosine annealing with lr_max=3e-5 was a ceiling we chose based on crash history. Mason's adaptive KL starts at 1e-3 and self-adjusts — no human guessing required.

4. **The AI Coach works, but needs guardrails.** Trial 11l proved the coach can break plateaus (gait 10→8.5 at iter 1200). It also proved unconstrained optimization drifts (velocity 5→14.26). The fix: tighter bounds, deferred activation, and LR/noise locked.

5. **Don't fight proven baselines.** Mason reached terrain ~6 without an AI coach. Our config couldn't break 4.83 with one. The simplest explanation is usually correct — the baseline was better.

---

*"The best code is no code. The best reward is no reward. And the best AI coach is one that mostly says 'no_change.'"*
