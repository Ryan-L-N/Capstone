# Spot Training Curriculum: Flat → Transition → Robust Easy → Robust

> **Proven recipe for teaching Spot to walk on rough terrain.**
> Each phase resumes from the previous phase's best checkpoint.
> Run phases individually — stop, verify, then proceed.
>
> **Hardware:** NVIDIA H100 NVL 96GB
> **Last updated:** March 6, 2026

---

## The Four Phases

```
Phase A          ──►  Phase A.5         ──►  Phase B-easy       ──►  Phase B
100% flat              50% flat +             12 types,               12 types,
                       gentle rough           3 difficulty rows       10 difficulty rows
500 iters, ~1.7hr     1000 iters, ~2hr       30K iters, ~21hr        30K iters, ~16hr
Learn to walk          Gentle obstacles       All types, easy         Full difficulty
```

**Why four phases?** Terrain complexity has two independent axes: terrain *type* novelty (stairs vs slopes vs gaps) and *difficulty* within each type (0.05m stairs vs 0.25m stairs). The policy can handle one axis at a time but crashes if both change at once. Phase A.5 introduces new types gently, Phase B-easy exposes all 12 types at low difficulty, Phase B cranks difficulty to max.

---

## Phase A: Flat Terrain Warmup

**Goal:** Learn basic locomotion — balance, gait, velocity tracking.

**When it's done:** time_out > 95%, flip_over < 2%, noise_std decreasing.

```bash
cd ~/IsaacLab
screen -dmS spot_train bash -c '
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab &&
    export OMNI_KIT_ACCEPT_EULA=YES && export PYTHONUNBUFFERED=1 &&
    ./isaaclab.sh -p ~/multi_robot_training/train_ppo.py --headless \
        --robot spot --terrain flat --num_envs 10000 --max_iterations 500 \
        --warmup_iters 50 --save_interval 50 --lr_max 3e-4 \
        --no_wandb --seed 42 2>&1 | tee ~/phase_a.log
'
```

**Key settings:**
| Parameter | Value | Why |
|-----------|-------|-----|
| terrain | flat | Pure locomotion signal, no terrain noise |
| num_envs | 10,000 | Enough for gradient quality, not too hot |
| max_iterations | 500 | Sufficient for flat terrain mastery |
| lr_max | 3e-4 | Safe ceiling (1e-3 causes value explosion) |
| warmup_iters | 50 | Quick warmup, then cosine decay |
| save_interval | 50 | Frequent checkpoints (21MB each) |

**What to watch:**
- Noise std should decrease (0.50 → 0.38 is ideal)
- Flip_over should drop below 5% by iter 200
- Episode length should max out at 1,500 by iter 200
- Value loss should stay below 1.0

**Expected results (from Trial 7b):**

| Iter | Reward | Ep Length | Flip Over | Time Out | Noise |
|------|--------|-----------|-----------|----------|-------|
| 50 | ~2 | ~30 | ~50% | ~1% | 0.52 |
| 100 | ~140 | ~1,150 | ~28% | ~72% | 0.54 |
| 200 | ~375 | 1,500 | ~5% | ~95% | 0.58 |
| 500 | ~567 | 1,500 | <1% | >99% | 0.38 |

**Output:** `model_498.pt` (~21MB)

---

## Phase A.5: Transition Terrain

**Goal:** Learn to handle gentle slopes, slight roughness, small stairs, and waves without forgetting how to walk.

**When it's done:** time_out > 90%, flip_over < 5%, noise_std stable (< 0.6).

**Prereq:** Phase A checkpoint with time_out > 95%.

```bash
cd ~/IsaacLab
screen -dmS spot_train bash -c '
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab &&
    export OMNI_KIT_ACCEPT_EULA=YES && export PYTHONUNBUFFERED=1 &&
    ./isaaclab.sh -p ~/multi_robot_training/train_ppo.py --headless \
        --robot spot --terrain transition --num_envs 10000 --max_iterations 1000 \
        --warmup_iters 50 --save_interval 100 --lr_max 3e-4 \
        --resume --load_run <PHASE_A_RUN_DIR> --load_checkpoint model_498.pt \
        --no_wandb --seed 42 2>&1 | tee ~/phase_a5.log
'
```

**Terrain mix (6 types, all gentle):**
| Type | Proportion | Max Difficulty | vs Robust |
|------|-----------|----------------|-----------|
| Flat plane | 50% | — | Safe zone |
| Gentle slopes | 15% | 14° (0.25 rad) | Half of robust's 29° |
| Random rough | 10% | 0.06m noise | Half of robust's 0.15m |
| Gentle stairs | 10% | 0.10m step | Half of robust's 0.25m |
| Wave terrain | 10% | 0.08m amplitude | Half of robust's 0.20m |
| Vegetation plane | 5% | — | Drag training |

**Grid:** 5 rows (difficulty) × 20 cols = 100 patches

**Key settings:**
| Parameter | Value | Why |
|-----------|-------|-----|
| terrain | transition | 50% flat + gentle rough |
| max_iterations | 1000 | ~500 new iters of terrain adaptation |
| save_interval | 100 | Checkpoint every 100 iters |
| num_rows | 5 | Fewer difficulty levels (gentler curriculum) |

**What to watch:**
- Flip_over should stay below 10% from the start (if >70%, the step is too big)
- Stumble penalty will be the biggest negative term — that's expected
- Terrain_levels should climb above 2.0
- Noise should stay below 0.6

**Expected results (from Trial 9, early data):**

| Iter | Reward | Ep Length | Flip Over | Time Out | Noise |
|------|--------|-----------|-----------|----------|-------|
| 500 (resume) | ~10 | ~150 | ~1% | ~10% | 0.39 |
| 525 | ~228 | ~760 | ~7% | ~51% | 0.50 |
| 750 | TBD | TBD | TBD | TBD | TBD |
| 1000 | ~325 | ~1,400 | ~5% | ~93% | ~1.00 |

**Output:** Best checkpoint from this run (~21MB)

---

## Phase B-easy: All 12 Types, Capped Difficulty

**Goal:** Expose the robot to every terrain type (stairs, gaps, slopes, obstacles, stepping stones) at easy-to-medium difficulty so it learns the *feel* of each type without getting destroyed.

**When it's done:** time_out > 80%, flip_over < 15%, terrain_levels climbing, value_loss < 50.

**Prereq:** Phase A.5 checkpoint with time_out > 90%.

**WARNING:** Do NOT skip this phase. Trial 10 tried going directly from Phase A.5 to full robust and crashed in 15 iterations (action_smoothness exploded to -103 trillion, 63% flip_over).

```bash
cd ~/IsaacLab
screen -dmS spot_train bash -c '
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab &&
    export OMNI_KIT_ACCEPT_EULA=YES && export PYTHONUNBUFFERED=1 &&
    ./isaaclab.sh -p ~/multi_robot_training/train_ppo.py --headless \
        --robot spot --terrain robust_easy --num_envs 20480 --max_iterations 30000 \
        --warmup_iters 500 --save_interval 100 --lr_max 5e-5 \
        --resume --load_run <PHASE_A5_RUN_DIR> --load_checkpoint <BEST_MODEL>.pt \
        --no_wandb --seed 42 2>&1 | tee ~/phase_b_easy.log
'
```

**Terrain:** Same 12 types as full robust (`ROBUST_TERRAINS_CFG`) but with `num_rows=3` (only easy/medium difficulty rows) and `num_cols=20`.

**What this caps:**
| Terrain Type | Full Robust (row 10) | Robust Easy (row 3) |
|-------------|---------------------|---------------------|
| Stairs up | 0.25m steps | ~0.10m steps |
| Gaps | 0.50m width | ~0.20m width |
| Slopes | 0.50 rad (29°) | ~0.15 rad (9°) |
| Random rough | 0.15m noise | ~0.06m noise |
| Stepping stones | 0.40m gaps | ~0.15m gaps |
| Discrete obstacles | 0.30m height | ~0.12m height |

**Key settings:**
| Parameter | Value | Why |
|-----------|-------|-----|
| terrain | robust_easy | All 12 types, 3 difficulty rows |
| num_envs | 20,480 | Full H100 scale for diverse terrain sampling |
| max_iterations | 30,000 | Long run — let the robot master all types |
| lr_max | 5e-5 | 1e-4 causes value explosion at ~1134; 3e-4 crashes instantly |
| save_interval | 100 | Frequent checkpoints (~65M steps each, prevents progress loss) |

**What to watch:**
- Flip_over should stay below 15% from the start (if >50%, something is wrong)
- Action smoothness should stay finite (if it spikes to millions, kill immediately)
- Value loss should stay below 100
- Terrain_levels should climb above 1.5

**Expected results (from Trial 10h, lr_max=5e-5):**

| Iter | Reward | Ep Length | Flip Over | Time Out | Value Loss | Terrain Levels |
|------|--------|-----------|-----------|----------|------------|----------------|
| 1000 (resume) | ~-16 | ~91 | ~8% | ~5% | ~39 | ~0.3 |
| ~1025 (danger zone) | ~+8 | ~430 | ~49% | ~23% | ~967 | ~0.5 |
| 1100 (recovery) | ~117 | ~1,099 | ~42% | ~56% | ~53 | ~0.6 |
| 1400 | ~145 | ~1,127 | ~24% | ~74% | ~7 | ~0.8 |
| 1608 | ~155 | ~1,180 | ~23% | ~75% | ~16 | ~0.8 |

**WARNING:** Expect a value_loss spike to ~1000 around iter 1025. At `lr_max=5e-5` this recovers and stays stable (<20). At `lr_max=1e-4` it recovers but re-explodes around iter 1134 (Trial 10g). At `lr_max=3e-4` the spike hits 4,670+ and crashes instantly (Trials 10c/10e/10f). Do NOT use lr_max > 5e-5 for Phase B-easy.

**Output:** Best checkpoint from this run (~21MB)

---

## Phase B: Full Robust Terrain

**Goal:** Master all 12 terrain types at full difficulty (10 rows).

**When it's done:** time_out > 80%, terrain_levels > 5.0, flip_over < 10%.

**Prereq:** Phase B-easy checkpoint with time_out > 80%.

```bash
cd ~/IsaacLab
screen -dmS spot_train bash -c '
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab &&
    export OMNI_KIT_ACCEPT_EULA=YES && export PYTHONUNBUFFERED=1 &&
    ./isaaclab.sh -p ~/multi_robot_training/train_ppo.py --headless \
        --robot spot --terrain robust --num_envs 20480 --max_iterations 30000 \
        --warmup_iters 500 --save_interval 500 --lr_max 3e-4 \
        --resume --load_run <PHASE_B_EASY_RUN_DIR> --load_checkpoint <BEST_MODEL>.pt \
        --no_wandb --seed 42 2>&1 | tee ~/phase_b.log
'
```

**Terrain mix (12 types, full difficulty):**
| Category | Types | Proportion |
|----------|-------|-----------|
| Geometric (A) | stairs up/down, boxes, stepping stones, gaps | 40% |
| Surface (B) | random rough, slopes up/down, wave, friction, vegetation | 35% |
| Compound (C) | HF stairs, discrete obstacles, repeated boxes | 25% |

**Grid:** 10 rows (difficulty) × 40 cols = 400 patches

### Difficulty Rows — What Each Level Means

All terrain parameters scale linearly from their min (row 0) to max (row 9).

#### Row 0 — Flat Playground
| Terrain | Description |
|---------|-------------|
| Stairs Up/Down | 5cm risers — gentle ramp |
| Boxes | 5cm blocks — pebble-like bumps |
| Stepping Stones | Wide stones (50cm), tight gaps (10cm) |
| Random Rough | ±2cm noise — slightly textured flat |
| Slopes | ~0° — essentially flat |
| Waves | 5cm amplitude — gentle rolls |
| HF Stairs | 5cm coarse steps |
| Discrete Obstacles | 5cm scattered blocks |
| Repeated Boxes | 20 small boxes (5cm, 30x30cm) |

#### Row 1 — Textured Ground
5-7cm features. Gravel paths, low curbs, cobblestone. Robot starts to feel terrain.

#### Row 2 — Bumpy Terrain
7-9cm features. Rough trails, uneven rubble. Requires deliberate foot placement.

#### Row 3 — Challenging Ground
*Where Phase B-easy (Trial 10k) maxed out at terrain 0.83*

10-12cm features. Rocky hiking trails, tall curbs. ~16° slopes.

#### Row 4 — Real Obstacles
*Where Trials 11/11b plateaued at 4.0-4.1. Trial 11c-v2 broke through.*

| Terrain | Description |
|---------|-------------|
| Stairs Up/Down | **14cm risers** — residential half-step |
| Boxes | 14cm blocks — construction site rubble |
| Stepping Stones | 39cm wide, 23cm gaps — deliberate stepping |
| Random Rough | ±8cm noise — rocky scramble |
| Slopes | **~22°** — steep hillside |
| Waves | 12cm amplitude — choppy terrain |
| Discrete Obstacles | **16cm blocks** — ankle-height barriers |

#### Row 5 — Serious Terrain
| Terrain | Description |
|---------|-------------|
| Stairs Up/Down | **16cm risers** — standard indoor stair height |
| Boxes | 16cm blocks — climbing over debris |
| Stepping Stones | 36cm wide, 27cm gaps — careful placement |
| Random Rough | ±9cm noise — rocky scramble trail |
| Slopes | **~28°** — steep trail switchback |
| Discrete Obstacles | 19cm blocks — shin-height obstacles |

#### Row 6 — Expert Terrain
*Approaching real-world disaster site difficulty*

| Terrain | Description |
|---------|-------------|
| Stairs Up/Down | **18cm risers** — tall standard stair |
| Boxes | 18cm blocks — serious rubble pile |
| Stepping Stones | 33cm wide, 30cm gaps — hard to bridge |
| Random Rough | ±11cm noise — boulder field |
| Slopes | **~33°** — steep mountain trail |
| Discrete Obstacles | **22cm blocks** — knee-height barriers |

#### Row 7 — Extreme Terrain
*Beyond typical quadruped benchmarks*

| Terrain | Description |
|---------|-------------|
| Stairs Up/Down | **21cm risers** — near Spot's leg reach limit |
| Boxes | 21cm blocks — climbing, not walking |
| Stepping Stones | 31cm wide, 33cm gaps — barely reachable |
| Random Rough | ±12cm noise — extreme scramble |
| Slopes | **~39°** — near-scramble steep |
| Discrete Obstacles | 24cm blocks |

#### Row 8 — Near Physical Limits
*Approaching Spot's mechanical limits*

| Terrain | Description |
|---------|-------------|
| Stairs Up/Down | **23cm risers** — requires full leg extension |
| Boxes | 23cm blocks — chest-height (for Spot) obstacles |
| Stepping Stones | 28cm wide, 37cm gaps — at Spot's stride limit |
| Slopes | **~44°** — nearly scrambling |
| Discrete Obstacles | **27cm blocks** — major barriers |

#### Row 9 — Maximum Difficulty
*Theoretical ceiling — at or beyond Spot's physical hardware limits*

| Terrain | Description |
|---------|-------------|
| Stairs Up/Down | **25cm risers** — 10 inches, taller than most real stairs |
| Boxes | 25cm blocks — over half Spot's 42cm standing height |
| Stepping Stones | **25cm wide, 40cm gaps** — extreme precision or leaping |
| Random Rough | **±15cm noise** — chaotic elevation map |
| Slopes | **~50°** — near-vertical, scrambling territory |
| Waves | **20cm amplitude** — violent undulations |
| Discrete Obstacles | **30cm blocks** — near body height |
| Repeated Boxes | **40 boxes, 20cm tall, 50x50cm** — full obstacle course |

**Practical meaning:**
- Levels 0-3: Walking on uneven ground (most indoor/outdoor flat terrain)
- Levels 4-5: Real obstacles — standard stairs, construction debris, steep hills
- Levels 6-7: Disaster-site terrain — rubble piles, extreme stairs, mountain scrambles
- Levels 8-9: At or beyond Spot's physical hardware limits — theoretical ceiling

**Key settings:**
| Parameter | Value | Why |
|-----------|-------|-----|
| terrain | robust | Full 12-type curriculum |
| num_envs | 20,480 | Full H100 scale for diverse sampling |
| max_iterations | 30,000 | Long training for terrain mastery |
| save_interval | 500 | ~60 checkpoints over the run |

**What to watch:**
- If flip_over > 50% in the first 100 iters, Phase B-easy didn't converge enough — train longer on robust_easy
- Terrain_levels should climb steadily (expect 0→5+ over 30K iters)
- Noise should stay below 0.8
- Value loss should stay below 100

**Output:** Best checkpoint — the final robust locomotion policy

---

## Reward Changes Across Phases

| Reward Term | Phase A | Phase A.5 | Phase B-easy | Phase B | Trial 11b | Trial 11c | Trial 11d | Trial 11e | Trial 11f | Trial 11h |
|-------------|---------|-----------|-------------|---------|-----------|-----------|-----------|-----------|-----------|-----------|
| undesired_contacts | -5.0 | -1.5 | -1.5 | -1.5 | -1.5 | -1.5 | -1.5 | -1.5 | -1.5 | -1.5 |
| body_scraping | — | -2.0 (new) | -2.0 | -2.0 | -2.0 | -2.0 | -2.0 | -2.0 | -2.0 | -2.0 |
| gait weight | 10.0 | 10.0 | 10.0 | 10.0 | 3.0 | **1.0** | 1.0 | 1.0 | 1.0 | 1.0 |
| gait std/max_err | 0.1/0.2 | 0.1/0.2 | 0.1/0.2 | 0.1/0.2 | 0.25/0.4 | **0.35/0.6** | 0.35/0.6 | 0.35/0.6 | 0.35/0.6 | 0.35/0.6 |
| action_smoothness | -1.0 | -1.0 | -1.0 | -1.0 | -0.3 | **-0.1** | -0.1 | -0.1 | -0.1 | -0.1 |
| base_lin_vel std | 1.0 | 1.0 | 1.0 | 1.0 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| foot_clearance | 2.0 | 2.0 | 2.0 | 2.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 |
| joint_pos | -0.7 | -0.7 | -0.7 | -0.7 | -0.7 | **-0.2** | -0.2 | -0.2 | -0.2 | -0.2 |
| base_motion | -2.0 | -2.0 | -2.0 | -2.0 | -2.0 | **-0.5** | -0.5 | -0.5 | -0.5 | -0.5 |
| stumble | -0.1 | -0.1 | -0.1 | -0.1 | -0.1 | **-0.02** | -0.02 | -0.02 | -0.02 | **0.0 (disabled)** |
| action_scale | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | **0.3** | 0.3 | 0.3 | 0.3 | 0.3 |
| terrain_relative_height | — | — | — | — | — | -1.0 (fixed 0.30m) | -1.0 (terrain_scaled: 0.42m easy → 0.25m hard) | -2.0 (curriculum-level-based: 0.42m easy → 0.35m hard) | -2.0 (variance-based, NaN) | **-2.0 (variance-based + nan_to_num + error clamped [0,1])** |
| body_height_tracking | -1.0 (disabled on rough) | same | same | same | same | replaced | replaced | replaced | replaced | replaced |
| velocity command | UniformVelocity | same | same | same | same | same | TerrainScaledVelocity (sprint on easy, careful on hard) | same | same | same |

The `undesired_contacts` and `body_scraping` changes are baked into `spot_ppo_env_cfg.py` — they apply to all phases. On flat terrain they're negligible (body rarely contacts ground). On rough terrain they prevent belly-dragging without over-punishing legitimate bumps.

The Trial 11b reward changes target the terrain ~4.1 plateau: the strict trot gait enforcer prevented the robot from discovering terrain-adaptive strategies (non-trot footwork, dynamic corrections) needed for hard terrain.

---

## Quick Checklist: Before Every Phase

- [ ] GPU clean: `nvidia-smi` shows 0 MiB, no processes
- [ ] No zombies: `ps aux | grep defunct` returns nothing
- [ ] Previous phase checkpoint verified (correct reward, noise stable)
- [ ] Code deployed: `scp` updated files to H100
- [ ] TensorBoard launched on port 6006
- [ ] Log file being written (`tail -f ~/phase_X.log`)

## Quick Checklist: Go/No-Go for Next Phase

- [ ] time_out > 90% (policy survives most episodes)
- [ ] flip_over < 5% (not falling over)
- [ ] noise_std < 0.6 (policy is converging, not uncertain)
- [ ] value_loss < 10 (value function is stable)
- [ ] terrain_levels climbing (curriculum is advancing)

If ANY metric fails, do NOT proceed. Diagnose first.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| noise_std hits ceiling (1.0) | Terrain too hard, no gradient | Go back one phase, train longer |
| value_loss > 1000 | LR too high or noise exploding | Kill immediately, resume from last checkpoint with lower lr_max (5e-5 for B-easy) |
| flip_over > 70% on new terrain | Curriculum step too large | Add intermediate terrain phase |
| reward negative and falling | Penalties dominate rewards | Check per-term breakdown, lower dominant penalty |
| terrain_levels stuck at 0-1 | Robot can't advance curriculum | Check if too many terrain types are novel. Also check if noise_std is at ceiling — lower max_noise_std to let policy be precise on hard terrain (Bug #26) |
| Training crashes with inf | Value explosion | Resume from 2-3 checkpoints back, use lr_max=5e-5 for B-easy |
| action_smoothness explodes to trillions | Chaotic falls on hard terrain | Terrain step too big; use robust_easy first |
| `normal expects std >= 0.0` crash | Policy std hit NaN from gradient explosion | `_sanitize_std()` in training_utils.py handles NaN/Inf/negative. Also ensure `lr_max=5e-5` for B-easy (1e-4 still explodes at ~1134, 3e-4 crashes instantly). Resume from earlier checkpoint. |

---

## Run History

| Trial | Phase | Terrain | Iters | lr_max | Result | Checkpoint |
|-------|-------|---------|-------|--------|--------|------------|
| 7b | A | flat | 500 | 3e-4 | SUCCESS — 99.3% survival, noise 0.38 | model_498.pt |
| 8 | B | robust | ~40 | 3e-4 | FAILED — 96.5% flip, terrain shock | — |
| 9 | A.5 | transition | 1000 | 3e-4 | SUCCESS — 92.9% survival, gait 8.58 | model_998.pt |
| 10 | B | robust | ~15 | 3e-4 | FAILED — 63% flip, action_smooth=-103T, crash | — |
| 10b | B-easy | robust_easy | ~20 | 3e-4 | FAILED — 52% flip, height_tracking=-52, crash | — |
| 10c | B-easy | robust_easy | ~25 | 3e-4 | FAILED — 59% flip, value explosion, Bug #22 fixed | — |
| 10d | B-easy | robust_easy | ~319 | 1e-4 | FAILED — 71% survival at crash, NaN std (Bug #24) | model_1000.pt |
| 10e | B-easy | robust_easy | ~35 | 3e-4 | FAILED — clamp doesn't fix NaN, wrong lr | — |
| 10f | B-easy | robust_easy | ~35 | 3e-4 | FAILED — NaN sanitizer works but lr too high, zombie policy | — |
| 10g | B-easy | robust_easy | ~134 | 1e-4 | FAILED — value explosion at iter ~1134 (2.4×10²¹) | — |
| 10h | B-easy | robust_easy | ~4037 | 5e-5 | FAILED — peaked at 155 (iter 2000), value loss cascade to NaN. Curriculum stalled at 0.8 (noise_std=1.0 too high) | model_2000.pt |
| 10k | B-easy | robust_easy | 5002 | 3e-5 | FLATLINED — reward ~216, terrain 0.83 (3-row ceiling), flip 14%, value loss 9.6. Ran 20h with 40960 envs. Policy extracted max from 3-row terrain. | model_5000.pt |
| 11 | B | robust | 6600+ | 3e-5 | PLATEAUED — terrain 4.1, reward ~250, flip 12%. Gait enforcer (weight=10, std=0.1) prevents non-trot strategies needed for hard terrain. | model_6600.pt |
| 11b | B | robust | 7400 | 3e-5 | PLATEAUED — terrain 4.0, reward ~178. Gait loosened but penalties (joint_pos, base_motion, stumble, action_smooth) still blocking hard-terrain strategies. | model_7400.pt |
| 11c | B | robust | 7600 | 3e-5 | SHORT — Tier 1+2 applied but crawling behavior observed. Penalty reduction without height enforcement caused belly-drag exploit. | model_7600.pt |
| 11c-v2 | B | robust | 8200 | 3e-5 | PLATEAUED — terrain ~4.5, height penalty oscillating -0.5 to -6.0. Fixed 0.30m target caused knee-walking on flat ground. | model_8200.pt |
| 11d | B | robust | 11800 | 3e-5 | PLATEAUED — terrain ~5.0, reward 242.5, 92.8% survival. Best checkpoint ever but policy still crawling on flat ground. TerrainScaledVelocity + height (0.42m easy → 0.25m hard) broke past 4.1 ceiling but 0.25m hard target too low. | model_11800.pt |
| 11e | B | robust | 14000 | 3e-5 | REPLACED — height_hard 0.25→0.35, weight -1→-2. Height penalty still oscillating (-1.2 to -5.0) because curriculum-level-based conditioning is too indirect. Policy still knee-walking in teleop at iter 14000. | model_14000.pt |
| 11f | B | robust | 14100 | 3e-5 | FAILED — NaN. Introduced variance-based height conditioning (Option B). `ray_hits_w` returns `inf` for missed rays → height penalty NaN. Fixed with `torch.nan_to_num()`, but model_14100.pt CORRUPTED (17 tensors with NaN from gradient propagation). | model_14100.pt (CORRUPTED) |
| 11g | B | robust | ~0 | 3e-5 | FAILED — NaN cascade. Resumed from corrupted model_14100.pt → entire training NaN from start. Also bumped stumble -0.02→-1.0. Killed immediately. | — |
| **11h** | **B** | **robust** | **ongoing** | **3e-5** | **IN PROGRESS — Resumed from CLEAN model_14000.pt (11e). Variance-based height + `nan_to_num` + error clamped [0,1]. Stumble DISABLED (0.0). Bugs #28/#28b/#28c fixed. Early metrics (iter 31): reward 15.3, terrain 1.29, flip 83%, noise 0.56, value_loss 0.21 (stable, no NaN). Run dir: `2026-03-06_20-57-21`** | **TBD** |

**Key insight from B-easy attempts:** Three knobs matter: (1) LR must decrease aggressively for terrain transitions (3e-4→1e-4→5e-5→3e-5). (2) max_noise_std must decrease for later phases (1.0→0.7) to let the policy be precise on hard terrain. (3) Value loss watchdog (Bug #25) prevents oscillation cascades that LR reduction alone can't stop.

**Key insight from Trial 11 plateau:** The GaitReward (weight=10.0, std=0.1, 6 multiplicative sub-terms) is the primary bottleneck for terrain levels >4. It enforces strict diagonal trot gait, which becomes suboptimal on steep stairs and large obstacles where the robot needs dynamic corrections or non-trot strategies. Lowering gait weight to 3.0 and loosening tolerance (std=0.25, max_err=0.4) lets the policy discover terrain-adaptive gaits.

**Key insight from Trial 11c-v2 plateau:** A fixed height target (0.30m) teaches the robot to always crouch, even on flat ground (knee-walking problem). The fix: make height target terrain-scaled — stand tall (0.42m) on easy terrain, crouch (0.25m) on hard terrain. Similarly, terrain-scaled velocity commands prevent the robot from being asked to sprint on level 8 stairs.

**Key insight from Trial 11d plateau:** Terrain-scaled height with height_hard=0.25m still causes crawling because the majority of robots (at curriculum level ~5) get a target of ~0.33m — still a crouch. Plus 5,600 iters of "crouch is good" training before 11d means the asymmetric signal is too weak. Fix: raise height_hard from 0.25→0.35 (level 5 target becomes 0.381m — proper upright walking) and double weight from -1.0→-2.0 for stronger gradient signal.

**Key insight from Trial 11e:** Curriculum-level-based height conditioning is too indirect — it tells the robot "you're on row 5 of the curriculum, so crouch this much" but the policy can't observe curriculum level at eval time. The fix: condition height target on **height scan variance** instead. Low variance = flat ground = stand tall (0.42m). High variance = rough ground = crouch (0.35m). This is a direct per-step signal the policy can actually observe and learn from, and it works at eval time too.

**Key insight from Trial 11f/11g:** Variance-based height conditioning is correct in concept but `ray_hits_w` returns `inf` for missed rays, causing NaN in variance computation. The fix requires `torch.nan_to_num()` before computing variance. Also, `torch.square(height_error)` can produce enormous values (32.0+) when robots fall off terrain, causing gradient explosion. Must clamp height error to [0, 1] before squaring. AND: a single NaN-corrupted checkpoint (model_14100.pt with 17 NaN tensors) propagates NaN through the entire training from iter 0 — always verify checkpoints with `torch.isnan()` before resuming.

**Bug #28 (CommandTermCfg inheritance):** Custom command configs MUST inherit from `CommandTermCfg` (not standalone `@configclass`), and custom command classes must implement `_resample_command`, `_update_command`, `_update_metrics` — NOT override `reset`/`compute`/`_resample` directly.

**Bug #28b (Stumble penalty uses absolute Z):** `stumble_penalty` compares foot height against 0.15m in WORLD FRAME. On elevated terrain (stairs at Z=1.0m), every foot contact registers as a "stumble." Weight -1.0 caused 75% flip-over. DISABLED (weight=0.0).

**Bug #28c (Unbounded height error):** `torch.square(relative_height - target)` can produce enormous values (32.0+) when robots fall. Fix: `torch.clamp(height_error, 0.0, 1.0)` before squaring.

---

*"Don't throw the robot off a cliff. Walk it to the edge first. Then walk it to a shorter cliff."*
