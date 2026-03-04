# Spot Training Curriculum: Flat → Transition → Robust Easy → Robust

> **Proven recipe for teaching Spot to walk on rough terrain.**
> Each phase resumes from the previous phase's best checkpoint.
> Run phases individually — stop, verify, then proceed.
>
> **Hardware:** NVIDIA H100 NVL 96GB
> **Last updated:** March 3, 2026

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

| Reward Term | Phase A | Phase A.5 | Phase B-easy | Phase B |
|-------------|---------|-----------|-------------|---------|
| undesired_contacts | -5.0 | -1.5 | -1.5 | -1.5 |
| body_scraping | — | -2.0 (new) | -2.0 | -2.0 |
| All other terms | unchanged | unchanged | unchanged | unchanged |

The `undesired_contacts` and `body_scraping` changes are baked into `spot_ppo_env_cfg.py` — they apply to all phases. On flat terrain they're negligible (body rarely contacts ground). On rough terrain they prevent belly-dragging without over-punishing legitimate bumps.

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
| terrain_levels stuck at 0-1 | Robot can't advance curriculum | Check if too many terrain types are novel |
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
| **10h** | **B-easy** | **robust_easy** | **30K** | **5e-5** | **IN PROGRESS — iter 1608, reward 155, 75% survival** | **model_1600.pt** |
| 11 | B | robust | 30K | TBD | PLANNED — after 10h | TBD |

**Key insight from B-easy attempts:** LR must decrease aggressively for terrain transitions. `lr_max=3e-4` → instant crash. `lr_max=1e-4` → crash at iter ~1134-1319. `lr_max=5e-5` → stable (Trial 10h at iter 1608+). The NaN sanitizer (Bug #24) prevents std crashes but can't save a policy corrupted by high LR.

---

*"Don't throw the robot off a cliff. Walk it to the edge first. Then walk it to a shorter cliff."*
