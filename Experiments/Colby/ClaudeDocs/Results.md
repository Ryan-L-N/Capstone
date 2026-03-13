# Training Results — Spot RL (MS for Autonomy)

*Colby Johnson — AI2C Tech Capstone, Carnegie Mellon University*
*Each run gets one card. Update in-place when a run completes or is retired.*

---

## How to Use This Doc

- **One card per run** — fill in as training progresses, finalize when done
- **Summary table** at top always reflects the current best results across all runs
- **Failures get their own section** — what broke and what we learned
- **PPT/report ready** — numbers here go directly into slides; no digging through logs
- **TensorBoard** — `http://172.24.254.24:6006` — use this for live visual monitoring during runs (reward curves, terrain level, gait quality). Screenshots from here make great slides.

---

## Summary Table (Best Results to Date)

| Run | Goal | Key Metric | Best Value | Status |
|-----|------|-----------|-----------|--------|
| H100-Rough-30K | Rough terrain locomotion | Terrain Level | 4.42 | ✅ Complete |
| MH-1 | Mason Hybrid + AI Coach | Terrain Level | — | ❌ Failed (gait destroyed) |
| MH-2 | Mason Hybrid + VLM Coach | Terrain Level | In progress | 🔄 Active |

---

## Run Cards

---

### RUN-001 — H100 Rough Terrain (30K Iterations)

**Date:** February 2026
**Hardware:** H100 NVL 96GB
**Goal:** Train Spot to walk on rough terrain (stairs, rubble, uneven ground)
**Config:** 8,192 parallel envs, RSL-RL PPO, 235-dim obs, 12-dim action

#### Final Metrics
| Metric | Start | Final | Change |
|--------|------:|------:|--------|
| Mean Reward | -0.90 | +143.74 | **+144.64** |
| Episode Length | 20 steps | 573 steps | **+28.6×** |
| Gait Quality | 0.06 | 5.28 | **+88×** |
| Terrain Level | 3.18 | 4.42 | **+1.24** |
| Flip-Over Rate | ~50% | <1% | — |
| Time-Out Rate | ~1% | >99% | — |

**Total training:** ~53 hours, 5.9B timesteps, 30,000 iterations
**Checkpoint:** `C:\IsaacLab\logs\rsl_rl\spot_rough\48h_run\model_29999.pt`

#### Outcome
✅ Successfully deployed in 100m obstacle course with WASD + Xbox teleop.
Spot walks, trots, climbs 0.75m stairs, traverses rubble pool, handles 120 dynamic bricks.

#### Key Takeaway
> Height scan fill value must be **0.0** (not 1.0) for flat ground deployment.
> Using 1.0 produces action norm 7.42 → robot collapses in 1.5 seconds.

---

### RUN-002 — Mason Hybrid MH-1

**Date:** March 10, 2026
**Hardware:** H100 NVL 96GB
**Goal:** Break through terrain level 4.83 ceiling using Mason's 11-term config + AI Coach
**Config:** [512,256,128] network (800K params), adaptive KL LR, AI Coach active

#### What Happened
Training started well. AI Coach intervened and progressively boosted the velocity
tracking reward in an attempt to improve terrain traversal:

| Iteration | Velocity Reward Weight | Gait Quality |
|-----------|----------------------|-------------|
| 0 | 5.0 (baseline) | Normal |
| ~500 | 9.2 (coach adjusted) | Degrading |
| ~1000 | 14.26 (coach adjusted) | **Destroyed** |

Positive feedback loop: higher velocity reward → robot tries to move faster →
falls more → coach sees low reward → boosts velocity again. Gait collapsed entirely.

#### Outcome
❌ **FAILED.** Run retired. Terrain level never surpassed 4.83 ceiling.

#### Key Takeaway
> AI Coach must have strict velocity bounds (3.0–7.0 max).
> Coach should run in **deferred/silent mode** for 300+ iterations before intervening.
> LR changes should be **disabled** for the coach entirely.

---

### RUN-003 — Mason Hybrid MH-2

**Date:** March 11, 2026 — 🔄 **ACTIVE**
**Hardware:** H100 NVL 96GB
**Goal:** Mason Hybrid with gait-quality-first coach activation + VLM visual feedback
**Config:** Same as MH-1 but coach in deferred mode, tighter bounds, VLM enabled

#### Changes from MH-1
- Coach silent for first 300+ iterations (no interventions)
- Velocity reward bounds tightened: 3.0–7.0 (was 1.0–15.0)
- LR changes disabled from coach
- VLM frame averaging added (5-frame buffer from envs 0, 10, 50)

#### Metrics (In Progress)
| Metric | Value | Iteration |
|--------|-------|-----------|
| Terrain Level | — | — |
| Gait Quality | — | — |
| Mean Reward | — | — |

*Update this card when MH-2 completes.*

---

## Failures & What We Learned

| Run | Failure Mode | Root Cause | Fix Applied |
|-----|-------------|-----------|------------|
| MH-1 | Gait destroyed mid-training | AI Coach positive feedback loop on velocity reward | Tighter coach bounds, deferred mode, disable LR changes |
| Trial 10 (Alex) | action_smoothness → -103 trillion | Skipped Phase A.5, went directly to full robust terrain | Never skip A.5; always use 4-phase curriculum |
| 5K model | Robot barely walked | Undertrained (only 5K iters, laptop GPU) | Full H100 run (30K iters, 5.9B timesteps) |
| Early deployment | Robot collapsed in 1.5s | Height scan fill = 1.0 instead of 0.0 | Fixed fill value to 0.0 |
| Phase B-easy (lr=3e-4) | value_loss → 4,670+ at iter ~25 | LR too high for rough terrain phase | Use lr_max=5e-5 for Phase B+ |

---

## Benchmark Targets (ARL Delivery)

Targets from the project plan. Fill in actual results as runs complete.

| Scenario | Target Success Rate | Target Time | Actual | Status |
|----------|--------------------|-----------:|--------|--------|
| Sparse room (10 objects) | >95% | <30s | — | ⏳ |
| Dense room (40 objects) | >85% | <60s | — | ⏳ |
| Two-room + doorway | >80% | <90s | — | ⏳ |
| Stairs (2 floors) | >75% | <120s | — | ⏳ |
| Full obstacle course (100m) | Completion | — | ✅ Working | ✅ |

---

## Instructions for Claude (Results Maintenance)

When updating this file after a new training run:

1. **Add a new Run Card** using the template below — one card per run, never delete old ones
2. **Update the Summary Table** at the top with the new run's best metric
3. **If the run failed**, add a row to the Failures table with root cause + fix
4. **Finalize the card** when a run is retired (mark ✅ Complete or ❌ Failed)
5. Do NOT copy raw TensorBoard log dumps here — extract only the meaningful numbers
6. Numbers should be copy-paste ready for PowerPoint slides

### New Run Card Template
```
### RUN-00X — [Short Name]

**Date:** [Month Year]
**Hardware:** [GPU]
**Goal:** [One sentence]
**Config:** [num_envs, key hyperparams]

#### Final Metrics
| Metric | Start | Final | Change |
|--------|------:|------:|--------|
| Mean Reward | | | |
| Episode Length | | | |
| Terrain Level | | | |
| Gait Quality | | | |

**Total training:** [hours], [timesteps], [iterations]
**Checkpoint:** [path]

#### Outcome
[✅ / ❌] [What worked or what broke, 2-3 sentences max]

#### Key Takeaway
> [The single most important thing this run taught us]
```
