# SIM_TO_REAL Implementation Plan (Revised: 3-Master Strategy)

## Why This Exists

Our Spot RL training excels on flat/friction terrain (distilled_6899: 100%
friction completion) but struggles on boulders (30.4m, 4/5) and stairs (27.6m,
3/5). A third-party evaluation before our CMU PhD meeting identified 10
sim-to-real risks, all now mitigated.

This pipeline trains 3 terrain masters from proven checkpoints, with all S2R
mitigations baked in, then distills into a single deployment-ready generalist.

---

## Why 3 Masters Instead of 6 Experts

### Original Plan (6 experts)
Train 6 specialists from scratch: friction, stairs_up, stairs_down, boulders,
slopes, mixed_rough. Each gets 80% specialty terrain + 20% flat.

### What We Learned
1. **Fine-tuning beats from-scratch.** Model_1400 (fine-tuned from distilled_6899
   for boulders) reached terrain 5.39 in 1400 iters. Training from scratch would
   take 5000+ iters to reach the same level.
2. **Terrain families share kinematics.** Stairs up and stairs down use the same
   step-over motion. Friction and grass both need stability (grass drag is velocity
   command scaling, not physics force).
3. **Parallel training is efficient.** 3 masters fit in ~40GB on the H100 (96GB total),
   training simultaneously with separate dashboards for live tuning.

### The 3 Masters

| Master | Covers | Starting Point | Terrain |
|--------|--------|----------------|---------|
| **Flat Master** | Friction + Grass | distilled_6899 (100% friction) | 45% flat, 25% rough, 15% wave, 15% slopes |
| **Stair Master** | Stairs Up + Down | model_6300 (parkour, terrain 5.79) | 90% stairs (up+down), 10% flat |
| **Boulder Master** | Obstacle Fields | model_1400 (boulders, 22.6m eval) | 85% boulders (curriculum), 15% flat |

---

## Reward Philosophy

Each master has a distinct reward personality tuned for its terrain family:

### Flat Master: Restraint and Precision
On low-friction surfaces, the failure cascade is: jerky motion -> foot slip ->
lateral tilt -> fall. On grass (velocity-scaled commands), inefficient gait
means no progress. The solution is the same: the smoothest, most stable gait.

Key insight: **gait 15.0** (highest of any expert) + **action_smoothness -1.5**
(strictest) + **foot_slip -3.0** (6x base). The policy learns micro-adjustments
to maintain traction rather than aggressive locomotion.

### Stair Master: Controlled Climbing
Stairs need moderate foot clearance, strict body orientation (no diagonal
walking), and knee-bend stepping (not goosestepping). The pitch/roll split
(separate base_pitch and base_roll penalties) gives fine-grained control.

Key insight: **gait 12.5** + **foot_clearance 4.5** + **base_orientation -1.0**
enforces straight-ahead climbing with natural knee bend.

### Boulder Master: Step Over Everything
Boulders demand the highest foot clearance and contact avoidance. The irregular
polyhedral shapes (D8/D10/D12/D20 in eval) create unpredictable contact angles
with lower effective friction (multiply combine mode: 0.32-0.8).

Key insight: **foot_clearance 7.0** (highest) + **foot_slip -1.5** +
**undesired_contacts -3.5** creates aggressive stepping with body protection.

---

## Training Configuration

### Common to All Masters
- Network: [512, 256, 128] MLP, ELU activations
- Resume: actor_only_resume (fresh critic, staged actor warmup)
- Warmup: 300 critic-only, then layer-by-layer actor unfreeze (300/500/700)
- LR: adaptive KL, bounds 1e-6 to 3e-5
- Noise: max 0.5, min 0.3
- Save: every 100 iterations
- S2R: full hardening (delay, noise, dropout, DR, pushes)
- Emergency stop: value loss > 100K (raised for warmup compatibility)
- Standing stability: stand_still_scale 20-30, 25-30% standing envs

### Per-Master
| Param | Flat | Stair | Boulder |
|-------|------|-------|---------|
| num_envs | 2048 | 4096 | 4096 |
| max_iterations | 5000 | 5000 | 5000 |
| VRAM | ~8 GB | ~16 GB | ~16 GB |
| TensorBoard | :6011 | :6007 | :6009 |
| Dashboard | :6012 | :6008 | :6010 |

---

## Live Control Panel

Each master has a web dashboard for real-time reward weight tuning:
- **CLI:** `python -m control_panel.cli set foot_clearance 5.0`
- **Web:** Slider interface at dashboard port
- **Guardrails:** Sign preservation, absolute bounds, 50% max delta
- **Audit:** All changes logged to JSONL with timestamps
- **Expert name:** Displayed in dashboard header (e.g. "Training Control Panel -- Boulder Master")

---

## Distillation Plan (After Masters Converge)

1. Evaluate each master on all 4 environments (100 episodes each)
2. Select best checkpoint per master based on target-terrain score
3. Create 3-source distillation:
   - Router: 235 -> 64 -> 3 (softmax) -- simpler than 6-expert router
   - Student: [512, 256, 128] MLP
   - Alpha annealing: 0.8 -> 0.2 (expert-heavy to PPO-heavy)
4. Train at 20 Hz (decimation=25) on DISTILLATION_TERRAINS_CFG
5. Evaluate on 4-env gauntlet + 5-ring composite gauntlet

---

## Success Criteria

| Metric | Target | Pre-Master Best | Gap |
|--------|--------|-----------------|-----|
| Friction | 49.5m (5/5) | 49.0m (5/5) | 0.5m |
| Grass | 49.5m (5/5) | 28.2m (3/5) | 21.3m |
| Boulder | 49.5m (5/5) | 30.4m (4/5) | 19.1m |
| Stairs | 49.5m (5/5) | 27.6m (3/5) | 21.9m |
| Composite gauntlet | 600/600 | ~200/600 | ~400 |
| Flip rate | 0% | <10% | Improve |

---

## Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| 3 master configs + terrain | 2 hours | DONE |
| Parallel master training | ~8 hours | IN PROGRESS |
| Master evaluation | 1 hour | Pending |
| Distillation training | ~8 hours | Pending |
| Final evaluation | 1 hour | Pending |
| **Total** | **~20 hours** | |

---

*AI2C Tech Capstone -- MS for Autonomy, March 2026*
