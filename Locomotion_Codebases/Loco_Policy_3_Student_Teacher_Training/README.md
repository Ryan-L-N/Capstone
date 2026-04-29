# SIM_TO_REAL: 3-Master Expert Distillation Pipeline

Train 3 terrain-master Spot policies from proven checkpoints, each dominating
its terrain family with full sim-to-real hardening, then distill into a single
deployment-ready generalist that crushes all 4 eval environments.

**Target:** 49.5 m (5/5 zones) on Friction, Grass, Boulder, AND Stairs.

## Evolution: 6 Experts to 3 Masters

The original plan called for 6 from-scratch terrain specialists. In practice,
we discovered that **fine-tuning proven checkpoints with focused reward tuning
outperforms training from scratch**. Three masters -- each covering a terrain
family -- provide better coverage with less training time:

| Old (6 experts) | New (3 masters) | Why |
|-----------------|-----------------|-----|
| Friction, Slopes, Mixed Rough | **Flat Master** | Friction + grass are both flat-terrain stability problems. Grass drag is velocity scaling in eval, not physics force -- stability IS the grass solution. |
| Stairs Up, Stairs Down | **Stair Master** | Up and down share the same step-over kinematics. One expert handles both. |
| Boulders | **Boulder Master** | Boulders require unique high-clearance + slip-resistant gait distinct from stairs. |

---

## The 3 Masters

### Master 1: Flat Master (Friction + Grass)
- **Starting checkpoint:** distilled_6899.pt (100% friction completion, 28.2m grass)
- **Terrain:** 45% flat + 25% random rough + 15% wave + 15% gentle slopes
- **Philosophy:** Stability above all. Smoothest, most efficient gait possible.
- **Key rewards:** gait 15.0, action_smoothness -1.5, foot_slip -3.0, base_motion -3.0
- **Standing:** stand_still_scale 30.0, 30% standing envs
- **Training:** 2048 envs, 5000 iters, actor_only_resume + staged warmup

### Master 2: Stair Master (Stairs Up + Down)
- **Starting checkpoint:** model_6300.pt (obstacle parkour, terrain 5.79)
- **Terrain:** 90% stairs (45% up + 45% down) + 10% flat
- **Philosophy:** Controlled knee-bend stepping, straight-ahead traversal.
- **Key rewards:** gait 12.5, foot_clearance 4.5, base_roll -5.0, base_orientation -1.0
- **Standing:** stand_still_scale 20.0, 25% standing envs
- **Training:** 4096 envs, 5000 iters, actor_only_resume + staged warmup

### Master 3: Boulder Master (Obstacle Fields)
- **Starting checkpoint:** model_1400.pt (boulder expert, terrain 5.39, 22.6m eval)
- **Terrain:** 85% boulders (curriculum matching 4_env_test zones) + 15% flat
- **Philosophy:** High foot clearance, slip resistance on irregular polyhedra.
- **Key rewards:** foot_clearance 7.0, foot_slip -1.5, undesired_contacts -3.5
- **Standing:** stand_still_scale 20.0, 25% standing envs
- **Training:** 4096 envs, 5000 iters, actor_only_resume + staged warmup

---

## Parallel Training (2026-03-26)

All 3 masters train simultaneously on the H100:

| Master | Screen | VRAM | TensorBoard | Dashboard |
|--------|--------|------|-------------|-----------|
| Stair Master | s2r_parkour | ~16 GB | :6007 | :6008 |
| Boulder Master | s2r_boulder | ~16 GB | :6009 | :6010 |
| Flat Master | s2r_flat | ~8 GB | :6011 | :6012 |
| **Total** | | **~40 GB / 96 GB** | | |

### Staged Actor Warmup (all masters)
- Phase 1 (iter 0-300): All actor frozen, fresh critic calibrates
- Phase 2 (iter 300-500): Last actor layer unfreezes (128 to 12)
- Phase 3 (iter 500-700): Middle layer unfreezes (256 to 128)
- Phase 4 (iter 700+): Full unfreeze (all layers fine-tuning)

---

## Sim-to-Real Hardening

Every master trains with these mitigations from step 0:

| Feature | Value | Risk Addressed |
|---------|-------|---------------|
| Action delay | 40 ms (2 steps at 50 Hz) | Actuator latency |
| Observation delay | 20 ms (1 step at 50 Hz) | Sensor latency |
| Height scan dropout | 5% rays zeroed | Sensor dropout |
| IMU drift | Ornstein-Uhlenbeck process | Correlated noise |
| Observation noise | Per-channel Gaussian | Idealized sensors |
| Motor power penalty | -0.0005 weight | Energy efficiency |
| Torque limit penalty | -0.225 weight | Motor limits |
| External pushes | 3 N every 7-12 s | Disturbances |
| Mass randomization | 5 kg | Mass variation |
| Friction randomization | 0.3-1.0 static, 0.3-0.8 dynamic | Surface variation |

---

## Reward Tuning by Master

### Flat Master -- The Stability Stack
Strictest penalties of any expert. On ice, jerky = slip = fall.
- gait: 15.0 (perfect trot rhythm is survival)
- action_smoothness: -1.5 (smoothest of any expert)
- foot_slip: -3.0 (6x base -- every slip = death)
- base_roll: -5.0 (MAX -- lateral tilt = unrecoverable)
- base_pitch: -2.0 (strict -- nothing to climb)
- base_motion: -3.0 (no bobbing, no swaying)
- air_time_variance: -3.0 (symmetric gait timing)
- stand_still_scale: 30.0 (strongest standing stability)

### Stair Master -- Controlled Climbing
Moderate penalties, strong gait + orientation enforcement.
- gait: 12.5 (strong trot maintenance)
- foot_clearance: 4.5 (step over 3-25cm stair risers)
- base_roll: -5.0 (prevent diagonal walking)
- base_orientation: -1.0 (anti-angle enforcement)
- base_motion: -2.0 (prevent lateral drift)
- stand_still_scale: 20.0 (zero-cmd stability)

### Boulder Master -- High Clearance + Slip Resistance
Most permissive body orientation, highest clearance and contact penalties.
- foot_clearance: 7.0 (clear 25-35cm rocks)
- foot_slip: -1.5 (low-friction boulder surfaces)
- undesired_contacts: -3.5 (body hitting rocks = bad)
- action_smoothness: -0.8 (smooth prevents slip on irregular surface)
- base_roll: -5.0 (lateral tipover is #1 failure)
- stand_still_scale: 20.0 (zero-cmd stability)

---

## 4-Environment Eval Results

### Pre-Master Training (best checkpoints)
| Env | Best Policy | Progress | Zone |
|-----|-------------|----------|------|
| Friction | distilled_6899 | 49.0m (100/100) | 5/5 |
| Grass | distilled_6899 | 28.2m | 3/5 |
| Boulder | obstacle_44400 | 30.4m | 4/5 |
| Stairs | parkour_v3_5200 | 27.6m | 3/5 |

---

## Distillation (Next Phase)

After all 3 masters converge:
1. Evaluate each master on all 4 environments
2. Select best checkpoint per master
3. Distill 3 masters + distilled_6899 base into final student
4. Train at 20 Hz with balanced all-terrain curriculum
5. Evaluate on 4-env gauntlet + 5-ring composite

---

## Project Structure

```
SIM_TO_REAL/
  README.md                        This file
  PLAN.md                          Implementation plan
  RISK_MATRIX.md                   10-risk sim-to-real analysis
  configs/
    base_s2r_env_cfg.py            Base environment (all S2R hardening)
    terrain_cfgs.py                All terrain configurations
    expert_flat_master_cfg.py      Flat Master config
    expert_obstacle_parkour_cfg.py Stair Master config
    expert_boulder_master_cfg.py   Boulder Master config
  control_panel/
    dashboard.py                   Web dashboard (per-expert named)
    cli.py                         Command-line weight control
    hot_reload.py                  YAML-based live parameter updates
    guardrails.py                  Safety bounds for weight changes
  wrappers/                        S2R wrappers (delay, noise, dropout)
  scripts/
    train_expert.py                Unified training (all expert types)
  deploy/
    DEPLOYMENT_CHECKLIST.md        5-stage testing protocol
  checkpoints/                     Base checkpoints for fine-tuning
  logs/                            TensorBoard + training logs
```

---

*AI2C Tech Capstone -- MS for Autonomy, March 2026*
*3-Master Expert Distillation for sim-to-real quadruped locomotion*
