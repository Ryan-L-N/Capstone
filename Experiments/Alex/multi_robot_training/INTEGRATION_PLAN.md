# Integration Plan: Robust RL Paper into Multi-Robot Terrain Training

> **Paper:** "Reinforcement Learning for Robust and Adaptive Quadruped Locomotion" — Ashutosh Gupta (Dec 2025, Oregon State University)
>
> **Robots:** Boston Dynamics Spot (Spirit 40) + Ghost Robotics Vision60
>
> **Hardware:** NVIDIA H100 NVL 96GB
>
> **Date:** February 2026

---

## Executive Summary

This document outlines the unified multi-robot training system that integrates the key techniques from the Robust RL thesis into a **shared infrastructure** for both Spot and Vision60. The paper's 100-hour config demonstrates state-of-the-art blind quadruped locomotion using 12 terrain types, 19 reward terms, cosine LR annealing, and aggressive domain randomization — but it only targets Spot.

**Goals:**
1. **Share** terrain curriculum, reward functions, and training utilities between both robots
2. **Train separate policies** per robot using direct PPO (Phase 1), then student-teacher distillation (Phase 2)
3. **Integrate Weights & Biases** for experiment tracking and cross-robot comparison
4. **Extend** the 4-arena evaluation pipeline to both robots

The existing `100hr_env_run/`, `vision60_training/`, and `hybrid_ST_RL/` pipelines remain untouched as historical reference. All new work goes into `multi_robot_training/`.

---

## Table of Contents

1. [Paper Techniques Being Integrated](#1-paper-techniques-being-integrated)
2. [Architecture & Directory Layout](#2-architecture--directory-layout)
3. [What Is Shared vs Robot-Specific](#3-what-is-shared-vs-robot-specific)
4. [Step-by-Step Implementation](#4-step-by-step-implementation)
5. [Weights & Biases Integration](#5-weights--biases-integration)
6. [Training Commands](#6-training-commands)
7. [Evaluation Commands](#7-evaluation-commands)
8. [Reward Term Reference](#8-reward-term-reference)
9. [Reward Conflict Analysis](#9-reward-conflict-analysis)
10. [Vision60 Weight Adjustments](#10-vision60-weight-adjustments)
11. [Verification Plan](#11-verification-plan)
12. [Known Gotchas](#12-known-gotchas)

---

## 1. Paper Techniques Being Integrated

### 1.1 Terrain Curriculum (12 Types)

| # | Terrain Type | Proportion | Category |
|---|-------------|-----------|----------|
| 1 | Pyramid stairs up | 10% | Geometric |
| 2 | Pyramid stairs down | 10% | Geometric |
| 3 | Random grid boxes | 10% | Geometric |
| 4 | Stepping stones | 5% | Geometric |
| 5 | Gaps | 5% | Geometric |
| 6 | Random rough | 10% | Surface |
| 7 | Pyramid slope up | 7.5% | Surface |
| 8 | Pyramid slope down | 7.5% | Surface |
| 9 | Wave terrain | 5% | Surface |
| 10 | Friction plane | 5% | Surface |
| 11 | Vegetation plane | 5% | Surface |
| 12 | HF stairs up | 10% | Compound |
| 13 | Discrete obstacles | 5% | Compound |
| 14 | Repeated boxes | 5% | Compound |

**Grid:** 10 rows (difficulty 0-9) × 40 columns (variety) = 400 patches, 8m × 8m each.

### 1.2 Reward Terms (19 Total)

**7 Positive (task) rewards:**
- `base_linear_velocity` (7.0) — track commanded forward/lateral speed
- `gait` (10.0) — enforce trot gait timing
- `base_angular_velocity` (5.0) — track commanded yaw rate
- `foot_clearance` (3.5) — encourage swing foot lift
- `air_time` (3.0 Spot / 2.0 V60) — encourage proper gait timing
- `velocity_modulation` (2.0) — terrain-adaptive speed tracking
- `vegetation_drag` (-0.001) — physics modifier + small penalty

**12 Negative (penalty) rewards:**
- `base_orientation` (-5.0) — penalize tilt
- `base_motion` (-4.0 / -3.0) — penalize body bouncing
- `foot_slip` (-3.0 / -2.0) — penalize foot sliding during contact
- `action_smoothness` (-2.0) — penalize jerky actions
- `body_height_tracking` (-2.0) — keep nominal height
- `stumble` (-2.0) — penalize hitting obstacles at knee height
- `air_time_variance` (-1.0) — symmetric gait enforcement
- `joint_pos` (-1.0) — penalize deviation from default
- `contact_force_smoothness` (-0.5) — gentle foot placement
- `joint_vel` (-0.05 / -0.03) — smooth joint movement
- `joint_torques` (-0.002) — energy efficiency
- `joint_acc` (-0.0005) — smooth acceleration

### 1.3 Cosine LR Annealing

```
LR(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * progress))
```

With linear warmup: `lr_min → lr_max` over `warmup_iters`, then cosine decay.
Default: 1e-3 → 1e-5 over 60K iterations with 3K warmup.

### 1.4 Vegetation Drag Physics

Custom physics modifier + reward term that applies velocity-dependent drag forces to feet:
- **Terrain-aware:** No drag on friction_plane, always drag on vegetation_plane, randomized on other terrains
- **Tiered sampling:** 25% clean, 25% light (0.5-5.0), 25% medium (5.0-12.0), 25% heavy (12.0-20.0)
- **Formula:** F_drag = -c_drag × v_foot (horizontal only, contact-gated)

### 1.5 Domain Randomization

| Parameter | Spot (Fixed) | Vision60 (Progressive) |
|-----------|-------------|----------------------|
| Friction | [0.05, 1.5] | [0.3, 1.3] → [0.1, 1.5] |
| Mass offset | ±8 kg | ±5 → ±7 kg |
| Push velocity | ±1.5 m/s | ±0.5 → ±1.0 m/s |
| External force | ±8 N | ±3 → ±6 N |
| External torque | ±3 Nm | ±1 → ±2.5 Nm |

---

## 2. Architecture & Directory Layout

```
multi_robot_training/
    __init__.py
    INTEGRATION_PLAN.md              ← This document

    shared/                          ← Robot-agnostic modules
        __init__.py
        terrain_cfg.py               # ROBUST_TERRAINS_CFG (12 types, 400 patches)
        scratch_terrain_cfg.py       # SCRATCH_TERRAINS_CFG (7 types, for warmup)
        reward_terms.py              # 5 custom reward functions
        lr_schedule.py               # Cosine LR annealing utilities
        dr_schedule.py               # Progressive domain randomization
        training_utils.py            # Noise clamping, TF32, common args

    configs/                         ← Per-robot configurations
        __init__.py
        robot_params.py              # RobotParams dataclass
        spot_params.py               # SPOT_PARAMS instance
        vision60_params.py           # V60_PARAMS instance
        spot_ppo_env_cfg.py          # Spot Phase 1 env (12 terrains, 19 rewards)
        vision60_ppo_env_cfg.py      # Vision60 Phase 1 env (UPGRADED)
        spot_ppo_cfg.py              # Spot PPO runner (W&B enabled)
        vision60_ppo_cfg.py          # Vision60 PPO runner (W&B enabled)
        spot_teacher_env_cfg.py      # Spot Phase 2 teacher (+privileged dims)
        vision60_teacher_env_cfg.py  # Vision60 Phase 2 teacher
        distill_ppo_cfg.py           # Shared distillation PPO config

    train_ppo.py                     # Phase 1: --robot spot|vision60
    train_teacher.py                 # Phase 2a: Teacher with privileged obs
    train_distill.py                 # Phase 2b: Student distillation
    play.py                          # Inference / visualization

    eval/                            # Evaluation extension
        __init__.py
        vision60_rough_terrain_policy.py  # Vision60 deployment wrapper
```

---

## 3. What Is Shared vs Robot-Specific

### Shared (robot-agnostic)

| Module | Why Shared |
|--------|-----------|
| `terrain_cfg.py` | Pure heightfield geometry — no robot reference |
| `reward_terms.py` | Uses `SceneEntityCfg` params for body names |
| `lr_schedule.py` | Pure math (cosine + linear warmup) |
| `dr_schedule.py` | Operates on env config event terms generically |
| `training_utils.py` | TF32 config, noise clamping, argparser |

### Robot-Specific

| Parameter | Spot | Vision60 |
|-----------|------|----------|
| Foot body names | `".*_foot"` | `"lower.*"` |
| Gait pairs | `(fl_foot+hr_foot, fr_foot+hl_foot)` | `(lower0+lower3, lower2+lower1)` |
| Termination bodies | `["body", ".*leg"]` | `["body"]` |
| Body height | 0.42m | 0.55m |
| Foot clearance | 0.10m | 0.08m |
| Stumble knee height | 0.15m | 0.20m |
| PD gains (Kp/Kd) | 60/1.5 | 80/2.0 |
| Spawn height | 0.5m | 0.6m |
| Joint reset function | `spot_mdp.reset_joints_around_default` | `mdp.reset_joints_by_offset` |
| DR mode | Fixed (aggressive) | Progressive (mild → hard) |

---

## 4. Step-by-Step Implementation

### Phase 1: Direct PPO Training

1. `train_ppo.py --robot spot` — Full DR, 12 terrains, 19 rewards, cosine LR
2. `train_ppo.py --robot vision60` — Progressive DR, same terrains/rewards

Both produce checkpoints in `logs/rsl_rl/{robot}_robust_ppo/`.

### Phase 2a: Teacher Training

1. `train_teacher.py --robot spot --checkpoint <phase1_best>`
2. Weight surgery extends 235→N dim input layer (N determined at runtime)
3. Teacher trains with privileged friction + contact force observations

### Phase 2b: Student Distillation

1. `train_distill.py --robot spot --student_checkpoint <phase1> --teacher_checkpoint <phase2a>`
2. BC coefficient anneals: 0.8 → 0.2 over training
3. Student learns teacher's behavior using only standard 235-dim obs

---

## 5. Weights & Biases Integration

RSL-RL's `OnPolicyRunner` natively supports W&B:

```python
@configclass
class SpotPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    logger = "wandb"                         # was "tensorboard"
    wandb_project = "capstone-quadruped-rl"  # W&B project name
    experiment_name = "spot_robust_ppo"      # W&B run group
```

### Setup on H100

```bash
pip install wandb
wandb login  # paste API key
```

### Custom Metrics Logged

The training script logs additional custom metrics every 100 iterations:
- `lr` — current learning rate
- `dr/fraction` — DR expansion progress (Vision60 only)
- `dr/push_vel` — current push velocity range
- `dr/ext_force` — current external force range
- `distill/bc_coef` — behavior cloning coefficient (Phase 2b only)

### W&B Advantages over TensorBoard

- Side-by-side Spot vs Vision60 comparison dashboards
- Hyperparameter sweep tracking
- Team sharing without SSH tunnel to H100
- Automatic GPU metrics (temp, VRAM, utilization)
- Model artifact versioning for checkpoint management

---

## 6. Training Commands

### Phase 1 — Spot (H100)

```bash
cd ~/IsaacLab
./isaaclab.sh -p ~/multi_robot_training/train_ppo.py --headless \
    --robot spot --num_envs 20480 --max_iterations 60000 \
    --lr_max 1e-3 --lr_min 1e-5 --warmup_iters 3000
```

### Phase 1 — Vision60 (H100)

```bash
cd ~/IsaacLab
./isaaclab.sh -p ~/multi_robot_training/train_ppo.py --headless \
    --robot vision60 --num_envs 20480 --max_iterations 60000 \
    --lr_max 1e-3 --lr_min 1e-5 --warmup_iters 3000 \
    --dr_expansion_iters 15000
```

### Phase 1 — Local Debug

```bash
isaaclab.bat -p /path/to/train_ppo.py --headless \
    --robot spot --num_envs 64 --max_iterations 10 --no_wandb
```

### Phase 2a — Teacher

```bash
./isaaclab.sh -p ~/multi_robot_training/train_teacher.py --headless \
    --robot spot --checkpoint ~/logs/rsl_rl/spot_robust_ppo/.../model_best.pt \
    --num_envs 8192 --max_iterations 10000
```

### Phase 2b — Distillation

```bash
./isaaclab.sh -p ~/multi_robot_training/train_distill.py --headless \
    --robot spot \
    --student_checkpoint ~/logs/rsl_rl/spot_robust_ppo/.../model_best.pt \
    --teacher_checkpoint ~/logs/rsl_rl/spot_teacher/.../model_best.pt \
    --num_envs 8192 --max_iterations 10000
```

### Inference / Visualization

```bash
./isaaclab.sh -p ~/multi_robot_training/play.py \
    --robot spot --checkpoint ~/logs/rsl_rl/spot_robust_ppo/.../model_best.pt \
    --num_envs 50
```

---

## 7. Evaluation Commands

The 4-arena evaluation pipeline (`4_env_test/`) can be extended by:

1. Adding `--robot spot|vision60` argument to `run_capstone_eval.py`
2. Loading `Vision60RoughTerrainPolicy` from `multi_robot_training/eval/`
3. Using Vision60-specific PD gains (80/2.0) and default joints

```bash
# Spot evaluation (existing)
./isaaclab.sh -p src/run_capstone_eval.py --headless \
    --env stairs --policy rough --num_episodes 1000

# Vision60 evaluation (after integration)
./isaaclab.sh -p src/run_capstone_eval.py --headless \
    --robot vision60 --env stairs --policy rough --num_episodes 1000
```

---

## 8. Reward Term Reference

| # | Term | Type | Weight (Spot) | Weight (V60) | Source |
|---|------|------|--------------|-------------|--------|
| 1 | `base_linear_velocity` | + | 7.0 | 7.0 | spot_mdp |
| 2 | `gait` | + | 10.0 | 10.0 | spot_mdp (GaitReward) |
| 3 | `base_angular_velocity` | + | 5.0 | 5.0 | spot_mdp |
| 4 | `foot_clearance` | + | 3.5 | 3.5 | spot_mdp |
| 5 | `air_time` | + | 3.0 | **2.0** | spot_mdp |
| 6 | `velocity_modulation` | + | 2.0 | 2.0 | shared/reward_terms |
| 7 | `vegetation_drag` | - | -0.001 | -0.001 | shared/reward_terms |
| 8 | `base_orientation` | - | -5.0 | -5.0 | spot_mdp |
| 9 | `base_motion` | - | -4.0 | **-3.0** | spot_mdp |
| 10 | `foot_slip` | - | -3.0 | **-2.0** | spot_mdp |
| 11 | `action_smoothness` | - | -2.0 | -2.0 | spot_mdp |
| 12 | `body_height_tracking` | - | -2.0 | -2.0 | shared/reward_terms |
| 13 | `stumble` | - | -2.0 | -2.0 | shared/reward_terms |
| 14 | `air_time_variance` | - | -1.0 | -1.0 | spot_mdp |
| 15 | `joint_pos` | - | -1.0 | -1.0 | spot_mdp |
| 16 | `contact_force_smoothness` | - | -0.5 | -0.5 | shared/reward_terms |
| 17 | `joint_vel` | - | -0.05 | **-0.03** | spot_mdp |
| 18 | `joint_torques` | - | -0.002 | -0.002 | spot_mdp |
| 19 | `joint_acc` | - | -0.0005 | -0.0005 | spot_mdp |

**Bold** = adjusted for Vision60.

---

## 9. Reward Conflict Analysis

### 7 Potential Conflicts Identified

1. **Gait vs Air Time (MODERATE):** Keep 10.0/3.0 ratio. Reduce air_time to 2.0 for Vision60.
2. **Foot Slip vs Linear Velocity (HIGH):** velocity_modulation (+2.0) bridges this gap. Start V60 at slip=-2.0.
3. **Foot Clearance vs Base Motion (LOW):** Well-balanced via tanh velocity gate.
4. **Base Orientation vs Height Tracking (LOW):** Orientation dominates as intended.
5. **Action Smoothness vs Stumble (MODERATE):** Reduce smoothness to -1.0 if policy too cautious.
6. **Contact Force Smoothness vs Clearance (LOW):** Well-designed complementary pair.
7. **Vegetation Drag vs Motion Rewards (SPECIAL):** Physics IS the signal, -0.001 weight is just a nudge.

---

## 10. Vision60 Weight Adjustments

Vision60 is lighter (13.6 kg vs ~32 kg for Spot) but has different dynamics:

| Term | Spot | V60 | Rationale |
|------|------|-----|-----------|
| `air_time` | 3.0 | 2.0 | Lighter → easier airtime, reduce bouncing |
| `foot_slip` | -3.0 | -2.0 | Different traction characteristics |
| `base_motion` | -4.0 | -3.0 | Different inertial properties |
| `joint_vel` | -0.05 | -0.03 | Different joint dynamics |

### Tuning Protocol (4 phases)

1. **Sanity (iters 0-1K):** Only velocity tracking + gait — verify basic locomotion
2. **Core locomotion (iters 1K-5K):** Add all 14 core terms
3. **Robustness (iters 5K-15K):** Enable remaining 5 terms (velocity_modulation, vegetation_drag, body_height, contact_force_smoothness, stumble)
4. **Advanced (iters 15K+):** Full DR expansion, all rewards at target weights

Note: The current implementation enables all 19 terms from the start. Use the tuning protocol if training instability is observed.

---

## 11. Verification Plan

| # | Test | Criteria |
|---|------|---------|
| 1 | Local smoke test (64 envs, 10 iters) | Both `--robot spot` and `--robot vision60` launch without errors |
| 2 | Observation check | Both produce 235-dim observations |
| 3 | Terrain check | 400 patches render, all 12 types visible |
| 4 | Reward check | All 19 terms produce non-zero values after 10 iters |
| 5 | W&B check | Runs appear in `capstone-quadruped-rl` project |
| 6 | H100 scale test (20K envs, 100 iters) | No OOM, collision buffer sized correctly |
| 7 | Phase 2 teacher | Privileged obs layer extends correctly |
| 8 | Phase 2 distill | BC loss decreases over training |
| 9 | Evaluation | Both robots complete 4-arena gauntlet |
| 10 | Reward balance | No single term > 50% of total after 1000 iters |

---

## 12. Known Gotchas

1. **SimulationApp import order:** Create `AppLauncher` BEFORE importing omni/isaaclab modules
2. **Never call `simulation_app.close()`** — use `os._exit(0)` to avoid CUDA deadlock
3. **Height scan fill = 0.0** for flat terrain (NOT 1.0 — causes policy collapse)
4. **Quaternion format:** [w, x, y, z] scalar-first throughout
5. **DOF ordering:** Type-grouped (all hx, all hy, all kn), not leg-grouped
6. **Friction combine mode:** "multiply" in all environments
7. **Vision60 URDF path:** Must be at `~/vision60_training/urdf/vision60_v5.urdf` on H100
8. **GPU PhysX buffers:** Set `gpu_collision_stack_size = 2**31` for 20K+ envs with 12 terrain types
9. **Progressive DR:** Only for Vision60 — Spot uses fixed aggressive DR from the start
10. **W&B requires `pip install wandb` and `wandb login`** on H100 (one-time setup)
11. **Obs normalization:** Enabled for Vision60, disabled for Spot (matching successful training configs)
12. **Joint reset function:** Spot uses `spot_mdp.reset_joints_around_default`, Vision60 uses generic `mdp.reset_joints_by_offset`

---

*Created for AI2C Tech Capstone — MS for Autonomy, February 2026*
