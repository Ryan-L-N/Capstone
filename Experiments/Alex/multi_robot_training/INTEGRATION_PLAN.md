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
13. [Training Trial Log](#13-training-trial-log)
14. [Lessons Learned from Trials](#14-lessons-learned-from-trials)

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

### 1.2 Reward Terms (19 Total) — Paper-Matched Coefficients

> **Note:** These weights were updated on Feb 28, 2026 to match the paper's exact values. Our original hand-tuned weights were 2-6x too harsh and caused training collapse (see Section 13, Trial 1).

**7 Positive (task) rewards:**
- `base_linear_velocity` (5.0) — track commanded forward/lateral speed
- `gait` (5.0) — enforce trot gait timing (mode_time=0.2, vel_thresh=0.25)
- `base_angular_velocity` (5.0) — track commanded yaw rate
- `foot_clearance` (0.75) — encourage swing foot lift (target=0.125m)
- `air_time` (5.0 Spot / 3.0 V60) — encourage proper gait timing
- `velocity_modulation` (2.0) — terrain-adaptive speed tracking
- `vegetation_drag` (-0.001) — physics modifier + small penalty

**12 Negative (penalty) rewards:**
- `base_orientation` (-3.0) — penalize tilt
- `base_motion` (-2.0 / -1.5) — penalize body bouncing
- `foot_slip` (-0.5 / -0.3) — penalize foot sliding during contact
- `action_smoothness` (-1.0) — penalize jerky actions
- `body_height_tracking` (-1.0) — keep nominal height
- `stumble` (-0.1) — penalize hitting obstacles at knee height
- `air_time_variance` (-1.0) — symmetric gait enforcement
- `joint_pos` (-0.7) — penalize deviation from default (all joints, vel_thresh=0.25)
- `contact_force_smoothness` (-0.01) — gentle foot placement
- `joint_vel` (-0.01) — smooth joint movement (all joints, not just hips)
- `joint_torques` (-5e-4) — energy efficiency
- `joint_acc` (-1e-4) — smooth acceleration (all joints, not just hips)

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

> **IMPORTANT:** These are the **paper-matched** coefficients deployed in Trial 2 (Feb 28, 2026). The original custom-tuned values from Trial 1 were 2-6x too harsh and caused training collapse. See Section 13 for the full trial history.

| # | Term | Type | Weight (Spot) | Weight (V60) | Key Params | Source |
|---|------|------|--------------|-------------|------------|--------|
| 1 | `base_linear_velocity` | + | 5.0 | 5.0 | std=0.25 | spot_mdp |
| 2 | `gait` | + | 5.0 | 5.0 | std=0.1, max_err=0.2, vel_thresh=0.25 | spot_mdp (GaitReward) |
| 3 | `base_angular_velocity` | + | 5.0 | 5.0 | std=0.25 | spot_mdp |
| 4 | `foot_clearance` | + | 0.75 | 0.75 | std=0.05, tanh=2.0, target=0.125 | spot_mdp |
| 5 | `air_time` | + | 5.0 | **3.0** | mode_time=0.2, vel_thresh=0.25 | spot_mdp |
| 6 | `velocity_modulation` | + | 2.0 | 2.0 | — | shared/reward_terms |
| 7 | `vegetation_drag` | phys | -0.001 | -0.001 | — | shared/reward_terms |
| 8 | `action_smoothness` | - | -1.0 | -1.0 | — | spot_mdp |
| 9 | `air_time_variance` | - | -1.0 | -1.0 | vel_thresh=0.25 | spot_mdp |
| 10 | `base_motion` | - | -2.0 | **-1.5** | — | spot_mdp |
| 11 | `base_orientation` | - | -3.0 | -3.0 | — | spot_mdp |
| 12 | `foot_slip` | - | -0.5 | **-0.3** | vel_thresh=0.25 | spot_mdp |
| 13 | `joint_acc` | - | -1e-4 | -1e-4 | joint_names=".*" | spot_mdp |
| 14 | `joint_pos` | - | -0.7 | -0.7 | joint_names=".*", stand_still=5.0, vel_thresh=0.25 | spot_mdp |
| 15 | `dof_pos_limits` | - | -5.0 | -5.0 | joint_names=".*" | mdp |
| 16 | `joint_torques` | - | -5e-4 | -5e-4 | — | spot_mdp |
| 17 | `joint_vel` | - | -1e-2 | -1e-2 | joint_names=".*" | spot_mdp |
| 18 | `body_height_tracking` | - | -1.0 | -1.0 | target=0.42 (Spot) / 0.55 (V60) | shared/reward_terms |
| 19 | `contact_force_smoothness` | - | -0.01 | -0.01 | — | shared/reward_terms |
| 20 | `stumble` | - | -0.1 | -0.1 | knee_height=0.15 (Spot) / 0.20 (V60) | shared/reward_terms |

**Bold** = adjusted for Vision60 (heavier robot, different dynamics).

### What Changed from Our Original Weights (Trial 1 → Trial 2)

| Term | Trial 1 (custom) | Trial 2 (paper) | Factor |
|------|------------------|-----------------|--------|
| `base_linear_velocity` | +12.0 | +5.0 | 2.4x lower |
| `gait` | +15.0 | +5.0 | 3x lower |
| `foot_clearance` | +3.5 | +0.75 | 4.7x lower |
| `foot_slip` | -3.0 | -0.5 | 6x gentler |
| `joint_acc` | -5e-4 | -1e-4 | 5x gentler |
| `joint_vel` | -5e-2 | -1e-2 | 5x gentler |
| `joint_torques` | -2e-3 | -5e-4 | 4x gentler |
| `joint_pos` | -2.0 | -0.7 | 2.9x gentler |

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

Vision60 is lighter (13.6 kg vs ~32 kg for Spot) but has different dynamics. Starting from the paper-matched Spot coefficients:

| Term | Spot (paper) | V60 (adjusted) | Rationale |
|------|-------------|----------------|-----------|
| `air_time` | 5.0 | 3.0 | Lighter → easier airtime, reduce bouncing |
| `foot_slip` | -0.5 | -0.3 | Different traction characteristics |
| `base_motion` | -2.0 | -1.5 | Different inertial properties |

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
13. **Don't hand-tune reward weights.** Use the paper's exact coefficients. Our custom weights failed twice — once too lenient (do-nothing policy), once too harsh (training collapse). The paper's values encode a working equilibrium.
14. **Two Isaac Sims on one GPU = half speed each.** Sequential training is faster than parallel. Use `CUDA_VISIBLE_DEVICES` on multi-GPU machines only.
15. **Check reward *parameters*, not just weights.** `mode_time`, `velocity_threshold`, `target_height`, and `joint_names` reshape the reward surface as much as the weight values do.
16. **GPU reset clears zombies but also wipes screen sessions.** After resetting the GPU to clear D-state zombies, you need to relaunch all screen sessions (training, TensorBoard).
17. **body_contact termination must be DISABLED for Spot.** The paper deliberately removed it (commented out in `spot_env_cfg.py` lines 528-532). Spot's body geometry causes false positives during normal locomotion. Use `body_flip_over` (bad_orientation at 150°) + `undesired_contacts` reward (-2.0) instead.
18. **Set `disable_contact_processing = True`** in `__post_init__`. Matches the paper's PhysX config and affects contact force detection sensitivity.
19. **Action scale 0.2, not 0.25.** Paper uses 0.2 for Spot. Larger scales cause more violent random movements during early exploration, contributing to instability.
20. **Friction minimum must be ≥ 0.3.** Near-zero friction (0.02-0.05) is essentially ice — the robot slides uncontrollably and can't generate corrective forces. Paper uses 0.3 minimum.

---

## 13. Training Trial Log

### Trial 1: Custom-Tuned Weights (Feb 27, 2026) — FAILED

**Setup:** Spot + Vision60 simultaneous, 10K envs each, shared H100
**Log dir:** `spot_robust_ppo/2026-02-27_19-33-40/`

**What happened:**
- Both robots launched on shared GPU (~27s/iter each, half speed due to contention)
- Killed V60 to give Spot full GPU → V60 became D-state zombie
- Spot sped up to 17.2s/iter but training was diverging
- After 1,750 iterations: body_contact 99.9%, episode length 3.9 steps, terrain_levels 0.009
- Robot actively getting worse, not just failing to learn

**Root cause:** Reward weights 2-6x harsher than the paper's validated coefficients. Chaotic gradient landscape — robot pulled toward movement by huge rewards (+12.0, +15.0) while being crushed by disproportionate penalties (-3.0 slip, -5e-4 joint acc, -5e-2 joint vel).

**Checkpoints saved:** model_0.pt, model_500.pt, model_1000.pt, model_1500.pt (all non-functional)

---

### Trial 2: Paper-Matched Coefficients (Feb 28, 2026) — FAILED

**Setup:** Spot only, 10K envs, solo H100 (clean GPU after reset)
**Log dir:** `spot_robust_ppo/2026-02-28_08-52-14/`

**Changes from Trial 1:**
- All 19 reward weights set to paper's exact values (see Section 8)
- Parameter corrections: mode_time 0.3→0.2, velocity_threshold 0.5→0.25, target_height 0.10→0.125
- Joint names for acc/vel changed from hip-only (`.*_h[xy]`) to all joints (`.*`)

**Results after 1,471 iterations (~7 hours):**
- body_contact: 12.6% → 99.3% (same death spiral as Trial 1)
- episode_length: 27.0 → 4.68 steps
- terrain_levels: 3.39 → 0.017
- mean_reward: -4.09 → +0.20 (positive but stagnant)
- time_out peaked at 3.1% mid-run then fell to 0.7%

**Root cause:** Full config comparison against the paper's actual `spot_env_cfg.py` revealed that **body_contact termination was deliberately disabled** in the paper. The paper uses a flip-over check (150°) + reward penalty (-2.0) instead. Our termination was killing 99% of episodes before the robot could learn anything. Also found: friction too extreme (min 0.02 vs paper's 0.3), action scale too large (0.25 vs 0.2), missing `disable_contact_processing`.

**Outcome:** FAILED — same trajectory as Trial 1. Correct rewards are useless when termination kills episodes in <5 steps.

---

### Trial 3: Structural Fixes (Feb 28, 2026) — IN PROGRESS

**Setup:** Spot only, 10K envs, solo H100 (clean GPU after reset)
**Log dir:** `spot_robust_ppo/2026-02-28_16-05-11/`
**TensorBoard:** `http://172.24.254.24:6006`

**Changes from Trial 2 (4 structural fixes from full config comparison):**
1. **REMOVED** `body_contact` termination → Added `body_flip_over` (bad_orientation at 150°)
2. **ADDED** `undesired_contacts` reward penalty (weight=-2.0) — soft penalty replaces hard kill
3. **RAISED** friction minimums: static 0.05→0.3, dynamic 0.02→0.3
4. **REDUCED** action scale: 0.25→0.2
5. **ADDED** `disable_contact_processing = True`

**Iteration 0 results (vs Trial 2 iteration 0):**
| Metric | Trial 2 | Trial 3 | Change |
|--------|---------|---------|--------|
| episode_length | 27.0 | **31.77** | +18% — surviving full episodes |
| body termination | 12.6% (contact) | **0.24%** (flip-over) | 50x fewer kills |
| time_out | 1.0% | **1.14%** | More episodes reaching time limit |
| terrain_levels | 3.39 | **3.50** | Starting on real terrain |
| mean_reward | -4.09 | **-6.93** | More negative but from longer episodes |

**Performance:** ~18.4s/iter, ~17,350 fps, 38C
**ETA:** ~9.5 hours for 30K iterations

**What to watch for:**
- episode_length should stay >20 and climb toward 30+
- terrain_levels should remain >2.0 and advance
- mean_reward should trend upward from -6.93
- gait and velocity rewards should grow as robot learns to walk

---

### Trial 4: Vision60 (PLANNED)

After Spot Trial 3 succeeds:
- Apply same structural fixes (flip-over termination, undesired_contacts, raised friction)
- Paper-matched coefficients with V60-specific adjustments (Section 10)
- Progressive DR schedule (mild → aggressive over 15K iters)
- Solo H100, 10K envs

---

## 14. Lessons Learned from Trials

### Lesson 1: Use Published Coefficients Exactly

Our hand-tuned weights failed twice (Bug #3 do-nothing, Bug #11 too-harsh). The paper's coefficients encode a working equilibrium across all 19 terms that isn't obvious from reading the equations. Each weight interacts with every other weight through the gradient. Start from the paper's exact numbers, then tune individually if needed.

### Lesson 2: Reward Weights Are a System, Not Individual Knobs

Changing `foot_slip` from -0.5 to -3.0 doesn't just make slip 6x more penalized. It changes the *relative* gradient contribution of every other term. The policy will over-optimize for slip avoidance at the expense of velocity tracking, gait timing, and everything else. Always consider the full reward landscape, not individual terms.

### Lesson 3: Check Parameters, Not Just Weights

Two reward functions with the same weight can behave completely differently based on internal parameters. Our `mode_time=0.3` vs the paper's `0.2` changed gait reward timing. Our `velocity_threshold=0.5` vs `0.25` changed what counted as "moving." Our `joint_names=".*_h[xy]"` vs `".*"` changed which joints were penalized. These shape the reward surface as much as the weights do.

### Lesson 4: Sequential Beats Parallel on Single GPU

Two Isaac Sim instances on one H100 each ran at ~50% speed. Sequential training was 3 days faster for the same total work. GPU PhysX context-switching between two simulations wastes more than it saves.

### Lesson 5: Monitor Training From Iteration 1

Our Trial 1 ran for 1,750 iterations (~8 hours) before we checked the metrics and discovered it was diverging. Earlier monitoring would have caught the problem within the first 100 iterations — body_contact was already climbing, episode_length was already dropping. Set up TensorBoard before launching, and check within the first 30 minutes.

### Lesson 6: Compare the FULL Config, Not Just Rewards

We matched the paper's reward coefficients exactly (Trial 2) and still failed — because the `body_contact` termination was killing 99% of episodes in <5 steps. The paper deliberately disabled this termination and replaced it with a soft reward penalty. Termination conditions, friction ranges, action scale, and physics settings (like `disable_contact_processing`) are equally important as reward weights. When reproducing a paper, diff **every section** of the config.

### Lesson 7: Prefer Soft Penalties Over Hard Terminations

A hard termination (instant episode death) gives zero gradient — the policy learns "don't be here" without learning what to do instead. A reward penalty (-2.0 for undesired contacts) provides a smooth gradient that teaches gradual avoidance. The paper's approach of replacing body_contact termination with undesired_contacts reward went from 99.3% kill rate to 0.24% flip-over rate — a 400x improvement in episode survival at iteration 0.

---

*Created for AI2C Tech Capstone — MS for Autonomy, February 2026*
*Last updated: February 28, 2026 — Trial 3 (structural fixes: no body contact kill) in progress*
