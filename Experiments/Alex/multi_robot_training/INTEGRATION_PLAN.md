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
| 21 | `undesired_contacts` | - | **-1.5** | -1.5 | body_names=["body"], threshold=1.0 | mdp |
| 22 | `body_scraping` | - | **-2.0** | -2.0 | contact_thresh=1.0, vel_thresh=0.3 | shared/reward_terms |

**Bold** = adjusted for Vision60 or changed for Phase B. `undesired_contacts` was -5.0 in Phase A, lowered to -1.5 for rough terrain. `body_scraping` is new for Phase B.

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
21. **init_noise_std=1.0 is too high for Spot.** With action_scale=0.2, it gives ±11° random joint movements that flip the robot. Use 0.5 (±6°) and cap max_noise_std at 1.0.
22. **RSL-RL schedule="adaptive" overrides manual LR settings.** Use schedule="fixed" when applying your own LR schedule (e.g., cosine annealing via monkey-patch).
23. **Start with reference DR, not aggressive DR.** Zero external forces/torques, ±5kg mass, ±0.5 m/s pushes. Increase only after the robot learns basic locomotion.
24. **Start on flat terrain.** The robot must learn to walk before it can handle rough terrain. Use `--terrain flat` for the first 500 iterations, then resume with `--terrain robust`. Trying to learn locomotion and terrain traversal simultaneously gives no useful gradient signal.

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

### Trial 3: Structural Fixes (Feb 28, 2026) — FAILED

**Setup:** Spot only, 10K envs, solo H100 (clean GPU after reset)
**Log dir:** `spot_robust_ppo/2026-02-28_16-05-11/`

**Changes from Trial 2 (5 structural fixes from full config comparison):**
1. **REMOVED** `body_contact` termination → Added `body_flip_over` (bad_orientation at 150°)
2. **ADDED** `undesired_contacts` reward penalty (weight=-2.0) — soft penalty replaces hard kill
3. **RAISED** friction minimums: static 0.05→0.3, dynamic 0.02→0.3
4. **REDUCED** action scale: 0.25→0.2
5. **ADDED** `disable_contact_processing = True`

**Results (8,745 iterations before crash):**

| Metric | Iter 0 | Iter 100 | Iter 1000 | Iter 8000 | Iter 8745 |
|--------|--------|----------|-----------|-----------|-----------|
| body_flip_over | 0.24% | **97.8%** | 91.4% | 96.4% | 96.0% |
| episode_length | 31.8 | 21.7 | 82.4 | 45.2 | 21.2 |
| terrain_levels | 3.50 | 0.048 | 0.014 | 0.033 | 0.040 |
| mean_reward | -6.93 | -4.42 | +0.78 | -1.57 | -8.67 |
| value_loss | 13.8 | 1.83 | 1.63 | 1.78 | **inf** |
| noise_std | 1.0 | **1.68** | 1.20 | 1.39 | **1.79** |
| learning_rate | 1e-4 | 1e-4 | 1e-4 | ~0 | **0.01** |

**Root causes (two independent bugs):**

1. **Exploration noise death spiral:** `init_noise_std=1.0` with `action_scale=0.2` gives actions ~N(0, 0.2) rad — violent enough to flip the robot. Without body_contact termination to punish bad exploration instantly, the flip-over termination at 150° is too lenient. Most actions flip the robot → no useful gradient → noise_std grows → more violent actions → more flips. By iter 100, noise_std hit 1.68 and 97.8% of episodes ended in flip-over.

2. **LR schedule conflict:** RSL-RL's `schedule="adaptive"` overrides the cosine annealing set by our training script. The adaptive schedule crushed LR to ~0 (no learning) for 8,700 iterations, then spiked it to 0.01 → value function exploded to infinity → crash.

**Also discovered:** DR was far too aggressive vs the reference config. External forces (±8N), torques (±3Nm), push velocity (±1.5 m/s) all 3-16x stronger than the working Isaac Lab reference.

---

### Trial 4: DR + LR + Reward Fixes (Mar 2, 2026) — FAILED

**Setup:** Spot only, 10K envs, solo H100
**Log dir:** `spot_robust_ppo/2026-03-02_10-39-33/`

**Changes from Trial 3:**
1. **ZEROED** external forces (±8N → 0) and torques (±3Nm → 0) on reset
2. **REDUCED** mass DR: ±8kg → ±5kg (match reference)
3. **REDUCED** push velocity: ±1.5 → ±0.5 m/s (match reference)
4. **INCREASED** push interval: (5, 12) → (10, 15)s (match reference)
5. **REDUCED** joint reset velocity: ±3.0 → ±2.5 (match reference)
6. **INCREASED** gait weight: 5.0 → 10.0 (match reference)
7. **INCREASED** foot_clearance weight: 0.75 → 2.0 (match reference)
8. **FIXED** LR schedule: `adaptive` → `fixed` (so cosine annealing isn't overridden)

**Results (crashed at ~iter 80):**

| Metric | Iter 0 | Iter 8 | Iter 77 |
|--------|--------|--------|---------|
| body_flip_over | 0.24% | 76.1% | **95.5%** |
| episode_length | 31.9 | 82.3 | 117.9 |
| terrain_levels | 3.50 | 2.05 | 0.14 |
| mean_reward | -5.83 | -12.3 | -11.1 |
| value_loss | 12.5 | 8.6 | **212,568** |
| noise_std | 1.0 | 1.02 | **2.0** (ceiling) |

**Improvement over Trial 3:** DR fixes delayed the death spiral by ~70 iterations (76% flip_over at iter 8 vs Trial 3's 97.8% at iter 100). Episode length reached 117 steps (vs Trial 3's max of 82). But the fundamental noise spiral still dominated.

**Root cause:** Same exploration noise death spiral as Trial 3. The `init_noise_std=1.0` is too high for Spot — even with gentle DR, the random actions flip the robot. Noise_std hit the 2.0 ceiling by iter 77, value function exploded, training crashed.

**Key insight:** The DR fixes helped (delayed failure by 70 iters, 4x longer episodes at peak) but are insufficient. The noise initialization is the root cause — the robot must start with gentle enough actions to NOT flip over, so it can learn what walking looks like before exploring aggressively.

---

### Trial 5: Noise Stabilization (Mar 2, 2026) — FAILED

**Setup:** Spot only, 10K envs, solo H100
**Log dir:** `spot_robust_ppo/2026-03-02_12-15-09/`

**Changes from Trial 4 (targeting the noise death spiral):**
1. **REDUCED** `init_noise_std`: 1.0 → 0.5
2. **TIGHTENED** `max_noise_std`: 2.0 → 1.0
3. **INCREASED** `undesired_contacts` penalty: -2.0 → -5.0
4. All Trial 4 DR fixes retained
5. `schedule="fixed"` retained

**Results (~100 iterations):**

| Metric | Trial 4 (iter 77) | **Trial 5 (iter 1)** | **Trial 5 (iter 26)** | **Trial 5 (iter 101)** |
|--------|---|---|---|---|
| flip_over | 95.5% | **21.6%** | **80.6%** | **96.2%** |
| episode_length | 117.9 | **62.6** | **216.2** | **54.2** |
| noise_std | 2.0 | **0.50** | **0.58** | **1.0 (ceiling)** |
| value_loss | 212,568 | **6.07** | **7.86** | **3.71** |
| terrain_levels | 0.14 | **3.30** | **1.62** | **0.08** |

**Progress over previous trials:**
- Noise ceiling at 1.0 prevented value function explosion (3.71 vs 212K/inf)
- Episode length peaked at **216 steps** at iter 26 (best ever, 2x Trial 4's peak)
- Training remained numerically stable (no crash)
- But flip_over still climbed to 96.2% by iter 100

**Root cause:** Even with init_noise_std=0.5, noise grows to 1.0 (ceiling) within 100 iterations. The fundamental issue: the robot is on 12-type rough terrain from step 0. It can't learn to walk and handle terrain simultaneously. The noise spiral is a symptom — the lack of learning signal is the cause.

---

### Trial 6: Flat Terrain Warmup — First Attempt (Mar 2, 2026) — FAILED (value explosion)

**Setup:** Spot only, 10K envs, solo H100
**Log dir:** `spot_robust_ppo/2026-03-02_13-01-42/`

**Strategy:** Two-phase training — learn to walk on flat first, then transfer to rough terrain.
- 100% flat terrain (MeshPlaneTerrainCfg), 500 iterations
- All Trial 5 noise fixes (init=0.5, max=1.0), all Trial 4 DR fixes
- `warmup_iters=500`, `save_interval=500`, `lr_max=1e-3`

**Results:**

| Iter | Reward | Ep Length | Flip Over | Noise Std | Value Loss |
|------|--------|-----------|-----------|-----------|------------|
| 2 | 2.87 | 70 | 50.8% | 0.50 | 0.397 |
| 70 | 51.5 | 767 | 49.4% | 0.52 | 0.316 |
| 199 | 374 | 1,500 | 5.4% | 0.58 | 0.216 |
| 273 | 318 | 1,279 | 3.8% | 0.77 | 1,233 |
| 274 | -8,844 | 1,282 | 4.5% | 0.77 | 3.3B |
| 279 | — | — | — | 0.77 | **inf** |

**The robot learned to walk!** This was the first trial where the robot actually acquired locomotion — reward hit 384, episode length maxed at 1,500, only 5.4% flip-over. But then the value function exploded to infinity at iteration 274.

**Root cause (Bug #18):** Two issues compounded:
1. `warmup_iters=500` on a 500-iteration run meant the LR never stopped climbing — it linearly increased the entire run. By iter 273, LR ≈ 5.5e-4 and still rising.
2. `save_interval=500` meant no checkpoints were saved before the crash. The entire learned policy was lost.

---

### Trial 7: Flat Terrain — LR Fix (Mar 2, 2026) — FAILED (value explosion, but checkpoints saved)

**Setup:** Spot only, 10K envs, solo H100
**Log dir:** `spot_robust_ppo/2026-03-02_15-21-03/`

**Changes from Trial 6:**
- `warmup_iters=50` (was 500 — quick warmup, then cosine decay)
- `save_interval=50` (was 500 — checkpoint every 50 iterations)
- `lr_max=1e-3` (unchanged)

**Results:**

| Iter | Reward | Ep Length | Flip Over | Noise Std | Value Loss |
|------|--------|-----------|-----------|-----------|------------|
| 2 | 1.6 | 30 | 0.03% | 0.52 | 0.975 |
| 77 | 64.5 | 993 | 45.1% | 0.60 | 0.241 |
| 100 | 133 | 1,181 | 39.2% | 0.70 | 0.610 |
| 101 | 136 | 1,246 | 38.1% | **0.86** | 0.599 |
| 102 | -7.3B | 822 | 39.3% | 0.86 | **3.4e24** |

Same value explosion, but earlier (iter 102 vs 274). The key: noise spiked from 0.70 to 0.86 between iterations 100-101, then value loss detonated.

**Root cause:** `lr_max=1e-3` is too high. With warmup_iters=50, the LR peaked at 1e-3 by iter 50 and was ~9.7e-4 at iter 100. In Trial 6, the same explosion happened at iter 274 when LR climbed to ~5.5e-4. The value function can't handle LR above ~5e-4 with growing noise.

**Saved:** `model_50.pt` (noise=0.59, clean), `model_100.pt` (noise=0.70, right before explosion)

---

### Trial 7b: Flat Terrain — Lower LR, Resume (Mar 2, 2026) — SUCCESS (Phase A complete)

**Setup:** Spot only, 10K envs, solo H100
**Log dir:** `spot_robust_ppo/2026-03-02_15-55-25/`

**Changes from Trial 7:**
- Resumed from Trial 7's `model_50.pt` (robot already learning)
- `lr_max=3e-4` (was 1e-3 — stay below instability threshold)
- `warmup_iters=50`, `save_interval=50` (retained)

**Results:**

| Iter | Reward | Ep Length | Flip Over | Time Out | Noise Std | Value Loss |
|------|--------|-----------|-----------|----------|-----------|------------|
| 51 (resume) | 1.6 | 30 | 0.03% | 1.0% | 0.52 | 0.975 |
| 66 | 34.4 | 396 | 47.8% | 15.4% | 0.51 | 0.082 |
| 93 | 92 | 1,042 | 58.5% | 37.4% | 0.53 | 0.161 |
| 110 | 141 | 1,155 | 28.1% | 71.9% | 0.54 | 0.220 |
| 199 | 375 | 1,500 | 5.4% | 94.6% | 0.58 | 0.216 |
| 300 | 520 | 1,500 | 1.5% | 98.5% | 0.42 | 0.100 |
| 498 | **567** | **1,500** | **0.7%** | **99.3%** | **0.38** | **0.09** |

**Phase A: COMPLETE.** The robot has learned to walk. Key achievements:
- **99.3% of episodes survive the full episode length** (only 0.7% flip over)
- **Noise std decreased from 0.52 to 0.38** — the policy learned to be precise
- **Episode length maxed at 1,500** — Spot walks for the entire episode
- **Value loss stable at 0.09** — no explosion, clean convergence
- **10 checkpoints saved** (model_50 through model_498)
- **Total: 143M timesteps, 1.7 hours, 24K steps/sec**

**Checkpoint:** `model_498.pt` — ready for Phase B (rough terrain transfer)

---

### Trial 8: Phase B — Direct Rough Terrain Transfer (Mar 2, 2026) — FAILED (noise death spiral)

**Setup:** Spot only, 20,480 envs, solo H100
**Log dir:** `spot_robust_ppo/2026-03-02_19-33-05/`
**Resume from:** Trial 7b `model_498.pt` (flat terrain walking policy)

**Reward changes from Phase A:**
1. **LOWERED** `undesired_contacts`: -5.0 → **-1.5** (rough terrain = unavoidable body bumps)
2. **ADDED** `body_scraping` penalty: **-2.0** (penalizes belly-dragging while moving)

**Results (~40 iterations after resume, ~538 total):**

| Metric | Iter 500 (resume) | Iter 538 |
|--------|-------------------|----------|
| flip_over | 95.0% | **96.5%** |
| noise_std | 0.50 | **1.00 (ceiling)** |
| value_loss | — | **482,431** |
| reward | -54 | **-101** (getting worse) |
| terrain_levels | 1.30 | 1.18 (stuck) |

**Root cause:** The jump from 100% flat → 12-type rough terrain was too large. Even with curriculum starting at easy difficulty (row 0-1), the easiest rough terrain patches (stairs, slopes, gaps) are dramatically different from flat. The flat-trained policy immediately fails on ~95% of terrain patches → no useful gradient → noise grows to ceiling → death spiral. This is the same failure mode as Trials 3-5, but now caused by terrain shock instead of noise initialization.

**Key insight:** Curriculum learning needs smaller steps. Flat → 12-type robust is like teaching someone who just learned to walk on a sidewalk to immediately traverse an obstacle course. We need an intermediate step.

---

### Trial 9: Phase A.5 — Transition Terrain (Mar 2, 2026) — IN PROGRESS

**Setup:** Spot only, 10K envs, solo H100
**Log dir:** `spot_robust_ppo/2026-03-02_19-53-XX/`
**Resume from:** Trial 7b `model_498.pt` (flat terrain walking policy)

**Strategy:** Intermediate terrain between flat and robust — 6 terrain types, all gentle:
- 50% flat (safe zone, policy can still practice walking)
- 15% gentle slopes (max 0.25 rad ~14°, half of robust's 0.5 rad)
- 10% slight random roughness (noise 0.01-0.06, half of robust's 0.02-0.15)
- 10% gentle stairs (max 0.10m step, half of robust's 0.25m)
- 10% wave terrain (amplitude 0.02-0.08, half of robust's 0.05-0.20)
- 5% vegetation plane (drag training)

**Config:** `--terrain transition`, `num_rows=5` (fewer difficulty levels), `num_cols=20`

**Early results (iter 502, ~4 iterations after resume):**

| Metric | Failed Phase B (iter 538) | **Phase A.5 (iter 502)** |
|--------|---------------------------|--------------------------|
| flip_over | 96.5% | **1.2%** |
| noise_std | 1.00 (ceiling) | **0.39** (stable) |
| value_loss | 482,431 | **8.5** |
| reward | -101 | **+10.35** |
| terrain_levels | 1.18 (stuck) | **2.29** (climbing) |
| time_out | 2.8% | **10.1%** |

The transition terrain is working — robot survives, noise stable, curriculum advancing. After 1000 iterations, will proceed to full robust terrain (Phase B).

**Launch command:**
```bash
train_ppo.py --robot spot --terrain transition --num_envs 10000 --max_iterations 1000 \
    --warmup_iters 50 --save_interval 100 --lr_max 3e-4 --resume \
    --load_run 2026-03-02_15-55-25 --load_checkpoint model_498.pt --no_wandb
```

---

### Trial 10: Phase B — Full Robust Terrain (Mar 2, 2026) — FAILED (action smoothness explosion)

**Setup:** Spot only, 20,480 envs, solo H100
**Log dir:** `spot_robust_ppo/2026-03-02_21-58-52/`
**Resume from:** Trial 9 `model_998.pt` (transition terrain policy)

**Results (~15 iterations after resume):**

| Metric | Iter 999 (resume) | Iter ~1013 (crash) |
|--------|-------------------|-------------------|
| flip_over | — | **63%** |
| action_smoothness | — | **-103 TRILLION** |
| value_loss | — | exploding |
| reward | -16 | getting worse |

**Crash:** `RuntimeError: normal expects all elements of std >= 0.0` — the action smoothness reward exploded to -103 trillion, corrupted the policy parameters, noise std went negative, training crashed after ~15 iterations.

**Root cause (Bug #21):** Even with transition terrain training, jumping from 50% flat to ~10% flat (full robust) was too steep. Flip_over at 63% (better than Trial 8's 95%, so transition helped) but still too much failure for stable gradients. The action smoothness term is especially vulnerable — when the robot is falling chaotically on unfamiliar terrain, actions become extreme and the squared-difference penalty explodes.

**Key insight:** The terrain curriculum progression needs FOUR steps, not three. The robot needs to see all 12 terrain types at reduced difficulty before facing them at full difficulty.

---

### Trial 10b: Phase B — Robust Easy (Mar 2, 2026) — IN PROGRESS (overnight)

**Setup:** Spot only, 20,480 envs, solo H100
**Log dir:** `spot_robust_ppo/2026-03-02_22-18-XX/`
**Resume from:** Trial 9 `model_998.pt` (transition terrain policy)

**Strategy:** All 12 robust terrain types but with `num_rows=3` instead of 10. The robot sees every terrain type (stairs, gaps, slopes, obstacles, stepping stones) but only at easy-to-medium difficulty. Max stair height ~0.10m instead of 0.25m, max gap width ~0.20m instead of 0.50m, etc.

**Config:** `--terrain robust_easy` — uses `ROBUST_TERRAINS_CFG` with `num_rows=3`, `num_cols=20`

**Early results (iter 999, first iteration after resume):**

| Metric | Trial 10 (full robust) | **Trial 10b (robust_easy)** |
|--------|----------------------|---------------------------|
| flip_over | 63% → crash | **8.6%** |
| action_smoothness | -103 trillion → crash | **-0.35** (normal) |
| value_loss | exploding → crash | **39.7** (stable) |
| ep_length | 63 → crash | **89.8** |
| reward | -16 → crash | **-26.6** |

Numerically stable — no explosion risk. Running overnight (~21 hour ETA, 30K iterations).

**Launch command:**
```bash
train_ppo.py --robot spot --terrain robust_easy --num_envs 20480 --max_iterations 30000 \
    --warmup_iters 500 --save_interval 500 --lr_max 3e-4 --resume \
    --load_run 2026-03-02_19-53-10 --load_checkpoint model_998.pt --no_wandb
```

---

### Trial 10c: Phase B — Robust Easy, Height Fix (Mar 2, 2026) — FAILED (value explosion)

**Setup:** Spot only, 20,480 envs, solo H100
**Log dir:** `spot_robust_ppo/2026-03-02_22-34-19/`
**Resume from:** Trial 9 `model_998.pt`

**Fix from Trial 10b:** Disabled `body_height_tracking` (weight=0.0) for non-flat terrain. The reward used world-frame Z position, not height above terrain — penalizing the robot for standing on top of stairs (Bug #22).

**Results (~25 iterations):**

| Iter | Value Loss | Flip Over | Reward |
|------|-----------|-----------|--------|
| 1001 | 31 | 15.1% | -28 |
| 1006 | 101 | 38.8% | -25 |
| ~1025 | **4,670** | **59.4%** | +17 |

**Root cause:** `lr_max=3e-4` is too high for the terrain distribution shift. The value function was trained on transition terrain (50% flat) and now sees robust_easy (0% flat, 12 types). Value predictions are wildly wrong → large gradient updates at 3e-4 LR → value function oscillates instead of converging → crash.

---

### Trial 10d: Phase B — Robust Easy, Lower LR (Mar 2, 2026) — IN PROGRESS (overnight)

**Setup:** Spot only, 20,480 envs, solo H100
**Log dir:** `spot_robust_ppo/2026-03-02_22-50-53/`
**Resume from:** Trial 9 `model_998.pt`

**Changes from Trial 10c:**
- `lr_max=1e-4` (was 3e-4 — let value function adapt gradually)
- `body_height_tracking` disabled (retained from 10c)

**Results (~160 iterations in, stabilizing):**

| Iter | Reward | Ep Length | Flip Over | Time Out | Value Loss | Gait |
|------|--------|-----------|-----------|----------|------------|------|
| 1000 (resume) | -16 | 91 | 7.9% | 5.0% | 38.8 | 0.27 |
| 1006 | -25 | 197 | 38.8% | 11.5% | 101 | 0.67 |
| ~1025 | +8 | 430 | 49.4% | 23.0% | 967 | 1.82 |
| 1094 | **+117** | **1,099** | **42.3%** | **55.8%** | **53** | **4.73** |
| 1122 | +106 | 958 | 38.2% | 60.0% | 198 | 4.86 |
| 1160 | **+116** | **1,041** | **33.5%** | **64.3%** | **7.9** | **4.97** |

**The lower LR worked.** Value loss peaked at ~967 (vs 4,670 with 3e-4) then came back down to 7.9. Flip_over peaked at 49% then started decreasing (42% → 38% → 33.5%). By iter 1160, two-thirds of episodes are surviving and gait is near-perfect at 4.97/5.0.

**Running overnight.** 30K iterations, ~12.5 hours remaining. First checkpoint at iter 1500.

---

### Trial 11: Phase B — Full Robust (PLANNED)

Resume from Trial 10d's best checkpoint with full `--terrain robust` (all 12 types, 10 difficulty rows, `lr_max=1e-4`). Robot will know all terrain types at easy difficulty — full difficulty is refinement.

---

### Trial 12: Vision60 (PLANNED)

After Spot Phase B succeeds — same four-phase approach (flat → transition → robust_easy → robust) with V60-specific adjustments.

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

### Lesson 8: Exploration Noise Can Create a Death Spiral

`init_noise_std=1.0` with `action_scale=0.2` produces actions of ~N(0, 0.2) rad (±11°). For a top-heavy robot like Spot, this is violent enough to flip it over. Without hard termination to immediately punish flipping, the flip-over threshold (150°) is too lenient — the robot flails for dozens of steps before dying. The result: most episodes end in flip-over → no useful gradient signal → noise_std *increases* (to "explore" more) → even more violent actions → more flip-overs. This positive feedback loop ran in both Trial 3 (noise hit 1.79) and Trial 4 (noise hit 2.0 ceiling). **Fix:** Start with `init_noise_std=0.5` and cap at 1.0. The initial actions ~N(0, 0.1) rad (±6°) are gentle enough to maintain balance.

### Lesson 9: Cosine LR and RSL-RL's Adaptive Schedule Conflict

RSL-RL's `schedule="adaptive"` adjusts the learning rate based on KL divergence INSIDE the `PPO.update()` call. Our cosine annealing sets the LR BEFORE calling `update()`. The adaptive schedule overrides it — our cosine value is immediately replaced by the KL-based adjustment. In Trial 3, this caused the LR to collapse to near-zero for 8,700 iterations (no learning), then spike to 0.01 (value function exploded to infinity). **Fix:** Set `schedule="fixed"` so RSL-RL doesn't touch the LR, and let our cosine annealing be the sole LR controller.

### Lesson 11: Learn to Walk Before Learning to Climb

Five consecutive trials (Trials 1-5) failed with 95%+ termination rates. Every trial tried to teach the robot locomotion and rough terrain traversal simultaneously. On 12-type terrain with stairs, gaps, slopes, and obstacles, even gentle exploration causes the robot to trip and flip — giving the policy no useful gradient for learning basic walking. **Fix:** Start on 100% flat terrain (Phase A) so the only learning signal is locomotion — balance, gait, velocity tracking. Once the robot can walk, rough terrain becomes a fine-tuning task (Phase B), not a survival task. This is standard curriculum learning: master the prerequisite before adding complexity.

### Lesson 10: Match the Reference Config's DR Before Adding Your Own

Our aggressive DR (±8N forces, ±3Nm torques, ±1.5 m/s pushes, ±8kg mass) was 3-16x stronger than the Isaac Lab reference config for Spot (zero forces/torques, ±0.5 pushes, ±5kg mass). The reference values represent a working starting point. Start there, prove the robot can learn, THEN gradually increase DR. Jumping straight to extreme DR means the robot can't learn basic balance, let alone handle perturbations.

### Lesson 12: Save Checkpoints Frequently During Experimental Runs

Trial 6 ran beautifully for 273 iterations — the robot learned to walk, reward hit 384, 94.6% survival — then the value function exploded and the entire policy was lost because `save_interval=500` meant zero intermediate checkpoints. **Fix:** Set `save_interval=50` (or even 25) during experimental/warmup runs. Disk is cheap. Checkpoints are 21MB each. Losing 5 hours of training because you saved too infrequently is unrecoverable.

### Lesson 14: Adjust Penalties Conservatively When Changing Terrain

Phase B introduces 12 terrain types where body contact is physically unavoidable (stairs, slopes, obstacles). Keeping `undesired_contacts` at -5.0 would teach the robot to stand still — the safest way to avoid contact. We lowered it to -1.5 (still penalized, not dominant) and added a targeted `body_scraping` term (-2.0) that specifically penalizes belly-dragging at speed while allowing momentary bumps. The principle: when the environment changes, adjust the penalties that conflict with the new physics, and prefer targeted terms over blanket punishments.

### Lesson 15: Curriculum Steps Must Be Small Enough to Transfer

Trial 8 proved that flat → 12-type robust terrain is too large a jump. The flat-trained policy (99.3% survival on flat) immediately failed on 96.5% of rough terrain patches — even the easiest curriculum rows. The terrain *types* were the problem, not just difficulty. Stairs, gaps, and slopes are qualitatively different from flat — the robot has never experienced height changes under its feet. Trial 9's transition terrain (50% flat + gentle slopes/rough/stairs) bridged the gap: 1.2% flip-over from iteration 1. The lesson: curriculum steps must be small enough that the policy can transfer without catastrophic forgetting. If survival drops below ~70% on the new terrain, the step is too big.

### Lesson 17: Learning Rate Must Scale Down at Each Terrain Transition

Trial 10c used `lr_max=3e-4` (safe on flat and transition terrain) but crashed in 25 iterations on robust_easy. Trial 10d used `lr_max=1e-4` and stabilized. Each terrain transition creates a massive distribution shift in the value function — predictions trained on the old terrain are wildly wrong for the new terrain. High LR causes the value function to overcorrect, oscillate, and explode. Lower LR lets it converge gradually. The safe LR appears to decrease with each phase: 3e-4 for flat→transition, 1e-4 for transition→robust_easy.

### Lesson 18: World-Frame Rewards Break on Non-Flat Terrain

`body_height_tracking` used `root_pos_w[:, 2]` (world-frame Z) with a target of 0.42m. On flat terrain at ground z=0, this works. On stairs at ground z=1.0, the robot at z=1.42 gets penalized `(1.42 - 0.42)² = 1.0`. The penalty grows quadratically with terrain height, reaching -52 on rough terrain (Bug #22). Any reward term that uses absolute world-frame position instead of terrain-relative position will break on non-flat terrain. Disable or rewrite these terms before rough terrain training.

### Lesson 16: Separate Terrain Type Exposure from Difficulty Scaling

Trial 10 showed that even after transition training, jumping to full robust (12 types × 10 difficulty rows) crashed in 15 iterations. Trial 10b uses the same 12 types but only 3 difficulty rows — and it's stable (8.6% flip_over). The insight: there are two independent axes of terrain complexity — the number of terrain *types* (qualitative novelty) and the *difficulty* within each type (quantitative challenge). The policy can handle learning many new terrain types if the difficulty is capped, or handling hard difficulty on familiar types — but not both at once. The four-phase curriculum (flat → transition → robust_easy → robust) separates these axes.

### Lesson 13: Learning Rate Must Match Training Stability

A `lr_max=1e-3` that works for a 60K-iteration run with 500-iteration warmup is far too aggressive for a 500-iteration warmup run. Trial 7 exploded at iter 102 when the LR hit ~9.7e-4. Trial 7b succeeded with `lr_max=3e-4` — the value function stayed stable for the entire 500 iterations. The lesson: the maximum safe learning rate depends on the training phase. Short warmup runs need lower LR because the policy and value function are still fragile. Once the policy is stable (Phase B), higher LR may be safe.

---

*Created for AI2C Tech Capstone — MS for Autonomy, February 2026*
*Last updated: March 2, 2026 — Trial 10d robust_easy RUNNING OVERNIGHT (lr=1e-4, 64% survival at iter 1160)*
