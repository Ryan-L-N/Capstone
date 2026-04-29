# ARL Delivery Package — Spot Rough Terrain Locomotion

**AI2C Tech Capstone — MS for Autonomy**
**Date:** February 16, 2026

---

## What This Is

This package contains the complete Spot rough terrain locomotion system: a trained RL policy (30,000 iterations on H100), a 100-meter obstacle course with 12 terrain segments, dual gait switching (FLAT/ROUGH), and WASD + Xbox controller teleop.

The trained policy takes 235 sensor observations and outputs 12 joint position targets at 50 Hz, enabling Spot to walk, trot, climb stairs, and traverse uneven terrain.

---

## Quick Start

```bash
# Requires: Isaac Sim 5.1.0, Isaac Lab 2.3.0, Python 3.11, CUDA GPU
# Activate the isaaclab conda environment first

# Run the obstacle course with trained policy
cd 07_Launcher/
python launch.py grass-obstacle

# Controls:
#   WASD / Left Stick    — Move forward/back, turn
#   G / RB               — Toggle FLAT ↔ ROUGH gait
#   M / LB               — Toggle FPV camera
#   SHIFT / A             — Cycle drive mode
#   R / Y                — Reset to start
#   ESC                  — Exit
```

---

## Package Contents

```
ARL_DELIVERY/
|
|-- README.md                          <-- This document
|
|-- 01_Documentation/                  <-- Project docs, plans, debug history
|   |-- FORTHETEAM.md                  <-- Complete project walkthrough (start here)
|   |-- PROJECT_PLAN.md                <-- Overall project plan
|   |-- GRASS_EXPERIMENT_README.md     <-- Grass experiment status & results
|   |-- ROUGH_POLICY_DEBUG_HANDOFF.md  <-- Full deployment debug history + resolution
|   |-- TELEOP_PLAN.md                 <-- Teleop system design
|   |-- Isaac_on_H-100.md              <-- H100 server setup & usage
|
|-- 02_Obstacle_Course/                <-- Main deliverable: 100m obstacle course
|   |-- spot_obstacle_course.py        <-- 1753 lines — full course with dual gait
|   |-- spot_rough_terrain_policy.py   <-- Trained policy wrapper (loads model_29999.pt)
|
|-- 03_Rough_Terrain_Policy/           <-- Policy deployment standalone
|   |-- spot_rough_terrain_policy.py   <-- Policy wrapper module
|   |-- test_rough_standalone.py       <-- Standalone test (A/B/C phases)
|   |-- play_rough_teleop.py           <-- Isaac Lab env teleop (GPU PhysX)
|
|-- 04_Teleop_System/                  <-- WASD + Xbox teleop with grass terrain
|   |-- spot_teleop.py                 <-- 1142 lines — full teleop system
|   |-- grass_physics_config.py        <-- Grass friction & physics config
|
|-- 05_Training_Package/               <-- Everything to train the policy
|   |-- README.md                      <-- Training guide (hardware, setup, monitoring)
|   |-- spot_rough_48h_cfg.py          <-- Main training script
|   |-- train_spot_rough_48h.sh        <-- Shell launcher
|   |-- debug_10iter.sh                <-- Quick verification (2 min)
|   |-- eval_checkpoints.sh            <-- Post-training evaluation
|   |-- TRAINING_PLAN.md               <-- Detailed training rationale
|   |-- LESSONS_LEARNED.md             <-- H100 deployment issues & fixes
|   |-- 48h_training_docs/             <-- Training completion & results
|   |   |-- 48_hr_training_test.md     <-- Final metrics + download plan
|   |   |-- LESSONS_LEARNED.md         <-- Debug run + deployment lessons
|   |   |-- TRAINING_PLAN.md           <-- Hardware budget & iteration planning
|   |-- isaac_lab_spot_configs/        <-- Isaac Lab Spot rough terrain configs
|       |-- rough_env_cfg.py
|       |-- __init__.py
|       |-- agents/rsl_rl_ppo_cfg.py
|       |-- mdp/rewards.py, events.py
|
|-- 06_Core_Library/                   <-- Shared simulation utilities
|   |-- sim_app.py                     <-- SimulationApp factory
|   |-- world_factory.py               <-- World creation with standard physics
|   |-- navigation.py                  <-- Robot navigation controller
|   |-- lighting.py                    <-- Scene lighting presets
|   |-- data_collector.py              <-- Experiment metrics recording
|   |-- markers.py                     <-- Visual markers (goals, targets)
|
|-- 07_Launcher/                       <-- Central experiment launcher
|   |-- launch.py                      <-- Interactive launcher
|   |-- experiment_registry.py         <-- All experiment IDs → scripts
|
|-- 08_Lessons_Learned/                <-- All lessons learned documents
|   |-- grass_lessons_learned.md       <-- 25+ lessons (ES-001 through ES-026)
|   |-- h100_training_lessons.md       <-- H100 setup, observation space, deployment
|   |-- eureka_lessons.md              <-- Eureka reward function issues
```

---

## Key Technical Details

| Parameter | Value |
|-----------|-------|
| Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.0 |
| Python | 3.11.x |
| Training | 30,000 iterations, H100 NVL, 8,192 parallel envs |
| Checkpoint | `model_29999.pt` (6.6 MB) |
| Observation | 235 dims (48 proprioceptive + 187 height scan) |
| Action | 12 joint position offsets, scale = 0.25 |
| Network | 235 -> 512 -> 256 -> 128 -> 12 (ELU) |
| PD Gains | Kp=60, Kd=1.5 |
| Physics | 500 Hz, GPU PhysX, 4/0 solver iterations |
| Control | 50 Hz (decimation = 10) |

---

## Training Results

| Metric | Start | Final | Change |
|--------|------:|------:|--------|
| Mean Reward | -0.90 | +143.74 | +144.64 |
| Episode Length | 20 steps | 573 steps | 28.6x |
| Gait Quality | 0.06 | 5.28 | 88x |
| Terrain Level | 3.18 | 4.42 | +1.24 |
| Total Experience | — | 5.9B timesteps | ~53 hours |

---

## Obstacle Course Layout

```
 0m     10m    20m    30m 35m    45m 50m    60m 65m    75m 80m    90m    100m
 |START | WARM |GRASS |BRK| STEPS|FLT|RUBBLE|FLT|BLOCKS|FLT|BRICKS|FINISH|
 | flat |H1 grs|+stone|   | 0.75m|   |-0.5m |   |static|   |120dyn|  fin |
```

12 segments testing: flat walking, grass friction, stone obstacles, stair climbing (0.75m peak), rubble pool descent (-0.5m), large block navigation, and dynamic instability (120 loose bricks).

---

## Critical Deployment Notes

1. **Height scan = 0.0** (not 1.0) — fills 187 height_scan dims with 0.0 for flat ground
2. **GPU PhysX required** — `backend="torch"`, `device="cuda:0"` in World()
3. **CUDA tensors only** — GPU PhysX silently ignores numpy arrays
4. **Checkpoint path** — `C:\IsaacLab\logs\rsl_rl\spot_rough\48h_run\model_29999.pt`

See `01_Documentation/ROUGH_POLICY_DEBUG_HANDOFF.md` Section 16 for full deployment debug history.

---

## Recommended Reading Order

1. `01_Documentation/FORTHETEAM.md` — Start here. Complete project walkthrough.
2. `05_Training_Package/README.md` — How to train the policy on your hardware.
3. `02_Obstacle_Course/spot_obstacle_course.py` — The main demo script.
4. `08_Lessons_Learned/grass_lessons_learned.md` — 25+ documented issues and fixes.

---

**AI2C Tech Capstone Team — February 2026**
