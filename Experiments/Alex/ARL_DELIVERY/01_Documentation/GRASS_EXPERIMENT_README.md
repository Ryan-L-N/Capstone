# Experimental Design: Tall Grass Navigation

## Project Overview

This experimental design investigates quadruped robot navigation through **simulated tall grass terrain** at various heights and densities in NVIDIA Omniverse Isaac Sim. The study aims to understand how vegetation characteristics affect locomotion, sensor performance, and learned navigation policies.

> ⚠️ **SCOPE**: Single 60ft × 30ft (18.3m × 9.1m) room with procedurally generated grass terrain.

---

## CURRENT STATUS (February 16, 2026)

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Baseline | ✅ Complete | SpotFlatTerrainPolicy baseline established |
| Phase 2: Grass Height | ✅ Complete | Friction-based grass, 100% TCR all heights |
| Phase 3: Grass Density | ✅ Complete | 100% TCR all densities |
| Phase 4: Combined | ✅ Complete | 40% TCR with obstacles |
| Phase 5: Advanced RL | ✅ Complete | 48h H100 training (30k iters) + deployed |
| Phase 5b: Obstacle Course | ✅ Complete | 100m course, dual gait switching (FLAT/ROUGH) |
| Phase 6: Cross-Platform | ⏳ Pending | |
| Phase 7: Analysis | ⏳ Pending | |

### Phase 5 — RL Training Summary

**Eureka attempt (Feb 4)**: Failed — robot learned to balance, not walk (stability > locomotion rewards).

**H100 Training (Feb 13-16)**: SUCCESS
- **Training**: 30,000 iterations on H100 NVL, 8,192 parallel envs, ~53 hours
- **Final reward**: +143.74 (from -0.90), episode length 573 steps (from 20)
- **Terrain level**: 4.42 (curriculum mastered rough terrain)
- **Checkpoint**: `model_29999.pt` (6.6 MB)

**Standalone Deployment (Feb 16)**: SUCCESS
- Deployed in 100m obstacle course with WASD + Xbox teleop
- Dual gait switching: G key / RB button toggles FLAT ↔ ROUGH policy
- Key fix: height scan fill value 1.0 → 0.0 (see lessons_learned.md ES-025)
- GPU PhysX required to match training dynamics

### Key Files
- `code/spot_obstacle_course.py` — 100m obstacle course (1753 lines)
- `code/spot_rough_terrain_policy.py` — Trained policy wrapper for deployment
- `code/spot_teleop.py` — WASD/Xbox teleop with grass terrain
- `48h_training/` — Training plan, lessons learned, completion docs
- `ROUGH_POLICY_DEBUG_HANDOFF.md` — Full deployment debug history + resolution

---

## Research Questions

1. **Height Impact**: How does grass height affect navigation success rate and locomotion stability?
2. **Density Impact**: How does grass density affect path planning and energy expenditure?
3. **Sensor Degradation**: How does tall grass affect LiDAR and visual sensor performance?
4. **Policy Robustness**: Can RL policies trained in grass generalize across height/density variations?
5. **Cross-Platform Transfer**: Do grass navigation policies transfer between Spot and Vision 60?

---

## Environment Specification

| Parameter | Value |
|-----------|-------|
| **Room Size** | 60 ft × 30 ft (18.3m × 9.1m) |
| **Base Terrain** | Flat ground plane |
| **Vegetation** | Procedural grass (blade geometry or particle system) |
| **Grass Coverage** | Configurable zones within room |

---

## Experimental Variables

### Independent Variables (Manipulated)

| Variable | Symbol | Levels | Phase Active |
|----------|--------|--------|--------------|
| Grass Height | H | H0 (None), H1 (0.1m), H2 (0.3m), H3 (0.5m), H4 (0.7m) | Phase 2 |
| Grass Density | G | G0 (None), G1 (Sparse), G2 (Moderate), G3 (Dense) | Phase 3 |
| Obstacle Presence | O | O0 (None), O1 (Sparse obstacles in grass) | Phase 4 |
| Training Iterations | I | I1 (100), I2 (250), I3 (500) | Phase 5 |
| Reward Configuration | R | R1-R4 (Speed, Balanced, Efficiency, Recovery) | Phase 5 |
| Platform | P | Spot (R1), Vision 60 (R2) | All phases |

### Dependent Variables (Measured)

| Variable | Symbol | Description |
|----------|--------|-------------|
| Task Completion Rate | TCR | % of successful navigation runs |
| Navigation Time | T_nav | Time to reach target (seconds) |
| Collision Count | CC | Contact events with obstacles |
| Path Efficiency | η_path | Straight-line / actual distance |
| Stability Score | S | Balance metric (0-1) |
| Energy Proxy | E_exp | Cumulative joint torque |
| Sensor Occlusion Rate | SOR | % of sensor readings blocked by grass |

---

## Phase Structure (7 Phases)

| Phase | Name | Focus | Runs (Est.) |
|-------|------|-------|-------------|
| 1 | Baseline | No grass (flat room reference) | 30 |
| 2 | Grass Height | H1-H4 height levels | 60 |
| 3 | Grass Density | G1-G3 density levels | 45 |
| 4 | Combined Obstacles | Grass + sparse obstacles | 60 |
| 5 | Advanced RL | Train policies in grass | 156 |
| 6 | Cross-Platform | Spot ↔ V60 transfer | 80 |
| 7 | Analysis | Statistical synthesis | — |

**Estimated Total**: ~430 runs per round × 2 rounds + 80 cross-platform = **~940 runs**

---

## Platform Strategy

| Round | Platform | Purpose |
|-------|----------|---------|
| **Round 1** | Boston Dynamics Spot | Design validation (current) |
| **Round 2** | Ghost Robotics Vision 60 | Replication (awaiting SDK) |

> 📋 **PHASED APPROACH**: Spot validates design; V60 replicates identical protocol.

---

## Grass Simulation Approach

### Implementation Options in Omniverse

| Method | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| **Blade Geometry** | Accurate physics | High compute cost | Use for dense areas |
| **Particle System** | Fast rendering | Limited physics | Use for visual only |
| **Height Field + Friction** | Fast, physics-based | Less realistic | Use for large areas |
| **Hybrid** | Balance of above | Complex setup | **Recommended** |

### Grass Physics Model

```
Properties per grass patch:
- Height: 0.1m - 0.7m
- Stiffness: Moderate (bends, doesn't break)
- Friction: μ = 0.3-0.6 (height dependent)
- Recovery: Springs back after contact
- Density: Blades per m² (configurable)
```

---

## Success Criteria

| Metric | Acceptable | Good | Excellent |
|--------|------------|------|-----------|
| TCR (H1-H2) | ≥ 70% | ≥ 85% | ≥ 95% |
| TCR (H3-H4) | ≥ 50% | ≥ 70% | ≥ 85% |
| Transfer Gap | ≤ 25% | ≤ 15% | ≤ 10% |

---

## Directory Structure

```
experimental_design_grass/
├── README.md                           ← You are here
├── lessons_learned.md
├── phases/
│   ├── phase_1_baseline.md
│   ├── phase_2_grass_height.md
│   ├── phase_3_grass_density.md
│   ├── phase_4_combined_obstacles.md
│   ├── phase_5_advanced_rl.md
│   ├── phase_6_cross_platform.md
│   └── phase_7_analysis.md
├── variables/
│   ├── environment.md
│   ├── grass_height.md
│   ├── grass_density.md
│   ├── object_density.md
│   ├── robot_platforms.md
│   ├── training_iterations.md
│   └── rewards_and_penalties.md
└── experiments/
    ├── experiment_matrix.md
    ├── controlled_variables.md
    ├── dependent_independent_variables.md
    └── sim_to_real_transfer.md
```

---

## Key Differences from Flat Room Experiment

| Aspect | Flat Room | Grass Terrain |
|--------|-----------|---------------|
| Primary Challenge | Obstacle avoidance | Vegetation traversal |
| Sensor Impact | Minimal | High (occlusion) |
| Locomotion | Standard gait | Modified gait (high-step) |
| Physics | Rigid body only | Deformable vegetation |
| Phases | 6 | 7 (added combined phase) |

---

## Cross-References

- Related: `../experimental_design_flat_room/` — Obstacle navigation baseline
- Platform specs: `/variables/robot_platforms.md`
- Lessons: `/lessons_learned.md`
