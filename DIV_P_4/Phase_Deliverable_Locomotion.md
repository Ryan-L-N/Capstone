# Locomotion Policy Development

## 1. Overview

This document details every locomotion policy developed during this project phase, including training curriculum design, reward and penalty structures, iterations trained, and performance evaluation data. All locomotion policies were trained for the Boston Dynamics Spot quadruped using NVIDIA Isaac Lab and RSL-RL (PPO) on an NVIDIA H100 GPU.

## 2. Policy Architecture

All locomotion policies share the same input/output specification:

| Property | Value |
|----------|-------|
| Observation dimensions | 235 (187 height scan + 48 proprioception) |
| Action dimensions | 12 (joint position targets) |
| Control frequency | 50 Hz |
| Physics frequency | 500 Hz (decimation = 10) |
| PD gains | Kp = 60.0, Kd = 1.5 |
| Action scale | 0.2 rad offset from default standing pose |

Two network architectures were used:

| Architecture | Hidden layers | Parameters | Used by |
|-------------|--------------|------------|---------|
| Large | [1024, 512, 256] | ~1.2M | AI-Coached v8 |
| Standard | [512, 256, 128] | 286,604 | Mason Hybrid, S2R experts, all final policies |

The standard [512, 256, 128] architecture proved sufficient for all terrain conditions and became the default for all subsequent training.

## 3. Reward Function Design

### 3.1 Full 19-Term Reward Function

The locomotion reward function evolved through multiple iterations. The final version contains 19 terms:

**Positive Rewards (Incentives)**

| Term | Weight | Description |
|------|--------|-------------|
| `base_linear_velocity` | 12.0 | Reward for matching commanded forward/lateral speed |
| `gait` | 15.0 | Diagonal trot synchronization (FL+HR, FR+HL alternating) |
| `air_time` | 3.0 | Reward for proper swing phase (feet off ground ~0.3s) |
| `foot_clearance` | 2.0 | Reward for lifting feet during swing (obstacle clearance) |
| `velocity_modulation` | 2.0 | Adaptive speed on difficult terrain |
| `base_angular_velocity` | 1.0 | Reward for matching commanded yaw rate |
| `body_height_tracking` | 1.0 | Maintain target standing height (flat terrain only) |

**Negative Rewards (Penalties)**

| Term | Weight | Description |
|------|--------|-------------|
| `body_contact` | -10.0 | Body (non-foot) ground contact — catastrophic |
| `dof_pos_limits` | -10.0 | Approaching joint mechanical limits |
| `base_orientation` | -5.0 | Excessive body roll/pitch |
| `base_motion` | -4.0 | Unwanted vertical/lateral body velocity |
| `foot_slip` | -3.0 | Feet sliding during ground contact |
| `joint_pos` | -2.0 | Extreme joint angle deviations from default |
| `action_smoothness` | -0.5 | Rapid changes in control commands |
| `stumble` | -0.3 | Tripping events (foot catching terrain) |
| `joint_vel` | -0.05 | Excessive joint angular velocity |
| `contact_force_smoothness` | -0.02 | Impact forces during foot contact |
| `joint_torques` | -0.005 | Motor effort (energy efficiency) |
| `vegetation_drag` | -0.001 | Moving through vegetation (with physics drag) |

### 3.2 Reward Design Principles

Key lessons learned through extensive experimentation:

1. **Gait weight must stay at 15.0 or above.** Lower values produce bouncing exploits where the robot hops instead of walking.
2. **Tune complementary rewards together.** Foot clearance, action smoothness, and joint position all govern leg-lift behavior — changing one without adjusting the others produces suboptimal gaits.
3. **Gate all vertical velocity rewards on forward progress.** Ungated vz rewards are exploited by the robot standing on two legs (Bug #36).
4. **Clamp all unbounded penalty terms.** Squared errors and norms can explode to NaN without clamping (Bug #29).
5. **Use terrain-relative height, not world-frame Z.** World-frame height is meaningless on elevated terrain (Bug #22).

## 4. Training Curriculum

### 4.1 Phase Progression

| Phase | Terrain | Environments | Iterations | Purpose |
|-------|---------|-------------|------------|---------|
| A (Flat) | 100% smooth ground | 4,096 | 500 | Basic walking gait |
| A.5 (Transition) | 50% flat + 50% gentle | 4,096 | 1,000 | Gait adaptation |
| B-easy (Robust Easy) | 12 types, low difficulty | 8,192 | 30,000 | Terrain generalization |
| B (Robust) | 12 types, full difficulty | 8,192 | 30,000+ | Master all terrains |

### 4.2 12-Terrain Curriculum (Phase B)

Each training run uses a grid of 10 difficulty rows x 40 terrain columns = 400 patches (8m x 8m each):

| Category | Terrain Type | Proportion | Description |
|----------|-------------|-----------|-------------|
| Geometric (40%) | Pyramid stairs up | 8% | Ascending stair pyramids |
| | Pyramid stairs down | 8% | Descending stair pyramids |
| | Random grid boxes | 8% | Repeated box obstacles |
| | Stepping stones | 8% | Sparse elevated platforms |
| | Repeated boxes | 8% | Uniform obstacle grids |
| Surface (35%) | Random rough | 10% | Bumpy ground (0.02-0.12m) |
| | Slopes up | 5% | Inclined planes |
| | Slopes down | 5% | Declined planes |
| | Wave terrain | 5% | Sinusoidal ground |
| | Friction plane | 5% | Variable friction (0.05-0.9) |
| | Vegetation plane | 5% | Ground with drag forces |
| Compound (25%) | HF stairs | 10% | Height-field linear stairs |
| | Discrete obstacles | 15% | Random polyhedra |

Robots are automatically promoted to harder difficulty rows when they consistently survive, and demoted when they fall. This ensures training always occurs at the challenge frontier.

## 5. Locomotion Policies Developed

### 5.1 Phase A — Flat Terrain (Trial 7b)

**Objective:** Learn basic forward walking on smooth ground.

| Property | Value |
|----------|-------|
| Checkpoint | `model_498.pt` |
| Iterations | 500 |
| Survival rate | 99.3% |
| Action noise | 0.38 std |
| Architecture | [512, 256, 128] |

**Result:** Clean, stable walking gait. All robots survived the flat terrain consistently. This checkpoint served as the starting point for Phase A.5.

### 5.2 Phase A.5 — Transition (Trial 9)

**Objective:** Adapt flat-terrain gait to mild perturbations.

| Property | Value |
|----------|-------|
| Checkpoint | `model_998.pt` |
| Iterations | 1,000 |
| Survival rate | 92.9% |
| Gait score | 8.58 |
| Resume from | Phase A model_498 |

**Result:** Gait maintained quality while adapting to gentle slopes and small bumps. 7% failure rate on transition terrain was acceptable for moving to Phase B.

### 5.3 AI-Coached v8 (Trial 11l)

**Objective:** Push terrain mastery using AI-guided reward weight tuning.

| Property | Value |
|----------|-------|
| Checkpoint | `model_10600.pt` |
| Architecture | [1024, 512, 256] |
| Iterations | 10,600 |
| Training time | ~47 hours |
| Total steps | ~2.0 billion |
| Peak terrain | 4.83 (best ever at the time) |
| Learning rate | Adaptive with KL target 0.01 |

**AI Coach System:** A Claude Sonnet API integration that analyzed training metrics every 250 iterations and suggested reward weight adjustments within safety guardrails (max 20% change, max 3 weights per consultation).

**Final baked weights:** gait=8.5, base_linear_velocity=6.0, base_angular_velocity=6.0, action_smoothness=-1.2, base_motion=-2.4, base_orientation=-2.4, joint_pos=-0.3

**Lessons learned:**
- The [1024, 512, 256] network plateaued at terrain ~4.8 despite 47 hours of training
- The AI coach tended to progressively loosen penalties without a revert-if-failed mechanism
- The smaller [512, 256, 128] architecture ultimately achieved comparable results more efficiently

### 5.4 Mason Hybrid (Trial 12)

**Objective:** Combine Mason's proven baseline weights with the smaller [512, 256, 128] network.

| Property | Value |
|----------|-------|
| Checkpoint | `mason_hybrid_best_33200.pt` |
| Architecture | [512, 256, 128] |
| Iterations | 33,200 |
| Peak terrain | 3.83 |
| Flip rate | 0% |

**4-Environment Evaluation:**

| Environment | Distance | Zones |
|-------------|----------|-------|
| Friction | 49.5m | 5/5 |
| Grass | 49.5m | 5/5 |
| Boulder | 21.4m | 3/5 |
| Stairs | 11.5m | 2/5 |

**Key insight:** Mason's conservative weights + smaller network = rock-solid gait with 0% flip rate. Excellent on flat/drag terrain, but limited on obstacles and stairs.

### 5.5 Trial 12b — Obstacle Focus

**Objective:** Push boulder and stair performance using targeted training.

| Property | Value |
|----------|-------|
| Checkpoint | `model_44400.pt` (best), `model_54599.pt` (final) |
| Starting weights | foot_clearance=2.0, action_smoothness=-1.0, joint_pos=-0.3 |
| Terrain mix | 60% boulders/stairs |
| Iterations | 54,600 |
| Peak terrain | 4.38 |
| Flip rate | 0% |

**4-Environment Evaluation (model_44400):**

| Environment | Distance | Zones | vs. Mason Hybrid |
|-------------|----------|-------|-----------------|
| Friction | 42.2m | 5/5 | -7.3m |
| Grass | 31.7m | 4/5 | -17.8m |
| Boulder | **30.4m** | **4/5** | **+9.0m** |
| Stairs | 15.7m | 2/5 | +4.2m |

**Key insight:** Loosening `base_orientation` from -3.0 to -2.0 unlocked lateral tilt needed for boulder traversal. Tuning rewards along the same kinematic chain (foot_clearance + action_smoothness + joint_pos) produced the largest boulder improvement.

### 5.6 S2R Flat Master v3

**Objective:** Create a friction/grass specialist with sim-to-real hardening.

| Property | Value |
|----------|-------|
| Checkpoint | `flat_v3_3700.pt` |
| Architecture | [512, 256, 128] |
| Iterations | 3,700 (best out of 5,000) |
| Terrain | 45% flat, 25% rough, 15% wave, 15% slopes |
| Environments | 2,048 |
| S2R wrappers | Action delay (40ms), obs delay (20ms), sensor noise |

**Key reward weights:** gait=15.0 (highest), action_smoothness=-1.5 (strictest), foot_slip=-3.0, base_roll=-5.0

**4-Environment Evaluation:**

| Environment | Distance | Zones |
|-------------|----------|-------|
| Friction | **43.4m** | **5/5** |

**Lesson:** Best checkpoint was at iteration 3,700, not the final 5,000. Later iterations showed regression — a common pattern requiring periodic evaluation during training.

### 5.7 S2R Boulder V6 — Best Single Policy

**Objective:** Boulder field specialist starting from the obstacle-focused training.

| Property | Value |
|----------|-------|
| Checkpoint | `boulder_v6_expert_4500.pt` |
| Architecture | [512, 256, 128] |
| Iterations | 4,500 (crashed at 4,589 due to value loss spike) |
| Peak terrain | 6.13 |
| Environments | 4,096 |
| S2R wrappers | Full sim-to-real hardening |

**4-Environment Evaluation:**

| Environment | Distance | Zones | Speed (m/s) |
|-------------|----------|-------|-------------|
| Friction | **49.5m** | **5/5** | **0.99** |
| Grass | 21.2m | 3/5 | 0.35 |
| Boulder | **26.6m** | **3/5** | **0.31** |
| Stairs | **22.9m** | **3/5** | **0.31** |

**This is the best single policy produced.** Notable achievements:
- 22.9m on stairs without any dedicated stair training — better than all stair-specialist models
- Perfect friction completion at nearly 1.0 m/s
- Strong boulder performance at 26.6m
- Selected as the frozen locomotion backbone for Phase C navigation training

### 5.8 S2R Boulder V7d

**Objective:** Break the boulder zone 3 wall using rear/front foot clearance rewards.

| Property | Value |
|----------|-------|
| Checkpoint | `boulder_v7d_expert_4999.pt` |
| Iterations | 5,000 |
| Terrain | 5.12 (never recovered to V6's 6.13) |
| Flip rate | 4.8% |
| New rewards | rear_clearance_bonus, front_clearance, riser_collision |

**4-Environment Evaluation:**

| Environment | V6 (4500) | V7d (4999) | Delta |
|-------------|-----------|-----------|-------|
| Friction | 49.5m (0.99 m/s) | 49.5m (0.26 m/s) | -0.73 m/s speed |
| Grass | 21.2m (0.35 m/s) | 30.6m (0.10 m/s) | +9.4m, -0.25 m/s |
| Boulder | 26.6m (0.31 m/s) | 20.8m (0.27 m/s) | -5.8m |
| Stairs | 22.9m (0.31 m/s) | 22.0m (0.07 m/s) | -0.9m, -0.24 m/s |

**Conclusion:** Clearance rewards traded boulder distance for grass distance and made the gait significantly slower across all environments. Boulder V6 remains superior on both distance and speed.

### 5.9 S2R Stair Specialists (V5-V7b)

**Objective:** Break the 22m stair wall where step height exceeds 13cm.

**Approaches tested:**

| Version | Approach | Best Distance | Result |
|---------|----------|--------------|--------|
| V5a-f | Various reward tuning | 21.7m | Plateau at zone 3 |
| V5h-i | SURGE reward tuning | 21.7m | No improvement |
| V6a | Bidirectional stairs | Killed (ballet exploit) | Bug #36 — vz reward exploited |
| V6b | Fixed bidirectional | 20.6m | Height curriculum broken |
| V6c | Abs height curriculum | 20.6m | Oscillated 5.0-5.8, no gain |
| V7a | Slow curriculum + narrow | 21.6m | Gait not improving |
| V7b | Front clearance + riser | Crashed iter 1553 | 60% flip, value loss spike |

**The 22m stair wall:** Every approach plateaus at approximately 21-22m, where step height jumps to 13cm+. The robot needs fundamentally different behavior (hop/lunge) rather than incremental reward tuning. This remains an unsolved challenge.

## 6. Comparative Evaluation

### 6.1 All Policies — 4-Environment Results

| Policy | Friction | Grass | Boulder | Stairs |
|--------|---------|-------|---------|--------|
| Mason Hybrid (33200) | 49.5m (5/5) | 49.5m (5/5) | 21.4m (3/5) | 11.5m (2/5) |
| Trial 12b (44400) | 42.2m (5/5) | 31.7m (4/5) | 30.4m (4/5) | 15.7m (2/5) |
| AI-Coached v8 (10600) | 49.5m (5/5) | 41.2m (5/5) | 21.4m (3/5) | 11.5m (2/5) |
| **Boulder V6 (4500)** | **49.5m (5/5)** | 21.2m (3/5) | **26.6m (3/5)** | **22.9m (3/5)** |
| Boulder V7d (4999) | 49.5m (5/5) | 30.6m (4/5) | 20.8m (3/5) | 22.0m (3/5) |
| Stair V7b (1500) | — | — | — | 21.5m (3/5) |
| Flat Master v3 (3700) | 43.4m (5/5) | — | — | — |

### 6.2 Speed Comparison (Boulder V6 vs V7d)

| Environment | V6 Speed | V7d Speed | Ratio |
|-------------|----------|-----------|-------|
| Friction | 0.99 m/s | 0.26 m/s | 3.8x faster |
| Grass | 0.35 m/s | 0.10 m/s | 3.5x faster |
| Boulder | 0.31 m/s | 0.27 m/s | 1.1x faster |
| Stairs | 0.31 m/s | 0.07 m/s | 4.4x faster |

Boulder V6 is the dominant policy on both distance and speed metrics.

## 7. Teammate Locomotion Work

### 7.1 Ryan's Mason Hybrid Baseline

Ryan developed the Mason Hybrid baseline policy (`mason_hybrid_best_33200.pt`) which served as a foundation for multiple team members:
- Used by Colby as the frozen locomotion backbone in his CombinedPolicyTraining navigation system
- The [512, 256, 128] architecture and conservative weight configuration established the standard for subsequent training

### 7.2 Colby

Colby did not develop standalone locomotion policies. His navigation work (CombinedPolicyTraining) uses Ryan's `mason_hybrid_best_33200.pt` as a frozen locomotion backbone, with the navigation policy outputting velocity commands that the frozen loco policy converts to joint actions.

### 7.3 Cole

Cole did not develop standalone locomotion policies. His navigation system (RL_FOLDER_VS2/VS3) uses a `SpotFlatTerrainPolicy` as the frozen locomotion backbone, following the same hierarchical approach where high-level velocity commands are executed by a pre-trained walking controller.

## 8. Key Lessons Learned

1. **Smaller networks work.** The [512, 256, 128] architecture matched or exceeded the [1024, 512, 256] network on all metrics while training faster and being more stable.

2. **0% flip rate is achievable.** Mason's conservative weight philosophy (high gait weight, moderate penalties) produces gaits that never flip, even on difficult terrain.

3. **Best checkpoint != final checkpoint.** Training regression is common — periodic evaluation is essential. Boulder V6's best was at iter 4,500 out of a run that crashed at 4,589.

4. **Specialize then generalize.** Training on 60% obstacle terrain (Trial 12b) produced the best boulder results. The specialize-then-refine strategy works.

5. **The 22m stair wall is real.** Despite 9 stair-specific training runs with different approaches, no policy has broken past zone 3 (13cm+ steps). This likely requires a fundamentally different locomotion behavior.

6. **Speed matters as much as distance.** Boulder V7d went further on grass (+9.4m) but was 3.5x slower — a worse policy when time is considered.

7. **Domain randomization is essential.** S2R-hardened policies (with action delay, sensor noise, mass randomization) maintain strong performance while being deployable to real hardware.
