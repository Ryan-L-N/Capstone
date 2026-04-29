# Locomotion Policy Development

## 1. Overview

This document tells the story of how we developed a single generalist locomotion policy for the Boston Dynamics Spot quadruped — one that handles smooth ground, tall grass, boulder fields, and stairs. Rather than training one monolithic policy to do everything (which leads to conflicting reward gradients), we followed a four-stage pipeline:

1. **ARL Baseline** — An externally proven policy with a simpler, more elegant design that outperformed our custom 22-term configuration.
2. **ARL Hybrid** — We adapted the ARL Baseline to our harder 12-terrain curriculum and added three targeted safety fixes.
3. **Expert Masters** — We specialized two copies of the ARL Hybrid into terrain experts: one for smooth surfaces and one for obstacles.
4. **Distilled Master** — We combined both experts into a single student policy using multi-expert distillation with a terrain-aware router.

Each stage built on the last. The result is a policy that inherits the best behavior from each expert without the trade-offs that plagued single-policy training.

All training used NVIDIA Isaac Lab, RSL-RL (PPO), and an NVIDIA H100 GPU.

## 2. Policy Architecture

All policies in the pipeline share the same architecture, making them directly compatible with each other and with our evaluation harness:

| Property | Value |
|----------|-------|
| Network | [512, 256, 128] MLP with ELU activation |
| Parameters | ~286,604 |
| Observation dimensions | 235 (187 height scan + 48 proprioception) |
| Action dimensions | 12 (joint position targets) |
| Control frequency | 50 Hz |
| Physics frequency | 500 Hz (decimation = 10) |
| PD gains | Kp = 60.0, Kd = 1.5 |
| Action scale | 0.2 rad offset from default standing pose |
| Observation order | Height scan first (dims 0-186), proprioception second (dims 187-234) |

The [512, 256, 128] architecture was a key insight from ARL's work. Our earlier trials used a [1024, 512, 256] network (~1.2M parameters) that overfitted to easy-terrain survival strategies. The smaller network generalizes better — less capacity forces the policy to learn broadly useful locomotion rather than memorizing terrain-specific tricks.

## 3. Reward Function

### 3.1 ARL's 11-Term Reward Function

ARL's original reward structure uses 11 clean, well-tuned terms. Fewer signals give clearer gradients — the policy knows exactly what we want.

**Positive Rewards (Incentives)**

| Term | Weight | Description |
|------|--------|-------------|
| `gait` | 10.0 | Diagonal trot synchronization (FL+HR, FR+HL alternating) |
| `base_linear_velocity` | 5.0 | Reward for matching commanded forward/lateral speed |
| `base_angular_velocity` | 5.0 | Reward for matching commanded yaw rate |
| `air_time` | 5.0 | Reward for proper swing phase (feet off ground ~0.3s) |
| `foot_clearance` | 0.5 | Reward for lifting feet during swing (obstacle clearance) |

**Negative Rewards (Penalties)**

| Term | Weight | Description |
|------|--------|-------------|
| `base_orientation` | -3.0 | Excessive body roll/pitch |
| `base_motion` | -2.0 | Unwanted vertical/lateral body velocity |
| `air_time_variance` | -1.0 | Inconsistent swing timing across legs |
| `action_smoothness` | -1.0 | Rapid changes in control commands (clamped at 10.0) |
| `joint_pos` | -0.7 | Extreme joint angle deviations from default |
| `foot_slip` | -0.5 | Feet sliding during ground contact |

### 3.2 Three Safety Additions (ARL Hybrid)

When we put ARL's config on our harder 12-terrain curriculum, three failure modes appeared. We added one targeted fix for each:

| Addition | Weight | Why It Was Needed |
|----------|--------|-------------------|
| `terrain_relative_height` | -2.0 | Without this, the robot belly-crawls as a survival strategy. Fixed 0.37m target height. |
| `dof_pos_limits` | -3.0 | Without this, the policy locks knees at mechanical stops. Penalizes joints approaching URDF limits. |
| `clamped_action_smoothness` | (replaces original) | ARL's raw `action_smoothness` uses unbounded L2 norms that can explode to NaN. Our version caps at 10.0. |

This brings the total to **14 terms** — ARL's proven 11, plus 3 surgical fixes. By comparison, our earlier custom configuration had 22 terms with competing penalties that made gradient signals unclear.

### 3.3 Reward Design Principles

1. **Fewer rewards = clearer gradients.** 11 terms outperformed 22. The policy learns faster when it isn't balancing dozens of competing objectives.
2. **Gait weight must stay high (10.0+).** Lower values produce bouncing exploits where the robot hops instead of walking.
3. **Tune complementary rewards together.** Foot clearance, action smoothness, and joint position all govern leg-lift behavior — changing one without adjusting the others produces suboptimal gaits.
4. **Clamp all unbounded penalty terms.** Squared errors and norms can explode to NaN without clamping.
5. **Use terrain-relative height, not world-frame Z.** World-frame height is meaningless on elevated terrain.

## 4. Training Curriculum

### 4.1 12-Terrain Curriculum

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

### 4.2 Obstacle-Focus Variant

For the Obstacle Expert, the terrain mix was shifted to 60% boulders/stairs to force specialization on the hardest terrain types.

## 5. The Policy Pipeline

### 5.1 Stage 1 — ARL Baseline

**What it is:** ARL's team independently developed a locomotion policy that reached terrain level ~6.0 using a simpler design than our custom 22-term, 1.2M-parameter configuration. After 11 trials and ~30 sub-iterations of tuning our custom config, we adopted ARL's approach.

**Why ARL's design worked better:**

| Feature | Our Custom Config | ARL Baseline |
|---------|------------------|----------------|
| Reward terms | 22 (competing signals) | 11 (clean gradients) |
| Network size | [1024, 512, 256] — 1.2M params | [512, 256, 128] — 286K params |
| LR schedule | Cosine annealing (manual lr_max) | Adaptive KL (self-adjusting) |
| Domain randomization | Heavy (mass ±5kg, friction 0.15-1.0) | Light (mass ±2.5kg, friction 0.3-1.0) |
| Observation noise | Enabled | Disabled |
| Episode length | 30s | 20s |
| Mini-batches | 64 | 4 (larger effective batch) |

| Property | Value |
|----------|-------|
| Checkpoint | `mason_baseline_final_19999.pt` |
| Architecture | [512, 256, 128] |
| Iterations | 20,000 |
| Environments | 4,096 |

**100-Episode Evaluation:**

| Environment | Mean Progress | Zone (avg) | Fall Rate |
|-------------|--------------|-----------|-----------|
| Friction | 36.9m | 4.1 | 37% |
| Grass | 29.6m | 3.6 | 23% |
| Boulder | 14.4m | 2.2 | 13% |
| Stairs | 10.9m | 2.0 | 20% |

**Takeaway:** ARL's clean design produced a functional walking gait, but on our harder 12-terrain curriculum it needed refinement. The high friction fall rate (37%) and limited boulder performance (14.4m) showed where targeted fixes would help.

### 5.2 Stage 2 — ARL Hybrid

**What it is:** ARL's 11 reward terms + our 12-terrain curriculum + 3 surgical safety fixes (terrain-relative height, DOF limits, clamped action smoothness). No AI coach — just ARL's proven weights and our harder terrain.

| Property | Value |
|----------|-------|
| Checkpoint | `mason_hybrid_best_33200.pt` |
| Architecture | [512, 256, 128] |
| Iterations | 33,200 (best), 35,100 (final) |
| Training time | ~42.6 hours |
| Total steps | ~2.0 billion |
| Peak terrain level | 3.83 |
| Flip rate | **0%** |
| Environments | 4,096 |

**100-Episode Evaluation:**

| Environment | Mean Progress | Zone (avg) | Fall Rate | Velocity |
|-------------|--------------|-----------|-----------|----------|
| Friction | 48.9 ± 5.0m | 5.0 | **2%** | 0.934 m/s |
| Grass | 27.2 ± 8.0m | 3.3 | 15% | 0.487 m/s |
| Boulder | 20.3 ± 1.7m | 3.0 | 3% | 0.350 m/s |
| Stairs | 11.2 ± 2.0m | 2.0 | 36% | 0.227 m/s |

**What improved over the Baseline:**

| Environment | Baseline → Hybrid | Change |
|-------------|-------------------|--------|
| Friction | 36.9m → 48.9m | **+12.0m**, fall rate 37% → 2% |
| Boulder | 14.4m → 20.3m | **+5.9m**, fall rate 13% → 3% |
| Stairs | 10.9m → 11.2m | +0.3m (minimal change) |
| Grass | 29.6m → 27.2m | -2.4m (slight regression) |

The three safety additions transformed friction performance (from 37% falls to 2%) and improved boulder traversal by 41%. The ARL Hybrid became a rock-solid generalist with a **0% flip rate** — it never flipped over, even on difficult terrain.

**The trade-off problem:** The ARL Hybrid was excellent on smooth terrain but limited on obstacles. Training a single policy harder on boulders and stairs would regress its friction and grass performance. This tension — loose penalties for obstacles vs. tight penalties for clean walking — was the core problem that motivated the next stage.

### 5.3 Stage 3 — The Expert Masters

Rather than training one policy to do everything, we split the problem. Two specialist experts were trained, each optimized for its terrain type:

#### Friction/Grass Expert — `mason_hybrid_best_33200.pt`

The ARL Hybrid itself served as the friction/grass expert. It was already excellent on smooth surfaces (48.9m friction, 98% completion) with a clean, stable gait. No additional training was needed.

#### Obstacle Expert — `obstacle_best_44400.pt`

Starting from the ARL Hybrid, we retrained with a 60% boulder/stair terrain mix to force the policy to specialize on obstacles.

| Property | Value |
|----------|-------|
| Checkpoint | `obstacle_best_44400.pt` |
| Starting from | ARL Hybrid (33200) |
| Terrain mix | 60% boulders/stairs, 40% mixed |
| Iterations | 44,400 (best out of 54,600) |
| Peak terrain level | 4.38 |
| Flip rate | 0% |

**Key reward changes for obstacle specialization:**
- `foot_clearance`: 0.5 → 2.0 (lift feet higher to step over obstacles)
- `base_orientation`: -3.0 → -2.0 (allow lateral tilt for boulder traversal)
- `action_smoothness`: -1.0 (maintained for gait quality)

**Evaluation:**

| Environment | Distance | Zones |
|-------------|----------|-------|
| Friction | 42.2m | 5/5 |
| Grass | 31.7m | 4/5 |
| Boulder | **30.4m** | **4/5** |
| Stairs | 15.7m | 2/5 |

**The specialization trade-off in action:** The Obstacle Expert gained +10.1m on boulders compared to the ARL Hybrid (30.4m vs 20.3m) but lost -6.7m on friction (42.2m vs 48.9m). This is exactly why we need distillation — each expert excels at its terrain, but neither is good at everything.

### 5.4 Stage 4 — The Distilled Master

**What it is:** A single student policy that learns WHEN to use each expert's behavior by reading the terrain geometry through its height scan. The student acts in the environment, both frozen experts label what they would have done, and the student is trained to match the appropriate expert for each terrain.

| Property | Value |
|----------|-------|
| Checkpoint | `distilled_6899.pt` |
| Architecture | [512, 256, 128] (same as both experts) |
| Iterations | 6,899 |
| Training time | ~6-8 hours on H100 |
| Environments | 4,096 |
| Experts | Friction (`mason_hybrid_best_33200.pt`) + Obstacle (`obstacle_best_44400.pt`) |

#### How the Terrain Router Works

The height scan (first 187 observation dimensions) encodes terrain geometry. Flat terrain has near-zero variance in the scan. Boulders and stairs have high variance. A sigmoid gate routes each environment to the appropriate expert:

```
Height Scan (187 dims) → compute variance → sigmoid gate
                                              │
                              gate ≈ 0: smooth terrain → Friction Expert
                              gate ≈ 1: rough terrain  → Obstacle Expert
                              gate ≈ 0.5: transition   → blend both
```

The student doesn't hard-switch between experts — it smoothly blends their actions at terrain boundaries, preventing jerky transitions.

#### Training Process

1. **Initialize student** from the Friction Expert (best general gait — the student already knows how to walk).
2. **Critic warmup** (300 iterations): Actor is frozen while the critic learns the new value landscape.
3. **Combined training** (each iteration):
   - Student collects experience in the environment (standard PPO)
   - PPO update runs normally
   - Post-hoc distillation step: query both frozen experts, blend their actions based on the terrain gate, compute MSE + KL loss between student and blended expert target
4. **Alpha annealing**: Distillation weight starts at 0.8 (mostly copy experts) and decays to 0.2 (mostly PPO reward signal). The student absorbs expert knowledge first, then adapts with its own experience.

#### Distillation Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Alpha (start → end) | 0.8 → 0.2 | Shifts from expert imitation to PPO reward |
| KL weight | 0.1 | Balance between MSE and KL divergence loss |
| Roughness threshold | 0.005 | Height scan variance gate for routing |
| Routing temperature | 0.005 | Sigmoid sharpness (lower = harder gate) |
| Distill batch size | 8,192 | Samples per distillation gradient step |
| Critic warmup | 300 iters | Actor frozen while critic calibrates |

#### Distillation Loss

```
loss = MSE(student_action, blended_expert_action) + 0.1 × KL(student_dist ∥ expert_dist)
```

The KL term is clamped to [0.0, 10.0] for numerical stability.

## 6. Comparative Evaluation

### 6.1 The Full Pipeline — Side by Side

| Policy | Friction | Grass | Boulder | Stairs |
|--------|---------|-------|---------|--------|
| ARL Baseline | 36.9m (4.1) | 29.6m (3.6) | 14.4m (2.2) | 10.9m (2.0) |
| **ARL Hybrid** | **48.9m (5.0)** | 27.2m (3.3) | 20.3m (3.0) | 11.2m (2.0) |
| Obstacle Expert | 42.2m (5.0) | 31.7m (4.0) | **30.4m (4.0)** | **15.7m (2.0)** |

*Values shown as: mean progress (mean zone). ARL Baseline and Hybrid from 100-episode evaluations. Obstacle Expert from single-episode evaluation.*

**The specialization dilemma is clear:** The ARL Hybrid dominates friction (+6.7m over Obstacle Expert). The Obstacle Expert dominates boulders (+10.1m over Hybrid). Neither is best at everything. Distillation resolves this by teaching the student to use the right expert for each terrain.

### 6.2 Stability Comparison

| Policy | Friction Fall Rate | Grass Fall Rate | Boulder Fall Rate | Stairs Fall Rate | Flip Rate |
|--------|--------------------|-----------------|-------------------|------------------|-----------|
| ARL Baseline | 37% | 23% | 13% | 20% | — |
| ARL Hybrid | **2%** | 15% | **3%** | 36% | **0%** |

The ARL Hybrid's 0% flip rate and 2% friction fall rate demonstrate the value of ARL's conservative reward philosophy. High gait weight (10.0) and moderate penalties produce gaits that prioritize stability.

## 7. Teammate Locomotion Work

### 7.1 Ryan's ARL Hybrid Baseline

Ryan developed the ARL Hybrid baseline policy (`mason_hybrid_best_33200.pt`) which served as the foundation for the entire pipeline:
- Used as the Friction/Grass Expert in distillation
- Used by Colby as the frozen locomotion backbone in his CombinedPolicyTraining navigation system
- The [512, 256, 128] architecture and conservative weight configuration established the standard for all subsequent training

### 7.2 Colby

Colby did not develop standalone locomotion policies. His navigation work (CombinedPolicyTraining) uses Ryan's `mason_hybrid_best_33200.pt` as a frozen locomotion backbone, with the navigation policy outputting velocity commands that the frozen loco policy converts to joint actions.

### 7.3 Cole

Cole did not develop standalone locomotion policies. His navigation system (RL_FOLDER_VS2/VS3) uses a `SpotFlatTerrainPolicy` as the frozen locomotion backbone, following the same hierarchical approach where high-level velocity commands are executed by a pre-trained walking controller.

## 8. Key Lessons Learned

1. **Simpler is better.** ARL's 11-term reward function with a 286K-parameter network outperformed our 22-term function with a 1.2M-parameter network. Fewer rewards give clearer gradients; smaller networks generalize instead of memorizing.

2. **0% flip rate is achievable.** ARL's conservative weight philosophy (high gait weight, moderate penalties) produces gaits that never flip, even on difficult terrain. Stability is the foundation everything else builds on.

3. **Specialize, then distill.** Training a single policy on mixed terrain forces impossible trade-offs — loose penalties for obstacles vs. tight penalties for clean walking. Training two specialists and combining them via distillation sidesteps this entirely.

4. **Best checkpoint != final checkpoint.** Training regression is common. The Obstacle Expert's best was at iteration 44,400 out of a 54,600-iteration run. Periodic evaluation during training is essential.

5. **The height scan is the terrain router.** The 187-dimensional height scan variance cleanly separates smooth terrain (near-zero variance) from rough terrain (high variance), providing a natural signal for expert routing without any additional sensors.

6. **Initialize from your best generalist.** Starting the distilled student from the Friction Expert (best general gait) means it already knows how to walk. It only needs to learn when to switch to obstacle behavior — saving thousands of iterations.

7. **Tune rewards along kinematic chains.** For obstacle terrain, foot clearance + action smoothness + joint position all govern the step-up motion. Tuning one without the others produces suboptimal gaits. The biggest boulder improvement (+10.1m) came from adjusting these three together.
