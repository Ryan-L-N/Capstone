# Training Methodology: Boston Dynamics Spot Quadruped Locomotion Policy

## AI2C Tech Capstone — MS for Autonomy, Carnegie Mellon University, February 2026

---

## 1. Introduction

This document describes the complete training methodology for a robust
locomotion policy for the Boston Dynamics Spot quadruped robot. The policy is
trained entirely in simulation using NVIDIA Isaac Sim and Isaac Lab, with the
goal of sim-to-real transfer to physical hardware.

The final approach — **Hybrid Student-Teacher Reinforcement Learning** — emerged
from two prior training attempts: a successful but limited 48-hour baseline, and
a failed 100-hour from-scratch run. Each attempt informed the design decisions
for the next. The document covers the full journey, the curriculum choices, the
reward shaping rationale, and the literature motivating each decision.

---

## 2. Background: Reinforcement Learning for Legged Locomotion

Modern legged locomotion policies are trained using **Proximal Policy
Optimization (PPO)** in massively parallel GPU-based simulators. The robot
learns by trial and error across thousands of simultaneous environments, each
presenting different terrain configurations and physical disturbances.

Key concepts used throughout:

- **Domain Randomization (DR):** Randomly varying physics parameters
  (friction, mass, external forces) during training so the policy generalizes
  to real-world uncertainty (Tobin et al., 2017; Peng et al., 2018).
- **Terrain Curriculum:** Automatically promoting robots to harder terrain
  when they succeed and demoting them when they fail, keeping training at
  the learning frontier (Rudin et al., 2022).
- **Teacher-Student Distillation:** Training a privileged "teacher" with
  access to ground-truth terrain information, then distilling its behavior
  into a "student" that operates from onboard sensors only (Lee et al.,
  2020; Cheng et al., 2024).

---

## 3. Phase 1: The 48-Hour Rough Terrain Baseline

### 3.1 Overview

The first training run produced a working locomotion policy for Spot on 6
standard terrain types from NVIDIA's reference configuration. This became the
foundation for all subsequent work.

**Hardware:** NVIDIA H100 NVL 96GB HBM3
**Duration:** ~48 hours (27,500 iterations)
**Result:** A functional policy that walks, trots, follows velocity commands,
and handles moderate rough terrain.

### 3.2 Architecture

The policy uses a standard actor-critic architecture from RSL-RL (Rudin et
al., 2022):

```
Actor:  235 → [512] → [256] → [128] → 12    (ELU activation)
Critic: 235 → [512] → [256] → [128] → 1     (ELU activation)
```

**Observation space (235 dimensions):**

| Component | Dims | Description |
|-----------|------|-------------|
| Base linear velocity | 3 | Body-frame linear velocity (x, y, z) |
| Base angular velocity | 3 | Body-frame angular velocity (roll, pitch, yaw) |
| Projected gravity | 3 | Gravity vector projected into body frame |
| Velocity commands | 3 | Target linear (x, y) and angular (yaw) velocity |
| Joint positions | 12 | Relative to default stance (12 joints: 3 per leg) |
| Joint velocities | 12 | Relative joint velocities |
| Last action | 12 | Previous control output (for temporal context) |
| Height scan | 187 | 17×11 grid of terrain heights, 0.1m resolution |

The height scan provides a 1.6m × 1.0m footprint of terrain height
measurements around the robot, giving the policy local terrain awareness
without explicit terrain classification.

**Action space (12 dimensions):**
Joint position targets for all 12 Spot joints, scaled by 0.25 and added to
the default standing pose. The physics engine enforces these targets through
PD position controllers (Kp=60, Kd=1.5) at 500 Hz, with the policy running
at 50 Hz (decimation=10).

### 3.3 Terrains

The baseline used Isaac Lab's `ROUGH_TERRAINS_CFG` — 6 terrain types on a
10×20 grid (200 patches):

1. Pyramid stairs (ascending)
2. Pyramid stairs (descending)
3. Random grid boxes
4. Random rough heightfield
5. Pyramid slopes (ascending)
6. Pyramid slopes (descending)

### 3.4 Reward Structure

The 48hr policy used 14 reward terms balancing task completion against
movement efficiency:

**Task rewards (positive):** Incentivize forward progress, velocity tracking,
proper gait timing, and foot clearance over obstacles.

| Term | Weight | Purpose |
|------|--------|---------|
| `base_linear_velocity` | +7.0 | Track commanded forward/lateral speed |
| `base_angular_velocity` | +5.0 | Track commanded yaw rate |
| `gait` | +10.0 | Enforce diagonal trot pattern (FL-HR, FR-HL synchronization) |
| `air_time` | +5.0 | Reward proper swing/stance phase timing |
| `foot_clearance` | +2.5 | Lift feet high enough to clear obstacles |

**Efficiency penalties (negative):** Prevent the policy from finding degenerate
solutions — shaking, energy-wasting gaits, excessive joint stress.

| Term | Weight | Purpose |
|------|--------|---------|
| `base_orientation` | -5.0 | Stay upright (penalize roll/pitch) |
| `base_motion` | -3.0 | Minimize vertical bounce and lateral drift |
| `action_smoothness` | -2.0 | Smooth joint commands (no jitter) |
| `foot_slip` | -1.0 | Penalize feet sliding on ground |
| `joint_pos` | -1.0 | Stay near default stance when stopped |
| `joint_acc` | -5e-4 | Smooth joint trajectories |
| `joint_torques` | -2e-3 | Minimize energy consumption |
| `joint_vel` | -2e-2 | Controlled, deliberate movements |
| `air_time_variance` | -1.0 | Symmetric gait (equal swing times per leg) |

The gait reward (weight 10.0) is the heaviest single term. It enforces a
**trot gait** where diagonal leg pairs move in synchronization — this is
Spot's natural and most stable gait pattern. Without this reward, policies
tend to discover degenerate strategies like dragging feet or hopping
(Margolis et al., 2022).

### 3.5 Domain Randomization

The 48hr run applied moderate DR at startup:

| Parameter | Range | Purpose |
|-----------|-------|---------|
| Static friction | [0.5, 1.25] | Surface grip variation |
| Dynamic friction | [0.4, 1.0] | Sliding resistance variation |
| Base mass offset | ±5.0 kg | Payload/battery variation |
| External force | ±3.0 N | Wind/disturbance robustness |
| External torque | ±1.0 Nm | Rotational disturbance |
| Push velocity | ±0.5 m/s | Impulse recovery (every 10-15s) |

**Key limitation:** Friction was randomized only at simulation startup (once per
environment lifetime), not at each episode reset. This meant each robot
experienced only one friction value for its entire training trajectory.

### 3.6 PPO Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 3e-4 | Standard PPO rate for locomotion (Schulman et al., 2017) |
| LR schedule | Adaptive KL | Automatically adjusts LR to maintain KL ≈ 0.01 |
| Entropy coef | 0.008 | Moderate exploration for terrain diversity |
| Num envs | 8,192 | H100 throughput optimal |
| Steps/env | 24 | ~0.48s of experience per collection |
| Mini-batches | 8 | 24,576 samples per mini-batch |
| Learning epochs | 5 | Number of PPO passes per iteration |
| Clip param | 0.2 | Standard PPO clipping |
| Gamma | 0.99 | Discount factor (values long-term survival) |
| Lambda | 0.95 | GAE advantage estimation |
| Episode length | 20s | Maximum time before forced reset |

### 3.7 Results

After 27,500 iterations (~48 hours), the policy achieved:

- **Flat terrain:** Near-perfect locomotion, velocity tracking within 5%
- **Rough terrain (6 types):** Functional traversal of moderate obstacles
- **Stairs:** Ascends/descends up to ~15cm step height
- **Terrain curriculum level:** ~4-5 (moderate difficulty)

**Weaknesses identified in evaluation:**

| Environment | Fall Rate | Distance | Issue |
|-------------|-----------|----------|-------|
| Friction (μ 1.0→0.2) | 69% | 26.1m | Cannot handle low friction |
| Grass (drag 0→20 N·s/m) | 19% | 24.1m | Slows but mostly copes |
| Boulder (rough + boxes) | 70% | 13.4m | Falls on complex obstacles |
| Stairs (5-25cm steps) | 14% | 11.3m | Slow but stable |

The policy was never trained on friction below 0.5 or vegetation drag, so
these weaknesses were expected.

---

## 4. Phase 2: The Failed 100-Hour From-Scratch Attempt

### 4.1 Motivation

To address the 48hr policy's weaknesses, we designed a more ambitious training
run with 12 terrain types, aggressive DR, and a larger network. The goal was to
train a single policy that handles all terrain types from scratch.

### 4.2 What Changed

| Decision | 48hr Run | 100hr Run | Rationale |
|----------|----------|-----------|-----------|
| Architecture | [512, 256, 128] | [1024, 512, 256] | Bigger network for harder task |
| Terrain types | 6 | 12 | Added stairs-down, stepping stones, gaps, wave, friction/vegetation planes |
| Num envs | 8,192 | 65,536 | Fill H100 VRAM |
| Grid size | 10×20 (200 patches) | 10×40 (400 patches) | More terrain variety |
| Friction range | [0.5, 1.25] | [0.05, 1.5] | Train for icy surfaces |
| Push velocity | ±0.5 m/s | ±1.5 m/s | Stronger disturbances |
| Episode length | 20s | 30s | Longer episodes for hard terrain |
| Initialization | Random | Random | No checkpoint — from scratch |

### 4.3 Why It Failed

After 10,000 iterations (~17.6 billion timesteps, ~100 hours of wall time):

- **Terrain level:** Stuck at 0.0 (easiest terrain, never promoted)
- **Body contact termination:** 100% (every episode ended by falling)
- **Episode length:** ~7 seconds (learned to stand, not walk)
- **Mean reward:** Oscillating between 1.1-1.3 (no upward trend)

**Root cause analysis:**

1. **Architecture mismatch:** The [1024, 512, 256] network could not load the
   working [512, 256, 128] checkpoint because RSL-RL uses `strict=True` in
   `load_state_dict()`. Training started from random weights.

2. **Maximum difficulty from iteration 0:** With friction as low as 0.05 (oil
   on polished steel), even a policy that discovers standing is immediately
   knocked over. The gradient signal from robots on friction 0.05 contradicts
   the gradient from robots on friction 1.5 — the network receives incoherent
   learning signals.

3. **No progressive curriculum for DR:** The terrain curriculum (automatic
   promotion/demotion) worked, but DR was fixed at maximum ranges. Even on
   the easiest terrain (level 0), robots experienced extreme friction values
   and aggressive pushes. The terrain curriculum couldn't advance because the
   policy kept falling due to DR, not terrain geometry.

4. **Contradictory gradients:** In a single PPO batch, some robots needed
   slow, cautious movements (low friction) while others benefited from
   aggressive strides (high friction). The averaged gradient was a compromise
   that suited neither condition — the policy converged to a minimal-movement
   strategy (standing still).

5. **Too many simultaneous challenges:** Learning to walk, handle 12 terrain
   types, resist aggressive pushes, and adapt to extreme friction ranges — all
   at once from random initialization — exceeded the learning capacity of the
   system. Each challenge is learnable individually, but the combinatorial
   difficulty was overwhelming.

This failure mode is well-documented in the curriculum learning literature.
Narvekar et al. (2020) show that presenting all tasks simultaneously leads
to catastrophic interference, while sequential or progressive presentation
enables incremental skill acquisition. Our failure is a direct instance of
this phenomenon applied to domain randomization.

---

## 5. Phase 3: Hybrid Student-Teacher Reinforcement Learning

### 5.1 Design Philosophy

The hybrid approach draws on three key insights from the literature:

**Insight 1 — Start from what works (transfer learning):**
Rather than training from scratch, initialize from the working 48hr policy
and build on its existing locomotion competence. This is standard in deep
learning (Zhuang et al., 2020) but often overlooked in RL for robotics
where researchers default to tabula rasa training.

**Insight 2 — Progressive difficulty (curriculum learning):**
The CMU Extreme Parkour paper (Cheng et al., 2024) demonstrates that even
with a capable initialization, the environment difficulty must match the
policy's current ability. They use automatic terrain curriculum with
promotion/demotion thresholds. We extend this concept to domain
randomization itself — not just terrain geometry, but friction, forces,
and mass perturbations all start easy and progressively harden.

**Insight 3 — Privileged information enables faster learning (teacher-student):**
The ETH Zurich ANYmal Parkour work (Hoeller et al., 2024) and the CMU
Extreme Parkour paper both show that a teacher with access to ground-truth
terrain properties (friction, terrain type, contact forces) learns
significantly faster than a student operating from noisy sensor data alone.
The teacher's behavior can then be distilled into a deployable student,
transferring implicit terrain understanding.

### 5.2 Stage 1: Progressive Fine-Tuning

#### 5.2.1 Warm Start

The training initializes from `model_27500.pt` — the 48hr rough policy
checkpoint. This gives us:

- A policy that already walks with a stable diagonal trot
- Terrain awareness from the height scan (187 dims)
- Reasonable joint coordination and energy efficiency
- A starting terrain curriculum level of ~3-4 (not 0)

The optimizer state is **not** loaded (`load_optimizer=False`). The 48hr
training used LR=3e-4 with adaptive KL; our fine-tuning uses LR=1e-4.
Loading the old optimizer's momentum terms would cause the new LR to be
effectively overridden by stale gradient statistics, potentially causing
destructive updates in the first few hundred iterations (Li et al., 2019).

The action noise standard deviation is set to 0.65, matching the
checkpoint's converged value. The 48hr run started at 0.8 and naturally
decayed. Resetting to 0.8 would add unnecessary exploration noise to an
already-competent policy, potentially causing temporary performance
regression.

#### 5.2.2 Progressive Domain Randomization

This is the central innovation. Instead of applying full DR from iteration 0,
each DR parameter linearly interpolates from a "safe" starting value to its
final target over 15,000 iterations (60% of training):

| Parameter | Start (iter 0) | End (iter 15K) | 48hr Reference |
|-----------|---------------|----------------|----------------|
| Static friction | [0.3, 1.3] | [0.1, 1.5] | [0.5, 1.25] |
| Dynamic friction | [0.25, 1.1] | [0.08, 1.2] | [0.4, 1.0] |
| Push velocity | ±0.5 m/s | ±1.0 m/s | ±0.5 m/s |
| Push interval | [10, 15]s | [6, 13]s | [10, 15]s |
| External force | ±3.0 N | ±6.0 N | ±3.0 N |
| External torque | ±1.0 Nm | ±2.5 Nm | ±1.0 Nm |
| Mass offset | ±5.0 kg | ±7.0 kg | ±5.0 kg |

The starting values are chosen to be near the 48hr training conditions,
ensuring the checkpoint's existing competence is not immediately overwhelmed.
The end values represent the full target difficulty — harder than the failed
100hr run's friction range but with external forces capped at more reasonable
levels (±6.0 N vs the 100hr's ±8.0 N).

**Implementation detail:** The `physics_material` and `add_base_mass`
randomization events are changed from `mode="startup"` (randomized once at
simulation start) to `mode="reset"` (re-randomized at each episode reset).
This is essential for progressive DR — as the schedule expands the friction
range, new episodes sample from the expanded range while in-progress
episodes continue with their existing values. Without this change, expanding
the range would have no effect until the entire simulation restarted.

The progressive schedule is implemented by monkey-patching the RSL-RL
`runner.alg.update()` method. Before each PPO update, the DR parameters in
the environment config are modified in-place based on the current iteration.
This pattern is lightweight and avoids modifying the RSL-RL source code.

#### 5.2.3 Terrain Configuration: 12 Types

The terrain grid uses 10 rows (difficulty progression) × 40 columns
(terrain variety) = 400 patches, each 8m × 8m. The 12 terrain types are
organized into three categories:

**Category A — Geometric Obstacles (40%):**

| Terrain | Proportion | Difficulty Range | Purpose |
|---------|-----------|-----------------|---------|
| Pyramid stairs up | 10% | 5-25cm step height | Ascending staircase traversal |
| Pyramid stairs down | 10% | 5-25cm step height | Descending (harder — must control momentum) |
| Random grid boxes | 10% | 5-25cm box height | Boulder/rubble proxy |
| Stepping stones | 5% | Up to 15cm height | Precise foot placement |
| Gaps | 5% | 10-50cm width | Stride planning, potential jumping |

**Category B — Surface Variation (35%):**

| Terrain | Proportion | Difficulty Range | Purpose |
|---------|-----------|-----------------|---------|
| Random rough | 10% | 2-15cm noise | General uneven ground |
| Slopes up | 7.5% | 0-0.5 slope | Uphill climbing |
| Slopes down | 7.5% | 0-0.5 slope | Downhill control |
| Wave terrain | 5% | 5-20cm amplitude | Natural undulating surfaces |
| Friction plane | 5% | Flat, friction only | Pure low-friction challenge |
| Vegetation plane | 5% | Flat, drag only | Pure vegetation drag challenge |

**Category C — Compound Challenges (25%):**

| Terrain | Proportion | Difficulty Range | Purpose |
|---------|-----------|-----------------|---------|
| HF stairs up | 10% | 5-20cm step height | Noisy/debris-covered stairs |
| Discrete obstacles | 5% | 5-30cm height | Scattered blocks |
| Repeated boxes | 5% | 5-20cm height | Regular obstacle patterns |

The friction plane and vegetation plane are intentionally flat — they isolate
the challenge of low-friction locomotion and drag resistance from terrain
geometry. This separation allows the policy to learn each skill
independently before encountering them combined on other terrain types.

The terrain curriculum (`mdp.terrain_levels_vel`) automatically promotes
robots to harder rows when they traverse >50% of expected distance and
demotes them when they fail. This self-regulating mechanism, validated by
Rudin et al. (2022) and the Scaling Rough Terrain Locomotion paper (2026),
finds the learning frontier without manual tuning.

#### 5.2.4 Reward Structure: 19 Terms

The reward function includes all 14 terms from the 48hr run plus 5
custom terms addressing specific weaknesses identified in evaluation:

**New positive reward:**

| Term | Weight | Purpose | Literature |
|------|--------|---------|-----------|
| `velocity_modulation` | +2.0 | Accept slower speeds on hard terrain | Inspired by Extreme Parkour's universal reward (Cheng et al., 2024) |

The velocity modulation reward estimates terrain difficulty from the variance
of foot contact forces. On easy terrain (low variance), the robot should
track the full commanded velocity. On hard terrain (high force variance), the
robot receives credit for maintaining reduced but non-zero speed. This
prevents two failure modes: freezing on hard terrain (which would maximize
survival but produce zero velocity reward) and charging recklessly (which
would maximize velocity reward but cause frequent falls).

**New penalties:**

| Term | Weight | Purpose | Literature |
|------|--------|---------|-----------|
| `vegetation_drag` | -0.001 | Penalize time in drag zones; also applies physical drag forces | Novel — combines physics modifier with reward |
| `body_height_tracking` | -2.0 | Maintain 0.42m standing height | Prevents unnatural crouching (Margolis et al., 2022) |
| `contact_force_smoothness` | -0.5 | Gentle foot placement (penalize GRF spikes) | Reduces slip on low-friction (Hwangbo et al., 2019) |
| `stumble` | -2.0 | Penalize shin/knee contact above 15cm height | Encourages obstacle clearance (Cheng et al., 2024) |

The **vegetation drag reward** is unique in that it functions as both a
physics modifier and a reward term. It applies velocity-dependent drag
forces (F = -c·v) to the robot's feet when they contact the ground,
simulating grass, mud, or shallow water resistance. The drag coefficient
is terrain-aware: zero on friction plane columns, always positive on
vegetation plane columns, and randomly tiered on other terrains. This dual
role (physics + reward) ensures the policy experiences realistic drag forces
during training while also receiving a gradient signal to minimize time
spent in drag-affected zones.

The **stumble penalty** fires when a foot has contact forces above 5N while
elevated above knee height (15cm). This indicates the robot is hitting the
side of an obstacle — tripping — rather than stepping over it. The penalty
encourages the policy to increase foot clearance near obstacles, directly
addressing the boulder terrain weakness.

#### 5.2.5 Fine-Tuning Hyperparameters

| Parameter | 48hr Run | Hybrid Fine-Tune | Rationale |
|-----------|---------|-------------------|-----------|
| Learning rate | 3e-4 | **1e-4** | 3× lower to prevent catastrophic forgetting (Kirkpatrick et al., 2017) |
| Desired KL | 0.01 | **0.008** | Tighter policy updates — smaller steps from a good starting point |
| Entropy coef | 0.008 | **0.005** | Less exploration needed from warm start — policy already knows how to walk |
| Init noise std | 0.8 | **0.65** | Match checkpoint's converged action noise |
| Episode length | 20s | **30s** | Longer episodes needed for harder terrains and evaluating sustained locomotion |
| Num envs | 8,192 | **16,384** | Better terrain coverage (1,365 robots per terrain type vs 683) |
| Max iterations | 27,500 | **25,000** | Fewer iterations needed due to warm start and larger batch |
| Mini-batches | 8 | 8 | 49K samples per mini-batch (up from 24K due to more envs) |

The lower learning rate is the single most important hyperparameter change.
With a warm start, large learning rates cause **catastrophic forgetting** —
the policy rapidly loses its existing walking ability in the first few
hundred iterations before slowly relearning it (French, 1999). A rate of
1e-4 combined with adaptive KL (target 0.008) ensures each update makes
small, controlled improvements without destroying existing capabilities.

#### 5.2.6 GPU Parallelism and Scaling

Isaac Lab runs physics simulation on the GPU itself (GPU PhysX). Simulating
16,384 robots costs approximately 2× the wall-clock time of 8,192 — not
16,384/8,192 = 2× proportionally, because the GPU cores process
environments in parallel. But each PPO update sees 2× more diverse
experience, producing more accurate gradients that converge in fewer
iterations.

| Envs | Steps/iter | ~Time/iter | Robots per terrain | Total steps | Wall time |
|------|-----------|------------|-------------------|-------------|-----------|
| 8,192 | 196K | ~6s | 683 | 5.9B @ 30K | ~50h |
| **16,384** | **393K** | **~10s** | **1,365** | **9.8B @ 25K** | **~69h** |
| 32,768 | 786K | ~18s | 2,730 | 11.8B @ 15K | ~75h |
| 65,536 | 1.57M | ~40s | 5,461 | 15.7B @ 10K | ~111h |

We chose 16,384 environments because:

1. **Terrain coverage:** 1,365 robots per terrain type ensures every terrain
   is well-represented in each gradient update.
2. **Total experience budget:** 9.8 billion steps — 1.9× the successful
   48hr run's 5.3B, accounting for doubling the terrain count from 6 to 12.
3. **Wall time:** ~69 hours fits the 72-hour compute budget.
4. **Diminishing returns:** Going from 8K→16K doubles per-terrain coverage
   (meaningful). Going 16K→32K adds less value but costs 80% more time.

#### 5.2.7 Stage 1 Attempt #1: Value Function Mismatch Collapse

The first deployment of Stage 1 on the H100 (February 23, 2026) failed
catastrophically within 300 iterations (~50 minutes of wall time). This
section documents the failure mode and its resolution, as the root cause
reveals a subtle but critical pitfall in transfer learning for RL.

**Timeline of collapse:**

| Iteration | Terrain Level | Body Contact % | VF Loss | Noise Std | Episode Length |
|-----------|--------------|----------------|---------|-----------|----------------|
| 0 | 3.46 | 37% | 5,298 | 0.65 | 12.1s |
| 50 | 2.15 | 48% | 890 | 0.42 | 9.8s |
| 100 | 0.81 | 72% | 412 | 0.28 | 7.2s |
| 200 | 0.05 | 95% | 285 | 0.18 | 5.1s |
| 300 | 0.00 | 100% | 198 | 0.15 | 4.8s |

The policy collapsed from a functional walking state (terrain level 3.46,
37% body contact) to complete failure (terrain level 0, 100% body contact)
in approximately 300 iterations.

**Root cause — value function reward landscape mismatch:**

The 48hr checkpoint's critic was trained to predict cumulative returns
under a 14-term reward function. Stage 1 uses a 19-term reward function
(5 additional terms: velocity modulation, vegetation drag, body height
tracking, contact force smoothness, stumble penalty). When the full
checkpoint was loaded (both actor and critic), the critic's value
predictions were calibrated for the wrong reward landscape.

The failure chain:

1. The loaded critic predicted returns based on 14 reward terms
2. Actual returns included 5 additional penalty terms, producing
   systematically lower returns than predicted
3. The GAE advantage estimator computed large negative advantages for
   walking behavior (expected high value, received lower actual return)
4. PPO interpreted walking as "worse than expected" and shifted
   probability mass toward alternative (standing) behaviors
5. The action noise standard deviation collapsed (0.65 → 0.15) as the
   policy concentrated on a narrow set of low-motion actions
6. Reduced exploration prevented recovery — the policy could no longer
   discover that walking was viable in the new reward landscape

This is a specific instance of **value function interference** in transfer
learning for RL. Unlike supervised learning where loading a pre-trained
model and fine-tuning is straightforward, RL value functions encode
task-specific reward expectations. When the reward function changes, the
value function becomes a liability rather than an asset — it provides
actively misleading training signals.

The initial VF loss of 5,298 (compared to a converged value of ~50-100)
quantifies the severity of the mismatch. The critic's predictions were
off by orders of magnitude, generating pathological advantage estimates
that systematically penalized the actor's existing locomotion behavior.

**Observation on noise collapse:** The action noise parameter (`log_std`
in RSL-RL) is a learnable parameter updated by the PPO loss. When
advantages are consistently negative for high-noise actions (exploring
diverse walking strategies) and less negative for low-noise actions
(staying near the current mean, which converges to standing still), the
gradient drives noise toward zero. This constitutes an **exploration
death spiral** — the policy requires exploration to discover that walking
works in the new reward landscape, but the value function mismatch
penalizes exploration itself.

This failure mode is analogous to the cold-start problem described by
Abel et al. (2018) in transfer RL, where stale value functions in new
environments cause performance worse than random initialization. Our
empirical evidence confirms that for PPO with GAE, loading a mismatched
critic is strictly worse than starting with a randomly initialized critic.

#### 5.2.8 Stage 1 Attempt #2: Corrective Measures

Four modifications were implemented to address the value function
mismatch collapse:

**Fix 1 — Actor-only checkpoint loading:**

Rather than loading the full checkpoint (actor + critic + optimizer),
only the actor weights and action noise parameter are loaded. The critic
is left with its random initialization, forcing it to learn the new
19-term reward landscape from scratch. Implementation extracts keys
prefixed with `actor.` and the `std` parameter from the state dict,
loads them into the actor subnetwork via `strict=True`, and explicitly
skips all `critic.` keys.

The rationale: the actor encodes locomotion behavior (how to walk),
which transfers directly regardless of reward function changes. The
critic encodes value predictions (expected cumulative return), which do
not transfer when the reward function changes. Selectively loading only
the actor preserves behavioral competence while allowing the critic to
calibrate to the new reward landscape without interference.

**Fix 2 — Critic warmup (actor freeze):**

For the first 1,000 iterations, the actor's parameters are frozen
(`requires_grad=False`). During this phase, only the critic and noise
parameters receive gradient updates. This provides the randomly
initialized critic approximately 1,000 iterations × 393,216
samples/iteration = 393 million training samples to calibrate its value
predictions before the actor begins adapting.

Without this warmup, the first actor update would be based on a randomly
initialized critic's advantage estimates — high-variance noise that could
push the actor in an arbitrary direction, potentially destroying its
existing walking ability. The warmup ensures the first meaningful actor
update is guided by a reasonably calibrated value function.

**Fix 3 — Action noise floor:**

After each PPO update, the action noise standard deviation is clamped to
a minimum of 0.4:

```python
with torch.no_grad():
    policy.std.data.clamp_(min=math.log(min_noise_std))
```

This prevents the exploration death spiral observed in Attempt #1. Even
if the value function temporarily produces pathological advantages that
would drive noise toward zero, the floor maintains sufficient
stochasticity for the policy to explore diverse walking strategies. The
value 0.4 was chosen as approximately 60% of the checkpoint's converged
noise (0.65), providing a meaningful exploration floor without excessive
random behavior.

**Fix 4 — Learning rate reset at actor unfreeze:**

During the critic warmup phase, the frozen actor produces identical
actions regardless of state. This results in KL divergence ≈ 0 between
successive policy versions. The adaptive KL schedule (Section 5.2.5)
interprets KL ≈ 0 as insufficient policy change and repeatedly doubles
the learning rate. Over 1,000 warmup iterations, this inflated the LR
from 1e-4 to approximately 0.01 — a 100× increase.

When the actor unfreezes at iteration 1,000, the first gradient update
at LR = 0.01 would be catastrophically large, immediately destroying the
walking behavior. The fix resets both `runner.alg.learning_rate` and all
optimizer parameter group learning rates to the original configured
value (1e-4) at the moment the actor is unfrozen.

This interaction between actor freezing and adaptive LR scheduling is,
to our knowledge, undocumented in the RSL-RL literature. It arises
specifically from the combination of (a) freezing a subset of parameters
while (b) using a KL-adaptive learning rate schedule — a configuration
that does not occur in standard training.

**Attempt #2 validation (smoke test, 4,096 envs, 30 iterations):**

| Metric | Attempt #1 @ iter 300 | Attempt #2 @ iter 30 |
|--------|----------------------|---------------------|
| Terrain level | 0.00 (collapsed) | 2.67 (stable) |
| Body contact % | 100% | 56% |
| VF loss | 198 (falling from 5,298) | Dropping (normal convergence) |
| Noise std | 0.15 (collapsed) | 0.69 (healthy, above floor) |
| Episode length | 4.8s | Growing |

The production run was launched with 16,384 environments at 25,000
iterations. Initial metrics at iteration 1 showed terrain level 2.82
and body contact rate 37.2%, consistent with the checkpoint's baseline
performance — confirming that the actor's locomotion ability was
preserved through the selective loading and warmup process. Training is
ongoing as of February 24, 2026.

### 5.3 Stage 2: Teacher-Student Distillation (Optional)

Stage 2 is executed only if Stage 1 evaluation shows insufficient
improvement on specific terrain types:

| Environment | Metric | Trigger for Stage 2 |
|-------------|--------|---------------------|
| Friction | Fall rate > 50% | (48hr baseline: 69%) |
| Boulder | Fall rate > 60% | (48hr baseline: 70%) |
| Stairs | Velocity < 0.2 m/s | (48hr baseline: 0.22 m/s) |

#### 5.3.1 Phase 2a: Teacher Training

The teacher receives **privileged observations** — information available in
simulation but not on the real robot:

| Privileged Observation | Dims | Purpose |
|----------------------|------|---------|
| Terrain friction coefficient | 1 | Know exact surface grip |
| Terrain type (one-hot) | 12 | Know which terrain type |
| Per-foot contact forces | 4 | Clean force magnitudes |
| Terrain slope at robot position | 2 | Know incline direction |
| **Total teacher obs** | **254** | 235 standard + 19 privileged |

The teacher's network uses the same [512, 256, 128] architecture. To load
the Stage 1 checkpoint, **weight surgery** is performed: the first layer's
weight matrix is extended from shape [512, 235] to [512, 254] by
zero-initializing 19 new input columns. All other layers are copied
directly. This preserves the policy's existing behavior while giving the
teacher capacity to learn from the new inputs.

Zero-initialization of the new columns is deliberate — it means the
teacher's initial behavior is identical to the Stage 1 policy (the new
inputs have no effect). As training progresses, the teacher gradually learns
to use the privileged information to improve its terrain handling.

This approach is directly adapted from the CMU Extreme Parkour paper (Cheng
et al., 2024), where the teacher observes exact terrain geometry ("scandots")
while the student uses depth camera images. Our adaptation replaces scandots
with terrain friction and type, which are the primary sources of uncertainty
in our evaluation environments.

#### 5.3.2 Phase 2b: Student Distillation

The student uses the standard 235-dim observation space — the deployable
policy. It learns from a combined loss:

```
loss = (1 - β) × PPO_loss + β × BC_loss
```

Where:
- **PPO_loss:** Standard proximal policy gradient with clipped surrogate
- **BC_loss:** MSE between student actions and teacher actions (detached)
- **β (BC coefficient):** Anneals from 0.8 → 0.2 over training

The annealing schedule means the student initially follows the teacher
closely (β=0.8, 80% behavior cloning) and gradually transitions to
autonomous RL refinement (β=0.2). This is critical — pure behavior cloning
(β=1.0) would limit the student to the teacher's performance ceiling, while
pure RL (β=0.0) would ignore the teacher entirely. The annealing schedule,
used in DAgger (Ross et al., 2011) and CTS (2024), gives the student a
strong initialization from the teacher while allowing RL to discover
improvements.

---

## 6. Observation Noise and Sim-to-Real Transfer

All observations are corrupted with additive uniform noise during training:

| Observation | Noise Range | Sim-to-Real Motivation |
|-------------|-------------|------------------------|
| Base linear velocity | ±0.15 m/s | IMU drift, state estimator error |
| Base angular velocity | ±0.15 rad/s | Gyroscope noise |
| Projected gravity | ±0.05 | Accelerometer noise in attitude estimation |
| Joint positions | ±0.05 rad | Encoder quantization and backlash |
| Joint velocities | ±0.5 rad/s | Numerical differentiation noise |
| Height scan | ±0.15 m | Depth sensor noise, terrain aliasing |

The height scan is additionally clipped to [-1.0, 1.0] to prevent extreme
values from distorting the policy's terrain representation.

These noise levels are approximately 50% higher than the 48hr run's values.
The increase is deliberate — the policy trained under noisier conditions
generalizes better to real-world sensor uncertainty. This follows the
observation of Peng et al. (2018) that observation noise is one of the most
effective sim-to-real transfer techniques, often more important than dynamics
randomization.

---

## 7. Physics and Control Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Physics timestep | 0.002s (500 Hz) | Sufficient for contact dynamics |
| Control decimation | 10 (50 Hz policy) | Standard for legged locomotion |
| Episode length | 30s | Long enough to evaluate sustained locomotion |
| PD gains | Kp=60, Kd=1.5 | Isaac Lab Spot defaults |
| Action scale | 0.25 | Prevents large joint jumps |
| Friction combine mode | Multiply | Robot friction × terrain friction |
| PhysX solver | 4 position iters | GPU-optimized settings |
| GPU collision stack | 2^30 (1 GB) | Sized for 16K envs with complex terrain |

The 50 Hz control rate is standard across legged locomotion literature
(Hwangbo et al., 2019; Rudin et al., 2022; Cheng et al., 2024). Higher
rates would increase computational cost without meaningful improvement in
locomotion quality. Lower rates make the policy too slow to react to terrain
disturbances.

---

## 8. Termination Conditions

| Condition | Type | Purpose |
|-----------|------|---------|
| `time_out` | Timeout (30s) | End episode after maximum duration |
| `body_contact` | Fatal | Body or leg contacts ground (threshold: 1.0 N) |
| `terrain_out_of_bounds` | Timeout | Robot wanders off terrain grid |

The `body_contact` termination is the primary learning signal for stability.
When the robot's body or upper legs touch the ground, the episode
immediately ends with a truncated return. This creates strong selection
pressure for policies that maintain balance — robots that fall lose all
future reward for that episode.

---

## 9. Evaluation Protocol

Performance is measured on four purpose-built evaluation environments from
the `4_env_test/` suite, each isolating a specific real-world challenge:

| Environment | Challenge | Evaluation |
|-------------|-----------|------------|
| **Friction** | Flat terrain, friction decreasing from 1.0 to 0.2 | 200 episodes, measure fall rate and distance |
| **Grass** | Flat terrain, vegetation drag 0 to 20 N·s/m | 200 episodes, measure fall rate and distance |
| **Boulder** | Random rough + boxes, increasing difficulty | 200 episodes, measure fall rate and distance |
| **Stairs** | Step height 5cm to 25cm | 200 episodes, measure velocity and fall rate |

Each environment tests a different axis of robustness. A policy that
performs well on all four demonstrates generalized terrain competence, not
just specialization on training terrain types.

---

## 10. References

1. **Cheng, X., Shi, K., Agarwal, A., & Pathak, D.** (2024). Extreme
   Parkour with Legged Robots. *ICRA 2024*. arXiv: 2309.14341.
   https://extreme-parkour.github.io/

2. **Hoeller, D., et al.** (2024). ANYmal Parkour: Learning Agile
   Navigation for Quadrupedal Robots. *Science Robotics*.
   DOI: 10.1126/scirobotics.adi7566.

3. **Rudin, N., Hoeller, D., Reist, P., & Hutter, M.** (2022). Learning
   to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning.
   *Conference on Robot Learning (CoRL)*.

4. **Margolis, G., et al.** (2022). Walk These Ways: Tuning Robot Control
   for Generalization with Multiplicity of Behavior. *CoRL 2022*.

5. **Hwangbo, J., et al.** (2019). Learning Agile and Dynamic Motor Skills
   for Legged Robots. *Science Robotics*. DOI: 10.1126/scirobotics.aau5872.

6. **Lee, J., Hwangbo, J., Wellhausen, L., Koltun, V., & Hutter, M.**
   (2020). Learning Quadrupedal Locomotion over Challenging Terrain.
   *Science Robotics*. DOI: 10.1126/scirobotics.abc5986.

7. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.**
   (2017). Proximal Policy Optimization Algorithms. arXiv: 1707.06347.

8. **Peng, X. B., Andrychowicz, M., Zaremba, W., & Abbeel, P.** (2018).
   Sim-to-Real Transfer of Robotic Control with Dynamics Randomization.
   *ICRA 2018*. arXiv: 1710.06537.

9. **Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., & Abbeel,
   P.** (2017). Domain Randomization for Transferring Deep Neural Networks
   from Simulation to the Real World. *IROS 2017*. arXiv: 1703.06907.

10. **Narvekar, S., et al.** (2020). Curriculum Learning for Reinforcement
    Learning Domains: A Framework and Survey. *JMLR*, 21(181), 1-50.

11. **Kirkpatrick, J., et al.** (2017). Overcoming Catastrophic Forgetting
    in Neural Networks. *PNAS*, 114(13), 3521-3526.

12. **French, R. M.** (1999). Catastrophic Forgetting in Connectionist
    Networks. *Trends in Cognitive Sciences*, 3(4), 128-135.

13. **Li, Z., & Hoiem, D.** (2019). Learning without Forgetting.
    *IEEE Transactions on Pattern Analysis and Machine Intelligence*.

14. **Ross, S., Gordon, G. J., & Bagnell, J. A.** (2011). A Reduction of
    Imitation Learning and Structured Prediction to No-Regret Online
    Learning. *AISTATS 2011*.

15. **CTS: Concurrent Teacher-Student Reinforcement Learning for Legged
    Locomotion.** (2024). ResearchGate: 383934038.

16. **Scaling Rough Terrain Locomotion with Automatic Curriculum RL.**
    (2026). arXiv: 2601.17428.

17. **NVIDIA Isaac Lab Documentation.** (2025).
    https://isaac-sim.github.io/IsaacLab/

18. **RSL-RL: Robotic Systems Lab Reinforcement Learning.**
    https://github.com/leggedrobotics/rsl_rl

19. **Zhuang, F., et al.** (2020). A Comprehensive Survey on Transfer
    Learning. *Proceedings of the IEEE*, 109(1), 43-76.

20. **Parkour in the Wild: Unified Agile Locomotion.** (2025). ETH Zurich
    / NVIDIA. arXiv: 2505.11164.

---

21. **Abel, D., Dabney, W., Harutyunyan, A., Ho, M. K., Littman, M. L.,
    Precup, D., & Singh, S.** (2018). Policy and Value Transfer in Lifelong
    Reinforcement Learning. *ICML 2018*.

---

*Document created February 23, 2026. Updated February 24, 2026 with
Attempt #1 failure analysis and Attempt #2 corrective measures.
Stage 1 training in progress on NVIDIA H100 NVL — 16,384 environments,
25,000 iterations.*
