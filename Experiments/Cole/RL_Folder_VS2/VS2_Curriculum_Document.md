# RL_FOLDER_VS2: Curriculum-Based Navigation Policy Training

## 1. Training Architecture Overview

The VS2 training system implements a **hierarchical reinforcement learning (RL)** architecture for training a Boston Dynamics Spot robot to autonomously navigate waypoints within a simulated arena using NVIDIA Isaac Sim. The system uses a high-level navigation policy trained via Proximal Policy Optimization (PPO) that issues velocity commands to a pre-trained low-level locomotion controller (SpotFlatTerrainPolicy).

### 1.1 Environment Configuration

| Parameter | Value |
|-----------|-------|
| Arena | 50m diameter circular arena (25m radius) |
| Waypoints per episode | 25 |
| Waypoint capture radius | 0.25m |
| Observation space | 16 LiDAR raycasts (5m range) + stage encoding |
| Action space | [vx, vy, omega] velocity commands |
| Forward velocity range | -0.5 to 5.0 m/s |
| Lateral velocity range | -0.5 to 0.5 m/s |
| Turning rate range | -1.5 to 1.5 rad/s |
| Physics timestep | 0.002s (500 Hz) |
| Control frequency | 20 Hz |
| Steps per PPO iteration | 6,000 (300s of simulation) |

### 1.2 Scoring System

The environment maintains an internal scoring system that governs episode duration:

- **Initial score:** 300 points
- **Time decay:** -1.0 point/second (active from Stage 4 onward)
- **Waypoint bonus:** +15 points per capture
- **Episode termination:** Score reaches 0 (timeout), robot falls, or all 25 waypoints captured

### 1.3 Policy Network

| Component | Configuration |
|-----------|--------------|
| Architecture | 3-layer MLP [256, 256, 128] |
| Activation | ReLU |
| Action distribution | Gaussian (fixed std = 0.3) |
| Total parameters | 157,191 |

### 1.4 PPO Hyperparameters

Three learning rate variants were trained in parallel to evaluate convergence behavior:

| Parameter | Conservative | Moderate (Base) | Aggressive |
|-----------|-------------|-----------------|------------|
| Learning rate | 5.0e-5 | 1.0e-4 | 2.0e-4 |
| Clip parameter | 0.2 | 0.2 | 0.2 |
| Value coefficient | 0.5 | 0.5 | 0.5 |
| Entropy coefficient | 0.01 | 0.01 | 0.01 |
| Discount factor (gamma) | 0.99 | 0.99 | 0.99 |
| GAE lambda | 0.95 | 0.95 | 0.95 |
| Max gradient norm | 0.5 | 0.5 | 0.5 |
| Target KL divergence | 0.015 | 0.015 | 0.015 |
| PPO epochs per batch | 10 | 10 | 10 |

The target KL divergence threshold (0.015) and multi-epoch training (10 epochs) were added after a **critical training collapse incident** (see Section 5) where the original implementation lacked early stopping, resulting in catastrophic KL divergences exceeding 34.0.

---

## 2. Curriculum Design (8-Stage Progressive Training)

The VS2 curriculum implements an 8-stage progressive training pipeline that systematically increases task complexity. Advancement from one stage to the next requires achieving a sustained success rate above a threshold within a sliding window of recent episodes.

### 2.1 Stage Definitions

| Stage | Name | Waypoint Distance | Obstacles | Time Limit | Success Criterion | Window / Threshold |
|-------|------|-------------------|-----------|------------|-------------------|--------------------|
| 1 | Random Walking | None (no waypoints) | None | 120-180s | Remain upright for full duration | 50 episodes / 80% |
| 2 | Waypoints 5m | 5m | None | No limit | Capture all 25 waypoints | 100 episodes / 80% |
| 3 | Waypoints 10m | 10m | None | No limit | Capture all 25 waypoints | 100 episodes / 80% |
| 4 | Waypoints 20m | 20m | None | No limit | Capture all 25 waypoints | 100 episodes / 80% |
| 5 | Waypoints 20m→40m | First: 20m, subsequent: 40m | None | No limit | Capture all 25 waypoints | 100 episodes / 80% |
| 6 | Add Light Obstacles | First: 20m, subsequent: 40m | 25% light | No limit | Capture all 25 waypoints | 100 episodes / 80% |
| 7 | Add Heavy Obstacles | First: 20m, subsequent: 40m | 10% light + 10% heavy | No limit | Capture all 25 waypoints | 100 episodes / 80% |
| 8 | Add Small Static Obstacles | First: 20m, subsequent: 40m | 10% light + 10% heavy + 10% small static | No limit | Capture all 25 waypoints | 100 episodes / 90% |

### 2.2 Obstacle Types

| Type | Mass Range | Behavior |
|------|-----------|----------|
| Light | ≤ 0.45 kg | Pushable with minimal effort |
| Heavy | ≥ 65.4 kg | Immovable (static); robot must navigate around |
| Small Static | 4.3–10.2 cm diameter | Small ground-level static obstacles |

### 2.3 Curriculum Progression Logic

- **Stages 1–3:** No time decay on the internal score; episodes focus on stability and basic navigation learning.
- **Stages 4+:** Time decay is active (-1 point/second), creating urgency to complete waypoints efficiently.
- **Stages 2–5:** Progress shaping rewards are active (distance-based guidance toward waypoints).
- **Stages 6–8:** Push reward/penalty system activates alongside obstacle environments.

---

## 3. Reward and Penalty Design

The VS2 reward system consists of 15 distinct reward/penalty signals organized into navigation rewards, movement quality penalties, push interaction rewards, and terminal event signals.

### 3.1 Navigation Rewards

#### Waypoint Capture Reward
**Weight:** +50.0 per waypoint captured

Flat bonus applied each time the robot enters the 0.25m capture radius of the current waypoint. For a complete episode capturing all 25 waypoints, the maximum waypoint reward is +1,250.

#### Progress Shaping (Stages 2–5)
**Weight:** α = 5.0

$$r_{progress} = 5.0 \times (d_{t-1} - d_t)$$

A potential-based shaping reward proportional to the change in distance to the current waypoint. Positive when the robot moves closer, negative when it moves farther away.

#### Heading Reward
**Weight:** 0.2

$$r_{heading} = 0.2 \times \max(0,\; \hat{v} \cdot \hat{w})$$

Rewards alignment between the robot's velocity direction and the direction toward the current waypoint. Only active when the robot's horizontal speed exceeds 0.1 m/s. The dot product ranges from -1 (facing away) to +1 (facing directly toward), clamped at 0 (no reward for heading away).

#### Proximity Deceleration Reward
**Weight:** 0.15

$$r_{decel} = 0.15 \times \max(0,\; 1.0 - |v_{horiz} - 0.2 \times d_{wp}|)$$

Active within 2.5m of a waypoint. Rewards the robot for matching an ideal approach speed that decreases linearly with distance (0.5 m/s at 2.5m, 0 m/s at capture).

#### Static Turn Reward
**Weight:** 0.5

$$r_{turn} = \begin{cases} 0.5 \times \min\left(\frac{|\omega|}{2.0},\; 1.0\right) & \text{if } |v_x| < 0.3 \text{ and } |\omega| > 0.3 \\ 0 & \text{otherwise} \end{cases}$$

Rewards the robot for turning in place rather than attempting wide arcs. Activates when forward speed is below 0.3 m/s and angular velocity exceeds 0.3 rad/s.

#### Efficiency Bonus
**Weight:** 2.0

$$r_{efficiency} = 2.0 \times \text{score}_{remaining}$$

Applied once when the robot captures all 25 waypoints. Rewards the robot proportional to remaining internal score points, incentivizing faster completion.

#### Speed Reward (Tiered Distance-Based)
| Speed Range | Reward per Meter |
|-------------|-----------------|
| < 0.9 m/s (Tier 1) | 0.0 (no reward) |
| 0.9–1.79 m/s (Tier 2) | 0.25 |
| ≥ 1.79 m/s (Tier 3) | 0.50 |

Only active when the forward path is clear (nearest obstacle > 2.0m) and the robot is not near a waypoint (distance > 3.0m).

### 3.2 Movement Quality Penalties

#### Lateral Velocity Penalty
**Weight:** -0.25

$$p_{lateral} = -0.25 \times \min\left(\frac{|v_y|}{0.5},\; 1.0\right)$$

Always active. Penalizes strafing (sideways motion) to prevent drift. The penalty saturates at |vy| = 0.5 m/s.

#### Cross-Track Error Penalty
**Weight:** -0.4

$$p_{cross} = -0.4 \times |\hat{v} \cdot \hat{w}_{\perp}|$$

Penalizes the component of velocity perpendicular to the ideal straight-line path toward the current waypoint. Only active when distance to waypoint > 0.5m and speed > 0.1 m/s.

#### Wrong Direction Penalty
**Weight:** 2.0× multiplier on negative progress shaping

When the robot moves away from the waypoint (negative progress), the progress shaping penalty is doubled via this multiplier, strengthening the signal to turn around.

### 3.3 Push Interaction Rewards (Stages 6–8)

| Signal | Weight | Condition | Purpose |
|--------|--------|-----------|---------|
| Push Exploration | +0.01 | Contact force > 0.4, joint effort < 0.7 | Reward initial light contact with obstacles |
| Push Sustained | +0.01 | Above + speed > 0.2 m/s | Reward moving while in contact (active pushing) |
| Push Stuck | -1.0 | Contact force > 0.4, speed < 0.2 m/s | Penalize high contact without progress |
| Push Wasted Effort | -1.0 | Contact force > 0.3, joint effort > 0.8 | Penalize high effort without productive pushing |

### 3.4 Terminal Event Penalties

| Event | Reward | Score Impact | Effect |
|-------|--------|-------------|--------|
| Fall | -500.0 | -500.0 | Immediate episode termination |
| Boundary Exit | -5.0 | — | Immediate episode termination |
| Timeout (score ≤ 0) | -500.0 | — | Episode ends when score depleted |

### 3.5 Stage 1 Special: Stagnation Penalty

During Stage 1 (Random Walking), a stagnation penalty of -1.0/second is applied when the robot fails to move at least 0.05m per control step, preventing the policy from learning to stand still.

---

## 4. Training Results and Iterations

### 4.1 Stage Progression Summary

| Stage | Conservative (5e-5) | Moderate (1e-4) | Aggressive (2e-4) |
|-------|---------------------|-----------------|-------------------|
| 1 – Random Walking | ✅ Completed | ✅ Completed | ✅ Completed |
| 2 – Waypoints 5m | ✅ Completed | ✅ Completed | ✅ Completed |
| 3 – Waypoints 10m | ✅ Completed | ✅ Completed | ✅ Completed* |
| 4 – Waypoints 20m | ✅ Completed | ✅ Completed | ✅ Completed* |
| 5 – Waypoints 20m→40m | ✅ Completed (~127 iters) | ✅ Completed (~127 iters) | ✅ Completed (~127 iters) |
| 6 – Light Obstacles | ✅ Completed (~221 iters) | ✅ Completed (~221 iters) | ✅ Completed (~3 saves at 80%) |
| 7 – Heavy Obstacles | Training (408+ iters, 70%) | ✅ **Completed** (stage_7_complete.pt) | Not reached |
| 8 – Small Static Obstacles | Not reached | Not reached | Not reached |

*Aggressive variant has no stage_3_complete.pt or stage_4_complete.pt checkpoint files, suggesting these stages were passed quickly or through an earlier training run before checkpoint convention was established.

### 4.2 Total Iterations per Variant

Training iterations were accumulated across multiple sequential runs:

| Run Phase | Conservative | Moderate | Aggressive |
|-----------|-------------|----------|------------|
| Stage 5 (10,000 max) | ~127 iterations to threshold | ~127 iterations to threshold | ~127 iterations to threshold |
| Stage 5→6 transition logs | 221 iterations (stage 5 continued + stage 6) | 221 iterations | 408 iterations (stayed on stage 5 longer) |
| Stage 6 (1,000 max) | 221 iterations to threshold | 221 iterations | Stage 6 reached only briefly (~3 checkpoints) |
| Stage 7 (1,500 max) | 408+ iterations (ongoing, 70% success) | 408+ iterations (**Stage 7 completed**) | Not attempted |

### 4.3 Performance Data at Final Stage

#### Moderate Variant (Best Performer) — Stage 7: Add Heavy Obstacles

The moderate variant (LR = 1.0e-4) successfully completed Stage 7, navigating 25 waypoints through an environment with 10% light + 10% heavy obstacles.

**Typical successful episode (Stage 7):**

| Metric | Value |
|--------|-------|
| Waypoints captured | 25/25 |
| Episode length | ~380–420 seconds |
| Total reward | ~4,700–4,850 |
| Game score | ~280–295 |
| Success rate | 80%+ (threshold met) |

**Reward component breakdown (successful episode):**

| Component | Typical Value | Interpretation |
|-----------|--------------|----------------|
| progress_shaping | +4,860 | Strong distance-closing behavior |
| lateral_penalty | -1,750 | Significant strafing still present |
| heading_reward | +1,420 | Good waypoint heading alignment |
| push_wasted | -1,340 | High wasted effort near obstacles |
| waypoint_capture | +1,250 | All 25 waypoints × 50.0 |
| efficiency_bonus | +570 | Completed with ~285 score remaining |
| cross_track_error | -500 | Moderate path deviation |
| static_turn_reward | +185 | Active turning-in-place behavior |
| push_stuck | -75 | Some stuck-on-obstacle events |
| push_sustained | +58 | Some successful pushing |
| push_exploration | +41 | Contact exploration active |
| decel_reward | +25 | Some deceleration near waypoints |

**PPO training stability (Stage 7):**

| Metric | Value |
|--------|-------|
| Policy loss | ~0.0000 |
| Value loss | 3,300–6,700 (decreasing) |
| Entropy | 4.029 (stable) |
| Approx KL | 0.0000–0.0014 (well within safety threshold) |
| PPO epochs | 10/10 (no early stopping triggered) |

#### Conservative Variant — Stage 7: Add Heavy Obstacles

The conservative variant (LR = 5.0e-5) was still training on Stage 7 at the time of assessment.

**Performance at 408+ iterations (Stage 7):**

| Metric | Value |
|--------|-------|
| Success rate | 70% (target: 80%) |
| Mean waypoints | 23.0/25 (on successful episodes) |
| Total reward | ~4,400–4,600 per success |
| Fall rate | ~33% of failure episodes |

Typical successful episode achieved 25/25 waypoints in ~420–430 seconds with higher lateral penalties (-2,000) and more conservative heading alignment compared to the moderate variant.

#### Aggressive Variant — Stage 6: Add Light Obstacles

The aggressive variant (LR = 2.0e-4) reached Stage 6 but did not advance to Stage 7.

**Performance at Stage 6:**

| Metric | Value |
|--------|-------|
| Success rate | 80% (at threshold) |
| Mean waypoints | 25/25 (on successful episodes) |
| Episode length | ~330–340 seconds (fastest) |
| Total reward | ~4,900–4,935 per success |
| Game score | ~340 (highest efficiency) |

Despite being the fastest learner in early stages and achieving the highest per-episode scores, the aggressive variant's training instability prevented reliable advancement to Stage 7.

---

## 5. Training Collapse Incident and Recovery

### 5.1 Incident Summary

On March 7, 2026, all three training runs experienced **catastrophic policy collapse** during an overnight training session. The root cause was identified as missing PPO safety mechanisms in the original implementation.

### 5.2 What Happened

1. **March 5–6:** Training progressed normally. Run 3 achieved 23% success rate (best), Run 1 reached 13%.
2. **March 6, 22:00–23:00:** KL divergence began climbing from normal levels to 4–7 (safe range: < 0.03).
3. **March 7, 03:00–05:00:** KL divergence exploded to 10–34. Run 1 peaked at **KL = 34.37** (over 1,000× the safe threshold). All policies collapsed to degenerate states with 0 waypoint captures.

### 5.3 Root Cause

The original PPO implementation lacked four critical safeguards:

| Missing Feature | Impact |
|----------------|--------|
| Early stopping (target KL) | Policy could diverge without limit |
| Multi-epoch training | Only 1 update per batch (insufficient learning) |
| Minibatch training | Full batch updates caused gradient instability |
| Learning rate control | 3.0e-4 was too aggressive for long-term stability |

### 5.4 Fixes Implemented

1. **Reduced base learning rate:** 3.0e-4 → 1.0e-4 (70% reduction)
2. **Added target KL early stopping:** Training halts within an epoch if KL > 0.015
3. **Multi-epoch training:** 10 PPO epochs per batch (up from 1)
4. **Three learning rate variants:** Conservative (5e-5), Moderate (1e-4), Aggressive (2e-4) to empirically determine optimal rate

These fixes stabilized training. Post-fix KL divergence consistently remained below 0.003 across all three variants, with the moderate variant ultimately completing the full 7-stage curriculum.

---

## 6. Key Findings and Analysis

### 6.1 Learning Rate Comparison

| Metric | Conservative (5e-5) | Moderate (1e-4) | Aggressive (2e-4) |
|--------|---------------------|-----------------|-------------------|
| Final stage reached | 7 (training) | **7 (completed)** | 6 (completed) |
| Stage 7 success rate | 70% | **80%+** | N/A |
| Episode efficiency | Slowest (420–450s) | Balanced (380–420s) | Fastest (330–340s) |
| KL stability | Most stable | Stable | KL early stops triggered occasionally |
| Lateral penalty | Highest (-2,000) | Moderate (-1,750) | Lowest (-1,570) |
| Best model path | — | **Best_Nav_Policy/stage_7_complete.pt** | — |

### 6.2 Dominant Reward Signals

Across all three variants, the reward signal was dominated by:

1. **Progress shaping (+4,860):** The largest positive signal. The alpha=5.0 multiplier on distance reduction toward waypoints was the primary driver of goal-directed behavior.
2. **Lateral velocity penalty (-1,600 to -2,000):** The largest persistent penalty. All variants exhibited significant strafing behavior despite the penalty, suggesting the Spot locomotion controller naturally produces lateral drift.
3. **Heading reward (+1,200 to +1,530):** Second largest positive signal. Effective in orienting the robot toward waypoints.
4. **Push wasted effort (-1,300 to -1,450):** In obstacle stages, this was the dominant negative signal—indicating frequent unproductive contact with obstacles.

### 6.3 Areas for Improvement Identified

1. **Lateral drift:** All variants accumulated -1,600 to -2,000 in lateral penalties per episode. The lateral_velocity_penalty weight (0.25) may be insufficient to overcome the locomotion controller's natural tendency.
2. **Push system inefficiency:** The push_wasted penalty consistently exceeds push_exploration and push_sustained rewards combined, suggesting the robot has not learned efficient obstacle interaction strategies.
3. **Stage 8 unreached:** No variant reached Stage 8 (Small Static Obstacles at 90% threshold). The 90% success requirement with three obstacle types represents a significant jump from Stage 7's 80% threshold with two obstacle types.
4. **Episode length:** Even in successful episodes, completion takes 330–430 seconds for 25 waypoints spanning a 50m arena, indicating room for speed optimization.

### 6.4 Best Policy

The best-performing policy from VS2 training is:

**`Best_Nav_Policy/stage_7_complete.pt`** (Moderate variant, LR = 1.0e-4)

This policy can navigate 25 waypoints in a 50m arena with 10% light + 10% heavy obstacle coverage at an 80%+ success rate.
