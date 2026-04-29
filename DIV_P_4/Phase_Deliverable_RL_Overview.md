# Reinforcement Learning for Quadruped Locomotion

## 1. Introduction

This document provides a high-level overview of the Reinforcement Learning (RL) methods used to train locomotion and navigation policies for the Boston Dynamics Spot quadruped robot. It covers the theoretical foundations, algorithm selection, and the engineering considerations specific to training a four-legged robot to traverse complex terrain autonomously.

## 2. Reinforcement Learning Fundamentals

### 2.1 The Markov Decision Process (MDP)

RL frames robot control as a Markov Decision Process, defined by:

- **State (s):** The robot's current configuration — joint angles, velocities, body orientation, and surrounding terrain geometry.
- **Action (a):** The control signal sent to the robot — 12 joint position targets (one per degree of freedom).
- **Reward (r):** A scalar feedback signal that measures how well the robot is performing (e.g., forward velocity, gait quality, energy efficiency).
- **Transition (T):** The physics simulator advances the world state based on the action taken.
- **Policy (pi):** A neural network that maps observations to actions — the "brain" of the robot.

The goal is to find a policy that maximizes the cumulative discounted reward over time:

```
J(pi) = E[sum_{t=0}^{T} gamma^t * r_t]
```

where `gamma = 0.99` is the discount factor that prioritizes near-term rewards.

### 2.2 Value Functions

Two value functions guide learning:

- **V(s):** The expected cumulative reward from state s — "how good is this situation?"
- **A(s, a):** The advantage of taking action a over the average — "how much better is this action than typical?"

The advantage function is estimated using Generalized Advantage Estimation (GAE) with `lambda = 0.95`, which balances bias and variance in the reward signal.

### 2.3 Why Not Supervised Learning?

Locomotion cannot be solved with supervised learning because there is no pre-existing dataset of correct joint commands for every terrain condition. The robot must discover effective movement strategies through trial-and-error interaction with the physics simulator. RL enables this by:

1. Generating experience through simulation (billions of physics steps)
2. Evaluating outcomes via the reward function
3. Updating the policy to increase the probability of high-reward actions

## 3. Proximal Policy Optimization (PPO)

### 3.1 Algorithm Overview

PPO (Schulman et al., 2017) is the RL algorithm used throughout this project. It is the standard choice for robotic locomotion because it provides:

- **Stability:** The clipped surrogate objective prevents catastrophically large policy updates that can destroy a working gait.
- **Sample efficiency:** On-policy learning with mini-batch updates extracts maximum learning from each batch of experience.
- **Parallelism:** The algorithm scales linearly across thousands of parallel environments on a single GPU.

### 3.2 PPO Update Rule

PPO constrains updates using a clipped objective:

```
L_clip = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
```

where `r_t = pi_new(a|s) / pi_old(a|s)` is the probability ratio and `eps = 0.2` is the clipping threshold. This ensures the new policy stays within 20% of the old policy's action probabilities.

### 3.3 PPO Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Clip parameter | 0.2 | Constrains policy update magnitude |
| Value loss coefficient | 1.0 | Weight of critic loss in total loss |
| Entropy coefficient | 0.01 | Encourages exploration |
| Learning epochs | 5-8 | Number of passes over each data batch |
| Mini-batches | 4-8 | Data splits per learning epoch |
| Learning rate | 1e-4 to 3e-5 | Step size for gradient descent |
| Discount (gamma) | 0.99 | Reward time horizon |
| GAE lambda | 0.95 | Advantage estimation smoothing |
| Max gradient norm | 1.0 | Prevents gradient explosion |

## 4. Quadruped-Specific Considerations

### 4.1 Action Space: 12 Degrees of Freedom

The Spot robot has 4 legs, each with 3 actuated joints:

- **HX (Hip Abduction):** Spreads the leg laterally
- **HY (Hip Flexion):** Swings the leg forward/backward
- **KN (Knee):** Bends the lower leg

The 12-dimensional action space uses **type-grouped ordering** (all HX joints, then all HY, then all KN), not leg-grouped ordering. This is a critical implementation detail — mismatched ordering produces nonsensical gaits.

```
Action indices: [fl_hx, fr_hx, hl_hx, hr_hx,   # abduction
                 fl_hy, fr_hy, hl_hy, hr_hy,   # hip flexion
                 fl_kn, fr_kn, hl_kn, hr_kn]   # knee
```

Actions are position targets scaled by 0.2 radians from default standing pose, executed through PD controllers with Kp=60 and Kd=1.5.

### 4.2 Observation Design

The policy receives a 235-dimensional observation vector combining:

| Component | Dimensions | Purpose |
|-----------|-----------|---------|
| Height scan | 187 | 17x11 grid of terrain elevation measurements, 0.1m resolution |
| Base linear velocity | 3 | Body-frame forward, lateral, and vertical speed |
| Base angular velocity | 3 | Roll, pitch, and yaw rates |
| Projected gravity | 3 | Gravity vector in body frame (encodes orientation) |
| Velocity commands | 3 | Target forward, lateral, and yaw velocity |
| Joint positions | 12 | Current joint angles relative to default pose |
| Joint velocities | 12 | Current joint angular velocities |
| Previous actions | 12 | Last policy output (enables action smoothness) |

The height scan provides a local terrain map around the robot, enabling the policy to "see" upcoming obstacles without a camera. This is the primary exteroceptive input for locomotion policies.

### 4.3 Reward Shaping for Locomotion

Quadruped locomotion requires carefully designed reward functions that balance multiple competing objectives:

**Positive rewards (incentives):**
- Forward velocity tracking — move at the commanded speed
- Gait synchronization — maintain a diagonal trot pattern (front-left with hind-right)
- Foot clearance — lift feet high enough to clear obstacles
- Air time — proper swing phase duration for each leg

**Negative rewards (penalties):**
- Body orientation — penalize excessive roll and pitch
- Joint torque — minimize energy consumption
- Foot slip — penalize feet sliding on contact
- Action smoothness — prevent jerky, oscillating commands

The reward function typically contains 15-19 terms. The relative weights determine the character of the learned gait — higher gait weight produces a more regular trot, while higher velocity weight produces faster but potentially less stable movement.

**Key insight:** Sparse rewards (e.g., "did the robot reach the goal?") do not work for locomotion. The robot needs continuous feedback on the quality of each step to learn effective movement patterns. Reward shaping provides this dense signal.

### 4.4 Curriculum Learning

Training a robot on the hardest terrain from the start leads to failure — the untrained policy immediately falls and receives only negative rewards. Curriculum learning solves this by gradually increasing difficulty:

**Training Phases:**

| Phase | Terrain | Purpose |
|-------|---------|---------|
| A (Flat) | 100% smooth ground | Learn basic walking gait |
| A.5 (Transition) | 50% flat + 50% gentle terrain | Adapt gait to mild perturbations |
| B-easy (Robust Easy) | 12 terrain types, low difficulty | Develop terrain-adaptive behavior |
| B (Robust) | 12 terrain types, full difficulty | Master all terrain conditions |
| C (Navigation) | Visual terrain with depth camera | Learn to navigate using vision |

Within each phase, the terrain curriculum uses an automatic promotion/demotion system: robots that consistently survive on their current difficulty level are promoted to harder terrain, while those that fall are demoted to easier patches. This ensures each robot trains at its challenge frontier.

**12 Terrain Types:**
- Geometric: Pyramid stairs (up/down), random grid boxes, stepping stones
- Surface: Random rough, slopes (up/down), wave terrain, friction plane, vegetation plane
- Compound: Height-field stairs, discrete obstacles, repeated boxes

### 4.5 The Sim-to-Real Gap

Policies trained in simulation often fail on real hardware due to differences between simulated and real physics. Key sources of this gap include:

| Gap Source | Simulation | Real World |
|-----------|-----------|------------|
| Actuator response | Instantaneous | 40-60ms latency |
| Sensor readings | Perfect | Noisy, delayed, dropout |
| Ground contact | Idealized | Variable friction, deformable |
| Robot mass | Exact | Payload variation, wear |
| External forces | None | Wind, bumps, human contact |

**Mitigation strategies (Domain Randomization):**
- Randomize friction coefficients (0.3-1.0) during training
- Add action delay (40ms) to simulate actuator latency
- Add observation noise to simulate sensor imperfections
- Apply random external push forces every 10-15 seconds
- Randomize robot mass (plus/minus 2.5kg)
- Drop 5% of height scan rays to simulate sensor dropout

By training across these randomized conditions, the policy learns robust behaviors that transfer more reliably to the real robot.

### 4.6 Safety Constraints

Quadruped training must incorporate safety to prevent hardware damage during deployment:

- **Joint position limits:** Hard constraints prevent joints from exceeding mechanical range
- **Torque limits:** Hip joints capped at 45 Nm, knee joints at 100 Nm
- **Flip detection:** Episodes terminate immediately when the robot body inverts
- **Bad orientation detection:** Training penalizes excessive roll/pitch angles
- **Body contact penalty:** Heavy penalty when the robot's body (not feet) contacts the ground

## 5. Training Infrastructure

### 5.1 Software Stack

| Component | Technology | Role |
|-----------|-----------|------|
| Physics simulator | NVIDIA Isaac Lab (PhysX GPU) | Simulates robot + terrain at 500 Hz |
| RL framework | RSL-RL | PPO implementation with parallel env support |
| Deep learning | PyTorch 2.7.0 (CUDA) | Neural network training |
| Configuration | `@configclass` decorators | Type-safe environment/algorithm configs |
| Visualization | TensorBoard | Training metrics monitoring |

### 5.2 Hardware

- **GPU:** NVIDIA H100 NVL (96 GB HBM3 VRAM)
- **Sustained safe load:** 8,192 parallel environments (49C, 171W)
- **Maximum tested:** 65,536 parallel environments (100% GPU, 65C, 299W)

### 5.3 Parallel Environment Scaling

The key to efficient RL training is massive parallelism. Instead of one robot learning slowly, we simulate thousands of robots simultaneously:

```
Training throughput = num_envs x physics_steps_per_second
                    = 8192 x 500 Hz
                    = 4,096,000 physics steps/second
                    ~ 10 billion steps per 24-hour run
```

Each parallel environment contains a complete robot and terrain patch, running independent physics. All environments share the same policy network, and gradients are aggregated across the full batch for each PPO update.

### 5.4 Training Timescales

| Phase | Typical Duration | Iterations | Total Steps |
|-------|-----------------|------------|-------------|
| Phase A (Flat) | 2-4 hours | 500 | ~50M |
| Phase A.5 (Transition) | 4-8 hours | 1,000 | ~100M |
| Phase B (Robust) | 24-48 hours | 20,000-35,000 | ~2B |
| S2R Expert | 10-20 hours | 5,000 | ~500M |
| Phase C (Navigation) | 6-30 hours | 5,000 | ~60M |

## 6. Neural Network Architecture

### 6.1 Locomotion Policy (MLP)

The locomotion policy is a fully connected Multi-Layer Perceptron:

```
Input:  235 dimensions (height scan + proprioception)
Hidden: [512, 256, 128] with ELU activations
Output: 12 dimensions (joint position targets)
Total:  286,604 trainable parameters
```

The actor (policy) and critic (value function) share the same architecture but have separate weights. The critic outputs a single scalar value estimate.

### 6.2 Navigation Policy (CNN + MLP)

For visual navigation (Phase C), a Convolutional Neural Network processes depth images:

```
Depth Image: (1, 32, 32) single-channel depth
  -> Conv2d(1, 32, kernel=5, stride=2) + ELU
  -> Conv2d(32, 64, kernel=3, stride=2) + ELU
  -> Conv2d(64, 64, kernel=3, stride=2) + ELU
  -> Flatten -> Linear(*, 128) + ELU

Features (128) + Proprioception (12) = 140 dimensions
  -> Actor MLP [256, 128] -> 3 (velocity commands)
  -> Critic MLP [256, 128] -> 1 (value estimate)
Total: 489,799 trainable parameters
```

The navigation policy operates at 10 Hz (vs. 50 Hz for locomotion) and outputs high-level velocity commands [vx, vy, yaw_rate] that are executed by a frozen locomotion policy underneath.

## 7. Summary

Reinforcement learning enables quadruped robots to discover locomotion strategies that would be extremely difficult to hand-engineer. The combination of PPO, massive parallelism, curriculum learning, and domain randomization produces policies that can traverse friction surfaces, grass, boulders, and stairs — all from a single training pipeline. The hierarchical approach (frozen locomotion + trainable navigation) further extends these capabilities to visual terrain navigation using depth cameras.
