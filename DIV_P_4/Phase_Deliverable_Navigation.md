# Navigation Policy Development

## 1. Overview

This document details the navigation policy development efforts across the team. Three distinct approaches were pursued in parallel, each with different sensing modalities, network architectures, and training curricula. All approaches share the same hierarchical principle: a high-level navigation policy outputs velocity commands that are executed by a frozen locomotion policy.

```
Navigation Policy (high-level)    -->  Frozen Loco Policy (low-level)  -->  Physics
[where to go]                          [how to walk]                        [joint forces]
velocity commands [vx, vy, wz]         12 joint position targets            500 Hz PhysX
```

## 2. Team Navigation Approaches — Comparison

| Aspect | Alex (NAV_ALEX) | Colby (CombinedPolicy) | Cole (RL_FOLDER_VS3) |
|--------|----------------|----------------------|---------------------|
| **Sensing** | Depth camera (32x32) | Depth camera (64x64) | Multi-layer raycasts (48 rays) |
| **Observation dims** | 1,036 | 4,108 | 75 |
| **Network** | CNN + MLP | CNN + MLP | MLP only |
| **Nav frequency** | 10 Hz | 10 Hz | 20 Hz |
| **Loco backbone** | Boulder V6 (4500) | Mason Hybrid (33200) | SpotFlatTerrainPolicy |
| **Training framework** | RSL-RL + Isaac Lab | RSL-RL + Isaac Lab | Custom PPO |
| **Curriculum** | 6-level terrain | Terrain-based | 7-stage task-based |
| **AI Coach** | Yes (Claude Sonnet) | No | No |
| **Unique feature** | Terrain-adaptive height | rsl_rl 5.0.1 compat layer | Object pushing (Stage 3) |
| **Status** | Active training | Blocked (API compat) | Active development |

## 3. Alex — NAV_ALEX (Phase C)

### 3.1 Architecture

NAV_ALEX implements a hierarchical two-brain control system:

```
Depth Camera (32x32, 30m range)
        |
        v
[CNN Encoder: 3 conv layers -> 128-dim features]
        |
+ Proprioception (12 dims: velocity, angular vel, gravity, prev action)
        |
        v
[Navigation Policy MLP, 10 Hz]  <-- BEING TRAINED
        |
        v
Velocity Command [vx, vy, wz]
        |
        v
[Frozen Boulder V6 Policy, 50 Hz]  <-- FROZEN (286K params)
        |
        v
12 Joint Targets --> Spot Robot (500 Hz physics)
```

**Rate hierarchy:**
- 500 Hz: PhysX GPU physics solver
- 50 Hz: Frozen locomotion policy (5 substeps per nav step)
- 10 Hz: Navigation policy makes velocity command decisions

### 3.2 Depth Camera Specification

| Property | Value |
|----------|-------|
| Type | RayCasterCamera (no Vulkan required) |
| Resolution | 32 x 32 pixels (optimized from 64x64 for training speed) |
| Field of view | ~90 degrees |
| Range | 30 meters |
| Update rate | 10 Hz |
| Mount position | +0.3m forward, +0.3m above body center |
| Tilt | 10 degrees downward |
| Data type | distance_to_image_plane |
| Limitation | Static meshes only (terrain, not dynamic obstacles) |

The 30-meter range provides approximately 10 seconds of lookahead at 3 m/s, enabling route planning around distant obstacles.

### 3.3 CNN Architecture

```
Input:  (N, 1, 32, 32)  -- single-channel depth image
Conv2d(1, 32, kernel=5, stride=2) + ELU  --> (N, 32, 14, 14)
Conv2d(32, 64, kernel=3, stride=2) + ELU --> (N, 64, 6, 6)
Conv2d(64, 64, kernel=3, stride=2) + ELU --> (N, 64, 2, 2)
Flatten                                   --> (N, 256)
Linear(256, 128) + ELU                    --> (N, 128)
```

The CNN backbone is shared between actor and critic networks. The flattened CNN features (128) are concatenated with proprioception (12) to form a 140-dimensional combined feature vector.

**Actor:** MLP [256, 128] -> 3 (velocity commands [vx, vy, wz])
**Critic:** MLP [256, 128] -> 1 (value estimate)
**Total parameters:** 489,799

### 3.4 Observation Space (1,036 dimensions)

| Index Range | Component | Dimensions | Notes |
|-------------|-----------|-----------|-------|
| [0:1024] | Depth image (flattened) | 1,024 | 32x32, normalized [0,1], 30m range |
| [1024:1027] | Body linear velocity | 3 | Body-frame, noise std=0.15 |
| [1027:1030] | Body angular velocity | 3 | Body-frame, noise std=0.15 |
| [1030:1033] | Projected gravity | 3 | Noise std=0.05 |
| [1033:1036] | Previous action | 3 | No noise |

### 3.5 Action Space

| Index | Component | Range | Unit |
|-------|-----------|-------|------|
| 0 | vx (forward) | [-1.0, 3.0] | m/s |
| 1 | vy (lateral) | [-1.5, 1.5] | m/s |
| 2 | wz (yaw) | [-2.0, 2.0] | rad/s |

Forward range is asymmetric (3x faster forward than backward) to encourage forward progress.

### 3.6 Reward System (8 Terms)

All rewards follow Bug #29 convention: `compute -> clamp -> nan_to_num -> isfinite guard`.

| Term | Weight | Clamp Range | Purpose |
|------|--------|-------------|---------|
| `forward_velocity` | +10.0 | [-1.0, 3.0] | World-frame +X speed |
| `survival` | +1.0 | N/A (constant) | Per-step alive bonus |
| `terrain_traversal` | +2.0 | [0.0, 1.0] | Cumulative X-distance / 50m |
| `terrain_relative_height` | -2.0 | [0.0, 1.0] | Height deviation from target |
| `drag_penalty` | -1.5 | [0.0, 3.0] | Anti-crawl (low height + forward vel) |
| `cmd_smoothness` | -1.0 | [0.0, 5.0] | L2 norm of velocity command delta |
| `lateral_velocity` | -0.3 | [0.0, 2.0] | vy squared (light for obstacle avoidance) |
| `angular_velocity` | -0.5 | [0.0, 3.0] | wz squared (anti-spinning) |
| `vegetation_drag` | -0.001 | [0.0, 50.0] | Physics drag on feet + small penalty |

**Anti-crawl system:** Two-term defense prevents the robot from belly-sliding:
- `terrain_relative_height` adapts target height based on terrain roughness (0.42m flat, 0.35m rough)
- `drag_penalty` activates when body height < 0.25m AND forward velocity > 0

### 3.7 Terrain Curriculum (6 Levels x 10 Types)

| Terrain Type | Proportion | Level 1 (Easy) | Level 6 (Hard) |
|-------------|-----------|-----------------|-----------------|
| Flat | 10% | Smooth | Smooth |
| Random rough | 15% | 0.02m bumps | 0.12m bumps |
| Boxes | 15% | 2-8cm | 8-25cm |
| Stairs up | 12% | 3cm steps | 20cm steps |
| Stairs down | 8% | 3cm steps | 20cm steps |
| Wave | 10% | 5cm amplitude | 15cm amplitude |
| Discrete obstacles | 10% | 5cm | 20cm |
| Boulders | 15% | 0.1m | 0.8m |
| Friction plane | 5% | Low friction | Low friction |
| Vegetation plane | 5% | Drag forces | Drag forces |

Boulders have the highest weight (15%) as the primary navigation challenge.

### 3.8 AI Coach Integration

The optional AI Coach consults Claude Sonnet every 250 training iterations:

| Property | Value |
|----------|-------|
| API | Anthropic Claude Sonnet |
| Cost | ~$0.01/consultation, ~$2-4 per full run |
| Interval | Every 250 iterations |
| Max weight changes | 3 per consultation |
| Max delta | 20% per weight |
| Guardrails | Sign constraints, terrain-gated loosening, emergency LR halving |

### 3.9 Training Configuration

| Property | Value |
|----------|-------|
| Frozen loco backbone | Boulder V6 (4500) — best single policy |
| Environments | 512 |
| Steps per env | 24 |
| Iteration time | ~21 seconds on H100 |
| Max iterations | 5,000 |
| Save interval | 100 |
| Learning rate | 1e-4 (fixed) |
| PPO clip | 0.2 |
| Entropy coefficient | 0.01 |

**Performance optimization:** Original configuration (2048 envs, 128 steps, 64x64 depth) took 125s/iter. After reducing to 512 envs, 24 steps, and 32x32 depth, iteration time dropped to 21s — a 6x speedup with ETA of ~6 hours for 5,000 iterations.

### 3.10 Training Status

| Metric | Iter 0 | Iter 5 |
|--------|--------|--------|
| Mean reward | -0.73 | -2.41 |
| Episode length | 1.0s | — |
| Value function loss | 31.86 | — |
| Forward velocity | -0.012 | — |
| NaN crashes | None | None |

Training is actively running on the H100. The NaN issue from the initial run (vegetation_drag reward producing NaN contact forces) was resolved by adding explicit NaN guards on sensor inputs, actor outputs, and reward returns.

### 3.11 Bugs Fixed During Development

| Bug | Description | Fix |
|-----|-------------|-----|
| NaN actor mean | Depth camera returns NaN on init frames | `nan_to_num` on observation input |
| NaN vegetation_drag | Contact forces NaN early in episode | `nan_to_num` on sensor data + output clamp |
| NaN reward propagation | Single NaN reward poisons value loss | `nan_to_num` on reward in NavEnvWrapper |
| H100 deployment (22 bugs) | Import paths, API differences, config formats | Documented in H100_DEPLOYMENT_BUGS.md |

## 4. Colby — CombinedPolicyTraining

### 4.1 Architecture

Colby's approach follows the same two-brain principle as NAV_ALEX:

```
Depth Camera (64x64)
        |
        v
[CNN Encoder: 128-dim features]
        |
+ Proprioception (12 dims)
        |
        v
[Navigation Policy (ActorCriticCNN), 10 Hz]  <-- BEING TRAINED
        |
        v
Velocity Command [vx, vy, wz]
        |
        v
[Frozen Mason Hybrid Policy, 50 Hz]  <-- FROZEN (Ryan's checkpoint)
        |
        v
12 Joint Targets --> Spot Robot
```

### 4.2 Key Differences from NAV_ALEX

| Aspect | NAV_ALEX | Colby |
|--------|---------|-------|
| Depth resolution | 32x32 (1,024 dims) | 64x64 (4,096 dims) |
| Total obs dims | 1,036 | 4,108 |
| Loco backbone | Boulder V6 (4500) | Mason Hybrid (33200) |
| Code location | `NAV_ALEX/source/nav_locomotion/` | `CombinedPolicyTraining/` |
| RSL-RL compat | Direct integration | Custom adapter (`cnn_compat.py`) |
| AI Coach | Yes | No |

### 4.3 Implementation

Colby's implementation includes:

- **`train_combined.py`** (335 lines): Training entry point. Imports Alex's `nav_locomotion` package as a library. Handles Isaac Sim's DLL conflict by importing torch before AppLauncher.
- **`cnn_compat.py`** (217 lines): Adapter classes (`ActorCNNWrapper`, `CriticCNNWrapper`) that bridge RSL-RL's old `ActorCriticCNN` API to the new 5.0.1 `MLPModel` interface.
- **`run_combined_nav_loco.sh`**: Launch script supporting both local (16 envs) and H100 (2048 envs) configurations.
- **`install_prerequisites.sh`**: Dependency installer for nav_locomotion, isaaclab, rsl-rl 5.0.1, torch cu128.

### 4.4 Reward Configuration (9 Terms)

| Term | Weight | Purpose |
|------|--------|---------|
| `forward_velocity` | 10.0 | Forward progress |
| `survival` | 1.0 | Stay alive |
| `terrain_traversal` | 2.0 | Distance progress |
| `terrain_relative_height` | -2.0 | Anti-crawl |
| `drag_penalty` | -1.5 | Low-height penalty |
| `cmd_smoothness` | -1.0 | Smooth commands |
| `lateral_velocity` | -0.3 | Minimize strafing |
| `angular_velocity` | -0.5 | Minimize spinning |
| `vegetation_drag` | -0.001 | Drag physics |

### 4.5 Training Status: BLOCKED

Colby encountered a series of escalating errors during development:

| Error | Cause | Status |
|-------|-------|--------|
| DLL conflict (WinError 1114) | Isaac Sim CUDA 11 vs torch CUDA 12 | Fixed (import order) |
| CPU-only torch | Isaac Sim ships CPU torch by default | Fixed (install cu128) |
| KeyError: 'class_name' | RSL-RL 5.0.1 API break | **UNRESOLVED** |

The final blocker is RSL-RL 5.0.1's replacement of the combined `ActorCritic` class with separate `actor`/`critic` `MLPModel` objects. Colby's `cnn_compat.py` adapter classes are defined but not found by RSL-RL's `resolve_callable()` during OnPolicyRunner initialization.

**Checkpoints saved:** model_0.pt, model_50.pt, model_99.pt (from partial runs before the API error), model_final.pt. None represent meaningfully trained policies.

## 5. Cole — RL_FOLDER_VS3

### 5.1 Architecture

Cole takes a fundamentally different approach — no depth camera, using multi-layer raycasts instead:

```
48 Raycasts (16 rays x 3 height layers)
+ IMU Data (roll, pitch, acceleration)
+ Foot Contacts (4 binary)
+ Leg Joint Summary (4 averages)
+ Waypoint Info (dx, dy, distance)
+ Stage Encoding (7 one-hot)
        |
        v
[MLP Policy [256, 256, 128], 20 Hz]  <-- BEING TRAINED
        |
        v
Velocity Command [vx, vy, omega]
        |
        v
[Frozen SpotFlatTerrainPolicy, 50 Hz]  <-- FROZEN
        |
        v
Joint Targets --> Spot Robot
```

### 5.2 Observation Space (75 dimensions)

| Component | Dimensions | Description |
|-----------|-----------|-------------|
| Base velocity | 3 | vx, vy, omega |
| Heading | 2 | sin(yaw), cos(yaw) |
| IMU data | 4 | roll, pitch, accel_x, accel_y |
| Waypoint info | 3 | dx_robot, dy_robot, distance |
| Multi-layer raycasts | 48 | 16 rays x 3 heights (-0.2m, 0.0m, +0.2m) |
| Foot contacts | 4 | Binary contact per foot |
| Leg joint summary | 4 | Average joint angle per leg |
| Stage encoding | 7 | One-hot curriculum stage |

The 48-ray multi-layer raycast provides obstacle distance information at three heights, enabling the policy to distinguish between low obstacles (step over), medium obstacles (push), and tall obstacles (navigate around).

### 5.3 Action Space

| Component | Range | Unit |
|-----------|-------|------|
| vx (forward) | [-0.5, 5.0] | m/s |
| vy (lateral) | [-0.5, 0.5] | m/s |
| omega (yaw) | [-1.5, 1.5] | rad/s |

### 5.4 Seven-Stage Curriculum

Cole's most distinctive contribution is a task-based curriculum with 7 progressive stages:

| Stage | Name | Duration | Objective |
|-------|------|----------|-----------|
| 0 | Stability Foundation | 180s | Remain stable, no falling (5% obstacles) |
| 1 | Enhanced Stability | 180s | Stable with 10% obstacle density |
| 2 | Object Pushing | 300s | Push 5 lightweight objects 1m+ each |
| 3 | Short-Range Navigation | Varies | Capture 25 waypoints at 5m intervals |
| 4 | Medium-Range Navigation | Varies | Capture 25 waypoints at 10m intervals |
| 5 | Long-Range Navigation | Varies | Capture 25 waypoints at 20m intervals |
| 6 | Expert Navigation | Varies | First=20m, subsequent=40m waypoints |

**Stage advancement requires both:**
- 80% success rate over last 100 episodes
- Minimum 100 iterations at current stage (50 for stages 0-1)

### 5.5 Object Pushing (Stage 2) — Unique Feature

Cole's system uniquely trains the robot to interact with objects:

| Object Type | Mass | Behavior |
|-------------|------|----------|
| Light | < 0.45 kg | Pushable by Spot |
| Medium | 32.7 kg (Spot's mass) | Pushable with effort |
| Heavy | > 65.4 kg | Immovable, navigate around |

**Push rewards:**
- Exploration: 8.0 (first contact with light object)
- Sustained: 12.0 (maintaining contact while moving)
- Success: 15.0 (1m+ push distance achieved)
- Wasted effort: -1.0 (high contact without progress)

### 5.6 Reward Design

**Core waypoint navigation rewards:**

| Term | Weight | Purpose |
|------|--------|---------|
| Waypoint capture | +25.0 | Per waypoint reached |
| Progress shaping | 10.0 | Distance-based approach reward |
| Heading alignment | 0.2 | Facing waypoint direction |
| Speed reward | 0.3 | Fast movement on clear path |
| Fall penalty | -100.0 | Catastrophic fall |
| Boundary penalty | -5.0 | Soft arena boundary |
| Stillness penalty | -1.0 | Forces locomotion learning |
| Timeout penalty | -100.0 | Episode time limit |

**Scoring system:**
- Initial points: 300.0
- Time decay: 1.0 point/second (stages 4+ only)
- Waypoint bonus: 25.0 per waypoint captured
- Efficiency bonus: remaining_time x 0.5 when objectives complete

### 5.7 Arena Configuration

| Property | Value |
|----------|-------|
| Shape | Circular, 50m diameter |
| Robot start | Center (0, 0, 0.7) |
| Fall threshold | 0.25m z-height |
| Waypoint count | 25 (A through Y) |
| Waypoint capture radius | 0.5m |
| Boundary margin | 2.0m |

### 5.8 Three Learning Rate Configurations

| Config | Learning Rate | Push Exploration | Push Success |
|--------|-------------|-----------------|--------------|
| Aggressive | 2.0e-4 | 10.0 | 18.0 |
| Moderate | 1.0e-4 | 8.0 | 15.0 |
| Conservative | 5.0e-5 | 6.0 | 12.0 |

### 5.9 Network Architecture

| Property | Value |
|----------|-------|
| Input | 75 dimensions |
| Hidden layers | [256, 256, 128] |
| Activation | ReLU |
| Output | 3 dimensions (vx, vy, omega) |
| Action std | 0.3 (fixed Gaussian) |

### 5.10 PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| Clip parameter | 0.2 |
| Value coefficient | 0.5 |
| Entropy coefficient | 0.01 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Max gradient norm | 0.5 |
| Target KL | 0.015 |
| PPO epochs | 10 |
| Steps per iteration | 6,000 |

### 5.11 Evolution from VS2 to VS3

| Aspect | VS2 | VS3 |
|--------|-----|-----|
| Observation dims | 32 | 75 |
| Raycasts | 16 (single layer) | 48 (3 layers) |
| Foot contacts | No | Yes (4 dims) |
| Leg joints | No | Yes (4 dims) |
| IMU data | No | Yes (4 dims) |
| Stages | 8 | 7 (consolidated) |
| Logging | Detailed per-component | Simplified |

### 5.12 Testing Environments

Two testing environments validate trained policies:

- **`Testing_Environment.py`**: Tests VS2 policies (32-dim obs) in circular 50m arena with 25 waypoints and mixed obstacles
- **`VS3_Testing_Environment.py`**: Tests VS3 policies (75-dim obs) with the same arena and obstacle configuration

## 6. Comparative Analysis

### 6.1 Sensing Trade-offs

| Approach | Sensing | Range | Resolution | Computation |
|----------|---------|-------|-----------|-------------|
| Alex (CNN) | Depth camera | 30m | 32x32 = 1024 pixels | Heavy (raycaster bottleneck) |
| Cole (MLP) | Raycasts | ~5m | 48 rays | Light (fast inference) |

Alex's depth camera provides long-range terrain awareness (30m lookahead) but is computationally expensive. Cole's raycasts are fast but limited to ~5m range, requiring reactive rather than deliberate navigation.

### 6.2 Curriculum Philosophy

| Approach | Curriculum Type | Key Idea |
|----------|---------------|----------|
| Alex | Terrain difficulty (6 levels) | Progressive terrain complexity, auto-promote/demote |
| Colby | Same as Alex (shared codebase) | Same terrain curriculum |
| Cole | Task complexity (7 stages) | Stability -> pushing -> navigation progression |

Cole's task-based curriculum is unique in starting with stability training before any navigation objectives, and including an explicit object-pushing training phase.

### 6.3 Training Status Summary

| Approach | Status | Iterations | Key Result |
|----------|--------|-----------|------------|
| Alex (NAV_ALEX) | **Active training** | ~5 of 5,000 | NaN fixed, 21s/iter, healthy metrics |
| Colby | **Blocked** | 0 (partial runs) | RSL-RL 5.0.1 API incompatibility |
| Cole | **Active development** | Testing 3 LR configs | VS3 architecture finalized |
