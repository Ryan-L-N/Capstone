# Phase 2 Deliverables — 4-Environment Comparative Evaluation

**Team:** AI2C Tech Capstone — MS for Autonomy
**Date:** February 19, 2026
**Version:** 1.0

---

## 1. Data Dictionary

### Purpose

This data dictionary defines every variable recorded during the 4-environment comparative evaluation of Boston Dynamics Spot locomotion policies. Each evaluation episode produces one JSON record exported to JSONL files. The dictionary is structured so a non-team member can clearly interpret the data.

### Data Variables

| Variable | Descriptive Name | Definition | Type | Range / Values | Required | Notes |
|---|---|---|---|---|---|---|
| `episode_id` | Episode Identifier | Unique ID encoding environment, policy, and episode number | string | Format: `{env}_{policy}_ep{NNNN}` (e.g., `friction_flat_ep0042`) | Yes | Length: 20-28 characters |
| `policy` | Policy Type | Which locomotion policy controlled the robot | string | `"flat"` or `"rough"` | Yes | flat = NVIDIA pre-trained baseline; rough = custom 48h H100-trained |
| `environment` | Environment Name | Which of the 4 test environments was used | string | `"friction"`, `"grass"`, `"boulder"`, `"stairs"` | Yes | Each environment has 5 progressive difficulty zones |
| `completion` | Course Completed | Whether the robot traversed the full 50m arena (x >= 49.0m) | boolean | `true` / `false` | Yes | Threshold: `COMPLETION_X = 49.0` meters |
| `progress` | Max Forward Progress | Maximum x-position (meters) achieved during the episode | numeric (float) | 0.0 to 50.0 | Yes | Rounded to 3 decimal places |
| `zone_reached` | Highest Zone Reached | Highest difficulty zone the robot entered (1 = easiest, 5 = hardest) | numeric (int) | 1 to 5 | Yes | Derived from `progress`: zone = floor(x / 10) + 1, capped at 5 |
| `time_to_complete` | Completion Time | Simulation time (seconds) when robot first crossed x = 49.0m | numeric (float) or null | 0.0 to 120.0 | Optional | `null` if robot did not complete the course |
| `stability_score` | Composite Stability Score | Weighted sum of body oscillation metrics (lower = more stable) | numeric (float) | 0.0 to ~5.0 | Yes | Formula: `1.0*mean_roll + 1.0*mean_pitch + 10.0*height_variance + 0.5*mean_ang_vel` |
| `mean_roll` | Mean Absolute Roll | Average absolute roll angle across all control steps | numeric (float) | 0.0 to ~1.57 (radians) | Yes | Rounded to 6 decimal places |
| `mean_pitch` | Mean Absolute Pitch | Average absolute pitch angle across all control steps | numeric (float) | 0.0 to ~1.57 (radians) | Yes | Rounded to 6 decimal places |
| `height_variance` | Body Height Variance | Variance of body height above expected ground surface across all steps | numeric (float) | 0.0 to ~1.0 (m^2) | Yes | Uses `ground_height_fn` for stairs; 0.0 baseline for flat envs |
| `mean_ang_vel` | Mean Angular Velocity | Average magnitude of the 3-axis angular velocity vector | numeric (float) | 0.0 to ~10.0 (rad/s) | Yes | `||[wx, wy, wz]||` per step, then averaged |
| `fall_detected` | Fall Detected | Whether the robot's body dropped below the fall threshold | boolean | `true` / `false` | Yes | Threshold: body height < 0.15m above expected ground surface |
| `fall_location` | Fall X-Position | X-position where the fall was first detected | numeric (float) or null | 0.0 to 50.0 | Optional | `null` if no fall occurred |
| `fall_zone` | Fall Zone | Zone in which the fall occurred (1-indexed) | numeric (int) or null | 1 to 5 | Optional | `null` if no fall occurred |
| `mean_velocity` | Mean Forward Velocity | Average forward (x-axis) velocity across all control steps | numeric (float) | -2.0 to 3.0 (m/s) | Yes | Negative values indicate backwards movement |
| `total_energy` | Total Actuator Energy | Cumulative sum of `|torque * joint_velocity|` across all joints and steps | numeric (float) | 0.0 to ~100000.0 | Yes | 0.0 when torque data is unavailable from the policy |
| `episode_length` | Episode Duration | Simulation time elapsed from first to last recorded step | numeric (float) | 0.0 to 120.0 (seconds) | Yes | Max 120.0s (6000 control steps at 50 Hz) |

### Experimental Design Variables (Independent)

| Variable | Definition | Type | Values | Notes |
|---|---|---|---|---|
| Policy | Locomotion controller being tested | categorical | flat, rough | 2 levels |
| Environment | Terrain challenge type | categorical | friction, grass, boulder, stairs | 4 levels |
| Zone | Progressive difficulty level within each environment | ordinal | 1-5 | Each zone spans 10m |
| Episode | Trial replicate number | numeric (int) | 0 to N-1 | 100 per combination (production) |

### Zone Difficulty Parameters

**Friction:** Static friction coefficient from 0.90 (60-grit sandpaper) down to 0.05 (oil on polished steel)

**Grass:** Drag coefficient from 0.5 (light fluid) up to 20.0 (dense brush), with proxy stalk cylinders

**Boulder:** Polyhedra edge length from 3-5 cm (gravel) up to 80-120 cm (large boulders); shapes: D8, D10, D12, D20

**Stairs:** Step riser height from 3 cm (access ramp) up to 23 cm (maximum challenge); 33 steps per zone, 5 transition steps at zone boundaries

---

## 2. Exploratory Data Analysis

### Purpose

This EDA examines the structure and quality of the per-episode metrics collected during the 4-environment evaluation to confirm data integrity before drawing conclusions about policy performance.

### Data Distributions

**Completion (boolean):** Expect highly unequal groups. On easy terrain (friction zone 1-2), both policies should complete most episodes. On harder terrain (stairs zone 4-5, boulder zone 5), completions should be rare. This asymmetry is expected and central to the research question.

**Progress (float, 0-50m):** Expected to be right-skewed for hard environments (many episodes cluster at early zones) and left-skewed or uniform for easy environments. Histograms per (env, policy) should show this pattern.

**Zone Reached (int, 1-5):** Ordinal; report counts per zone. Unequal groups expected — most runs should reach zone 2-3, fewer reach zone 4-5.

**Stability Score (float, composite):** Expected to be right-skewed (most episodes are reasonably stable, with a tail of high-instability episodes before falls).

**Mean Velocity (float):** Should cluster around 0.8-1.0 m/s for successful traversals (target vx = 1.0 m/s) and drop toward 0 for stalled/fallen episodes.

**Episode Length (float, 0-120s):** Bimodal — episodes that complete early (< 60s) and timeouts at 120s. Falls truncate episodes.

**Fall Location / Zone:** Categorical counts — report fall frequency per zone per environment. Expect falls to concentrate in higher-numbered zones.

### Missing Data

| Variable | Can Be Missing? | Reason | Handling |
|---|---|---|---|
| `time_to_complete` | Yes (`null`) | Only populated when robot reaches x >= 49.0m | Expected missing — no imputation needed |
| `fall_location` | Yes (`null`) | Only populated when a fall is detected | Expected missing — no imputation needed |
| `fall_zone` | Yes (`null`) | Only populated when a fall is detected | Expected missing — no imputation needed |
| `total_energy` | Always present | 0.0 when torque data unavailable | Flag 0.0 values — may indicate torque passthrough unavailable |
| All other fields | Never missing | Always recorded by MetricsCollector | N/A |

The three nullable fields (`time_to_complete`, `fall_location`, `fall_zone`) are conditionally missing by design. There is no systematic cause for unexpected missing data — every control step records all required sensor readings.

### Outliers

**Progress:** Outliers may appear if the robot gets stuck in geometry (progress = 0) or glitches through terrain (progress >> expected). Check for episodes with progress < 1.0m (stuck at spawn) or progress > 50.0m (impossible — arena is 50m).

**Stability Score:** Extreme values (> 2.0) likely correspond to tumbling episodes just before fall detection. These are legitimate data reflecting policy failure, not measurement errors.

**Episode Length:** Episodes significantly shorter than ~20s with no fall suggest potential simulation errors (should be investigated). Episodes at exactly 120.0s are timeouts (expected).

**Mean Velocity:** Negative values are possible (robot walks backward) and legitimate. Values outside [-2.0, 3.0] would indicate a measurement error (command clamps enforce this range).

### Descriptive Statistics

Per (environment, policy) group, report:
- **Progress:** min, mean, median, max, std
- **Stability Score:** min, mean, median, max, std
- **Completion Rate:** count(completion=true) / N
- **Fall Rate:** count(fall_detected=true) / N
- **Mean Velocity:** min, mean, median, max, std
- **Zone Distribution:** count of episodes reaching each zone (1-5)
- **Episode Length:** min, mean, median, max, std

### Statistical Tests

- **Welch's t-test:** Compare mean progress between flat vs rough policies per environment (unequal variance assumed)
- **Cohen's d:** Effect size for progress differences
- **Two-proportion z-test:** Compare completion rates between flat vs rough per environment
- **Significance level:** alpha = 0.05

### Visualizations

1. **Completion Rate Bar Chart:** Grouped bars (flat vs rough) per environment
2. **Progress Box Plots:** One subplot per environment, flat vs rough side by side
3. **Fall Zone Heatmap:** Rows = (env, policy), columns = zones 1-5, color = fall percentage
4. **Stability by Zone Line Plot:** Mean stability score vs zone, one line per (env, policy) combination

---

## 3. Data Schema

### Purpose

This schema defines the JSON structure for each episode record, enabling automated validation of output data. The format is JSONL (one JSON object per line).

### File Naming Convention

```
{environment}_{policy}_episodes.jsonl
```

Examples: `friction_flat_episodes.jsonl`, `stairs_rough_episodes.jsonl`

### JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Capstone Episode Record",
  "description": "Per-episode evaluation metrics for the 4-environment Spot locomotion comparative test.",
  "type": "object",
  "required": [
    "episode_id", "policy", "environment", "completion", "progress",
    "zone_reached", "stability_score", "mean_roll", "mean_pitch",
    "height_variance", "mean_ang_vel", "fall_detected", "mean_velocity",
    "total_energy", "episode_length"
  ],
  "properties": {
    "episode_id": {
      "type": "string",
      "pattern": "^(friction|grass|boulder|stairs)_(flat|rough)_ep[0-9]{4}$",
      "description": "Unique identifier: {env}_{policy}_ep{NNNN}"
    },
    "policy": {
      "type": "string",
      "enum": ["flat", "rough"],
      "description": "Locomotion policy type"
    },
    "environment": {
      "type": "string",
      "enum": ["friction", "grass", "boulder", "stairs"],
      "description": "Test environment name"
    },
    "completion": {
      "type": "boolean",
      "description": "Whether robot reached x >= 49.0m"
    },
    "progress": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 55.0,
      "description": "Maximum x-position achieved (meters)"
    },
    "zone_reached": {
      "type": "integer",
      "minimum": 1,
      "maximum": 5,
      "description": "Highest difficulty zone entered"
    },
    "time_to_complete": {
      "type": ["number", "null"],
      "minimum": 0.0,
      "maximum": 120.0,
      "description": "Sim seconds to reach x=49m, or null if incomplete"
    },
    "stability_score": {
      "type": "number",
      "minimum": 0.0,
      "description": "Composite stability metric (lower = more stable)"
    },
    "mean_roll": {
      "type": "number",
      "minimum": 0.0,
      "description": "Mean absolute roll angle (radians)"
    },
    "mean_pitch": {
      "type": "number",
      "minimum": 0.0,
      "description": "Mean absolute pitch angle (radians)"
    },
    "height_variance": {
      "type": "number",
      "minimum": 0.0,
      "description": "Variance of body height above ground (m^2)"
    },
    "mean_ang_vel": {
      "type": "number",
      "minimum": 0.0,
      "description": "Mean angular velocity magnitude (rad/s)"
    },
    "fall_detected": {
      "type": "boolean",
      "description": "Whether body dropped below 0.15m above ground"
    },
    "fall_location": {
      "type": ["number", "null"],
      "minimum": 0.0,
      "maximum": 50.0,
      "description": "X-position of fall, or null if no fall"
    },
    "fall_zone": {
      "type": ["integer", "null"],
      "minimum": 1,
      "maximum": 5,
      "description": "Zone where fall occurred, or null if no fall"
    },
    "mean_velocity": {
      "type": "number",
      "description": "Mean forward velocity (m/s)"
    },
    "total_energy": {
      "type": "number",
      "minimum": 0.0,
      "description": "Cumulative |torque * joint_vel| energy (Nm*rad/s)"
    },
    "episode_length": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 125.0,
      "description": "Episode duration in simulation seconds"
    }
  },
  "additionalProperties": false
}
```

### Validation

The schema can be validated using Python's `jsonschema` library:

```python
import json
import jsonschema

with open("schema.json") as f:
    schema = json.load(f)

with open("friction_flat_episodes.jsonl") as f:
    for line in f:
        record = json.loads(line.strip())
        jsonschema.validate(record, schema)
```

### Example Record

```json
{
  "episode_id": "friction_flat_ep0000",
  "policy": "flat",
  "environment": "friction",
  "completion": false,
  "progress": 11.819,
  "zone_reached": 2,
  "time_to_complete": null,
  "stability_score": 0.096244,
  "mean_roll": 0.005696,
  "mean_pitch": 0.00688,
  "height_variance": 0.000111,
  "mean_ang_vel": 0.165116,
  "fall_detected": false,
  "fall_location": null,
  "fall_zone": null,
  "mean_velocity": 0.9849,
  "total_energy": 0.0,
  "episode_length": 119.98
}
```

---

## 4. Model Documentation

### Overview

This project evaluates two locomotion policies for the Boston Dynamics Spot quadruped robot in simulation:

1. **Flat Terrain Policy (Baseline):** NVIDIA's pre-trained locomotion controller bundled with Isaac Sim 5.1.0. It uses 48-dimensional proprioceptive observations (body velocity, gravity, joint positions/velocities, previous actions) to produce 12 joint position offsets. This policy was designed for flat, obstacle-free environments and serves as the control group.

2. **Rough Terrain Policy (Experimental):** A custom-trained PPO reinforcement learning policy that extends the flat baseline with 187-dimensional height scan observations (total: 235 dims). Trained for 48 hours (30,000 iterations) on an NVIDIA H100 NVL GPU using Isaac Lab's RSL-RL framework, this policy learns to perceive and adapt to uneven terrain through a learned mapping from local height measurements to gait adjustments.

The research question is whether the additional height-scan perception and rough-terrain training significantly improves traversal performance across progressively difficult environments compared to the proprioception-only baseline.

### Specifications

**Flat Terrain Policy:**
- Source: `omni.isaac.quadruped.robots.SpotFlatTerrainPolicy` (NVIDIA Isaac Sim 5.1.0)
- Observations: 48 dimensions (proprioception only)
- Actions: 12 joint position offsets
- Architecture: Pre-compiled NVIDIA module (not modifiable)

**Rough Terrain Policy:**
- Source: `src/spot_rough_terrain_policy.py` (custom wrapper)
- Architecture: MLP with 4 layers — `[235 -> 512 -> 256 -> 128 -> 12]`
- Activation: ELU between each hidden layer
- Observations: 235 dimensions
  - `[0:3]` Base linear velocity (body frame)
  - `[3:6]` Base angular velocity (body frame)
  - `[6:9]` Projected gravity vector
  - `[9:12]` Velocity commands (vx, vy, omega_z)
  - `[12:24]` Joint positions relative to default
  - `[24:36]` Joint velocities
  - `[36:48]` Previous action
  - `[48:235]` Height scan (17x11 grid, 0.1m resolution, 1.6m x 1.0m)
- Actions: 12 joint position offsets, scaled by 0.25
- Decimation: 10 (policy runs at 50 Hz, physics at 500 Hz)
- PD Gains: Kp = 60.0, Kd = 1.5 (all joints)
- Effort Limits: Hips = 45 Nm, Knees = 110 Nm (angle-dependent via RemotizedPD lookup table)
- Solver: 4 position iterations, 0 velocity iterations
- Checkpoint: `model_29999.pt` (6.6 MB, 30,000 training iterations)

### Model Run

**Platform:** NVIDIA H100 NVL (95,830 MiB VRAM), CUDA 13.0, Driver 580.126.16

**Software Stack:**
- NVIDIA Isaac Sim 5.1.0
- Isaac Lab 0.54.2
- PyTorch 2.7.0+cu128
- Python 3.11.14 (Miniconda `env_isaaclab`)
- Ubuntu Linux (ai2ct2 server)

**Execution:**
- Headless mode (no GUI rendering) via `./isaaclab.sh -p run_capstone_eval.py --headless`
- Per-episode wall time: ~41 seconds (120s simulation time at faster-than-realtime)
- Full production run: 100 episodes x 8 combinations = ~9 hours
- GPU memory usage: ~4.5 GB per run (well within 96 GB capacity)

**Cost:** University research GPU allocation (no direct monetary cost). H100 NVL compute time breakdown:
- Training the rough terrain policy: ~48 hours (30,000 PPO iterations, 8,192 parallel envs)
- Production evaluation: ~9 hours (100 episodes x 8 combinations)
- Debug validation: ~34 minutes (5 episodes x 8 combinations)
- Total H100 usage: ~58 hours

### Training

The rough terrain policy was trained using Isaac Lab's RSL-RL (PPO) framework over 48 hours on the NVIDIA H100 NVL. The training configuration files are located in `ARL_DELIVERY/05_Training_Package/`.

#### Algorithm and Scale

- **Algorithm:** Proximal Policy Optimization (PPO) [3]
- **Framework:** RSL-RL (Isaac Lab's RL training library) [4]
- **Task:** `Isaac-Velocity-Rough-Spot-v0` (Isaac Lab built-in rough terrain locomotion)
- **Duration:** ~48 hours wall-clock on NVIDIA H100 NVL
- **Iterations:** 30,000 (final checkpoint: `model_29999.pt`)
- **Parallel Environments:** 8,192 simultaneous robots
- **Steps per Iteration:** 196,608 (8,192 envs x 24 steps)
- **Total Timesteps:** ~3.3 billion
- **Throughput:** ~36,000 steps/second
- **GPU Temperature:** ~49C (safe operating range)

#### PPO Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Learning Rate | 3e-4 | Adaptive schedule via desired KL |
| Desired KL | 0.01 | Target KL divergence for adaptive LR |
| Entropy Coefficient | 0.008 | Increased from default 0.005 for exploration |
| GAE Lambda | 0.95 | Generalized Advantage Estimation |
| Discount (gamma) | 0.99 | Standard |
| PPO Clip Range | 0.2 | Standard |
| Learning Epochs | 5 | Gradient updates per iteration |
| Mini-batches | 8 | Batch size: 196,608 / 8 = 24,576 |
| Max Gradient Norm | 1.0 | Gradient clipping |
| Initial Noise Std | 0.8 | Action exploration (decreased from 1.0) |
| Observation Normalization | Disabled | Critical: no running-mean normalization |
| Save Interval | 500 | 60 checkpoints over 30k iterations |

#### Neural Network Architecture

```
Actor:  235 -> [512, ELU] -> [256, ELU] -> [128, ELU] -> 12
Critic: 235 -> [512, ELU] -> [256, ELU] -> [128, ELU] -> 1
```

- Total parameters: ~500K
- No observation or action normalization layers
- Standard architecture consistent with ANYmal-C and Unitree Go2 quadruped training

#### Observation Space (235 Dimensions)

| Component | Dims | Noise | Description |
|---|---|---|---|
| base_lin_vel | 3 | +/-0.1 | Body-frame linear velocity |
| base_ang_vel | 3 | +/-0.1 | Body-frame angular velocity |
| projected_gravity | 3 | +/-0.05 | Gravity projected into body frame |
| velocity_commands | 3 | None | Target [vx, vy, omega_z] |
| joint_pos (relative) | 12 | +/-0.05 | Joint angles relative to default stance |
| joint_vel (relative) | 12 | +/-0.5 | Joint velocities |
| last_action | 12 | None | Previous policy output |
| height_scan | 187 | +/-0.1, clipped [-1, 1] | 17x11 grid, 0.1m resolution, 1.6m x 1.0m |
| **Total** | **235** | | |

#### Reward Function (14 Terms)

The reward function was tuned for long-duration training with stronger penalties than standard configurations to produce cleaner, more efficient gaits.

**Positive (task) rewards:**

| Term | Weight | Function |
|---|---|---|
| Gait | +10.0 | Trot enforcer: diagonal pairs synchronized (FL-HR, FR-HL) |
| Linear velocity tracking | +7.0 | Exponential kernel on XY velocity error |
| Angular velocity tracking | +5.0 | Exponential kernel on yaw rate error |
| Air time | +5.0 | Rewards appropriate swing/stance phase timing |
| Foot clearance | +2.5 | Rewards foot height during swing phase (target 0.12m for stair clearance) |

**Negative (penalty) rewards:**

| Term | Weight | Purpose |
|---|---|---|
| Base orientation | -5.0 | Penalizes roll/pitch deviation from upright |
| Base motion | -3.0 | Penalizes vertical bouncing and lateral sway |
| Action smoothness | -2.0 | Penalizes jerky action changes between steps |
| Foot slip | -1.0 | Penalizes feet sliding during ground contact |
| Joint position deviation | -1.0 | Penalizes deviation from default stance |
| Air time variance | -1.0 | Penalizes asymmetric gait patterns |
| Joint velocity | -0.02 | Penalizes high joint velocities |
| Joint torques | -0.002 | Penalizes torque usage for energy efficiency |
| Joint acceleration | -0.0005 | Penalizes joint acceleration for smoothness |

#### Terrain Curriculum

Training uses NVIDIA's `ROUGH_TERRAINS_CFG` with 6 procedurally generated terrain types:

1. Flat ground (baseline)
2. Random uniform roughness
3. Slope terrain (ascending/descending)
4. Stair terrain (ascending/descending)
5. Pyramid stairs
6. Discrete obstacles (boxes)

Terrain difficulty progresses automatically: robots with longer episodes are promoted to harder terrain, while robots that fall frequently are moved to easier terrain. Levels range from 0 to 5+, with the curriculum advancing throughout training.

#### Domain Randomization

| Parameter | Range | When Applied | Purpose |
|---|---|---|---|
| Surface friction | Static: [0.5, 1.25], Dynamic: [0.4, 1.0] | Startup | Surface variation |
| Base mass offset | +/-5.0 kg | Startup | Payload robustness |
| Joint position reset | +/-0.2 rad around default | Episode reset | Varied start poses |
| Joint velocity reset | +/-2.5 rad/s | Episode reset | Varied dynamics |
| Base pose reset | +/-0.5m XY, +/-pi yaw | Episode reset | Random spawn location |
| External force/torque | +/-3.0 N force, +/-1.0 Nm torque | Episode reset | Perturbation handling |
| Push perturbation | +/-0.5 m/s XY velocity | Every 10-15s | Recovery training |

#### Training Phases

**Phase 1 — Foundation (iterations 0-10,000):** Learn to stand, walk, and follow velocity commands. Terrain starts at easy levels. Gait reward dominates. High entropy encourages exploration.

**Phase 2 — Rough Terrain Mastery (iterations 10,000-20,000):** Climb stairs, traverse slopes, navigate obstacles. Height scan becomes critical for terrain awareness. Foot clearance reward drives high stepping over obstacles.

**Phase 3 — Robustness and Efficiency (iterations 20,000-30,000):** Polish gait quality, minimize energy usage. Handle push perturbations and recover from disturbances. Action smoothness penalties eliminate jitter. Most environments running on difficult terrain by this phase.

#### Training Monitoring

| Metric | Healthy Range | Red Flag |
|---|---|---|
| mean_reward | Increasing trend | Flatline after 5k iterations |
| mean_episode_length | Growing toward 600 steps (20s) | Dropping below 200 steps (4s) |
| terrain_levels | Increasing from 0 toward 5+ | Stuck at 0-1 after 10k iterations |
| policy_loss | Decreasing trend | NaN or diverging |
| Episode_Termination/timeout | >30% by end of training | <10% indicates frequent falling |
| Episode_Termination/body_contact | Decreasing below 70% | Stuck above 90% after 5k |

#### Physics and Control Configuration

| Parameter | Value |
|---|---|
| Physics timestep | 1/500 s (500 Hz) |
| Control frequency | 1/50 s (50 Hz, decimation = 10) |
| Action scale | 0.25 |
| Observation clipping | [-100, 100] |
| Action clipping | [-100, 100] |
| PD gains | Kp = 60.0, Kd = 1.5 |
| Hip effort limit | 45 Nm (DelayedPD actuator) |
| Knee effort limit | angle-dependent, max 113 Nm (RemotizedPD lookup table) |
| Joint velocity limit | 12.0 rad/s |
| Joint friction | 0.0 (training config) |
| Joint armature | 0.0 (training config) |
| Solver iterations | 4 position, 0 velocity |
| Self-collision | Enabled |
| Max depenetration velocity | 1.0 m/s |

#### Dataset and Reproduction

There is no fixed dataset — the training environment procedurally generates randomized rough terrain at each episode reset. Domain randomization (friction, mass, forces) provides additional diversity. To reproduce:

1. Clone the IsaacLab repository and install RSL-RL
2. Copy training configs from `ARL_DELIVERY/05_Training_Package/`
3. Run: `isaaclab.sh -p train_48h_spot.py --task Isaac-Velocity-Rough-Spot-v0 --headless --num_envs 8192 --max_iterations 30000`
4. Monitor via TensorBoard: `tensorboard --logdir logs/rsl_rl/spot_rough/`
5. Final checkpoint: `logs/rsl_rl/spot_rough/48h_run/model_29999.pt`

### Evaluation

**Procedure:** Each (environment, policy) combination runs N episodes. The robot spawns at (0, 15, 0.6) facing +X and follows 6 waypoints along the arena centerline at vx = 1.0 m/s. Episodes terminate on: completion (x >= 49m), fall (body < 0.15m above ground), or timeout (120s).

**Metrics Collected Per Episode:**
- Completion (boolean) and time to complete
- Maximum forward progress (meters) and highest zone reached
- Stability score (composite: roll + pitch + height variance + angular velocity)
- Fall detection with location and zone
- Mean forward velocity and total actuator energy

**Analysis Pipeline:**
1. Raw data: JSONL files per (env, policy) combination
2. Summary statistics: `reporter.py` computes per-group aggregates
3. Statistical tests: Welch's t-test (progress), two-proportion z-test (completion rate), Cohen's d (effect size)
4. Visualizations: Completion rate bars, progress boxplots, fall zone heatmaps, stability-by-zone curves

**Evaluation Scale:**
- Debug validation: 5 episodes x 8 combinations (40 total) — confirms pipeline integrity
- Production: 100 episodes x 8 combinations (800 total) — statistical significance

**Preliminary Debug Results (5 episodes each, all 8 combinations):**
All combinations completed without crashes. Example: `friction_flat` achieved mean progress of 12.0m (zone 2), mean stability 0.096, no falls detected, mean velocity 0.99 m/s, episode length 119.98s (timeout). Full production results pending.

### References

[1] NVIDIA, "Isaac Sim Documentation," 2025. [Online]. Available: https://docs.omniverse.nvidia.com/isaacsim/latest/

[2] NVIDIA, "Isaac Lab Documentation," 2025. [Online]. Available: https://isaac-sim.github.io/IsaacLab/

[3] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," arXiv preprint arXiv:1707.06347, 2017.

[4] N. Rudin, D. Hoeller, P. Reist, and M. Hutter, "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning," Proc. Conf. Robot Learn. (CoRL), 2022.

[5] Boston Dynamics, "Spot Robot," 2024. [Online]. Available: https://bostondynamics.com/products/spot/

---

## Reflection

Building this deliverable required synthesizing implementation details from across the entire codebase into standardized documentation formats. The data dictionary exercise clarified which variables are required vs optional and exposed that `total_energy` reports 0.0 when torque data is unavailable — a limitation worth noting in the final analysis. The JSON schema formalization revealed that our JSONL output already conforms to a clean, validatable structure with no ambiguous types.

The model documentation — particularly the 48-hour training section — forced us to consolidate scattered configuration files (reward weights, PPO hyperparameters, domain randomization ranges, terrain curriculum) into a single reference. This process revealed that several reward weights had been tuned specifically for long-duration training (e.g., action smoothness increased from -1.0 to -2.0, joint torques penalty increased 4x) compared to standard Isaac Lab defaults. Documenting the 14-term reward function and 7 domain randomization parameters makes the training fully reproducible. The model documentation also forced us to articulate the exact sim-to-sim gap between Isaac Lab training (explicit PD actuators with RemotizedPD knee lookup) and standalone deployment (PhysX position drive), which was a major source of debugging time during development.

## Changes to Previous Deliverables

- None at this time. This is the initial Phase 2 submission.
