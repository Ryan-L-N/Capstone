# Phase 2 Deliverables — 5-Environment Comparative Evaluation

**Team:** AI2C Tech Capstone — MS for Autonomy
**Date:** February 19, 2026
**Version:** 2.6 (Updated March 7, 2026)

---

## 1. Data Dictionary

### Purpose

This data dictionary defines every variable recorded during the 5-environment comparative evaluation of Boston Dynamics Spot locomotion policies. Each evaluation episode produces one JSON record exported to JSONL files. The dictionary is structured so a non-team member can clearly interpret the data.

### Data Variables

| Variable | Descriptive Name | Definition | Type | Range / Values | Required | Notes |
|---|---|---|---|---|---|---|
| `episode_id` | Episode Identifier | Unique ID encoding environment, policy, and episode number | string | Format: `{env}_{policy}_ep{NNNN}` (e.g., `friction_flat_ep0042`) | Yes | Length: 20-30 characters |
| `policy` | Policy Type | Which locomotion policy controlled the robot | string | `"flat"` or `"rough"` | Yes | flat = NVIDIA pre-trained baseline; rough = 4-phase curriculum-trained |
| `environment` | Environment Name | Which of the 5 test environments was used | string | `"friction"`, `"grass"`, `"boulder"`, `"stairs"`, `"obstacle"` | Yes | Each environment has 5 progressive difficulty zones |
| `completion` | Course Completed | Whether the robot traversed the full arena (x >= 49.0m for linear courses, or reached goal for obstacle) | boolean | `true` / `false` | Yes | Threshold: `COMPLETION_X = 49.0` meters (linear) or goal proximity (obstacle) |
| `progress` | Max Forward Progress | Maximum x-position (meters) achieved during the episode | numeric (float) | 0.0 to 50.0 | Yes | Rounded to 3 decimal places |
| `zone_reached` | Highest Zone Reached | Highest difficulty zone the robot entered (1 = easiest, 5 = hardest) | numeric (int) | 1 to 5 | Yes | Derived from `progress`: zone = floor(x / 10) + 1, capped at 5 |
| `time_to_complete` | Completion Time | Simulation time (seconds) when robot first crossed completion threshold | numeric (float) or null | 0.0 to 120.0 | Optional | `null` if robot did not complete the course |
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
| Environment | Terrain challenge type | categorical | friction, grass, boulder, stairs, obstacle | 5 levels |
| Zone | Progressive difficulty level within each environment | ordinal | 1-5 | Each zone spans 10m (linear) or 20m radius (obstacle) |
| Episode | Trial replicate number | numeric (int) | 0 to N-1 | 100 per combination (production) |

### Zone Difficulty Parameters

**Friction:** Static friction coefficient from 0.90 (60-grit sandpaper) down to 0.05 (oil on polished steel)

**Grass:** Drag coefficient from 0.5 (light fluid) up to 20.0 (dense brush), with proxy stalk cylinders

**Boulder:** Polyhedra edge length from 3-5 cm (gravel) up to 80-120 cm (large boulders); shapes: D8, D10, D12, D20

**Stairs:** Step riser height from 3 cm (access ramp) up to 23 cm (maximum challenge); 33 steps per zone, 5 transition steps at zone boundaries

**Obstacle:** 100m x 100m enclosed field with 360 randomly placed obstacles (100 large furniture + 250 small clutter + 5 cars + 5 trucks). Fixed start position at (-45, 0), randomized goal at 75m+ distance. Obstacle types: couch, chair, table, shelf, ottoman, bed, cabinet, small_clutter, car, truck. Zones defined by Euclidean distance from start (5 x 20m bands: zone 1 = 0-20m, zone 2 = 20-40m, zone 3 = 40-60m, zone 4 = 60-80m, zone 5 = 80-100m). Source: `Cole/Testing_Environments/Testing_Environment_1.py`.

---

## 2. Exploratory Data Analysis

### Purpose

This EDA examines the structure and quality of the per-episode metrics collected during the 5-environment evaluation to confirm data integrity before drawing conclusions about policy performance.

### Data Distributions

**Completion (boolean):** Expect highly unequal groups. On easy terrain (friction zone 1-2), both policies should complete most episodes. On harder terrain (stairs zone 4-5, boulder zone 5), completions should be rare. The obstacle environment should show lower completion rates overall due to the dense obstacle field and longer traversal distance (75m+). This asymmetry is expected and central to the research question.

**Progress (float, 0-50m):** Expected to be right-skewed for hard environments (many episodes cluster at early zones) and left-skewed or uniform for easy environments. Histograms per (env, policy) should show this pattern. Obstacle environment progress may show a different distribution shape due to the non-linear navigation path.

**Zone Reached (int, 1-5):** Ordinal; report counts per zone. Unequal groups expected — most runs should reach zone 2-3, fewer reach zone 4-5. Obstacle zones are defined by radial distance from start rather than linear x-position.

**Stability Score (float, composite):** Expected to be right-skewed (most episodes are reasonably stable, with a tail of high-instability episodes before falls). Obstacle collisions may produce stability spikes distinct from terrain-induced instability.

**Mean Velocity (float):** Should cluster around 0.8-1.0 m/s for successful traversals (target vx = 1.0 m/s) and drop toward 0 for stalled/fallen episodes. Obstacle navigation may produce lower mean velocities due to path deviations around obstacles.

**Episode Length (float, 0-120s):** Bimodal — episodes that complete early (< 60s) and timeouts at 120s. Falls truncate episodes.

**Fall Location / Zone:** Categorical counts — report fall frequency per zone per environment. Expect falls to concentrate in higher-numbered zones.

### Missing Data

| Variable | Can Be Missing? | Reason | Handling |
|---|---|---|---|
| `time_to_complete` | Yes (`null`) | Only populated when robot reaches completion threshold | Expected missing — no imputation needed |
| `fall_location` | Yes (`null`) | Only populated when a fall is detected | Expected missing — no imputation needed |
| `fall_zone` | Yes (`null`) | Only populated when a fall is detected | Expected missing — no imputation needed |
| `total_energy` | Always present | 0.0 when torque data unavailable | Flag 0.0 values — may indicate torque passthrough unavailable |
| All other fields | Never missing | Always recorded by MetricsCollector | N/A |

The three nullable fields (`time_to_complete`, `fall_location`, `fall_zone`) are conditionally missing by design. There is no systematic cause for unexpected missing data — every control step records all required sensor readings.

### Outliers

**Progress:** Outliers may appear if the robot gets stuck in geometry (progress = 0) or glitches through terrain (progress >> expected). Check for episodes with progress < 1.0m (stuck at spawn) or progress > 50.0m (impossible — arena is 50m for linear courses). Obstacle environment episodes may legitimately show low progress if the robot is blocked by dense obstacle clusters early on.

**Stability Score:** Extreme values (> 2.0) likely correspond to tumbling episodes just before fall detection. These are legitimate data reflecting policy failure, not measurement errors. Obstacle collisions may produce transient stability spikes.

**Episode Length:** Episodes significantly shorter than ~20s with no fall suggest potential simulation errors (should be investigated). Episodes at exactly 120.0s are timeouts (expected).

**Mean Velocity:** Negative values are possible (robot walks backward) and legitimate. Values outside [-2.0, 3.0] would indicate a measurement error (command clamps enforce this range).

### Descriptive Statistics

Per (environment, policy) group (10 combinations: 5 environments x 2 policies), report:
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

1. **Completion Rate Bar Chart:** Grouped bars (flat vs rough) per environment (5 environment groups)
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

Examples: `friction_flat_episodes.jsonl`, `stairs_rough_episodes.jsonl`, `obstacle_rough_episodes.jsonl`

### JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Capstone Episode Record",
  "description": "Per-episode evaluation metrics for the 5-environment Spot locomotion comparative test.",
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
      "pattern": "^(friction|grass|boulder|stairs|obstacle)_(flat|rough)_ep[0-9]{4}$",
      "description": "Unique identifier: {env}_{policy}_ep{NNNN}"
    },
    "policy": {
      "type": "string",
      "enum": ["flat", "rough"],
      "description": "Locomotion policy type"
    },
    "environment": {
      "type": "string",
      "enum": ["friction", "grass", "boulder", "stairs", "obstacle"],
      "description": "Test environment name"
    },
    "completion": {
      "type": "boolean",
      "description": "Whether robot reached completion threshold"
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
      "description": "Sim seconds to reach completion threshold, or null if incomplete"
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

### 4.1 Overview

This project evaluates two locomotion policies for the Boston Dynamics Spot quadruped robot in simulation:

1. **Flat Terrain Policy (Baseline):** NVIDIA's pre-trained locomotion controller bundled with Isaac Sim 5.1.0. It uses 48-dimensional proprioceptive observations (body velocity, gravity, joint positions/velocities, previous actions) to produce 12 joint position offsets. This policy was designed for flat, obstacle-free environments and serves as the control group.

2. **Rough Terrain Policy (Experimental):** A custom-trained PPO reinforcement learning policy that extends the flat baseline with 187-dimensional height scan observations (total: 235 dims). Trained across a 4-phase curriculum on an NVIDIA H100 NVL GPU using Isaac Lab's RSL-RL framework, this policy learns to perceive and navigate complex terrain through a learned mapping from local height measurements to gait adjustments. The 4-phase curriculum progressively introduces terrain difficulty — from flat ground to 11 custom terrain types at full difficulty — producing a dramatically more capable policy than a single-phase training approach.

The research question is whether the additional height-scan perception and curriculum-trained rough-terrain policy significantly improves traversal performance across progressively difficult environments compared to the proprioception-only baseline.

### 4.2 Specifications

**Flat Terrain Policy:**
- Source: `omni.isaac.quadruped.robots.SpotFlatTerrainPolicy` (NVIDIA Isaac Sim 5.1.0)
- Observations: 48 dimensions (proprioception only)
- Actions: 12 joint position offsets
- Architecture: Pre-compiled NVIDIA module (not modifiable)

**Rough Terrain Policy:**
- Source: `multi_robot_training/train_ppo.py` (unified training script)
- Architecture: MLP with 4 layers (see Section 4.4 for diagram)
  - Actor: `235 -> [1024, ELU] -> [512, ELU] -> [256, ELU] -> 12`
  - Critic: `235 -> [1024, ELU] -> [512, ELU] -> [256, ELU] -> 1`
- Total parameters: ~1.8M (actor ~900K + critic ~900K)
- Observations: 235 dimensions
  - `[0:3]` Base linear velocity (body frame)
  - `[3:6]` Base angular velocity (body frame)
  - `[6:9]` Projected gravity vector
  - `[9:12]` Velocity commands (vx, vy, omega_z)
  - `[12:24]` Joint positions relative to default
  - `[24:36]` Joint velocities
  - `[36:48]` Previous action
  - `[48:235]` Height scan (17x11 grid, 0.1m resolution, 1.6m x 1.0m)
- Actions: 12 joint position offsets, scaled by 0.3
- init_noise_std: 0.5
- Decimation: 10 (policy runs at 50 Hz, physics at 500 Hz)
- PD Gains: Kp = 60.0, Kd = 1.5 (all joints)
- Effort Limits: Hips = 45 Nm, Knees = 110 Nm (angle-dependent via RemotizedPD lookup table)
- Solver: 4 position iterations, 0 velocity iterations
- Production checkpoint: Final Phase B model (training ongoing as of March 2026)

### 4.3 Model Run

**Platform:** NVIDIA H100 NVL (95,830 MiB VRAM), CUDA 13.0, Driver 580.126.16

**Software Stack:**
- NVIDIA Isaac Sim 5.1.0
- Isaac Lab 0.54.2
- PyTorch 2.7.0+cu128
- Python 3.11.14 (Miniconda `env_isaaclab`)
- Ubuntu Linux (ai2ct2 server)

**Execution:**
- Headless mode (no GUI rendering) via `./isaaclab.sh -p train_ppo.py --headless`
- Per-episode eval wall time: ~41 seconds (120s simulation time at faster-than-realtime)
- Full production run: 100 episodes x 10 combinations = ~11 hours
- GPU memory usage: ~4.5 GB per evaluation run (well within 96 GB capacity)

**Cost:** University research GPU allocation (no direct monetary cost). H100 NVL compute time breakdown:
- Training (all phases and trials): ~120+ hours across 11+ trial runs
- Production evaluation: ~11 hours (100 episodes x 10 combinations)
- Debug validation: ~42 minutes (5 episodes x 10 combinations)
- Total H100 usage: ~132+ hours

### 4.4 Training

The rough terrain policy was trained using a 4-phase curriculum on the NVIDIA H100 NVL. Training is defined in `multi_robot_training/configs/` and executed by `multi_robot_training/train_ppo.py`.

#### Algorithm

- **Algorithm:** Proximal Policy Optimization (PPO) [3]
- **Framework:** RSL-RL (Isaac Lab's RL training library) [4]
- **LR Schedule:** Cosine annealing with linear warmup (custom `shared/lr_schedule.py`)
- **Value Loss Watchdog:** Halves learning rate when `value_loss > 100` for 50 consecutive iterations to break oscillation cascades (Bug #25)

#### 4-Phase Curriculum

Training progresses through four phases of increasing terrain difficulty. Each phase resumes from the previous phase's best checkpoint:

| Phase | Terrain | Iters | lr_max | Envs | Result |
|---|---|---|---|---|---|
| A (flat) | 100% flat | 500 | 3e-4 | 20,480 | `model_498.pt` — 99.3% survival, noise 0.38 |
| A.5 (transition) | 50% flat + gentle rough | 1,000 | 3e-4 | 20,480 | `model_998.pt` — 92.9% survival, gait 8.58 |
| B-easy (robust_easy) | 11 terrains, 3 difficulty rows | 5,002 | 3e-5 | 40,960 | `model_5000.pt` — reward 216, terrain 0.83 |
| B (robust) | 11 terrains, 10 difficulty rows | ongoing | 3e-5 | 5,000 | Trial 11j — variance-based height conditioning (with NaN guard + error clamping) + terrain-scaled velocity + explicit noise clamp [0.3, 0.5] + clamped penalty terms (Bug #29) |

- **Phase A** builds a stable flat-ground gait with high survival rate as the foundation.
- **Phase A.5** introduces gentle rough terrain while retaining 50% flat ground to prevent catastrophic forgetting of the base gait.
- **Phase B-easy** exposes the policy to all 11 terrain types at reduced difficulty (3 rows) with a conservative learning rate (3e-5) discovered through multiple crash trials.
- **Phase B** is the final phase with full difficulty (10 rows), producing the production rough terrain policy.

#### PPO Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Clip Range | 0.2 | Standard PPO clipping |
| Entropy Coefficient | 0.01 | Encourages exploration |
| Value Loss Coefficient | 1.0 | |
| Discount (gamma) | 0.99 | |
| GAE Lambda | 0.95 | Generalized Advantage Estimation |
| Learning Epochs | 4-8 | 8 default; reduced to 4 for Phase B stability |
| Mini-batches | 64 | |
| Max Gradient Norm | 1.0 | Gradient clipping |
| init_noise_std | 0.5 | Reduced from 0.8 to prevent flip-over spiral |
| Desired KL | 0.01 | Target KL divergence |
| Steps per Env | 32 | Per-iteration rollout length |
| Observation Normalization | Disabled | No running-mean normalization |
| Save Interval | 100 | ~65M steps between checkpoints |

#### Neural Network Architecture

```
Actor:  235 -> [1024, ELU] -> [512, ELU] -> [256, ELU] -> 12
Critic: 235 -> [1024, ELU] -> [512, ELU] -> [256, ELU] -> 1
```

- Total parameters: ~1.8M (actor ~900K + critic ~900K)
- No observation or action normalization layers
- Wider architecture (1024->512->256) compared to standard ANYmal-C configs (512->256->128) to handle the terrain diversity of the 4-phase curriculum

#### Observation Space (235 Dimensions)

| Component | Dims | Noise | Description |
|---|---|---|---|
| base_lin_vel | 3 | +/-0.15 | Body-frame linear velocity |
| base_ang_vel | 3 | +/-0.15 | Body-frame angular velocity |
| projected_gravity | 3 | +/-0.05 | Gravity projected into body frame |
| velocity_commands | 3 | None | Target [vx, vy, omega_z] |
| joint_pos (relative) | 12 | +/-0.05 | Joint angles relative to default stance |
| joint_vel (relative) | 12 | +/-0.5 | Joint velocities |
| last_action | 12 | None | Previous policy output |
| height_scan | 187 | +/-0.15, clipped [-1, 1] | 17x11 grid, 0.1m resolution, 1.6m x 1.0m |
| **Total** | **235** | | |

#### Reward Function

The reward function uses 19 terms tuned for curriculum-based training with terrain-adaptive weights.

**Task rewards (positive incentives):**

| Term | Weight | Function |
|---|---|---|
| gait | +1.0 | Trot enforcer: diagonal pairs synchronized (FL-HR, FR-HL), std=0.35, max_err=0.6 (loosened from 10.0/0.1/0.2 to allow terrain-adaptive gaits) |
| air_time | +5.0 | Rewards appropriate swing/stance phase timing (mode_time=0.2s) |
| base_linear_velocity | +5.0 | Exponential kernel on XY velocity error (std=0.5) |
| base_angular_velocity | +5.0 | Exponential kernel on yaw rate error (std=2.0) |
| foot_clearance | +3.0 | Rewards foot height during swing phase (target 0.125m) |
| velocity_modulation | +2.0 | Velocity-dependent reward scaling (std=0.5) |

**Penalty rewards (negative incentives):**

| Term | Weight | Purpose |
|---|---|---|
| dof_pos_limits | -5.0 | Hard penalty for hitting joint limits |
| base_orientation | -3.0 | Penalizes roll/pitch deviation from upright |
| body_scraping | -2.0 | Penalizes body contact while moving (velocity > 0.3 m/s) |
| undesired_contacts | -1.5 | Penalizes body-ground contact (threshold 1.0 N) |
| terrain_relative_height | -2.0 | Variance-based height target: 0.42m on flat ground (scan variance ≤ 0.001) → 0.35m on rough ground (scan variance ≥ 0.02), interpolated by height scan variance. Uses ray-cast ground Z for relative height. Direct per-step signal that works at eval time. NaN-guarded with `nan_to_num` (missed rays return `inf`), height error clamped to [0, 1] to prevent gradient explosion |
| air_time_variance | -1.0 | Penalizes asymmetric gait patterns |
| base_motion | -0.5 | Penalizes vertical bouncing and lateral sway |
| joint_pos | -0.2 | Penalizes deviation from default stance (stand_still_scale=5.0) |
| foot_slip | -0.5 | Penalizes feet sliding during ground contact |
| action_smoothness | -0.1 | Penalizes jerky action changes between steps |
| stumble | 0.0 (disabled) | Penalizes knee contact below 0.15m at force > 5.0 N — DISABLED because world-frame Z comparison breaks on elevated terrain (Bug #28b) |
| joint_vel | -1e-2 | Penalizes high joint velocities |
| contact_force_smoothness | -0.01 | Penalizes abrupt contact force changes |
| vegetation_drag | -1e-3 | Drag penalty for vegetation terrain (max drag 20.0) |
| joint_torques | -5e-4 | Penalizes torque usage for energy efficiency |
| joint_acc | -1e-4 | Penalizes joint acceleration for smoothness |

#### Terrain (11 Types, 3 Categories)

Training uses a custom `ROBUST_TERRAINS_CFG` (`shared/terrain_cfg.py`) with 11 procedurally generated terrain types across a 10-row x 40-column grid (8m x 8m patches). Difficulty increases with row number; the curriculum automatically promotes robots with longer survival to harder rows.

**Category A — Geometric Obstacles (40%):**

| Terrain | Proportion | Parameters |
|---|---|---|
| pyramid_stairs_up | 10% | step_height [0.05, 0.25]m, step_width 0.3m, platform 3.0m |
| pyramid_stairs_down | 10% | step_height [0.05, 0.25]m, step_width 0.3m, platform 3.0m |
| boxes | 10% | grid_width 0.45m, grid_height [0.05, 0.25]m, platform 2.0m |
| stepping_stones | 10% | stone_height_max 0.15m, stone_width [0.25, 0.5]m, distance [0.1, 0.4]m |

**Category B — Surface Variation (40%):**

| Terrain | Proportion | Parameters |
|---|---|---|
| random_rough | 10% | noise [0.02, 0.15]m, noise_step 0.02 |
| hf_pyramid_slope_up | 7.5% | slope [0.0, 0.5], platform 2.0m |
| hf_pyramid_slope_down | 7.5% | slope [0.0, 0.5], platform 2.0m |
| wave_terrain | 5% | amplitude [0.05, 0.2]m, num_waves 3 |
| friction_plane | 5% | Flat plane (low-friction challenge, no geometry) |
| vegetation_plane | 5% | Flat plane (drag challenge, no geometry) |

**Category C — Compound Challenges (20%):**

| Terrain | Proportion | Parameters |
|---|---|---|
| hf_stairs_up | 10% | step_height [0.05, 0.20]m, step_width 0.3m |
| discrete_obstacles | 5% | obstacle_width [0.25, 0.75]m, height [0.05, 0.30]m, 40 per patch |
| repeated_boxes | 5% | 20-40 objects, height [0.05, 0.20]m, size [0.3, 0.5]m |

#### Difficulty Rows — What Each Curriculum Level Means

All terrain parameters scale linearly from their minimum (row 0) to maximum (row 9). The curriculum automatically promotes robots that survive longer to harder rows and demotes robots that fail.

| Row | Stairs | Boxes | Rough Noise | Slopes | Stepping Stones | Real-World Equivalent |
|-----|--------|-------|-------------|--------|-----------------|----------------------|
| **0** | 5cm ramp | 5cm pebbles | ±2cm | ~0° flat | 50cm wide, 10cm gaps | Smooth sidewalk |
| **1** | 7cm curb | 7cm cobblestone | ±3.5cm | ~5° | 47cm, 13cm gaps | Gravel path |
| **2** | 9cm step | 9cm rubble | ±5cm | ~11° | 44cm, 17cm gaps | Rough hiking trail |
| **3** | 12cm step | 12cm chunks | ±6cm | ~17° | 42cm, 20cm gaps | Rocky trail |
| **4** | **14cm** half-stair | 14cm debris | ±8cm | **~22°** | 39cm, 23cm gaps | Construction site |
| **5** | **16cm** standard stair | 16cm rubble | ±9cm | **~28°** | 36cm, 27cm gaps | Real stairs, steep hills |
| **6** | **18cm** tall stair | 18cm pile | ±11cm | **~33°** | 33cm, 30cm gaps | Disaster site |
| **7** | **21cm** near leg limit | 21cm climbing | ±12cm | **~39°** | 31cm, 33cm gaps | Extreme terrain |
| **8** | **23cm** full extension | 23cm chest-high | ±13cm | **~44°** | 28cm, 37cm gaps | Near Spot's physical limits |
| **9** | **25cm** max | 25cm (>half Spot height) | **±15cm** | **~50°** | **25cm, 40cm gaps** | Beyond real-world stairs |

**Practical interpretation:**
- **Rows 0-3:** Walking on uneven ground (most indoor/outdoor surfaces)
- **Rows 4-5:** Real obstacles — standard stairs, construction debris, steep hills
- **Rows 6-7:** Disaster-site terrain — rubble piles, extreme stairs, mountain scrambles
- **Rows 8-9:** At or beyond Spot's physical hardware limits — theoretical ceiling

**Training progress:** The rough terrain policy peaked at terrain level ~5.0 (Trial 11d, best ever), meaning it can handle ~16cm stairs (real indoor stairs), ±10cm rough ground, and ~28° slopes. Trial 11j (ongoing) uses **height scan variance conditioning** with **clamped penalty terms** (Bug #29). Trial 11f introduced variance conditioning but failed due to NaN from missed ray hits returning `inf`; Trial 11g failed from resuming a corrupted checkpoint; Trial 11h failed at iter 105 because `--max_noise_std 0.5` was omitted from the launch command (defaults to 1.0), causing noise to climb and `action_smoothness` to explode to -1.3 trillion (Bug #28d). Trial 11i fixed all prior issues (`nan_to_num`, error clamp, stumble disabled, explicit noise clamp) but FAILED at iter 82 — `action_smoothness` exploded to -921,693 and value loss hit 6.3e18, even with noise clamped at 0.5. Root cause: Isaac Lab's built-in penalty functions return unbounded L2 norms that amplify value function instability (Bug #29). Trial 11j adds clamped wrapper functions for all unbounded penalties and resumes from clean model_14000.pt (run dir `2026-03-07_08-14-41`). Early metrics stable: reward 15-29, terrain 1.1, zero NaN.

#### Domain Randomization

| Parameter | Range | When Applied | Purpose |
|---|---|---|---|
| Static friction | [0.3, 1.5] | Startup (256 buckets) | Surface variation |
| Dynamic friction | [0.3, 1.2] | Startup (256 buckets) | Surface variation |
| Base mass offset | +/-5.0 kg | Startup | Payload robustness |
| Joint position reset | +/-0.2 rad around default | Episode reset | Varied start poses |
| Joint velocity reset | +/-2.5 rad/s | Episode reset | Varied dynamics |
| Base pose reset | +/-0.5m XY, +/-pi yaw | Episode reset | Random spawn location |
| Push perturbation | +/-0.5 m/s XY velocity | Every 10-15s | Recovery training |

#### Safety Mechanisms

Five layers of protection guard against training instability, discovered through multiple crash-debug cycles:

1. **NaN Sanitizer (Bug #24):** `_sanitize_std()` in `shared/training_utils.py` explicitly detects NaN, Inf, and negative values in the policy's standard deviation parameter, then replaces and clamps them. PyTorch's `clamp_()` alone does NOT fix NaN (`NaN.clamp_(min=0.3)` is still NaN), so explicit detection is required.

2. **Pre-Forward Safety Clamp:** Monkey-patches `policy.act()` to call `_sanitize_std()` before every single forward pass, catching corruption that occurs mid-update within the learning epochs x mini-batches PPO update cycle. Prevents `RuntimeError: normal expects all elements of std >= 0.0`.

3. **Value Loss Watchdog (Bug #25):** Monitors `value_loss` each iteration. When it exceeds 100 for 50 consecutive iterations, the learning rate is halved to break oscillation cascades that otherwise escalate to NaN.

4. **Noise Clamping [0.3, 0.5] (Bugs #26, #28d):** Post-update safety clamp on the policy's exploration noise standard deviation. The upper bound must be explicitly passed via `--max_noise_std` (train_ppo.py defaults to 1.0). Trial 11h crashed at iter 105 because this flag was omitted — noise climbed to 1.0, causing `action_smoothness` to explode to -1.3 trillion and triggering NaN. ALWAYS pass `--max_noise_std 0.5 --min_noise_std 0.3` explicitly.

5. **Clamped Penalty Terms (Bug #29):** Isaac Lab's built-in penalty functions (`action_smoothness_l2`, `joint_acc_l2`, `joint_torques_l2`, `joint_vel_l2`) return unbounded L2 norms. When the value function goes unstable, a positive feedback loop forms: policy spike -> norm explosion -> penalty explosion -> value loss explosion -> NaN. Trial 11i failed at iter 82 with `action_smoothness` at -921,693 and `value_loss` at 6.3e18 despite noise clamped at 0.5. Fixed with clamped wrapper functions in `shared/reward_terms.py`: `clamped_action_smoothness_penalty` [0, 10], `clamped_joint_acceleration_penalty` [0, 10000], `clamped_joint_torques_penalty` [0, 1000], `clamped_joint_velocity_penalty` [0, 50], `contact_force_smoothness_penalty` [0, 500]. All wrappers include `torch.where(torch.isfinite(...))` NaN safety.

Additionally, `body_height_tracking` was replaced by `terrain_relative_height_penalty` (Bug #22/#27) which uses the height scanner's center ray hit to compute local ground Z, enabling correct height enforcement on elevated terrain. The height target is driven by **height scan variance**: the variance of the 187 ray Z-hits determines terrain roughness per robot per step. Flat ground (variance ≤ 0.001) → target 0.42m (full stand), rough ground (variance ≥ 0.02) → target 0.35m (moderate crouch), with linear interpolation between. This direct per-step signal replaced the earlier curriculum-level-based approach (Trial 11e) which was too indirect and produced oscillating penalties. Weight -2.0. The variance computation is NaN-guarded with `torch.nan_to_num()` (missed rays return `inf`) and the height error is clamped to [0, 1] before squaring to prevent gradient explosion from fallen robots (Bug #28c). The stumble penalty was disabled (Bug #28b) because it uses world-frame Z, which incorrectly penalizes all foot contacts on elevated terrain.

**Terrain-Scaled Velocity Commands** (`shared/terrain_velocity_command.py`): A custom `TerrainScaledVelocityCommand` class queries each robot's terrain curriculum level and interpolates the velocity command range — sprint commands (0.5-3.0 m/s) on easy terrain, careful walk commands (0.0-1.0 m/s) on hard terrain. This teaches the policy to map height-scan patterns to appropriate speeds proactively.

#### Physics and Control Configuration

| Parameter | Value |
|---|---|
| Physics timestep | 1/500 s (500 Hz) |
| Control frequency | 1/50 s (50 Hz, decimation = 10) |
| Action scale | 0.3 |
| Observation clipping | [-100, 100] |
| Action clipping | [-100, 100] |
| PD gains | Kp = 60.0, Kd = 1.5 |
| Hip effort limit | 45 Nm (DelayedPD actuator) |
| Knee effort limit | angle-dependent, max ~110 Nm (RemotizedPD lookup table) |
| Joint velocity limit | 12.0 rad/s |
| Solver iterations | 4 position, 0 velocity |
| Self-collision | Enabled |
| Max depenetration velocity | 1.0 m/s |
| Episode length | 30.0 s (training) |

#### Dataset and Reproduction

There is no fixed dataset — the training environment procedurally generates randomized terrain at each episode reset. Domain randomization (friction, mass, pushes) provides additional diversity. To reproduce:

1. Clone the repository and install Isaac Lab + RSL-RL
2. Training configs: `multi_robot_training/configs/`
3. Phase A: `isaaclab.sh -p train_ppo.py --headless --terrain flat --num_envs 20480 --max_iterations 500 --lr_max 3e-4`
4. Phase A.5: `isaaclab.sh -p train_ppo.py --headless --terrain transition --num_envs 20480 --max_iterations 1000 --lr_max 3e-4 --resume model_498.pt`
5. Phase B-easy: `isaaclab.sh -p train_ppo.py --headless --terrain robust_easy --num_envs 40960 --max_iterations 5000 --lr_max 3e-5 --resume model_998.pt`
6. Phase B: `isaaclab.sh -p train_ppo.py --headless --terrain robust --num_envs 5000 --lr_max 3e-5 --learning_epochs 4 --save_interval 100 --resume model_5000.pt`
7. Monitor via TensorBoard: `tensorboard --logdir logs/rsl_rl/spot_robust_ppo/`

### 4.5 Evaluation

**Procedure:** Each (environment, policy) combination runs N episodes. For linear courses (friction, grass, boulder, stairs), the robot spawns at (0, 15, 0.6) facing +X and follows 6 waypoints along the arena centerline at vx = 1.0 m/s. For the obstacle environment (`Cole/Testing_Environments/Testing_Environment_1.py`), the robot spawns at (-45, 0, 0.7) and navigates to a randomized goal 75m+ away through 360 obstacles using a go-to-goal controller. Episodes terminate on: completion, fall (body < 0.15m above ground), or timeout (120s).

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
- Debug validation: 5 episodes x 10 combinations (50 total) — confirms pipeline integrity
- Production: 100 episodes x 10 combinations (1,000 total) — statistical significance

**Preliminary Debug Results (5 episodes each, all 10 combinations):**
All combinations completed without crashes. Example: `friction_flat` achieved mean progress of 12.0m (zone 2), mean stability 0.096, no falls detected, mean velocity 0.99 m/s, episode length 119.98s (timeout). Full production results pending.

### References

[1] NVIDIA, "Isaac Sim Documentation," 2025. [Online]. Available: https://docs.omniverse.nvidia.com/isaacsim/latest/

[2] NVIDIA, "Isaac Lab Documentation," 2025. [Online]. Available: https://isaac-sim.github.io/IsaacLab/

[3] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," arXiv preprint arXiv:1707.06347, 2017.

[4] N. Rudin, D. Hoeller, P. Reist, and M. Hutter, "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning," Proc. Conf. Robot Learn. (CoRL), 2022.

[5] Boston Dynamics, "Spot Robot," 2024. [Online]. Available: https://bostondynamics.com/products/spot/

---

## 5. Reflection

Building this deliverable required synthesizing implementation details from across the entire codebase into standardized documentation formats. The data dictionary exercise clarified which variables are required vs optional and exposed that `total_energy` reports 0.0 when torque data is unavailable — a limitation worth noting in the final analysis. The JSON schema formalization revealed that our JSONL output already conforms to a clean, validatable structure with no ambiguous types.

The model documentation underwent a complete rewrite for Version 2.0 to reflect the transition from a single-phase 48-hour training approach to a 4-phase curriculum. The original model (`model_29999.pt`, 30K iterations, 512->256->128 network, 14 rewards) has been replaced by a curriculum-trained policy with a larger 1024->512->256 network, 19 reward terms, 11 custom terrain types, and multiple safety mechanisms. This rewrite was necessary because the curriculum approach fundamentally changed the training methodology — rather than one long run, training now consists of four carefully sequenced phases (flat -> transition -> robust_easy -> robust) where each phase builds on the previous phase's learned behaviors.

The curriculum approach proved essential for training stability. Single-phase training on hard terrain consistently diverged or stalled, as documented in our Bug Museum (Bugs #22-26). Each bug discovery — from NaN propagation through standard deviation parameters to value loss oscillation cascades — required a targeted safety mechanism. These hard-won lessons are now baked into the training infrastructure as automatic safeguards, making the training pipeline significantly more robust than the original 48-hour approach.

Adding the 5th evaluation environment (obstacle navigation) was a team collaboration effort. Cole's Testing_Environment_1.py provides a qualitatively different challenge from the four linear-course environments: the robot must navigate a dense 100m x 100m field of 360 randomly placed obstacles to reach a randomized goal, testing spatial reasoning and collision avoidance rather than terrain traversal. This addition strengthens the evaluation by testing whether rough-terrain training transfers to obstacle avoidance, a capability not explicitly trained but potentially emergent from height-scan perception.

---

## 6. Changes to Previous Deliverables

**Version 2.0 (March 5, 2026):**

- **Section 4 (Model Documentation): Full rewrite.** Replaced the single-phase 48h training documentation (`model_29999.pt`, 14 rewards, 6 terrain types, 512->256->128 network, 8192 envs) with the current 4-phase curriculum training documentation (19 rewards, 11 custom terrains, 1024->512->256 network, up to 40,960 envs). Added subsections for the curriculum phases, safety mechanisms, and updated hyperparameters.

- **Sections 1-3: Added 5th evaluation environment (obstacle navigation).** Cole's Testing_Environment_1.py — a 100m x 100m obstacle arena with 360 randomly placed objects — was added as the `"obstacle"` environment across the data dictionary, EDA, and JSON schema. Updated all combination counts from 8 (4 envs x 2 policies) to 10 (5 envs x 2 policies).

- **Section 5 (Reflection): Updated** to discuss curriculum learning vs single-phase training, bugs discovered, and team collaboration on the obstacle environment.

**Version 2.1 (March 6, 2026):**

- **Section 4.4 (Reward Function): Updated weights** to reflect Trial 11d tuning — gait loosened (10.0→1.0, std 0.1→0.35), penalties reduced (action_smoothness -1.0→-0.1, joint_pos -0.7→-0.2, base_motion -2.0→-0.5, stumble -0.1→-0.02), foot_clearance increased (2.0→3.0), action_scale increased (0.2→0.3). Replaced `body_height_tracking` with `terrain_relative_height` using terrain-scaled targets (0.42m easy → 0.25m hard).

- **Section 4.4 (Safety Mechanisms): Added** terrain-scaled velocity commands (`TerrainScaledVelocityCommand`) and terrain-scaled height penalty documentation. These two features teach the policy to map terrain difficulty to appropriate speed and posture.

- **Section 4.4 (Training Progress): Updated** from Trial 11 (terrain 3.77) to Trial 11d (terrain 4.5+) with terrain-scaled velocity and height.

**Version 2.2 (March 6, 2026):**

- **Section 4.4 (Reward Function): Updated** `terrain_relative_height` weight from -1.0 to -2.0, height_hard from 0.25m to 0.35m (Trial 11e fix for persistent crawling behavior).

- **Section 4.4 (Training Progress): Updated** from Trial 11d (terrain 4.5+) to Trial 11d peak (terrain ~5.0, best ever) and Trial 11e (ongoing, stronger height signal).

**Version 2.3 (March 6, 2026):**

- **Section 4.4 (Reward Function): Updated** `terrain_relative_height` from curriculum-level-based to height-scan-variance-based conditioning (Trial 11f). Height target now driven by the variance of the 187 height scan rays — flat ground (var ≤ 0.001) → 0.42m, rough ground (var ≥ 0.02) → 0.35m. Direct per-step signal that works at eval time.

- **Section 4.4 (Training Progress): Updated** to Trial 11f (variance-based height). Trial 11e replaced after 14000 iters — curriculum-level conditioning too indirect, policy still knee-walking.

**Version 2.4 (March 6, 2026):**

- **Section 4.4 (Training Progress): Updated** to Trial 11h. Trial 11f FAILED (NaN from missed ray `inf`), Trial 11g FAILED (corrupted checkpoint cascade). Trial 11h resumes from clean model_14000.pt with three fixes: `nan_to_num` for ray hits, height error clamped [0, 1], stumble disabled.

- **Section 4.4 (Reward Function): Updated** stumble penalty from -0.02 to 0.0 (disabled, Bug #28b — world-frame Z breaks on elevated terrain). Updated terrain_relative_height description to note NaN guard and error clamping (Bug #28c).

- **Section 4.4 (Safety Mechanisms): Updated** to document NaN guard on height scan variance, error clamping, and stumble penalty disabling.

**Version 2.5 (March 7, 2026):**

- **Section 4.4 (Training Progress): Updated** to Trial 11i. Trial 11h FAILED at iter 105 — forgot `--max_noise_std 0.5` flag, noise climbed to 1.0, `action_smoothness` exploded to -1.3 trillion (Bug #28d). Trial 11i resumes from clean model_14000.pt with explicit `--max_noise_std 0.5 --min_noise_std 0.3`.

- **Section 4.4 (Safety Mechanisms): Updated** noise clamping from [0.3, 0.7] to [0.3, 0.5] and added Bug #28d documentation — `--max_noise_std` must always be passed explicitly (train_ppo.py defaults to 1.0).

**Version 2.6 (March 7, 2026):**

- **Section 4.4 (Training Progress): Updated** to Trial 11j. Trial 11i FAILED at iter 82 — `action_smoothness` exploded to -921,693, value loss 6.3e18, even with noise clamped at 0.5. Root cause: unbounded L2 penalty norms (Bug #29). Trial 11j resumes from clean model_14000.pt with clamped penalty wrappers (run dir `2026-03-07_08-14-41`). Early metrics stable: reward 15-29, terrain 1.1, zero NaN.

- **Section 4.4 (Safety Mechanisms): Added** Bug #29 clamped penalty terms as 5th safety layer. Isaac Lab's built-in penalty functions return unbounded L2 norms that amplify value function instability into a positive feedback loop. Fixed with clamped wrappers: `clamped_action_smoothness_penalty` [0, 10], `clamped_joint_acceleration_penalty` [0, 10000], `clamped_joint_torques_penalty` [0, 1000], `clamped_joint_velocity_penalty` [0, 50], `contact_force_smoothness_penalty` [0, 500]. All include `torch.where(torch.isfinite(...))` NaN safety.
