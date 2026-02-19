# Training Environment 1 — Flat Terrain Friction Study
## Reinforcement Learning Plan for Spot in IsaacSim

**Goal:** Start from IsaacSim's stock `SpotFlatTerrainPolicy` and improve locomotion quality across varying ground friction conditions. The robot walks straight forward across a 100m track — this track is the mechanism to exercise locomotion, not a navigation challenge. The policy learns to walk stably, efficiently, and adaptably depending on the surface underfoot.

---

## 1. Environment Overview

| Parameter | Value |
|---|---|
| Arena size | 100m (X) × 30m (Y) |
| Arena centered at | (0, 0, 0) |
| Robot start | X = −48m |
| Walk direction | +X (straight forward) |
| Ground type | Flat, infinite plane |
| Ground friction | Configurable (see §3) |
| Number of robots | 5 (parallel training) |
| Episode timeout | 120 seconds |
| Physics rate | 500 Hz |
| Control rate | 50 Hz |

Spot spawns at the west end and walks straight east. The direction is fixed — the robot is not asked to find a goal. The only environmental variable is the friction coefficient of the ground plane. A robot that reaches X = +48m is simply reset; reaching the end is incidental, not the objective.

---

## 2. Architecture: Locomotion Command Policy

```
┌─────────────────────────────────────────────────────────────┐
│  Locomotion Command Policy (to be trained/improved)         │
│  Input: body state + friction  →  Output: [vx, vy, ω]       │
│  Learns to issue commands that produce high-quality gait    │
└─────────────────────────┬───────────────────────────────────┘
                          │  velocity command [vx, vy, ω]
┌─────────────────────────▼───────────────────────────────────┐
│  SpotFlatTerrainPolicy (stock IsaacSim)                     │
│  Input: velocity command  →  Output: joint torques          │
│  Pre-trained gait primitive — treated as a black box        │
└─────────────────────────────────────────────────────────────┘
```

The `SpotFlatTerrainPolicy` handles joint-level gait generation. The Locomotion Command Policy learns what velocity commands to issue — and at what magnitude — to produce the best locomotion outcome on a given surface. On high friction, it can command full speed confidently. On low friction, it should command lower speeds, smoother transitions, and smaller lateral/yaw components to avoid slipping and falling.

**Baseline:** Run the stock policy with a fixed forward command (vx = 1.5, vy = 0, ω = 0) at each friction level and record locomotion metrics. This is the performance floor.

---

## 3. Friction Configuration

The friction variable is the sole environmental parameter. It is set at the start of each episode (not mid-episode) via the USD physics material API.

### Discrete Test Surfaces

Values are for rubber foot pad (Spot's feet) on each surface. μₛ = static friction, μₖ = dynamic (kinetic) friction.

| Label | μₛ (static) | μₖ (dynamic) | Represents |
|---|---|---|---|
| `asphalt_dry` | 0.75 | 0.65 | Dry pavement — comfortable baseline |
| `asphalt_wet` | 0.50 | 0.40 | Rain on pavement |
| `grass_dry` | 0.40 | 0.35 | Dry grass field |
| `grass_wet` | 0.25 | 0.20 | Wet grass |
| `mud` | 0.20 | 0.15 | Saturated soil (approximation — see note below) |
| `snow` | 0.15 | 0.10 | Packed snow |
| `ice` | 0.07 | 0.05 | Ice — worst case |

> **Mud note:** A single friction coefficient cannot capture mud's viscous drag or foot-suction effects. These values are an approximation. A fuller mud model requires PhysX deformable terrain, which is out of scope for Environment 1.

```python
# Controlled via PhysicsMaterialAPI on the ground plane prim
FRICTION_CONFIG = {
    "asphalt_dry": {"static": 0.75, "dynamic": 0.65},
    "asphalt_wet": {"static": 0.50, "dynamic": 0.40},
    "grass_dry":   {"static": 0.40, "dynamic": 0.35},
    "grass_wet":   {"static": 0.25, "dynamic": 0.20},
    "mud":         {"static": 0.20, "dynamic": 0.15},
    "snow":        {"static": 0.15, "dynamic": 0.10},
    "ice":         {"static": 0.07, "dynamic": 0.05},
    "random":      {"static": (0.05, 0.80), "dynamic": (0.05, 0.70)},  # Uniform sample
}
```

**Changing friction at runtime** requires accessing the USD prim directly:

```python
from pxr import UsdPhysics

def set_ground_friction(stage, static_friction, dynamic_friction):
    """Update ground plane friction material in-place."""
    material_path = "/World/defaultGroundPlane/PhysicsMaterial"
    material = UsdPhysics.MaterialAPI.Get(stage, material_path)
    material.GetStaticFrictionAttr().Set(static_friction)
    material.GetDynamicFrictionAttr().Set(dynamic_friction)
```

This can be called at episode reset without restarting the simulation. Verify the exact prim path on first run — it may vary by IsaacSim version (see §12).

---

## 4. Observation Space

Each robot receives an 11-dimensional observation vector at every control step (50 Hz). All observations describe the robot's body state — no goal direction or distance features.

| Index | Feature | Description |
|---|---|---|
| 0 | `vel_x` | Actual body linear velocity in X (forward), m/s |
| 1 | `vel_y` | Actual body linear velocity in Y (lateral), m/s |
| 2 | `vel_z` | Actual body linear velocity in Z (vertical), m/s |
| 3 | `ang_vel_x` | Body angular velocity — roll rate, rad/s |
| 4 | `ang_vel_y` | Body angular velocity — pitch rate, rad/s |
| 5 | `ang_vel_z` | Body angular velocity — yaw rate, rad/s |
| 6 | `roll` | Body roll angle, rad |
| 7 | `pitch` | Body pitch angle, rad |
| 8 | `height` | Body Z position normalized by nominal height (0.7m) |
| 9 | `friction` | Current static friction coefficient (known to agent) |
| 10 | `cmd_vx` | The forward velocity the policy commanded last step |

**Notes:**
- No goal position, no heading error, no distance to goal — this is locomotion, not navigation.
- `cmd_vx` is included so the policy can observe its own prior action and learn smooth command transitions.
- `friction` is provided explicitly so the policy can adapt its behavior to the known surface condition. In future environments this could be estimated from slip ratio or IMU data instead.
- All velocities are in the robot's world frame from `spot.robot.get_linear_velocity()` and `spot.robot.get_angular_velocity()`.

---

## 5. Action Space

The policy outputs a 3-dimensional continuous action vector:

| Index | Signal | Range | Notes |
|---|---|---|---|
| 0 | `vx` — forward velocity | [0.0, 2.0] m/s | Policy must keep this positive to keep walking |
| 1 | `vy` — lateral velocity | [−0.3, 0.3] m/s | Narrow range — only small corrections needed on flat terrain |
| 2 | `ω` — yaw rate | [−0.3, 0.3] rad/s | Narrow range — no turning needed, minor drift correction only |

The lateral and yaw ranges are tighter than the robot's physical limits because on this flat, straight track there is no reason for large lateral or turning commands. Overuse of vy/ω on low-friction surfaces causes instability.

These are passed directly to `spot.forward(step_size, action)`.

---

## 6. Reward Function

Rewards are computed every control step (50 Hz) and summed over the episode. All rewards reflect locomotion quality — not navigation progress.

```
R_total = R_forward + R_stability + R_height + R_smoothness + R_efficiency + R_alive + R_fall
```

| Component | Formula | Scale | Purpose |
|---|---|---|---|
| **Forward motion** | `vel_x` if `vel_x > 0`, else `2 × vel_x` | × 2.0 | Reward walking forward; penalize moving backward |
| **Stability — roll** | `−roll²` | × 3.0 | Penalize body tilt left/right |
| **Stability — pitch** | `−pitch²` | × 3.0 | Penalize body tilt forward/back |
| **Stability — angular rates** | `−(ang_vel_x² + ang_vel_y²)` | × 1.0 | Penalize oscillating/rocking |
| **Height maintenance** | `−(height − 1.0)²` | × 2.0 | Keep body near nominal height (height is normalized, so 1.0 = 0.7m) |
| **Command smoothness** | `−(vx_cmd − vx_cmd_prev)²` | × 1.0 | Penalize sudden velocity command changes (jerk) |
| **Lateral/yaw cost** | `−(vy_cmd² + ω_cmd²)` | × 0.5 | Discourage unnecessary lateral/yaw commands |
| **Alive** | +1 per step if `pos_z > 0.35m` | × 0.2 | Small survival incentive |
| **Fall penalty** | One-time penalty on fall | −100 | Large penalty for losing balance |

**Design rationale:**
- `R_forward` keeps the robot moving without prescribing a destination.
- Stability rewards dominate — a slow, stable walk scores better than a fast, wobbly one.
- `R_smoothness` encourages the policy to change speed gradually, which is critical for low-friction surfaces where sudden acceleration causes slipping.
- There is no goal-reaching bonus. The episode ends on fall or timeout.

---

## 7. Termination Conditions

An episode ends early if:

- **Fall:** Body Z position < 0.25m (body contact with ground)
- **Lateral escape:** Body Y position outside [−13m, +13m] (drifted too far sideways)
- **Timeout:** Episode time > 120 seconds

If the robot reaches X = +48m (the far end), it is simply reset in place to X = −48m and the episode continues — locomotion quality is measured continuously, not as a one-time achievement.

---

## 8. Baseline Measurement (Untrained Model)

Before any training begins, the stock `SpotFlatTerrainPolicy` is run with a fixed, constant forward command through all 7 friction environments. No learning occurs. This establishes the performance floor that the trained policy must beat.

**Command issued every episode:** `[vx = 1.5, vy = 0.0, ω = 0.0]` — a straightforward walk at 1.5 m/s with no lateral or turning correction.

**Protocol:** 20 episodes per surface, 5 parallel robots each episode = 100 independent runs per surface, 700 total runs across all 7 surfaces.

### Measurements Recorded Per Step (50 Hz)

| Measurement | Source | Notes |
|---|---|---|
| Body position (x, y, z) | `get_world_pose()` | Track drift, height loss |
| Linear velocity (vx, vy, vz) | `get_linear_velocity()` | Actual vs. commanded speed |
| Angular velocity (wx, wy, wz) | `get_angular_velocity()` | Roll/pitch rate oscillation |
| Roll angle | Derived from quaternion | Body tilt left/right |
| Pitch angle | Derived from quaternion | Body tilt forward/back |
| Joint positions (12) | `get_joint_positions()` | Gait pattern reference |
| Joint velocities (12) | `get_joint_velocities()` | Limb speed reference |
| Joint torques (12) | `get_applied_action()` | Energy reference |
| Fall event | `pos_z < 0.25m` | Binary, logged with timestamp |

### Metrics Aggregated Per Episode

| Metric | Formula | Purpose |
|---|---|---|
| **Survival time** | Time until fall, or 120s | Primary stability indicator |
| **Fall occurred** | Bool | Binary outcome |
| **Mean actual vx** | `mean(vel_x)` over episode | Speed achieved |
| **Slip ratio** | `mean(vel_x) / 1.5` | How much speed is lost to slipping |
| **Mean \|roll\|** | `mean(abs(roll))` in degrees | Lateral stability |
| **Mean \|pitch\|** | `mean(abs(pitch))` in degrees | Fore-aft stability |
| **Std roll** | `std(roll)` | Oscillation severity |
| **Std pitch** | `std(pitch)` | Oscillation severity |
| **Mean roll rate** | `mean(abs(ang_vel_x))` | Rocking motion proxy |
| **Mean pitch rate** | `mean(abs(ang_vel_y))` | Bobbing motion proxy |
| **Total joint work** | `Σ abs(torque × joint_vel × dt)` | Energy consumed |
| **Cost of Transport** | `total_work / (mass × g × distance)` | Efficiency (dimensionless) |
| **Distance traveled** | `Σ vel_x × dt` | Total forward progress |

### Expected Results Table (to be filled in after running)

| Surface | μₛ | Survival (s) | Fall rate | Slip ratio | Mean \|roll\| (°) | Mean \|pitch\| (°) | CoT |
|---|---|---|---|---|---|---|---|
| `asphalt_dry` | 0.75 | — | — | — | — | — | — |
| `asphalt_wet` | 0.50 | — | — | — | — | — | — |
| `grass_dry` | 0.40 | — | — | — | — | — | — |
| `grass_wet` | 0.25 | — | — | — | — | — | — |
| `mud` | 0.20 | — | — | — | — | — | — |
| `snow` | 0.15 | — | — | — | — | — | — |
| `ice` | 0.07 | — | — | — | — | — | — |

Fill in this table after running the baseline. It becomes the reference for evaluating trained policy improvement in §14.

### Output Files

```
checkpoints/
└── phase0_baseline.npz     ← per-step arrays for all runs (position, velocity, joint data)

logs/
└── baseline_summary.csv    ← one row per episode: surface, robot_id, all aggregated metrics
```

---

## 9. Training Strategy

The baseline (§8) completes once all 700 runs are logged and the results table is filled in. That CSV is the reference for all phase comparisons.

### Phase 1 — High Friction Warm-Up (μ = 0.8)
Train the Locomotion Command Policy with high friction. Easy footing. Policy learns to:
- Maintain stable forward walking at a self-selected speed
- Keep body level and height consistent
- Issue smooth commands

**Advance criteria:** Mean body roll < 5°, mean body pitch < 5°, fall rate < 10%, over 50 episodes.

### Phase 2 — Medium Friction (μ = 0.5)
Transfer the Phase 1 policy. Policy must:
- Reduce commanded speed to maintain the same stability metrics
- Adjust smoothness threshold — transitions that were safe at high friction may cause slipping here

**Advance criteria:** Same stability targets as Phase 1 over 50 episodes.

### Phase 3 — Low Friction (μ = 0.2)
Transfer the Phase 2 policy. Hardest condition. Policy must:
- Command significantly reduced forward velocity (likely 0.5–0.8 m/s)
- Minimize lateral and yaw commands entirely
- Prioritize stability over speed

**Advance criteria:** Mean body roll < 8°, mean body pitch < 8°, fall rate < 25%, over 50 episodes.

### Phase 4 — Randomized Friction
Randomize friction uniformly in [0.1, 0.9] each episode. Policy must generalize — using the `friction` observation to adapt its command magnitude without separate training phases.

**Advance criteria:** Locomotion stability metrics from Phase 1 maintained on average across the full friction range.

---

## 10. Learning Algorithm

**Option A — Cross-Entropy Method (CEM)** *(recommended for first pass)*
- Rank episodes by total locomotion reward, update policy toward top-k
- No gradient computation or replay buffer needed
- Works well when the state space is small (11-dim) and policy is compact
- Simple to debug — reward shaping issues are immediately visible

**Option B — Proximal Policy Optimization (PPO)** *(better long-term)*
- Better sample efficiency
- Handles the continuous reward signal well
- Requires value network; use `stable-baselines3` or a custom implementation
- Verify `stable-baselines3` compatibility with Python 3.11 before installing into the IsaacSim venv

**Suggested network architecture (both options):**
```
Input (11) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(3) → Tanh → scale to action range
```

The network is intentionally small — the locomotion command problem on flat terrain is low-dimensional.

---

## 11. File Structure

```
Experiments/Ryan/Training Environment/
├── training_env_1_plan.md          ← this file
├── training_env_1.py               ← main entry point (run this)
├── baseline_runner.py              ← runs §8 baseline (no training, just measurement)
├── env_config.py                   ← all config parameters (friction, arena, rewards)
├── loco_policy.py                  ← locomotion command policy (network + CEM/PPO update)
├── reward_fn.py                    ← reward function, isolated for easy tuning
├── robot_state.py                  ← RobotState class (obs collection, reward, reset)
├── metrics.py                      ← logging, CSV export, per-episode tracking
├── checkpoints/                    ← saved policy weights per phase
│   ├── phase1_best.npz
│   ├── phase2_best.npz
│   └── phase3_best.npz
└── logs/
    ├── baseline_summary.csv        ← one row per episode from §8
    └── baseline_raw.npz            ← per-step arrays for all baseline runs
```

---

## 12. Key Metrics to Track Per Episode

| Metric | Description |
|---|---|
| `friction` | Friction coefficient for this episode |
| `total_reward` | Cumulative locomotion reward |
| `fall` | Did robot fall? (bool) |
| `survival_time` | How many seconds before fall (or 120s if no fall) |
| `mean_vel_x` | Mean actual forward velocity achieved |
| `mean_cmd_vx` | Mean commanded forward velocity |
| `slip_ratio` | `mean_vel_x / mean_cmd_vx` (1.0 = no slip) |
| `mean_roll_abs` | Mean absolute body roll, degrees |
| `mean_pitch_abs` | Mean absolute body pitch, degrees |
| `mean_roll_rate` | Mean absolute roll angular velocity (oscillation proxy) |
| `cmd_smoothness` | Mean squared step-to-step change in vx command |

Log to CSV per episode. Print summary every 10 episodes to console.

---

## 13. Implementation Notes

### Python version requirement
IsaacSim requires **Python 3.11** exactly. All scripts must be run using the Python interpreter bundled with IsaacSim (or the `isaacSim_env` virtual environment in this repo), not a system Python. Do not use `python` or `python3` directly from the shell — use the IsaacSim launcher or the venv interpreter:

```bash
# From project root
./isaacSim_env/Scripts/python.exe training_env_1.py
./isaacSim_env/Scripts/python.exe baseline_runner.py --headless
```

Any third-party packages (NumPy, PyTorch, stable-baselines3, etc.) must be installed into this Python 3.11 environment, not a system-level environment. Verify before adding a package:

```bash
./isaacSim_env/Scripts/python.exe -c "import torch; print(torch.__version__)"
```

### Friction material path
When using `world.scene.add_default_ground_plane()`, the material prim is typically at:
```
/World/defaultGroundPlane/Looks/theGrid/PhysicsMaterial
```
Verify on first run with a stage traversal — the path can vary by IsaacSim version.

### Robot reset pattern
From `robots.py`: use `set_world_pose()` + `set_linear_velocity()` + `set_angular_velocity()` to reset. Do **not** re-create the robot prim each episode. On reset, place the robot back at X = −48m with zero velocity and the same starting orientation.

### Parallel robot efficiency
All 5 robots run the same episode with the same friction setting. Each robot produces independent locomotion data, giving 5× throughput with no extra physics overhead.

### Stabilization period
Keep a 3-second stabilization window (from `test_env1.py`) before issuing any velocity commands. This prevents penalizing the policy for instability caused by physics settling at spawn.

### Physics callback pattern
Use `world.add_physics_callback("control", on_physics_step)` at 500 Hz. Decimate to 50 Hz inside the callback (every 10th step) for the policy control loop.

### Extracting actual velocity and attitude
```python
pos, quat = spot.robot.get_world_pose()
vel = spot.robot.get_linear_velocity()       # [vx, vy, vz] world frame
ang_vel = spot.robot.get_angular_velocity()  # [wx, wy, wz] world frame

# Roll and pitch from quaternion
w, x, y, z = quat[0], quat[1], quat[2], quat[3]
roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
```

---

## 14. Success Criteria for Environment 1

Environment 1 is considered complete when the trained policy outperforms the §8 baseline across all friction levels on **locomotion metrics** (not navigation metrics):

1. **Fall rate** reduced by ≥ 50% vs. baseline at μ = 0.2
2. **Mean body roll** ≤ 6° at all friction levels (vs. baseline measurement)
3. **Mean body pitch** ≤ 6° at all friction levels
4. **Slip ratio** ≥ 0.85 at μ = 0.5 (robot achieves ≥ 85% of its commanded speed)
5. **Command smoothness** improved by ≥ 30% vs. baseline (less jerk in velocity commands)

Once these are met, proceed to Environment 2 (add terrain variation — slopes, rough terrain — where locomotion adaptability is exercised more aggressively).

---

*Last updated: 2026-02-18*
