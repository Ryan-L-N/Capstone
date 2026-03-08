# Cole — Circular Training Environment Specification

**File:** `Testing_Environment_2.py`  
**Location:** `Experiments/Cole/Testing_Environments/`  
**Author:** Cole (MS for Autonomy Project)  
**Date:** February 2026  
**Simulator:** NVIDIA Isaac Sim 5.1 / Isaac Lab  

---

## 1. Environment Overview

A circular, flat-terrain training arena designed for reinforcement-learning–based navigation of a Boston Dynamics Spot quadruped robot. Each episode fully randomizes obstacle placement, obstacle properties, and waypoint positions so that the policy generalizes rather than memorizing a fixed map.

---

## 2. Environment Geometry

| Property | Value |
|---|---|
| Shape | Circle |
| Diameter | 50 m |
| Radius | 25 m |
| Center | (0, 0) |
| Terrain | Flat (ground plane, z = 0) |
| Boundary wall height | 1.5 m |
| Boundary wall thickness | 0.3 m |
| Boundary approximation | 64-segment polygon wall |
| Stage units | Meters |
| Physics timestep | 1/500 s (500 Hz) |
| Rendering timestep | 10/500 s (50 Hz) |

The circular boundary is approximated in USD using 64 thin box segments arranged in a ring. Each segment subtends 360°/64 = 5.625° of arc and is tangent to the inner circle radius.

---

## 3. Obstacle Specifications

### 3.1 General Requirements

- Obstacles **randomly generate each episode** (positions, shapes, sizes, weights all re-drawn).  
- Combined obstacle footprint = **20 % of the total arena area**.  
  - Arena area: π × 25² ≈ **1,963 m²**  
  - Target obstacle footprint: ≈ **392.7 m²**  
- All objects are referred to as **obstacles** throughout code and documentation.  
- Obstacles maintain a minimum clearance of **3 m** from the arena boundary and **5 m** from every waypoint.

### 3.2 Allowed Shapes

| Shape ID | USD Primitive / Mesh | Notes |
|---|---|---|
| `rectangle` | `UsdGeom.Mesh` | Rectangular footprint, random aspect ratio |
| `square` | `UsdGeom.Mesh` | Equal-sided footprint |
| `trapezoid` | `UsdGeom.Mesh` | Custom 4-vertex footprint |
| `sphere` | `UsdGeom.Sphere` | Uniform radius, rests on ground |
| `diamond` | `UsdGeom.Mesh` | Rhombus/diamond footprint |
| `oval` | `UsdGeom.Sphere` (scaled) | Non-uniform XY scale → ellipsoid |
| `cylinder` | `UsdGeom.Cylinder` | Upright cylinder |

### 3.3 Size Constraints

| Limit | Value | Notes |
|---|---|---|
| Minimum footprint | ≥ 9 in² (≈ 0.0058 m²) | Smallest permissible obstacle |
| Maximum footprint | ≤ Spot bounding box | Spot ≈ 1.1 m × 0.5 m → 0.55 m² |
| Maximum individual height | 1.0 m | Prevents visual clutter |

### 3.4 Weight Constraints & Interaction Behavior

| Category | Mass Range | Physics Behavior | Spot Behavior |
|---|---|---|---|
| Light | 0 – 0.45 kg (< 1 lb) | Low mass, physics-enabled, can be displaced | **Spot may push it aside** |
| Heavy | 0.45 kg – 32.7 kg (Spot's weight) | High mass, near-static | **Spot must navigate around it** |

- Light obstacles: `UsdPhysics.MassAPI` mass set ≤ 0.45 kg; no lock.  
- Heavy obstacles: `UsdPhysics.MassAPI` mass set > 0.45 kg up to 32.7 kg; no lock (relying on inertia).  
- All obstacles have `UsdPhysics.CollisionAPI` and `UsdPhysics.RigidBodyAPI` applied.  
- Light obstacles are colored **orange** (RGB 1.0, 0.55, 0.0).  
- Heavy obstacles are colored **steel blue** (RGB 0.27, 0.51, 0.71).  

### 3.5 Obstacle Count Estimation

Target obstacle coverage of ~392.7 m² with average footprint of ~0.15 m²  
→ approximately **≈ 60 – 80 obstacles per episode** (recalculated each episode based on random sizes).

---

## 4. Spot Robot Configuration

### 4.1 Model

| Property | Value |
|---|---|
| Robot | Boston Dynamics Spot |
| Isaac Sim class | `SpotFlatTerrainPolicy` (`omni.isaac.quadruped.robots`) |
| RL policy | `FlatTerrain` (built-in locomotion policy) |
| Initial position | (0, 0, 0.7) — center of arena |
| Bounding box (approx.) | 1.1 m (L) × 0.5 m (W) × 0.6 m (H) |
| Weight | ~32.7 kg |

### 4.2 Sensors

All standard sensors included by `SpotFlatTerrainPolicy` plus supplementary sensor configuration:

| Sensor | Type | Notes |
|---|---|---|
| Front camera | RGB, 640 × 480 | Forward-facing |
| Depth camera | Depth, 640 × 480 | Used for proximity detection |
| IMU | Linear accel + angular vel | Built-in base link |
| LiDAR | 360° planar scan, 10 m range | 1440 scan points at 0.25° res |
| Joint encoders | 12 joints (3/leg × 4 legs) | Position + velocity |
| Contact sensors | 4 foot pads | Ground contact detection |

### 4.3 Speed Control

| Parameter | Value |
|---|---|
| Maximum forward speed | 5 mph ≈ **2.235 m/s** |
| Minimum speed (near obstacle) | 0.3 m/s |
| Proximity slowdown radius | 2.0 m |
| Speed scaling function | Linear: `v = v_max × (d / r_slow)`, clamped to [v_min, v_max] |

Speed is computed each step by:
1. Querying all obstacle positions.
2. Finding the nearest obstacle distance `d`.
3. Scaling the commanded forward speed proportionally when `d < r_slow`.

Formula:

```
v_cmd = v_max × clamp(d / r_slow, v_min / v_max, 1.0)
```

---

## 5. Waypoints

### 5.1 Properties

| Property | Value |
|---|---|
| Count | 25 |
| Labels | A – Y (alphabetical) |
| Start point | A = (0, 0) — same as Spot's start |
| Minimum spacing | Each waypoint ≥ 25 m from the previous |
| Randomization | Positions re-drawn each episode |
| Clearance from boundary | ≥ 2 m inside circle radius |
| Visual marker | **Flag-on-pole** — grey pole (r = 0.05 m, h = 2.5 m) with coloured banner (0.7 m × 0.4 m) at top |
| Pole colour | Light grey (RGB 0.88, 0.88, 0.88) |
| Banner colour — Waypoint A | Bright green (RGB 0.2, 0.9, 0.2) — start marker |
| Banner colour — Waypoints B–Y | Bright yellow (RGB 1.0, 0.95, 0.0) |
| Distinctiveness | Flag silhouette and colour are visually distinct from all obstacle shapes and colours |

### 5.2 Visit Order

Spot must visit waypoints in alphabetical order: **A → B → C → … → Y**.  
A waypoint is "reached" when Spot's base position is within **1.5 m** of the waypoint center.

### 5.3 Placement Algorithm

```
waypoints = [A = (0, 0)]
for each label B … Y:
    repeat up to MAX_ATTEMPTS times:
        sample angle θ uniformly in [0, 2π)
        d = 25 m  (exact spacing)
        candidate = prev_waypoint + (d·cos θ, d·sin θ)
        if candidate is inside circle (radius 23 m) and > 25 m from all previous waypoints:
            accept
    if no valid placement found:
        relax spacing to best available within circle
```

Waypoints are stored in a dict `{label: np.array([x, y])}`.

### 5.4 Waypoint System Specification (Final Optimized Version)

#### Spacing Rules

| Waypoint | Placement Rule |
|---|---|
| A | Placed exactly **24 m** from the start point (0, 0) |
| B – Z | Each placed at least **30 m** from the **previous** waypoint |
| Non-adjacent pairs | No spacing requirement |

#### Sequential Spawning

Only **one waypoint exists in the scene at any time**. Waypoints are spawned and despawned on demand:

1. **Episode start** — Spawn waypoint **A** exactly 24 m from (0, 0) in a random valid direction.
2. When Spot reaches the active waypoint:
   - Award **+15 points**
   - Log the waypoint index
   - **Despawn** the current waypoint
   - **Spawn the next waypoint** at the required distance from the current one (re-rolling direction if outside arena)
3. Repeat until all waypoints have been collected or the episode terminates.

#### Placement Rules

A waypoint placement is **accepted** if:

- The candidate position is inside the **25 m radius arena** (minimum 2 m clearance from boundary).
- The required offset from the previous waypoint lands within those bounds.
- If the candidate falls outside the arena, **re-roll the direction** until a valid placement is found.

Each waypoint must:
- Include a **flag-on-pole marker** for visibility (grey pole + coloured banner).
- Be **visually distinct** from all obstacle shapes and colours.

#### Logging

At the end of every episode, the following are appended to `training_log.csv`:

| Column | Description |
|---|---|
| `Episode` | Episode number (1-indexed, persists across runs) |
| `Waypoints_Reached` | Total waypoints collected that episode |
| `Time_Elapsed` | Sim-seconds elapsed before termination |
| `Final_Score` | Score at episode end |

The visit **order** is also printed to the console each episode (e.g. `A → B → C → …`).

---

## 6. Episode Randomization

Every call to `env.reset()` randomizes:

| Element | Method |
|---|---|
| Obstacle positions | Uniform random inside circle, clearance-checked |
| Obstacle shapes | Uniform sample from 7 shape types |
| Obstacle sizes | Uniform random within per-shape bounds |
| Obstacle weights | Uniform random in [0.05 kg, 32.7 kg] |
| Obstacle count | Recomputed to achieve ≈ 20% area coverage |
| Waypoint positions | Chain placement from (0,0) with 25 m spacing |

All randomization uses `numpy.random` with a per-episode seed (derived from episode counter) for reproducibility during debugging.

---

## 7. Reward Function (RL Policy)

### 7.1 Episode Scoring Framework

Spot begins every episode with a **300-point bank**. The score acts as the primary RL reward signal and as the episode clock — the episode terminates as soon as the score reaches zero.

| Event | Score / Reward | Notes |
|---|---|---|
| Episode start | **+300 pts** | Fixed starting bank |
| Time decay | **−1 pt / sim-second** | Applied continuously every step |
| Waypoint reached (in order) | **+15 pts** | Awarded per waypoint A → Y |
| Fall detected (base z < 0.3 m) | Score → **0**, episode ends | Fall sets score to zero immediately |
| Score reaches 0 | Episode terminates | Acts as a soft time limit |

### 7.2 Termination Conditions

| Condition | Trigger | Result |
|---|---|---|
| Score depletion | `score ≤ 0` | Episode ends (`reason = score_depleted`) |
| Fall | `pos.z < 0.3 m` | Score forced to 0; episode ends (`reason = fall`) |
| Course complete | All 24 waypoints collected (B–Y) | Episode ends (`reason = complete`) |

### 7.3 Reward Modularity

All shaping terms are computed inside `CircularWaypointEnv.compute_reward()`, which returns a named component dictionary. This makes it trivial to add or disable shaping without touching `step()`:

```python
def compute_reward(self, pos, dist_to_wp, nearest_obs) -> dict:
    components = {}
    components["time_decay"] = -TIME_DECAY_PER_SEC * PHYSICS_DT  # always active
    # Future hooks (uncomment to enable):
    # components["energy"]     = PENALTY_ENERGY_COEFF * motor_effort
    # components["smoothness"] = REWARD_SMOOTHNESS * smoothness_metric
    # components["obs_avoid"]  = 0.05 * min(nearest_obs / OBSTACLE_SLOW_RADIUS, 1.0)
    return components
```

### 7.4 Waypoint Tracking & CSV Logging

For every episode, the following are tracked and appended to `training_log.csv`:

| Column | Description |
|---|---|
| `Episode` | Episode number (1-indexed, persists across runs) |
| `Waypoints_Reached` | Total waypoints collected that episode |
| `Time_Elapsed` | Sim-seconds elapsed before termination |
| `Final_Score` | Score at episode end |

- The CSV file is created on first run with a header row; subsequent runs **append** (never overwrite).
- File location: `Experiments/Cole/Testing_Environments/training_log.csv`
- Waypoint **visit order** is also logged to the console as `A → C → B → …` each episode.

---

## 8. Observation Space

| Component | Dimension | Description |
|---|---|---|
| Base linear velocity | 3 | vx, vy, vz |
| Base angular velocity | 3 | wx, wy, wz |
| Gravity vector (projected) | 3 | g in body frame |
| Joint positions | 12 | All 12 joints |
| Joint velocities | 12 | All 12 joints |
| Previous action | 12 | Last joint command |
| Current waypoint vector | 2 | (dx, dy) to active waypoint |
| Distance to current waypoint | 1 | Scalar distance |
| Nearest obstacle distance | 1 | From LiDAR min |
| **Total** | **49** | |

---

## 9. Action Space

| Property | Value |
|---|---|
| Type | Continuous |
| Dimension | 3 |
| Components | [forward_speed, lateral_speed, yaw_rate] |
| Forward speed bounds | [0, 2.235] m/s |
| Lateral speed bounds | [−1.0, 1.0] m/s |
| Yaw rate bounds | [−1.5, 1.5] rad/s |

---

## 10. File Structure

```
Experiments/Cole/Testing_Environments/
├── Cole_md.md                     ← This specification file
├── Lessons_Learned.md             ← Running lessons-learned log
└── Testing_Environment_2.py       ← Full training environment implementation
```

Dependencies:
- `omni.isaac.quadruped.robots.SpotFlatTerrainPolicy`
- `omni.isaac.core.World`
- `omni.isaac.sensor` (Camera)
- `pxr` (USD: UsdGeom, UsdPhysics, UsdLux, Gf)
- `numpy`
- `isaacsim.SimulationApp`
- `Experiments/Cole/Spots/Spot_1.py` (SpotRobot wrapper)

---

## 11. Environment Constants Summary

```python
ARENA_RADIUS          = 25.0          # meters
ARENA_CENTER          = (0.0, 0.0)
WALL_SEGMENTS         = 64            # polygon approx of circle
WALL_HEIGHT           = 1.5           # meters
SPOT_START_POS        = (0.0, 0.0, 0.7)
SPOT_MAX_SPEED        = 2.235         # m/s  (5 mph)
SPOT_MIN_SPEED        = 0.3           # m/s
OBSTACLE_AREA_FRAC    = 0.20          # 20% of arena
WAYPOINT_COUNT        = 25
WAYPOINT_SPACING      = 25.0          # meters (min distance between consecutive)
WAYPOINT_REACH_DIST   = 1.5           # meters (threshold to "reach" a waypoint)
OBSTACLE_MAX_MASS     = 32.7          # kg (Spot's weight)
OBSTACLE_LIGHT_THRESH = 0.45          # kg (1 lb)
PHYSICS_DT            = 1/500         # seconds
RENDERING_DT          = 10/500        # seconds

# Reward / Scoring
EPISODE_START_SCORE   = 300.0         # points at episode start
TIME_DECAY_PER_SEC    = 1.0           # points lost per sim-second
WAYPOINT_REWARD       = 15.0          # points awarded per waypoint collected

# Waypoint flag geometry
WP_POLE_HEIGHT        = 2.5           # meters
WP_POLE_RADIUS        = 0.05          # meters
WP_FLAG_WIDTH         = 0.7           # meters
WP_FLAG_HEIGHT        = 0.40          # meters

# CSV log
CSV_LOG_PATH          = "Testing_Environments/training_log.csv"
CSV_HEADERS           = ["Episode", "Waypoints_Reached", "Time_Elapsed", "Final_Score"]
```

---

---

## 12. Changelog

| Date | Change |
|---|---|
| 2026-02-18 | Initial specification created |
| 2026-02-18 | Reward system updated: 300-pt starting bank, −1 pt/s decay, +15 per waypoint, fall → score = 0 |
| 2026-02-18 | Waypoint markers updated from plain cylinder to flag-on-pole (grey pole + coloured banner) |
| 2026-02-18 | Added CSV episode logging: `training_log.csv` with Episode, Waypoints_Reached, Time_Elapsed, Final_Score |
| 2026-02-18 | Reward function made modular via `compute_reward()` with named component dict |
| 2026-02-18 | Added Section 5.4 — Waypoint System Specification (Optimized): sequential single-waypoint spawning, chain spacing rule, placement re-roll logic, CSV logging spec |
| 2026-02-18 | Section 5.4 updated to Final Optimized Version: A placed 24 m from start, B–Z each ≥ 30 m from previous, sequential spawn/despawn, re-roll on out-of-bounds |

---

*Last updated: February 18, 2026*
