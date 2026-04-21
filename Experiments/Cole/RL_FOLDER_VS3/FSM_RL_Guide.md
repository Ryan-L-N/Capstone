# FSM + RL Navigation System — How It Works & How to Use It

**File:** `Testing_Environment.py`  
**Policy:** `RL_FOLDER_VS3/navigation_policy.py`  
**Checkpoint used:** `aggressive/checkpoints/stage_4_complete.pt`

---

## Overview

The system uses a **two-layer architecture**:

1. **RL Policy (NavigationPolicy)** — a trained PyTorch MLP that reads sensor observations and outputs raw velocity commands `[vx, vy, omega]`.
2. **FSM (Finite State Machine)** — a deterministic 4-state override layer that intercepts the RL output and modifies or replaces it based on geometry rules.

The RL policy handles general waypoint navigation (long-range planning, smooth paths). The FSM handles cases the policy handles poorly: large heading misalignment and close-range obstacle avoidance. Neither component works optimally alone.

```
Sensors → Observation Vector → RL Policy → Raw [vx, vy, ω]
                                                   ↓
                                        FSM Priority Check
                                                   ↓
                                   Final Command → SpotFlatTerrainPolicy
```

---

## The RL Policy

### Architecture

`NavigationPolicy` is a **3-layer MLP** (256 → 256 → 128 → 3) with ReLU activations and a Tanh output layer.

- **Input:** 34-dimensional observation vector
- **Output:** `[vx, vy, omega]` in the range `[-1, 1]` (scaled after inference)
- **Critic:** Separate MLP (256 → 128 → 64 → 1) used only during training for PPO value estimates

### Observation Vector (34 dims)

| Index | Size | Description |
|-------|------|-------------|
| 0–2   | 3    | Base velocity `[vx, vy, omega]` from Spot's own sensors |
| 3–4   | 2    | Heading encoded as `[sin(yaw), cos(yaw)]` |
| 5–7   | 3    | Waypoint in robot frame: `[dx_robot, dy_robot, distance]` |
| 8–25  | 18   | Obstacle ray distances (normalized 0.0–1.0, 5 m range) |
| 26–31 | 6    | Stage one-hot encoding (which training stage was active) |
| 32–33 | 2    | Mode encoding: `[is_turning, is_approaching]` |

**Ray layout:** 18 rays spaced 20° apart around 360°. The three forward-facing rays are:
- Ray 0 → straight ahead
- Ray 1 → 20° left
- Ray 17 → 20° right (wraps around)

**Ray normalization:** `0.0 = obstacle right at sensor`, `1.0 = clear at 5 m`. An obstacle 3 m away returns `0.6`.

### Action Scaling

Raw policy outputs `[-1, 1]`. Scaled before the FSM sees them:

```python
vx    = clip(raw_vx,    -1, 1) * (5.0 if raw_vx > 0 else 0.5)   # forward: up to 5 m/s
vy    = clip(raw_vy,    -1, 1) * 0.5                              # strafe:  ±0.5 m/s
omega = clip(raw_omega, -1, 1) * 1.5                              # yaw:     ±1.5 rad/s
```

---

## The FSM

### How States Are Determined

Every policy inference step (20 Hz), three geometry variables are computed from the ray data:

```python
_DODGE_THRESHOLD = 3.0 / 5.0   # 0.6 normalized → obstacle within 3 m
_SLOW_THRESHOLD  = 5.0 / 5.0   # 1.0 normalized → obstacle within 5 m (full range)

_ray_ahead = obstacle_distances[0]
_ray_left  = obstacle_distances[1]
_ray_right = obstacle_distances[17]

_obstacle_in_front   = any ray < _DODGE_THRESHOLD   (< 3 m)
_obstacle_approaching = any ray < _SLOW_THRESHOLD    (< 5 m)
_min_fwd_ray = min(_ray_ahead, _ray_left, _ray_right)
```

### State Priority (Highest to Lowest)

```
┌──────────────────────────────────────────────────────────────────┐
│  Priority 1 — HEADING CORRECTION   (if |heading_error| > 30°)   │
│  Priority 2 — OBSTACLE NUDGE       (if obstacle within 3 m)      │
│  Priority 3 — OBSTACLE SLOWDOWN    (if obstacle within 5 m)      │
│  Priority 4 — DEFAULT RL           (otherwise)                   │
└──────────────────────────────────────────────────────────────────┘
```

---

### State 1: Heading Correction

**Condition:** `|heading_error| > 30°`

The robot is pointed more than 30° away from the target waypoint. The RL policy tends to wander or spiral when misaligned, so the FSM locks in a yaw-forward command:

```python
self._yaw_dir = 0.0
command = [min(1.0, vx), 0.0, omega]
```

- `vx` from RL (capped at 1.0 so it doesn't sprint while turning)
- `vy` forced to 0 (no strafing while correcting heading)
- `omega` from RL (the policy's heading correction is trusted here)

**Effect:** Robot turns toward the waypoint while moving forward. Exits when heading error drops under 30°.

---

### State 2: Obstacle Nudge

**Condition:** any forward-cone ray < 3 m (`_obstacle_in_front = True`)

An obstacle is close in the forward cone. The FSM picks a yaw direction **once** (the first time it enters this state) and holds it until the obstacle clears:

```python
if self._yaw_dir == 0.0:
    self._yaw_dir = -1.5 if _ray_left < _ray_right else 1.5
command = [min(1.0, vx), 0.0, self._yaw_dir]
```

- Yaws **away from the closer side** (left ray closer → yaw right, and vice versa)
- Holds the fixed `±1.5 rad/s` yaw rate — does not re-evaluate each step
- `_yaw_dir` resets to 0.0 when the state exits

**Why hold the direction?** Without this latch, the robot oscillates — it yaws right, the right ray becomes closer, so it yaws left, repeat. Holding the direction lets it sweep past the obstacle smoothly.

---

### State 3: Obstacle Slowdown

**Condition:** any forward-cone ray < 5 m (`_obstacle_approaching = True`) AND not already in nudge state

Obstacle is in the area but not yet critical. The FSM reduces forward speed proportionally to the gap remaining:

```python
_slow_vx = vx * (_min_fwd_ray / _SLOW_THRESHOLD)
command = [max(0.1, _slow_vx), vy, omega]
```

- `_slow_vx` approaches 0 as the closest ray approaches 0 (obstacle at sensor)
- Floored at `0.1` — robot never completely stops in this state
- `vy` and `omega` from RL unchanged — steering is still fully RL-controlled

**Effect:** Robot slows to creep speed before entering an obstacle cluster, giving it time to steer around rather than snap suddenly.

---

### State 4: Default RL

**Condition:** No heading misalignment, no obstacle in cone

Full RL control. No overrides:

```python
command = [vx, vy, omega]
```

This is the normal cruising state between waypoints in open terrain.

---

## Control Loop Timing

The policy runs at **20 Hz** (every 25 physics steps at 500 Hz). Between inference calls the last command is replayed unchanged:

```
Physics: 500 Hz  ─────────────────────
Policy:   20 Hz  ──|    |    |    |───
                   ↑ infer  ↑ infer
                   ├─────── 25 steps ──┤
                     cached command
                     replayed each step
```

The FSM runs inside each inference call, so it updates at 20 Hz too.

---

## Scoring

| Event | Points |
|-------|--------|
| Start of episode | 300 |
| Per second elapsed | −1 |
| Each waypoint collected | +15 |
| Maximum possible (25 WPs, perfect) | 375 final |

The practical ceiling is ~308 for waypoint A (6 steps ≈ 0.3 sec elapsed before collection).

---

## How to Run

### Basic test (1 episode, GUI)

```powershell
& "c:\isaac-sim\python.bat" `
  "Experiments\Cole\Baseline_Folder\Testing_Environment.py" `
  --episodes 1 `
  --checkpoint "Experiments\Cole\RL_FOLDER_VS3\aggressive\checkpoints\stage_4_complete.pt"
```

### Headless, multiple episodes

```powershell
& "c:\isaac-sim\python.bat" `
  "Experiments\Cole\Baseline_Folder\Testing_Environment.py" `
  --headless `
  --episodes 5 `
  --checkpoint "Experiments\Cole\RL_FOLDER_VS3\aggressive\checkpoints\stage_4_complete.pt"
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | None | Path to `.pt` policy file. **Required** for RL to run. |
| `--episodes` | 1 | Number of episodes to run sequentially. |
| `--headless` | False | Disable GUI for faster simulation. |
| `--seed` | None (random) | Fix RNG seed for reproducibility. |

### CSV Output

Results append to `Experiments/Cole/Baseline_Folder/Baseline_CSV.csv` after each episode:

```
Episode,Waypoints_Reached,Failure_Reason,Final_Score
1,6,Fell Over,265.3
2,12,Ran Out of Points,0.0
```

Failure reasons:
- `Fell Over` — Spot's z-height dropped below 0.25 m
- `Ran Out of Points` — Score hit 0 before timeout
- `Completed` — All 25 waypoints collected

---

## Tuning the FSM

All thresholds are inline constants in `Testing_Environment.py` inside the policy inference block (~line 1660):

| Variable | Value | Effect of increasing |
|----------|-------|---------------------|
| `_DODGE_THRESHOLD` | `3.0 / 5.0 = 0.6` | Nudge activates farther from obstacles |
| `_SLOW_THRESHOLD` | `5.0 / 5.0 = 1.0` | Slowdown zone grows (covers full ray range) |
| `30°` heading threshold | `math.radians(30)` | Robot turns more aggressively before moving |
| `self._yaw_dir = ±1.5` | `1.5 rad/s` | Faster/slower nudge yaw rate |

**Known tradeoffs:**
- Increasing heading threshold → fewer oscillations near waypoints, but more cautious alignment stalls
- Increasing `_DODGE_THRESHOLD` → earlier obstacle detection, but more false positives in dense arenas
- Decreasing `_yaw_dir` magnitude → smoother nudge but may not escape tight obstacle gaps in time

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│                    OBSERVATION (34 dims)                  │
│  [vel(3)] [heading(2)] [wp_robot(3)] [rays(18)] [enc(8)] │
└───────────────────────────┬──────────────────────────────┘
                            │
                   NavigationPolicy MLP
                  256 → 256 → 128 → 3
                   (Tanh output)
                            │
                  Raw [vx, vy, ω] in [-1,1]
                            │
                   Action Scaling
                  vx × 5.0, vy × 0.5, ω × 1.5
                            │
              ┌─────────────▼────────────────┐
              │        FSM PRIORITY          │
              │  1. |heading_err| > 30°?     │ → Yaw-forward command
              │  2. Any fwd ray < 3 m?       │ → Obstacle nudge
              │  3. Any fwd ray < 5 m?       │ → Proportional slowdown
              │  4. Default                  │ → Full RL command
              └─────────────┬────────────────┘
                            │
               Final [vx, vy, ω] cached
                            │
           SpotFlatTerrainPolicy (500 Hz, Isaac Sim)
                            │
                   Spot robot joints
```
