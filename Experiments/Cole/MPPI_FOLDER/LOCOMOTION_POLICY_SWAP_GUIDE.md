# Swapping the Locomotion Policy in the MPPI Testing Environment

This guide explains every place in `Testing_Environment_MPPI.py` that must be
changed when replacing the default `SpotFlatTerrainPolicy` with a different
locomotion policy (e.g. `SpotRoughTerrainPolicy` backed by a custom PPO
checkpoint).

---

## Architecture Overview

The environment uses a two-layer stack:

```
MPPI Navigator (20 Hz)
   └─ produces [vx, vy, omega] velocity command
         │
         ▼
Locomotion Policy (50 Hz)
   └─ converts velocity command → 12 joint position targets
         │
         ▼
Spot Robot (PhysX simulation)
```

The locomotion policy is stored in `env.spot`. The `forward(dt, command)`
method is the only interface MPPI calls each physics step — any policy that
exposes this method can be dropped in.

---

## Step-by-Step Swap Instructions

### 1 — Import the new policy class

**File location:** near line 48, in the import block.

**Current (flat policy):**
```python
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
```

**For `SpotRoughTerrainPolicy` (parkour / rough-terrain PPO):**
```python
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy  # keep — robot still needs this
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'PARKOUR_NAV_handoff', 'code'))
from spot_rough_terrain_policy import SpotRoughTerrainPolicy   # noqa: E402
```

> **Why keep the flat import?**  
> `SpotRoughTerrainPolicy` shares the robot articulation that `SpotFlatTerrainPolicy`
> creates. You must still construct the flat policy first; the rough policy wraps it.

---

### 2 — Construct and initialize the policy

**File location:** around line 1277–1295, inside `main()`, after `world.reset()`.

**Current (flat policy only):**
```python
spot_prim_path = "/World/Spot"
spot = SpotFlatTerrainPolicy(
    prim_path=spot_prim_path,
    name="Spot",
    position=np.array([SPOT_START_X, SPOT_START_Y, SPOT_START_Z])
)
print(f"[OK] SpotFlatTerrainPolicy created at {spot_prim_path}")

world.reset()
print("[OK] World reset")

spot.initialize()
spot.robot.set_joints_default_state(spot.default_pos)
print("[OK] Spot (flat) initialized")

setup_spot_sensors(spot_prim_path)

env = CircularWaypointEnv(world, stage, rng)
env.spot = spot
```

**Swapped to `SpotRoughTerrainPolicy`:**
```python
spot_prim_path = "/World/Spot"
spot = SpotFlatTerrainPolicy(
    prim_path=spot_prim_path,
    name="Spot",
    position=np.array([SPOT_START_X, SPOT_START_Y, SPOT_START_Z])
)
print(f"[OK] SpotFlatTerrainPolicy created at {spot_prim_path}")

world.reset()
print("[OK] World reset")

spot.initialize()
spot.robot.set_joints_default_state(spot.default_pos)
print("[OK] Spot (flat) initialized")

# --- Swap in the rough / parkour locomotion policy ---
_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'PARKOUR_NAV_handoff', 'parkour_phase3_7000.pt'
)
spot_rough = SpotRoughTerrainPolicy(
    flat_policy=spot,
    checkpoint_path=_CHECKPOINT,
    action_scale=0.3,            # MUST match the training config
    robot_prim_path=spot_prim_path,
)
spot_rough.initialize()
spot_rough.apply_gains()         # sets Kp=60, Kd=1.5, PhysX solver 4/0 iterations
print("[OK] SpotRoughTerrainPolicy initialized")
# --- End swap ---

setup_spot_sensors(spot_prim_path)

env = CircularWaypointEnv(world, stage, rng)
env.spot = spot_rough            # <-- point env at the new policy
```

---

### 3 — Add `post_reset()` to the episode reset

**File location:** around line 958–966, inside `CircularWaypointEnv.reset()`.

The flat policy has no internal state to clear between episodes. The rough
policy tracks `_previous_action`, `action`, and `_policy_counter`; these must
be zeroed on reset to prevent stale state from one episode leaking into the
next.

**Current (flat, no extra call needed):**
```python
self.spot.robot.set_joints_default_state(self.spot.default_pos)
print(f"[OK] Spot reset to start ({SPOT_START_X}, {SPOT_START_Y}, {SPOT_START_Z})")
```

**Add for rough policy:**
```python
self.spot.robot.set_joints_default_state(self.spot.default_pos)
self.spot.post_reset()           # clears action buffers between episodes
print(f"[OK] Spot reset to start ({SPOT_START_X}, {SPOT_START_Y}, {SPOT_START_Z})")
```

---

### 4 — (No changes required) MPPI command call site

The `forward(dt, command)` call at line 1082 does **not** need to change:

```python
self.spot.forward(step_size, self._cached_command)
```

Both `SpotFlatTerrainPolicy` and `SpotRoughTerrainPolicy` expose the same
`forward(dt, np.ndarray[3])` signature. MPPI is unaware of which policy is
underneath.

> **Note on `vy`:** The comment at line 1072 states that `SpotFlatTerrainPolicy`
> only accepts `[vx, 0.0, omega]`. The rough policy also zeroes `vy` internally,
> so the existing `self._cached_command = np.array([_cmd[0], 0.0, _cmd[2]])` line
> is correct for both policies and does not need to change.

---

## Critical Parameters to Match

When loading any PPO checkpoint, these values **must** match what was used
during training or the policy will produce wrong joint targets:

| Parameter | Where to set | Notes |
|---|---|---|
| `action_scale` | `SpotRoughTerrainPolicy(..., action_scale=X)` | `parkour_phase3_7000.pt` uses **0.3**. Other checkpoints may use 0.2 or 0.4. |
| Kp / Kd | Set automatically by `apply_gains()` | `parkour_phase3_7000.pt` trains with Kp=60, Kd=1.5 |
| PhysX solver iterations | Set automatically by `apply_gains()` | Must be 4/0; default 32/32 amplifies commands ~10× |
| Spawn Z | `SPOT_START_Z` constant (~line 68) | Rough policy was trained at z=0.55; flat policy default is z=0.7 |
| DOF order | Handled inside `SpotRoughTerrainPolicy` | Type-grouped `[hx×4, hy×4, kn×4]` — do not reorder manually |

---

## Reverting to the Flat Policy

To go back to `SpotFlatTerrainPolicy`:

1. Remove the `sys.path.insert` and `SpotRoughTerrainPolicy` import lines.
2. Remove the `spot_rough` construction block; change `env.spot = spot_rough` back to `env.spot = spot`.
3. Remove `self.spot.post_reset()` from `CircularWaypointEnv.reset()`.

Everything else (MPPI parameters, obstacle fill rates, waypoints) is unaffected
by the locomotion policy swap.

---

## File Locations Reference

| File | Purpose |
|---|---|
| `Testing_Environment_MPPI.py` | Main test environment — all edits above go here |
| `mppi_navigator.py` | MPPI planner — no changes needed for a policy swap |
| `../PARKOUR_NAV_handoff/parkour_phase3_7000.pt` | PPO checkpoint (parkour policy) |
| `../PARKOUR_NAV_handoff/code/spot_rough_terrain_policy.py` | `SpotRoughTerrainPolicy` wrapper class |
| `../PARKOUR_NAV_handoff/CLAUDE_CONTEXT.md` | Full architecture notes for the parkour policy |

---

---

# Using `mppi_navigator.py` — Standalone Guide

`MPPINavigator` is a pure Python/NumPy class with **no Isaac Sim dependency**.
It can be imported and used in any script, notebook, or headless test harness.

---

## What It Does

Every call to `solve()` does the following in ~8 ms on CPU:

1. Samples **K=512** random velocity perturbation sequences over a **H=25-step** horizon
2. Simulates each as a unicycle trajectory (x, y, yaw)
3. Scores each trajectory on goal distance, heading alignment, obstacle clearance, and arena boundary
4. Returns the **information-theoretic weighted-best** first command `[vx, vy, omega]`
5. Warm-starts the next call using the shifted optimal sequence

---

## Import

```python
import sys
import os
sys.path.insert(0, r"C:\path\to\MPPI_FOLDER")   # or a relative path
from mppi_navigator import MPPINavigator
```

No additional packages required beyond `numpy`.

---

## Minimal Usage

```python
import numpy as np
from mppi_navigator import MPPINavigator

nav = MPPINavigator()          # all defaults — ready to use

pos    = np.array([0.0, 0.0])  # current robot (x, y) in metres
yaw    = 0.0                   # current heading in radians
target = np.array([10.0, 5.0]) # waypoint (x, y)
obstacles = []                  # empty — no obstacles

cmd = nav.solve(pos, yaw, target, obstacles)
# cmd = [vx, vy, omega]   e.g. [1.23, 0.0, 0.31]
```

---

## Obstacles Format

Pass a list of `(ox, oy, radius)` tuples — one per obstacle bounding circle:

```python
obstacles = [
    (3.0,  2.0, 0.6),   # obstacle at (3, 2) with 0.6 m radius
    (-1.0, 4.0, 0.4),
]
cmd = nav.solve(pos, yaw, target, obstacles)
```

The `r_safe` margin (default 0.4 m) is added on top of the obstacle radius, so
the effective avoidance distance is `radius + r_safe`.

---

## Episode Reset

Call `nav.reset()` at the start of each new episode to clear the warm-start
state. Without this, the previous episode's optimal sequence bleeds into the
first solve of the new episode.

```python
nav.reset()   # clears internal warm-start buffer
```

---

## Testing in Random Environments (No Isaac Sim)

You can run hundreds of random trials in seconds using a simple unicycle simulator:

```python
import numpy as np
import math
from mppi_navigator import MPPINavigator

def unicycle_step(x, y, yaw, cmd, dt=0.05):
    vx, vy, omega = cmd
    x   += (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
    y   += (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
    yaw += omega * dt
    return x, y, yaw

def random_obstacles(n=20, arena_r=25.0, rng=None):
    rng = rng or np.random.default_rng()
    r   = rng.uniform(2.0, arena_r - 2.0, n)
    ang = rng.uniform(0, 2 * math.pi, n)
    ox  = r * np.cos(ang)
    oy  = r * np.sin(ang)
    rad = rng.uniform(0.3, 0.8, n)
    return list(zip(ox.tolist(), oy.tolist(), rad.tolist()))


rng = np.random.default_rng(42)
nav = MPPINavigator(arena_radius=25.0)

successes = 0
N_TRIALS  = 50

for trial in range(N_TRIALS):
    nav.reset()
    x, y, yaw = 0.0, 0.0, 0.0
    target = np.array([rng.uniform(-20, 20), rng.uniform(-20, 20)])
    obstacles = random_obstacles(rng=rng)

    for step in range(500):           # max 500 steps × 0.05 s = 25 s
        cmd = nav.solve(
            pos=np.array([x, y]),
            yaw=yaw,
            target=target,
            obstacles=obstacles,
        )
        x, y, yaw = unicycle_step(x, y, yaw, cmd)

        dist = math.sqrt((x - target[0])**2 + (y - target[1])**2)
        if dist < 1.0:
            successes += 1
            break

print(f"Success rate: {successes}/{N_TRIALS} = {100*successes/N_TRIALS:.0f}%")
```

---

## Constructor Parameters

All parameters have defaults matching the Isaac Sim arena setup:

| Parameter | Default | Description |
|---|---|---|
| `horizon` | 25 | Lookahead steps. `horizon × dt` = lookahead time (1.25 s). |
| `num_samples` | 512 | Number of sampled trajectories K. More = smoother but slower. |
| `temperature` | 0.03 | MPPI λ. Lower = greedier selection; higher = softer blending. |
| `dt` | 0.05 | Control timestep in seconds (matches 20 Hz nav rate). |
| `sigma_vx` | 0.8 | Forward speed noise std. |
| `sigma_vy` | 0.0 | Lateral speed noise std. (0 = no lateral motion) |
| `sigma_omega` | 0.25 | Turn rate noise std. |
| `vx_min/max` | 0.0 / 3.0 | Forward speed bounds m/s. |
| `omega_min/max` | -1.5 / 1.5 | Turn rate bounds rad/s. |
| `w_goal` | 20.0 | Terminal goal cost weight. |
| `w_heading` | 1.5 | Running heading error cost weight. |
| `w_obs` | 80.0 | Obstacle penalty weight. Raise if robot clips obstacles. |
| `w_bound` | 50.0 | Arena boundary penalty weight. |
| `r_safe` | 0.4 | Safety margin added to obstacle radius (metres). |
| `arena_radius` | 25.0 | Arena radius for boundary cost (metres). |

---

## Tuning Tips

- **Robot overshoots waypoints** → increase `w_heading` or decrease `vx_max`
- **Robot clips obstacles** → increase `w_obs` or `r_safe`
- **Robot spins in place** → increase `w_goal` or raise `sigma_vx`
- **Planning too slow on CPU** → reduce `num_samples` to 256
- **Robot won't turn sharply** → increase `sigma_omega` or widen `omega_min/max`
