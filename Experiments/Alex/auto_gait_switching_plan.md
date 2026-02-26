# Terrain-Aware Auto Gait Switching for Spot Lava Arena

## Context

The `spot_lava_arena.py` controller currently has three gait modes (FLAT, ROUGH, PARKOUR), but switching between them is **manual only** (G key / Xbox RB button). Inspired by Boston Dynamics' Hazard Avoidance Service API — where terrain is classified by hazard type and the robot adapts its locomotion accordingly — this plan adds **automatic gait switching** based on real-time terrain analysis from PhysX raycasting.

### Boston Dynamics Hazard Avoidance Concept (from `hazard_avoidance.proto`)

The BD API classifies terrain into types like:
- `PREFER_AVOID_WEAK/STRONG` — terrain the robot should avoid stepping on
- `NEVER_STEP_ON` — forbidden footholds
- `PREFER_STEP_ON` — safe terrain the robot should prefer
- `NEVER_STEP_ACROSS` — full no-go zones

We adapt this concept into our Isaac Sim simulation by using the rough terrain policy's existing **187-ray PhysX height scanner** to classify local terrain difficulty and automatically select the best gait.

---

## Architecture Overview

```
                   ┌─────────────────────────┐
                   │   SpotRoughTerrainPolicy │
                   │   ._cast_height_rays()   │
                   │   (187 PhysX rays,       │
                   │    17x11 grid, 1.6x1.0m) │
                   └───────────┬─────────────┘
                               │ np.ndarray (187,)
                               ▼
                   ┌─────────────────────────┐
                   │ TerrainDifficultyAssessor│
                   │                         │
                   │ 1. Compute variance     │
                   │ 2. Compute peak-to-peak │
                   │ 3. Weighted metric      │
                   │ 4. EMA smoothing        │
                   │ 5. Hysteresis check     │
                   │ 6. Confirmation count   │
                   └───────────┬─────────────┘
                               │ recommended gait or None
                               ▼
                   ┌─────────────────────────┐
                   │   on_physics_step()     │
                   │   (every 0.5s scan)     │
                   │                         │
                   │ Update gait_idx[0]      │
                   │ Apply gains/props       │
                   │ Trigger stabilization   │
                   └─────────────────────────┘
```

---

## Key Discovery: Height Scanner Already Exists

`SpotRoughTerrainPolicy._cast_height_rays()` (line 226 in `spot_rough_terrain_policy.py`):
- Casts **187 rays** in a 17x11 grid (1.6m x 1.0m) ahead of the robot
- Uses PhysX `raycast_closest()` — works **independently** of which policy is active
- Returns `np.ndarray (187,)` of height differences relative to body height
- Handles self-hit avoidance (skips rays hitting `/World/Spot`)
- Available at runtime: `spot_rough._cast_height_rays()` and `spot_rough._height_scan`

This scanner works even when the flat policy is driving the robot, since all policies share the same robot body.

---

## Terrain Difficulty Metric

```python
difficulty = 0.6 * variance(height_scan) + 0.4 * (peak_to_peak / 2.0)^2
```

| Difficulty Score | Terrain Type | Expected Gait |
|-----------------|--------------|---------------|
| ~0.00 | Perfectly flat | FLAT |
| 0.01 - 0.03 | Slightly uneven | FLAT |
| 0.04 - 0.08 | Moderate rubble | ROUGH |
| 0.08 - 0.12 | Heavy rubble | ROUGH |
| > 0.12 | Extreme obstacles | PARKOUR |

### Anti-Oscillation: Hysteresis + Confirmation

**Hysteresis** — different thresholds for switching up vs. down:
- FLAT → ROUGH: difficulty > **0.04**
- ROUGH → FLAT: difficulty < **0.02** (gap prevents oscillation at boundary)
- ROUGH → PARKOUR: difficulty > **0.12**
- PARKOUR → ROUGH: difficulty < **0.08**

**Confirmation** — 3 consecutive readings above/below threshold required:
- At 0.5s scan interval, a gait switch takes **1.5 seconds** of sustained terrain change
- Prevents transient spikes (e.g., stepping on one large rock) from triggering false switches

**EMA Smoothing** — exponential moving average (alpha=0.3) smooths the difficulty signal over time.

---

## File to Modify

**Primary:** `spot_lava_arena.py`
```
C:\Users\Gabriel Santiago\OneDrive\Desktop\Nvidia Omniverse\
  AI2C_Tech_Capstone_MS_for_Autonomy\MS_for_autonomy\
  experimental_design_grass\code\spot_lava_arena.py
```

**Reference (read-only):** `spot_rough_terrain_policy.py` in the same directory

---

## Implementation Steps

### Step 1: Add Configuration Constants

**Location:** After `GAIT_SWITCH_STABILIZE = 0.5` (line 213)

```python
# Terrain-aware auto-gait switching
TERRAIN_SCAN_INTERVAL     = 0.5    # seconds between assessments (~2Hz)
TERRAIN_EMA_ALPHA         = 0.3    # EMA smoothing factor (0=slow, 1=instant)
TERRAIN_FLAT_TO_ROUGH     = 0.04   # difficulty threshold: upgrade to ROUGH
TERRAIN_ROUGH_TO_FLAT     = 0.02   # difficulty threshold: downgrade to FLAT
TERRAIN_ROUGH_TO_PARKOUR  = 0.12   # difficulty threshold: upgrade to PARKOUR
TERRAIN_PARKOUR_TO_ROUGH  = 0.08   # difficulty threshold: downgrade from PARKOUR
TERRAIN_CONFIRM_READINGS  = 3      # consecutive readings before switching

AUTO_GAIT_HUD_NAMES = {"FLAT": "A-FLAT", "ROUGH": "A-ROUGH", "PARKOUR": "A-PARK"}
```

Also add `XBOX_BTN_START = 7` near existing Xbox constants (line 209).

---

### Step 2: Add `TerrainDifficultyAssessor` Class

**Location:** After `VelocitySmoother` class (~line 328)

```python
class TerrainDifficultyAssessor:
    """Analyzes height-scan data to compute terrain difficulty and recommend gait.

    Uses SpotRoughTerrainPolicy._cast_height_rays() which casts 187 PhysX rays
    in a 17x11 grid ahead of the robot. Works independently of active policy.
    """

    GAIT_LEVEL = {"flat": 0, "rough": 1, "parkour": 2}

    def __init__(self, spot_rough_policy, has_parkour=False):
        self._spot_rough = spot_rough_policy
        self._has_parkour = has_parkour
        self._ema_difficulty = 0.0
        self._confirm_up_to_rough = 0
        self._confirm_down_to_flat = 0
        self._confirm_up_to_parkour = 0
        self._confirm_down_to_rough = 0
        self.last_raw_difficulty = 0.0
        self.last_ema_difficulty = 0.0

    def reset(self):
        """Reset all state. Call on robot reset."""
        self._ema_difficulty = 0.0
        self._confirm_up_to_rough = 0
        self._confirm_down_to_flat = 0
        self._confirm_up_to_parkour = 0
        self._confirm_down_to_rough = 0
        self.last_raw_difficulty = 0.0
        self.last_ema_difficulty = 0.0

    def assess(self, current_gait_type):
        """Run one terrain assessment cycle.

        Returns:
            str or None: Recommended gait ("flat"/"rough"/"parkour"), or None.
        """
        try:
            height_scan = self._spot_rough._cast_height_rays()
        except Exception:
            return None

        if height_scan is None or len(height_scan) == 0:
            return None

        # Compute raw difficulty
        variance = np.var(height_scan)
        ptp = np.ptp(height_scan)
        raw_difficulty = 0.6 * variance + 0.4 * (ptp / 2.0) ** 2
        self.last_raw_difficulty = raw_difficulty

        # EMA smoothing
        self._ema_difficulty = (
            TERRAIN_EMA_ALPHA * raw_difficulty
            + (1.0 - TERRAIN_EMA_ALPHA) * self._ema_difficulty
        )
        self.last_ema_difficulty = self._ema_difficulty
        difficulty = self._ema_difficulty

        # Check transitions with hysteresis + confirmation
        if current_gait_type == "flat":
            if difficulty > TERRAIN_FLAT_TO_ROUGH:
                self._confirm_up_to_rough += 1
            else:
                self._confirm_up_to_rough = 0
            if self._confirm_up_to_rough >= TERRAIN_CONFIRM_READINGS:
                self._confirm_up_to_rough = 0
                return "rough"

        elif current_gait_type == "rough":
            if difficulty < TERRAIN_ROUGH_TO_FLAT:
                self._confirm_down_to_flat += 1
                self._confirm_up_to_parkour = 0
            elif self._has_parkour and difficulty > TERRAIN_ROUGH_TO_PARKOUR:
                self._confirm_up_to_parkour += 1
                self._confirm_down_to_flat = 0
            else:
                self._confirm_down_to_flat = 0
                self._confirm_up_to_parkour = 0
            if self._confirm_down_to_flat >= TERRAIN_CONFIRM_READINGS:
                self._confirm_down_to_flat = 0
                return "flat"
            if self._confirm_up_to_parkour >= TERRAIN_CONFIRM_READINGS:
                self._confirm_up_to_parkour = 0
                return "parkour"

        elif current_gait_type == "parkour":
            if difficulty < TERRAIN_PARKOUR_TO_ROUGH:
                self._confirm_down_to_rough += 1
            else:
                self._confirm_down_to_rough = 0
            if self._confirm_down_to_rough >= TERRAIN_CONFIRM_READINGS:
                self._confirm_down_to_rough = 0
                return "rough"

        return None
```

---

### Step 3: Extract Gait-Switch Helper Functions

**Location:** After `flat_saved_props` is populated (~line 1493)

Currently the actuator restore logic is **duplicated** in the G key handler (lines 1194-1215) and Xbox RB handler (lines 1857-1882). Extract into two shared helpers:

```python
def _switch_to_rough_or_parkour():
    """Apply rough/parkour actuator gains."""
    if spot_rough is not None:
        spot_rough.apply_gains()

def _switch_to_flat():
    """Restore flat policy actuator properties from saved state."""
    try:
        av = spot_flat.robot._articulation_view
        if 'kps' in flat_saved_props:
            av.set_gains(kps=flat_saved_props['kps'], kds=flat_saved_props['kds'])
        if 'efforts' in flat_saved_props:
            av.set_max_efforts(values=flat_saved_props['efforts'])
        # ... (frictions, armatures, max_vels, pos_iters, vel_iters)
    except Exception as e:
        print(f"[GAIT] Restore error: {e}")
```

Then refactor both manual gait handlers (G key + Xbox RB) to call these helpers.

---

### Step 4: Add State Variables

**Location:** After existing state vars (~line 1164)

```python
auto_gait_active = [False]
last_terrain_scan_time = [0.0]
```

---

### Step 5: Instantiate Assessor

**Location:** After policies are initialized (~line 1501)

```python
terrain_assessor = None
if spot_rough is not None:
    terrain_assessor = TerrainDifficultyAssessor(
        spot_rough_policy=spot_rough,
        has_parkour=(spot_parkour is not None))
```

---

### Step 6: Add Controls

**Keyboard — N key** (in `on_keyboard_event`, after H key block ~line 1250):
- Toggle `auto_gait_active[0]`
- Print status

**Xbox — Start button** (in main loop Xbox section ~line 1884):
- Same toggle behavior

**Update docstring** (lines 1-38) and **controls printout** (lines 1274-1300) to document the new bindings.

---

### Step 7: Integrate into `on_physics_step()`

**Location:** After gait_switch_timer guard (~line 1723), before velocity command computation

```python
# Terrain-aware auto-gait (runs at ~2Hz)
if (auto_gait_active[0]
        and terrain_assessor is not None
        and sim_time[0] - last_terrain_scan_time[0] >= TERRAIN_SCAN_INTERVAL):
    last_terrain_scan_time[0] = sim_time[0]
    current_gait = GAIT_MODES[gait_idx[0]]
    recommended = terrain_assessor.assess(current_gait["policy_type"])

    if recommended is not None and recommended != current_gait["policy_type"]:
        # Find matching gait index, validate availability
        new_idx = next((i for i, g in enumerate(GAIT_MODES)
                        if g["policy_type"] == recommended), None)
        if new_idx is not None:
            if recommended == "parkour" and spot_parkour is None:
                new_idx = None
            elif recommended == "rough" and spot_rough is None:
                new_idx = None

        if new_idx is not None and new_idx != gait_idx[0]:
            old_name = GAIT_MODES[gait_idx[0]]["name"]
            gait_idx[0] = new_idx
            new_gait = GAIT_MODES[new_idx]
            if new_gait["policy_type"] in ("rough", "parkour"):
                _switch_to_rough_or_parkour()
            else:
                _switch_to_flat()
            print(f"\n  >> AUTO-GAIT: {old_name} -> {new_gait['name']} "
                  f"(difficulty={terrain_assessor.last_ema_difficulty:.4f})")
```

**Why this location:** Placed after all early-return guards (selfright, recovery, stabilization), so terrain assessment never fires during those states. Setting `gait_idx[0]` here means the policy switch takes effect on the next physics step (lines 1613-1628), which is the correct behavior.

---

### Step 8: Update HUD

**Location:** Lines 1774, 1787-1792

```python
# Gait display with auto-gait prefix
gait_name = GAIT_MODES[gait_idx[0]]['name']
gait_str = AUTO_GAIT_HUD_NAMES.get(gait_name, f"A-{gait_name}") if auto_gait_active[0] else gait_name

# Add difficulty to status line
terrain_str = ""
if auto_gait_active[0] and terrain_assessor is not None:
    terrain_str = f" D:{terrain_assessor.last_ema_difficulty:.3f}"

print(f"\r  [...] {gait_str:>7s} ... | {status}{terrain_str}", end="     ")
```

---

### Step 9: Reset Assessor on Robot Reset

**Location:** In the reset block (~line 1648)

```python
if terrain_assessor is not None:
    terrain_assessor.reset()
```

---

## Edge Cases Handled

| Scenario | Behavior |
|----------|----------|
| `spot_rough is None` | `terrain_assessor = None`, auto-gait toggle prints "unavailable" |
| `spot_parkour is None` | `has_parkour=False`, ROUGH→PARKOUR never triggers |
| Selfright / recovery / stabilization | Assessment skipped (early returns above it) |
| Raycast failure (robot airborne) | `assess()` catches exception, returns `None` |
| Robot reset (R key) | `terrain_assessor.reset()` clears all state |
| Manual G key while auto-gait is ON | Works; auto re-evaluates on next scan |
| Terrain boundary oscillation | Hysteresis gap + 3-reading confirmation prevents it |

---

## New Controls Summary

| Input | Action |
|-------|--------|
| **N** (keyboard) | Toggle auto-gait on/off |
| **Start** (Xbox) | Toggle auto-gait on/off |
| **G** (keyboard) | Manual gait cycle (still works with auto-gait on) |
| **RB** (Xbox) | Manual gait cycle (still works with auto-gait on) |

---

## Verification Plan

1. **Baseline** — Run with auto-gait OFF. Confirm zero behavior change from current code.
2. **Flat ground** — Toggle N on flat terrain. HUD shows `A-FLAT` with `D:0.000`.
3. **Enter rock field** — Walk into arena. Difficulty rises → gait switches FLAT→ROUGH after ~1.5s.
4. **Exit rock field** — Walk out. Difficulty drops → gait switches ROUGH→FLAT.
5. **Boundary patrol** — Walk along arena edge. Confirm no rapid FLAT↔ROUGH oscillation.
6. **Parkour zone** — Navigate to heavy rubble. Confirm ROUGH→PARKOUR at D>0.12 (if parkour policy loaded).
7. **Manual override** — While auto-gait is on, press G. Confirm it works. Auto re-evaluates after 0.5s.
8. **Reset** — Press R. Confirm assessor resets, gait returns to FLAT.
9. **No rough policy** — Test with `SpotRoughTerrainPolicy` import disabled. Confirm auto-gait is gracefully unavailable.

---

## Estimated Changes

- **~170 new lines** added to `spot_lava_arena.py` (from ~1911 to ~2081 lines)
- **~40 lines refactored** (G key + Xbox RB handlers → use shared helpers)
- **0 changes** to `spot_rough_terrain_policy.py`
