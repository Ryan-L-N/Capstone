# Spot Teleoperation Script for Grass Experiments

## Context

The grass experiment has completed Phases 1-4 using autonomous NavigationController. Now we need a hands-on way to **manually drive Spot through grass terrain** with WASD keyboard controls, switch between drive modes with Shift, and feel like you're operating a real robot. This gives direct, interactive testing of how Spot handles different grass heights.

The only working locomotion policy is `SpotFlatTerrainPolicy` (built-in to Isaac Sim). "Policy switching" is implemented as **drive modes** -- different velocity profiles that change how WASD maps to robot commands.

---

## Controls

```
MOVEMENT:
  W / S         Forward / Backward
  A / D         Turn left / Turn right
  SPACE         Emergency stop (zero velocity)

MODE:
  SHIFT         Cycle drive mode: MANUAL -> SMOOTH -> PATROL -> AUTO-NAV -> ...

TERRAIN:
  0             No grass (baseline, friction 0.80)
  1             H1 grass (0.1m, friction 0.80)
  2             H2 grass (0.3m, friction 0.85)
  3             H3 grass (0.5m, friction 0.90)
  4             H4 grass (0.7m, friction 0.95)

RUBBLE:
  H             Cycle rubble level: CLEAR -> LIGHT -> MODERATE -> HEAVY

CAMERA:
  M             Toggle FPV camera (first-person onboard view)

SPEED:
  UP / DOWN     Increase / decrease speed multiplier (+/-0.1)

RECOVERY:
  X             Toggle selfright mode (physics-based rollover recovery)
                  In selfright mode: A/D = roll left/right
                  Auto-exits when robot is upright for 0.3s

OTHER:
  R             Reset robot to start position
  ESC           Exit simulation
```

---

## Rubble System (H key to cycle)

Press H to cycle through rubble levels. Rubble pieces are rigid bodies with physics (mass, collision, friction) scattered across the grass zone. Spot must push through or navigate around them.

| Level | Pieces | Composition | Description |
|-------|--------|-------------|-------------|
| **CLEAR** | 0 | -- | No rubble |
| **LIGHT** | ~13 | Mostly small bricks + few blocks | Scattered debris |
| **MODERATE** | ~28 | Mixed all sizes | Cluttered terrain |
| **HEAVY** | ~50 | Full range incl. 2x2ft crates | Dense rubble field |

**Piece types:**
- Small bricks (0.1m, ~1 kg) - tan/brown
- Medium blocks (0.2m, ~5-7 kg) - gray
- Large slabs (0.3-0.4m, ~10-12 kg) - dark gray
- Big crates (0.6m/2x2ft, ~23 kg/50 lbs) - dark brown

---

## Drive Modes (Shift to cycle)

All modes use the same `SpotFlatTerrainPolicy.forward()` underneath. The difference is how WASD input maps to velocity commands:

| Mode | Max vx | Max wz | Smoothing | Feel |
|------|--------|--------|-----------|------|
| **MANUAL** | 1.5 m/s | 1.0 rad/s | None (instant) | Arcade / snappy |
| **SMOOTH** | 1.2 m/s | 0.8 rad/s | Accel 1.5, Decel 3.0 | Real robot inertia |
| **PATROL** | 0.6 m/s | 0.5 rad/s | Accel 1.0, Decel 2.5 | Slow & careful (tall grass) |
| **AUTO-NAV** | 1.0 m/s | 0.8 rad/s | Accel 1.2, Decel 3.0 | Autonomous + WASD override |

- **MANUAL**: Direct control, instant velocity response, allows turning from standstill
- **SMOOTH**: Velocity ramps up/down exponentially (like a real robot with inertia). Decel is 2x faster than accel for safety. Enforces ES-003 minimum forward speed while turning
- **PATROL**: Lower speeds optimized for H3/H4 grass. Slowest acceleration for most careful movement
- **AUTO-NAV**: NavigationController drives to room waypoints autonomously. Any WASD key pressed = manual override. Release WASD = resume auto-nav. Waypoints cycle through room corners

---

## Architecture

### VelocitySmoother class
Ramps `current_vx` and `current_wz` toward target velocities per physics step. Uses different rates for acceleration vs deceleration. Called at 500Hz in the physics callback.

### DriveController class
Holds current mode, VelocitySmoother, and optional NavigationController (for AUTO-NAV). Main method: `compute_command(key_state, sim_time, position, yaw)` returns `[vx, 0.0, wz]`.

### Environment setup functions
Copied from `phase_2_friction_grass.py`:
- `create_room(stage)` -- 18.3m x 9.1m x 3.0m room with walls
- `create_room_lighting(stage)` -- dome + distant light
- `create_grass_material(stage, friction)` -- PhysX material with friction combine mode
- `create_grass_zone_visual(stage, zone, height_key)` -- green floor overlay
- `switch_grass(height_key)` -- runtime grass switching (updates friction + visual color)

### Keyboard handler
Uses `carb.input` -- identical pattern from `Cole_working_sim/spot_walk_keyboard_control.py`.

### Physics callback (500Hz)
Follows ES-010B pattern with `physics_ready` flag. Reads `key_state` dict, calls `drive_controller.compute_command()`, sends result to `spot.forward(step_size, command)`.

### Selfright Mode (X key to toggle, A/D to roll)
Physics-based rollover recovery inspired by real Spot's `selfright` command from the [Boston Dynamics SDK](https://dev.bostondynamics.com/python/bosdyn-client/src/bosdyn/client/robot_command.html). Since `SpotFlatTerrainPolicy` has no built-in self-righting motor sequence, recovery uses direct angular velocity control on the robot's rigid body.

**How it works:**
1. Press **X** to enter SELFRIGHT mode (locomotion policy disabled, `spot.forward()` not called)
2. Press **A** to roll left or **D** to roll right — applies phase-dependent angular torque around the body's forward axis. Release A/D and ground friction naturally damps the roll.
3. When the robot's roll AND pitch are both below 35 degrees for 0.3 seconds continuously, selfright auto-exits and enters a 1.5-second stabilization period (zero velocity commands to let the policy re-settle).
4. Press **X** again at any time to cancel selfright mode.

**Sim-to-real physics model — phase-dependent forces simulating leg ground-reaction:**

| Roll angle | Phase | Torque gain | Upward lift | What it simulates |
|-----------|-------|-------------|-------------|-------------------|
| >120 deg | PUSH | 100% (12 rad/s²) | Full | Legs fully extended, pushing hard off ground |
| 60-120 deg | ASSIST | 60% | Partial | Gravity starting to help, less leg force needed |
| 30-60 deg | GUIDE | 25% | None | Past tipping point, gravity does the work |
| <30 deg | SETTLE | 10% | None | Gentle correction, prevent over-rotation |

When A/D is released, angular velocity decays naturally via damping coefficient 3.0 (simulating ground friction and gravity opposing free rotation). Upward ground-reaction force only applies when roll >45 degrees and scales linearly — simulating the vertical component of legs pushing against the floor. Near upright, no upward force (legs are properly supporting).

**Constants:** Peak torque 12.0 rad/s², max roll velocity 2.5 rad/s (deliberate ~3s roll), ground lift 0.8 m/s max (phase-scaled), damping 3.0.

The HUD shows `SELFRIGHT(R:±deg P:±deg) A/D=roll` during selfright mode, or `ROLLED! (X=selfright)` when rollover is detected but selfright is not yet active.

### FPV Camera System (M key to toggle)
A `UsdGeom.Camera` attached to `/World/Spot/body/fpv_camera` that auto-follows the robot body. Camera is positioned at front of Spot's head (0.4m forward, 0.15m up) with wide 18mm focal length for immersive FOV. The USD camera default look direction (-Z) is rotated via quaternion `(0.5, 0.5, -0.5, -0.5)` to face +X (forward) with +Z up. Viewport switching uses `omni.kit.viewport.utility.get_active_viewport()` to swap between the default orbit camera and the FPV camera.

### Terminal HUD
Overwrites same terminal line every 0.5s showing: sim time, drive mode, camera mode (FPV/ORB), grass height, position, yaw, vx, wz, distance from start.

---

## Critical patterns

| Pattern | Source | Why |
|---------|--------|-----|
| `SimulationApp` before all imports | Every script | Crashes otherwise |
| `physics_ready = [False]` skip first callback | ES-010B | Robot flip bug |
| `vy = 0` always | ES-004 | SpotFlatTerrainPolicy constraint |
| Quaternion `[w,x,y,z]` | ES-002 | Isaac Sim 5.1.0 format |
| Min 50% forward speed while turning | ES-003 | Quadrupeds need momentum |
| Dead zone 0.1 rad, turn gain 0.8 | ES-005 | Prevents oscillation |
| Physics at 500Hz, render at 50Hz | All scripts | Robot stability |
