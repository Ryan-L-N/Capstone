# For The Team: How This Project Actually Works

**AI2C Tech Capstone — MS for Autonomy**
*Last updated: February 16, 2026*

If you're reading this, you're probably staring at a folder with a couple thousand lines of Python, three different robots, and wondering where to even start. This document will walk you through the whole thing — the architecture, the decisions, the bugs that nearly broke us, and the lessons that made us better engineers.

Grab some coffee. This is going to be a good read.

---

## The 30-Second Version

We're teaching robots to walk through difficult terrain inside a physics simulator. Three robots, three very different challenges:

1. **Spot** (Boston Dynamics' quadruped) — Navigate through grass, rubble, stairs, and a 100-meter obstacle course
2. **Vision 60** (Ghost Robotics' hexapod) — Stand up and walk from scratch using reinforcement learning
3. **Iris** (Quadcopter) — Fly, search, detect, and intercept a ground target

Everything runs in **NVIDIA Isaac Sim 5.1.0**, a simulator so accurate that policies trained inside it can (theoretically) transfer to real hardware. We use **Isaac Lab 2.3.0** for RL training and **Pegasus Simulator** for the drone.

---

## How the Codebase is Organized

Think of the project like a restaurant. There's a kitchen (shared library), a menu (experiment registry), a front door (launcher), and three dining rooms (experimental designs).

```
MS_for_autonomy/
│
├── launch.py                    ← The front door. Run any experiment from here.
├── experiment_registry.py       ← The menu. Lists all 20+ experiments.
├── open_isaac_sim.py            ← Opens a blank sim (like preheating the oven)
│
├── core/                        ← The kitchen. Shared code everyone uses.
│   ├── sim_app.py               ← Starts the simulator (MUST come first)
│   ├── world_factory.py         ← Creates physics worlds
│   ├── navigation.py            ← Robot navigation (point-to-point)
│   ├── lighting.py              ← Scene lighting presets
│   ├── data_collector.py        ← Records experiment metrics
│   └── markers.py               ← Visual markers (goals, targets)
│
├── experimental_design_grass/   ← Dining room #1: Spot on terrain
├── experimental_design_vision60_alpha/  ← Dining room #2: Vision 60
├── experimental_design_quad_drone/      ← Dining room #3: Iris drone
└── experimental_design_flat_room/       ← Dining room #4: (docs only, future)
```

### The Launcher System

Instead of memorizing file paths, you just run:

```bash
python launch.py grass-teleop      # Drive Spot with WASD + Xbox
python launch.py v60-urdf          # Test Vision 60's URDF
python launch.py drone-p1          # Fly the Iris drone
python launch.py --list            # See all experiments
python launch.py --list grass      # See just Spot experiments
```

The `experiment_registry.py` maps friendly IDs to scripts. Adding a new experiment is one dictionary entry. This might seem like overkill for a capstone project, but when you have 20+ scripts across three robots, not having a central launcher means someone on the team will inevitably run the wrong file and waste 30 minutes waiting for Isaac Sim to boot... only to realize they launched Phase 2 instead of Phase 3.

**Lesson: Build your tool infrastructure early.** The launcher took 30 minutes to write. It's saved hours of confusion.

---

## The Core Library: Why Boilerplate Extraction Matters

Early in the project, every script started with the same 40 lines:

```python
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "width": 1280, ...})
# NOW you can import omni stuff
from omni.isaac.core import World
world = World(physics_dt=1/500, rendering_dt=10/500, ...)
world.scene.add_default_ground_plane()
# ... 20 more lines of setup
```

Copy-paste this into 15 scripts and you've got a maintenance nightmare. Change the physics timestep? Update it in 15 places. Forget one? Enjoy a physics bug that only appears in that one script.

So we extracted it into `core/`. Now every script starts with:

```python
from core import create_simulation, create_world
simulation_app = create_simulation()
world = create_world()
```

Two lines. One source of truth. If you need to change physics settings, you change `WorldConfig` in `world_factory.py` and every experiment gets the update.

**Lesson: The DRY principle ("Don't Repeat Yourself") isn't just about saving keystrokes. It's about ensuring consistency.** When your simulation has a physics timestep of 0.002 seconds, every experiment needs to agree on that number. One script running at 0.001 will produce subtly different results, and you'll spend a week wondering why your data doesn't match.

---

## The Critical Rule That Breaks Everything

Here is the single most important technical fact in this entire project:

> **`SimulationApp` MUST be created BEFORE any `omni.isaac` imports.**

This isn't a suggestion. It's not a best practice. It's a hard requirement. Import `omni.isaac.core` before creating `SimulationApp` and your script crashes with a cryptic error about missing Carbonite plugins. Every team member has hit this at least once.

```python
# WRONG — will crash
from omni.isaac.core import World
from isaacsim import SimulationApp
app = SimulationApp({})  # Too late, omni.isaac already tried to load

# RIGHT — create app first, then import
from isaacsim import SimulationApp
app = SimulationApp({})
from omni.isaac.core import World  # Now safe
```

This is why `core/sim_app.py` exists as a separate module — it ensures the app is created as the very first action. Every other core module imports after it.

**Lesson: When a framework has hard initialization requirements, build guardrails around them.** Don't trust yourself (or your teammates) to remember the correct import order at 2 AM before a deadline. Encode the constraint in the architecture.

---

## Experiment #1: Spot on Grass Terrain

### What We Built

The Spot experiment evolved through five phases, each building on the last:

1. **Phase 1** — Baseline: Spot walks across a flat room. Measures speed, path efficiency, success rate.
2. **Phase 2** — Grass friction: Simulate grass by changing ground friction. Higher grass = lower friction = harder to walk.
3. **Phase 3** — Density: Add more grass patches and obstacles.
4. **Phase 4** — Combined: Everything at once — tall grass, rubble, friction changes.
5. **Phase 5** — RL Training: Train Spot to handle it all using reinforcement learning.

Phase 5 led us down the deepest rabbit hole of the project.

### The Teleop System (spot_teleop.py)

This is our Swiss Army knife — 1,142 lines of battle-tested code that lets you drive Spot around with a keyboard or Xbox controller. It has:

- **4 drive modes**: MANUAL (raw WASD), SMOOTH (velocity ramping), PATROL (waypoint following), AUTO-NAV (fully autonomous)
- **Xbox controller support** with analog stick input, deadzone handling, and button mapping
- **Self-righting mode** — if Spot rolls over, it can flip itself back (inspired by real Boston Dynamics behavior)
- **FPV camera** — first-person view from Spot's perspective
- **Dynamic environment** — switch grass heights (H1-H4) and rubble levels on the fly
- **HUD** — real-time display of speed, heading, position, and drive mode

The velocity smoother is worth calling out:

```python
class VelocitySmoother:
    """Ramps velocity changes instead of snapping to target."""
    def update(self, target, dt):
        diff = target - self.current
        max_change = self.accel * dt
        self.current += np.clip(diff, -max_change, max_change)
        return self.current
```

Without this, the robot jerks violently when you press a key. With it, velocity changes are smooth and gradual — like the difference between slamming the gas pedal and gently accelerating. Small class, huge impact on usability.

### The Obstacle Course (spot_obstacle_course.py)

A 100-meter-long gauntlet with 12 terrain segments:

```
START → Warm-Up Grass → Grass+Stones → Break → STAIRS (0.75m) →
Flat → RUBBLE POOL (-0.5m deep) → Flat → LARGE BLOCKS →
Flat → INSTABILITY FIELD (120 loose bricks) → FINISH
```

This is 1,633 lines of code — the largest file in the project. It creates every obstacle programmatically: 10 stair steps, 40 rubble pieces, 20 navigation blocks, 120 dynamic bricks. The instability field is particularly interesting — 120 small rigid bodies that shift and tumble when Spot steps on them, testing balance and recovery.

One key design decision: **no default ground plane**. Isaac Sim's `add_default_ground_plane()` creates an infinite flat surface at Z=0. But the rubble pool dips to Z=-0.5m. An infinite ground plane would block the descent. So we built custom ground segments — individual cubes for each flat section, overlapping by 0.01m to prevent gaps where Spot could fall through the world.

**Lesson: Default convenience functions (like "add ground plane") are great until your use case exceeds their assumptions.** Know what the defaults do, so you know when to replace them.

### The Rough Terrain Policy Saga

This is the story of a bug that took days to diagnose and ultimately pointed us to a deeper truth about simulation.

**The setup**: We trained a neural network policy (PPO, 235 observations → 12 joint actions) to make Spot walk on rough terrain using Isaac Lab's training pipeline. The policy worked *perfectly* in Isaac Lab's test environment — the robot walked, handled bumps, climbed small obstacles.

Then we deployed it in our obstacle course. The robot collapsed in 1.5 seconds.

**What we checked (and ruled out)**:
- Model weights? ✅ Verified identical outputs between our loader and Isaac Lab's.
- Quaternion convention? ✅ [w,x,y,z] everywhere, gravity projection correct.
- Joint ordering? ✅ Same 12-DOF order in both environments.
- Default positions? ✅ Identical between training and deployment.
- Height scan values? ~~✅ Always 1.0 due to a 20m Z-offset.~~ ❌ **THIS WAS WRONG — see below.**

**What was actually wrong (Round 1 — 5k model)**: The initial 5,000-iteration policy was simply undertrained. Even in Isaac Lab's play.py, the robot barely walked. We'd been debugging deployment when the policy itself was the problem. This led to the 48-hour H100 training run.

**What was actually wrong (Round 2 — 30k model)**: After training completed (30,000 iterations, H100), the properly trained policy STILL fell in standalone deployment. This time, we found the real bug: **the height scan fill value was 1.0, but should have been 0.0.**

Our original source code tracing of Isaac Lab's RayCaster concluded that height_scan = 1.0 (due to a 20m Z-offset, clipped to 1.0). This analysis was wrong. Running the actual training environment and printing raw observations proved it:

```python
# Actual training observations (from Isaac Lab env):
height_scan range: [-0.000002, 0.148083]
height_scan mean:  0.003959
# It's approximately 0.0 on flat ground, NOT 1.0!
```

A parameter sweep of the trained actor showed extreme sensitivity to this value:

```python
# height_scan = 0.0  →  action norm = 3.08  (normal walking)
# height_scan = 0.2  →  action norm = 2.37  (optimal)
# height_scan = 1.0  →  action norm = 7.42  (CATASTROPHIC — instant fall)
```

With `hs=1.0`, the policy produced joint commands 2.4x larger than normal, immediately destabilizing the robot. The one-line fix:

```python
# BEFORE: obs[48:235] = 1.0  ← Robot falls instantly
# AFTER:  obs[48:235] = 0.0  ← Robot walks normally
```

We also switched from CPU to GPU PhysX to match training dynamics, and wrapped the robot API in a `NumpyRobotWrapper` to handle CUDA tensor conversions (GPU PhysX silently ignores numpy arrays).

**The result**: The obstacle course now works. Spot walks in ROUGH gait, responds to WASD/Xbox controls, and can be driven through all 12 terrain segments.

**Lesson 1: Never trust source code tracing alone.** A 2-minute script printing actual observation values would have caught this immediately. Instead, we spent days tracing through `observations.py`, `ray_caster.py`, and `combine_frame_transforms` — and still got it wrong. Runtime truth beats static analysis.

**Lesson 2: When debugging, verify your golden reference first.** For the 5k model, "it works in play.py" was actually "it barely moves in play.py." For the 30k model, the play.py reference was genuine — which correctly pointed us at the deployment wrapper as the bug location.

---

## Experiment #2: Vision 60 Bring-Up

### Why This Was Hard

Spot has a pre-trained walking policy from NVIDIA. Vision 60 has *nothing*. We had to teach it to stand up and walk from scratch. This is like the difference between customizing a car and building one from raw parts.

### The Gravity Bug

Early in Vision 60 development, the robot would launch into the sky at simulation start. After much confusion, we found the cause: **positive gravity**.

```python
# WRONG — robot flies into space
physics_context.set_gravity(9.81)   # Gravity pointing UP

# RIGHT — robot stays on ground
physics_context.set_gravity(-9.81)  # Gravity pointing DOWN
```

This seems obvious in hindsight, but Isaac Sim's API takes a scalar, not a vector. Is positive "downward" (like acceleration due to gravity) or "upward" (like the Z-axis direction)? The answer is: positive = upward, which is the opposite of what most physics textbooks use.

**Lesson: When a robot does something physically impossible (flying, teleporting, passing through walls), the bug is almost always in your coordinate conventions.** Check signs, check axes, check units.

### The Joint Angle Paradox

Vision 60's joints behave counterintuitively: **smaller angles make the robot LOWER, larger angles make it HIGHER**. This is because of how the knee linkage works — extending the knee (larger angle) actually pushes the body up, like straightening your legs from a squat.

```python
# Vision 60 heights
folded  = {"hip": 0.6, "knee": 1.2}   # h ≈ 0.41m (crouching)
standing = {"hip": 0.9, "knee": 1.8}  # h ≈ 0.55m (standing)
```

Three team members independently discovered this the hard way. The third one said "I wish someone had written this down." That's why `lessons_learned.md` exists.

### Training From Scratch

The RL training pipeline for Vision 60 uses a custom environment (not Isaac Lab's built-in tasks) because Vision 60 isn't a standard Isaac Lab robot. The observation space is 188 dimensions — joint positions, velocities, gravity projection, velocity commands, and a height scan grid.

After 32 checkpoints of training, Vision 60 can stand. It can't walk yet. The transition from standing to walking is the hardest part — the policy needs to simultaneously maintain balance AND generate forward motion. It's like learning to ride a bicycle: you have to commit to moving forward to stay upright, but committing to motion feels terrifying when you're barely balanced.

---

## Experiment #3: Iris Quadcopter

### Cascaded PID: Why Simple Controllers Still Matter

In an era of neural network everything, the drone's hover controller is a beautifully simple cascaded PID:

```
Position Error → PID → Desired Velocity → PID → Desired Attitude → PID → Motor Commands
```

Three PID loops, nested inside each other. The outer loop (position) runs slowly and outputs a velocity target. The middle loop (velocity) outputs an attitude target. The inner loop (attitude) outputs motor commands. Each loop runs faster than the one above it.

Results? **±5mm position hold** in a 100m arena. That's 100x better than the ±500mm target. Sometimes the simple approach crushes it.

**Lesson: Don't reach for the neural network until you've tried the PID controller.** RL is powerful but opaque, slow to train, and hard to debug. PIDs are transparent, fast to tune, and predictable. Use RL when PID genuinely can't solve the problem (like rough terrain locomotion), not because it's trendy.

### The Upside-Down Camera

The drone's camera was pointing at the sky. For a target detection system. The camera config specified `pitch = 90.0` degrees, which seemed correct for "pointing down." Except Pegasus uses the convention that **negative pitch = down**.

```python
# Points at the sky
camera_pitch = 90.0

# Points at the ground
camera_pitch = -90.0
```

We diagnosed this by adding debug image saves. The camera was capturing gorgeous images of clouds and horizon. Not exactly useful for detecting ground targets.

| Pitch | What the Camera Sees |
|-------|---------------------|
| +90°  | Blue sky, clouds    |
| +20°  | Horizon, buildings  |
| -90°  | Ground, target      |

**Lesson: When a sensor gives bad data, visualize its raw output before assuming the processing pipeline is wrong.** If we'd looked at the camera images first, we'd have found this in 5 minutes instead of 2 hours.

---

## The H100 Training Run: Industrial-Scale RL

### Why We Needed It

The laptop training (5,000 iterations, ~45 minutes on RTX 4090) produced a policy that couldn't walk. The H100 training (30,000 iterations, ~53 hours) produced one that can.

The math is simple: more iterations = more experience. At 8,192 parallel environments and 24 steps per iteration, the policy processes **196,608 timesteps per iteration**. Over 30,000 iterations, that's **5.9 billion timesteps** of walking experience. The laptop run only saw 491 million.

### Reward Engineering

Training a robot to walk is really about designing the right incentive structure. Here's what we incentivize and penalize:

**Positive rewards (do more of this):**
| Reward | Weight | Why |
|--------|--------|-----|
| Trot gait pattern | +10.0 | Enforce diagonal foot pair timing (like a horse's trot) |
| Forward velocity tracking | +7.0 | Go where commanded |
| Yaw velocity tracking | +5.0 | Turn when commanded |
| Air time (foot swing) | +5.0 | Lift feet properly, don't drag |
| Foot clearance | +2.5 | Step high enough to clear obstacles |

**Penalties (do less of this):**
| Penalty | Weight | Why |
|---------|--------|-----|
| Base orientation deviation | -5.0 | Stay upright, don't tilt |
| Base vertical/lateral motion | -3.0 | Don't bounce or sway |
| Action smoothness | -2.0 | Don't jerk the joints |
| Joint torque | -0.002 | Don't waste energy |
| Joint position deviation | -1.0 | Stay near default stance |
| Foot slip | -1.0 | Don't slide feet on ground |

This is reward engineering, and it's more art than science. The gait reward at +10.0 is the strongest signal — it basically says "above all else, maintain a proper trot." Without it, the robot might discover that hopping on two legs is technically faster, but completely impractical for real deployment.

The penalties are where the training philosophy lives. A weight of -0.002 on joint torques barely matters in early training when the robot is just trying not to fall. But by iteration 20,000, when walking is easy, those small penalties shape *how* the robot walks — efficiently, smoothly, with minimal energy waste. It's like a music student: first learn to play the notes (stay upright), then learn to play them beautifully (smooth, efficient gait).

### The 235-Dimension Surprise

One thing you learn quickly in this project: **verify your assumptions with actual output**. We assumed the observation space was 208 dimensions (48 proprioceptive + 160 height scan). The debug run revealed it's actually **235 dimensions** — the height scan grid is 17x11 = 187 points, not 16x10 = 160.

Why? The `GridPattern(resolution=0.1, size=[1.6, 1.0])` generates points from -0.8 to +0.8 at 0.1m intervals. That's 17 points (including both endpoints), not 16 (excluding one). A classic off-by-one error in our mental model.

If we'd deployed the policy with a 208-dimension observation vector, indices 160-207 would contain garbage data, and the policy would behave unpredictably. The 10-iteration debug run caught this before we wasted 48 hours training a model we couldn't deploy.

**Lesson: Run a small test before the big run. Always.** The 10-iteration debug took 2 minutes. It caught a bug that would have invalidated a 53-hour training run.

### Windows → Linux: The CRLF Trap

We wrote shell scripts on Windows. Windows uses `\r\n` (carriage return + newline) for line endings. Linux uses `\n`. When we uploaded the scripts to the H100 server, bash saw `\r` as part of each command:

```
$ bash train.sh
/home/t2user/train.sh: line 11: $'\r': command not found
```

Fix: `sed -i "s/\r$//" *.sh`

This is a classic cross-platform bug that every engineer hits at least once. Now you know about it before you hit it.

---

## How the Technologies Connect

Here's how everything fits together:

```
┌─────────────────────────────────────────────────────────────────┐
│                    NVIDIA Isaac Sim 5.1.0                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  GPU PhysX    │  │  RTX Renderer│  │  USD Stage (scene)   │  │
│  │  500Hz physics│  │  50Hz render │  │  Prims, materials,   │  │
│  │  rigid bodies │  │  ray tracing │  │  robots, terrain     │  │
│  └──────┬───────┘  └──────────────┘  └──────────┬───────────┘  │
│         │                                        │              │
│  ┌──────┴────────────────────────────────────────┴───────────┐  │
│  │                   Python API Layer                         │  │
│  │  SimulationApp → World → Scene → Robot (ArticulationView) │  │
│  └──────┬───────────────────────────────────────────────────┘  │
└─────────┼───────────────────────────────────────────────────────┘
          │
    ┌─────┴──────────────────────────────────────────┐
    │              Our Code Layer                     │
    │                                                 │
    │  ┌─────────┐  ┌───────────┐  ┌──────────────┐  │
    │  │ core/   │  │ Teleop    │  │ RL Training  │  │
    │  │ library │  │ (WASD/Xbox│  │ (RSL-RL PPO) │  │
    │  │         │  │  control) │  │              │  │
    │  └─────────┘  └───────────┘  └──────┬───────┘  │
    │                                      │          │
    │  ┌───────────────────────────────────┴───────┐  │
    │  │         Isaac Lab 2.3.0                   │  │
    │  │  ManagerBasedRLEnv → Rewards → Curriculum │  │
    │  │  Observations → Actions → Terminations    │  │
    │  └───────────────────────────────────────────┘  │
    │                                                 │
    │  ┌───────────────────────────────────────────┐  │
    │  │       Pegasus Simulator (Drone only)      │  │
    │  │  Vehicle → Backend → Motor Mixing → PID   │  │
    │  └───────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────┘
```

**Isaac Sim** is the foundation — it runs the physics (GPU PhysX at 500Hz) and renders the scene. **Isaac Lab** sits on top and provides the RL training infrastructure — it manages observations, rewards, terminations, and curriculum progression. **Pegasus** is a separate layer for aerial vehicles with its own control pipeline.

Our code interfaces with all three, plus provides the teleop layer, the experiment management, and the custom environments (obstacle course, grass terrain, etc.).

---

## Best Practices We Learned the Hard Way

### 1. Physics Callbacks, Not Render Loops

Physics runs at 500Hz. Rendering runs at 50Hz. If you put your robot control in the render loop, you're only commanding the robot 50 times per second while physics is simulating 500 steps. The robot will be uncontrolled for 90% of the simulation.

```python
# WRONG — controls at 50Hz, physics at 500Hz
while simulation_app.is_running():
    world.step(render=True)  # Only runs at render rate
    robot.set_velocity(...)  # Misses 9 out of 10 physics steps

# RIGHT — controls at 500Hz via callback
def on_physics_step(step_size):
    robot.set_velocity(...)  # Runs every physics step

world.add_physics_callback("control", on_physics_step)
```

### 2. Quaternions: Know Your Convention

Isaac Sim uses [w, x, y, z] (scalar-first). Pegasus/SciPy uses [x, y, z, w] (scalar-last). Mix them up and your robot will face backwards, rotate wrong, or compute gravity in the wrong direction.

We have a utility function `quat_to_yaw()` in `core/navigation.py` that handles this. Use it. Don't write your own unless you enjoy debugging rotation math.

### 3. Save and Restore State During Mode Switches

When switching between drive modes or gait policies, save the current PhysX state first:

```python
# Before switch
saved_gains = av.get_gains()
saved_solver = av.get_solver_position_iteration_counts()

# Switch to new mode...

# If it fails, restore
av.set_gains(*saved_gains)
av.set_solver_position_iteration_counts(saved_solver)
```

We learned this when switching from the flat terrain policy to the rough terrain policy. Without save/restore, a failed switch leaves the robot in a broken intermediate state.

### 4. Debug Runs Before Real Runs

Always run 10 iterations before 30,000. Always test with 4,096 envs before 8,192. Always verify on localhost before uploading to the server.

The 10-iteration debug run on the H100 caught: CRLF line endings, missing config files, conda initialization issues, EULA prompts, and a 235-vs-208 observation dimension mismatch. Total time: 2 minutes. Problems it prevented: wasting 53 hours on a broken pipeline.

### 5. Document Your Bugs

Every experimental design folder has a `lessons_learned.md`. These aren't optional documentation — they're the institutional memory of the project. When someone says "I think the camera might be upside down," you can search lessons_learned.md for "camera" and find Lesson 7 in the drone folder: "Pegasus uses NEGATIVE pitch to point down."

The most valuable documentation isn't API references or architecture diagrams. It's "here's a thing that broke and here's exactly how we fixed it."

---

## The Numbers That Matter

### Spot Obstacle Course
- **Course length**: 100m with 12 terrain segments
- **Total prims**: ~246 objects (160 dynamic rigid bodies)
- **Steps**: 0.75m peak (5 up, 5 down)
- **Rubble pool**: 0.5m deep with 40 debris pieces
- **Instability field**: 120 small dynamic bricks

### RL Training (H100) — COMPLETE
- **Observation space**: 235 dimensions (48 proprioceptive + 187 height scan)
- **Action space**: 12 dimensions (joint position offsets)
- **Network**: 235 → 512 → 256 → 128 → 12 (ELU, ~350K parameters)
- **Training environments**: 8,192 parallel simulations
- **Throughput**: 30,000 steps/second
- **Total experience**: 5.9 billion timesteps
- **Training time**: ~53 hours on H100 NVL
- **Final reward**: +143.74 (from -0.90 at start)
- **Episode length**: 573 steps / 11.5 seconds (from 20 steps / 0.4s)
- **Terrain level**: 4.42 (curriculum reached hard terrain)
- **Gait quality**: 5.28 (88x improvement from start)
- **Deployed**: Successfully in obstacle course with WASD + Xbox teleop

### Drone
- **Position hold accuracy**: ±5mm (100x better than ±500mm target)
- **Detection rate**: 100% directly above target
- **Detection confidence**: 0.89 at optimal position

### Vision 60
- **DOF**: 12 joints (3 per leg × 4 legs)
- **Observation space**: 188 dimensions
- **Current status**: Can stand, learning to walk

---

## Things That Will Bite You (So They Don't)

| Pitfall | What Happens | How to Avoid |
|---------|--------------|--------------|
| Import `omni.isaac` before `SimulationApp` | Crash with Carbonite error | Always use `core/sim_app.py` |
| Use numpy arrays with GPU PhysX | Silently ignored (no error!) | Convert to CUDA tensors or use NumpyRobotWrapper |
| Height scan fill value = 1.0 | Policy outputs catastrophic actions (norm 7.4) | Use 0.0 for flat ground — always verify with actual training obs |
| Positive gravity in Isaac Sim | Robot launches into space | Always use negative values |
| Windows CRLF in Linux shell scripts | `$'\r': command not found` | `sed -i "s/\r$//"` after upload |
| Camera pitch = +90° in Pegasus | Camera points at sky | Use -90° for downward |
| `conda activate` in SSH command | `conda: command not found` | Source conda hook explicitly |
| `Gf.Quatd` for orientation ops | Silent type mismatch | Always use `Gf.Quatf` |
| Robot control in render loop | Jerky, unresponsive motion | Use `add_physics_callback()` |
| Default ground plane with rubble pool | Robot can't descend below Z=0 | Build custom ground segments |
| vy ≠ 0 in `spot.forward()` | Undefined behavior | Always pass `[vx, 0.0, wz]` |
| OneDrive syncing `.git/worktrees/` | Permission denied on git ops | `git worktree prune` (harmless) |

---

## How Good Engineers Think

If there's one meta-lesson from this project, it's this: **good engineering is about managing uncertainty systematically**.

We didn't know if the rough terrain policy would work when deployed. So we built a test (play_rough_teleop.py) that isolated the policy in its training environment. When it barely worked there, we knew the problem was training, not deployment.

We didn't know if the H100 could handle 8,192 environments. So we ran a stress test with increasing env counts and measured throughput, temperature, and physics stability at each level.

We didn't know if our reward weights would produce a good gait. So we ran 10 iterations, verified the reward terms were active and correctly weighted, and confirmed the training metrics were trending in the right direction before committing to 30,000 iterations.

At every point of uncertainty, the pattern is the same:
1. **Identify what you don't know**
2. **Design the smallest possible test that answers the question**
3. **Run the test before committing to the expensive path**

This is the difference between engineering and hoping.

---

## File Quick Reference

| What You Want To Do | File to Look At |
|---------------------|-----------------|
| Drive Spot around | `experimental_design_grass/code/spot_teleop.py` |
| Run the obstacle course | `experimental_design_grass/code/spot_obstacle_course.py` |
| Rough terrain policy wrapper | `experimental_design_grass/code/spot_rough_terrain_policy.py` |
| Understand RL training | `experimental_design_grass/48h_training/TRAINING_PLAN.md` |
| See the training script | `experimental_design_grass/48h_training/spot_rough_48h_cfg.py` |
| Debug rough policy deployment | `experimental_design_grass/ROUGH_POLICY_DEBUG_HANDOFF.md` |
| Test Vision 60 | `experimental_design_vision60_alpha/code/vision60_working.py` |
| Fly the drone | `experimental_design_quad_drone/code/phase_1_stable_flight.py` |
| Add a new experiment | `experiment_registry.py` (add entry) + `launch.py` (auto-discovers) |
| Change physics settings | `core/world_factory.py` |
| Debug a robot bug | Check `lessons_learned.md` in the relevant experiment folder |
| Connect to H100 | `experimental_design_grass/Isaac_on_H-100.md` |
| Package for ARL delivery | `ARL_DELIVERY/` folder at project root |

---

## Final Thought

This project has ~7,000 lines of Python across 40+ files, three robot platforms, two RL frameworks, one drone simulator, a 30k-iteration H100 training run, and a partridge in a pear tree. It can feel overwhelming.

But strip away the complexity and it's fundamentally simple: we're building a world, putting a robot in it, and teaching it to move. Everything else — the reward engineering, the tensor debugging, the SSH config headaches — is just the cost of making that simple idea work in practice.

The gap between "simple idea" and "working implementation" is where engineering lives. And if this document helps you cross that gap a little faster, it's done its job.

Good luck. And remember: check your quaternion convention.

— The Team, February 2026
