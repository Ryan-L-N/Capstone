# FORTHERAWDOGS.md

## Teaching a Robot Dog to Walk Through Hell — A Technical Deep-Dive

> *"In theory, there is no difference between theory and practice. In practice, there is."*
> — Yogi Berra (also every robotics engineer ever)

This document explains the entire AI2C Tech Capstone project — from the big picture down to the gnarliest bugs we squashed. If you're reading this, you're either a teammate trying to understand the codebase, a future student inheriting this work, or someone who just saw "FORTHERAWDOGS" in a git log and got curious. Either way, welcome.

---

## Table of Contents

1. [The 30-Second Pitch](#the-30-second-pitch)
2. [Why This Matters (Beyond the Grade)](#why-this-matters)
3. [The Tech Stack — What We're Standing On](#the-tech-stack)
4. [How a Robot Learns to Walk (RL Crash Course)](#how-a-robot-learns-to-walk)
5. [The Codebase — A Map of the Territory](#the-codebase)
6. [The Training Pipeline — Building the Brain](#the-training-pipeline)
7. [The Evaluation Pipeline — The Obstacle Course](#the-evaluation-pipeline)
8. [The Numbers That Matter](#the-numbers-that-matter)
9. [Bugs, Disasters, and Hard-Won Lessons](#bugs-disasters-and-hard-won-lessons)
10. [How Good Engineers Think](#how-good-engineers-think)
11. [What I'd Do Differently](#what-id-do-differently)

---

## The 30-Second Pitch

We trained a simulated Boston Dynamics Spot robot to walk across ice, through tall grass, over boulders, and up staircases — all using reinforcement learning in NVIDIA's Isaac Sim physics simulator. The robot sees the world through 235 numbers (its body state + a height map of the ground ahead), decides how to move 12 joints 50 times per second, and learns entirely through trial and error across billions of attempts.

Think of it like teaching a toddler to walk, except the toddler has four legs, the floor keeps changing material, and you have 20,480 toddlers learning simultaneously on an NVIDIA H100 GPU.

---

## Why This Matters

Every military base, disaster site, and construction zone has terrain that wheeled robots can't handle. Boston Dynamics Spot can physically handle these environments, but its stock locomotion is conservative — it slows down or refuses to traverse surfaces it hasn't been programmed for.

Our project trains a single neural network policy that handles *all* of these terrains:

| Terrain | Real-World Analog | Why It's Hard |
|---------|-------------------|---------------|
| **Low friction** | Ice, wet metal, oily floors | Robot's feet slide — can't push off |
| **Vegetation** | Tall grass, brush, mud | Drag forces resist movement |
| **Boulders** | Rubble fields, rocky ground | Uneven footing, trip hazards |
| **Stairs** | Building access, loading docks | Precise stepping, elevation change |

The goal: one policy to rule them all, trained in simulation, deployable to real hardware.

---

## The Tech Stack

Here's everything we're standing on, and why we chose it:

### NVIDIA Isaac Sim + Isaac Lab
**What it is:** A physics simulator built on NVIDIA Omniverse that can simulate thousands of robots simultaneously on a GPU.

**Why we chose it:** Isaac Lab (the RL framework built on top of Isaac Sim) can run 20,480 parallel environments on a single H100 GPU. That means our robot gets 20,480x more practice per second compared to training on a single environment. A 100-hour training run generates ~39 billion timesteps — that's roughly 1,200 years of robot walking experience compressed into four days.

**The catch:** Isaac Sim is powerful but opinionated. It has strict rules about initialization order (more on this in the bugs section), and the GPU PhysX engine has behaviors that don't always match the documentation. We fought it a lot. We won most of those fights.

### RSL-RL (PPO Implementation)
**What it is:** A lean PyTorch implementation of Proximal Policy Optimization, built by the Robotic Systems Lab at ETH Zurich.

**Why we chose it:** It's what Isaac Lab ships with, it's battle-tested on real quadrupeds (ANYmal), and it's simple enough to modify. We monkey-patched its learning rate scheduler to add cosine annealing, which took about 20 lines of code.

### Python 3.11 + PyTorch 2.7 + CUDA 12.8
The standard ML stack. Python for glue, PyTorch for neural networks, CUDA for GPU acceleration. Nothing exotic here — we wanted reliability over novelty.

### H100 NVL 96GB (Training) + RTX 2000 Ada 8GB (Development)
We developed and debugged locally on a laptop GPU (8GB), then deployed to an H100 server (96GB) for real training. This "develop small, train big" workflow saved us from burning expensive GPU hours on bugs.

**Key insight:** Always test with 1-4 environments locally before launching 20,480 on the server. Every bug that crashes 1 environment will crash 20,480 environments and waste your time.

---

## How a Robot Learns to Walk

If you've never worked with reinforcement learning, here's the mental model:

### The Game Loop

Imagine a video game where the player is the robot and the score is a reward function. Every 0.02 seconds (50Hz), the game:

1. **Shows the player the screen** (235 numbers describing body state + terrain ahead)
2. **Player presses buttons** (12 joint position commands)
3. **Physics happens** (10 tiny physics steps at 500Hz simulate what those commands do)
4. **Score updates** (+points for walking forward, -points for falling, jerky movement, etc.)

The neural network IS the player. At first it mashes buttons randomly. After billions of attempts, it learns to walk smoothly.

### The Reward Function — Teaching Without Words

This is the most critical design decision in the whole project. You can't tell a neural network "walk nicely" — you have to express that numerically. Our reward function has 19 terms:

**The Carrots (positive rewards):**
- `gait` (+10.0): Reward for maintaining a trot pattern (diagonal legs move together)
- `base_linear_velocity` (+7.0): Reward for moving at the commanded speed
- `base_angular_velocity` (+5.0): Reward for turning at the commanded rate
- `foot_clearance` (+3.5): Reward for lifting feet 10cm off the ground (clears obstacles)
- `air_time` (+3.0): Reward for proper swing phase timing
- `velocity_modulation` (+2.0): Accept slower speeds on harder terrain

**The Sticks (penalties):**
- `foot_slip` (-3.0): Heavy penalty for feet sliding on the ground
- `base_motion` (-4.0): Penalty for bouncing and swaying
- `stumble` (-2.0): Penalty for tripping over obstacles
- `action_smoothness` (-2.0): Penalty for jerky joint commands
- `base_orientation` (-5.0): Penalty for tilting (stay upright!)
- `body_height_tracking` (-2.0): Penalty for crouching or standing too tall

**The Analogy:** Think of reward shaping like raising a kid. You can't just say "be good" — you have to specifically praise sharing (gait reward) and specifically call out hitting (foot slip penalty). If your rewards are vague, the robot finds loopholes. If they're too specific, it becomes brittle.

One real example: in early training, we rewarded `air_time` too heavily (+5.0). The robot learned to *bounce* on low-friction surfaces — feet in the air means no slipping! We dropped it to +3.0 and tripled the `foot_slip` penalty to fix this.

### The Terrain Curriculum — Difficulty Ramp

Nobody throws a toddler onto an ice rink on day one. Our terrain generator creates a 10x40 grid of terrain patches, where rows represent difficulty levels:

- **Row 0 (easy):** Tiny stairs, gentle slopes, mild bumps
- **Row 5 (medium):** Standard stairs, moderate obstacles
- **Row 9 (hell):** 25cm stairs, 50cm gaps, violent terrain

Robots that walk well get promoted to harder rows. Robots that fall get sent back. This is called **curriculum learning**, and it's stolen directly from how video games teach you to play — easy levels first, boss fights later.

---

## The Codebase

```
Capstone/Experiments/Alex/
├── 4_env_test/              # Evaluation pipeline (Phase 4)
│   ├── src/
│   │   ├── run_capstone_eval.py          # Main episode runner
│   │   ├── spot_rough_terrain_policy.py  # Policy wrapper (559 lines)
│   │   ├── configs/
│   │   │   ├── eval_cfg.py               # Physics constants, thresholds
│   │   │   └── zone_params.py            # 4 environments x 5 zones each
│   │   ├── envs/
│   │   │   ├── base_arena.py             # Shared arena setup
│   │   │   ├── friction_env.py           # 5 friction zones (sandpaper → oil)
│   │   │   ├── grass_env.py              # 5 drag zones (air → dense brush)
│   │   │   ├── boulder_env.py            # Polyhedra obstacles (D8/D10/D12/D20)
│   │   │   └── stairs_env.py             # 5 step-height zones (3cm → 23cm)
│   │   ├── metrics/
│   │   │   ├── collector.py              # Per-step data accumulator
│   │   │   └── reporter.py               # Stats, t-tests, plots
│   │   └── navigation/
│   │       └── waypoint_follower.py      # Velocity commands toward goal
│   ├── scripts/                          # Shell scripts for batch runs
│   ├── checkpoints/                      # model_29999.pt lives here
│   └── results/                          # JSONL episodes + CSV reports + plots
│
├── 100hr_env_run/           # Training pipeline (Phase 5)
│   ├── train_100hr.py                    # Main training script with cosine LR
│   ├── play_100hr.py                     # Visualization/export script
│   ├── configs/
│   │   ├── terrain_cfg.py                # 12 terrain types, 400 patches
│   │   ├── env_cfg.py                    # Full environment config (577 lines)
│   │   └── ppo_cfg.py                    # PPO hyperparameters
│   ├── rewards/
│   │   └── reward_terms.py               # VegetationDrag + 4 custom rewards
│   └── scripts/
│       ├── train_h100.sh                 # Full 100hr launch (screen + TB)
│       ├── test_h100.sh                  # 1000-iter validation run
│       └── train_local_debug.sh          # Local sanity check
│
└── FORTHERAWDOGS.md         # You are here
```

### How the Pieces Connect

**Training flow:**
```
terrain_cfg.py (defines 12 terrain types)
       ↓
env_cfg.py (wires terrain + rewards + domain randomization + observations)
       ↓
ppo_cfg.py (sets network architecture + optimizer + rollout length)
       ↓
train_100hr.py (creates environment, creates runner, monkey-patches LR, calls learn())
       ↓
RSL-RL OnPolicyRunner.learn() runs the training loop
       ↓
Saves model_XXXXX.pt checkpoints every 1000 iterations
```

**Evaluation flow:**
```
run_capstone_eval.py (parses args, creates SimulationApp)
       ↓
envs/__init__.py → build_environment("friction") → friction_env.py
       ↓
spot_rough_terrain_policy.py loads model_29999.pt, wraps flat policy
       ↓
For each episode:
  waypoint_follower.py generates velocity commands
  policy.forward(dt, command) → 12 joint actions
  collector.py records every step
       ↓
After all episodes:
  reporter.py generates summary.csv + statistical_tests.csv + plots/
```

**The key insight:** Training and evaluation are *completely separate systems*. Training uses Isaac Lab's manager-based RL (highly parallel, 20K+ envs, automatic reward/observation/terrain management). Evaluation uses standalone Isaac Sim (one robot, one episode at a time, manual physics stepping). This separation is intentional — evaluation needs precise control over individual environments that the training framework abstracts away.

---

## The Training Pipeline

### The Network: A Brain in Three Layers

```
Input:  235 numbers (what the robot "sees")
         ↓
Layer 1: 1024 neurons + ELU activation ("what patterns exist?")
         ↓
Layer 2: 512 neurons + ELU activation ("what do the patterns mean?")
         ↓
Layer 3: 256 neurons + ELU activation ("what should I do?")
         ↓
Output: 12 numbers (how much to move each joint)
```

Total: ~2 million trainable parameters. That's tiny by LLM standards (GPT-4 has ~1.8 trillion), but for a locomotion policy, it's actually quite large. The previous 48-hour policy used [512, 256, 128] (~500K params). We scaled up 4x because the 12-terrain environment is much more complex.

**Why MLP and not a transformer/CNN?** Locomotion policies need to run at 50Hz on embedded hardware. An MLP with 2M params takes microseconds to evaluate. A transformer would be overkill for a 235-dim input — there's no sequential or spatial structure to exploit.

### Domain Randomization — Prepare for Everything

Here's a dirty secret about sim-to-real transfer: the simulation is always wrong. The real world has air resistance, actuator backlash, sensor noise, and a thousand other things your simulator doesn't model perfectly. The solution? **Make the simulation wrong in random directions** so the policy learns to be robust to *any* wrongness.

Every episode, we randomly change:

| What We Randomize | Range | Real-World Equivalent |
|---|---|---|
| Surface friction | 0.05 — 1.5 | Oil on steel → sandpaper |
| Robot mass | ±8 kg | Carrying different payloads |
| Push disturbances | ±1.5 m/s | Getting bumped by wind/people |
| Sustained force | ±8 N | Dragging a cable, uphill wind |
| Joint position noise | ±0.2 rad | Encoder drift |
| Joint velocity noise | ±3.0 rad/s | Noisy IMU readings |
| PD gains (Kp, Kd) | ±20% | Actuator model uncertainty |

The idea: if you can walk on ice while carrying 8 extra kilos and being pushed sideways, you can probably walk on a normal sidewalk just fine.

### The Learning Rate Schedule — Fast Then Careful

Imagine learning to drive. At first, you make big corrections (swerving, slamming brakes). As you get better, your corrections get smaller and more precise. Our learning rate does the same thing:

```
Iterations 0-3000:     Warmup (1e-5 → 1e-3, linear ramp)
Iterations 3000-57000: Cosine decay (1e-3 → 1e-5)
Iterations 57000-60000: Hold at 1e-5 (fine-tuning)
```

The warmup prevents early catastrophic gradient updates (the network is random at first — big learning rates cause explosions). The cosine decay gives a smooth transition from "explore broadly" to "refine what works."

### Terrain-Aware VegetationDragReward — The Trickiest Piece

This reward term does double duty — it's both a **physics modifier** and a **reward signal**:

1. **Physics:** Applies velocity-dependent drag forces to robot feet (F = -c * v), simulating grass/brush resistance
2. **Reward:** Penalizes the robot proportionally to drag magnitude (teaches it to push through)

The clever part: it's **terrain-aware**. When `curriculum=True` in the terrain config, columns are deterministically assigned to terrain types. The reward term replicates this assignment logic to know which column is "friction_plane" (drag = 0, friction is the only challenge) vs. "vegetation_plane" (drag > 0, always). This prevents the robot from getting confused by drag forces on terrain where friction is supposed to be the challenge.

**Why not just apply drag everywhere?** Because then the robot can't distinguish "this surface is slippery" from "this surface has drag." Separating them gives cleaner training signal and lets the policy develop distinct strategies for each.

---

## The Evaluation Pipeline

### The Four Gauntlets

Each test environment is a 50-meter course with 5 zones of increasing difficulty. The robot starts at x=0 and tries to reach x=49m. It's like a video game level with checkpoints.

#### Friction: The Ice Rink Gauntlet

```
Zone 1 (0-10m)   Zone 2 (10-20m)  Zone 3 (20-30m)  Zone 4 (30-40m)  Zone 5 (40-50m)
  Sandpaper         Dry rubber        Wet concrete       Wet ice        Oil on steel
  mu=0.90           mu=0.60           mu=0.35            mu=0.15         mu=0.05
  [EASY]            [MODERATE]        [CHALLENGING]      [HARD]          [NEARLY IMPOSSIBLE]
```

Physics insight: At mu=0.05, a 32kg robot can only push with 0.05 * 32 * 9.81 = 15.7N before its feet slide. That's barely enough to accelerate, let alone maintain speed.

#### Grass: The Swamp Gauntlet

```
Zone 1: Light fluid (c=0.5)    — Like walking through air
Zone 2: Thin grass (c=2.0)     — Ankle-high lawn
Zone 3: Medium grass (c=5.0)   — Knee-deep field
Zone 4: Thick grass (c=10.0)   — Dense underbrush
Zone 5: Dense brush (c=20.0)   — Wading through a swamp
```

The drag coefficient `c` means at 1 m/s, the retarding force is `c` Newtons. At c=20 and 4 feet in contact, that's 80N of resistance — significant for a 32kg robot.

#### Boulders: The Rubble Field

```
Zone 1: Gravel         (4500 rocks, 3-5cm)    — Crunchy but walkable
Zone 2: River rocks    (2400 rocks, 10-15cm)   — Watch your step
Zone 3: Large rocks    (1200 rocks, 25-35cm)   — Knee-height obstacles
Zone 4: Small boulders (600 rocks, 50-70cm)    — Hip-height obstacles
Zone 5: Large boulders (300 rocks, 80-120cm)   — Bigger than the robot
```

Each boulder is a random polyhedron (D8, D10, D12, or D20 shape) with random orientation. This isn't neat rows of boxes — it's chaos, just like real rubble.

#### Stairs: The Climb

```
Zone 1: 3cm steps   — Access ramp (barely noticeable)
Zone 2: 8cm steps   — Low residential stairs
Zone 3: 13cm steps  — Standard residential stairs
Zone 4: 18cm steps  — Steep commercial stairs
Zone 5: 23cm steps  — Maximum challenge (Spot's physical limit)
```

The cumulative elevation gain is 21.45 meters — that's a 7-story building. And unlike real stairs, there's no railing to grab.

### Production Results (40 episodes per combo, Feb 19, 2026)

Here's what we found with the 48-hour rough policy (model_29999.pt):

| Environment | Policy | Mean Progress | Fall Rate | Key Finding |
|---|---|---|---|---|
| **Friction** | flat | 38.7m (zone 4!) | 15% | Flat policy actually better here |
| **Friction** | rough | 27.1m (zone 3) | 70% | Rough policy *worse* — over-corrects on ice |
| **Grass** | flat | 27.1m (zone 3) | 10% | Decent baseline |
| **Grass** | rough | 25.0m (zone 3) | 17.5% | No improvement — never trained on drag |
| **Boulder** | flat | 10.8m (zone 2) | 60% | Expected — can't see terrain |
| **Boulder** | rough | 13.4m (zone 2) | 67.5% | Marginal improvement |
| **Stairs** | flat | 7.2m (zone 1-2) | 100% | Falls every single time |
| **Stairs** | rough | 11.4m (zone 2) | 15% | Clear winner — height scan helps |

**The uncomfortable truth:** The 48-hour rough policy was only clearly better on stairs. On friction, it was actually *worse* (Cohen's d = -1.18, p < 0.001). On grass, no statistically significant difference. On boulders, marginal improvement.

**Why?** Because the 48-hour training had:
- Friction range [0.5, 1.25] — never saw friction below 0.5 (ice starts at ~0.15)
- No vegetation drag simulation at all
- Boulder proxies maxed at 20cm — real test has 120cm boulders
- Only 8cm max stair height in training vs. 23cm in test

This is exactly why we built the 100-hour training config — to close these gaps.

---

## The Numbers That Matter

### Observation Space: 235 Dimensions

```
Indices  [0:3]    — Linear velocity (body frame)     "How fast am I going?"
Indices  [3:6]    — Angular velocity (body frame)     "Am I spinning?"
Indices  [6:9]    — Projected gravity vector           "Which way is down?"
Indices  [9:12]   — Velocity commands (vx, vy, omega)  "Where should I go?"
Indices  [12:24]  — Joint positions (relative)         "Where are my legs?"
Indices  [24:36]  — Joint velocities                   "How fast are my legs moving?"
Indices  [36:48]  — Last action                        "What did I just do?"
Indices  [48:235] — Height scan (187 rays)             "What's the ground like ahead?"
```

The height scan is a 17x11 grid of raycast points projected from the robot's body, measuring ground height relative to the body. On flat ground, all 187 values are near 0.0. On stairs, you see the step pattern. On boulders, you see irregular bumps.

**Critical detail:** The flat policy ignores indices [48:235] entirely. The rough policy needs them. If you fill them with the wrong default value (1.0 instead of 0.0), the policy thinks there are tall obstacles everywhere and panics. We learned this the hard way.

### Action Space: 12 Joints

```
Joints:
  fl_hx, fr_hx, hl_hx, hr_hx   — Hip abduction/adduction (leg spread)
  fl_hy, fr_hy, hl_hy, hr_hy   — Hip flexion/extension (leg forward/back)
  fl_kn, fr_kn, hl_kn, hr_kn   — Knee flexion/extension (leg bend)
```

Each joint gets a position offset that's added to the default standing pose. The PD controller (Kp=60, Kd=1.5) converts position targets to torques. The action scale is 0.25, meaning the network output is multiplied by 0.25 before being applied — this prevents wild joint swings during early training when the network outputs random garbage.

### Throughput Numbers

| Hardware | Envs | Steps/sec | Time for 1B steps |
|---|---|---|---|
| RTX 2000 Ada (8GB) | 4 | ~50 | ~8 months |
| RTX 2000 Ada (8GB) | 64 | ~800 | ~15 days |
| H100 NVL (96GB) | 8,192 | ~35,000 | ~8 hours |
| H100 NVL (96GB) | 20,480 | ~49,000 | ~5.7 hours |

This is why we need the H100. A 39-billion-step training run takes ~100 hours on H100 but would take *15 years* on the laptop GPU with 4 envs.

---

## Bugs, Disasters, and Hard-Won Lessons

This section is the most valuable part of this document. Every bug here cost us hours or days.

### Bug #1: The Silent Import Catastrophe

**Symptom:** Everything imports fine. No errors. Robot spawns. Then... nothing. No movement, no physics, no errors. Just a motionless robot in a void.

**Root cause:** Isaac Sim requires `SimulationApp` to be created BEFORE any `omni.*` imports. If you import `omni.isaac.core` first, it initializes with a null simulation context. No error. No warning. Just silent failure.

```python
# WRONG — dies silently
from omni.isaac.core import World
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

# CORRECT — works
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from omni.isaac.core import World  # Now safe
```

**Lesson:** When a framework has strict initialization order requirements, put a big comment at the top of every file that uses it. We added comments like `# !! SimulationApp MUST be created before this line !!` to every script.

### Bug #2: The Robot That Wouldn't Move

**Symptom:** Robot spawns correctly, physics is running, but legs are limp. Robot immediately collapses like its batteries died.

**Root cause:** `spot.initialize()` was called before the physics engine was actually stepping. The robot's articulation wasn't registered with the physics solver yet.

**Fix:** Initialize the robot *inside the first physics callback*, after at least one physics step has occurred:

```python
world.step(render=False)  # One physics step first
spot.initialize()         # NOW the articulation exists in PhysX
spot.post_reset()         # Reset state to default pose
```

**Lesson:** GPU-accelerated physics frameworks often defer setup until the first step. "Creating" an object and "registering it with the solver" are separate operations that happen at different times.

### Bug #3: The Closure Variable Trap

**Symptom:** `UnboundLocalError: cannot access local variable 'control_step'`

**Root cause:** Python closures. When you modify a variable inside a nested function, Python treats it as local unless you declare `nonlocal`:

```python
control_step = 0

def on_physics_step():
    nonlocal control_step  # WITHOUT THIS LINE: crash
    control_step += 1      # Python thinks this is a new local variable
```

**Lesson:** This bites everyone exactly once per project in Python. If you're modifying *any* outer variable inside a callback/closure, slap `nonlocal` on it immediately.

### Bug #4: The Zombie That Wouldn't Die (Server Down for Hours)

**Symptom:** Killed a training run mid-execution. Next Isaac Sim process hangs on startup. `nvidia-smi` hangs. Server effectively bricked.

**Root cause:** `SimulationApp.close()` triggers NVIDIA GPU driver cleanup that deadlocks in the Linux kernel (D-state — uninterruptible sleep). The process becomes a zombie: unkillable even by `kill -9`, holding GPU memory hostage. The only fix was a physical power cycle of the server.

**Fix:** Three-layer defense:

```python
# Layer 1: Never call close()
import os
os._exit(0)  # Brutal but reliable — skips GPU cleanup entirely

# Layer 2: Signal handler
import signal
def graceful_exit(sig, frame):
    save_metrics()  # Save your data first!
    os._exit(0)     # Then die immediately
signal.signal(signal.SIGINT, graceful_exit)

# Layer 3: Shell-level timeout
timeout --foreground -k 30 7200 python train.py
```

**Lesson:** This is the most important lesson in the entire project. **Never call `SimulationApp.close()`**. Just `os._exit(0)`. Yes, it's ugly. Yes, it skips Python cleanup. No, you do not have a better option. The GPU driver bug has been reported but not fixed as of Isaac Sim 5.1.

### Bug #5: The Height Scan Value That Broke Everything

**Symptom:** Rough policy works great in training. In standalone evaluation with flat terrain, robot immediately collapses with action norms 2.4x higher than expected.

**Root cause:** In training, the height scanner returns actual terrain heights relative to the robot. On flat ground, values are ~0.0. In standalone evaluation, we don't have Isaac Lab's RayCaster, so we fill the 187 height scan values manually. We initially filled with 1.0 ("maximum height difference"). The policy interpreted this as "I'm standing on top of a cliff" and panicked.

```python
# WRONG: action_norm = 7.42, robot collapses
height_scan = np.ones(187) * 1.0

# CORRECT: action_norm = 3.08, robot walks normally
height_scan = np.zeros(187)  # Flat ground = 0.0 relative height
```

**Lesson:** When deploying a trained model, every input must match the training distribution. Check the actual training values (we found: range [-0.000002, 0.148], mean 0.004). When in doubt, use the mean, not a guess.

### Bug #6: PhysX Collision Stack Overflow (20K Envs)

**Symptom:** Training launches, terrain generates, then PhysX spams errors about `collisionStackSize buffer overflow` and contacts are dropped. Robots fall through terrain.

**Root cause:** The default GPU PhysX collision stack is 64 MB (2^26 bytes). With 20,480 environments and 12 complex terrain types, the collision detection generates ~475 MB of contact data per frame.

**Fix:** Increase the buffer in the environment config:

```python
self.sim.physx.gpu_collision_stack_size = 2**30  # 1 GB (was 64 MB)
self.sim.physx.gpu_max_rigid_contact_count = 2**23
self.sim.physx.gpu_max_rigid_patch_count = 2**23
```

**Lesson:** GPU physics engines use fixed-size pre-allocated buffers (unlike CPU physics which can dynamically grow). When scaling up, always check buffer sizes. The error messages are helpful — they tell you the minimum size needed.

### Bug #7: Windows CRLF on Linux Server

**Symptom:** `$'\r': command not found` when running shell scripts on H100 after SCP from Windows.

**Root cause:** Windows saves files with `\r\n` line endings. Linux expects `\n` only. The `\r` gets interpreted as part of the command.

**Fix:** `sed -i 's/\r$//' script.sh` after every SCP from Windows.

**Lesson:** Configure your editor to use LF line endings for shell scripts, or add a `dos2unix` step to your deployment scripts. This will bite you every single time you develop on Windows and deploy to Linux.

### Bug #8: The Unicode Arrow That Crashed Windows

**Symptom:** `UnicodeEncodeError: 'charmap' codec can't encode character '\u2192'` when running on Windows.

**Root cause:** A `print()` statement used `→` (Unicode right arrow) in a status message. Windows console uses cp1252 encoding, which can't represent this character. The same code works fine on Linux (UTF-8).

**Fix:** Replace `→` with `->` in print statements. Unicode in comments and docstrings is fine (not printed to console).

**Lesson:** If your code might run on Windows, keep print statements ASCII-only. Or set `PYTHONIOENCODING=utf-8`, but that doesn't always work with all terminal emulators.

### Bug #9: Isaac Lab Reward Parameter Validation

**Symptom:** `ValueError: The term 'vegetation_drag' expects mandatory parameters: ['asset_cfg', 'sensor_cfg'] and optional parameters: ['drag_max', 'contact_threshold'], but received: ['asset_cfg', 'sensor_cfg', 'drag_max', 'contact_threshold', 'vegetation_terrain_name', 'friction_terrain_name']`

**Root cause:** Isaac Lab's `RewardManager` validates the `params` dict against the `__call__` method signature. If you add new parameters to the config but don't add them to the function signature, it rejects them.

**Fix:** Add all params to the `__call__` signature, even if they're only used in `__init__`:

```python
def __call__(self, env, asset_cfg, sensor_cfg,
             drag_max=8.0, contact_threshold=1.0,
             vegetation_terrain_name="vegetation_plane",  # Must be here
             friction_terrain_name="friction_plane",       # Must be here
             ) -> torch.Tensor:
```

**Lesson:** Frameworks that use introspection (checking function signatures at runtime) are powerful but unforgiving. Read the validation code, not just the documentation.

---

## How Good Engineers Think

Here are the meta-lessons — the ways of thinking that matter more than any specific bug fix.

### 1. Test Small Before Going Big

We never once launched a 20,480-env training run without first testing with 1-4 envs locally. Every H100 deployment followed this checklist:
1. Run with 1 env, 5 iterations locally (catches import errors, config bugs)
2. Run with 4 envs, 100 iterations locally (catches reward NaN, terrain generation issues)
3. Run with 20,480 envs, 1000 iterations on H100 (catches memory, throughput, scaling bugs)
4. Then — and only then — launch the full 60,000-iteration training

The 15 minutes spent on steps 1-2 saved us from wasting hours on step 3.

### 2. Make the Invisible Visible

Every print statement in our training script has `flush=True`. Why? Because Isaac Sim buffers stdout. Without `flush=True`, you see nothing for minutes, then a wall of text. When debugging, silence is your enemy.

We also added `PYTHONUNBUFFERED=1` to every launch script, periodic status prints every 1000 iterations, and piped all output to log files with `tee`. When something goes wrong at iteration 47,000 of a 100-hour run, you need to know exactly what was happening.

### 3. Separate Concerns Ruthlessly

Training and evaluation are completely separate codebases that share only the checkpoint file. This seems wasteful until you realize:
- Training uses Isaac Lab's manager-based RL (parallel, automatic, abstracted)
- Evaluation uses standalone Isaac Sim (sequential, manual, precise)
- They have different physics solvers, different terrain generators, different everything

Trying to share code between them would create a fragile coupling. Keeping them separate means we can modify one without breaking the other. The checkpoint `.pt` file is the only interface between them.

### 4. Automate the Boring Parts

Our shell scripts handle:
- Conda activation (different paths on Windows vs. H100)
- Screen session management (kill old, launch new, persist after SSH disconnect)
- TensorBoard alongside training
- Log file capture with timestamps
- Graceful error handling (don't abort all combos if one fails)
- Report generation after all episodes complete

The first version of each script was 5 lines. The final versions are 100+ lines each. The extra 95 lines handle every edge case we hit during development. This is normal — robust automation is always longer than naive automation.

### 5. Document the Weird Stuff

Normal code is self-documenting. Weird code needs comments. Here are examples of comments that earned their keep:

```python
# CRITICAL: height_scan fill must be 0.0 for flat ground, NOT 1.0
# See LESSONS_LEARNED.md "Height Scan Fill Value" — wrong value
# causes 2.4x action norm spike and immediate collapse

# PhysX GPU: numpy arrays are SILENTLY IGNORED by set_gains()
# Must use CUDA tensors. No error, no warning, just limp robot.

# Do NOT call simulation_app.close() — causes unkillable zombie
# process that holds GPU memory and requires power cycle
```

These aren't redundant comments — they're saving the next person (or future you) from repeating a painful debugging session.

### 6. Version Everything, Track Everything

Every training run saves:
- Full environment config (YAML)
- Full agent config (YAML)
- Training parameters (plain text)
- TensorBoard logs (scalars for all 19 reward terms + losses + throughput)
- Checkpoints every 1000 iterations

When experiment #37 produces better results than experiment #36, you need to know exactly what changed. "I think I tweaked the learning rate" is not a reproducibility strategy.

### 7. Expect Failure, Plan for Recovery

Our training script supports `--resume --load_run <dir> --load_checkpoint model_XXXXX.pt`. If the server crashes at iteration 45,000, we lose at most 1000 iterations of work (the save interval). This took 30 minutes to implement and saves potentially days of retraining.

Our eval script saves metrics every 50 episodes and has a signal handler that saves on Ctrl-C. If episode 743 out of 800 crashes, you still have 742 episodes of data.

**The pattern:** Always ask "what happens if this crashes halfway?" and build the recovery path before you need it.

---

## What I'd Do Differently

### Start with the Evaluation Framework

We built the training pipeline first and the evaluation pipeline second. In hindsight, we should have built evaluation first. Why? Because the evaluation environments *define the success criteria*. The 48-hour policy scored poorly on friction because we didn't know friction would be tested at mu=0.05 until we built the friction environment.

Build the test before the code. This is just TDD (Test-Driven Development) applied to robotics.

### Use a Configuration Management System

Our configs are scattered across Python dataclasses, YAML files, command-line arguments, and shell script variables. A single source of truth (like Hydra or OmegaConf) would make experiment tracking much cleaner. Right now, to understand a training run, you need to check:
1. `env_cfg.py` (environment parameters)
2. `ppo_cfg.py` (algorithm parameters)
3. `train_100hr.py` (command-line overrides)
4. `train_h100.sh` (shell-level overrides)

That's four places where a learning rate might be defined or overridden.

### Invest in Automated Testing Earlier

We wrote 84 unit tests for the evaluation pipeline — but only after the core code was "done." Writing tests alongside the code (not after) would have caught several bugs earlier. The `conftest.py` fixtures and mock objects were worth every line.

### Plan for Sim-to-Real from Day One

The entire project lives in simulation. We haven't deployed to a physical Spot yet. Some of our design decisions (like using Isaac Lab's specific PhysX solver settings with 4 position iterations and 0 velocity iterations) might not transfer well. If we'd planned for real hardware deployment from the start, we might have chosen different solver settings or added more realistic actuator models.

---

## Final Thoughts

This project is fundamentally about *closing the gap between simulation and reality*. Every terrain type, every domain randomization parameter, every reward term is an attempt to make our simulated training so diverse that the real world holds no surprises.

We're not there yet. The 48-hour policy proved that rough terrain training helps for stairs but hurts for friction. The 100-hour training config addresses this with 12 terrain types (vs 6), friction down to 0.05 (vs 0.5), and explicit vegetation drag simulation.

The tools we used (Isaac Sim, Isaac Lab, RSL-RL, PyTorch) are the same tools used by NVIDIA, ETH Zurich, and CMU for state-of-the-art quadruped locomotion research. The difference between a capstone project and a research paper isn't the tools — it's the depth of understanding, the thoroughness of evaluation, and the honesty about what doesn't work yet.

Speaking of which: if the 100-hour policy doesn't beat the flat policy on friction, that's a real result. It means our reward function or domain randomization still has gaps. Science doesn't fail when the experiment gives an unexpected result — it fails when you don't run the experiment at all.

Now go check TensorBoard. The training's been running for a while.

---

*Created for AI2C Tech Capstone — MS for Autonomy, February 2026*
*If this file helped you, pay it forward. Document your bugs.*
