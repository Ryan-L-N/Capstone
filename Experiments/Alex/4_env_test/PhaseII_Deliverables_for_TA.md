# Phase 2 Deliverables — Simplified Guide for TAs

**Team:** AI2C Tech Capstone — MS for Autonomy
**Date:** March 5, 2026

> **What this document is:** A plain-language version of our Phase 2 deliverables for TA reviewers who may not have a robotics background. All the same information is here — just explained without assuming you know reinforcement learning or robot simulation.

---

## What We're Doing (The Big Picture)

We're teaching a simulated robot dog (Boston Dynamics Spot) to walk across difficult terrain using AI. Think of it like a video game where instead of a human playing, an AI learns to control the dog's 12 leg joints by trial and error — millions of times — until it gets good at walking.

We have **two AI controllers** ("policies") we're comparing:

1. **Flat Policy (Baseline):** A pre-built controller from NVIDIA that only knows how to walk on flat ground. It can feel its own body (joint angles, speed, etc.) but cannot "see" the ground ahead. Think of it as a blindfolded dog walking by feel alone.

2. **Rough Policy (Ours):** Our custom-trained controller that can both feel its body AND "see" the ground using a height scanner (like a grid of laser rangefinders measuring ground height in front of the robot). We trained this one on progressively harder terrain over 120+ hours of GPU time.

**The question:** Does our terrain-aware policy actually perform better than the blind baseline when the ground gets hard?

To answer this, we test both policies across **5 different environments** (friction, grass, boulders, stairs, and obstacles) and measure how far each one gets, whether it falls, how stable it is, and how fast it moves.

---

## 1. Data Dictionary (What We Measure)

Every time the robot runs through a test course, we record one "episode" of data. Here's what each measurement means:

### Core Measurements

| What We Measure | What It Means | Example |
|---|---|---|
| **episode_id** | A unique label for this specific test run | `friction_flat_ep0042` = friction course, flat policy, run #42 |
| **policy** | Which AI controller was driving | `"flat"` (baseline) or `"rough"` (ours) |
| **environment** | Which test course | `"friction"`, `"grass"`, `"boulder"`, `"stairs"`, or `"obstacle"` |
| **completion** | Did the robot reach the end? | `true` or `false` |
| **progress** | How far did it get (in meters)? | 0 to 50 meters (50m = full course) |
| **zone_reached** | What difficulty level did it reach? (1=easy, 5=hard) | Each course has 5 progressively harder zones |
| **time_to_complete** | How many seconds to finish (if it finished) | `null` if it didn't finish |
| **stability_score** | How wobbly was the robot? (lower = more stable) | Combines roll, pitch, height bounce, and spin |
| **fall_detected** | Did the robot fall over? | `true` or `false` |
| **fall_location** | Where (in meters) did it fall? | `null` if no fall |
| **mean_velocity** | Average forward speed (m/s) | Target is 1.0 m/s (a brisk walk) |
| **total_energy** | How much motor effort was used | Higher = less energy-efficient |
| **episode_length** | How long the test lasted (seconds) | Max 120s, shorter if robot fell |

### The 5 Test Environments

Each environment gets progressively harder as the robot moves forward (5 zones, each 10m long for linear courses):

| Environment | What Changes | Zone 1 (Easy) | Zone 5 (Hard) |
|---|---|---|---|
| **Friction** | How slippery the ground is | Sandpaper-like grip | Nearly ice |
| **Grass** | How thick the vegetation is | Light grass | Dense brush that resists movement |
| **Boulder** | Size of rocks on the ground | Pebbles (3-5 cm) | Large boulders (80-120 cm) |
| **Stairs** | How tall each step is | Shallow ramp (3 cm steps) | Steep stairs (23 cm steps) |
| **Obstacle** | Density of furniture and vehicles | Open area near start | Dense clutter further from start |

The **obstacle** environment is different from the others — instead of a straight 50m course, it's a 100m x 100m open field filled with 360 randomly placed objects (couches, chairs, tables, cars, trucks, etc.). The robot starts at one side and has to navigate to a goal 75+ meters away. Zones are based on distance from the start point (5 bands of 20m each). This environment was built by team member Cole.

### Experimental Design

- **2 policies** x **5 environments** = **10 combinations**
- **100 episodes** per combination = **1,000 total test runs** for the final evaluation
- This gives us enough data for statistically meaningful comparisons

---

## 2. Exploratory Data Analysis (How We Check Our Data)

Before we draw any conclusions, we verify the data makes sense.

### What We Expect to See

- **Completion rates** should vary a lot: both policies probably complete easy zones, but the rough policy should handle hard zones better than the flat one.
- **Progress** on hard environments (boulders, stairs) should be right-skewed — meaning most runs stop early, with a few making it far.
- **Stability scores** should be low for most runs (stable walking) with a tail of high values right before falls.
- **Speed** should be around 1.0 m/s when things are going well, dropping to 0 when the robot is stuck or about to fall.
- **Episode lengths** should be bimodal — either the robot finishes quickly or hits the 120-second timeout.
- **Obstacle environment** may show lower speeds (robot has to weave around objects) and different failure patterns (collision-induced falls vs. terrain-induced falls).

### Handling Missing Values

Three fields can be `null` — this is by design, not a bug:
- `time_to_complete` is `null` when the robot didn't finish
- `fall_location` and `fall_zone` are `null` when the robot didn't fall

### Statistical Tests We Run

| Test | What It Tells Us |
|---|---|
| **Welch's t-test** | Is the average progress statistically different between flat vs rough? |
| **Cohen's d** | How big is the difference (effect size)? |
| **Two-proportion z-test** | Is the completion rate statistically different between flat vs rough? |

We use alpha = 0.05 (standard significance threshold — meaning we need 95% confidence to declare a difference real).

### Plots We Generate

1. Bar chart comparing completion rates (flat vs rough for each environment)
2. Box plots showing the spread of progress distances
3. Heatmap showing where falls happen (which zones are most dangerous)
4. Line plot tracking stability across zones

---

## 3. Data Schema (How Data Is Stored)

### File Format

Each test produces a `.jsonl` file (JSON Lines — one JSON object per line, like a simple database). Files are named by environment and policy:

```
friction_flat_episodes.jsonl
stairs_rough_episodes.jsonl
obstacle_rough_episodes.jsonl
```

### What a Single Record Looks Like

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

**Reading this example:** The flat policy ran in the friction environment. It didn't finish the course (only got to 11.8m, zone 2). It was very stable (score 0.096), didn't fall, moved at about 1.0 m/s, and the episode timed out at 120 seconds.

### Validation

We have a formal JSON Schema that validates every record's types and ranges. This catches any corrupt or malformed data before analysis. Validated with Python's `jsonschema` library.

---

## 4. Model Documentation (How We Built the AI)

### 4.1 The Two Policies

**Flat Policy (Baseline):**
- Made by NVIDIA, comes pre-installed with Isaac Sim (their robot simulator)
- "Feels" its body with 48 sensor values (speed, joint angles, balance, etc.)
- Cannot see the terrain ahead — walks blind
- We did NOT modify this; it's our control group

**Rough Policy (Ours):**
- Custom-built by our team
- "Feels" its body (same 48 values) PLUS "sees" the ground with 187 height measurements = **235 total inputs**
- The height scanner works like a grid of 187 tiny downward-pointing rangefinders (17 x 11 grid covering 1.6m x 1.0m in front of the robot)
- **Outputs:** 12 numbers — one target angle for each leg joint
- Trained across a 4-phase curriculum that gradually increases terrain difficulty

### 4.2 Network Architecture (The Brain)

The rough policy's "brain" is a neural network — a mathematical function that maps 235 inputs to 12 outputs through layers of computation:

```
235 inputs --> 1024 neurons --> 512 neurons --> 256 neurons --> 12 outputs
                 (ELU)           (ELU)           (ELU)
```

- **~1.8 million trainable parameters** (numbers the AI adjusts during learning)
- There's actually two copies of this network: an "actor" (decides what to do) and a "critic" (evaluates how good the current situation is) — standard for our training algorithm
- ELU = a type of activation function (makes the network non-linear so it can learn complex patterns)

### 4.3 Compute Resources

Everything runs on an NVIDIA H100 GPU (a very powerful AI accelerator with 96 GB of memory) hosted on a university server.

| What | Time |
|---|---|
| Training (all phases and trial runs) | ~120+ hours |
| Final evaluation (1,000 test episodes) | ~11 hours |
| Debug/testing | ~1 hour |
| **Total GPU time** | **~132+ hours** |

### 4.4 Training (How the AI Learned)

#### The Algorithm: PPO

We use **Proximal Policy Optimization (PPO)** — one of the most popular reinforcement learning algorithms. Here's how it works at a high level:

1. Thousands of simulated robots walk around simultaneously (up to 40,960 at once!)
2. Each robot tries to follow velocity commands (walk forward, turn, etc.)
3. Good behaviors earn positive rewards; bad behaviors earn penalties
4. After a batch of experience, the AI updates its neural network to do more of what worked
5. Repeat millions of times

**Key settings:**

| Setting | Value | What It Means |
|---|---|---|
| Clip range | 0.2 | Limits how much the AI can change per update (prevents wild swings) |
| Discount factor | 0.99 | How much the AI cares about future vs. immediate rewards (0.99 = very forward-looking) |
| Learning rate | 3e-5 to 3e-4 | How big each learning step is (smaller = safer but slower) |
| Mini-batches | 64 | Data is split into 64 chunks per update for efficiency |
| Learning epochs | 4-8 | How many times the AI re-examines each batch of data |

#### The 4-Phase Curriculum

Instead of throwing the robot onto hard terrain immediately (which causes the training to crash), we gradually increase difficulty — like teaching a kid to ride a bike with training wheels first:

| Phase | What the Robot Trains On | Duration | Result |
|---|---|---|---|
| **A — Flat Ground** | 100% flat terrain, 20,480 robots | 500 iterations | Learned to walk: 99.3% survival rate |
| **A.5 — Transition** | 50% flat + 50% gentle bumps | 1,000 iterations | Learned basic rough walking: 92.9% survival |
| **B-easy — Easy Rough** | All 11 terrain types at low difficulty | 5,002 iterations | Can handle varied terrain at moderate levels |
| **B — Full Rough** | All 11 terrain types at max difficulty | Ongoing (Trial 11d) | Terrain level 4.5+ — climbing real stairs, navigating debris |

Each phase starts from where the previous one left off (transfer learning).

#### What the AI Is Rewarded For

The reward function has 19 terms — think of it as a scorecard that tells the AI what "good walking" looks like:

**Things that earn points (positive rewards):**
- Walking at the commanded speed (+5.0)
- Turning at the commanded rate (+5.0)
- Maintaining a trot gait — diagonal legs loosely in sync (+1.0, loosened from +10.0 to allow terrain-adaptive gaits)
- Lifting feet high enough to clear obstacles (+3.0)
- Proper foot timing — not dragging feet (+5.0)

**Things that lose points (penalties):**
- Falling over or tilting too much (-3.0 to -5.0)
- Not maintaining proper height above ground (-1.0) — adapts per terrain: stand tall on flat ground, crouch on hard terrain
- Hitting body on ground while moving (-1.5 to -2.0)
- Bouncing, swaying, or wobbly movement (-0.5)
- Feet slipping on the ground (-0.5)
- Deviating from default joint positions (-0.2)
- Jerky, twitchy leg movements (-0.1)
- Tripping on obstacles (-0.02)
- Using too much motor power (-0.0005)
- Hitting joint limits (-5.0)

The weights (numbers in parentheses) determine how much each term matters. Higher weights mean the AI cares more about that behavior.

#### The 11 Training Terrains

The simulated world contains 11 different terrain types that the robot must learn to handle:

| Category | Terrain Types | What They Test |
|---|---|---|
| **Geometric (40%)** | Pyramid stairs (up/down), random boxes, stepping stones | Climbing and precise foot placement |
| **Surface (40%)** | Rough ground, slopes (up/down), waves, low-friction planes, vegetation (high drag) | Adapting to different ground properties |
| **Compound (20%)** | Heightfield stairs, discrete obstacles, repeated box patterns | Combined challenges |

Difficulty increases across 10 rows — robots that survive longer get promoted to harder terrain automatically (curriculum learning). Think of it like a video game with 10 difficulty levels:

#### What Each Difficulty Level Looks Like

| Level | Stair Height | Ground Roughness | Slope Angle | Think of It As... |
|-------|-------------|-----------------|-------------|-------------------|
| **0** | 5cm (speed bump) | Barely bumpy | Flat | Walking on a sidewalk |
| **1** | 7cm (low curb) | Gravel-like | ~5° | Walking on a gravel path |
| **2** | 9cm (thick book) | Bumpy trail | ~11° | Hiking on a rough trail |
| **3** | 12cm (tall curb) | Rocky | ~17° | Scrambling over a rocky trail |
| **4** | **14cm (half-stair)** | Very rocky | **~22°** | **Walking through a construction site** |
| **5** | **16cm (real stair)** | Rubble field | **~28°** | **Climbing actual stairs and steep hills** |
| **6** | 18cm (tall stair) | Boulder-like | ~33° | Navigating a disaster site |
| **7** | 21cm (very tall stair) | Extreme rocks | ~39° | Extreme terrain — near the robot's physical limits |
| **8** | 23cm (full leg reach) | Chaotic surface | ~44° | At the robot's hardware limits |
| **9** | 25cm (maximum) | Violent chaos | ~50° | Beyond what real stairs look like |

**For context:** A standard indoor stair is about 16-18cm (levels 5-6). The robot (Spot) is only 42cm tall, so a 25cm obstacle at level 9 is over half its body height — like a human trying to climb over a waist-high wall with every step.

**Where our robot is now:** The training (Trial 11d) has reached level 4.5+, meaning it can handle construction-site-level terrain including 14cm stairs, rough ground with ±8cm bumps, and 22° slopes. Previous training attempts couldn't get past level 4.1 — multiple rounds of reward tuning, plus two new features, finally broke through:
- **Terrain-scaled velocity:** The robot automatically runs fast on easy terrain and walks carefully on hard terrain (instead of being asked to sprint on stairs)
- **Terrain-scaled height:** The robot stands tall on flat ground (42cm) and crouches on hard terrain (25cm), instead of always crouching everywhere

#### Domain Randomization (Making Training Robust)

To prevent the AI from memorizing one specific environment, we randomize:
- **Ground friction:** Varies from slippery (0.3) to grippy (1.5)
- **Robot weight:** +/- 5 kg (simulates carrying different payloads)
- **Random pushes:** Every 10-15 seconds, the robot gets shoved (teaches it to recover from disturbances)
- **Start position:** Random location and facing direction each episode

#### Safety Mechanisms (Preventing Training Crashes)

Training this AI is unstable — numbers can blow up to infinity ("NaN errors") or oscillate wildly, crashing the entire training run. We discovered and fixed four critical failure modes:

1. **NaN Sanitizer:** Catches and fixes corrupted numbers in the neural network before they cause a crash. Normal math cleanup functions don't work on NaN (Not a Number) — you need explicit detection first.

2. **Pre-Forward Safety Clamp:** Checks the network's exploration randomness before every single decision. Catches corruption that happens during the network update process.

3. **Value Loss Watchdog:** Monitors a key training metric and automatically reduces the learning rate if it detects oscillation. Like cruise control that taps the brakes when the car starts shaking.

4. **Noise Clamping:** Limits how random the AI's actions can be. Too much randomness causes the robot to fall constantly on hard terrain, preventing it from learning. Capping randomness at 0.7 (from 1.0) fixed this.

These safeguards were developed through painful trial-and-error across 11+ training attempts. Without them, training reliably crashes within hours.

5. **Terrain-Scaled Commands:** The robot receives speed commands adapted to its current terrain difficulty — sprinting on easy ground, walking carefully on hard terrain. This prevents the AI from being asked to do something impossible (like sprint on steep stairs) and teaches it to associate what it "sees" in the ground ahead with the right speed and posture.

#### Physics Configuration

| Setting | Value | Plain English |
|---|---|---|
| Physics rate | 500 Hz | The simulator computes physics 500 times per second |
| Control rate | 50 Hz | The AI makes a decision 50 times per second |
| Action scale | 0.3 | Each decision moves joints by at most 30% of their range |
| PD gains | Kp=60, Kd=1.5 | How "stiff" vs "damped" the leg joints are |
| Episode length | 30 seconds | Each practice run lasts 30 simulated seconds |

### 4.5 Evaluation (How We Test)

**For linear courses (friction, grass, boulder, stairs):**
- Robot spawns at one end of a 50m course facing forward
- It follows waypoints at 1.0 m/s (a brisk walking pace)
- The course gets progressively harder (5 zones of 10m each)
- Episode ends when: it reaches the end, it falls, or 120 seconds pass

**For obstacle course:**
- Robot spawns in a 100m x 100m field filled with 360 objects
- It navigates to a randomly placed goal 75+ meters away
- Must weave around furniture, cars, and trucks
- Source: `Cole/Testing_Environments/Testing_Environment_1.py`

**Scale:**
- Debug: 5 episodes x 10 combinations = 50 runs (~42 minutes)
- Production: 100 episodes x 10 combinations = 1,000 runs (~11 hours)

### References

[1] NVIDIA, "Isaac Sim Documentation," 2025. https://docs.omniverse.nvidia.com/isaacsim/latest/

[2] NVIDIA, "Isaac Lab Documentation," 2025. https://isaac-sim.github.io/IsaacLab/

[3] J. Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.

[4] N. Rudin et al., "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning," CoRL 2022.

[5] Boston Dynamics, "Spot Robot," 2024. https://bostondynamics.com/products/spot/

---

## 5. Reflection

This project applies reinforcement learning — the same type of AI behind game-playing systems like AlphaGo — to a practical robotics challenge: teaching a robot dog to walk over difficult terrain. The key insight is that by gradually increasing difficulty (our 4-phase curriculum) rather than training on everything at once, we can produce a much more capable and stable AI controller.

The journey from Phase 1 (a simple 48-hour training run) to the current curriculum approach involved discovering and fixing five critical bugs that caused training to crash. Each crash taught us something new about the fragility of reinforcement learning at scale — for example, that a single corrupted number (NaN) in one neural network parameter can cascade through the entire training process in milliseconds. The safety mechanisms we built are now automatic, making the system robust enough for multi-day training runs.

Adding Cole's obstacle navigation environment as a 5th test arena adds an important dimension to the evaluation. The four linear courses test terrain traversal — can the robot walk over difficult ground? The obstacle course tests spatial navigation — can the robot plan a path around objects? This is a capability the rough policy was never explicitly trained for, but might emerge naturally from its height-scan perception.

---

## 6. Changes from Version 1.0

- **Section 4 completely rewritten:** Old version documented a single 48-hour training run (30K iterations, 14 rewards, 6 terrain types, smaller network). New version documents the 4-phase curriculum training (120+ hours, 19 rewards, 11 terrains, larger network, safety mechanisms).
- **5th environment added everywhere:** Cole's obstacle navigation arena added to Sections 1, 2, 3, and 4.5. All combination counts updated from 8 to 10.
- **Reflection updated:** Added discussion of curriculum learning and team collaboration.

**Version 2.1 (March 6, 2026):**

- **Section 4.4 updated:** Reward weights updated to reflect Trial 11d tuning (gait loosened from +10 to +1, penalties reduced). Added terrain-scaled velocity commands and terrain-scaled height targets. Updated training progress from level 4 to 4.5+.
- **Safety Mechanisms:** Added terrain-scaled commands as a 5th mechanism.
