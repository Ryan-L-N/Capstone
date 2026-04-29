# Sim-to-Real (S2R) Wrappers — Plain Language Guide

*For discussion with CMU PhD reviewers — March 2026*

---

## What Is Sim-to-Real?

Training a robot in simulation is like learning to drive in a video game. The game has perfect physics, instant response, and pixel-perfect vision. Then you sit in a real car — there's steering play, brake lag, dirty windshields, and potholes. The skills transfer, but the imperfections throw you off.

**Sim-to-Real (S2R)** is the engineering discipline of making simulation imperfect *on purpose*, so the policy isn't surprised by reality.

Our S2R system has three wrappers that sit between the policy brain and the simulated environment, deliberately corrupting the signals in both directions.

---

## The Three Wrappers

### 1. Action Delay — "Steering Lag"

**The analogy:** Imagine driving a car where the steering wheel has a half-second delay. You turn left, and half a second later the car actually turns. You'd have to *anticipate* every turn — start turning before the curve, not during it.

**What it does:** When the policy outputs a joint command at time T, that command goes into a ring buffer. The environment doesn't execute it until time T+40ms (2 control steps later). The robot is always executing commands from 40ms ago.

**Why it exists:** Real Boston Dynamics Spot has ~40-60ms between "brain says move" and "motors actually move." This is communication latency (WiFi/ethernet), command processing, and motor controller response time.

**What happens without it:** A policy trained with instant response over-corrects on real hardware. It sees itself tilting, sends a correction, but the correction arrives 40ms late — by then it's already tilted further. The correction overshoots, causing oscillation that builds until the robot falls. This is the same phenomenon as when you over-steer a car on ice.

```
Simulation (no delay):
  Brain: "tilt detected → correct NOW" → Robot corrects → Stable

Real hardware (40ms delay):
  Brain: "tilt detected → correct NOW"
  ...40ms passes...
  Robot: "executing correction" → But tilt is now WORSE → Overcorrection → Oscillation → Fall
```

**Current setting:** 2 steps at 50Hz = **40ms delay**

---

### 2. Observation Delay — "Seeing the Past"

**The analogy:** Imagine playing basketball while watching yourself on a TV with a 1-second delay. You see yourself at the free throw line, but you've already taken the shot. Every decision you make is based on where you *were*, not where you *are*.

**What it does:** Instead of giving the policy the current sensor readings, it gives readings from 20ms ago. The policy makes decisions based on stale information.

**Why it exists:** Real IMUs (inertial measurement units) and joint encoders have processing pipelines. The sensor reads a value, digitizes it, timestamps it, sends it over a bus, and the policy receives it ~10-20ms later. By then, the robot has moved.

**What happens without it:** The policy trusts that sensor readings are instantaneous. On real hardware, it reads "I'm at 5 degrees tilt" but the robot is actually at 6 degrees. Every correction is slightly wrong. Over many steps, these small errors accumulate.

```
Combined with action delay:
  Robot is at state S(now)
  Policy receives S(now - 20ms)     ← observation delay
  Policy computes correction
  Correction arrives at S(now + 40ms) ← action delay
  Total: policy is correcting for a state that's 60ms in the past
```

**Current setting:** 1 step at 50Hz = **20ms delay**

---

### 3. Sensor Noise — "Dirty Glasses + Faulty Instruments"

This wrapper does three things at once:

#### a) Height Scan Dropout (5% of rays)

**The analogy:** Imagine navigating a room by shining 187 flashlight beams at the floor and measuring how far away the floor is. Now randomly turn off 9 of those flashlights every second. Sometimes you lose the ones pointing at the stairs ahead. Sometimes you lose the ones under your feet. You have to navigate with incomplete terrain information.

**What it does:** Each control step, 5% of the 187 height scan rays are randomly zeroed. The policy sees "flat ground" (0.0) where there might actually be a boulder or a stair.

**Why it exists:** Real depth cameras (Intel RealSense, LiDAR) have blind spots. Reflective surfaces (water puddles, glass) return no signal. Dark surfaces absorb the laser. Moving legs occlude the view. At any moment, some portion of terrain data is simply missing.

#### b) IMU Drift (Ornstein-Uhlenbeck process)

**The analogy:** Imagine a speedometer that slowly drifts. At first it reads 0 mph correctly. An hour later it reads 3 mph while you're parked. The error is small at any moment but persistent — it doesn't jump around randomly, it *wanders* slowly.

**What it does:** Adds a slow random walk to the base velocity readings (both linear and angular). The drift persists across steps — it's not random noise that averages out, it's a bias that accumulates and then slowly reverts.

**Why it exists:** Real IMUs drift with temperature changes, vibration, and magnetic interference. A cheap IMU might drift 0.1 m/s per minute. The policy must learn to not fully trust velocity readings, and to use other cues (joint angles, terrain changes) to cross-check.

#### c) Spike Noise (0.1% probability)

**The analogy:** Static on a radio. Most of the time the signal is clear, but occasionally there's a loud crackle that completely corrupts one reading.

**What it does:** Each sensor channel has a 0.1% chance per step of getting a large random value added. This simulates electromagnetic interference or sensor hardware glitches.

**Current settings:**
| Parameter | Value | Real-world equivalent |
|-----------|-------|----------------------|
| Ray dropout | 5% | LiDAR occlusion / depth camera blind spots |
| IMU drift rate | 0.002 | Temperature-induced accelerometer drift |
| Spike probability | 0.1% | EMI / hardware glitch |

---

## The Env-Level Hardening (On Top of Wrappers)

Beyond the wrappers, the environment config itself also adds noise and randomization.

**The analogy:** The wrappers corrupt the *communication* (like a bad phone line). The env-level hardening changes the *world itself* (like training in rain, wind, and different shoes every day).

**IMPORTANT:** After 10 deployment bugs, we learned that env-level hardening must stay
at Mason's proven values. Wider DR (friction 0.15, mass ±5 kg, pushes ±3 N) caused
instant falls. The progressive wrapper handles S2R robustness instead.

| What | Mason Default | Our S2R Setting | Status | Real-World Equivalent |
|------|--------------|----------------|--------|----------------------|
| Height scan noise | ±0.1m | ±0.1m (Mason's) | KEPT | Depth camera accuracy at range |
| Velocity noise | ±0.1 m/s | ±0.1 m/s (Mason's) | KEPT | IMU noise floor |
| Joint position noise | ±0.05 rad | ±0.05 rad (Mason's) | KEPT | Encoder quantization + backlash |
| Joint velocity noise | ±0.5 rad/s | ±0.5 rad/s (Mason's) | KEPT | Numerical differentiation noise |
| Friction range | 0.3 - 1.0 | 0.3 - 1.0 (Mason's) | KEPT | ~~0.15-1.3 caused falls~~ |
| Mass randomization | ±2.5 kg | ±2.5 kg (Mason's) | KEPT | ~~±5.0 too aggressive~~ |
| Push forces | ±0.5 m/s / 10-15s | ±0.5 m/s / 10-15s (Mason's) | KEPT | ~~±3.0 N caused falls~~ |
| Motor power penalty | None | **-0.005 weight (NEW)** | ADDED | Battery life conservation |
| Torque limit penalty | None | **-0.3 weight (NEW)** | ADDED | Real motor can't exceed 45 Nm (hip) |
| Joint torques weight | -5e-4 | **-1e-3 (increased)** | CHANGED | Torque awareness |
| Body contact | **Hard termination** | **Soft penalty -1.5** | CHANGED | ~~Hard kill prevented exploration~~ |
| Observation corruption | Disabled | **Enabled** | CHANGED | Sensor noise applied |

The key insight: **physics DR stays safe, S2R comes from progressive wrappers.**

---

## Why We Can't Turn Everything On At Once

**The analogy:** Teaching someone to juggle. You don't hand them 5 balls on day one while wearing oven mitts on a moving bus. You start with 1 ball, then 2, then 3. The oven mitts come after they can juggle 3 comfortably.

We tried 4 different approaches before finding what works:

| Attempt | What we did | What happened | Bug # |
|---------|-------------|---------------|-------|
| 1 | From scratch + full S2R wrappers + wider DR | Standing still collapse — never learned to walk | S2R-6 |
| 2 | From distilled_6899 + full S2R wrappers | Same collapse — policy never trained with delay | S2R-6 |
| 3 | From hybrid_nocoach + wider DR (friction 0.15) | 100% body contact — instant falls on ice | S2R-8 |
| 4 | From hybrid_nocoach + Mason DR + hard body termination | Gait collapsed after 300-iter warmup — hard kill prevented recovery | S2R-9 |
| 5 | From hybrid_nocoach + Mason DR + soft termination + lr=3e-5 | Gait collapsed after warmup — LR too aggressive | S2R-11 |
| **6** | **From hybrid_nocoach + Mason DR + soft termination + lr=1e-5 + gradual layer unfreeze + progressive S2R** | **Terrain 5.18 friction, 4.97 stairs — stable and climbing!** | **Current** |

### The Winning Formula

1. **Base checkpoint:** `hybrid_nocoach_19999.pt` (proven walking gait)
2. **Physics DR:** Mason's safe values (friction 0.3-1.0, mass ±2.5 kg)
3. **Body contact:** Soft penalty (-1.5), not hard termination
4. **Learning rate:** 1e-5 (very conservative fine-tuning)
5. **Gradual actor unfreeze:**
   - Iter 0-300: ALL actor frozen (critic calibrates)
   - Iter 300-500: Only output layer unfreezes (128→12)
   - Iter 500-700: Middle layer unfreezes (256→128)
   - Iter 700+: All layers unfrozen (full fine-tuning)
6. **Progressive S2R:** Wrappers scale 0→100% with terrain curriculum level

---

## Our Training Strategy

### Current Strategy: Progressive S2R Integrated into Curriculum

After 10 deployment bugs and 4 failed approaches, we landed on the cleanest solution:
**S2R intensity scales automatically with terrain curriculum level.**

```
Terrain Row 0-2 (easy):   S2R scale = 0.0  → Clean signals, learn to walk on specialty terrain
Terrain Row 3-4 (medium): S2R scale = 0.3  → 12ms delay, 1.5% dropout
Terrain Row 5-6 (hard):   S2R scale = 0.6  → 24ms delay, 3% dropout
Terrain Row 7-9 (expert):  S2R scale = 1.0  → Full 40ms delay, 5% dropout, IMU drift
```

**Base checkpoint:** `hybrid_nocoach_19999.pt` (Mason baseline, proven 0% flip gait)
**Physics DR:** Mason's safe values (friction 0.3-1.0, mass ±2.5 kg)
**Body contact:** Soft penalty (-1.5), NOT hard termination
**Learning rate:** 1e-5 (very conservative — 3e-5 was too aggressive, destroyed gait)

This eliminates separate phases entirely. The robot learns terrain AND S2R together,
with terrain leading and S2R following. No phase transitions, no manual intervention.

**The analogy:** Teaching someone to juggle while gradually introducing a rocking boat.
The boat starts still (row 0-2), rocks gently as they improve (row 3-6), and reaches
full ocean waves only when they're an expert juggler (row 7+).

### Gradual Actor Unfreeze — "Waking Up One Limb at a Time"

When we load the Mason baseline into a new training environment with different reward
weights, the brain (neural network) needs to adapt. But if the entire brain changes
at once, it's like a surgeon rewiring your whole nervous system in one operation —
you'd lose the ability to walk while your body figures out the new wiring.

**The analogy:** Imagine you're an expert piano player, and someone changes the tuning
of your piano. If they change ALL 88 keys at once, you can't play anything — every
note sounds wrong and you have no reference point. But if they change just the last
octave first (the high notes), you can still play with the other 7 octaves as anchor
while your ear adjusts to the new high notes. Then they change the next octave, and
so on. By the end, the whole piano is retuned and you can still play beautifully.

Our neural network has 4 layers:
```
Layer 1 (input):   235 observations → 512 neurons   "Perceives the world"
Layer 2 (deep):    512 → 256 neurons                 "Understands patterns"
Layer 3 (middle):  256 → 128 neurons                 "Plans movements"
Layer 4 (output):  128 → 12 joint commands            "Executes actions"
```

**The unfreeze schedule:**
```
Iter 0-300:    ALL layers FROZEN         Critic learns new reward landscape
               (Mason gait runs unchanged)  while the actor performs perfectly

Iter 300-500:  Layer 4 UNFREEZES         Only the final "execution" layer adapts
               (output: 128→12)          Core walking patterns stay locked
               Layers 1-3 still frozen   Like tuning only the last octave

Iter 500-700:  Layer 3 UNFREEZES         "Movement planning" layer adapts
               (middle: 256→128)         Deep understanding still locked
               Layers 1-2 still frozen   Like tuning 2 octaves now

Iter 700+:     ALL layers UNFROZEN       Full brain adapts freely
               (full fine-tuning)        But by now each layer has a stable
                                         anchor from the layers above it
```

**Why this works:** Each unfrozen layer has the stability of the still-frozen layers
above it as a reference. The output layer adjusts its commands while the deeper layers
still "see" and "plan" the same way. By the time the deep layers unfreeze, the output
layer has already settled into its new range, providing stability from below.

**Results so far:**
- Previous approach (unfreeze everything at iter 300): terrain crashed from 5.0 → 0.0
- Gradual unfreeze: terrain held at 5.2 (friction) and **climbed to 5.7** (stairs)
- Both survived Phase 2 (iter 300) and Phase 3 (iter 500) with zero collapse

### What we tried and why it failed

| Approach | Result | Bug # |
|----------|--------|-------|
| From scratch + full S2R | Standing still collapse (never learned to walk) | S2R-6 |
| From distilled_6899 + full S2R | Same collapse (policy never saw delay) | S2R-6 |
| From hybrid_nocoach + wider DR | 100% body contact falls (friction 0.15 = ice) | S2R-8 |
| From hybrid_nocoach + Mason DR + hard termination | Falls after warmup, no recovery | S2R-9 |
| **From hybrid_nocoach + Mason DR + soft termination + progressive S2R** | **Terrain 5.27 and climbing** | **Current** |

### Distillation (after expert training)
Distill all 6 S2R-hardened experts into one student at **20 Hz** (real Spot control rate)
with full S2R wrappers active from step 0 (the student inherits S2R robustness from experts).

---

## Curriculum Levels in Plain Language

The terrain curriculum has 10 difficulty rows (0-9). The terrain level metric is the
average row across all 4096 training robots. Here's what each level means in
real-world terms:

### Friction Expert Curriculum
| Row | Friction Coeff (mu) | Real-World Equivalent | Difficulty |
|-----|--------------------|-----------------------|------------|
| 0 | 1.5 | Rubber gym mat | Trivial |
| 1 | 1.3 | Dry concrete | Easy |
| 2 | 1.1 | Asphalt road | Easy |
| 3 | 0.9 | Indoor tile | Moderate |
| 4 | 0.7 | Polished wood floor | Moderate |
| 5 | 0.5 | Wet concrete | Hard |
| 6 | 0.35 | Smooth metal plate | Hard |
| 7 | 0.2 | Wet tile | Very hard |
| 8 | 0.1 | Oiled surface | Extreme |
| 9 | 0.05 | Black ice | Nearly impossible |

**Our friction expert is at level 5.2** = walking confidently on wet concrete.

### Stairs Expert Curriculum
| Row | Step Height | Real-World Equivalent | Difficulty |
|-----|-----------|------------------------|------------|
| 0 | 3 cm | Door threshold | Trivial |
| 1 | 5 cm | Sidewalk curb cut | Easy |
| 2 | 8 cm | Low garden step | Easy |
| 3 | 10 cm | Short porch step | Moderate |
| 4 | 13 cm | Standard residential stair | Moderate |
| 5 | 15 cm | Code-minimum stair height | Hard |
| 6 | 18 cm | Steep residential stair | Hard |
| 7 | 20 cm | Commercial staircase | Very hard |
| 8 | 23 cm | Industrial stair / loading dock | Extreme |
| 9 | 25 cm | Maximum — taller than Spot's shin | Nearly impossible |

**Our stairs expert is at level 5.7** = climbing 16-17cm stairs (steep residential).

### Boulder Expert Curriculum (future)
| Row | Obstacle Height | Real-World Equivalent |
|-----|---------------|------------------------|
| 0 | 5 cm | Loose gravel / small rocks |
| 3 | 13 cm | Fallen branch / curb |
| 5 | 18 cm | Large rock / construction debris |
| 7 | 23 cm | Boulder / collapsed wall chunk |
| 9 | 30 cm | Major obstacle (half of Spot's leg length) |

### What the Numbers Mean for the CMU Discussion

- **Level 3.5** (Mason baseline plateau) = "walks on moderate terrain reliably"
- **Level 5.0** = "handles challenging real-world terrain most humans would notice"
- **Level 7.0** = "traverses terrain that would make a human careful where they step"
- **Level 9.0** = "handles extreme terrain at the physical limits of the hardware"

Our experts are currently at **5.2 (friction) and 5.7 (stairs)** — already past
the Mason baseline's 3.74 and into "challenging real-world terrain" territory.
The previous best terrain level across all our trials was 4.83 (AI-coached v8),
so these experts are already setting new records while simultaneously learning
sim-to-real robustness via the progressive wrapper.

---

## Key Numbers for Discussion

| Parameter | Our Value | Industry Standard | Source |
|-----------|-----------|-------------------|--------|
| Action delay | 40ms | 30-60ms | Spot SDK measured latency |
| Observation delay | 20ms | 10-30ms | IMU/encoder bus latency |
| Height scan dropout | 5% | 2-10% | RealSense D435 spec at 3m range |
| Friction range | 0.3-1.0 (Mason safe) | 0.1-1.5 | ETH Zurich ANYmal deployment |
| Mass randomization | ±2.5 kg (Mason safe) | ±10-20% | Standard DR for quadrupeds |
| Control frequency | 50 Hz (train) → 20 Hz (deploy) | 20-50 Hz | Spot SDK command rate |
| Network | [512, 256, 128] MLP | [256-1024] MLP or LSTM | ETH/MIT quadruped papers |

---

## Questions the PhD Students Might Ask

**Q: Why not use LSTM instead of MLP for temporal modeling?**
A: MLP inference is <10ms (critical for 20Hz control). LSTM adds ~5-10ms and complexity. We compensate with domain randomization — the policy learns to be robust without explicit system identification. ETH Zurich's production ANYmal deployment also uses MLP.

**Q: 40ms delay seems low — real systems can have 100ms+. Why not more?**
A: 40ms is conservative for Spot's wired internal bus. External WiFi control could add 50-100ms more. We start with the nominal case. If field testing shows higher latency, we retrain with 3-4 step delay (60-80ms).

**Q: Why not train with wrappers from scratch?**
A: We tried — the policy collapsed to standing still within 1000 iterations. The combined effect of delay + noise + wider DR creates too large a search space for PPO to navigate from random initialization. Fine-tuning from a walking policy provides the exploration scaffold.

**Q: How do you validate that sim S2R matches real S2R?**
A: Our deployment pipeline includes a `calibration.py` tool that measures real Spot latency, friction, and joint zero offsets. We compare measured values against training assumptions. If they differ significantly, we retrain with corrected parameters.

---

*AI2C Tech Capstone — MS for Autonomy, March 2026*
