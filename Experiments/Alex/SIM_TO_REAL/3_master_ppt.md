# Slide 1: 3-Master Expert Training Strategy

## From One Policy to Three Specialists

**Problem:** A single RL policy can't master every terrain. Our generalist Spot walks perfectly on flat ground but struggles on stairs (27.6m/50m) and boulders (22.6m/50m).

**Solution:** Train 3 terrain-master policies in parallel, each fine-tuned from our best checkpoint for its terrain family.

### The 3 Masters

| Master | Terrain Focus | Starting Policy | Key Training Signal |
|--------|--------------|-----------------|-------------------|
| **Flat Master** | Friction + Grass (ice to oil, light to dense brush) | Distilled Generalist (100% friction completion) | Maximum stability: smoothest gait, zero slipping, rock-solid standing |
| **Stair Master** | Stairs Up + Down (3cm ramps to 23cm commercial stairs) | Obstacle Parkour Expert (terrain level 5.79) | Controlled knee-bend stepping, straight-ahead traversal, no diagonal drift |
| **Boulder Master** | Irregular Rock Fields (gravel to 50cm boulders) | Boulder Expert (22.6m eval progress) | High foot clearance over rocks, slip resistance on uneven contact surfaces |

### Parallel Training on H100 GPU

- All 3 masters train simultaneously (40 GB / 96 GB VRAM used)
- Each has a live web dashboard for real-time reward tuning
- 5,000 iterations per master (~8 hours each)
- Staged warmup: freeze actor, calibrate critic, then gradually unfreeze layers

### Next Step: Distill Into One

After training, the 3 masters are distilled into a **single generalist policy** using a learned attention router that automatically selects the right expert behavior based on what the robot sees through its height scanner.

---

# Slide 2: Bridging the Sim-to-Real Gap

## Training With Intentional Imperfection

**The Problem:** Policies trained in perfect simulation fail on real hardware because the real world is messy -- sensors lag, motors are imprecise, surfaces are unpredictable.

**Our Solution:** We corrupt the simulation during training so the policy learns to handle real-world imperfections from day one.

### 7 Sim-to-Real Mitigations (Active During ALL Training)

| What We Corrupt | How | Why It Matters |
|-----------------|-----|---------------|
| **Motor Response** | Add 40ms delay to every action | Real actuators don't respond instantly |
| **Sensor Readings** | Add 20ms delay + Gaussian noise to observations | Real IMU/joint encoders are noisy and lagged |
| **Terrain Perception** | Randomly zero out 5% of height scan rays | Real depth cameras have dropouts and occlusions |
| **IMU Drift** | Simulate slow-drifting bias on gyroscope | Real IMUs accumulate error over time |
| **Robot Weight** | Randomize mass by +/- 5 kg each episode | Real payload varies; battery weight shifts |
| **Surface Grip** | Randomize friction 0.3 to 1.0 every reset | Real surfaces range from ice to rubber |
| **External Pushes** | Apply random 3N force every 7-12 seconds | Wind, bumps, and operator contact happen |

### The Result

A policy that has **never seen a perfect simulation** -- so when it encounters the imperfect real world, it already knows how to cope. This is the core of sim-to-real transfer.

### Training to Deployment Pipeline

```
[3 Master Experts] --> [Distilled Student Policy] --> [ONNX Export] --> [Boston Dynamics Spot SDK]
   (50 Hz sim)            (20 Hz, all terrain)        (optimized)        (20 Hz real-time)
```

The final distilled policy runs at 20 Hz to match the real Spot SDK control rate, with a safety layer monitoring joint limits, orientation, and torque before every command reaches the robot.
