# Dual-Brain Policy Training — Why It Works

---

## The Core Idea

Instead of training one giant policy to handle everything at once, split the problem into two specialized layers — one that handles **how to move**, one that handles **where to go**.

---

## The Two Layers

### Low-level (Body) — Locomotion
- Controls 12 joints at **50 Hz**
- Input: proprioception (235-dim) + velocity command
- Trained by Ryan, **frozen forever** — 286,604 parameters, never updated again
- Only answers: *"given a direction, how do I physically execute it?"*

### High-level (Eyes) — Navigation
- Reads a **64×64 depth camera** + 12-dim proprioception = 4,108 inputs
- Runs at **10 Hz** — outputs just 3 numbers: [vx, vy, ωz]
- **489,799 parameters** being actively trained
- Only answers: *"given what I see, where should I go?"*

---

## Why Not One Big Policy?

| End-to-End | Dual-Brain |
|---|---|
| Learns actuator physics AND perception simultaneously | Each layer solves one problem |
| Enormous, unstable search space | Nav policy outputs only **3 values** |
| Change the task → retrain everything | Swap only the layer relevant to the new task |
| Can't tell if failure is perception or control | Failures isolate to one brain |

---

## What the Smoke Test Proved

- **489,799-parameter CNN policy** trained through 100 iterations without NaN or crashes
- Value loss stable at ~13.7 (expected — random policy, no learned behavior yet)
- **92% of episodes ended in bad orientation** — expected at iteration 0 with a random nav policy
- Episode length: 7.5 steps — Spot falls fast with random velocity commands
- Full pipeline validated: Isaac Sim → depth camera → CNN → frozen loco → joints → physics

---

## Key Architecture Numbers

| Component | Detail |
|---|---|
| Depth image | 64×64 = 4,096 pixels, flattened |
| Proprioception | 12-dim (lin_vel, ang_vel, gravity, prev_action) |
| CNN encoder | 3 conv layers → 128-dim feature vector |
| Actor MLP | [256, 128] → 3-dim velocity command |
| Critic MLP | [256, 128] → 1-dim value estimate |
| Frozen loco | [512, 256, 128] MLP, 286K params, Ryan's checkpoint |
| Nav policy | CNN + MLP, 490K params, being trained |
| Physics rate | 500 Hz |
| Loco rate | 50 Hz (5 physics steps per loco step) |
| Nav rate | 10 Hz (5 loco steps per nav step) |

---

## The Signal Chain

```
64×64 depth image  +  proprioception (12)
           ↓
     CNN encoder  →  128-dim features
           ↓
     Actor MLP    →  [vx, vy, ωz]   ← 3 numbers
           ↓
  Frozen Loco MLP →  12 joint targets
           ↓
     Spot moves
```

Two layers, two timescales, one robot.

---

## Key Benefits

- **Clean training signal** — frozen locomotion layer means the nav policy isn't fighting a moving target below it
- **Fast iteration** — nav experiments are cheap because the expensive loco layer is never retouched
- **Modularity** — drop in a better loco checkpoint without retraining nav, and vice versa
- **Reusability** — same frozen loco weights can serve waypoint-following, obstacle-avoidance, or any other nav policy you put on top
- **Biological parallel** — mirrors how the brain separates high-level intent (cortex) from low-level motor execution (cerebellum/spinal cord)
