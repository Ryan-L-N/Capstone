# Dual-Brain Policy Training — Why It Works

---

## The Core Idea

Instead of training one giant policy to handle everything at once, you split the problem into two specialized layers — one that handles **how to move**, and one that handles **where to go**. Each brain solves a much simpler problem than the combined one would be.

---

## The Two Layers

### Low-level (Body) — Locomotion
- Runs fast, controls raw actuators
- Only needs to answer: *"given a direction command, how do I physically execute that?"*
- Trained once, then frozen — never changes again

### High-level (Eyes) — Navigation
- Runs slower, uses perception (camera, sensors)
- Only needs to answer: *"given what I see, what direction should I go?"*
- This is the layer being actively trained

---

## Why Train Them This Way vs. End-to-End

| Training one big policy | Dual-brain approach |
|---|---|
| Must learn actuator control AND perception simultaneously | Each layer solves one problem |
| Enormous search space — millions of parameters fighting each other | Navigation only learns 3 output values |
| Any task change means retraining everything from scratch | Swap only the layer relevant to the new task |
| Hard to diagnose failures — is it perception or control? | Failures are isolated to one brain |

---

## Key Benefits

- **Training efficiency** — The lower layer is frozen, so the upper layer's training signal is clean and stable. It isn't fighting a moving target at the bottom.
- **Modularity** — A better lower-layer policy can be dropped in without retraining the upper layer, and vice versa.
- **Faster iteration** — You can run many navigation experiments cheaply because you never touch the expensive, already-trained locomotion layer.
- **Cleaner reward signal** — The navigation policy only has to learn high-level behavior. It doesn't have to figure out joint physics at the same time.
- **Reusability** — The locomotion layer is task-agnostic. The same frozen weights can serve a waypoint-following nav policy, an obstacle-avoidance nav policy, or anything else you put on top.

---

## The Hierarchy in Practice

```
Perception (camera/sensors)
        ↓
High-level policy  →  outputs a direction command
        ↓
Low-level policy   →  translates command to physical actions
        ↓
Robot moves
```

The two layers run at different rates — high-level thinks slowly (uses perception), low-level reacts quickly (uses proprioception). This mirrors how biological motor control works.
