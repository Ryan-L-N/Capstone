# Combined Nav + Loco Policy — How-To Guide

*For: Colby | Last updated: March 2026*

---

## How to Run

```bash
# Local smoke test (16 envs, 100 iterations — just to verify it starts)
bash Experiments/Colby/run_combined_nav_loco.sh --local

# Full training run on H100
bash Experiments/Colby/run_combined_nav_loco.sh --h100
```

The script automatically:
1. Installs Alex's nav package (`pip install -e`)
2. Validates Ryan's loco checkpoint exists
3. Launches training — no teammate files are touched

---

## 1. What Metrics of Success Should I See?

When training starts you'll get console output every iteration from RSL-RL.
The key numbers to watch, in order of importance:

### Primary — Is it learning to move?

| Metric | What it means | Good sign | Bad sign |
|--------|--------------|-----------|----------|
| `mean_reward` | Average total reward per episode | Climbing over time | Flat or dropping |
| `Nav/forward_distance` | Average meters traveled per 30s episode | Growing (target: 10m+) | Stuck near 0 |
| `Nav/survival_rate` | % of robots that don't fall during episode | Above 50%, trending up | Below 20% |
| `Curriculum/terrain_level` | Average difficulty level (1–6) | Slowly increasing | Stuck at 1 |

### Secondary — Is the training stable?

| Metric | What it means | Good sign | Bad sign |
|--------|--------------|-----------|----------|
| `value_loss` | How wrong the critic's predictions are | Decreasing | Spikes above 1000 |
| `Nav/flip_rate` | % of robots flipping over per episode | Below 30%, falling | Rising above 50% |
| `Nav/body_height` | Average body height above ground (m) | Near 0.42m | Below 0.25m (crawling) |
| `mean_episode_length` | How long episodes last (steps at 10 Hz) | Increasing toward 300 | Flat near 0 |

### TensorBoard (recommended view)
While training runs, open TensorBoard in a browser:
```bash
# On H100
http://172.24.254.24:6006

# Locally (run this in a separate terminal)
tensorboard --logdir Experiments/Alex/NAV_ALEX/logs/
```
This gives you live reward curves, terrain level, survival rate, and AI Coach decisions all in one place.

---

## 2. How Do I Tell If It's Performing Better?

There are three phases to expect — don't panic if early numbers look bad:

### Phase 1: First ~200 iterations — "Learning to walk"
- `survival_rate` is low (20–40%) — the loco policy is doing its job but the nav policy is sending random velocity commands
- `forward_distance` is small (0–3m) — robot is stumbling around
- `terrain_level` stays at 1 — it's failing too often to advance
- **This is normal.** The CNN is learning what the depth camera even means.

### Phase 2: ~200–2000 iterations — "Learning to move forward"
- `survival_rate` climbs above 50%
- `forward_distance` grows noticeably (3m → 8m → 15m+)
- `terrain_level` starts ticking above 1
- `mean_reward` rises steadily
- **This is the main learning phase.** The nav policy is discovering that moving forward = reward.

### Phase 3: 2000+ iterations — "Learning to navigate obstacles"
- `terrain_level` starts climbing toward 3–4
- The CNN is learning to use depth camera data to route around boulders and stairs
- Plateau here is normal — boulders and stairs are genuinely hard
- `forward_distance` in the 20–40m range = strong performance

### Rule of thumb
If after 500 iterations `forward_distance` is still below 2m and `survival_rate` is below 25%, something is broken (loco checkpoint mismatch, obs bug, etc.). Kill it and debug.

---

## 3. How Do I Create My Own Checkpoints?

### Automatic checkpoints (already built in)
RSL-RL saves checkpoints automatically to:
```
Experiments/Alex/NAV_ALEX/logs/spot_nav_explore_ppo/<timestamp>/
  ├── model_100.pt       ← every 100 iterations (--save_interval)
  ├── model_200.pt
  ├── model_500.pt
  └── model_final.pt     ← always saved when training ends or is interrupted
```

Change how often they save with `--save_interval`:
```bash
# Save every 50 iterations instead of 100
python scripts/rsl_rl/train_nav.py --save_interval 50 --loco_checkpoint ...
```

### Manual checkpoint (save right now mid-run)
`Ctrl+C` during training — the script catches the interrupt and saves `model_final.pt` before exiting. Safe to use.

### Resuming from a checkpoint
```bash
bash Experiments/Colby/run_combined_nav_loco.sh --local \
  # Add this flag to train_nav.py invocation inside the script:
  --resume Experiments/Alex/NAV_ALEX/logs/spot_nav_explore_ppo/<timestamp>/model_500.pt
```
Or edit `run_combined_nav_loco.sh` and add `--resume <path>` to the python command.

### Checkpoint file format
Each `.pt` file contains just the nav policy weights:
```python
{"model_state_dict": <ActorCriticCNN weights>}
```
The loco policy (Ryan's checkpoint) is always loaded separately and frozen — it's never saved into the nav checkpoint.

---

## 4. Capstone Presentation Section

> **Use this section for slides. Plain language, no jargon.**

---

### What We Built: A Two-Brain Control System for Spot

We combined two separately trained AI policies into a single control system. Think of it as giving Spot two specialized "brains" that work together in real time:

---

#### Brain 1 — The Locomotion Policy (the body)
**What it does:** Controls all 12 of Spot's joints 50 times per second. It keeps the robot stable, absorbs terrain impacts, and executes movement commands.

**How it was trained:** 33,200 iterations of PPO reinforcement learning across 8,192 parallel simulated Spot robots running on an H100 GPU. It learned on 12 types of terrain including stairs, rubble, and rough ground.

**Input:** Robot's own body state (velocity, joint positions, terrain scan) + a velocity command from Brain 2.
**Output:** Exact position target for each of 12 joints.
**Checkpoint:** `mason_hybrid_best_33200.pt` (Ryan's evaluation checkpoint)

---

#### Brain 2 — The Navigation Policy (the eyes)
**What it does:** Sees the world through a simulated 64×64 depth camera. Decides where to go and how fast at 10 times per second. Sends velocity commands down to Brain 1.

**How it is being trained:** PPO reinforcement learning across 2,048 parallel environments on the H100. The objective is simple: survive and move forward as far as possible in 30 seconds across 6 levels of terrain difficulty.

**Input:** 64×64 depth image (sees up to 30m ahead) + body velocity/orientation.
**Output:** Velocity command [forward speed, strafe, turn rate] → sent to Brain 1.
**Checkpoint:** Being trained now — saves every 100 iterations.

---

#### How They Work Together

```
What Spot sees (depth camera)
          ↓
  Navigation Policy (10 Hz)
  "I see a boulder — turn right"
          ↓
  Velocity Command [vx, vy, wz]
          ↓
  Locomotion Policy (50 Hz)       ← frozen, never retrained
  "Execute that command across this terrain"
          ↓
  12 Joint Targets → Spot moves
```

The key design choice: the locomotion policy is **frozen** while training the navigation policy. This means:
- We don't have to retrain walking from scratch every time we change the navigation goal
- The navigation policy can be swapped out (for a waypoint-following version, for instance) without touching the locomotion layer
- If a better locomotion policy is trained, it can be dropped in without retraining navigation

---

#### Why This Approach

| Traditional approach | Our approach |
|---|---|
| One monolithic policy controlling all 12 joints from raw camera input | Two specialized policies at different rates |
| Hard to train (12-dim joint space + high-dim vision = massive search space) | Navigation only needs to learn 3 numbers (vx, vy, wz) |
| Retrain everything when task changes | Swap only the layer that changed |
| Brittle — one failure mode affects everything | Loco failures and nav failures are diagnosable separately |

---

#### Current Status

| Component | Status | Key number |
|---|---|---|
| Locomotion policy | ✅ Trained | 33,200 iterations, terrain level 6 capable |
| Navigation policy | 🔄 Training now | 0 → 30,000 iterations planned |
| Combined system | ✅ Integrated | Running on H100 |
| Evaluation | Pending nav training | Target: 20m+ forward distance per episode |

---

#### What "Success" Looks Like

A successfully trained combined system will show Spot:
1. Walking upright and stably across all terrain types (locomotion policy, already working)
2. Using its depth camera to detect obstacles ahead (boulder, stairs, gap)
3. Routing around or over obstacles rather than walking into them
4. Sustaining forward progress across a 30-second episode without falling

The terrain curriculum means we'll have concrete difficulty benchmarks: the robot is graded on a 1–6 scale, where level 1 is flat ground and level 6 is 80cm boulders and 20cm stairs.
