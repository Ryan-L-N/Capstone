# Combined Nav + Loco Policy — How-To Guide

*For: Colby | Last updated: March 2026*

---

## Prerequisites — Read Before Running Anything

### Note: Isaac Lab

As of March 2026, `isaacSim_env` has Isaac Sim 5.1.0 and PyTorch but not Isaac Lab.
**`install_prerequisites.sh` will attempt to install it automatically** from the NVIDIA pip index.
If that fails (network/version issue), it prints the manual install command and continues setting up everything else.

The H100 `env_isaaclab` conda environment already has Isaac Lab — the H100 path is fully self-sufficient.

---

### Virtual Environment

**All packages must be installed inside your Isaac Sim virtual environment.**
The install script enforces this — it will refuse to run if no venv is active.

```bash
# Activate your venv first (if not already active)
source C:/Users/Colby/Documents/AI2C/Class/Capstone/Capstone/isaacSim_env/Scripts/activate  # Windows/Git Bash

# Then run the one-time setup
bash Experiments/Colby/CombinedPolicyTraining/install_prerequisites.sh --local   # local
bash Experiments/Colby/CombinedPolicyTraining/install_prerequisites.sh --h100    # H100
```

The install script sets up (all inside the venv):
- `nav_locomotion` — Alex's nav package (editable install, no files modified in his dir)
- `isaaclab` — installed from `isaacSim_env/isaaclab_src/source/isaaclab` (editable)
- `isaaclab_rl` — Isaac Lab's RSL-RL wrapper, installed from `isaacSim_env/isaaclab_src/source/isaaclab_rl` (editable). Required by `SpotNavPPORunnerCfg`.
- `tensorboard` — training metrics viewer
- `gymnasium` — RL environment interface
- `rsl-rl` — PPO runner (GitHub source)
- `h5py` — required by isaaclab's dataset utilities
- `torch (cu128)` — CUDA-enabled torch. Isaac Sim ships CPU-only by default; the script installs the CUDA build automatically if CUDA is not available.

**Known gotchas handled automatically by the install script:**
- `h5py` — isaaclab fails to import without it (`ModuleNotFoundError: No module named 'h5py'`)
- `isaaclab_rl` — required for config classes; missing causes first-pass `KeyError: 'class_name'`
- `torch CUDA` — Isaac Sim installs CPU-only torch; training requires the CUDA build
- **DLL load order (Windows)** — `train_combined.py` imports torch before `AppLauncher` to prevent Isaac Sim's CUDA 11 extscache DLLs from conflicting with torch's CUDA 12 DLLs (`WinError 1114 / c10.dll`). Do not move this import.
- **rsl_rl 5.0.1 API break** — rsl_rl 5.0.1 replaced the old combined `ActorCritic` class with separate `actor`/`critic` `MLPModel` objects. Alex's `ActorCriticCNN` uses the old API. Fixed via `cnn_compat.py` (adapter classes) + manual `runner_cfg_dict` in `train_combined.py`. Do not revert to `class_to_dict(SpotNavPPORunnerCfg)`.

Currently missing from the venv (handled by install script above):
`rsl-rl`, `gymnasium`, `tensorboard`

---

## How to Run

```bash
# Local smoke test (16 envs, 100 iterations — just to verify it starts)
bash Experiments/Colby/CombinedPolicyTraining/run_combined_nav_loco.sh --local

# Full training run on H100
bash Experiments/Colby/CombinedPolicyTraining/run_combined_nav_loco.sh --h100
```

The script automatically:
1. Installs Alex's nav package (`pip install -e`) — no files in his directory are modified
2. Validates Ryan's loco checkpoint exists
3. Launches `train_combined.py` — all output stays in `Experiments/Colby/CombinedPolicyTraining/`

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

# Locally (run this in a separate terminal, inside your venv)
tensorboard --logdir Experiments/Colby/CombinedPolicyTraining/logs/

# Or use the live terminal dashboard (no browser needed)
python Experiments/Colby/CombinedPolicyTraining/watch_training.py
```
This gives you live reward curves, terrain level, and survival rate all in one place.

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
`train_combined.py` saves checkpoints to **Colby's folder only**:
```
Experiments/Colby/CombinedPolicyTraining/logs/spot_nav_explore_ppo/<timestamp>/
  ├── model_100.pt       ← every 100 iterations (--save_interval)
  ├── model_200.pt
  ├── model_500.pt
  └── model_final.pt     ← always saved when training ends or is interrupted
```
No teammate directories are written to.

Change how often they save with `--save_interval`:
```bash
# Edit run_combined_nav_loco.sh and add --save_interval 50 to the python call,
# or invoke train_combined.py directly:
python Experiments/Colby/CombinedPolicyTraining/train_combined.py \
    --save_interval 50 \
    --loco_checkpoint Experiments/Ryan/checkpoints/mason_hybrid_best_33200.pt \
    --headless --num_envs 16 --max_iterations 100
```

### Manual checkpoint (save right now mid-run)
`Ctrl+C` during training — the script catches the interrupt and saves `model_final.pt` before exiting. Safe to use.

### Resuming from a checkpoint
Pass `--resume` directly to `train_combined.py` (or add it to the python call in `run_combined_nav_loco.sh`):
```bash
python Experiments/Colby/CombinedPolicyTraining/train_combined.py \
    --loco_checkpoint Experiments/Ryan/checkpoints/mason_hybrid_best_33200.pt \
    --resume Experiments/Colby/CombinedPolicyTraining/logs/spot_nav_explore_ppo/<timestamp>/model_500.pt \
    --headless --num_envs 2048 --max_iterations 30000
```

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
**Checkpoint:** Being trained now — saves every 100 iterations to `Experiments/Colby/CombinedPolicyTraining/logs/`.

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
| Locomotion policy | Trained | 33,200 iterations, terrain level 6 capable |
| Navigation policy | Training now | 0 → 30,000 iterations planned |
| Combined system | Integrated | Ready to run on H100 |
| Evaluation | Pending nav training | Target: 20m+ forward distance per episode |

---

#### What "Success" Looks Like

A successfully trained combined system will show Spot:
1. Walking upright and stably across all terrain types (locomotion policy, already working)
2. Using its depth camera to detect obstacles ahead (boulder, stairs, gap)
3. Routing around or over obstacles rather than walking into them
4. Sustaining forward progress across a 30-second episode without falling

The terrain curriculum means we'll have concrete difficulty benchmarks: the robot is graded on a 1–6 scale, where level 1 is flat ground and level 6 is 80cm boulders and 20cm stairs.
