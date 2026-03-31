# Policy Architecture & Team Work Summary

*Last updated: March 2026*

---

## What We're Actually Building

A hierarchical control stack for Spot:

```
Depth Camera (64×64)  →  CNN Encoder (128-dim)
                               |
                    + Proprioception (12-dim)
                               |
              [Alex's Nav Policy MLP, 10 Hz]   ← BEING TRAINED
                               |
                  Velocity Command [vx, vy, wz]
                               |
          [Frozen Loco Policy (RSL-RL), 50 Hz]  ← FROZEN
                               |
                  12 Joint Targets → Spot Robot
```

The nav policy learns to steer using depth camera data.
The loco policy handles actual joint control and is frozen during nav training.

---

## The Two Policies

### Navigation Policy — Alex (NAV_ALEX)
- **Location:** `Experiments/Alex/NAV_ALEX/`
- **Architecture:** CNN encoder (3 conv layers) + proprioception → MLP [256, 128] → [vx, vy, wz]
- **Obs:** 64×64 depth image (4096) + proprioception (12) = 4108 total
- **Output:** Velocity commands [vx, vy, wz] at 10 Hz
- **Framework:** Isaac Lab + RSL-RL (full extension, `pip install -e source/nav_locomotion/`)
- **Objective:** Maximize forward terrain traversal while surviving (exploration mode, no waypoints)
- **Status:** **No trained checkpoint yet** — training has not been run
- **Train script:** `scripts/rsl_rl/train_nav.py --loco_checkpoint <path>`
- **Referenced loco checkpoint:** `ai_coached_v8_10600.pt` — does NOT exist in the repo

### Locomotion Policy — Ryan's checkpoint (trains under Alex's nav)
- **Checkpoint:** `Experiments/Ryan/checkpoints/mason_hybrid_best_33200.pt`
- **Architecture:** `[512, 256, 128]` MLP, ELU activations (auto-detected by FrozenLocoPolicy)
- **Obs:** 235-dim (base_lin_vel + base_ang_vel + projected_gravity + velocity_commands + joint_pos + joint_vel + actions + height_scan)
- **Velocity command injection:** indices [9, 10, 11] in the 235-dim obs vector
- **Output:** 12 joint position targets at 50 Hz
- **Status:** Trained, verified compatible with Alex's `FrozenLocoPolicy` wrapper
- **Why Ryan's:** The checkpoint Alex's README points to (`ai_coached_v8_10600.pt`) doesn't exist in the repo. Ryan's mason_hybrid checkpoint is the same format and architecture — exact drop-in replacement.

---

## What Cole Has Been Doing (Parallel Nav System)

Cole built a **completely independent navigation system** in parallel with Alex. Same concept, different implementation.

### Cole VS2 (`Experiments/Cole/RL_Folder_VS2/`)
- Custom PPO training loop (no Isaac Lab/RSL-RL)
- Nav policy: 32-dim obs (velocity + heading + waypoint + 16 raycasts + stage one-hot)
- Output: [vx, vy, omega] to `SpotFlatTerrainPolicy` (built-in Isaac Sim, not RSL-RL trained)
- 8-stage curriculum: random walk → 5m → 10m → 20m → 40m waypoints → obstacles
- **Has a real trained checkpoint:** `checkpoints/run_6_aggressive/final_model.pt`
- Trained through Stage 5 (40m waypoint chains), 6 total runs with detailed performance reports

### Cole VS3 (`Experiments/Cole/RL_FOLDER_VS3/`)
- Upgraded version of VS2: 75-dim obs (adds IMU, foot contacts, multi-layer raycasts)
- 3 parallel training runs planned (conservative / moderate / aggressive LR)
- Checkpoint dirs set up (`VS3_checkpoints/`) but **no trained checkpoint yet**

### Key Differences: Cole vs Alex

| | Cole (VS2/VS3) | Alex (NAV_ALEX) |
|---|---|---|
| Sensor | Simple raycasts | 64×64 depth camera CNN |
| Loco base | SpotFlatTerrainPolicy (built-in) | Trained RSL-RL policy (frozen) |
| Objective | Waypoint capture curriculum | Terrain traversal/exploration |
| Framework | Custom PPO from scratch | Isaac Lab + RSL-RL |
| Trained checkpoint | **Yes (VS2)** | **No** |

Cole's work is simpler but more complete. Alex's is more ambitious but untrained.
Cole's waypoint curriculum structure could be valuable for future task-layer training on top of Alex's system.

---

## How to Run the Combined System

All scripts live in `Experiments/Colby/CombinedPolicyTraining/`. No teammate files are modified.

### Step 0 — Virtual Environment (required)
All packages install into the Isaac Sim venv. The install script will refuse to run without one active.
```bash
# Activate your venv first
source <path/to/isaacSim_env>/Scripts/activate   # Windows / Git Bash
source <path/to/isaacSim_env>/bin/activate        # Linux / Mac
```

### Step 1 — One-time install
```bash
bash Experiments/Colby/CombinedPolicyTraining/install_prerequisites.sh --local   # local
bash Experiments/Colby/CombinedPolicyTraining/install_prerequisites.sh --h100    # H100
```

Installs into the active venv only:
- `nav_locomotion` (Alex's package, editable install — his files unchanged)
- `anthropic` (AI coach)
- `tensorboard` (metrics)
- `gymnasium` (RL env interface)
- `rsl-rl` (PPO runner)

### Step 2 — Train
```bash
# Local smoke test (16 envs, 100 iterations)
bash Experiments/Colby/CombinedPolicyTraining/run_combined_nav_loco.sh --local

# H100 full run (2048 envs, 30000 iterations, with AI coach)
bash Experiments/Colby/CombinedPolicyTraining/run_combined_nav_loco.sh --h100
```

### What the training script does
1. Resolves paths relative to repo root
2. Validates Ryan's loco checkpoint exists
3. `pip install -e Experiments/Alex/NAV_ALEX/source/nav_locomotion/` (idempotent)
4. Launches `train_combined.py` — Colby's own script, saves all output to `CombinedPolicyTraining/logs/`

### Checkpoint output (Colby's folder only)
```
Experiments/Colby/CombinedPolicyTraining/logs/spot_nav_explore_ppo/<timestamp>/
  ├── model_100.pt
  ├── model_200.pt
  └── model_final.pt
```

### Monitor training
```bash
# Terminal dashboard
python Experiments/Colby/CombinedPolicyTraining/watch_training.py

# Browser (H100)
http://172.24.254.24:6006

# Browser (local)
tensorboard --logdir Experiments/Colby/CombinedPolicyTraining/logs/
```

---

## Status & Next Steps

**Done:**
- Combined training pipeline built (`train_combined.py`) — imports Alex's modules as a library, zero teammate files touched
- Install script handles all dependencies including Isaac Lab auto-install attempt
- All checkpoints and logs save to `Experiments/Colby/CombinedPolicyTraining/logs/` only
- H100 path (`env_isaaclab`) is self-sufficient — ready to run

**Remaining:**
- [ ] Run `install_prerequisites.sh --h100` on the H100 to confirm all imports pass
- [ ] Start NAV-001 training run — use `--no_coach` for the first smoke test iteration, then enable for full 30K run
- [ ] Fill in RUN-004 card in `Results.md` as training progresses
