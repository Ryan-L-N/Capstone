# 4-Environment Capstone Evaluation

Comparative evaluation of Boston Dynamics Spot locomotion policies across 4 test environments.

| Policy | Observations | Checkpoint | Training |
|--------|-------------|------------|----------|
| **Flat Baseline** (NVIDIA) | 48-dim proprioceptive | Pre-trained | Isaac Lab default |
| **Rough Terrain** (Custom) | 235-dim (48 proprio + 187 height scan) | `model_29999.pt` | 48h on H100, 30K iterations |

| Environment | Challenge | Zones |
|-------------|-----------|-------|
| **Friction** | Decreasing friction (sandpaper -> oil) | 5 zones, 10m each |
| **Grass** | Proxy stalks + drag forces | 5 zones, increasing density |
| **Boulder** | Mixed polyhedra (D8/D10/D12/D20) | 5 zones, increasing size |
| **Stairs** | Ascending steps | 5 zones, 3cm -> 23cm step height |

---

## Prerequisites

- Isaac Sim 5.1.0 + Isaac Lab 2.3.0
- Conda environment: `isaaclab311` (Python 3.11)
- H100 NVL server access (ai2ct2) for production runs
- Policy checkpoints (see `checkpoints/README.md`)

## Setup

```bash
# 1. Clone and navigate
cd Experiments/Alex/4_env_test/

# 2. Place checkpoints
cp /path/to/model_29999.pt checkpoints/
# See checkpoints/README.md for flat baseline location

# 3. Fix line endings (if uploading from Windows to Linux)
sed -i "s/\r$//" scripts/*.sh
chmod +x scripts/*.sh
```

## Workflow

### Step 1: Debug (5 episodes, <1 minute)

Run 5 quick episodes to verify everything loads and runs:

```bash
bash scripts/debug_5iter.sh
```

Check output in `results/debug/`. If issues found, document in `LESSONS_LEARNED.md` and fix.
Repeat up to 5 times until clean.

### Step 2: Production Run (8,000 episodes, ~1-1.5 hours)

```bash
bash scripts/run_full_eval.sh
```

Runs all 8 combinations (2 policies x 4 environments x 1000 episodes each).
Output: `results/{env}_{policy}_episodes.jsonl`

### Step 3: Rendered Visualization (80 episodes, ~3.5 hours)

```bash
bash scripts/run_rendered_viz.sh
```

Captures video + keyframe PNGs for 10 episodes x 2 policies x 4 environments.
Output: `results/rendered/`

### Step 4: Manual Teleop Walkthrough

```bash
bash scripts/run_teleop.sh friction    # or: grass, boulder, stairs
```

Xbox controller + keyboard. Press RB to switch gaits (FLAT/ROUGH), LB for FPV camera.

### Step 5: Analyze Results

```bash
python src/metrics/reporter.py --input results/
```

Generates `results/summary.csv` and plots in `results/plots/`.

---

## Folder Structure

```
4_env_test/
├── capstone_test.md          # Master test plan (full specification)
├── LESSONS_LEARNED.md        # Debug journal + critical deployment notes
├── README.md                 # This file
├── .gitignore
├── scripts/                  # Shell scripts for H100 execution
│   ├── debug_5iter.sh
│   ├── run_full_eval.sh
│   ├── run_rendered_viz.sh
│   └── run_teleop.sh
├── src/                      # Python source code
│   ├── run_capstone_eval.py  # Main headless/rendered entry point
│   ├── run_capstone_teleop.py
│   ├── envs/                 # Environment builders
│   ├── navigation/           # Waypoint follower
│   ├── metrics/              # Data collection & analysis
│   └── configs/              # Evaluation configuration
├── results/                  # Output (gitignored)
│   ├── debug/
│   └── plots/
└── checkpoints/              # Policy checkpoints
```

## Key References

| File | Location |
|------|----------|
| Master test plan | `capstone_test.md` |
| Rough env config | `../ARL_DELIVERY/05_Training_Package/isaac_lab_spot_configs/rough_env_cfg.py` |
| Grass physics | `../ARL_DELIVERY/04_Teleop_System/grass_physics_config.py` |
| Obstacle course | `../ARL_DELIVERY/02_Obstacle_Course/spot_obstacle_course.py` |
| Teleop system | `../ARL_DELIVERY/04_Teleop_System/spot_teleop.py` |
| Prior lessons | `../ARL_DELIVERY/08_Lessons_Learned/` |
