# Parallelize 4_env_test Evaluation Pipeline

## Context

The 4_env_test eval takes **~27 hours** to run 800 episodes (100 episodes x 8 combos) because it's fundamentally **sequential and single-threaded**: 1 robot, 1 episode at a time. Each episode simulates 600s of physics at 500Hz, taking ~150s wallclock. Meanwhile, the H100 has ~55 GB of free VRAM (training uses 39 GB) that the eval isn't using.

**Goal:** Run all 8 combos in parallel instead of sequentially, cutting eval time from ~27 hours to **~4 hours** (limited by the slowest combo).

---

## Why It's Slow

The current `run_h100_eval.sh` runs a nested loop:
```
for POLICY in flat rough; do
    for ENV in friction grass boulder stairs; do
        # Launch Isaac Sim, run 100 episodes, exit -- SEQUENTIALLY
    done
done
```

Each combo launches a separate Isaac Sim instance (creates `SimulationApp`, builds the scene, runs 100 episodes, exits). They don't share state, so there's **no reason they can't run in parallel**.

Each eval instance uses ~1-2 GB VRAM. With 55 GB free, we can easily run all 8 in parallel.

---

## Architecture Analysis

### Current Eval Pipeline (Sequential)

**Entry point:** `src/run_capstone_eval.py`

```python
for ep_idx in range(args.num_episodes):   # 100 episodes, serial
    # Reset robot to spawn
    spot.robot.set_world_pose(position=SPAWN_POS, orientation=SPAWN_QUAT)
    # Step loop: 6000 control steps @ 50Hz = 600s sim time
    for step in range(MAX_CONTROL_STEPS):
        spot.forward(PHYSICS_DT, cmd)
        world.step(render=not headless)
```

**Per-episode breakdown:**
- 10 stabilization steps (0.2s)
- 6,000 control steps (600s simulation time)
- ~150s wallclock per episode
- 100 episodes x 8 combos = 800 episodes = ~33 hours

### Comparison: Training vs Eval

| | Training | Eval |
|---|---|---|
| Parallelism | 65,536 robots | 1 robot |
| Framework | Isaac Lab manager-based RL | Standalone Isaac Sim World |
| Episodes | Continuous (parallel resets) | Sequential loop |
| VRAM | 39 GB | ~1-2 GB per instance |

---

## The Plan

### Step 1: Create `run_h100_eval_parallel.sh`

New script that launches **8 separate screen sessions** (one per combo):

```bash
POLICIES=(flat rough)
ENVIRONMENTS=(friction grass boulder stairs)
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR="${PROJECT_DIR}/results/h100_eval_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

for POLICY in "${POLICIES[@]}"; do
    for ENV in "${ENVIRONMENTS[@]}"; do
        COMBO="${ENV}_${POLICY}"
        screen -dmS "eval_${COMBO}" bash -c "
            source /home/t2user/miniconda3/etc/profile.d/conda.sh
            conda activate env_isaaclab
            cd ~/IsaacLab
            export OMNI_KIT_ACCEPT_EULA=YES
            export PYTHONUNBUFFERED=1
            ./isaaclab.sh -p ~/4_env_test/src/run_capstone_eval.py --headless \
                --num_episodes 100 --policy ${POLICY} --env ${ENV} \
                --output_dir ${OUTPUT_DIR} \
                > ${OUTPUT_DIR}/${COMBO}.log 2>&1
        "
        echo "Launched: ${COMBO}"
    done
done
```

Then a **monitor loop** that:
1. Polls every 60s for running screen sessions
2. Reports progress (tail last line of each log)
3. When all 8 screens have exited, runs the reporter

### Step 2: Deploy

```bash
scp run_h100_eval_parallel.sh t2user@172.24.254.24:~/4_env_test/scripts/
ssh t2user@172.24.254.24 "sed -i 's/\r$//' ~/4_env_test/scripts/run_h100_eval_parallel.sh"
```

### Step 3: Launch

```bash
ssh t2user@172.24.254.24
# Kill existing sequential eval
screen -S eval4env -X quit 2>/dev/null
# Launch parallel eval
bash ~/4_env_test/scripts/run_h100_eval_parallel.sh
```

---

## Expected Performance

| Metric | Sequential (current) | Parallel (proposed) |
|---|---|---|
| Total eval time | ~27-33 hours | **~4-5 hours** |
| GPU memory (eval only) | ~2 GB | ~16 GB (8 x 2 GB) |
| GPU memory (with training) | 41 GB | ~55 GB (of 96 GB) |
| Screen sessions | 1 (eval4env) | 8 (eval_friction_flat, ...) |
| Report generation | Same | Same (runs after all complete) |

---

## VRAM Budget (with 65K env training running)

| Process | VRAM |
|---|---|
| Training (65K envs) | ~39 GB |
| 8x eval instances | ~16 GB |
| TensorBoard | negligible |
| **Total** | **~55 GB / 96 GB** |

Headroom: ~41 GB free. Safe.

---

## Risk Mitigation

1. **VRAM overflow**: If 8 instances is too many, fall back to 4 parallel (two batches). Still 2x faster.
2. **Isaac Sim GPU context conflicts**: Each `SimulationApp` creates its own CUDA context. Multiple contexts on one GPU is supported but watch for OOM.
3. **CPU bottleneck**: 8 parallel Isaac Sim instances may saturate CPU cores. The H100 server should have enough cores (typically 64+).
4. **Log file conflicts**: Each combo writes to its own JSONL/log file (keyed by combo name), so no conflicts.

---

## Files Involved

- **CREATE:** `4_env_test/scripts/run_h100_eval_parallel.sh`
- **NO CHANGES to:** `run_capstone_eval.py`, environment scripts, metrics, reporter
- This is purely an **orchestration change** -- the eval code itself is untouched

---

## Date: 2026-02-20
## Status: PLANNED (not yet implemented)
## Author: Claude Code + Gabriel Santiago
