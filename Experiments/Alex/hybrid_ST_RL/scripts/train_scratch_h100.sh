#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# Attempt 5: H100 Production — From-Scratch Training with Terrain Curriculum
# ─────────────────────────────────────────────────────────────────────────
#
# Trains a fresh 235-dim policy from random initialization on the H100.
# Terrain curriculum starts flat and gradually introduces harder terrains.
# Progressive DR expands friction/push/force over 15K iterations.
#
# Expected: ~48 hours for 15K iterations with 16,384 envs.
#
# Usage:
#   ssh t2user@172.24.254.24
#   cd ~/IsaacLab
#   screen -S scratch
#   bash ~/hybrid_ST_RL/scripts/train_scratch_h100.sh 2>&1 | tee ~/scratch_attempt6_$(date +%F_%H-%M-%S).log
#
# TensorBoard (in a separate screen):
#   screen -S tb_scratch
#   conda activate env_isaaclab
#   tensorboard --logdir ~/IsaacLab/logs/rsl_rl/spot_scratch_terrain --port 6006 --bind_all
#
# Monitor:
#   screen -r scratch          # Attach to training
#   tail -f ~/scratch_*.log    # Watch log file
#   http://172.24.254.24:6006  # TensorBoard
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail

# Accept EULA
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

# ── Configuration ───────────────────────────────────────────────────────
NUM_ENVS=16384
MAX_ITERS=15000
DR_EXPANSION=15000
SEED=42

TRAIN_SCRIPT="${HOME}/hybrid_ST_RL/train_from_scratch.py"
# ────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "  Attempt 5 — H100 PRODUCTION"
echo "  Mode:       FROM SCRATCH (no checkpoint)"
echo "  Envs:       ${NUM_ENVS}"
echo "  Iterations: ${MAX_ITERS}"
echo "  DR expand:  ${DR_EXPANSION} iters"
echo "  Script:     ${TRAIN_SCRIPT}"
echo "  Time:       $(date)"
echo "============================================================"

# Activate conda env (source profile first for non-interactive shells)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# Launch training
./isaaclab.sh -p "${TRAIN_SCRIPT}" --headless \
    --num_envs ${NUM_ENVS} \
    --max_iterations ${MAX_ITERS} \
    --dr_expansion_iters ${DR_EXPANSION} \
    --seed ${SEED}

echo ""
echo "============================================================"
echo "  Attempt 5 training complete at $(date)"
echo "============================================================"
