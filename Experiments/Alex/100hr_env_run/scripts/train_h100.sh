#!/usr/bin/env bash
# =============================================================================
# 100-Hour Multi-Terrain Robust Locomotion Training — H100 NVL 96GB
# =============================================================================
#
# Launch full training run on H100 server (ai2ct2) in screen sessions.
# Expected: ~100 hours, ~39 billion timesteps, ~60,000 iterations.
#
# Sets up TWO screen sessions:
#   1. "train100hr" — the training process (persists after SSH disconnect)
#   2. "tb100hr"    — TensorBoard server
#
# Usage:
#   ssh ai2ct2
#   bash /path/to/train_h100.sh
#
# Then:
#   screen -r train100hr   # Attach to training output
#   screen -r tb100hr      # Attach to TensorBoard
#   Ctrl+A, D              # Detach from screen
#
# TensorBoard: http://ai2ct2:6006
# Live log:    tail -f ~/100hr_training.log
#
# Created for AI2C Tech Capstone — MS for Autonomy, February 2026
# =============================================================================

set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_100hr.py"
ISAACLAB_DIR="${HOME}/IsaacLab"
LOG_DIR="${ISAACLAB_DIR}/logs/rsl_rl/spot_100hr_robust"

NUM_ENVS=65536
MAX_ITERATIONS=30000
SEED=42
LR_MAX=1e-3
LR_MIN=1e-5
WARMUP_ITERS=3000

# --- Pre-flight checks ---
echo "============================================================"
echo "  100hr Multi-Terrain Robust Training — Pre-flight"
echo "============================================================"
echo "  Train script:   ${TRAIN_SCRIPT}"
echo "  IsaacLab dir:   ${ISAACLAB_DIR}"
echo "  Num envs:       ${NUM_ENVS}"
echo "  Max iterations: ${MAX_ITERATIONS}"
echo "  LR schedule:    ${LR_MAX} -> ${LR_MIN} (warmup ${WARMUP_ITERS})"
echo "  Seed:           ${SEED}"
echo "============================================================"

if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "[ERROR] Training script not found: ${TRAIN_SCRIPT}"
    echo "  Did you copy the 100hr_env_run/ folder to the server?"
    exit 1
fi

if [ ! -d "${ISAACLAB_DIR}" ]; then
    echo "[ERROR] IsaacLab directory not found: ${ISAACLAB_DIR}"
    exit 1
fi

echo ""
echo "[INFO] GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
echo ""

# --- Kill any existing screen sessions with same names ---
screen -S train100hr -X quit 2>/dev/null || true
screen -S tb100hr -X quit 2>/dev/null || true

# --- Launch TensorBoard in a screen session ---
echo "[INFO] Starting TensorBoard on port 6006..."
mkdir -p "${LOG_DIR}"
screen -dmS tb100hr bash -c "
    source /home/t2user/miniconda3/etc/profile.d/conda.sh
    conda activate env_isaaclab
    echo '[TB] TensorBoard starting...'
    echo '[TB] Log dir: ${LOG_DIR}'
    echo '[TB] URL: http://\$(hostname):6006'
    tensorboard --logdir '${LOG_DIR}' --bind_all --port 6006 2>&1
"
echo "[INFO] TensorBoard screen session: 'tb100hr'"
echo ""

# --- Launch training in a screen session ---
echo "[INFO] Starting training in screen session 'train100hr'..."
screen -dmS train100hr bash -c "
    source /home/t2user/miniconda3/etc/profile.d/conda.sh
    conda activate env_isaaclab
    cd '${ISAACLAB_DIR}'
    export OMNI_KIT_ACCEPT_EULA=YES
    export PYTHONUNBUFFERED=1

    echo '============================================================'
    echo '  65K ENV FULL RUN -- Starting at \$(date)'
    echo '  Expected: ~5-6 days, ~63B timesteps, resuming from 1000-iter test'
    echo '============================================================'

    ./isaaclab.sh -p '${TRAIN_SCRIPT}' \
        --headless \
        --num_envs ${NUM_ENVS} \
        --max_iterations ${MAX_ITERATIONS} \
        --seed ${SEED} \
        --lr_max ${LR_MAX} \
        --lr_min ${LR_MIN} \
        --warmup_iters ${WARMUP_ITERS} \
        --resume \
        --load_run 2026-02-20_17-53-31_multi_terrain_v1 \
        2>&1 | tee ~/100hr_training.log

    echo ''
    echo '============================================================'
    echo '  TRAINING COMPLETE at \$(date)'
    echo '============================================================'
    echo 'Press Enter to close this screen session...'
    read
"

echo ""
echo "============================================================"
echo "  LAUNCHED SUCCESSFULLY"
echo "============================================================"
echo ""
echo "  Training:    screen -r train100hr"
echo "  TensorBoard: screen -r tb100hr"
echo "  Live log:    tail -f ~/100hr_training.log"
echo "  TB URL:      http://$(hostname):6006"
echo ""
echo "  Detach from screen: Ctrl+A, D"
echo "  List screens:       screen -ls"
echo "  Kill training:      screen -S train100hr -X quit"
echo "============================================================"
