#!/usr/bin/env bash
# =============================================================================
# H100 Test Run — 1000 iterations with screen + TensorBoard
# =============================================================================
#
# Runs a short test (1000 iterations, ~20 min on H100) to verify:
#   - All 12 terrain types generate correctly
#   - Reward terms compute without NaN
#   - GPU memory fits within 96GB
#   - TensorBoard logging works
#   - Throughput measurement (target: >50K steps/sec)
#
# This script sets up TWO screen sessions:
#   1. "train100hr" — the training process
#   2. "tb100hr"    — TensorBoard server
#
# Usage:
#   ssh ai2ct2
#   cd ~/IsaacLab
#   bash /path/to/test_h100.sh
#
# Then:
#   screen -r train100hr   # Attach to training output
#   screen -r tb100hr      # Attach to TensorBoard
#   Ctrl+A, D              # Detach from screen
#
# TensorBoard will be at: http://ai2ct2:6006
#
# Created for AI2C Tech Capstone — MS for Autonomy, February 2026
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_100hr.py"
ISAACLAB_DIR="${HOME}/IsaacLab"
LOG_DIR="${ISAACLAB_DIR}/logs/rsl_rl/spot_100hr_robust"

# --- Test run parameters ---
NUM_ENVS=20480
MAX_ITERATIONS=1000
SEED=42
LR_MAX=1e-3
LR_MIN=1e-5
WARMUP_ITERS=100

echo "============================================================"
echo "  H100 TEST RUN — 1000 iterations"
echo "============================================================"
echo "  Train script:   ${TRAIN_SCRIPT}"
echo "  IsaacLab dir:   ${ISAACLAB_DIR}"
echo "  Num envs:       ${NUM_ENVS}"
echo "  Max iterations: ${MAX_ITERATIONS}"
echo "  LR schedule:    ${LR_MAX} -> ${LR_MIN} (warmup ${WARMUP_ITERS})"
echo "  Log dir:        ${LOG_DIR}"
echo "============================================================"

# --- Pre-flight checks ---
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
echo "[INFO] Access at: http://172.24.254.24:6006"
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
    echo '  H100 TEST RUN -- Starting at \$(date)'
    echo '============================================================'

    ./isaaclab.sh -p '${TRAIN_SCRIPT}' \
        --headless \
        --num_envs ${NUM_ENVS} \
        --max_iterations ${MAX_ITERATIONS} \
        --seed ${SEED} \
        --lr_max ${LR_MAX} \
        --lr_min ${LR_MIN} \
        --warmup_iters ${WARMUP_ITERS} \
        2>&1 | tee ~/100hr_test_run.log

    echo ''
    echo '============================================================'
    echo '  TEST RUN COMPLETE at \$(date)'
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
echo "  Live log:    tail -f ~/100hr_test_run.log"
echo "  TB URL:      http://$(hostname):6006"
echo ""
echo "  Detach from screen: Ctrl+A, D"
echo "  List screens:       screen -ls"
echo "============================================================"
