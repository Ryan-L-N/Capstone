#!/usr/bin/env bash
# =============================================================================
# Stage 2a: Teacher Training on H100 NVL 96GB
# =============================================================================
# Trains a teacher policy with privileged observations (254-dim) initialized
# from the best Stage 1 checkpoint via weight surgery.
#
# Prerequisites:
#   - Stage 1 completed with a best checkpoint
#   - IsaacLab installed at ~/IsaacLab
#
# Usage:
#   bash train_teacher_h100.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_teacher.py"

# --- Configuration (override via environment variables) ---
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERS="${MAX_ITERS:-20000}"
SEED="${SEED:-42}"

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

echo "============================================"
echo "  STAGE 2a: TEACHER TRAINING (H100)"
echo "============================================"
echo "  Script:      ${TRAIN_SCRIPT}"
echo "  Checkpoint:  ${STAGE1_CHECKPOINT:-NOT SET}"
echo "  Envs:        ${NUM_ENVS}"
echo "  Max iters:   ${MAX_ITERS}"
echo "  Seed:        ${SEED}"
echo "============================================"

if [ -z "${STAGE1_CHECKPOINT}" ]; then
    echo ""
    echo "ERROR: STAGE1_CHECKPOINT not set."
    echo "Usage: STAGE1_CHECKPOINT=/path/to/stage1_best.pt bash $0"
    echo ""
    exit 1
fi

if [ ! -f "${STAGE1_CHECKPOINT}" ]; then
    echo ""
    echo "ERROR: Checkpoint not found at ${STAGE1_CHECKPOINT}"
    exit 1
fi

cd ~/IsaacLab

# Launch in screen session
screen -dmS train_teacher bash -c "
    cd ~/IsaacLab && \
    OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
    ./isaaclab.sh -p '${TRAIN_SCRIPT}' --headless \
        --num_envs ${NUM_ENVS} \
        --max_iterations ${MAX_ITERS} \
        --seed ${SEED} \
        --checkpoint '${STAGE1_CHECKPOINT}' \
    2>&1 | tee ~/stage2a_teacher_training.log
"

echo ""
echo "Training launched in screen session 'train_teacher'"
echo "  Monitor:  screen -r train_teacher"
echo "  Log:      tail -f ~/stage2a_teacher_training.log"
echo ""

# Launch TensorBoard
screen -dmS tb_teacher bash -c "
    source ~/miniconda3/bin/activate isaaclab311 && \
    tensorboard --logdir ~/IsaacLab/logs/rsl_rl/spot_hybrid_st_rl --port 6007 --bind_all
"

echo "TensorBoard launched on port 6007"
echo "  http://$(hostname -I | awk '{print $1}'):6007"
echo "============================================"
