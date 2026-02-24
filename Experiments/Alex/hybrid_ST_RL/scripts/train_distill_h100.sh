#!/usr/bin/env bash
# =============================================================================
# Stage 2b: Student Distillation on H100 NVL 96GB
# =============================================================================
# Distills the teacher's behavior (254-dim privileged obs) into a student
# policy with standard 235-dim observations using combined PPO + BC loss.
#
# Prerequisites:
#   - Stage 1 best checkpoint (student initialization)
#   - Stage 2a best checkpoint (teacher for action queries)
#
# Usage:
#   STUDENT_CHECKPOINT=/path/to/stage1_best.pt \
#   TEACHER_CHECKPOINT=/path/to/stage2a_best.pt \
#   bash train_distill_h100.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_distill.py"

# --- Configuration (override via environment variables) ---
STUDENT_CHECKPOINT="${STUDENT_CHECKPOINT:-}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERS="${MAX_ITERS:-10000}"
BC_START="${BC_START:-0.8}"
BC_END="${BC_END:-0.2}"
SEED="${SEED:-42}"

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

echo "============================================"
echo "  STAGE 2b: STUDENT DISTILLATION (H100)"
echo "============================================"
echo "  Script:      ${TRAIN_SCRIPT}"
echo "  Student:     ${STUDENT_CHECKPOINT:-NOT SET}"
echo "  Teacher:     ${TEACHER_CHECKPOINT:-NOT SET}"
echo "  BC coef:     ${BC_START} -> ${BC_END}"
echo "  Envs:        ${NUM_ENVS}"
echo "  Max iters:   ${MAX_ITERS}"
echo "  Seed:        ${SEED}"
echo "============================================"

if [ -z "${STUDENT_CHECKPOINT}" ] || [ -z "${TEACHER_CHECKPOINT}" ]; then
    echo ""
    echo "ERROR: Both STUDENT_CHECKPOINT and TEACHER_CHECKPOINT must be set."
    echo "Usage:"
    echo "  STUDENT_CHECKPOINT=/path/to/stage1_best.pt \\"
    echo "  TEACHER_CHECKPOINT=/path/to/stage2a_best.pt \\"
    echo "  bash $0"
    echo ""
    exit 1
fi

if [ ! -f "${STUDENT_CHECKPOINT}" ]; then
    echo "ERROR: Student checkpoint not found at ${STUDENT_CHECKPOINT}"
    exit 1
fi

if [ ! -f "${TEACHER_CHECKPOINT}" ]; then
    echo "ERROR: Teacher checkpoint not found at ${TEACHER_CHECKPOINT}"
    exit 1
fi

cd ~/IsaacLab

# Launch in screen session
screen -dmS train_distill bash -c "
    cd ~/IsaacLab && \
    OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
    ./isaaclab.sh -p '${TRAIN_SCRIPT}' --headless \
        --num_envs ${NUM_ENVS} \
        --max_iterations ${MAX_ITERS} \
        --seed ${SEED} \
        --student_checkpoint '${STUDENT_CHECKPOINT}' \
        --teacher_checkpoint '${TEACHER_CHECKPOINT}' \
        --bc_start ${BC_START} \
        --bc_end ${BC_END} \
    2>&1 | tee ~/stage2b_distill_training.log
"

echo ""
echo "Distillation launched in screen session 'train_distill'"
echo "  Monitor:  screen -r train_distill"
echo "  Log:      tail -f ~/stage2b_distill_training.log"
echo ""

# Launch TensorBoard (if not already running)
if ! screen -list | grep -q tb_teacher; then
    screen -dmS tb_distill bash -c "
        source ~/miniconda3/bin/activate isaaclab311 && \
        tensorboard --logdir ~/IsaacLab/logs/rsl_rl/spot_hybrid_st_rl --port 6007 --bind_all
    "
    echo "TensorBoard launched on port 6007"
    echo "  http://$(hostname -I | awk '{print $1}'):6007"
else
    echo "TensorBoard already running (session tb_teacher)"
fi
echo "============================================"
