#!/usr/bin/env bash
# =============================================================================
# Stage 1: Progressive Fine-Tuning — H100 NVL 96GB
# =============================================================================
# Fine-tunes the 48hr rough policy on 12 terrain types with progressive DR.
#
# Usage:
#   bash ~/hybrid_ST_RL/scripts/train_finetune_h100.sh
#
# Override defaults:
#   NUM_ENVS=8192 MAX_ITERS=30000 bash ~/hybrid_ST_RL/scripts/train_finetune_h100.sh
# =============================================================================

set -euo pipefail

# ── Config (override via environment variables) ─────────────────────────
NUM_ENVS="${NUM_ENVS:-16384}"
MAX_ITERS="${MAX_ITERS:-25000}"
DR_EXPANSION="${DR_EXPANSION:-15000}"
SEED="${SEED:-42}"
CHECKPOINT="${CHECKPOINT:-/home/t2user/IsaacLab/logs/rsl_rl/spot_rough/2026-02-13_16-26-37_48h_proprioception/model_27500.pt}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_finetune.py"

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_FILE="${HOME}/hybrid_st_rl_stage1_${TIMESTAMP}.log"

echo "============================================"
echo "  HYBRID ST-RL — STAGE 1: FINE-TUNING"
echo "============================================"
echo "  Envs:          ${NUM_ENVS}"
echo "  Max iters:     ${MAX_ITERS}"
echo "  DR expansion:  ${DR_EXPANSION} iters"
echo "  Checkpoint:    ${CHECKPOINT}"
echo "  Train script:  ${TRAIN_SCRIPT}"
echo "  Log file:      ${LOG_FILE}"
echo "  Timestamp:     ${TIMESTAMP}"
echo "============================================"

# ── Verify checkpoint exists ────────────────────────────────────────────
if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi
echo "Checkpoint OK: $(ls -lh "${CHECKPOINT}" | awk '{print $5}')"

# ── Launch training in screen ──────────────────────────────────────────
screen -dmS finetune bash -c "
    source /home/t2user/miniconda3/etc/profile.d/conda.sh
    conda activate env_isaaclab
    cd /home/t2user/IsaacLab
    export OMNI_KIT_ACCEPT_EULA=YES
    export PYTHONUNBUFFERED=1

    echo '[START] Stage 1 fine-tuning at $(date)' | tee -a ${LOG_FILE}

    ./isaaclab.sh -p ${TRAIN_SCRIPT} --headless \
        --num_envs ${NUM_ENVS} \
        --max_iterations ${MAX_ITERS} \
        --dr_expansion_iters ${DR_EXPANSION} \
        --seed ${SEED} \
        --checkpoint ${CHECKPOINT} \
        2>&1 | tee -a ${LOG_FILE}

    echo '[DONE] Stage 1 complete at $(date)' | tee -a ${LOG_FILE}
"

echo ""
echo "Training launched in screen session 'finetune'"
echo "  Monitor: screen -r finetune"
echo "  Log:     tail -f ${LOG_FILE}"
echo ""

# ── Launch TensorBoard ──────────────────────────────────────────────────
TB_LOGDIR="/home/t2user/IsaacLab/logs/rsl_rl/spot_hybrid_st_rl"

screen -dmS tb_finetune bash -c "
    source /home/t2user/miniconda3/etc/profile.d/conda.sh
    conda activate env_isaaclab
    echo '[TB] TensorBoard starting...'
    echo '[TB] Log dir: ${TB_LOGDIR}'
    echo '[TB] URL: http://\$(hostname):6006'
    tensorboard --logdir '${TB_LOGDIR}' --bind_all --port 6006 2>&1
"

echo "TensorBoard launched in screen session 'tb_finetune'"
echo "  URL: http://$(hostname):6006"
echo ""
echo "  screen -ls   # list sessions"
echo "============================================"
