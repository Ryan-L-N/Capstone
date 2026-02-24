#!/usr/bin/env bash
# =============================================================================
# Stage 1: Progressive Fine-Tuning — H100 NVL 96GB
# =============================================================================
# Fine-tunes the 48hr rough policy on 12 terrain types with progressive DR.
#
# Attempt #4: Ultra-conservative PPO to prevent catastrophic forgetting at unfreeze.
#   - LR 1e-5 (was 1e-4), clip 0.1 (was 0.2), entropy 0.0 (was 0.005)
#   - 3 learning epochs (was 5), KL target 0.005 (was 0.008)
#   - Noise std permanently frozen at 0.65
#   - LR warmup: 2e-6 → 1e-5 over 1000 iters post-unfreeze
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
ACTOR_FREEZE="${ACTOR_FREEZE:-1000}"
LR_WARMUP="${LR_WARMUP:-1000}"
MIN_NOISE_STD="${MIN_NOISE_STD:-0.4}"
MAX_NOISE_STD="${MAX_NOISE_STD:-1.5}"
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
echo "  Actor freeze:  ${ACTOR_FREEZE} iters (critic warmup)"
echo "  LR warmup:     ${LR_WARMUP} iters post-unfreeze (2e-6 → 1e-5)"
echo "  Noise std:     permanently frozen at 0.65 (clamp [${MIN_NOISE_STD}, ${MAX_NOISE_STD}])"
echo "  Checkpoint:    ${CHECKPOINT}"
echo "  Train script:  ${TRAIN_SCRIPT}"
echo "  Log file:      ${LOG_FILE}"
echo "  Timestamp:     ${TIMESTAMP}"
echo "  Attempt:       #4 (ultra-conservative PPO + LR warmup + permanent std freeze)"
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
        --actor_freeze_iters ${ACTOR_FREEZE} \
        --lr_warmup_iters ${LR_WARMUP} \
        --min_noise_std ${MIN_NOISE_STD} \
        --max_noise_std ${MAX_NOISE_STD} \
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
