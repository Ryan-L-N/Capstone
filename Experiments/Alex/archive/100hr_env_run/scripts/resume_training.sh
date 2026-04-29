#!/usr/bin/env bash
# =============================================================================
# Resume Training from Checkpoint — H100 NVL 96GB
# =============================================================================
#
# Resumes a crashed or interrupted training run from the latest checkpoint.
#
# Usage:
#   bash /path/to/resume_training.sh [run_dir] [checkpoint]
#
# Examples:
#   # Resume from latest checkpoint in latest run:
#   bash resume_training.sh
#
#   # Resume from specific run directory:
#   bash resume_training.sh 2026-02-20_15-00-00_multi_terrain_v1
#
#   # Resume from specific checkpoint in specific run:
#   bash resume_training.sh 2026-02-20_15-00-00_multi_terrain_v1 model_10000.pt
#
# Created for AI2C Tech Capstone — MS for Autonomy, February 2026
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_100hr.py"
ISAACLAB_DIR="${HOME}/IsaacLab"

NUM_ENVS=20480
MAX_ITERATIONS=60000
SEED=42

LOAD_RUN="${1:-}"
LOAD_CHECKPOINT="${2:-model_.*\\.pt}"

echo "============================================================"
echo "  Resuming 100hr Training"
echo "============================================================"
echo "  Load run:        ${LOAD_RUN:-latest}"
echo "  Load checkpoint: ${LOAD_CHECKPOINT}"
echo "============================================================"

cd "${ISAACLAB_DIR}"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

RESUME_ARGS="--resume"
if [ -n "${LOAD_RUN}" ]; then
    RESUME_ARGS="${RESUME_ARGS} --load_run ${LOAD_RUN}"
fi

./isaaclab.sh -p "${TRAIN_SCRIPT}" \
    --headless \
    --num_envs ${NUM_ENVS} \
    --max_iterations ${MAX_ITERATIONS} \
    --seed ${SEED} \
    --lr_max 1e-3 \
    --lr_min 1e-5 \
    --warmup_iters 3000 \
    ${RESUME_ARGS} \
    --load_checkpoint "${LOAD_CHECKPOINT}"

echo "[$(date)] Resume training complete."
