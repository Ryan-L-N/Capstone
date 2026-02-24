#!/usr/bin/env bash
# =============================================================================
# Local Debug: Smoke test on RTX 2000 Ada (64 envs, 10 iterations)
# =============================================================================
# Verifies: checkpoint loads, 12 terrains generate, 18 reward terms compute,
# progressive DR updates, no NaN, no CUDA errors.
#
# Usage:
#   cd C:/IsaacLab
#   bash /path/to/train_local_debug.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_finetune.py"

# Local checkpoint path (copy model_27500.pt here for local testing)
CHECKPOINT="${CHECKPOINT:-${SCRIPT_DIR}/checkpoints/model_27500.pt}"

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

echo "============================================"
echo "  LOCAL DEBUG — 64 envs, 10 iterations"
echo "============================================"
echo "  Script:      ${TRAIN_SCRIPT}"
echo "  Checkpoint:  ${CHECKPOINT}"
echo "============================================"

if [ ! -f "${CHECKPOINT}" ]; then
    echo ""
    echo "WARNING: Checkpoint not found at ${CHECKPOINT}"
    echo "Copy model_27500.pt to ${SCRIPT_DIR}/checkpoints/ for local testing."
    echo "Running without checkpoint (from scratch) for debug only."
    echo ""
    ./isaaclab.sh -p "${TRAIN_SCRIPT}" --headless \
        --num_envs 64 \
        --max_iterations 10 \
        --dr_expansion_iters 5 \
        --seed 42
else
    ./isaaclab.sh -p "${TRAIN_SCRIPT}" --headless \
        --num_envs 64 \
        --max_iterations 10 \
        --dr_expansion_iters 5 \
        --seed 42 \
        --checkpoint "${CHECKPOINT}"
fi

echo ""
echo "Local debug complete. Check output for:"
echo "  - Checkpoint load success (reward should start positive)"
echo "  - Terrain level > 0 at start (confirms warm start)"
echo "  - No NaN in reward terms"
echo "  - DR expansion messages every 5 iterations"
echo "============================================"
