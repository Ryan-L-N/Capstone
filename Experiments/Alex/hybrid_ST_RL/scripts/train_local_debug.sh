#!/usr/bin/env bash
# =============================================================================
# Local Debug: Smoke test on RTX 2000 Ada (64 envs, 10 iterations)
# =============================================================================
# Verifies: actor-only loading, critic warmup, noise floor, 12 terrains,
# 19 reward terms, progressive DR, no NaN, no CUDA errors.
#
# Attempt #3: Tests all fixes (actor-only load, std freeze, noise clamp [min,max]).
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
        --actor_freeze_iters 3 \
        --min_noise_std 0.4 \
        --max_noise_std 1.5 \
        --seed 42
else
    ./isaaclab.sh -p "${TRAIN_SCRIPT}" --headless \
        --num_envs 64 \
        --max_iterations 10 \
        --dr_expansion_iters 5 \
        --actor_freeze_iters 3 \
        --min_noise_std 0.4 \
        --max_noise_std 1.5 \
        --seed 42 \
        --checkpoint "${CHECKPOINT}"
fi

echo ""
echo "Local debug complete. Check output for:"
echo "  - Actor-only load (SKIPPED critic keys message)"
echo "  - Actor FROZEN / UNFROZEN messages (critic warmup)"
echo "  - Noise std in [0.4, 1.5] (clamped both directions)"
echo "  - Noise std stable at ~0.65 during warmup (std frozen)"
echo "  - Terrain level > 0 at start (confirms warm start)"
echo "  - No NaN in reward terms"
echo "  - DR expansion messages"
echo "============================================"
