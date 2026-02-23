#!/usr/bin/env bash
# =============================================================================
# Local Debug Run — RTX 2000 Ada (Windows/MSYS2)
# =============================================================================
#
# Quick sanity check: 100 iterations with 64 envs.
# Verifies terrain generation, reward computation, and checkpoint saving.
# Expected: ~5 minutes.
#
# Usage:
#   cd /c/IsaacLab
#   bash /path/to/train_local_debug.sh
#
# Created for AI2C Tech Capstone — MS for Autonomy, February 2026
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_100hr.py"
ISAACLAB_DIR="/c/IsaacLab"

NUM_ENVS=64
MAX_ITERATIONS=100
SEED=42

echo "============================================================"
echo "  Local Debug Run — 100 iterations, 64 envs"
echo "============================================================"

cd "${ISAACLAB_DIR}"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

./isaaclab.bat -p "${TRAIN_SCRIPT}" \
    --headless \
    --num_envs ${NUM_ENVS} \
    --max_iterations ${MAX_ITERATIONS} \
    --seed ${SEED} \
    --lr_max 1e-3 \
    --lr_min 1e-5 \
    --warmup_iters 10

echo "Debug run complete."
