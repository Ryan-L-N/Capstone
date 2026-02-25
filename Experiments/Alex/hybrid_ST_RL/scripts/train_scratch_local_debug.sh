#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# Attempt 5: Local Debug — From-Scratch Training with Terrain Curriculum
# ─────────────────────────────────────────────────────────────────────────
#
# Runs 10 iterations with 64 envs on the local GPU (RTX 2000 Ada) to
# verify the training script works before deploying to H100.
#
# Usage (Windows — Git Bash / MSYS2):
#   cd C:\IsaacLab
#   bash /path/to/train_scratch_local_debug.sh
#
# Usage (Linux):
#   cd ~/IsaacLab
#   bash /path/to/train_scratch_local_debug.sh
#
# Validation checklist (check these in the output):
#   [x] "ATTEMPT 6: TRAIN FROM SCRATCH + TERRAIN CURRICULUM" banner appears
#   [x] "FROM SCRATCH (no checkpoint)" in banner
#   [x] "7 types (SCRATCH_TERRAINS_CFG, flat start)" in banner
#   [x] Observations: 235 dims (48 proprio + 187 height scan)
#   [x] No checkpoint loading messages
#   [x] No freeze/unfreeze messages
#   [x] Training starts immediately (no warmup phase)
#   [x] DR fraction increases from 0% toward 100%
#   [x] No crashes or CUDA errors
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail

# Accept EULA
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

# Path to the training script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/../train_from_scratch.py"

echo "============================================================"
echo "  Attempt 5 — LOCAL DEBUG (64 envs, 10 iters)"
echo "  Script: ${TRAIN_SCRIPT}"
echo "============================================================"

# Detect OS for isaaclab launcher
if [[ -f "./isaaclab.sh" ]]; then
    LAUNCHER="./isaaclab.sh"
elif [[ -f "./isaaclab.bat" ]]; then
    LAUNCHER="./isaaclab.bat"
else
    echo "ERROR: Run this script from the IsaacLab root directory"
    echo "  cd C:\\IsaacLab   (Windows)"
    echo "  cd ~/IsaacLab    (Linux)"
    exit 1
fi

${LAUNCHER} -p "${TRAIN_SCRIPT}" --headless \
    --num_envs 64 \
    --max_iterations 10 \
    --dr_expansion_iters 5 \
    --seed 42

echo ""
echo "============================================================"
echo "  Local debug complete — check output above for validation"
echo "============================================================"
