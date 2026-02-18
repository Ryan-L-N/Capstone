#!/usr/bin/env bash
# run_teleop.sh — Manual Xbox controller walkthrough
#
# Launches a single-robot environment for manual navigation with
# Xbox controller or keyboard. Supports gait switching (RB) and FPV camera (LB).
#
# Usage:
#   bash scripts/run_teleop.sh friction     # specify environment
#   bash scripts/run_teleop.sh stairs       # any of: friction, grass, boulder, stairs
#
# Controls:
#   Xbox:  Left Stick = move/turn, RB = cycle gait, LB = FPV camera,
#          A = drive mode, B = selfright, Y = reset, Back = E-stop
#   Keys:  WASD = move/turn, G = cycle gait, M = FPV camera,
#          SHIFT = drive mode, X = selfright, R = reset, SPACE = E-stop

set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV="${1:-friction}"
DEVICE="${2:-xbox}"

# --- Validate environment ---
case "$ENV" in
    friction|grass|boulder|stairs)
        ;;
    *)
        echo "ERROR: Unknown environment '$ENV'"
        echo "Valid environments: friction, grass, boulder, stairs"
        exit 1
        ;;
esac

echo "============================================"
echo "  4-ENV CAPSTONE TEST — MANUAL TELEOP"
echo "============================================"
echo "  Environment: $ENV"
echo "  Controller:  $DEVICE"
echo "  Gaits:       FLAT <-> ROUGH (press RB / G)"
echo "============================================"

# Source conda if available
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate isaaclab311
elif [ -f "/home/t2user/miniconda3/bin/conda" ]; then
    eval "$(/home/t2user/miniconda3/bin/conda shell.bash hook)"
    conda activate isaaclab311
fi

export OMNI_KIT_ACCEPT_EULA=YES

cd ~/IsaacLab 2>/dev/null || cd "$PROJECT_DIR"

./isaaclab.sh -p "$PROJECT_DIR/src/run_capstone_teleop.py" \
    --env "$ENV" \
    --device "$DEVICE"
