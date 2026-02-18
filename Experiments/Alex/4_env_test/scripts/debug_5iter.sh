#!/usr/bin/env bash
# debug_5iter.sh — Quick sanity check before production runs
#
# Runs 5 episodes with 2 parallel envs on a single environment.
# Designed to complete in <1 minute. Run this up to 5 times,
# documenting findings in LESSONS_LEARNED.md between each run.
#
# Usage:
#   bash scripts/debug_5iter.sh                    # default: friction env, flat policy
#   bash scripts/debug_5iter.sh grass rough        # specify env and policy
#   bash scripts/debug_5iter.sh boulder flat       # any combination

set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV="${1:-friction}"
POLICY="${2:-flat}"
NUM_ENVS=2
NUM_EPISODES=5
OUTPUT_DIR="$PROJECT_DIR/results/debug"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# --- Environment setup ---
echo "============================================"
echo "  4-ENV CAPSTONE TEST — DEBUG RUN"
echo "============================================"
echo "  Environment:  $ENV"
echo "  Policy:       $POLICY"
echo "  Num envs:     $NUM_ENVS"
echo "  Num episodes: $NUM_EPISODES"
echo "  Output:       $OUTPUT_DIR"
echo "  Timestamp:    $TIMESTAMP"
echo "============================================"

# Source conda if available (needed for non-interactive SSH)
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate isaaclab311
elif [ -f "/home/t2user/miniconda3/bin/conda" ]; then
    eval "$(/home/t2user/miniconda3/bin/conda shell.bash hook)"
    conda activate isaaclab311
fi

# EULA acceptance for headless mode
export OMNI_KIT_ACCEPT_EULA=YES

# Create output directory
mkdir -p "$OUTPUT_DIR"

# --- Run evaluation ---
echo ""
echo "Starting debug run at $(date)..."
echo ""

cd ~/IsaacLab 2>/dev/null || cd "$PROJECT_DIR"

./isaaclab.sh -p "$PROJECT_DIR/src/run_capstone_eval.py" --headless \
    --num_envs "$NUM_ENVS" \
    --num_episodes "$NUM_EPISODES" \
    --policy "$POLICY" \
    --env "$ENV" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/debug_${ENV}_${POLICY}_${TIMESTAMP}.log"

echo ""
echo "============================================"
echo "  DEBUG RUN COMPLETE"
echo "  Check: $OUTPUT_DIR"
echo "  Document findings in LESSONS_LEARNED.md"
echo "============================================"
