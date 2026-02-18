#!/usr/bin/env bash
# run_rendered_viz.sh — Rendered visualization runs with video capture
#
# Runs 10 rendered episodes × 5 parallel envs × 2 policies × 4 environments
# = 80 total episodes with video and keyframe capture.
#
# Expected runtime: ~3.5-4 hours (rendering is much slower than headless)
#
# Usage:
#   bash scripts/run_rendered_viz.sh                    # all combinations
#   bash scripts/run_rendered_viz.sh friction flat       # single combination

set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NUM_ENVS=5
NUM_EPISODES=10
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR="$PROJECT_DIR/results/rendered_${TIMESTAMP}"

POLICIES=("flat" "rough")
ENVIRONMENTS=("friction" "grass" "boulder" "stairs")

# Allow single combination override
if [ $# -ge 2 ]; then
    ENVIRONMENTS=("$1")
    POLICIES=("$2")
    OUTPUT_DIR="$PROJECT_DIR/results/rendered"
fi

# --- Environment setup ---
echo "============================================"
echo "  4-ENV CAPSTONE TEST — RENDERED VIZ"
echo "============================================"
echo "  Policies:     ${POLICIES[*]}"
echo "  Environments: ${ENVIRONMENTS[*]}"
echo "  Num envs:     $NUM_ENVS"
echo "  Num episodes: $NUM_EPISODES per combination"
echo "  Output:       $OUTPUT_DIR"
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
mkdir -p "$OUTPUT_DIR"

# --- Run all combinations ---
TOTAL=$((${#POLICIES[@]} * ${#ENVIRONMENTS[@]}))
CURRENT=0

for POLICY in "${POLICIES[@]}"; do
    for ENV in "${ENVIRONMENTS[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "--- [$CURRENT/$TOTAL] Rendered: Policy=$POLICY, Env=$ENV ---"

        LOG_FILE="$OUTPUT_DIR/${ENV}_${POLICY}_rendered.log"

        cd ~/IsaacLab 2>/dev/null || cd "$PROJECT_DIR"

        ./isaaclab.sh -p "$PROJECT_DIR/src/run_capstone_eval.py" \
            --num_envs "$NUM_ENVS" \
            --num_episodes "$NUM_EPISODES" \
            --policy "$POLICY" \
            --env "$ENV" \
            --rendered \
            --capture_video \
            --output_dir "$OUTPUT_DIR" \
            2>&1 | tee "$LOG_FILE"

        echo "  Completed: ${ENV}_${POLICY}"
    done
done

echo ""
echo "============================================"
echo "  RENDERED VISUALIZATION COMPLETE"
echo "  Videos: $OUTPUT_DIR/video/"
echo "  Frames: $OUTPUT_DIR/frames/"
echo "============================================"
