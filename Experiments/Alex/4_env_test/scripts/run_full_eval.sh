#!/usr/bin/env bash
# run_full_eval.sh — Full production evaluation (8,000 episodes)
#
# Runs 1000 episodes × 2 policies × 4 environments = 8,000 total episodes
# with 512 parallel environments per batch on H100.
#
# Expected runtime: ~1-1.5 hours on H100 NVL
#
# Usage:
#   bash scripts/run_full_eval.sh                  # run all 8 combinations
#   bash scripts/run_full_eval.sh friction flat     # run single combination

set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NUM_ENVS=512
NUM_EPISODES=100
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR="$PROJECT_DIR/results/full_${TIMESTAMP}"

POLICIES=("flat" "rough")
ENVIRONMENTS=("friction" "grass" "boulder" "stairs")

# Allow single combination override
if [ $# -ge 2 ]; then
    ENVIRONMENTS=("$1")
    POLICIES=("$2")
    OUTPUT_DIR="$PROJECT_DIR/results"
fi

# --- Environment setup ---
echo "============================================"
echo "  4-ENV CAPSTONE TEST — PRODUCTION RUN"
echo "============================================"
echo "  Policies:     ${POLICIES[*]}"
echo "  Environments: ${ENVIRONMENTS[*]}"
echo "  Num envs:     $NUM_ENVS"
echo "  Num episodes: $NUM_EPISODES per combination"
echo "  Output:       $OUTPUT_DIR"
echo "  Timestamp:    $TIMESTAMP"
echo "============================================"

# Source conda if available
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate env_isaaclab 2>/dev/null || conda activate isaaclab311
elif [ -f "/home/t2user/miniconda3/bin/conda" ]; then
    eval "$(/home/t2user/miniconda3/bin/conda shell.bash hook)"
    conda activate env_isaaclab 2>/dev/null || conda activate isaaclab311
fi

export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p "$OUTPUT_DIR"

# --- Run all combinations ---
TOTAL=$((${#POLICIES[@]} * ${#ENVIRONMENTS[@]}))
CURRENT=0
FAILED=0

START_TIME=$(date +%s)

for POLICY in "${POLICIES[@]}"; do
    for ENV in "${ENVIRONMENTS[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "--- [$CURRENT/$TOTAL] Policy=$POLICY, Env=$ENV ---"
        echo "Started at $(date)"

        LOG_FILE="$OUTPUT_DIR/${ENV}_${POLICY}.log"

        cd ~/IsaacLab 2>/dev/null || cd "$PROJECT_DIR"

        # Use timeout to prevent Isaac Sim shutdown hangs from blocking
        # the entire pipeline.  600s = 10 min safety margin per combo.
        # The os._exit(0) fix in run_capstone_eval.py should prevent
        # hangs, but this is a belt-and-suspenders safety net.
        if timeout 600 ./isaaclab.sh -p "$PROJECT_DIR/src/run_capstone_eval.py" --headless \
            --num_episodes "$NUM_EPISODES" \
            --policy "$POLICY" \
            --env "$ENV" \
            --output_dir "$OUTPUT_DIR" \
            2>&1 | tee "$LOG_FILE"; then
            echo "  PASSED: ${ENV}_${POLICY}"
        else
            EXIT_CODE=$?
            echo "  FAILED: ${ENV}_${POLICY} (exit code $EXIT_CODE)"
            FAILED=$((FAILED + 1))
            # Kill any lingering Isaac Sim processes from this combo
            pkill -f "run_capstone_eval.py.*--env $ENV.*--policy $POLICY" 2>/dev/null || true
            sleep 5
        fi
    done
done

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "============================================"
echo "  PRODUCTION RUN COMPLETE"
echo "  Total combinations: $TOTAL"
echo "  Failed: $FAILED"
echo "  Wall-clock time: ${ELAPSED} minutes"
echo "  Results: $OUTPUT_DIR"
echo "============================================"

if [ "$FAILED" -gt 0 ]; then
    echo "WARNING: $FAILED combination(s) failed. Check logs in $OUTPUT_DIR/*.log"
    exit 1
fi
