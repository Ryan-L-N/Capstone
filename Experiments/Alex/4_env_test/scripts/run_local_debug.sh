#!/usr/bin/env bash
# run_local_debug.sh — Local debug run on Windows (Isaac Sim 5.1)
#
# Runs 5 episodes × 2 policies × 4 environments = 40 total episodes
# headless on your local GPU, then generates the reporter visualizations.
#
# Prerequisites:
#   conda activate isaaclab311
#   set OMNI_KIT_ACCEPT_EULA=YES
#
# Usage (from 4_env_test/ directory):
#   bash scripts/run_local_debug.sh                # all 8 combos
#   bash scripts/run_local_debug.sh friction flat   # single combo

set -euo pipefail

# --- Conda activation (Windows: C:\miniconda3\envs\isaaclab311) ---
# Use C:\miniconda3 explicitly (NOT C:\Users\...\miniconda3)
eval "$(/c/miniconda3/Scripts/conda.exe shell.bash hook)"
conda activate isaaclab311

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

NUM_EPISODES=5
OUTPUT_DIR="$PROJECT_DIR/results/local_debug_${TIMESTAMP}"

POLICIES=("flat" "rough")
ENVIRONMENTS=("friction" "grass" "boulder" "stairs")

# Allow single combination override
if [ $# -ge 2 ]; then
    ENVIRONMENTS=("$1")
    POLICIES=("$2")
fi

# --- Environment setup ---
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

echo "============================================"
echo "  LOCAL DEBUG RUN — Isaac Sim 5.1 (Windows)"
echo "============================================"
echo "  Policies:     ${POLICIES[*]}"
echo "  Environments: ${ENVIRONMENTS[*]}"
echo "  Episodes:     $NUM_EPISODES per combo"
echo "  Output:       $OUTPUT_DIR"
echo "  Timestamp:    $TIMESTAMP"
echo "  Python:       $(which python)"
echo "============================================"
echo ""

# Verify Isaac Sim is available
python -c "from isaacsim import SimulationApp; print('Isaac Sim OK')" 2>/dev/null || {
    echo "ERROR: Cannot import isaacsim. Activate conda env first:"
    echo "  conda activate isaaclab311"
    exit 1
}

mkdir -p "$OUTPUT_DIR"

# --- Run all combinations ---
TOTAL=$(( ${#POLICIES[@]} * ${#ENVIRONMENTS[@]} ))
CURRENT=0
FAILED=0
RESULTS=()

OVERALL_START=$(date +%s)

for POLICY in "${POLICIES[@]}"; do
    for ENV in "${ENVIRONMENTS[@]}"; do
        CURRENT=$((CURRENT + 1))
        COMBO="${ENV}_${POLICY}"
        LOG_FILE="$OUTPUT_DIR/debug_${COMBO}.log"

        echo "--- [$CURRENT/$TOTAL] env=$ENV policy=$POLICY ---"
        echo "  Started: $(date)"

        COMBO_START=$(date +%s)

        # Run headless eval — redirect to log (no pipe, avoids orphan issues)
        if python "$PROJECT_DIR/src/run_capstone_eval.py" --headless \
            --num_episodes "$NUM_EPISODES" \
            --policy "$POLICY" \
            --env "$ENV" \
            --output_dir "$OUTPUT_DIR" \
            > "$LOG_FILE" 2>&1; then

            COMBO_END=$(date +%s)
            COMBO_TIME=$(( COMBO_END - COMBO_START ))
            echo "  PASSED: $COMBO (${COMBO_TIME}s)"
            RESULTS+=("PASS: $COMBO (${COMBO_TIME}s)")
            # Show key output lines
            tail -10 "$LOG_FILE" | grep -E "ep[0-9]|Saved|Evaluation|Exiting|complete" || true
        else
            COMBO_END=$(date +%s)
            COMBO_TIME=$(( COMBO_END - COMBO_START ))
            echo "  FAILED: $COMBO (exit code $?, ${COMBO_TIME}s)"
            RESULTS+=("FAIL: $COMBO (${COMBO_TIME}s)")
            FAILED=$((FAILED + 1))
            echo "  Last 5 lines of log:"
            tail -5 "$LOG_FILE" || true
        fi

        # Brief pause between combos to let GPU memory free
        sleep 3
        echo ""
    done
done

OVERALL_END=$(date +%s)
OVERALL_TIME=$(( (OVERALL_END - OVERALL_START) / 60 ))

echo "============================================"
echo "  DEBUG RESULTS"
echo "============================================"
for R in "${RESULTS[@]}"; do
    echo "  $R"
done
echo ""
echo "  Total time: ${OVERALL_TIME} minutes"
echo "  Failed: $FAILED / $TOTAL"
echo "============================================"
echo ""

# Gate: abort reporter if any combo failed
if [ "$FAILED" -gt 0 ]; then
    echo "WARNING: $FAILED combo(s) failed. Check logs in $OUTPUT_DIR/"
    echo "Skipping reporter — fix failures first."
    exit 1
fi

# --- Phase 2: Generate report ---
echo "============================================"
echo "  GENERATING REPORT"
echo "============================================"
echo ""

# Check what JSONL files we have
echo "JSONL files collected:"
ls -la "$OUTPUT_DIR"/*.jsonl 2>/dev/null || echo "  (none found)"
echo ""

REPORT_DIR="$OUTPUT_DIR/report"
if python "$PROJECT_DIR/src/metrics/reporter.py" --input "$OUTPUT_DIR" --output "$REPORT_DIR"; then
    echo ""
    echo "  Report generated: $REPORT_DIR/"
    echo ""
    echo "  Output files:"
    ls -la "$REPORT_DIR"/ 2>/dev/null || true
    if [ -d "$REPORT_DIR/plots" ]; then
        echo ""
        echo "  Plots:"
        ls -la "$REPORT_DIR/plots/" 2>/dev/null || true
    fi
else
    echo "  WARNING: Report generation failed"
    exit 1
fi

echo ""
echo "============================================"
echo "  LOCAL DEBUG COMPLETE"
echo "============================================"
echo "  Total wall-clock: ${OVERALL_TIME} minutes"
echo "  Episodes:  $TOTAL combos × $NUM_EPISODES = $(( TOTAL * NUM_EPISODES )) total"
echo "  Data:      $OUTPUT_DIR/*.jsonl"
echo "  Report:    $REPORT_DIR/"
echo "  Plots:     $REPORT_DIR/plots/"
echo ""
echo "  Next: review plots, then run production on H100"
echo "============================================"
