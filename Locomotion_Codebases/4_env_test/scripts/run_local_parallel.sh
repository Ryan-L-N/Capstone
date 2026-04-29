#!/usr/bin/env bash
# run_local_parallel.sh — Parallel 100-episode eval on local RTX 2000 Ada (8 GB)
#
# Runs 100 episodes × 2 policies × 4 environments = 800 total episodes
# in PARALLEL batches. RTX 2000 Ada (8 GB VRAM) can handle ~2 Isaac Sim
# instances at once (~2 GB each). So we run 2 combos in parallel (4 batches).
#
# Estimated runtime: ~14-16 hours (vs ~33 hours sequential)
#
# Usage (from 4_env_test/ directory):
#   bash scripts/run_local_parallel.sh
#
# Created for AI2C Tech Capstone — MS for Autonomy, February 2026

set -euo pipefail

# --- Conda activation (Windows: C:\miniconda3\envs\isaaclab311) ---
eval "$(/c/miniconda3/Scripts/conda.exe shell.bash hook)"
conda activate isaaclab311

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

NUM_EPISODES=100
PARALLEL_JOBS=2      # RTX 2000 Ada 8 GB: safe with 2 instances (~2 GB each)
OUTPUT_DIR="$PROJECT_DIR/results/parallel_${TIMESTAMP}"

# All 8 combos as "policy:env" pairs
COMBOS=(
    "flat:friction"
    "flat:grass"
    "flat:boulder"
    "flat:stairs"
    "rough:friction"
    "rough:grass"
    "rough:boulder"
    "rough:stairs"
)

# --- Environment setup ---
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

echo "============================================"
echo "  PARALLEL LOCAL EVAL — RTX 2000 Ada"
echo "============================================"
echo "  Combos:       ${#COMBOS[@]} total"
echo "  Parallel:     $PARALLEL_JOBS at a time"
echo "  Episodes:     $NUM_EPISODES per combo"
echo "  Output:       $OUTPUT_DIR"
echo "  Timestamp:    $TIMESTAMP"
echo "  Python:       $(which python)"
echo "============================================"
echo ""

# Verify Isaac Sim
python -c "from isaacsim import SimulationApp; print('Isaac Sim OK')" 2>/dev/null || {
    echo "ERROR: Cannot import isaacsim. Activate conda env first."
    exit 1
}

mkdir -p "$OUTPUT_DIR"

# --- Run combos in parallel batches ---
TOTAL=${#COMBOS[@]}
BATCH=0
FAILED=0
OVERALL_START=$(date +%s)

run_combo() {
    local POLICY="$1"
    local ENV="$2"
    local COMBO="${ENV}_${POLICY}"
    local LOG_FILE="$OUTPUT_DIR/${COMBO}.log"

    echo "  [START] $COMBO ($(date +%H:%M:%S))"

    if python "$PROJECT_DIR/src/run_capstone_eval.py" --headless \
        --num_episodes "$NUM_EPISODES" \
        --policy "$POLICY" \
        --env "$ENV" \
        --output_dir "$OUTPUT_DIR" \
        > "$LOG_FILE" 2>&1; then
        echo "  [DONE]  $COMBO — $(grep -c 'ep[0-9]' "$LOG_FILE" 2>/dev/null || echo '?') episodes"
    else
        echo "  [FAIL]  $COMBO (exit $?)"
        return 1
    fi
}

# Process combos in batches of PARALLEL_JOBS
for (( i=0; i<TOTAL; i+=PARALLEL_JOBS )); do
    BATCH=$((BATCH + 1))
    BATCH_SIZE=$PARALLEL_JOBS
    if (( i + BATCH_SIZE > TOTAL )); then
        BATCH_SIZE=$((TOTAL - i))
    fi

    echo ""
    echo "========== Batch $BATCH: combos $((i+1))-$((i+BATCH_SIZE)) of $TOTAL =========="

    PIDS=()
    COMBO_NAMES=()

    for (( j=0; j<BATCH_SIZE; j++ )); do
        IDX=$((i + j))
        IFS=':' read -r POLICY ENV <<< "${COMBOS[$IDX]}"
        COMBO_NAMES+=("${ENV}_${POLICY}")

        run_combo "$POLICY" "$ENV" &
        PIDS+=($!)
    done

    echo "  Launched ${#PIDS[@]} jobs: ${COMBO_NAMES[*]}"
    echo "  Waiting..."

    # Wait for all jobs in this batch
    BATCH_FAILED=0
    for PID in "${PIDS[@]}"; do
        if ! wait "$PID"; then
            BATCH_FAILED=$((BATCH_FAILED + 1))
        fi
    done
    FAILED=$((FAILED + BATCH_FAILED))

    # Progress report
    DONE=$((i + BATCH_SIZE))
    ELAPSED=$(( $(date +%s) - OVERALL_START ))
    ELAPSED_MIN=$((ELAPSED / 60))
    if [ "$DONE" -lt "$TOTAL" ]; then
        AVG=$((ELAPSED / DONE))
        ETA_MIN=$(( (AVG * (TOTAL - DONE)) / 60 ))
        echo "  Progress: $DONE/$TOTAL | Elapsed: ${ELAPSED_MIN}min | ETA: ~${ETA_MIN}min"
    fi

    # Brief pause between batches to let GPU cool
    sleep 5
done

OVERALL_END=$(date +%s)
OVERALL_MIN=$(( (OVERALL_END - OVERALL_START) / 60 ))

echo ""
echo "============================================"
echo "  ALL COMBOS COMPLETE"
echo "============================================"
echo "  Total time: ${OVERALL_MIN} minutes"
echo "  Failed:     $FAILED / $TOTAL"
echo ""

# --- Show per-combo episode counts ---
echo "  Per-combo results:"
for COMBO_PAIR in "${COMBOS[@]}"; do
    IFS=':' read -r POLICY ENV <<< "$COMBO_PAIR"
    COMBO="${ENV}_${POLICY}"
    LOG="$OUTPUT_DIR/${COMBO}.log"
    if [ -f "$LOG" ]; then
        EP_COUNT=$(grep -c 'ep[0-9]' "$LOG" 2>/dev/null || echo "0")
        FALLS=$(grep -c 'FELL' "$LOG" 2>/dev/null || echo "0")
        echo "    $COMBO: $EP_COUNT episodes, $FALLS falls"
    else
        echo "    $COMBO: NO LOG (did not run)"
    fi
done

echo ""

# --- Generate report ---
echo "============================================"
echo "  GENERATING REPORT"
echo "============================================"

REPORT_DIR="$OUTPUT_DIR/report"
if python "$PROJECT_DIR/src/metrics/reporter.py" --input "$OUTPUT_DIR" --output "$REPORT_DIR"; then
    echo ""
    echo "  Report:  $REPORT_DIR/"
    ls -la "$REPORT_DIR"/ 2>/dev/null || true
    if [ -d "$REPORT_DIR/plots" ]; then
        echo ""
        echo "  Plots:"
        ls -la "$REPORT_DIR/plots/" 2>/dev/null || true
    fi
else
    echo "  WARNING: Report generation failed"
fi

echo ""
echo "============================================"
echo "  PARALLEL EVAL COMPLETE"
echo "============================================"
echo "  Wall-clock:  ${OVERALL_MIN} minutes"
echo "  Episodes:    $TOTAL combos × $NUM_EPISODES = $(( TOTAL * NUM_EPISODES )) total"
echo "  Data:        $OUTPUT_DIR/*.jsonl"
echo "  Report:      $REPORT_DIR/"
echo "  Plots:       $REPORT_DIR/plots/"
if [ "$FAILED" -gt 0 ]; then
    echo ""
    echo "  WARNING: $FAILED combo(s) failed. Check logs in $OUTPUT_DIR/"
fi
echo "============================================"
