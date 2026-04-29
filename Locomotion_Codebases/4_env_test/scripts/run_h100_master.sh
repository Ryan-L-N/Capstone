#!/usr/bin/env bash
# run_h100_master.sh — Complete H100 production pipeline
#
# Phase 1: Debug all 8 combinations (5 episodes each, ~8 min)
# Phase 2: Full production run (1000 episodes each, ~1-1.5 hours)
# Phase 3: Generate summary report
#
# Usage:
#   screen -S capstone
#   bash scripts/run_h100_master.sh
#
# To resume after SSH disconnect:
#   screen -r capstone

set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

POLICIES=("flat" "rough")
ENVIRONMENTS=("friction" "grass" "boulder" "stairs")

DEBUG_EPISODES=5
PROD_EPISODES=100
PROD_OUTPUT="$PROJECT_DIR/results/full_${TIMESTAMP}"

# --- Environment setup ---
echo "============================================"
echo "  4-ENV CAPSTONE — H100 MASTER PIPELINE"
echo "============================================"
echo "  Started:      $(date)"
echo "  Project:      $PROJECT_DIR"
echo "  Combinations: ${#POLICIES[@]} policies x ${#ENVIRONMENTS[@]} envs = $(( ${#POLICIES[@]} * ${#ENVIRONMENTS[@]} ))"
echo "  Debug:        $DEBUG_EPISODES episodes each"
echo "  Production:   $PROD_EPISODES episodes each"
echo "  Output:       $PROD_OUTPUT"
echo "============================================"
echo ""

# Source conda (handle non-interactive SSH)
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate env_isaaclab 2>/dev/null || conda activate isaaclab311
elif [ -f "/home/t2user/miniconda3/bin/conda" ]; then
    eval "$(/home/t2user/miniconda3/bin/conda shell.bash hook)"
    conda activate env_isaaclab 2>/dev/null || conda activate isaaclab311
fi

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

# Verify Python and Isaac Sim
echo "Python: $(which python)"
python -c "import isaacsim; print(f'Isaac Sim OK')" 2>/dev/null || echo "WARNING: isaacsim import check skipped"
echo ""

# ══════════════════════════════════════════════════════════════════════
# PHASE 1: DEBUG ALL 8 COMBINATIONS
# ══════════════════════════════════════════════════════════════════════
echo "============================================"
echo "  PHASE 1: DEBUG RUNS (5 episodes each)"
echo "============================================"
echo ""

DEBUG_DIR="$PROJECT_DIR/results/debug"
mkdir -p "$DEBUG_DIR"

TOTAL=$(( ${#POLICIES[@]} * ${#ENVIRONMENTS[@]} ))
CURRENT=0
DEBUG_FAILED=0
DEBUG_RESULTS=()

PHASE1_START=$(date +%s)

for POLICY in "${POLICIES[@]}"; do
    for ENV in "${ENVIRONMENTS[@]}"; do
        CURRENT=$((CURRENT + 1))
        COMBO="${ENV}_${POLICY}"
        LOG_FILE="$DEBUG_DIR/debug_${COMBO}_${TIMESTAMP}.log"

        echo "--- [$CURRENT/$TOTAL] DEBUG: env=$ENV policy=$POLICY ---"
        echo "  Started: $(date)"

        cd ~/IsaacLab 2>/dev/null || cd "$PROJECT_DIR"

        COMBO_START=$(date +%s)

        # 420s timeout; --foreground kills process group; -k 30 sends SIGKILL
        # NOTE: No pipe to tee — pipes prevent timeout from killing grandchild Python.
        if timeout --foreground -k 30 420 ./isaaclab.sh -p "$PROJECT_DIR/src/run_capstone_eval.py" --headless \
            --num_episodes "$DEBUG_EPISODES" \
            --policy "$POLICY" \
            --env "$ENV" \
            --output_dir "$DEBUG_DIR" \
            > "$LOG_FILE" 2>&1; then

            COMBO_END=$(date +%s)
            COMBO_TIME=$(( COMBO_END - COMBO_START ))
            echo "  PASSED: $COMBO (${COMBO_TIME}s)"
            DEBUG_RESULTS+=("PASS: $COMBO (${COMBO_TIME}s)")
            tail -8 "$LOG_FILE" | grep -E "ep[0-9]|Saved|Evaluation|Exiting" || true
        else
            COMBO_END=$(date +%s)
            COMBO_TIME=$(( COMBO_END - COMBO_START ))
            echo "  FAILED: $COMBO (exit code $?, ${COMBO_TIME}s)"
            DEBUG_RESULTS+=("FAIL: $COMBO (${COMBO_TIME}s)")
            DEBUG_FAILED=$((DEBUG_FAILED + 1))
            tail -5 "$LOG_FILE" || true
        fi
        # Kill ANY lingering processes (broad match — arg order varies)
        pkill -f "run_capstone_eval" 2>/dev/null || true
        sleep 5
        echo ""
    done
done

PHASE1_END=$(date +%s)
PHASE1_TIME=$(( (PHASE1_END - PHASE1_START) / 60 ))

echo "============================================"
echo "  PHASE 1 RESULTS — DEBUG RUNS"
echo "============================================"
for R in "${DEBUG_RESULTS[@]}"; do
    echo "  $R"
done
echo "  Total time: ${PHASE1_TIME} minutes"
echo "  Failed: $DEBUG_FAILED / $TOTAL"
echo "============================================"
echo ""

# Gate: abort if any debug run failed
if [ "$DEBUG_FAILED" -gt 0 ]; then
    echo "ABORTING: $DEBUG_FAILED debug run(s) failed."
    echo "Check logs in $DEBUG_DIR/"
    echo "Fix issues and re-run."
    exit 1
fi

echo "All debug runs passed! Proceeding to production..."
echo ""

# ══════════════════════════════════════════════════════════════════════
# PHASE 2: FULL PRODUCTION (1000 episodes each)
# ══════════════════════════════════════════════════════════════════════
echo "============================================"
echo "  PHASE 2: PRODUCTION ($PROD_EPISODES episodes each)"
echo "============================================"
echo ""

mkdir -p "$PROD_OUTPUT"

CURRENT=0
PROD_FAILED=0
PROD_RESULTS=()

PHASE2_START=$(date +%s)

for POLICY in "${POLICIES[@]}"; do
    for ENV in "${ENVIRONMENTS[@]}"; do
        CURRENT=$((CURRENT + 1))
        COMBO="${ENV}_${POLICY}"
        LOG_FILE="$PROD_OUTPUT/${COMBO}.log"

        echo "--- [$CURRENT/$TOTAL] PRODUCTION: env=$ENV policy=$POLICY ---"
        echo "  Started: $(date)"
        echo "  Episodes: $PROD_EPISODES"

        cd ~/IsaacLab 2>/dev/null || cd "$PROJECT_DIR"

        COMBO_START=$(date +%s)

        # 7200s (2hr) timeout; --foreground kills process group; -k 30 SIGKILL
        # NOTE: No pipe to tee — pipes prevent timeout from killing grandchild Python.
        if timeout --foreground -k 30 7200 ./isaaclab.sh -p "$PROJECT_DIR/src/run_capstone_eval.py" --headless \
            --num_episodes "$PROD_EPISODES" \
            --policy "$POLICY" \
            --env "$ENV" \
            --output_dir "$PROD_OUTPUT" \
            > "$LOG_FILE" 2>&1; then

            COMBO_END=$(date +%s)
            COMBO_TIME=$(( COMBO_END - COMBO_START ))
            COMBO_MIN=$(( COMBO_TIME / 60 ))
            echo "  PASSED: $COMBO (${COMBO_MIN}min)"
            PROD_RESULTS+=("PASS: $COMBO (${COMBO_MIN}min)")
            tail -8 "$LOG_FILE" | grep -E "ep[0-9]|Saved|Evaluation|Exiting" || true
        else
            COMBO_END=$(date +%s)
            COMBO_TIME=$(( COMBO_END - COMBO_START ))
            COMBO_MIN=$(( COMBO_TIME / 60 ))
            echo "  FAILED: $COMBO (exit code $?, ${COMBO_MIN}min)"
            PROD_RESULTS+=("FAIL: $COMBO (${COMBO_MIN}min)")
            PROD_FAILED=$((PROD_FAILED + 1))
            tail -5 "$LOG_FILE" || true
        fi
        # Kill ANY lingering processes (broad match — arg order varies)
        pkill -f "run_capstone_eval" 2>/dev/null || true
        sleep 5
        echo ""
    done
done

PHASE2_END=$(date +%s)
PHASE2_TIME=$(( (PHASE2_END - PHASE2_START) / 60 ))

echo "============================================"
echo "  PHASE 2 RESULTS — PRODUCTION RUNS"
echo "============================================"
for R in "${PROD_RESULTS[@]}"; do
    echo "  $R"
done
echo "  Total time: ${PHASE2_TIME} minutes"
echo "  Failed: $PROD_FAILED / $TOTAL"
echo "============================================"
echo ""

# ══════════════════════════════════════════════════════════════════════
# PHASE 3: GENERATE REPORT
# ══════════════════════════════════════════════════════════════════════
echo "============================================"
echo "  PHASE 3: GENERATING REPORT"
echo "============================================"

cd "$PROJECT_DIR"

if python src/metrics/reporter.py --input "$PROD_OUTPUT" --output "$PROD_OUTPUT/report"; then
    echo "  Report generated: $PROD_OUTPUT/report/"
else
    echo "  WARNING: Report generation failed (non-critical)"
fi

# ══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
TOTAL_END=$(date +%s)
TOTAL_TIME=$(( (TOTAL_END - PHASE1_START) / 60 ))

echo ""
echo "============================================"
echo "  PIPELINE COMPLETE"
echo "============================================"
echo "  Total wall-clock: ${TOTAL_TIME} minutes"
echo "  Debug:      ${PHASE1_TIME} min ($TOTAL combos × $DEBUG_EPISODES eps)"
echo "  Production: ${PHASE2_TIME} min ($TOTAL combos × $PROD_EPISODES eps)"
echo "  Total episodes: $(( TOTAL * PROD_EPISODES ))"
echo "  Results: $PROD_OUTPUT"
echo "  Report:  $PROD_OUTPUT/report/"
echo ""
echo "  Debug failures:      $DEBUG_FAILED"
echo "  Production failures: $PROD_FAILED"
echo "============================================"

if [ "$PROD_FAILED" -gt 0 ]; then
    echo "WARNING: $PROD_FAILED production run(s) failed."
    exit 1
fi

echo "All done! Download results from $PROD_OUTPUT"
