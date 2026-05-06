#!/usr/bin/env bash
# run_canonical_eval.sh — verified-working per-env eval battery for 22100.
#
# After Apr 29 rendered-smoke tuning, these are the canonical per-env
# --target_vx / --zone_slowdown_cap settings. All 4 envs hit the
# "+1m past zone 3" depth target on 2/2 episodes (verified Apr 29
# with 22100, mason flag, action_scale=0.3).
#
# Usage:
#   bash scripts/run_canonical_eval.sh                       # default 5 eps, rendered
#   bash scripts/run_canonical_eval.sh 100                   # 100 eps, rendered
#   bash scripts/run_canonical_eval.sh 100 --headless        # 100 eps, headless (production)
#   bash scripts/run_canonical_eval.sh 5 --headless --ckpt /path/to/other.pt
#
# Args: $1 = NUM_EPISODES (default 5), then any extra args (e.g. --headless or
#            --ckpt to override the default 22100 ckpt).

set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NUM_EPISODES="${1:-5}"
shift || true

# Default to 22100 ship checkpoint
CKPT="${PROJECT_DIR}/../../Experiments/Ryan/Final_Capstone_Policy_22100/parkour_phasefwplus_22100.pt"
RENDER_FLAG="--rendered"
EXTRA_ARGS=()

# Parse extra args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --headless) RENDER_FLAG="--headless"; shift ;;
        --rendered) RENDER_FLAG="--rendered"; shift ;;
        --ckpt) CKPT="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

OUTPUT_DIR="$PROJECT_DIR/results/canonical_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# --- Environment setup ---
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate env_isaaclab 2>/dev/null || conda activate isaaclab311
fi
export OMNI_KIT_ACCEPT_EULA=YES

# Detect Python (Windows uses miniconda env directly, Linux uses isaaclab.sh)
if [ -f "/c/miniconda3/envs/isaaclab311/python.exe" ]; then
    PYTHON="/c/miniconda3/envs/isaaclab311/python.exe"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "ERROR: no python found"; exit 1
fi

COMMON_ARGS=(
    "$PROJECT_DIR/src/run_capstone_eval.py"
    --policy rough
    "$RENDER_FLAG"
    --num_episodes "$NUM_EPISODES"
    --max_episode_time 180
    --checkpoint "$CKPT"
    --mason
    --action_scale 0.3
    --output_dir "$OUTPUT_DIR"
)
COMMON_ARGS+=("${EXTRA_ARGS[@]}")

echo "============================================"
echo "  CANONICAL 4-ENV EVAL BATTERY"
echo "============================================"
echo "  Checkpoint:    $CKPT"
echo "  Episodes/env:  $NUM_EPISODES"
echo "  Mode:          $RENDER_FLAG"
echo "  Output:        $OUTPUT_DIR"
echo "============================================"

# Per-env canonical (target_vx, zone_slowdown_cap) — verified Apr 29
declare -A TARGET_VX=(
    [friction]=3.0
    [grass]=3.0
    [boulder]=2.0
    [stairs]=2.0
)
declare -A ZONE_CAP=(
    [friction]=1.0
    [grass]=3.0
    [boulder]=0.67
    [stairs]=1.0
)

for env in friction grass boulder stairs; do
    tgt="${TARGET_VX[$env]}"
    cap="${ZONE_CAP[$env]}"
    echo ""
    echo "============================================"
    echo "  $env  (target_vx=$tgt  zone_slowdown_cap=$cap)"
    echo "============================================"
    "$PYTHON" "${COMMON_ARGS[@]}" \
        --env "$env" \
        --target_vx "$tgt" \
        --zone_slowdown_cap "$cap" \
        2>&1 | grep -E '\[ ?[0-9]+/'"$NUM_EPISODES"'\]|FELL|FLIP|COMPLETE|TIMEOUT|Saved|Evaluation complete|Traceback|RuntimeError' | tail -$((NUM_EPISODES + 5))
done

echo ""
echo "============================================"
echo "  CANONICAL EVAL COMPLETE"
echo "  Results: $OUTPUT_DIR"
echo "============================================"
