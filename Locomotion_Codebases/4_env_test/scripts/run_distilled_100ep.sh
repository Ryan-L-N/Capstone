#!/bin/bash
# Run Distilled Policy (model_6899) through all 4 environments, 100 episodes each.
#
# This is the multi-expert distilled policy that combines:
#   - Friction expert (mason_hybrid_best_33200.pt) — smooth/drag terrain
#   - Obstacle expert (obstacle_best_44400.pt) — boulders/stairs
#
# Usage:
#   cd <repo_root>/Experiments/Alex/4_env_test
#   bash scripts/run_distilled_100ep.sh [--headless]
#
# Options:
#   --headless    Run without GUI (faster, for batch eval)
#   (default)     Rendered mode (watch the robot)
#
# Requirements:
#   - Conda env: isaaclab311
#   - Checkpoint: multi_robot_training/checkpoints/distilled_6899.pt
#   - Isaac Sim 5.1 via isaaclab311 environment

set -e

# ── Resolve paths relative to this script ──────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EVAL_ROOT/../.." && pwd)"

EVAL_SCRIPT="$EVAL_ROOT/src/run_capstone_eval.py"
CHECKPOINT="$REPO_ROOT/multi_robot_training/checkpoints/distilled_6899.pt"
OUTPUT_DIR="$EVAL_ROOT/results/distilled_100ep"

# ── Find Python ────────────────────────────────────────────────────────
# Try common conda paths
if [ -f "/c/miniconda3/envs/isaaclab311/python" ]; then
    PYTHON="/c/miniconda3/envs/isaaclab311/python"
elif command -v conda &> /dev/null; then
    PYTHON="$(conda run -n isaaclab311 which python 2>/dev/null || echo "")"
fi

if [ -z "$PYTHON" ] || [ ! -f "$PYTHON" ]; then
    echo "[ERROR] Cannot find isaaclab311 Python. Activate conda env first:"
    echo "  conda activate isaaclab311"
    echo "  python $EVAL_SCRIPT ..."
    exit 1
fi

# ── Parse args ─────────────────────────────────────────────────────────
HEADLESS_FLAG=""
if [[ "$1" == "--headless" ]]; then
    HEADLESS_FLAG="--headless"
fi

# ── Verify checkpoint exists ───────────────────────────────────────────
if [ ! -f "$CHECKPOINT" ]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT"
    echo "  Place distilled_6899.pt in multi_robot_training/checkpoints/"
    exit 1
fi

# ── Clear pycache ──────────────────────────────────────────────────────
rm -rf "$EVAL_ROOT/src/configs/__pycache__"
rm -rf "$EVAL_ROOT/src/__pycache__"
rm -rf "$EVAL_ROOT/src/envs/__pycache__"
rm -rf "$EVAL_ROOT/src/navigation/__pycache__"
rm -rf "$EVAL_ROOT/src/metrics/__pycache__"

mkdir -p "$OUTPUT_DIR"

# ── Run ────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Distilled Policy 100-Episode Evaluation"
echo "  Checkpoint: distilled_6899.pt"
echo "  Output:     $OUTPUT_DIR"
echo "  Mode:       ${HEADLESS_FLAG:-rendered}"
echo "  Started:    $(date)"
echo "=============================================="

for ENV in friction grass boulder stairs; do
    echo ""
    echo ">>> Starting $ENV (100 episodes) at $(date) ..."
    "$PYTHON" "$EVAL_SCRIPT" \
        --robot spot --policy rough --env "$ENV" --mason $HEADLESS_FLAG \
        --num_episodes 100 \
        --checkpoint "$CHECKPOINT" \
        --output_dir "$OUTPUT_DIR" 2>&1
    echo ">>> Finished $ENV at $(date)"
    echo ""
done

echo "=============================================="
echo "  All 4 environments complete!"
echo "  Finished: $(date)"
echo "  Results:  $OUTPUT_DIR"
echo "=============================================="
echo ""
echo "JSONL files saved to:"
ls -la "$OUTPUT_DIR"/*.jsonl 2>/dev/null || echo "  (no JSONL files found)"
