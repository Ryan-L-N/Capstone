#!/bin/bash
# 5-Ring Navigation Gauntlet — evaluation launcher
#
# Usage:
#   ./scripts/run_eval.sh                          # 100 episodes, rough policy, headless
#   ./scripts/run_eval.sh --num_episodes 5 --rendered  # 5 episodes, rendered
#   ./scripts/run_eval.sh --policy flat --num_episodes 20  # flat policy baseline

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Default checkpoint
CHECKPOINT="${CHECKPOINT:-checkpoints/model_29999.pt}"

# Timestamp for results
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_DIR="results/ring_${TIMESTAMP}"

echo "=============================================="
echo "  5-Ring Navigation Gauntlet"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output:     $OUTPUT_DIR"
echo "=============================================="

python src/run_ring_eval.py \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --headless \
    "$@"
