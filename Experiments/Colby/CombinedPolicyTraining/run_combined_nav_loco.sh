#!/usr/bin/env bash
# run_combined_nav_loco.sh
#
# Launches combined nav + loco training.
# Cleanup of orphaned Isaac Sim processes is handled automatically by
# isaac_cleanup.sh — fires on normal exit, crash, or Ctrl+C.
#
# Usage (local smoke test, 100 iters):
#   bash Experiments/Colby/CombinedPolicyTraining/run_combined_nav_loco.sh --local
#
# Usage (H100 full run):
#   bash Experiments/Colby/CombinedPolicyTraining/run_combined_nav_loco.sh --h100

set -e

# ---------------------------------------------------------------------------
# Resolve paths relative to repo root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

NAV_ALEX_DIR="$REPO_ROOT/Experiments/Alex/NAV_ALEX"
LOCO_CHECKPOINT="$REPO_ROOT/Experiments/Ryan/checkpoints/mason_hybrid_best_33200.pt"
TRAIN_SCRIPT="$SCRIPT_DIR/train_combined.py"

# ---------------------------------------------------------------------------
# Register automatic Isaac Sim process cleanup on exit
# ---------------------------------------------------------------------------
source "$SCRIPT_DIR/../ExitKillScript/isaac_cleanup.sh"

# ---------------------------------------------------------------------------
# Validate loco checkpoint
# ---------------------------------------------------------------------------
echo "=== Combined Nav + Loco Training ==="
echo "Train script      : $TRAIN_SCRIPT"
echo "Loco checkpoint   : $LOCO_CHECKPOINT"
echo "Checkpoint output : $SCRIPT_DIR/logs/"
echo ""

if [ ! -f "$LOCO_CHECKPOINT" ]; then
    echo "ERROR: Loco checkpoint not found: $LOCO_CHECKPOINT"
    exit 1
fi

# ---------------------------------------------------------------------------
# Install Alex's NAV_ALEX package (idempotent — no files modified in his dir)
# ---------------------------------------------------------------------------
echo "Installing nav_locomotion package (pip install -e)..."
pip install -e "$NAV_ALEX_DIR/source/nav_locomotion/" --quiet
echo "Package installed."
echo ""

# ---------------------------------------------------------------------------
# Determine run mode and launch
# ---------------------------------------------------------------------------
MODE="${1:---local}"

if [ "$MODE" = "--local" ]; then
    echo "Mode: LOCAL smoke test (16 envs, 100 iterations)"
    echo ""
    launch_training python "$TRAIN_SCRIPT" \
        --headless \
        --loco_checkpoint "$LOCO_CHECKPOINT" \
        --num_envs 16 \
        --max_iterations 100 \
        --save_interval 50

elif [ "$MODE" = "--h100" ]; then
    echo "Mode: H100 full training run (2048 envs, 30000 iterations)"
    echo ""
    # Activate conda env on H100 (bashrc not sourced in non-interactive SSH)
    eval "$(/home/t2user/miniconda3/bin/conda shell.bash hook)"
    conda activate env_isaaclab

    launch_training python "$TRAIN_SCRIPT" \
        --headless \
        --loco_checkpoint "$LOCO_CHECKPOINT" \
        --num_envs 2048 \
        --max_iterations 30000 \
        --save_interval 100

else
    echo "ERROR: Unknown mode '$MODE'. Use --local or --h100."
    exit 1
fi
