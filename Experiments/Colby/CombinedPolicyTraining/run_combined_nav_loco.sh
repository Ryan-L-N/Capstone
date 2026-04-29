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
# Resolve Python and validate loco checkpoint
# ---------------------------------------------------------------------------
MODE="${1:---local}"

if [ "$MODE" = "--local" ]; then
    VENV_SCRIPTS="$REPO_ROOT/isaacSim_env/Scripts"
    if [ ! -f "$VENV_SCRIPTS/python.exe" ]; then
        echo "ERROR: isaacSim_env not found at: $VENV_SCRIPTS"
        exit 1
    fi
    PYTHON="$VENV_SCRIPTS/python.exe"
elif [ "$MODE" = "--h100" ]; then
    eval "$(/home/t2user/miniconda3/bin/conda shell.bash hook)"
    conda activate env_isaaclab
    PYTHON=python
else
    echo "ERROR: Unknown mode '$MODE'. Use --local or --h100."
    exit 1
fi

echo "=== Combined Nav + Loco Training ==="
echo "Train script      : $TRAIN_SCRIPT"
echo "Loco checkpoint   : $LOCO_CHECKPOINT"
echo "Checkpoint output : $SCRIPT_DIR/logs/"
echo "Python            : $PYTHON"
echo ""

if [ ! -f "$LOCO_CHECKPOINT" ]; then
    echo "ERROR: Loco checkpoint not found: $LOCO_CHECKPOINT"
    exit 1
fi

# ---------------------------------------------------------------------------
# Install Alex's NAV_ALEX package (idempotent — no files modified in his dir)
# ---------------------------------------------------------------------------
echo "Installing nav_locomotion package (pip install -e)..."
NAV_LOCO_WIN="$(wslpath -w "$NAV_ALEX_DIR/source/nav_locomotion")"
"$PYTHON" -m pip install -e "$NAV_LOCO_WIN" --quiet
echo "Package installed."
echo ""

# ---------------------------------------------------------------------------
# Launch training
# ---------------------------------------------------------------------------
if [ "$MODE" = "--local" ]; then
    echo "Mode: LOCAL smoke test (16 envs, 100 iterations)"
    echo ""
    TRAIN_SCRIPT_WIN="$(wslpath -w "$TRAIN_SCRIPT")"
    LOCO_CHECKPOINT_WIN="$(wslpath -w "$LOCO_CHECKPOINT")"
    launch_training "$PYTHON" "$TRAIN_SCRIPT_WIN" \
        --headless \
        --loco_checkpoint "$LOCO_CHECKPOINT_WIN" \
        --num_envs 16 \
        --max_iterations 100 \
        --save_interval 50

elif [ "$MODE" = "--h100" ]; then
    echo "Mode: H100 full training run (2048 envs, 30000 iterations)"
    echo ""
    launch_training "$PYTHON" "$TRAIN_SCRIPT" \
        --headless \
        --loco_checkpoint "$LOCO_CHECKPOINT" \
        --num_envs 2048 \
        --max_iterations 30000 \
        --save_interval 100
fi
