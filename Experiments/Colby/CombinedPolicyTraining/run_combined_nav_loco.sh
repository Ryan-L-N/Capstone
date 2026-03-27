#!/usr/bin/env bash
# run_combined_nav_loco.sh
#
# Launches Alex's NAV_ALEX navigation policy training on top of
# Ryan's mason_hybrid locomotion checkpoint (the frozen loco layer).
#
# No teammate files are modified. This script just wires the two
# together via CLI args and installs the package if needed.
#
# Usage (local smoke test, 100 iters, no GUI):
#   bash Experiments/Colby/CombinedPolicyTraining/run_combined_nav_loco.sh --local
#
# Usage (H100 full run):
#   bash Experiments/Colby/CombinedPolicyTraining/run_combined_nav_loco.sh --h100
#
# Architecture being trained:
#   Depth Camera (64x64) -> CNN Encoder -> 128-dim
#     + Proprioception (12-dim)
#           |
#   [Alex's Nav Policy, 10 Hz]   <- BEING TRAINED
#           |
#   Velocity Command [vx, vy, wz]
#           |
#   [Ryan's Frozen Loco Policy, 50 Hz]  <- FROZEN (mason_hybrid_best_33200.pt)
#           |
#   12 Joint Targets -> Spot

set -e

# ---------------------------------------------------------------------------
# Resolve paths relative to repo root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

NAV_ALEX_DIR="$REPO_ROOT/Experiments/Alex/NAV_ALEX"
LOCO_CHECKPOINT="$REPO_ROOT/Experiments/Ryan/checkpoints/mason_hybrid_best_33200.pt"
TRAIN_SCRIPT="$SCRIPT_DIR/train_combined.py"

echo "=== Combined Nav + Loco Training ==="
echo "Train script      : $TRAIN_SCRIPT"
echo "Loco checkpoint   : $LOCO_CHECKPOINT"
echo "Checkpoint output : $SCRIPT_DIR/logs/"
echo ""

# ---------------------------------------------------------------------------
# Validate that the loco checkpoint exists
# ---------------------------------------------------------------------------
if [ ! -f "$LOCO_CHECKPOINT" ]; then
    echo "ERROR: Loco checkpoint not found. Run install_prerequisites.sh first."
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
# Determine run mode
# ---------------------------------------------------------------------------
MODE="${1:---local}"

if [ "$MODE" = "--local" ]; then
    echo "Mode: LOCAL smoke test (16 envs, 100 iterations, no coach)"
    echo "Use --h100 for a full production run."
    echo ""
    python "$TRAIN_SCRIPT" \
        --headless \
        --no_coach \
        --loco_checkpoint "$LOCO_CHECKPOINT" \
        --num_envs 16 \
        --max_iterations 100 \
        --save_interval 50

elif [ "$MODE" = "--h100" ]; then
    echo "Mode: H100 full training run (2048 envs, 30000 iterations, with AI coach)"
    echo ""
    # Activate conda env on H100 (bashrc not sourced in non-interactive SSH)
    eval "$(/home/t2user/miniconda3/bin/conda shell.bash hook)"
    conda activate env_isaaclab

    python "$TRAIN_SCRIPT" \
        --headless \
        --loco_checkpoint "$LOCO_CHECKPOINT" \
        --num_envs 2048 \
        --max_iterations 30000 \
        --save_interval 100 \
        --coach_interval 250

else
    echo "ERROR: Unknown mode '$MODE'. Use --local or --h100."
    exit 1
fi
