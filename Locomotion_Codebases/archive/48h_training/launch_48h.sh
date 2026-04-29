#!/bin/bash
# ==============================================================================
# 48-Hour ANYmal-C Training Launch Script
# ==============================================================================
# Deploys and runs the optimized 48h training on the H100 server.
# Designed to be run inside a GNU screen session for persistence.
#
# Usage:
#   screen -S train48h
#   bash launch_48h.sh
#   # Press Ctrl+A, D to detach
#   # screen -r train48h to reattach
#
# To resume from checkpoint:
#   bash launch_48h.sh --resume
#   bash launch_48h.sh --resume --checkpoint /path/to/model_XXXX.pt
# ==============================================================================

set -euo pipefail

# ─── Environment Setup ───────────────────────────────────────────────────────
echo "========================================================================"
echo "  48-HOUR TRAINING LAUNCHER"
echo "  $(date)"
echo "========================================================================"

# Source conda (required for non-interactive shells)
source /home/t2user/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# Set required environment variables
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

# Verify GPU is available
echo ""
echo "--- GPU Status ---"
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader
echo ""

# Verify Python environment
echo "--- Python Environment ---"
echo "  Python: $(python --version 2>&1)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo ""

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_PATH="/home/t2user/train_48h_spot.py"
LOG_BASE="/home/t2user/IsaacLab/logs/rsl_rl/spot_48h"

# Create log directory
mkdir -p "$LOG_BASE"

# ─── Pre-flight Checks ───────────────────────────────────────────────────────
echo "--- Pre-flight Checks ---"

# Check training script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Training script not found at $SCRIPT_PATH"
    echo "Upload it first:  scp train_48h_spot.py t2user@172.24.254.24:~/"
    exit 1
fi

# Check available disk space (need at least 20GB for checkpoints)
AVAIL_GB=$(df /home/t2user --output=avail -BG | tail -1 | tr -d ' G')
echo "  Available disk space: ${AVAIL_GB}GB"
if [ "$AVAIL_GB" -lt 20 ]; then
    echo "WARNING: Less than 20GB free. Checkpoints may fill disk."
fi

# Check GPU memory is mostly free
GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "  GPU memory in use: ${GPU_USED}MiB"
if [ "$GPU_USED" -gt 5000 ]; then
    echo "WARNING: GPU has ${GPU_USED}MiB in use. Kill other processes first."
fi

echo "  All checks passed."
echo ""

# ─── Launch Training ──────────────────────────────────────────────────────────
echo "========================================================================"
echo "  LAUNCHING 48-HOUR TRAINING"
echo "  Start: $(date)"
echo "  Script: $SCRIPT_PATH"
echo "  Args: $@"
echo "========================================================================"
echo ""

# Change to IsaacLab directory (some imports expect this)
cd /home/t2user/IsaacLab

# Run training with all passed arguments
# Output is both displayed and logged to a file
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_FILE="$LOG_BASE/training_stdout_${TIMESTAMP}.log"

python "$SCRIPT_PATH" \
    --num_envs 8192 \
    --max_iterations 17000 \
    "$@" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# ─── Post-Training ───────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "  TRAINING FINISHED"
echo "  End: $(date)"
echo "  Exit code: $EXIT_CODE"
echo "  Stdout log: $LOG_FILE"
echo "========================================================================"

# Final GPU status
echo ""
echo "--- Final GPU Status ---"
nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw \
    --format=csv,noheader

exit $EXIT_CODE
