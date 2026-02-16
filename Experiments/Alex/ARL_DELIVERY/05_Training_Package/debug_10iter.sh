#!/bin/bash
# ============================================================================
# DEBUG RUN: 10 iterations to verify training pipeline
# ============================================================================
#
# Usage:
#   bash ~/spot_48h/debug_10iter.sh
#
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_SCRIPT="${SCRIPT_DIR}/spot_rough_48h_cfg.py"

echo "============================================================"
echo "  DEBUG RUN: 10 iterations"
echo "  Date: $(date)"
echo "============================================================"

# ── Activate conda ──
eval "$(conda shell.bash hook)"
conda activate env_isaaclab

echo "[INFO] Verifying environment..."
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import isaacsim; print('  Isaac Sim: OK')"

echo ""
echo "[INFO] GPU status:"
nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total --format=csv
echo ""

cd ~/IsaacLab

echo "[INFO] Starting 10-iteration debug run..."
echo ""

./isaaclab.sh -p "${TRAINING_SCRIPT}" \
    --headless \
    --num_envs 4096 \
    --max_iterations 10

echo ""
echo "============================================================"
echo "  DEBUG RUN COMPLETE at $(date)"
echo "  Check logs: ls -lt ~/IsaacLab/logs/rsl_rl/spot_rough/"
echo "============================================================"
