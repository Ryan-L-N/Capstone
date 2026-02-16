#!/bin/bash
# ============================================================================
# 48-Hour Spot Rough Terrain Training — H100 NVL
# ============================================================================
#
# Usage:
#   1. scp this folder to the H100 server:
#      scp -r 48h_training/ t2user@172.24.254.24:~/spot_48h/
#
#   2. SSH to the server:
#      ssh t2user@172.24.254.24
#
#   3. Start a screen session:
#      screen -S spot_48h
#
#   4. Run this script:
#      bash ~/spot_48h/train_spot_rough_48h.sh
#
#   5. Detach: Ctrl+A, then D
#   6. Reconnect later: screen -r spot_48h
#
# ============================================================================

set -e

# ── Configuration ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_SCRIPT="${SCRIPT_DIR}/spot_rough_48h_cfg.py"
NUM_ENVS=8192
MAX_ITERATIONS=30000

echo "============================================================"
echo "  48-HOUR SPOT ROUGH TERRAIN TRAINING"
echo "  Server: $(hostname)"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Date:   $(date)"
echo "============================================================"
echo ""

# ── Activate conda ──
echo "[1/4] Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate env_isaaclab

echo "[2/4] Verifying environment..."
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import isaacsim; print('  Isaac Sim: OK')"
python -c "import isaaclab; print(f'  Isaac Lab: {isaaclab.__version__}')"
echo ""

# ── GPU status ──
echo "[3/4] GPU status before training:"
nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total --format=csv
echo ""

# ── Launch training ──
echo "[4/4] Launching training..."
echo "  Script:     ${TRAINING_SCRIPT}"
echo "  Envs:       ${NUM_ENVS}"
echo "  Iterations: ${MAX_ITERATIONS}"
echo "  Estimated:  ~48 hours"
echo ""
echo "  Logs: ~/IsaacLab/logs/rsl_rl/spot_rough/"
echo ""
echo "============================================================"
echo "  Training started at: $(date)"
echo "============================================================"
echo ""

cd ~/IsaacLab

./isaaclab.sh -p "${TRAINING_SCRIPT}" \
    --headless \
    --num_envs ${NUM_ENVS} \
    --max_iterations ${MAX_ITERATIONS}

echo ""
echo "============================================================"
echo "  Training finished at: $(date)"
echo "============================================================"
