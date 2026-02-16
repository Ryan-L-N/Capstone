#!/bin/bash
# ============================================================================
# Evaluate Spot Rough Terrain Checkpoints
# ============================================================================
#
# After 48h training completes, this script evaluates the top checkpoints
# to find the best-performing model.
#
# Usage:
#   bash ~/spot_48h/eval_checkpoints.sh <run_directory>
#
# Example:
#   bash ~/spot_48h/eval_checkpoints.sh 2026-02-13_10-00-00_48h_proprioception
#
# ============================================================================

set -e

RUN_DIR="${1:?Usage: eval_checkpoints.sh <run_directory_name>}"
LOG_ROOT="${HOME}/IsaacLab/logs/rsl_rl/spot_rough"
RUN_PATH="${LOG_ROOT}/${RUN_DIR}"

if [ ! -d "${RUN_PATH}" ]; then
    echo "[ERROR] Run directory not found: ${RUN_PATH}"
    echo "Available runs:"
    ls -1t "${LOG_ROOT}/" 2>/dev/null || echo "  (none)"
    exit 1
fi

echo "============================================================"
echo "  EVALUATING CHECKPOINTS"
echo "  Run: ${RUN_DIR}"
echo "============================================================"
echo ""

# ── Activate conda ──
eval "$(conda shell.bash hook)"
conda activate env_isaaclab

# ── Find checkpoints to evaluate ──
# Evaluate: last, and every 5000th iteration
CHECKPOINTS=()
for ckpt in "${RUN_PATH}"/model_*.pt; do
    basename=$(basename "$ckpt" .pt)
    iter=${basename#model_}
    # Evaluate every 5000 iterations + the last one
    if [[ "$iter" == "last" ]] || (( iter % 5000 == 0 && iter > 0 )); then
        CHECKPOINTS+=("$ckpt")
    fi
done

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "[ERROR] No checkpoints found in ${RUN_PATH}"
    exit 1
fi

echo "Evaluating ${#CHECKPOINTS[@]} checkpoints:"
for ckpt in "${CHECKPOINTS[@]}"; do
    echo "  - $(basename $ckpt)"
done
echo ""

# ── Run evaluations ──
cd ~/IsaacLab

for ckpt in "${CHECKPOINTS[@]}"; do
    ckpt_name=$(basename "$ckpt" .pt)
    echo "────────────────────────────────────────"
    echo "Evaluating: ${ckpt_name}"
    echo "────────────────────────────────────────"

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task=Isaac-Velocity-Rough-Spot-Play-v0 \
        --headless \
        --num_envs 50 \
        --load_run "${RUN_DIR}" \
        --load_checkpoint "${ckpt_name}.pt" \
        2>&1 | tail -20

    echo ""
done

echo "============================================================"
echo "  EVALUATION COMPLETE"
echo "  To copy the best checkpoint to your local machine:"
echo ""
echo "  scp t2user@172.24.254.24:${RUN_PATH}/model_XXXX.pt \\"
echo "      C:/IsaacLab/logs/rsl_rl/spot_rough/48h_run/model_best.pt"
echo "============================================================"
