#!/bin/bash
# ==========================================================================
# Train all 6 S2R terrain experts sequentially on H100
#
# Total: ~6 x 17h = ~102h (~4.3 days)
#
# Usage:
#   cd ~/SIM_TO_REAL
#   screen -S s2r_train
#   bash scripts/train_all_experts.sh
#
# Created for AI2C Tech Capstone — MS for Autonomy, March 2026
# ==========================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S2R_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$S2R_ROOT"

COMMON_ARGS="--headless --no_wandb --num_envs 4096 --max_iterations 10000 --save_interval 100 --max_noise_std 0.5"

echo "=========================================="
echo "  S2R Expert Training Pipeline"
echo "  6 experts, sequential execution"
echo "  Started: $(date)"
echo "=========================================="

for EXPERT in friction stairs_up stairs_down boulders slopes mixed_rough; do
    echo ""
    echo "=========================================="
    echo "  Training expert: $EXPERT"
    echo "  Started: $(date)"
    echo "=========================================="

    python scripts/train_expert.py --expert_type "$EXPERT" $COMMON_ARGS

    echo "=========================================="
    echo "  Expert $EXPERT COMPLETE: $(date)"
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "  ALL 6 EXPERTS TRAINED SUCCESSFULLY"
echo "  Finished: $(date)"
echo "=========================================="
echo ""
echo "Next step: Run distillation"
echo "  python scripts/train_distill_s2r.py --headless --no_wandb \\"
echo "    --expert_friction checkpoints/expert_friction/best.pt \\"
echo "    --expert_stairs_up checkpoints/expert_stairs_up/best.pt \\"
echo "    --expert_stairs_down checkpoints/expert_stairs_down/best.pt \\"
echo "    --expert_boulders checkpoints/expert_boulders/best.pt \\"
echo "    --expert_slopes checkpoints/expert_slopes/best.pt \\"
echo "    --expert_mixed_rough checkpoints/expert_mixed_rough/best.pt"
