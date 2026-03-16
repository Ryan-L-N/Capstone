#!/bin/bash
# Run Mason Baseline (model_19999) through all 4 environments, 100 episodes each, headless.
# Usage: bash scripts/run_mason_baseline_100ep.sh

PYTHON="/c/miniconda3/envs/isaaclab311/python"
SCRIPT="C:/Users/Gabriel Santiago/OneDrive/Desktop/Capstone Project/Capstone/Experiments/Alex/4_env_test/src/run_capstone_eval.py"
CHECKPOINT="C:/Users/Gabriel Santiago/OneDrive/Desktop/Capstone Project/Capstone/Experiments/Alex/multi_robot_training/checkpoints/mason_baseline_final_19999.pt"
OUTPUT_DIR="C:/Users/Gabriel Santiago/OneDrive/Desktop/Capstone Project/Capstone/Experiments/Alex/4_env_test/results/mason_baseline_100ep"

# Clear pycache
rm -rf "C:/Users/Gabriel Santiago/OneDrive/Desktop/Capstone Project/Capstone/Experiments/Alex/4_env_test/src/configs/__pycache__"
rm -rf "C:/Users/Gabriel Santiago/OneDrive/Desktop/Capstone Project/Capstone/Experiments/Alex/4_env_test/src/__pycache__"
rm -rf "C:/Users/Gabriel Santiago/OneDrive/Desktop/Capstone Project/Capstone/Experiments/Alex/4_env_test/src/envs/__pycache__"
rm -rf "C:/Users/Gabriel Santiago/OneDrive/Desktop/Capstone Project/Capstone/Experiments/Alex/4_env_test/src/navigation/__pycache__"
rm -rf "C:/Users/Gabriel Santiago/OneDrive/Desktop/Capstone Project/Capstone/Experiments/Alex/4_env_test/src/metrics/__pycache__"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "  Mason Baseline 100-Episode Evaluation"
echo "  Checkpoint: mason_baseline_final_19999.pt"
echo "  Output:     $OUTPUT_DIR"
echo "  Started:    $(date)"
echo "=============================================="

for ENV in friction grass boulder stairs; do
    echo ""
    echo ">>> Starting $ENV (100 episodes) at $(date) ..."
    "$PYTHON" "$SCRIPT" \
        --robot spot --policy rough --env "$ENV" --mason --headless \
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
