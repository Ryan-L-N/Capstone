#!/usr/bin/env bash
# run_h100_eval.sh — 100 episodes x 8 combos on H100 in a screen session
#
# Launches a screen session "eval4env" that runs all 8 policy/environment
# combinations (100 episodes each), then generates the report + plots.
#
# Usage:
#   ssh t2user@172.24.254.24
#   bash ~/4_env_test/scripts/run_h100_eval.sh
#
# Then:
#   screen -r eval4env       # Attach to eval output
#   Ctrl+A, D                # Detach
#
# Created for AI2C Tech Capstone — MS for Autonomy, February 2026

set -euo pipefail

PROJECT_DIR="$HOME/4_env_test"
ISAACLAB_DIR="$HOME/IsaacLab"

echo "============================================"
echo "  4-ENV EVAL — Launching on H100"
echo "============================================"
echo "  Project:    ${PROJECT_DIR}"
echo "  IsaacLab:   ${ISAACLAB_DIR}"
echo "  Episodes:   100 per combo (800 total)"
echo "============================================"

# Pre-flight
if [ ! -f "${PROJECT_DIR}/src/run_capstone_eval.py" ]; then
    echo "[ERROR] Eval script not found: ${PROJECT_DIR}/src/run_capstone_eval.py"
    exit 1
fi

if [ ! -f "${PROJECT_DIR}/src/checkpoints/model_29999.pt" ]; then
    echo "[ERROR] Checkpoint not found: ${PROJECT_DIR}/src/checkpoints/model_29999.pt"
    exit 1
fi

# Kill any existing eval screen
screen -S eval4env -X quit 2>/dev/null || true

echo "[INFO] Starting eval in screen session 'eval4env'..."
screen -dmS eval4env bash -c "
    source /home/t2user/miniconda3/etc/profile.d/conda.sh
    conda activate env_isaaclab
    export OMNI_KIT_ACCEPT_EULA=YES
    export PYTHONUNBUFFERED=1

    PROJECT_DIR=\"$HOME/4_env_test\"
    ISAACLAB_DIR=\"$HOME/IsaacLab\"
    TIMESTAMP=\$(date +\"%Y-%m-%d_%H-%M-%S\")
    OUTPUT_DIR=\"\${PROJECT_DIR}/results/h100_eval_\${TIMESTAMP}\"
    mkdir -p \"\${OUTPUT_DIR}\"

    NUM_EPISODES=100
    POLICIES=(flat rough)
    ENVIRONMENTS=(friction grass boulder stairs)

    echo '============================================'
    echo '  4-ENV CAPSTONE EVAL — H100'
    echo '  Started at '\$(date)
    echo '  Episodes: 100 per combo (800 total)'
    echo '  Output: '\${OUTPUT_DIR}
    echo '============================================'
    echo ''

    TOTAL=\$(( \${#POLICIES[@]} * \${#ENVIRONMENTS[@]} ))
    CURRENT=0
    FAILED=0
    START_TIME=\$(date +%s)

    for POLICY in \"\${POLICIES[@]}\"; do
        for ENV in \"\${ENVIRONMENTS[@]}\"; do
            CURRENT=\$((CURRENT + 1))
            COMBO=\"\${ENV}_\${POLICY}\"
            LOG_FILE=\"\${OUTPUT_DIR}/\${COMBO}.log\"

            echo \"--- [\${CURRENT}/\${TOTAL}] env=\${ENV} policy=\${POLICY} ---\"
            echo \"  Started: \$(date)\"
            echo \"  Episodes: \${NUM_EPISODES}\"

            COMBO_START=\$(date +%s)

            cd \"\${ISAACLAB_DIR}\"

            if timeout --foreground -k 30 7200 ./isaaclab.sh -p \"\${PROJECT_DIR}/src/run_capstone_eval.py\" --headless \
                --num_episodes \"\${NUM_EPISODES}\" \
                --policy \"\${POLICY}\" \
                --env \"\${ENV}\" \
                --output_dir \"\${OUTPUT_DIR}\" \
                > \"\${LOG_FILE}\" 2>&1; then

                COMBO_END=\$(date +%s)
                COMBO_TIME=\$(( (COMBO_END - COMBO_START) / 60 ))
                echo \"  PASSED: \${COMBO} (\${COMBO_TIME}min)\"
                tail -5 \"\${LOG_FILE}\" | grep -E 'ep[0-9]|Saved|Evaluation|Exiting|complete' || true
            else
                COMBO_END=\$(date +%s)
                COMBO_TIME=\$(( (COMBO_END - COMBO_START) / 60 ))
                echo \"  FAILED: \${COMBO} (exit \$?, \${COMBO_TIME}min)\"
                FAILED=\$((FAILED + 1))
                tail -5 \"\${LOG_FILE}\" || true
                pkill -f 'run_capstone_eval' 2>/dev/null || true
                sleep 5
            fi

            # ETA
            ELAPSED=\$(( \$(date +%s) - START_TIME ))
            if [ \"\${CURRENT}\" -lt \"\${TOTAL}\" ]; then
                AVG=\$(( ELAPSED / CURRENT ))
                ETA=\$(( (AVG * (TOTAL - CURRENT)) / 60 ))
                echo \"  Elapsed: \$((ELAPSED / 60))min | ETA: ~\${ETA}min\"
            fi
            echo ''
            sleep 3
        done
    done

    END_TIME=\$(date +%s)
    TOTAL_MIN=\$(( (END_TIME - START_TIME) / 60 ))

    echo '============================================'
    echo '  ALL COMBOS COMPLETE'
    echo \"  Total time: \${TOTAL_MIN} minutes\"
    echo \"  Failed: \${FAILED} / \${TOTAL}\"
    echo '============================================'
    echo ''

    # --- Generate report ---
    echo 'Generating report...'
    REPORT_DIR=\"\${OUTPUT_DIR}/report\"
    if python \"\${PROJECT_DIR}/src/metrics/reporter.py\" --input \"\${OUTPUT_DIR}\" --output \"\${REPORT_DIR}\"; then
        echo ''
        echo 'Report generated:'
        ls -la \"\${REPORT_DIR}/\" 2>/dev/null || true
        if [ -d \"\${REPORT_DIR}/plots\" ]; then
            echo ''
            echo 'Plots:'
            ls -la \"\${REPORT_DIR}/plots/\" 2>/dev/null || true
        fi
    else
        echo 'WARNING: Report generation failed'
    fi

    echo ''
    echo '============================================'
    echo '  4-ENV EVAL COMPLETE'
    echo \"  Results: \${OUTPUT_DIR}\"
    echo \"  Report:  \${OUTPUT_DIR}/report/\"
    echo \"  Plots:   \${OUTPUT_DIR}/report/plots/\"
    echo '============================================'
    echo 'Press Enter to close this screen session...'
    read
"

echo ""
echo "============================================"
echo "  EVAL LAUNCHED"
echo "============================================"
echo ""
echo "  Attach:     screen -r eval4env"
echo "  Detach:     Ctrl+A, D"
echo "  List:       screen -ls"
echo "============================================"
