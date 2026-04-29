#!/usr/bin/env bash
# ============================================================================
# H100 Parallel 4-Environment Evaluation
# ============================================================================
# Runs all 4 environments (friction, grass, boulder, stairs) in parallel
# on the H100 GPU (96 GB VRAM, ~4 GB per instance = ~16 GB total).
#
# Usage:
#   bash run_h100_parallel_eval.sh --debug          # 1 episode, sequential
#   bash run_h100_parallel_eval.sh                  # 100 episodes, parallel
#   bash run_h100_parallel_eval.sh --episodes 50    # custom episode count
#
# Outputs:
#   ~/4_env_test/results/mason_parallel_YYYY-MM-DD_HH-MM-SS/
#     ├── friction_rough_episodes.jsonl
#     ├── grass_rough_episodes.jsonl
#     ├── boulder_rough_episodes.jsonl
#     ├── stairs_rough_episodes.jsonl
#     ├── tensorboard/          (live TensorBoard events)
#     ├── friction_rough.log
#     ├── grass_rough.log
#     ├── boulder_rough.log
#     └── stairs_rough.log
# ============================================================================

set -euo pipefail

# ── Configuration ───────────────────────────────────────────────────────────
PROJECT_DIR="$HOME/4_env_test"
ISAACLAB_DIR="$HOME/IsaacLab"
ISAACLAB_PYTHON="${ISAACLAB_DIR}/isaaclab.sh -p"
EVAL_SCRIPT="${PROJECT_DIR}/src/run_capstone_eval.py"
TB_WATCHER="${PROJECT_DIR}/scripts/tb_watcher.py"
CHECKPOINT="$HOME/multi_robot_training_new/logs/rsl_rl/spot_hybrid_ppo/2026-03-11_11-28-30/model_19999.pt"
ENVS=(friction grass boulder stairs)
STAGGER_SECONDS=30
TB_PORT=6007

# ── Parse arguments ─────────────────────────────────────────────────────────
DEBUG=false
NUM_EPISODES=100
TIMEOUT_SEC=5400  # 90 minutes per env

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)     DEBUG=true; NUM_EPISODES=1; TIMEOUT_SEC=700; shift ;;
        --episodes)  NUM_EPISODES="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --timeout)   TIMEOUT_SEC="$2"; shift 2 ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR="${PROJECT_DIR}/results/mason_parallel_${TIMESTAMP}"

# ── Conda + environment setup ──────────────────────────────────────────────
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate env_isaaclab
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
CONDA_PYTHON="$(conda run -n env_isaaclab which python)"

# ── Preflight checks ───────────────────────────────────────────────────────
echo "============================================================"
echo "  H100 Parallel 4-Environment Evaluation"
echo "============================================================"
echo "  Mode:       $([ "$DEBUG" = true ] && echo 'DEBUG (1 ep, sequential)' || echo "PRODUCTION (${NUM_EPISODES} ep, parallel)")"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Timeout:    ${TIMEOUT_SEC}s per environment"
echo "  Envs:       ${ENVS[*]}"
echo "  TensorBoard: port ${TB_PORT}"
echo "============================================================"

# Verify files exist
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "[ERROR] Eval script not found: $EVAL_SCRIPT"
    exit 1
fi
if [ ! -f "$CHECKPOINT" ]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT"
    exit 1
fi
if [ ! -f "${ISAACLAB_DIR}/isaaclab.sh" ]; then
    echo "[ERROR] isaaclab.sh not found: ${ISAACLAB_DIR}/isaaclab.sh"
    exit 1
fi

# Clear pycache
echo "Clearing __pycache__..."
find "${PROJECT_DIR}/src" -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# Create output directory
mkdir -p "${OUTPUT_DIR}/tensorboard"

# Check GPU
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo ""

# ── TensorBoard watcher (background) ───────────────────────────────────────
# Tails JSONL files and writes TensorBoard events for live monitoring
echo "Starting TensorBoard watcher..."
${CONDA_PYTHON} "${TB_WATCHER}" \
    --results_dir "${OUTPUT_DIR}" \
    --tb_dir "${OUTPUT_DIR}/tensorboard" \
    --envs friction grass boulder stairs \
    --poll_interval 10 &
TB_WATCHER_PID=$!
echo "  TensorBoard watcher PID: ${TB_WATCHER_PID}"

# Start TensorBoard server
echo "Starting TensorBoard on port ${TB_PORT}..."
tensorboard --logdir "${OUTPUT_DIR}/tensorboard" --port ${TB_PORT} --bind_all \
    > "${OUTPUT_DIR}/tensorboard_server.log" 2>&1 &
TB_SERVER_PID=$!
echo "  TensorBoard server PID: ${TB_SERVER_PID}"
echo "  View at: http://$(hostname -I | awk '{print $1}'):${TB_PORT}"
echo ""

# ── Cleanup trap ────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[CLEANUP] Stopping TensorBoard watcher and server..."
    kill ${TB_WATCHER_PID} 2>/dev/null || true
    kill ${TB_SERVER_PID} 2>/dev/null || true
    # Don't kill eval processes here — they handle their own cleanup via os._exit
}
trap cleanup EXIT

# ── Run function ────────────────────────────────────────────────────────────
run_env() {
    local env_name=$1
    local log_file="${OUTPUT_DIR}/${env_name}_rough.log"

    echo "[${env_name}] Starting ${NUM_EPISODES} episodes..."

    timeout --foreground -k 30 ${TIMEOUT_SEC} \
        ${ISAACLAB_PYTHON} "${EVAL_SCRIPT}" \
            --headless --robot spot --policy rough --env "${env_name}" --mason \
            --num_episodes ${NUM_EPISODES} \
            --checkpoint "${CHECKPOINT}" \
            --output_dir "${OUTPUT_DIR}" \
        > "${log_file}" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[${env_name}] DONE (exit 0)"
    elif [ $exit_code -eq 124 ]; then
        echo "[${env_name}] TIMEOUT after ${TIMEOUT_SEC}s"
    else
        echo "[${env_name}] FAILED (exit ${exit_code})"
    fi
    return $exit_code
}

# ── Debug mode: sequential ──────────────────────────────────────────────────
if [ "$DEBUG" = true ]; then
    echo "=== DEBUG MODE: Running 1 episode per env, sequentially ==="
    echo ""

    PASS=0
    FAIL=0
    for env in "${ENVS[@]}"; do
        start_ts=$(date +%s)
        if run_env "$env"; then
            elapsed=$(( $(date +%s) - start_ts ))
            echo "  [${env}] OK (${elapsed}s)"
            PASS=$((PASS + 1))
        else
            elapsed=$(( $(date +%s) - start_ts ))
            echo "  [${env}] FAILED (${elapsed}s) — check ${OUTPUT_DIR}/${env}_rough.log"
            FAIL=$((FAIL + 1))
        fi
        echo ""
    done

    echo "============================================================"
    echo "  Debug complete: ${PASS} passed, ${FAIL} failed"
    if [ $FAIL -gt 0 ]; then
        echo "  Fix failures before running production!"
        exit 1
    else
        echo "  All 4 envs passed. Run without --debug for production."
    fi
    echo "============================================================"
    exit 0
fi

# ── Production mode: parallel ───────────────────────────────────────────────
echo "=== PRODUCTION MODE: ${NUM_EPISODES} episodes x 4 envs, parallel ==="
echo ""

declare -A PIDS
declare -A START_TIMES

LAUNCH_START=$(date +%s)

for env in "${ENVS[@]}"; do
    START_TIMES[$env]=$(date +%s)
    run_env "$env" &
    PIDS[$env]=$!
    echo "  Launched ${env} (PID ${PIDS[$env]})"

    # Stagger launches to avoid CUDA initialization race
    if [ "$env" != "${ENVS[-1]}" ]; then
        echo "  Waiting ${STAGGER_SECONDS}s before next launch..."
        sleep ${STAGGER_SECONDS}
    fi
done

echo ""
echo "All 4 environments launched. Monitoring..."
echo ""

# ── Monitor loop ────────────────────────────────────────────────────────────
declare -A EXIT_CODES
declare -A FINISHED
DONE_COUNT=0

while [ $DONE_COUNT -lt ${#ENVS[@]} ]; do
    sleep 30

    echo "--- Status check ($(date '+%H:%M:%S')) ---"
    for env in "${ENVS[@]}"; do
        if [ "${FINISHED[$env]:-}" = "true" ]; then
            continue
        fi

        if ! kill -0 ${PIDS[$env]} 2>/dev/null; then
            # Process finished
            wait ${PIDS[$env]} 2>/dev/null
            EXIT_CODES[$env]=$?
            FINISHED[$env]="true"
            DONE_COUNT=$((DONE_COUNT + 1))

            elapsed=$(( $(date +%s) - ${START_TIMES[$env]} ))
            if [ ${EXIT_CODES[$env]} -eq 0 ]; then
                echo "  [${env}] COMPLETED in ${elapsed}s"
            else
                echo "  [${env}] EXITED with code ${EXIT_CODES[$env]} after ${elapsed}s"
            fi
        else
            # Still running — show latest progress
            latest=$(grep -oP 'progress=\K[0-9.]+' "${OUTPUT_DIR}/${env}_rough.log" 2>/dev/null | tail -1)
            ep_count=$(grep -c '^\s*\[' "${OUTPUT_DIR}/${env}_rough.log" 2>/dev/null || echo 0)
            echo "  [${env}] running... ep ~${ep_count}/${NUM_EPISODES}, last progress: ${latest:-?}m"
        fi
    done
    echo ""
done

# ── Final report ────────────────────────────────────────────────────────────
TOTAL_TIME=$(( $(date +%s) - LAUNCH_START ))

echo "============================================================"
echo "  EVALUATION COMPLETE"
echo "============================================================"
echo ""

printf "  %-12s %-10s %-10s %-12s %-8s\n" "Environment" "Status" "Episodes" "JSONL Lines" "Time"
printf "  %-12s %-10s %-10s %-12s %-8s\n" "-----------" "------" "--------" "-----------" "----"

for env in "${ENVS[@]}"; do
    elapsed=$(( $(date +%s) - ${START_TIMES[$env]} ))
    jsonl_file="${OUTPUT_DIR}/${env}_rough_episodes.jsonl"

    if [ -f "$jsonl_file" ]; then
        lines=$(wc -l < "$jsonl_file")
        status="OK"
    else
        lines=0
        status="NO_DATA"
    fi

    if [ "${EXIT_CODES[$env]:-999}" -ne 0 ]; then
        status="FAILED"
    fi

    printf "  %-12s %-10s %-10s %-12s %-8s\n" "$env" "$status" "${lines}" "${lines} lines" "${elapsed}s"
done

echo ""
echo "  Total wall-clock: ${TOTAL_TIME}s ($(( TOTAL_TIME / 60 ))m $(( TOTAL_TIME % 60 ))s)"
echo "  Results: ${OUTPUT_DIR}"
echo "  TensorBoard: http://$(hostname -I | awk '{print $1}'):${TB_PORT}"
echo ""
echo "  To download results:"
echo "    scp -r t2user@$(hostname -I | awk '{print $1}'):${OUTPUT_DIR} ."
echo "============================================================"
