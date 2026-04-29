#!/bin/bash
# ==============================================================================
# 48-Hour Training Monitor Script
# ==============================================================================
# Run this script to check on the training progress.
# IMPORTANT: Use this from a SEPARATE SSH session, not the screen session.
#            Remember: ONE SSH session at a time rule. Disconnect after checking.
#
# Usage:
#   ssh t2user@172.24.254.24
#   bash monitor_48h.sh          # One-shot status check
#   bash monitor_48h.sh --loop   # Continuous monitoring (Ctrl+C to stop)
# ==============================================================================

LOOP_MODE=false
if [ "$1" == "--loop" ]; then
    LOOP_MODE=true
fi

check_status() {
    echo "========================================================================"
    echo "  48-HOUR TRAINING STATUS"
    echo "  $(date)"
    echo "========================================================================"

    # ─── GPU Status ──────────────────────────────────────────────────────
    echo ""
    echo "--- GPU Status ---"
    nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
        --format=csv 2>/dev/null || echo "  nvidia-smi unavailable"

    # ─── Training Process ────────────────────────────────────────────────
    echo ""
    echo "--- Training Process ---"
    TRAIN_PID=$(pgrep -f "train_48h_spot.py" 2>/dev/null)
    if [ -n "$TRAIN_PID" ]; then
        echo "  Status: RUNNING (PID: $TRAIN_PID)"
        # Process runtime
        ELAPSED=$(ps -o etime= -p "$TRAIN_PID" 2>/dev/null | tr -d ' ')
        echo "  Runtime: $ELAPSED"
        # Memory usage
        RSS=$(ps -o rss= -p "$TRAIN_PID" 2>/dev/null | tr -d ' ')
        if [ -n "$RSS" ]; then
            echo "  RAM usage: $((RSS / 1024)) MB"
        fi
    else
        echo "  Status: NOT RUNNING"
        echo "  Check screen session: screen -r train48h"
    fi

    # ─── Screen Sessions ─────────────────────────────────────────────────
    echo ""
    echo "--- Screen Sessions ---"
    screen -ls 2>/dev/null || echo "  No screen sessions found"

    # ─── Latest Checkpoints ──────────────────────────────────────────────
    echo ""
    echo "--- Latest Checkpoints ---"
    LOG_BASE="/home/t2user/IsaacLab/logs/rsl_rl/spot_48h"
    if [ -d "$LOG_BASE" ]; then
        LATEST_DIR=$(ls -td "$LOG_BASE"/*/ 2>/dev/null | head -1)
        if [ -n "$LATEST_DIR" ]; then
            echo "  Log dir: $LATEST_DIR"
            # Count checkpoints
            NUM_CKPT=$(find "$LATEST_DIR" -name "model_*.pt" 2>/dev/null | wc -l)
            echo "  Checkpoints saved: $NUM_CKPT"
            # Latest checkpoint
            LATEST_CKPT=$(ls -t "$LATEST_DIR"/model_*.pt 2>/dev/null | head -1)
            if [ -n "$LATEST_CKPT" ]; then
                CKPT_TIME=$(stat -c '%Y' "$LATEST_CKPT" 2>/dev/null)
                CKPT_DATE=$(date -d "@$CKPT_TIME" '+%Y-%m-%d %H:%M:%S' 2>/dev/null)
                CKPT_NAME=$(basename "$LATEST_CKPT")
                echo "  Latest: $CKPT_NAME ($CKPT_DATE)"
                # Estimate progress (model_XXXX.pt where XXXX is iteration)
                ITER_NUM=$(echo "$CKPT_NAME" | grep -oP '\d+')
                if [ -n "$ITER_NUM" ]; then
                    PROGRESS=$((ITER_NUM * 100 / 17000))
                    echo "  Progress: $ITER_NUM / 17000 iterations ($PROGRESS%)"
                    # Estimate remaining time
                    NOW=$(date +%s)
                    TRAIN_START=$(stat -c '%Y' "$LATEST_DIR" 2>/dev/null)
                    if [ -n "$TRAIN_START" ] && [ "$ITER_NUM" -gt 0 ]; then
                        ELAPSED_S=$((NOW - TRAIN_START))
                        SEC_PER_ITER=$((ELAPSED_S / ITER_NUM))
                        REMAINING_ITERS=$((17000 - ITER_NUM))
                        REMAINING_S=$((REMAINING_ITERS * SEC_PER_ITER))
                        REMAINING_H=$((REMAINING_S / 3600))
                        echo "  Avg iter time: ${SEC_PER_ITER}s"
                        echo "  Est. remaining: ~${REMAINING_H} hours"
                    fi
                fi
            fi
            # Disk usage
            DIR_SIZE=$(du -sh "$LATEST_DIR" 2>/dev/null | cut -f1)
            echo "  Disk usage: $DIR_SIZE"
        fi
    else
        echo "  No training logs found at $LOG_BASE"
    fi

    # ─── Latest stdout log ───────────────────────────────────────────────
    echo ""
    echo "--- Latest Training Output (last 10 lines) ---"
    LATEST_LOG=$(ls -t "$LOG_BASE"/training_stdout_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        tail -10 "$LATEST_LOG"
    else
        echo "  No stdout log found"
    fi

    # ─── System Health ───────────────────────────────────────────────────
    echo ""
    echo "--- System Health ---"
    echo "  CPU load: $(uptime | awk -F'load average:' '{print $2}')"
    echo "  RAM: $(free -h | awk '/Mem:/ {printf "%s used / %s total (%s free)", $3, $2, $4}')"
    echo "  Disk: $(df -h /home/t2user | awk 'NR==2 {printf "%s used / %s total (%s free)", $3, $2, $4}')"

    echo ""
    echo "========================================================================"
}

if [ "$LOOP_MODE" = true ]; then
    echo "Continuous monitoring mode. Press Ctrl+C to stop."
    echo "Refreshing every 60 seconds..."
    while true; do
        clear
        check_status
        sleep 60
    done
else
    check_status
fi
