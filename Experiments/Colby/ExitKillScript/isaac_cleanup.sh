#!/usr/bin/env bash
# isaac_cleanup.sh
# =================
# Source this at the top of any Isaac Sim / Isaac Lab training shell script.
# It registers a trap that automatically kills orphaned Isaac Sim and omni
# kit processes when the script exits — whether training finishes normally,
# crashes, or is interrupted with Ctrl+C.
#
# Usage in your training script:
#
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "$SCRIPT_DIR/../../Colby/ExitKillScript/isaac_cleanup.sh"
#
#   launch_training python train.py --headless --num_envs 2048 ...
#
# That's it. No other changes needed.

# Holds the PID of the training process launched by launch_training()
_ISAAC_TRAIN_PID=""

# ---------------------------------------------------------------------------
# cleanup — called automatically on EXIT, SIGINT, SIGTERM
# ---------------------------------------------------------------------------
_isaac_cleanup() {
    echo ""
    echo "[CLEANUP] Killing orphaned Isaac Sim processes..."

    # Kill the training process group if still alive
    if [ -n "$_ISAAC_TRAIN_PID" ] && kill -0 "$_ISAAC_TRAIN_PID" 2>/dev/null; then
        kill -- "-$_ISAAC_TRAIN_PID" 2>/dev/null || true
    fi

    # Kill any Isaac / omni kit subprocesses owned by this user
    pkill -u "$(whoami)" -f "kit/python"    2>/dev/null || true
    pkill -u "$(whoami)" -f "omni.isaac"    2>/dev/null || true
    pkill -u "$(whoami)" -f "omni.kit"      2>/dev/null || true

    # Grace period, then force-kill anything still alive
    sleep 2
    pkill -9 -u "$(whoami)" -f "kit/python" 2>/dev/null || true
    pkill -9 -u "$(whoami)" -f "omni.isaac" 2>/dev/null || true
    pkill -9 -u "$(whoami)" -f "omni.kit"   2>/dev/null || true

    echo "[CLEANUP] Done."

    # Show remaining GPU processes so you know if the card is actually clear
    if command -v nvidia-smi &>/dev/null; then
        REMAINING=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
        if [ -n "$REMAINING" ]; then
            echo "[CLEANUP] WARNING — GPU processes still running:"
            echo "$REMAINING"
        else
            echo "[CLEANUP] GPU clear."
        fi
    fi
}

trap _isaac_cleanup EXIT SIGINT SIGTERM

# ---------------------------------------------------------------------------
# launch_training <command> [args...]
# ---------------------------------------------------------------------------
# Runs your python training command in its own process group (setsid),
# captures its PID for cleanup, waits for it to finish, and returns its
# exit code to your script.
#
# Example:
#   launch_training python train.py --headless --num_envs 2048
# ---------------------------------------------------------------------------
launch_training() {
    set +e
    setsid "$@" &
    _ISAAC_TRAIN_PID=$!
    wait "$_ISAAC_TRAIN_PID"
    local exit_code=$?
    set -e
    echo ""
    echo "[RUN] Training exited with code $exit_code"
    return $exit_code
}
