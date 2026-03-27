#!/usr/bin/env bash
# install_prerequisites.sh
#
# One-time setup for the combined nav+loco training pipeline.
# Run this ONCE before your first training run (local or H100).
#
# IMPORTANT: For --local, your Isaac Sim venv must be active BEFORE running this
#            script, or it will activate isaacSim_env automatically. If neither
#            is possible, it will exit — never installs into the global Python.
#
# Usage:
#   Local  →  bash Experiments/Colby/CombinedPolicyTraining/install_prerequisites.sh --local
#   H100   →  bash Experiments/Colby/CombinedPolicyTraining/install_prerequisites.sh --h100

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
NAV_ALEX_DIR="$REPO_ROOT/Experiments/Alex/NAV_ALEX"
LOCO_CHECKPOINT="$REPO_ROOT/Experiments/Ryan/checkpoints/mason_hybrid_best_33200.pt"
COLBY_LOGS="$SCRIPT_DIR/logs"

MODE="${1:---local}"

echo "============================================================"
echo "  Combined Nav+Loco — Prerequisites Install"
echo "============================================================"
echo "  Repo root    : $REPO_ROOT"
echo "  NAV_ALEX dir : $NAV_ALEX_DIR"
echo "  Loco ckpt    : $LOCO_CHECKPOINT"
echo "  Log output   : $COLBY_LOGS"
echo "  Mode         : $MODE"
echo ""

# ---------------------------------------------------------------------------
# Step 0: Activate the right environment
# ---------------------------------------------------------------------------
echo "[0/6] Activating environment..."

if [ "$MODE" = "--h100" ]; then
    eval "$(/home/t2user/miniconda3/bin/conda shell.bash hook)"
    conda activate env_isaaclab
    PYTHON=python
    echo "      H100: conda env_isaaclab activated."
elif [ "$MODE" = "--local" ]; then
    # Resolve python directly from the venv — works even when PATH isn't updated by activate
    VENV_SCRIPTS="$REPO_ROOT/isaacSim_env/Scripts"
    if [ ! -f "$VENV_SCRIPTS/python.exe" ]; then
        echo ""
        echo "  ERROR: isaacSim_env not found at expected path:"
        echo "    $VENV_SCRIPTS/python.exe"
        echo ""
        echo "  Expected venv location: $REPO_ROOT/isaacSim_env"
        exit 1
    fi
    PYTHON="$VENV_SCRIPTS/python.exe"
    echo "      Local: using venv python at $PYTHON"
else
    echo "ERROR: Unknown mode '$MODE'. Use --local or --h100."
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1: Validate the loco checkpoint
# ---------------------------------------------------------------------------
echo ""
echo "[1/6] Checking loco checkpoint..."
if [ ! -f "$LOCO_CHECKPOINT" ]; then
    echo "  ERROR: Loco checkpoint not found:"
    echo "    $LOCO_CHECKPOINT"
    echo "  Make sure the repo is fully cloned and Ryan's checkpoints are present."
    exit 1
fi
echo "      OK: $(basename "$LOCO_CHECKPOINT") found."

# ---------------------------------------------------------------------------
# Step 2: Check Isaac Sim + Isaac Lab
# ---------------------------------------------------------------------------
echo ""
echo "[2/6] Checking Isaac Sim and Isaac Lab..."

_has_pkg() { "$PYTHON" -c "import importlib.util; exit(0 if importlib.util.find_spec('$1') else 1)" 2>/dev/null; }

# --- Isaac Sim ---
if _has_pkg isaacsim; then
    echo "      OK: isaacsim already installed."
else
    echo "      isaacsim not found — attempting pip install from NVIDIA index..."
    "$PYTHON" -m pip install isaacsim \
        --extra-index-url https://pypi.nvidia.com \
        --quiet \
        && echo "      OK: isaacsim installed." \
        || {
            echo ""
            echo "  WARNING: isaacsim pip install failed. If it's installed via the"
            echo "  Isaac Sim launcher (not pip), this is expected — Isaac Sim will"
            echo "  still be available at runtime."
        }
fi

# --- Isaac Lab ---
if _has_pkg isaaclab; then
    echo "      OK: isaaclab already installed."
else
    echo "      isaaclab not found — attempting pip install from NVIDIA index..."
    if "$PYTHON" -m pip install isaacsim-lab \
            --extra-index-url https://pypi.nvidia.com \
            --quiet 2>/dev/null; then
        echo "      OK: isaaclab installed from NVIDIA index."
    else
        echo "      NVIDIA index failed — attempting source install..."
        ISAACLAB_DIR="$REPO_ROOT/isaacSim_env/isaaclab_src"
        ISAACLAB_SRC="$ISAACLAB_DIR/source/isaaclab"
        if [ ! -d "$ISAACLAB_SRC" ]; then
            echo "      Cloning IsaacLab (shallow)..."
            git config --global core.longpaths true
            if ! git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR" --depth 1; then
                echo ""
                echo "  ERROR: git clone failed. Clone manually first, then re-run:"
                echo "    git config --global core.longpaths true"
                echo "    git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git \\"
                echo "      isaacSim_env\\isaaclab_src"
                echo "    Then re-run this script."
            fi
        else
            echo "      Source already cloned at $ISAACLAB_DIR"
        fi
        if [ -d "$ISAACLAB_SRC" ]; then
            ISAACLAB_WIN="$(wslpath -w "$ISAACLAB_SRC")"
            if "$PYTHON" -m pip install -e "$ISAACLAB_WIN" --quiet; then
                echo "      OK: isaaclab installed from source."
            else
                echo ""
                echo "  ERROR: isaaclab could not be installed. train_combined.py will fail."
                echo "  Manually run:"
                echo "    python -m pip install -e isaacSim_env\\isaaclab_src\\source\\isaaclab"
            fi
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Step 3: Install nav_locomotion package
# ---------------------------------------------------------------------------
echo ""
echo "[3/6] Installing nav_locomotion package (Alex's NAV_ALEX)..."
NAV_LOCO_WIN="$(wslpath -w "$NAV_ALEX_DIR/source/nav_locomotion")"
"$PYTHON" -m pip install -e "$NAV_LOCO_WIN" --quiet
echo "      OK: nav_locomotion installed (editable)."

# ---------------------------------------------------------------------------
# Step 4: Install remaining dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[4/6] Installing dependencies..."
"$PYTHON" -m pip install anthropic --quiet
"$PYTHON" -m pip install tensorboard --quiet
"$PYTHON" -m pip install gymnasium --quiet
"$PYTHON" -m pip install git+https://github.com/leggedrobotics/rsl_rl.git --quiet
echo "      OK: anthropic, tensorboard, gymnasium, rsl-rl installed."

# ---------------------------------------------------------------------------
# Step 5: Create Colby's log directory
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] Creating Colby's log directory..."
mkdir -p "$COLBY_LOGS"
echo "      OK: $COLBY_LOGS"
echo "      (train_combined.py writes all checkpoints here — no teammate dirs touched)"

# ---------------------------------------------------------------------------
# Step 5: Sanity check — verify key imports work
# ---------------------------------------------------------------------------
echo ""
echo "[6/6] Verifying key imports..."

"$PYTHON" - <<'PYCHECK'
import sys

failures = []

try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"  torch       : {torch.__version__}  (CUDA: {cuda_ok})")
    if not cuda_ok:
        print(f"  NOTE: CUDA not available. Training will run on CPU (slow).")
        print(f"        To enable GPU, reinstall torch with CUDA support:")
        print(f"          python -m pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print(f"        Replace cu121 with your CUDA version (check with: nvidia-smi)")
except ImportError as e:
    failures.append(f"torch: {e}")

try:
    import numpy as np
    print(f"  numpy       : {np.__version__}")
except ImportError as e:
    failures.append(f"numpy: {e}")

try:
    import anthropic
    print(f"  anthropic   : {anthropic.__version__}")
except ImportError as e:
    failures.append(f"anthropic: {e}")

try:
    import tensorboard
    print(f"  tensorboard : {tensorboard.__version__}")
except ImportError as e:
    failures.append(f"tensorboard: {e}")

try:
    import gymnasium
    print(f"  gymnasium      : {gymnasium.__version__}")
except ImportError as e:
    failures.append(f"gymnasium: {e}")

try:
    import rsl_rl
    ver = getattr(rsl_rl, '__version__', 'installed')
    print(f"  rsl_rl         : {ver}")
except ImportError as e:
    failures.append(f"rsl_rl: {e}")

try:
    import nav_locomotion
    print(f"  nav_locomotion : OK")
except ImportError as e:
    failures.append(f"nav_locomotion: {e}")

# Isaac Lab (bundled with Isaac Sim, not a standalone pip package — warn only)
try:
    import importlib.util
    if importlib.util.find_spec("isaaclab") is not None:
        print(f"  isaaclab    : found")
    else:
        print(f"  isaaclab    : not found as standalone package (OK if bundled with Isaac Sim)")
except Exception as e:
    print(f"  isaaclab    : check failed: {e}")

if failures:
    print("\n  FAILED IMPORTS:")
    for f in failures:
        print(f"    x {f}")
    sys.exit(1)
else:
    print("\n  All imports OK.")
PYCHECK

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Setup complete. You're ready to train."
echo ""
echo "  Run training:"
echo "    bash $SCRIPT_DIR/run_combined_nav_loco.sh --$( echo $MODE | tr -d '-' )"
echo ""
echo "  Monitor in a second terminal:"
echo "    python $SCRIPT_DIR/watch_training.py"
echo ""
echo "  Checkpoints will be saved to:"
echo "    $COLBY_LOGS/spot_nav_explore_ppo/<timestamp>/"
echo "============================================================"
