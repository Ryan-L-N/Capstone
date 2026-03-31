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
    echo "      H100: conda env_isaaclab activated."
elif [ "$MODE" = "--local" ]; then
    # Activate local venv if not already active
    if [ -z "$VIRTUAL_ENV" ]; then
        VENV="$REPO_ROOT/../isaacSim_env/Scripts/activate"
        if [ -f "$VENV" ]; then
            source "$VENV"
            echo "      Local: isaacSim_env activated."
        else
            echo ""
            echo "  ERROR: No virtual environment is active and isaacSim_env was not"
            echo "  found at the expected path:"
            echo "    $VENV"
            echo ""
            echo "  Activate your Isaac Sim venv first, then re-run this script:"
            echo "    source <path/to/isaacSim_env>/Scripts/activate   # Windows/Git Bash"
            echo "    source <path/to/isaacSim_env>/bin/activate        # Linux/Mac"
            echo ""
            echo "  This is required to prevent packages from polluting your global Python."
            exit 1
        fi
    else
        echo "      Local: venv already active ($VIRTUAL_ENV)."
    fi
    # Double-check we are actually inside a venv before installing anything
    if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
        echo "  ERROR: Could not confirm a virtual environment is active. Aborting."
        exit 1
    fi
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
# Step 2: Install Isaac Lab (if not already present)
# ---------------------------------------------------------------------------
echo ""
echo "[2/6] Checking / installing Isaac Lab..."

if python -c "import importlib.util; exit(0 if importlib.util.find_spec('isaaclab') else 1)" 2>/dev/null; then
    echo "      OK: isaaclab already installed."
else
    echo "      Not found — attempting pip install from NVIDIA index..."
    pip install isaacsim-lab \
        --extra-index-url https://pypi.nvidia.com \
        --quiet \
        && echo "      OK: isaaclab installed." \
        || {
            echo ""
            echo "  WARNING: Isaac Lab pip install failed."
            echo "  This is required for training. Manual install options:"
            echo "    pip install isaacsim-lab --extra-index-url https://pypi.nvidia.com"
            echo "    or install from source: https://isaac-sim.github.io/IsaacLab"
            echo "  Continuing setup — other packages will still be installed."
        }
fi

# ---------------------------------------------------------------------------
# Step 3: Install nav_locomotion package
# ---------------------------------------------------------------------------
echo ""
echo "[3/6] Installing nav_locomotion package (Alex's NAV_ALEX)..."
pip install -e "$NAV_ALEX_DIR/source/nav_locomotion/" --quiet
echo "      OK: nav_locomotion installed (editable)."

# ---------------------------------------------------------------------------
# Step 4: Install remaining dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[4/6] Installing dependencies..."
pip install anthropic --quiet
pip install tensorboard --quiet
pip install gymnasium --quiet
pip install rsl-rl --quiet
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

python - <<'PYCHECK'
import sys

failures = []

try:
    import torch
    print(f"  torch       : {torch.__version__}  (CUDA: {torch.cuda.is_available()})")
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
    print(f"  rsl_rl         : {rsl_rl.__version__}")
except ImportError as e:
    failures.append(f"rsl_rl: {e}")

try:
    import nav_locomotion
    print(f"  nav_locomotion : OK")
except ImportError as e:
    failures.append(f"nav_locomotion: {e}")

# Isaac Lab (only importable after SimulationApp — just check the package exists)
try:
    import importlib.util
    if importlib.util.find_spec("isaaclab") is not None:
        print(f"  isaaclab    : found")
    else:
        failures.append("isaaclab: not found — is Isaac Lab installed in this env?")
except Exception as e:
    failures.append(f"isaaclab check failed: {e}")

if failures:
    print("\n  FAILED IMPORTS:")
    for f in failures:
        print(f"    ✗ {f}")
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
