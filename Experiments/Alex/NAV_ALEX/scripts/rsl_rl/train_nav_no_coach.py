"""Phase C navigation training WITHOUT AI coach (baseline).

Convenience wrapper — just calls train_nav.py with --no_coach flag.
Use this for Phase C-0 warm-up (flat terrain, no coach, 2000-5000 iters)
to verify CNN trains and robot moves forward before enabling the coach.

Usage:
    python scripts/rsl_rl/train_nav_no_coach.py \
        --headless --no_wandb \
        --loco_checkpoint checkpoints/ai_coached_v8_10600.pt \
        --num_envs 512 --max_iterations 5000
"""

import sys
import os

# Inject --no_coach flag
sys.argv.append("--no_coach")

# Import and run the main training script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from train_nav import main

if __name__ == "__main__":
    main()
