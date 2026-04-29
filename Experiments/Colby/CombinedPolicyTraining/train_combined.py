"""
train_combined.py
=================
Combined nav + loco training script for Colby's capstone work.

Imports Alex's nav_locomotion modules (nav policy, loco wrapper, env wrapper)
and Ryan's locomotion checkpoint — but owns its own configuration,
log directory, and checkpoint output. Zero modifications to any teammate's files.

Architecture:
    Depth Camera (64x64) -> CNN Encoder (128-dim)
      + Proprioception (12-dim)
            |
    [Nav Policy MLP, 10 Hz]      <- BEING TRAINED
            |
    Velocity Command [vx, vy, wz]
            |
    [Frozen Loco Policy, 50 Hz]  <- Ryan's mason_hybrid_best_33200.pt (frozen)
            |
    12 Joint Targets -> Spot

Checkpoints saved to:
    Experiments/Colby/CombinedPolicyTraining/logs/spot_nav_explore_ppo/<timestamp>/

Usage:
    # Local smoke test
    python train_combined.py --headless --num_envs 16 --max_iterations 100 \
        --loco_checkpoint <path/to/mason_hybrid_best_33200.pt>

    # H100 full run
    python train_combined.py --headless --num_envs 2048 --max_iterations 30000 \
        --loco_checkpoint <path/to/mason_hybrid_best_33200.pt>

IMPORTANT: CLI args must be parsed before any Isaac Lab imports (AppLauncher rule).
Exit with os._exit(0) — never call simulation_app.close() (CUDA deadlock).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — everything relative to this file, never hardcoded
# ---------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent           # CombinedPolicyTraining/
REPO_ROOT = THIS_DIR.parents[2]                      # repo root
LOG_DIR = THIS_DIR / "logs"                          # our checkpoint output

# ---------------------------------------------------------------------------
# CLI args — MUST be parsed before any Isaac Lab / omni imports
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combined nav+loco training — Colby's capstone"
    )

    # Required
    parser.add_argument(
        "--loco_checkpoint", type=str, required=True,
        help="Path to frozen RSL-RL locomotion checkpoint (.pt)"
    )

    # Environment
    parser.add_argument("--num_envs",  type=int, default=2048)
    parser.add_argument("--depth_res", type=int, default=64)

    # Training
    parser.add_argument("--max_iterations", type=int, default=30000)
    parser.add_argument("--save_interval",  type=int, default=100)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--resume",         type=str, default=None,
                        help="Path to a nav checkpoint to resume from")

    # Isaac Lab
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no_wandb", action="store_true", default=True)

    return parser.parse_args()


args = parse_args()

# ---------------------------------------------------------------------------
# Pre-load torch before Isaac Sim starts.
# Isaac Sim's extension loader imports torch internally during AppLauncher
# init. If its older CUDA 11 extscache DLLs load first, torch's c10.dll
# crashes (WinError 1114). Importing torch here first puts it in sys.modules
# in a clean state so Isaac Sim's extensions reuse the already-loaded DLLs.
# ---------------------------------------------------------------------------
import torch  # noqa: E402 — must be before AppLauncher

# ---------------------------------------------------------------------------
# Isaac Lab boot — must happen before any omni/isaaclab imports
# ---------------------------------------------------------------------------

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Remaining imports (safe after AppLauncher)
# ---------------------------------------------------------------------------

import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

# Alex's modules — imported as a library, nothing in his directory is modified
import nav_locomotion  # noqa: F401 — triggers Isaac Lab gym env registration
from nav_locomotion.modules.depth_cnn import ActorCriticCNN, TOTAL_OBS_DIMS
from nav_locomotion.modules.loco_wrapper import FrozenLocoPolicy
from nav_locomotion.modules.nav_env_wrapper import NavEnvWrapper
from nav_locomotion.tasks.navigation.config.spot.nav_env_cfg import SpotNavExploreCfg

# cnn_compat: adapter classes that bridge ActorCriticCNN (old rsl_rl < 4.0 API) to
# rsl_rl 5.0.1's separate actor/critic MLPModel interface.
# CombinedPolicyTraining/ is on sys.path when running as "python train_combined.py",
# so "cnn_compat:ActorCNNWrapper" resolves correctly via resolve_callable().
import cnn_compat  # noqa: F401 — must be imported so resolve_callable can find it
import rsl_rl.models as _rsl_models
_rsl_models.ActorCNNWrapper = cnn_compat.ActorCNNWrapper
_rsl_models.CriticCNNWrapper = cnn_compat.CriticCNNWrapper


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  COMBINED NAV + LOCO TRAINING")
    print("=" * 60)
    print(f"  Loco checkpoint : {args.loco_checkpoint}")
    print(f"  Num envs        : {args.num_envs}")
    print(f"  Max iterations  : {args.max_iterations}")
    print(f"  Save interval   : {args.save_interval}")
    print(f"  Checkpoint dir  : {LOG_DIR}")
    print(f"  Device          : {device}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    env_cfg = SpotNavExploreCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = gym.make("Navigation-Explore-Spot-v0", cfg=env_cfg)
    device = str(env.unwrapped.device) if hasattr(env, "unwrapped") else device
    print(f"[TRAIN] Environment ready — device={device}")

    # ------------------------------------------------------------------
    # Frozen locomotion policy (Ryan's checkpoint, never updated)
    # ------------------------------------------------------------------
    loco_policy = FrozenLocoPolicy.from_checkpoint(
        args.loco_checkpoint, device=device
    )

    # ------------------------------------------------------------------
    # Nav environment wrapper (vel_cmd -> loco -> joints)
    # ------------------------------------------------------------------
    nav_env = NavEnvWrapper(env, loco_policy)

    # ------------------------------------------------------------------
    # RSL-RL runner  (rsl_rl 5.0.1 compatible config dict)
    # ------------------------------------------------------------------
    # rsl_rl 5.0.1 broke the old "policy" config format — it now expects separate
    # "actor" and "critic" dicts with class_name pointing to MLPModel subclasses.
    # We bypass class_to_dict(SpotNavPPORunnerCfg) and build the dict manually,
    # pointing class_name at our ActorCNNWrapper / CriticCNNWrapper adapters.
    # obs_groups tells rsl_rl which observation group to feed to each model.
    # NavEnvWrapper returns {"policy": tensor(N, 4108)} so both use ["policy"].

    # NavEnvWrapper has no .cfg attribute; Logger stores it but only uses it for
    # wandb/neptune (we use tensorboard), so an empty dict is safe.
    nav_env.cfg = {}

    runner_cfg_dict = {
        # --- Runner bookkeeping ---
        "seed": 42,
        "num_steps_per_env": 128,
        "max_iterations": args.max_iterations,
        "save_interval": args.save_interval,
        "logger": "tensorboard",
        "experiment_name": "spot_nav_explore_ppo",
        "run_name": "",
        "device": device,
        "check_for_nan": False,   # disable NaN check (slow + we handle NaN in cnn)
        "resume": False,
        "load_run": ".*",
        "load_checkpoint": "model_.*.pt",
        "clip_actions": None,
        "multi_gpu": None,        # overwritten by OnPolicyRunner._configure_multi_gpu

        # --- Observation routing ---
        # NavEnvWrapper.get_observations() returns {"policy": tensor(N, 4108)}
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},

        # --- Actor model (CNN adapter) ---
        "actor": {
            "class_name": "rsl_rl.models:ActorCNNWrapper",
            # No other fields — ActorCNNWrapper absorbs extras via **kwargs
        },

        # --- Critic model (CNN adapter) ---
        "critic": {
            "class_name": "rsl_rl.models:CriticCNNWrapper",
        },

        # --- PPO algorithm ---
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 8,
            "num_mini_batches": 8,
            "learning_rate": args.lr,
            "schedule": "fixed",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "optimizer": "adam",
            "normalize_advantage_per_mini_batch": False,
            "share_cnn_encoders": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }

    runner = OnPolicyRunner(
        nav_env,
        runner_cfg_dict,
        log_dir=str(LOG_DIR),
        device=device,
    )

    # runner.alg.actor is our ActorCNNWrapper; ._net is the actual ActorCriticCNN
    nav_policy = runner.alg.actor._net
    print(f"[TRAIN] ActorCriticCNN: "
          f"{sum(p.numel() for p in nav_policy.parameters()):,} params")

    # ------------------------------------------------------------------
    # Resume from checkpoint if specified
    # ------------------------------------------------------------------
    if args.resume:
        print(f"[TRAIN] Resuming from {args.resume}")
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        nav_policy.load_state_dict(ckpt.get("model_state_dict", ckpt))

    # ------------------------------------------------------------------
    # Training loop — patch update() to inject terrain level logging
    # ------------------------------------------------------------------
    original_update = runner.alg.update

    def patched_update(*a, **kw):
        result = original_update(*a, **kw)

        iteration = getattr(runner, "current_learning_iteration",
                            getattr(runner, "it", 0))

        # Log terrain curriculum level every 10 iterations
        # rsl_rl 5.0.1 uses runner.logger.writer (not runner.writer directly)
        writer = getattr(getattr(runner, "logger", None), "writer", None)
        if writer is not None and iteration % 10 == 0:
            try:
                terrain = nav_env._unwrapped.scene.terrain
                if hasattr(terrain, "terrain_levels"):
                    mean_level = terrain.terrain_levels.float().mean().item()
                    writer.add_scalar(
                        "Curriculum/terrain_level", mean_level, iteration
                    )
            except Exception:
                pass

        return result

    runner.alg.update = patched_update

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    nav_env.reset()
    print(f"[TRAIN] Starting — {args.max_iterations} iterations")

    try:
        runner.learn(
            num_learning_iterations=args.max_iterations,
            init_at_random_ep_len=True,
        )
    except KeyboardInterrupt:
        print("\n[TRAIN] Interrupted by user — saving final checkpoint...")
    except Exception as e:
        import traceback
        print(f"[TRAIN] Error: {e}")
        traceback.print_exc()

    # ------------------------------------------------------------------
    # Save final checkpoint to our directory
    # ------------------------------------------------------------------
    final_path = LOG_DIR / "spot_nav_explore_ppo" / "model_final.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": nav_policy.state_dict()}, final_path)
    print(f"[TRAIN] Final checkpoint saved: {final_path}")

    # Never call simulation_app.close() — causes CUDA deadlock (see Memory.md)
    os._exit(0)


if __name__ == "__main__":
    main()
