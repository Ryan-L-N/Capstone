"""
train_combined.py
=================
Combined nav + loco training script for Colby's capstone work.

Imports Alex's nav_locomotion modules (nav policy, loco wrapper, env wrapper,
AI coach) and Ryan's locomotion checkpoint — but owns its own configuration,
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
    python train_combined.py --headless --no_coach --num_envs 16 --max_iterations 100 \
        --loco_checkpoint <path/to/mason_hybrid_best_33200.pt>

    # H100 full run
    python train_combined.py --headless --num_envs 2048 --max_iterations 30000 \
        --loco_checkpoint <path/to/mason_hybrid_best_33200.pt> --coach_interval 250

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

    # AI Coach
    parser.add_argument("--coach_interval", type=int, default=250)
    parser.add_argument("--no_coach", action="store_true",
                        help="Disable AI coach (use for smoke tests)")

    # Isaac Lab
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no_wandb", action="store_true", default=True)

    return parser.parse_args()


args = parse_args()

# ---------------------------------------------------------------------------
# Isaac Lab boot — must happen before any omni/isaaclab imports
# ---------------------------------------------------------------------------

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Remaining imports (safe after AppLauncher)
# ---------------------------------------------------------------------------

import torch
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

# Alex's modules — imported as a library, nothing in his directory is modified
import nav_locomotion  # noqa: F401 — triggers Isaac Lab gym env registration
from nav_locomotion.modules.depth_cnn import ActorCriticCNN, TOTAL_OBS_DIMS
from nav_locomotion.modules.loco_wrapper import FrozenLocoPolicy
from nav_locomotion.modules.nav_env_wrapper import NavEnvWrapper
from nav_locomotion.tasks.navigation.config.spot.nav_env_cfg import SpotNavExploreCfg
from nav_locomotion.tasks.navigation.config.spot.agents.rsl_rl_ppo_cfg import SpotNavPPORunnerCfg
from nav_locomotion.ai_coach.guardrails import NAV_WEIGHT_BOUNDS
from isaaclab.utils import class_to_dict


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
    print(f"  AI Coach        : {'OFF' if args.no_coach else 'ON'}")
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
    # RSL-RL runner
    # ------------------------------------------------------------------
    runner_cfg = SpotNavPPORunnerCfg()
    runner_cfg.max_iterations = args.max_iterations
    runner_cfg.save_interval  = args.save_interval
    runner_cfg.logger = "tensorboard"

    # Inject our custom policy class into RSL-RL's namespace so it can
    # find it when instantiating via class_name="ActorCriticCNN"
    import rsl_rl.runners.on_policy_runner as _runner_module
    _runner_module.ActorCriticCNN = ActorCriticCNN

    runner = OnPolicyRunner(
        nav_env,
        class_to_dict(runner_cfg),
        log_dir=str(LOG_DIR),
        device=device,
    )

    nav_policy = runner.alg.policy
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
    # AI Coach (optional — skip with --no_coach for smoke tests)
    # ------------------------------------------------------------------
    coach = guardrails = actuator = metrics_collector = decision_log = None

    if not args.no_coach:
        try:
            from nav_locomotion.ai_coach.coach import Coach
            from nav_locomotion.ai_coach.guardrails import Guardrails
            from nav_locomotion.ai_coach.actuator import Actuator
            from nav_locomotion.ai_coach.metrics import MetricsCollector
            from nav_locomotion.ai_coach.decision_log import DecisionLog

            coach            = Coach(weight_bounds=NAV_WEIGHT_BOUNDS)
            guardrails       = Guardrails(weight_bounds=NAV_WEIGHT_BOUNDS)
            actuator         = Actuator(nav_env, runner)
            metrics_collector = MetricsCollector(nav_env, runner)
            decision_log     = DecisionLog(
                str(LOG_DIR / runner_cfg.experiment_name / "coach_decisions.jsonl")
            )
            print(f"[TRAIN] AI Coach enabled — every {args.coach_interval} iters")
        except Exception as e:
            print(f"[TRAIN] AI Coach init failed: {e} — continuing without coach")
            coach = None

    # ------------------------------------------------------------------
    # Training loop — intercept runner internals for coach + metrics
    # ------------------------------------------------------------------
    start_time   = time.time()
    reward_info  = {}
    recent_decisions = []
    halt_training    = False

    # Capture RSL-RL's internal log dict for the metrics collector
    original_log = getattr(runner, "_log_process", None) or getattr(runner, "log", None)

    def intercepted_log(info: dict, *a, **kw):
        nonlocal reward_info
        reward_info.update(info)
        if metrics_collector:
            metrics_collector.update_extras(info)
        if original_log:
            return original_log(info, *a, **kw)

    if hasattr(runner, "_log_process"):
        runner._log_process = intercepted_log
    elif hasattr(runner, "log"):
        runner.log = intercepted_log

    # Patch update() to inject terrain level logging + coach consultations
    original_update = runner.alg.update

    def patched_update(*a, **kw):
        nonlocal halt_training
        result = original_update(*a, **kw)

        iteration = getattr(runner, "current_learning_iteration",
                            getattr(runner, "it", 0))

        # Log terrain curriculum level every 10 iterations
        if runner.writer is not None and iteration % 10 == 0:
            try:
                terrain = nav_env._unwrapped.scene.terrain
                if hasattr(terrain, "terrain_levels"):
                    mean_level = terrain.terrain_levels.float().mean().item()
                    runner.writer.add_scalar(
                        "Curriculum/terrain_level", mean_level, iteration
                    )
            except Exception:
                pass

        # AI Coach consultation
        if (coach and metrics_collector
                and iteration > 0
                and iteration % args.coach_interval == 0):

            elapsed_hours = (time.time() - start_time) / 3600
            lr = runner.alg.optimizer.param_groups[0]["lr"]

            snapshot = metrics_collector.collect(
                iteration=iteration,
                elapsed_hours=elapsed_hours,
                reward_info=reward_info,
                lr=lr,
            )

            emergency = guardrails.check_emergency(snapshot)
            if emergency:
                if emergency == "halve_lr":
                    new_lr = actuator.emergency_halve_lr()
                    decision_log.log_emergency(
                        iteration, emergency,
                        f"Value loss {snapshot.value_loss:.1f} > 100 — LR → {new_lr:.2e}"
                    )
                elif emergency == "nan_halt":
                    decision_log.log_emergency(
                        iteration, emergency, "NaN in policy params"
                    )
                    print("[COACH] EMERGENCY: NaN — halting")
                    halt_training = True
                return result

            decision, latency = coach.get_decision(
                snapshot=snapshot,
                recent_history=metrics_collector.get_recent(5),
                recent_decisions=recent_decisions,
                plateau_detected=metrics_collector.is_plateau(),
            )
            recent_decisions.append(decision)
            if len(recent_decisions) > 5:
                recent_decisions.pop(0)

            print(f"[COACH] iter={iteration} action={decision.action} "
                  f"conf={decision.confidence:.2f} latency={latency:.0f}ms")

            applied = {}
            guardrail_msgs = []

            if decision.action == "adjust_weights" and decision.weight_changes:
                approved, msgs = guardrails.validate_weight_changes(
                    decision.weight_changes,
                    snapshot.current_weights,
                    snapshot.mean_terrain_level,
                )
                guardrail_msgs.extend(msgs)
                if approved:
                    applied = actuator.apply_weight_changes(approved)

            if decision.action == "adjust_lr" and decision.lr_change:
                validated_lr, msgs = guardrails.validate_lr_change(
                    decision.lr_change, lr
                )
                guardrail_msgs.extend(msgs)
                if validated_lr:
                    old_lr = actuator.apply_lr_change(validated_lr)
                    applied["lr"] = (old_lr, validated_lr)

            decision_log.log_decision(
                iteration=iteration,
                metrics={
                    "reward":      snapshot.mean_reward,
                    "terrain":     snapshot.mean_terrain_level,
                    "survival":    snapshot.survival_rate,
                    "body_height": snapshot.mean_body_height,
                    "value_loss":  snapshot.value_loss,
                },
                decision=decision,
                guardrail_msgs=guardrail_msgs,
                applied_changes=applied,
                api_latency_ms=latency,
            )
            for msg in guardrail_msgs:
                print(f"  {msg}")

            # Write nav + coach metrics to TensorBoard
            if runner.writer is not None:
                w, it = runner.writer, iteration
                w.add_scalar("Nav/forward_distance", snapshot.mean_forward_distance, it)
                w.add_scalar("Nav/body_height",      snapshot.mean_body_height,      it)
                w.add_scalar("Nav/survival_rate",    snapshot.survival_rate,          it)
                w.add_scalar("Nav/flip_rate",        snapshot.flip_rate,              it)
                action_map = {
                    "no_change": 0, "adjust_weights": 1,
                    "adjust_lr": 2, "emergency_stop": 3
                }
                w.add_scalar("AI_Coach/action",         action_map.get(decision.action, -1), it)
                w.add_scalar("AI_Coach/confidence",     decision.confidence,  it)
                w.add_scalar("AI_Coach/api_latency_ms", latency,              it)
                for term, weight in (snapshot.current_weights or {}).items():
                    w.add_scalar(f"Reward_Weights/{term}", weight, it)

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
        if halt_training:
            print("[TRAIN] Halted by AI Coach emergency (NaN)")
        else:
            import traceback
            print(f"[TRAIN] Error: {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Save final checkpoint to our directory
    # ------------------------------------------------------------------
    final_path = LOG_DIR / runner_cfg.experiment_name / "model_final.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": nav_policy.state_dict()}, final_path)
    print(f"[TRAIN] Final checkpoint saved: {final_path}")

    # Never call simulation_app.close() — causes CUDA deadlock (see Memory.md)
    os._exit(0)


if __name__ == "__main__":
    main()
