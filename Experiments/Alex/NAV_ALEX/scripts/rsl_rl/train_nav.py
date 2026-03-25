"""Phase C navigation training with AI coach integration.

Trains a depth-camera navigation policy on top of a frozen Phase B loco policy.
The nav policy outputs velocity commands [vx, vy, wz] at 10 Hz, which the frozen
loco policy converts to 12-dim joint actions at 50 Hz.

Usage (H100):
    python scripts/rsl_rl/train_nav.py \
        --headless --no_wandb \
        --loco_checkpoint checkpoints/ai_coached_v8_10600.pt \
        --num_envs 2048 --max_iterations 30000 \
        --coach_interval 250

Usage (local smoke):
    python scripts/rsl_rl/train_nav.py \
        --headless --no_wandb --no_coach \
        --loco_checkpoint checkpoints/ai_coached_v8_10600.pt \
        --num_envs 16 --max_iterations 100

Architecture:
    RSL-RL OnPolicyRunner
      -> ActorCriticCNN (depth CNN + MLP, 10 Hz)
        -> NavEnvWrapper
          -> FrozenLocoPolicy (50 Hz)
            -> Isaac Lab env (500 Hz physics)

IMPORTANT: Parse CLI args BEFORE importing Isaac Lab (AppLauncher requirement).
Exit with os._exit(0) to avoid CUDA deadlock (never call simulation_app.close()).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# ---------------------------------------------------------------------------
# CLI args — MUST be parsed before any Isaac Lab imports
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Phase C: Nav policy training with AI coach")

    # Required
    parser.add_argument("--loco_checkpoint", type=str, required=True,
                        help="Path to frozen Phase B loco checkpoint (.pt)")

    # Environment
    parser.add_argument("--num_envs", type=int, default=2048,
                        help="Number of parallel environments (default: 2048)")
    parser.add_argument("--depth_res", type=int, default=64,
                        help="Depth image resolution (default: 64)")

    # Training
    parser.add_argument("--max_iterations", type=int, default=30000,
                        help="Maximum training iterations (default: 30000)")
    parser.add_argument("--save_interval", type=int, default=100,
                        help="Checkpoint save interval (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to nav checkpoint to resume from")

    # AI Coach
    parser.add_argument("--coach_interval", type=int, default=250,
                        help="Coach consultation interval in iterations (default: 250)")
    parser.add_argument("--no_coach", action="store_true",
                        help="Disable AI coach (baseline training)")

    # Isaac Lab
    parser.add_argument("--headless", action="store_true",
                        help="Run headless (required on H100)")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")

    args = parser.parse_args()
    return args


args = parse_args()

# ---------------------------------------------------------------------------
# Isaac Lab imports (after CLI parsing)
# ---------------------------------------------------------------------------

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

# Local imports
import nav_locomotion  # noqa: F401 — triggers gym registration
from nav_locomotion.modules.depth_cnn import ActorCriticCNN, TOTAL_OBS_DIMS
from nav_locomotion.modules.loco_wrapper import FrozenLocoPolicy
from nav_locomotion.modules.nav_env_wrapper import NavEnvWrapper
from nav_locomotion.ai_coach.guardrails import NAV_WEIGHT_BOUNDS


def main():
    # ------------------------------------------------------------------
    # Create Isaac Lab environment
    # ------------------------------------------------------------------
    from nav_locomotion.tasks.navigation.config.spot.nav_env_cfg import SpotNavExploreCfg
    from nav_locomotion.tasks.navigation.config.spot.agents.rsl_rl_ppo_cfg import SpotNavPPORunnerCfg

    env_id = "Navigation-Explore-Spot-v0"
    env_cfg = SpotNavExploreCfg()
    env_cfg.scene.num_envs = args.num_envs

    env = gym.make(env_id, cfg=env_cfg)
    # Unwrap gym wrappers to get Isaac Lab env's device
    device = env.unwrapped.device if hasattr(env, "unwrapped") else "cuda:0"

    print(f"[TRAIN_NAV] Environment: {env_id}, {args.num_envs} envs, device={device}")

    # ------------------------------------------------------------------
    # Load frozen loco policy
    # ------------------------------------------------------------------
    loco_policy = FrozenLocoPolicy.from_checkpoint(
        args.loco_checkpoint, device=str(device)
    )

    # ------------------------------------------------------------------
    # Wrap env with nav wrapper (vel_cmd -> loco -> joints)
    # ------------------------------------------------------------------
    nav_env = NavEnvWrapper(env, loco_policy)

    # ------------------------------------------------------------------
    # Create RSL-RL runner with standard PPO config
    # ------------------------------------------------------------------
    from nav_locomotion.tasks.navigation.config.spot.agents.rsl_rl_ppo_cfg import (
        SpotNavPPORunnerCfg,
    )

    runner_cfg = SpotNavPPORunnerCfg()
    runner_cfg.max_iterations = args.max_iterations
    runner_cfg.save_interval = args.save_interval
    if args.no_wandb:
        runner_cfg.logger = "tensorboard"

    # Experiment directory
    log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Convert config class to dict for RSL-RL's OnPolicyRunner (expects dict)
    from isaaclab.utils import class_to_dict
    runner_cfg_dict = class_to_dict(runner_cfg)

    # Register ActorCriticCNN in rsl_rl's namespace so eval("ActorCriticCNN") works
    import rsl_rl.runners.on_policy_runner as _runner_module
    _runner_module.ActorCriticCNN = ActorCriticCNN

    runner = OnPolicyRunner(nav_env, runner_cfg_dict, log_dir=log_dir, device=str(device))

    # The runner already created ActorCriticCNN via the config class_name.
    # Access it for logging and optional checkpoint resume.
    cnn_policy = runner.alg.policy
    print(f"[TRAIN_NAV] ActorCriticCNN: {sum(p.numel() for p in cnn_policy.parameters()):,} params")

    # Load nav checkpoint if resuming
    if args.resume:
        print(f"[TRAIN_NAV] Resuming from {args.resume}")
        ckpt = torch.load(args.resume, weights_only=False, map_location=str(device))
        cnn_policy.load_state_dict(ckpt.get("model_state_dict", ckpt))

    # ------------------------------------------------------------------
    # AI Coach setup (optional)
    # ------------------------------------------------------------------
    coach = None
    guardrails = None
    actuator = None
    metrics_collector = None
    decision_log = None

    if not args.no_coach:
        try:
            from nav_locomotion.ai_coach.coach import Coach
            from nav_locomotion.ai_coach.guardrails import Guardrails
            from nav_locomotion.ai_coach.actuator import Actuator
            from nav_locomotion.ai_coach.metrics import MetricsCollector
            from nav_locomotion.ai_coach.decision_log import DecisionLog

            coach = Coach(weight_bounds=NAV_WEIGHT_BOUNDS)
            guardrails = Guardrails(weight_bounds=NAV_WEIGHT_BOUNDS)
            actuator = Actuator(nav_env, runner)
            metrics_collector = MetricsCollector(nav_env, runner)
            decision_log = DecisionLog(
                os.path.join(log_dir, runner_cfg.experiment_name, "coach_decisions.jsonl")
            )
            print("[TRAIN_NAV] AI Coach enabled — consulting every "
                  f"{args.coach_interval} iterations")
        except Exception as e:
            print(f"[TRAIN_NAV] AI Coach init failed: {e} — running without coach")
            coach = None

    # ------------------------------------------------------------------
    # Training loop — monkey-patch runner for coach integration
    # ------------------------------------------------------------------
    start_time = time.time()
    reward_info = {}
    recent_decisions = []
    halt_training = False

    # Intercept runner.log() to capture RSL-RL internal metrics
    # (mean_reward, value_loss, noise_std are local vars in learn(),
    # only accessible via log interception — see Bug #33)
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

    # Monkey-patch alg.update() to inject coach checks after each iteration
    original_update = runner.alg.update

    def patched_update(*update_args, **update_kwargs):
        nonlocal halt_training
        result = original_update(*update_args, **update_kwargs)

        # Get current iteration from runner
        iteration = getattr(runner, "current_learning_iteration",
                           getattr(runner, "it", 0))

        # Log terrain level every iteration (if terrain curriculum exists)
        if runner.writer is not None and iteration % 10 == 0:
            try:
                terrain = nav_env._unwrapped.scene.terrain
                if hasattr(terrain, "terrain_levels"):
                    mean_level = terrain.terrain_levels.float().mean().item()
                    runner.writer.add_scalar("Curriculum/terrain_level", mean_level, iteration)
            except Exception:
                pass

        # AI Coach consultation
        if (coach and metrics_collector and iteration > 0
                and iteration % args.coach_interval == 0):

            elapsed_hours = (time.time() - start_time) / 3600
            lr = runner.alg.optimizer.param_groups[0]["lr"]

            # Collect metrics
            snapshot = metrics_collector.collect(
                iteration=iteration,
                elapsed_hours=elapsed_hours,
                reward_info=reward_info,
                lr=lr,
            )

            # Emergency check first (priority over coach)
            emergency = guardrails.check_emergency(snapshot)
            if emergency:
                if emergency == "halve_lr":
                    new_lr = actuator.emergency_halve_lr()
                    decision_log.log_emergency(
                        iteration, emergency,
                        f"Value loss {snapshot.value_loss:.1f} > 100 — LR halved to {new_lr:.2e}"
                    )
                    print(f"[AI-COACH] EMERGENCY: {emergency}")
                elif emergency == "nan_halt":
                    decision_log.log_emergency(
                        iteration, emergency, "NaN detected in policy parameters"
                    )
                    print("[AI-COACH] EMERGENCY: NaN detected — halting training")
                    halt_training = True
                return result

            # Coach consultation
            recent_history = metrics_collector.get_recent(5)
            plateau = metrics_collector.is_plateau()

            decision, latency = coach.get_decision(
                snapshot=snapshot,
                recent_history=recent_history,
                recent_decisions=recent_decisions,
                plateau_detected=plateau,
            )
            recent_decisions.append(decision)
            if len(recent_decisions) > 5:
                recent_decisions.pop(0)

            print(
                f"[AI-COACH] iter={iteration} action={decision.action} "
                f"conf={decision.confidence:.2f} latency={latency:.0f}ms"
            )

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

            # Log decision
            decision_log.log_decision(
                iteration=iteration,
                metrics={
                    "reward": snapshot.mean_reward,
                    "terrain": snapshot.mean_terrain_level,
                    "survival": snapshot.survival_rate,
                    "body_height": snapshot.mean_body_height,
                    "value_loss": snapshot.value_loss,
                },
                decision=decision,
                guardrail_msgs=guardrail_msgs,
                applied_changes=applied,
                api_latency_ms=latency,
            )

            for msg in guardrail_msgs:
                print(f"  {msg}")

            # --- Write AI Coach + terrain metrics to TensorBoard ---
            if runner.writer is not None:
                writer = runner.writer
                it = runner.current_learning_iteration

                # Terrain curriculum level
                writer.add_scalar("Curriculum/terrain_level", snapshot.mean_terrain_level, it)

                # Nav-specific metrics
                writer.add_scalar("Nav/forward_distance", snapshot.mean_forward_distance, it)
                writer.add_scalar("Nav/body_height", snapshot.mean_body_height, it)
                writer.add_scalar("Nav/survival_rate", snapshot.survival_rate, it)
                writer.add_scalar("Nav/flip_rate", snapshot.flip_rate, it)

                # AI Coach decision
                action_map = {"no_change": 0, "adjust_weights": 1, "adjust_lr": 2, "emergency_stop": 3}
                writer.add_scalar("AI_Coach/action", action_map.get(decision.action, -1), it)
                writer.add_scalar("AI_Coach/confidence", decision.confidence, it)
                writer.add_scalar("AI_Coach/api_latency_ms", latency, it)

                # Current reward weights
                if snapshot.current_weights:
                    for term, weight in snapshot.current_weights.items():
                        writer.add_scalar(f"Reward_Weights/{term}", weight, it)

                # Applied weight changes
                for term, (old_val, new_val) in applied.items():
                    if term != "lr":
                        writer.add_scalar(f"Weight_Changes/{term}", new_val, it)

        return result

    runner.alg.update = patched_update

    # Reset the gym env before training (gym requires reset before step)
    nav_env.reset()

    print(f"[TRAIN_NAV] Starting training: {args.max_iterations} iters, "
          f"coach={'ON' if coach else 'OFF'}")

    try:
        runner.learn(
            num_learning_iterations=args.max_iterations,
            init_at_random_ep_len=True,
        )
    except KeyboardInterrupt:
        print("\n[TRAIN_NAV] Training interrupted by user")
    except Exception as e:
        if halt_training:
            print("[TRAIN_NAV] Training halted by emergency NaN detection")
        else:
            print(f"\n[TRAIN_NAV] Training error: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Save final checkpoint
    # ------------------------------------------------------------------
    final_path = os.path.join(log_dir, runner_cfg.experiment_name, "model_final.pt")
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save({"model_state_dict": cnn_policy.state_dict()}, final_path)
    print(f"[TRAIN_NAV] Final checkpoint saved: {final_path}")

    # Exit cleanly — never call simulation_app.close() (CUDA deadlock)
    os._exit(0)


if __name__ == "__main__":
    main()
