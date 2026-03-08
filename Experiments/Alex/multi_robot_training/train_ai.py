"""AI-Guided PPO Training for Spot.

Wraps the standard training loop with an LLM coach that monitors metrics
and adjusts reward weights at runtime. Trains through all phases
(flat → transition → robust_easy → robust) automatically.

Architecture:
    Training Loop (every iter) → Metrics Collector (every N iters) →
    Emergency Check → AI Coach (Claude API) → Guardrails → Actuator → Live Env

Usage (H100):
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/multi_robot_training/train_ai.py --headless \\
        --robot spot --start_phase flat --num_envs 5000 \\
        --max_noise_std 0.5 --min_noise_std 0.3 \\
        --coach_interval 100 --no_wandb

    # Resume mid-phase with AI coach:
    ./isaaclab.sh -p ~/multi_robot_training/train_ai.py --headless \\
        --robot spot --start_phase robust --num_envs 5000 \\
        --load_run 2026-03-08_12-40-08 --load_checkpoint model_1000.pt \\
        --coach_interval 100 --no_wandb

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

# ── 0. Parse args BEFORE any Isaac imports ──────────────────────────────
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AI-guided PPO training for Spot")
parser.add_argument("--robot", type=str, default="spot",
                    choices=["spot", "vision60"])
parser.add_argument("--start_phase", type=str, default="flat",
                    choices=["flat", "transition", "robust_easy", "robust"],
                    help="Which phase to start from")
parser.add_argument("--end_phase", type=str, default="robust",
                    choices=["flat", "transition", "robust_easy", "robust"],
                    help="Which phase to end at")
parser.add_argument("--num_envs", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--lr_max", type=float, default=None,
                    help="Override phase-specific lr_max")
parser.add_argument("--lr_min", type=float, default=1e-5)
parser.add_argument("--warmup_iters", type=int, default=50)
parser.add_argument("--max_noise_std", type=float, default=0.5)
parser.add_argument("--min_noise_std", type=float, default=0.3)
parser.add_argument("--num_learning_epochs", type=int, default=4)
parser.add_argument("--no_wandb", action="store_true", default=False)

# AI Coach args
parser.add_argument("--coach_interval", type=int, default=100,
                    help="Consult AI coach every N iterations")
parser.add_argument("--coach_model", type=str, default="claude-sonnet-4-20250514",
                    help="Claude model for coach decisions")
parser.add_argument("--no_coach", action="store_true", default=False,
                    help="Disable AI coach (run plain training)")
parser.add_argument("--anthropic_api_key", type=str, default=None,
                    help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")

# Resume args
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--load_checkpoint", type=str, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 1. Imports (AFTER SimulationApp) ────────────────────────────────────
import json
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from shared.lr_schedule import cosine_annealing_lr, set_learning_rate
from shared.training_utils import configure_tf32, register_std_safety_clamp

from ai_trainer.config import PHASE_CONFIGS, PHASE_ORDER, CoachConfig
from ai_trainer.metrics import MetricsCollector
from ai_trainer.coach import Coach
from ai_trainer.guardrails import Guardrails
from ai_trainer.actuator import Actuator
from ai_trainer.decision_log import DecisionLog

configure_tf32()


# ── 2. Config Loading ───────────────────────────────────────────────────

def load_robot_configs(robot: str):
    """Load env and PPO configs for the specified robot."""
    if robot == "spot":
        from configs.spot_ppo_env_cfg import SpotPPOEnvCfg
        from configs.spot_ppo_cfg import SpotPPORunnerCfg
        return SpotPPOEnvCfg(), SpotPPORunnerCfg(), "Isaac-Velocity-Spot-PPO-v0"
    elif robot == "vision60":
        from configs.vision60_ppo_env_cfg import Vision60PPOEnvCfg
        from configs.vision60_ppo_cfg import Vision60PPORunnerCfg
        return Vision60PPOEnvCfg(), Vision60PPORunnerCfg(), "Isaac-Velocity-Vision60-PPO-v0"
    else:
        raise ValueError(f"Unknown robot: {robot}")


def apply_phase_terrain(env_cfg, phase_name: str):
    """Apply terrain configuration for the given phase."""
    terrain = PHASE_CONFIGS[phase_name].terrain
    if terrain == "flat":
        if hasattr(env_cfg.scene, "terrain"):
            env_cfg.scene.terrain.terrain_type = "plane"
    elif terrain in ("transition", "robust_easy", "robust"):
        if hasattr(env_cfg, "terrain_cfg_name"):
            env_cfg.terrain_cfg_name = terrain


# ── 3. Single Phase Training ───────────────────────────────────────────

def train_phase(
    phase_name: str,
    env_cfg,
    agent_cfg,
    env_id: str,
    coach: Coach | None,
    coach_cfg: CoachConfig,
    resume_run: str | None = None,
    resume_checkpoint: str | None = None,
) -> tuple[str | None, bool]:
    """Train a single phase with optional AI coaching.

    Returns:
        (checkpoint_path, should_advance) — path to best checkpoint and
        whether go/no-go criteria were met for phase advancement.
    """
    phase_cfg = PHASE_CONFIGS[phase_name]

    # Apply CLI overrides
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    agent_cfg.save_interval = args_cli.save_interval
    agent_cfg.max_iterations = phase_cfg.max_iterations

    if args_cli.no_wandb:
        agent_cfg.logger = "tensorboard"

    lr_max = args_cli.lr_max if args_cli.lr_max else phase_cfg.lr_max
    lr_min = args_cli.lr_min

    # Noise bounds — mutable dict shared with safety clamp and actuator
    noise_bounds = {
        "min": max(args_cli.min_noise_std, phase_cfg.min_noise_std),
        "max": min(args_cli.max_noise_std, phase_cfg.max_noise_std),
    }

    # Logging
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)

    steps_per_iter = env_cfg.scene.num_envs * agent_cfg.num_steps_per_env

    print(f"\n{'='*70}", flush=True)
    print(f"  AI-GUIDED TRAINING — PHASE: {phase_name.upper()}", flush=True)
    print(f"  {'='*66}", flush=True)
    print(f"  Robot:            {args_cli.robot}", flush=True)
    print(f"  Terrain:          {phase_cfg.terrain}", flush=True)
    print(f"  Envs:             {env_cfg.scene.num_envs}", flush=True)
    print(f"  Max iterations:   {agent_cfg.max_iterations}", flush=True)
    print(f"  LR:               {lr_max:.1e} -> {lr_min:.1e}", flush=True)
    print(f"  Noise bounds:     [{noise_bounds['min']}, {noise_bounds['max']}]", flush=True)
    print(f"  AI Coach:         {'ENABLED' if coach else 'DISABLED'}", flush=True)
    if coach:
        print(f"  Coach interval:   every {coach_cfg.check_interval} iters", flush=True)
        print(f"  Coach model:      {coach_cfg.api_model}", flush=True)
    print(f"  Log dir:          {log_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # ── Create environment ──────────────────────────────────────────────
    EnvCfgClass = type(env_cfg)
    try:
        gym.register(
            id=env_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{EnvCfgClass.__module__}:{EnvCfgClass.__name__}",
            },
        )
    except gym.error.Error:
        pass  # Already registered

    env = gym.make(env_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── Create runner ───────────────────────────────────────────────────
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )

    # Resume from checkpoint if specified
    if resume_run and resume_checkpoint:
        resume_path = get_checkpoint_path(log_root_path, resume_run, resume_checkpoint)
        print(f"[AI-TRAIN] Resuming from {resume_path}", flush=True)
        runner.load(resume_path)

    # Register safety clamp
    register_std_safety_clamp(
        runner.alg.policy,
        min_std=noise_bounds["min"],
        max_std=noise_bounds["max"],
    )

    # ── Save config ─────────────────────────────────────────────────────
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # ── Setup AI Coach components ───────────────────────────────────────
    metrics_collector = MetricsCollector(env, runner, phase_name)
    guardrails = Guardrails(coach_cfg, phase_cfg)
    actuator = Actuator(env, runner)
    actuator.register_noise_control(noise_bounds)
    decision_log = DecisionLog(os.path.join(log_dir, coach_cfg.decision_log_path))

    # ── Training loop with AI coaching ──────────────────────────────────
    start_time = time.time()
    best_reward = -float("inf")
    best_checkpoint = None

    initial_lr = cosine_annealing_lr(
        0, agent_cfg.max_iterations, lr_max, lr_min, args_cli.warmup_iters
    )
    set_learning_rate(runner, initial_lr)

    # Monkey-patch the update function for LR schedule + AI coaching
    original_update = runner.alg.update
    _iteration_counter = [0]
    _log_interval = 100
    _lr_cooldown = [0]  # iterations remaining on emergency LR cooldown

    # Storage for per-iteration reward info (captured from runner logging)
    _last_reward_info = [{}]

    def update_with_ai_coach(*args, **kwargs):
        it = _iteration_counter[0]
        elapsed = time.time() - start_time
        hours = elapsed / 3600

        # -- LR schedule --
        if _lr_cooldown[0] > 0:
            _lr_cooldown[0] -= 1
        else:
            lr = cosine_annealing_lr(
                it, agent_cfg.max_iterations, lr_max, lr_min,
                args_cli.warmup_iters
            )
            set_learning_rate(runner, lr)

        # -- Run the actual PPO update --
        result = original_update(*args, **kwargs)

        # -- Collect reward info from env extras --
        try:
            extras = env.unwrapped.extras if hasattr(env.unwrapped, "extras") else {}
            log_data = extras.get("log", {})
            # Merge with any ep_infos from the runner
            if hasattr(runner, "ep_infos") and runner.ep_infos:
                for info in runner.ep_infos:
                    log_data.update(info)
            _last_reward_info[0] = log_data
        except Exception:
            pass

        # -- Periodic logging --
        if it % _log_interval == 0:
            current_lr = runner.alg.optimizer.param_groups[0]["lr"]
            fps = it * steps_per_iter / max(elapsed, 1) if it > 0 else 0
            print(
                f"[AI-TRAIN] iter={it:6d}/{agent_cfg.max_iterations}  "
                f"lr={current_lr:.2e}  elapsed={hours:.1f}h  fps={fps:.0f}",
                flush=True,
            )

        # -- AI Coach check (every N iterations) --
        if coach and it > 0 and it % coach_cfg.check_interval == 0:
            _run_coach_check(it, hours)

        _iteration_counter[0] += 1
        return result

    def _run_coach_check(it: int, hours: float):
        """Run the AI coach check at the current iteration."""
        current_lr = runner.alg.optimizer.param_groups[0]["lr"]

        # Collect metrics
        snapshot = metrics_collector.collect(
            iteration=it,
            elapsed_hours=hours,
            reward_info=_last_reward_info[0],
            lr=current_lr,
        )

        # Track best reward
        nonlocal best_reward, best_checkpoint
        if snapshot.mean_reward > best_reward:
            best_reward = snapshot.mean_reward
            best_checkpoint = os.path.join(log_dir, f"model_{it}.pt")

        # 1. Emergency checks (override coach)
        emergency = guardrails.check_emergency(snapshot)
        if emergency:
            if emergency == "nan_rollback":
                print(f"[AI-COACH] EMERGENCY: NaN detected at iter {it}!",
                      flush=True)
                decision_log.log_emergency(it, phase_name, "nan_rollback",
                                           "NaN in policy parameters")
                # Save what we can and exit
                env.close()
                return

            elif emergency == "halve_lr":
                new_lr = actuator.emergency_halve_lr()
                _lr_cooldown[0] = 50  # hold for 50 iters
                decision_log.log_emergency(
                    it, phase_name, "halve_lr",
                    f"Value loss {snapshot.value_loss:.1f} > 100, "
                    f"LR halved to {new_lr:.2e}")
                return

            elif emergency == "emergency_stop":
                print(f"[AI-COACH] EMERGENCY STOP at iter {it}!", flush=True)
                decision_log.log_emergency(
                    it, phase_name, "emergency_stop",
                    f"action_smoothness={snapshot.reward_breakdown.get('action_smoothness', 'N/A')}")
                return

        # 2. Consult AI coach
        if not coach.is_available:
            return

        plateau = metrics_collector.is_plateau()
        recent_history = list(metrics_collector.history)
        recent_decisions = decision_log.get_recent(coach_cfg.decision_history)

        decision, latency = coach.get_decision(
            snapshot, recent_history, recent_decisions, plateau)

        print(f"[AI-COACH] iter={it} action={decision.action} "
              f"confidence={decision.confidence:.2f} "
              f"latency={latency:.0f}ms", flush=True)
        if decision.reasoning:
            print(f"[AI-COACH] reason: {decision.reasoning}", flush=True)

        # 3. Apply through guardrails
        all_msgs = []
        applied = {}

        if decision.action == "adjust_weights" and decision.weight_changes:
            validated_weights, msgs = guardrails.validate_weight_changes(
                decision.weight_changes, snapshot.current_weights)
            all_msgs.extend(msgs)
            if validated_weights:
                applied = actuator.apply_weight_changes(validated_weights)
                for name, (old, new) in applied.items():
                    print(f"[AI-COACH] CHANGED {name}: {old:.4f} -> {new:.4f}",
                          flush=True)

        if decision.lr_change is not None:
            new_lr, msgs = guardrails.validate_lr_change(decision.lr_change)
            all_msgs.extend(msgs)
            if new_lr is not None:
                old_lr = actuator.apply_lr_change(new_lr)
                applied["lr"] = (old_lr, new_lr)
                print(f"[AI-COACH] CHANGED LR: {old_lr:.2e} -> {new_lr:.2e}",
                      flush=True)

        if decision.noise_change is not None:
            new_noise, msgs = guardrails.validate_noise_change(
                decision.noise_change)
            all_msgs.extend(msgs)
            if new_noise is not None:
                actuator.apply_noise_change(new_noise)
                applied["max_noise_std"] = (noise_bounds["max"], new_noise)
                print(f"[AI-COACH] CHANGED noise: {noise_bounds['max']:.2f} "
                      f"-> {new_noise:.2f}", flush=True)

        if all_msgs:
            for msg in all_msgs:
                print(f"[AI-COACH] guardrail: {msg}", flush=True)

        # 4. Log everything
        decision_log.log_decision(
            iteration=it,
            phase=phase_name,
            metrics=snapshot.to_dict(),
            decision=decision.to_dict(),
            guardrail_msgs=all_msgs,
            applied_changes={k: list(v) if isinstance(v, tuple) else v
                             for k, v in applied.items()},
            api_latency_ms=latency,
        )

    runner.alg.update = update_with_ai_coach

    # ── Run training ────────────────────────────────────────────────────
    print(f"\n[AI-TRAIN] Starting phase {phase_name}...", flush=True)
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    elapsed = time.time() - start_time
    print(f"\n{'='*70}", flush=True)
    print(f"  PHASE {phase_name.upper()} COMPLETE — {elapsed/3600:.1f} hours",
          flush=True)
    print(f"  Best reward: {best_reward:.1f}", flush=True)
    print(f"  Checkpoints: {log_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Check go/no-go for phase advancement
    should_advance = False
    if coach:
        go, failures = metrics_collector.go_no_go(phase_cfg)
        if go:
            print(f"[AI-TRAIN] GO for next phase! All criteria met.", flush=True)
            should_advance = True
        else:
            print(f"[AI-TRAIN] NO-GO for next phase: {failures}", flush=True)

    # Save final checkpoint
    final_ckpt = os.path.join(log_dir, f"model_{_iteration_counter[0]}.pt")
    actuator.save_checkpoint(final_ckpt)

    env.close()
    return final_ckpt, should_advance


# ── 4. Main ─────────────────────────────────────────────────────────────

def main():
    coach_cfg = CoachConfig(
        check_interval=args_cli.coach_interval,
        api_model=args_cli.coach_model,
    )

    # Initialize AI coach
    coach = None
    if not args_cli.no_coach:
        api_key = args_cli.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            start_phase_cfg = PHASE_CONFIGS[args_cli.start_phase]
            coach = Coach(coach_cfg, start_phase_cfg, api_key=api_key)
            print("[AI-TRAIN] AI Coach initialized", flush=True)
        else:
            print("[AI-TRAIN] WARNING: No API key, running without AI coach",
                  flush=True)

    # Determine phase range
    start_idx = PHASE_ORDER.index(args_cli.start_phase)
    end_idx = PHASE_ORDER.index(args_cli.end_phase)
    phases = PHASE_ORDER[start_idx:end_idx + 1]

    print(f"[AI-TRAIN] Training phases: {' → '.join(phases)}", flush=True)

    resume_run = args_cli.load_run
    resume_ckpt = args_cli.load_checkpoint

    for i, phase_name in enumerate(phases):
        print(f"\n[AI-TRAIN] === PHASE {i+1}/{len(phases)}: {phase_name} ===",
              flush=True)

        # Update coach phase config
        if coach:
            coach.update_phase(PHASE_CONFIGS[phase_name])

        # Load fresh configs for each phase (terrain changes need new env)
        env_cfg, agent_cfg, env_id = load_robot_configs(args_cli.robot)
        apply_phase_terrain(env_cfg, phase_name)

        checkpoint, should_advance = train_phase(
            phase_name=phase_name,
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            env_id=env_id,
            coach=coach,
            coach_cfg=coach_cfg,
            resume_run=resume_run,
            resume_checkpoint=resume_ckpt,
        )

        if not should_advance and i < len(phases) - 1:
            print(f"[AI-TRAIN] Phase {phase_name} did not meet advancement "
                  f"criteria. Stopping.", flush=True)
            # Write state file for potential manual resume
            state = {
                "last_phase": phase_name,
                "checkpoint": checkpoint,
                "next_phase": phases[i + 1] if i + 1 < len(phases) else None,
            }
            with open("ai_train_state.json", "w") as f:
                json.dump(state, f, indent=2)
            break

        # Next phase resumes from this phase's checkpoint
        if checkpoint:
            resume_run = os.path.dirname(checkpoint).split("/")[-1]
            resume_ckpt = os.path.basename(checkpoint)
        else:
            resume_run = None
            resume_ckpt = None

    print("\n[AI-TRAIN] All phases complete!", flush=True)


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)
