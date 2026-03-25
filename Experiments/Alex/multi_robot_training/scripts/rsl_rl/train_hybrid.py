"""Mason Hybrid Training — No AI Coach.

Mason's proven reward weights + our robust terrain + our safety fixes.
No AI coach, no VLM — just let the config do its thing.

Key features:
  - Mason's [512, 256, 128] network (800K params, generalizes better)
  - Mason's adaptive KL learning rate (no cosine annealing)
  - Mason's 11 reward terms + terrain_relative_height + dof_pos_limits
  - Our ROBUST_TERRAINS_CFG (12 types, 10 difficulty rows)
  - Fixed 0.37m standing height (anti-belly-crawl)
  - Clamped penalties (Bug #29 safety)
  - Value loss watchdog (Bug #25)
  - Noise std safety clamp

Usage (H100):
    python scripts/rsl_rl/train_hybrid.py \\
      --headless --no_wandb \\
      --num_envs 4096 --save_interval 100 \\
      --max_noise_std 1.0 --max_iterations 20000

Usage (local smoke test):
    python scripts/rsl_rl/train_hybrid.py \\
      --headless --no_wandb \\
      --num_envs 250 --max_iterations 100

Resume from checkpoint:
    python scripts/rsl_rl/train_hybrid.py \\
      --headless --no_wandb \\
      --num_envs 4096 --save_interval 100 \\
      --max_noise_std 1.0 \\
      --load_run 2026-03-11_XX-XX-XX --load_checkpoint model_XXXX.pt

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

# ── 0. Parse args BEFORE any Isaac imports ──────────────────────────────
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Mason Hybrid PPO training (no AI coach)")
parser.add_argument("--num_envs", type=int, default=4096,
                    help="Number of parallel environments (default 4096 — Mason's)")
parser.add_argument("--max_iterations", type=int, default=20000,
                    help="Max training iterations (default 20000 — Mason's)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--save_interval", type=int, default=100,
                    help="Checkpoint save interval (default 100 — never lose >65M steps)")
parser.add_argument("--no_wandb", action="store_true", default=False,
                    help="Disable Weights & Biases logging (use TensorBoard)")
parser.add_argument("--max_noise_std", type=float, default=1.0,
                    help="Maximum noise std ceiling (Mason's init is 1.0, adaptive KL manages)")
parser.add_argument("--min_noise_std", type=float, default=0.2,
                    help="Minimum noise std floor")
parser.add_argument("--load_run", type=str, default=None,
                    help="Run directory to resume from (e.g. 2026-03-11_10-52-57)")
parser.add_argument("--load_checkpoint", type=str, default="model_.*\\.pt",
                    help="Checkpoint regex to load (default: latest)")
parser.add_argument("--resume", action="store_true", default=False,
                    help="Resume from latest checkpoint in latest run")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 1. Imports (AFTER SimulationApp) ────────────────────────────────────
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Trigger gym registrations
import isaaclab_tasks  # noqa: F401
import quadruped_locomotion  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path
from quadruped_locomotion.utils.training_utils import (
    configure_tf32,
    clamp_noise_std,
    register_std_safety_clamp,
)

# Enable TF32
configure_tf32()


# ── 2. Config Loading ──────────────────────────────────────────────────

def load_configs():
    """Load Mason Hybrid env and agent configs."""
    from quadruped_locomotion.tasks.locomotion.config.spot.mason_hybrid_env_cfg import (
        SpotMasonHybridEnvCfg,
    )
    from quadruped_locomotion.tasks.locomotion.config.spot.agents.rsl_rl_mason_hybrid_cfg import (
        SpotMasonHybridPPORunnerCfg,
    )
    return SpotMasonHybridEnvCfg(), SpotMasonHybridPPORunnerCfg(), "Locomotion-MasonHybrid-Spot-v0"


# ── 3. Main ─────────────────────────────────────────────────────────────

def main():
    env_cfg, agent_cfg, env_id = load_configs()

    # Apply CLI overrides
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed
    agent_cfg.save_interval = args_cli.save_interval

    # W&B toggle
    if args_cli.no_wandb:
        agent_cfg.logger = "tensorboard"

    # Logging
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    steps_per_iter = env_cfg.scene.num_envs * agent_cfg.num_steps_per_env

    print(f"\n{'='*70}", flush=True)
    print(f"  MASON HYBRID TRAINING — NO AI COACH", flush=True)
    print(f"  {'='*66}", flush=True)
    print(f"  Config:           Mason's rewards + our terrain + safety fixes", flush=True)
    print(f"  Envs:             {env_cfg.scene.num_envs}", flush=True)
    print(f"  Max iterations:   {agent_cfg.max_iterations}", flush=True)
    print(f"  Save interval:    {agent_cfg.save_interval}", flush=True)
    print(f"  LR schedule:      ADAPTIVE KL (Mason's — starts 1e-3, self-adjusts)", flush=True)
    print(f"  Noise bounds:     [{args_cli.min_noise_std}, {args_cli.max_noise_std}]", flush=True)
    print(f"  Network:          {agent_cfg.policy.actor_hidden_dims}", flush=True)
    print(f"  Episode length:   {env_cfg.episode_length_s}s", flush=True)
    print(f"  Terrain:          ROBUST (12 types, 10 difficulty rows)", flush=True)
    print(f"  Height target:    0.37m fixed (anti-belly-crawl)", flush=True)
    print(f"  AI Coach:         DISABLED", flush=True)
    print(f"  Logger:           {agent_cfg.logger}", flush=True)
    print(f"  Log dir:          {log_dir}", flush=True)
    print(f"  Steps/iteration:  {steps_per_iter:,}", flush=True)
    print(f"  Est. total steps: {steps_per_iter * agent_cfg.max_iterations / 1e9:.1f}B", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Print reward weights for verification
    print("[CONFIG] Reward weights:", flush=True)
    for attr_name in sorted(dir(env_cfg.rewards)):
        attr = getattr(env_cfg.rewards, attr_name)
        if hasattr(attr, "weight"):
            print(f"  {attr_name}: {attr.weight}", flush=True)
    print(flush=True)

    # ── Create environment ──────────────────────────────────────────────
    env = gym.make(env_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── Create runner ───────────────────────────────────────────────────
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )

    # ── Safety: clamp std inside PPO update mini-batches ────────────────
    register_std_safety_clamp(
        runner.alg.policy, args_cli.min_noise_std, args_cli.max_noise_std
    )
    print(f"[SAFETY] Registered std clamp: [{args_cli.min_noise_std}, {args_cli.max_noise_std}]",
          flush=True)

    # ── Resume if requested ─────────────────────────────────────────────
    start_iteration = 0
    if args_cli.resume or args_cli.load_run:
        resume_path = get_checkpoint_path(
            log_root_path, args_cli.load_run, args_cli.load_checkpoint
        )
        print(f"[INFO] Resuming from: {resume_path}", flush=True)
        runner.load(resume_path)
        ckpt_name = os.path.basename(resume_path)
        try:
            start_iteration = int(ckpt_name.split("_")[1].split(".")[0]) + 1
            print(f"[INFO] Resuming from iteration {start_iteration}", flush=True)
        except (IndexError, ValueError):
            print("[WARN] Could not parse iteration from checkpoint name", flush=True)

    # ── Save config ─────────────────────────────────────────────────────
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    with open(os.path.join(log_dir, "params", "training_params.txt"), "w") as f:
        f.write(f"config: mason_hybrid (no AI coach)\n")
        f.write(f"num_envs: {env_cfg.scene.num_envs}\n")
        f.write(f"max_iterations: {agent_cfg.max_iterations}\n")
        f.write(f"lr_schedule: adaptive_kl (starts 1e-3)\n")
        f.write(f"noise_bounds: [{args_cli.min_noise_std}, {args_cli.max_noise_std}]\n")
        f.write(f"seed: {args_cli.seed}\n")
        f.write(f"steps_per_iter: {steps_per_iter}\n")
        f.write(f"network: {agent_cfg.policy.actor_hidden_dims}\n")
        f.write(f"terrain: robust (12 types)\n")
        f.write(f"height_target: 0.37m fixed\n")
        f.write(f"logger: {agent_cfg.logger}\n")

    # ── Value loss watchdog + noise clamp (monkey-patch PPO update) ─────
    _VL_THRESHOLD = 100.0
    _VL_COOLDOWN_ITERS = 50
    _vl_cooldown = [0]
    _vl_spike_count = [0]
    _iteration_counter = [start_iteration]
    _log_interval = 500

    original_update = runner.alg.update
    start_time = time.time()

    def update_with_safety(*args, **kwargs):
        """Wrapper: noise clamp + value loss watchdog. No LR override — adaptive KL handles it."""
        it = _iteration_counter[0]

        # Run the actual PPO update (adaptive KL adjusts LR internally)
        result = original_update(*args, **kwargs)

        # Value loss watchdog — detect instability
        vl = result.get("value_function", 0.0) if isinstance(result, dict) else 0.0
        if vl > _VL_THRESHOLD:
            _vl_spike_count[0] += 1
            _vl_cooldown[0] = _VL_COOLDOWN_ITERS
            print(
                f"[GUARD] Value loss spike: {vl:.1f} > {_VL_THRESHOLD} at iter {it} "
                f"(spike #{_vl_spike_count[0]}). Adaptive KL should self-correct.",
                flush=True,
            )

        if _vl_cooldown[0] > 0:
            _vl_cooldown[0] -= 1

        # Safety: clamp noise std
        clamp_noise_std(runner.alg.policy, args_cli.min_noise_std, args_cli.max_noise_std)

        # Periodic console logging
        if it % _log_interval == 0 and it > 0:
            elapsed = time.time() - start_time
            hours = elapsed / 3600
            iters_done = it - start_iteration
            fps = iters_done * steps_per_iter / max(elapsed, 1) if iters_done > 0 else 0
            noise = runner.alg.policy.std.mean().item() if hasattr(runner.alg.policy, "std") else 0

            # Get current LR from optimizer
            current_lr = runner.alg.optimizer.param_groups[0]["lr"]

            guard_str = f"  [VL spikes: {_vl_spike_count[0]}]" if _vl_spike_count[0] > 0 else ""

            print(
                f"[TRAIN] iter={it:6d}/{agent_cfg.max_iterations}  "
                f"lr={current_lr:.2e}  noise={noise:.3f}  "
                f"elapsed={hours:.1f}h  fps={fps:.0f}{guard_str}",
                flush=True,
            )

        _iteration_counter[0] += 1
        return result

    runner.alm = update_with_safety  # typo guard — overwrite the right thing
    runner.alg.update = update_with_safety

    # ── Run training ────────────────────────────────────────────────────
    print(f"[TRAIN] Starting Mason Hybrid training (adaptive KL LR)...", flush=True)

    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations - start_iteration,
        init_at_random_ep_len=True,
    )

    # ── Training complete ───────────────────────────────────────────────
    elapsed = time.time() - start_time
    hours = elapsed / 3600
    total_steps = (agent_cfg.max_iterations - start_iteration) * steps_per_iter

    print(f"\n{'='*70}", flush=True)
    print(f"  MASON HYBRID TRAINING COMPLETE", flush=True)
    print(f"  {'='*66}", flush=True)
    print(f"  Total time:       {hours:.1f} hours ({elapsed:.0f} seconds)", flush=True)
    print(f"  Total steps:      {total_steps / 1e9:.1f}B", flush=True)
    print(f"  Avg throughput:   {total_steps / max(elapsed, 1):.0f} steps/sec", flush=True)
    print(f"  VL spikes:        {_vl_spike_count[0]}", flush=True)
    print(f"  Checkpoints:      {log_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    # Use os._exit(0) to avoid CUDA deadlock on close
    import os as _os
    _os._exit(0)
