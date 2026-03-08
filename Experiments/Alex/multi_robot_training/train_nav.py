"""Phase C: Navigation Policy Training for Spot.

Trains a nav policy that outputs velocity commands [vx, vy, wz] to a frozen
Phase B locomotion policy. The nav policy uses LiDAR + depth camera to avoid
obstacles and reach goals.

Architecture:
    Nav Policy (10 Hz) --> [vx, vy, wz] --> Frozen Loco Policy (50 Hz) --> [12 joints]

Input modes (at inference time):
    1. Teleop: Human velocity commands bypass nav policy entirely
    2. Waypoint: Give (X,Y) goal, nav policy steers there
    3. Vector: Give heading + distance, converted to waypoint

Usage (H100):
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/multi_robot_training/train_nav.py --headless \\
        --loco_checkpoint logs/rsl_rl/spot_robust_ppo/BEST/model_XXXX.pt \\
        --num_envs 512 --max_iterations 20000 --save_interval 100 \\
        --no_wandb

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

# ── 0. Parse args BEFORE any Isaac imports ──────────────────────────────
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase C: Navigation PPO training for Spot")
parser.add_argument("--loco_checkpoint", type=str, required=True,
                    help="Path to frozen Phase B loco checkpoint (model_XXXX.pt)")
parser.add_argument("--num_envs", type=int, default=512,
                    help="Number of parallel environments (limited by camera rendering)")
parser.add_argument("--max_iterations", type=int, default=20000,
                    help="Max training iterations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--save_interval", type=int, default=100,
                    help="Checkpoint save interval")
parser.add_argument("--lr_max", type=float, default=3e-4,
                    help="Max learning rate")
parser.add_argument("--lr_min", type=float, default=1e-5,
                    help="Min learning rate")
parser.add_argument("--warmup_iters", type=int, default=50,
                    help="LR warmup iterations")
parser.add_argument("--arena_type", type=str, default="mixed",
                    choices=["sparse", "dense", "corridor", "mixed"],
                    help="Arena obstacle density")
parser.add_argument("--no_wandb", action="store_true", default=False,
                    help="Disable Weights & Biases logging")
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

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from shared.lr_schedule import cosine_annealing_lr, set_learning_rate
from shared.training_utils import configure_tf32
from shared.loco_wrapper import FrozenLocoPolicy

configure_tf32()


# ── 2. Config Loading ───────────────────────────────────────────────────

def load_nav_configs():
    """Load nav env and PPO configs."""
    from configs.spot_nav_env_cfg import SpotNavEnvCfg
    from configs.spot_nav_ppo_cfg import SpotNavPPORunnerCfg
    return SpotNavEnvCfg(), SpotNavPPORunnerCfg(), "Isaac-Nav-Spot-PPO-v0"


# ── 3. Main ─────────────────────────────────────────────────────────────

def main():
    env_cfg, agent_cfg, env_id = load_nav_configs()

    # Apply CLI overrides
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed
    agent_cfg.save_interval = args_cli.save_interval

    if args_cli.no_wandb:
        agent_cfg.logger = "tensorboard"

    # Load frozen loco policy
    loco_policy = FrozenLocoPolicy(
        checkpoint_path=args_cli.loco_checkpoint,
        obs_dim=235,
        action_dim=12,
        vel_cmd_indices=(9, 10, 11),
        device="cuda",
    )

    # Logging
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)

    steps_per_iter = env_cfg.scene.num_envs * agent_cfg.num_steps_per_env

    print(f"\n{'='*70}", flush=True)
    print(f"  PHASE C: NAVIGATION TRAINING — SPOT", flush=True)
    print(f"  {'='*66}", flush=True)
    print(f"  Envs:             {env_cfg.scene.num_envs}", flush=True)
    print(f"  Max iterations:   {agent_cfg.max_iterations}", flush=True)
    print(f"  Loco checkpoint:  {args_cli.loco_checkpoint}", flush=True)
    print(f"  Arena type:       {args_cli.arena_type}", flush=True)
    print(f"  Nav policy:       {agent_cfg.policy.actor_hidden_dims}", flush=True)
    print(f"  LR schedule:      cosine {args_cli.lr_max} -> {args_cli.lr_min}", flush=True)
    print(f"  Log dir:          {log_dir}", flush=True)
    print(f"  Steps/iteration:  {steps_per_iter:,}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # ── Create environment ──────────────────────────────────────────────
    EnvCfgClass = type(env_cfg)
    gym.register(
        id=env_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{EnvCfgClass.__module__}:{EnvCfgClass.__name__}",
        },
    )
    env = gym.make(env_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Store loco policy reference for the env step wrapper
    env.unwrapped.loco_policy = loco_policy

    # ── Create runner ───────────────────────────────────────────────────
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )

    # ── Save config ─────────────────────────────────────────────────────
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    with open(os.path.join(log_dir, "params", "training_params.txt"), "w") as f:
        f.write(f"loco_checkpoint: {args_cli.loco_checkpoint}\n")
        f.write(f"num_envs: {env_cfg.scene.num_envs}\n")
        f.write(f"max_iterations: {agent_cfg.max_iterations}\n")
        f.write(f"arena_type: {args_cli.arena_type}\n")
        f.write(f"lr_max: {args_cli.lr_max}\n")
        f.write(f"lr_min: {args_cli.lr_min}\n")
        f.write(f"nav_policy: {agent_cfg.policy.actor_hidden_dims}\n")

    # ── Training loop ───────────────────────────────────────────────────
    print(f"\n[NAV-TRAIN] Starting navigation training...", flush=True)
    start_time = time.time()

    initial_lr = cosine_annealing_lr(
        0, agent_cfg.max_iterations,
        args_cli.lr_max, args_cli.lr_min, args_cli.warmup_iters
    )
    set_learning_rate(runner, initial_lr)

    # LR schedule wrapper
    original_update = runner.alg.update
    _iteration_counter = [0]
    _lr_log_interval = 500

    def update_with_schedule(*args, **kwargs):
        it = _iteration_counter[0]
        lr = cosine_annealing_lr(
            it, agent_cfg.max_iterations,
            args_cli.lr_max, args_cli.lr_min, args_cli.warmup_iters
        )
        set_learning_rate(runner, lr)

        result = original_update(*args, **kwargs)

        if it % _lr_log_interval == 0:
            elapsed = time.time() - start_time
            hours = elapsed / 3600
            fps = it * steps_per_iter / max(elapsed, 1) if it > 0 else 0
            print(
                f"[NAV-TRAIN] iter={it:6d}/{agent_cfg.max_iterations}  "
                f"lr={lr:.2e}  elapsed={hours:.1f}h  fps={fps:.0f}",
                flush=True,
            )

        _iteration_counter[0] += 1
        return result

    runner.alg.update = update_with_schedule

    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    elapsed = time.time() - start_time
    print(f"\n{'='*70}", flush=True)
    print(f"  NAV TRAINING COMPLETE — {elapsed/3600:.1f} hours", flush=True)
    print(f"  Checkpoints: {log_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)
