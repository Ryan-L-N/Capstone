"""Phase 2a: Teacher Training with Privileged Observations.

Trains a teacher policy with privileged observations (friction, contact forces)
that are only available in simulation. Requires weight surgery to extend the
Phase 1 checkpoint from 235-dim to the teacher's expanded input dim.

Usage (H100 — Spot):
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/multi_robot_training/train_teacher.py --headless \\
        --robot spot --checkpoint /path/to/phase1_best.pt --num_envs 8192

Usage (H100 — Vision60):
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/multi_robot_training/train_teacher.py --headless \\
        --robot vision60 --checkpoint /path/to/phase1_best.pt --num_envs 8192

Created for AI2C Tech Capstone — MS for Autonomy, February 2026
"""

# ── 0. Parse args BEFORE any Isaac imports ──────────────────────────────
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 2a: Teacher training with privileged obs")
parser.add_argument("--robot", type=str, required=True, choices=["spot", "vision60"],
                    help="Robot to train: spot or vision60")
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--max_iterations", type=int, default=10000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to Phase 1 best checkpoint")
parser.add_argument("--no_wandb", action="store_true", default=False)
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

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from shared.training_utils import configure_tf32
configure_tf32()


# ── 2. Weight Surgery ───────────────────────────────────────────────────

def extend_checkpoint_for_teacher(checkpoint_path: str, standard_obs_dim: int, teacher_obs_dim: int):
    """Extend a Phase 1 checkpoint to accept teacher's privileged observations.

    Finds first layer weights shaped [hidden, standard_obs_dim] and extends
    them to [hidden, teacher_obs_dim] by appending zero-initialized columns.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    extra_dims = teacher_obs_dim - standard_obs_dim
    modified = {}
    for key, tensor in state_dict.items():
        if key.endswith(".0.weight") and tensor.shape[1] == standard_obs_dim:
            extra_cols = torch.zeros(tensor.shape[0], extra_dims)
            modified[key] = torch.cat([tensor, extra_cols], dim=1)
            print(f"  [SURGERY] {key}: {tensor.shape} -> {modified[key].shape}", flush=True)
        else:
            modified[key] = tensor

    if "model_state_dict" in checkpoint:
        checkpoint["model_state_dict"] = modified
        return checkpoint
    return modified


# ── 3. Main ─────────────────────────────────────────────────────────────

def main():
    robot = args_cli.robot

    # Load robot-specific teacher env config
    if robot == "spot":
        from configs.spot_teacher_env_cfg import SpotTeacherEnvCfg
        from configs.spot_ppo_cfg import SpotPPORunnerCfg
        env_cfg = SpotTeacherEnvCfg()
        agent_cfg = SpotPPORunnerCfg()
        env_id = "Isaac-Velocity-Teacher-Spot-v0"
        EnvCfgClass = SpotTeacherEnvCfg
    else:
        from configs.vision60_teacher_env_cfg import Vision60TeacherEnvCfg
        from configs.vision60_ppo_cfg import Vision60PPORunnerCfg
        env_cfg = Vision60TeacherEnvCfg()
        agent_cfg = Vision60PPORunnerCfg()
        env_id = "Isaac-Velocity-Teacher-Vision60-v0"
        EnvCfgClass = Vision60TeacherEnvCfg

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed
    agent_cfg.experiment_name = f"{robot}_teacher"

    if args_cli.no_wandb:
        agent_cfg.logger = "tensorboard"

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    print(f"\n{'='*70}", flush=True)
    print(f"  PHASE 2a: TEACHER TRAINING — {robot.upper()}", flush=True)
    print(f"  {'='*66}", flush=True)
    print(f"  Checkpoint:       {args_cli.checkpoint}", flush=True)
    print(f"  Envs:             {env_cfg.scene.num_envs}", flush=True)
    print(f"  Max iterations:   {agent_cfg.max_iterations}", flush=True)
    print(f"  Network:          {agent_cfg.policy.actor_hidden_dims}", flush=True)
    print(f"  Log dir:          {log_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Create environment
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

    # Determine teacher observation dimension from environment
    teacher_obs_dim = env.observation_space.shape[0]
    standard_obs_dim = 235
    print(f"[INFO] Teacher obs dim: {teacher_obs_dim} (standard: {standard_obs_dim})", flush=True)

    # Create runner
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )

    # Weight surgery: extend checkpoint
    print("[INFO] Performing weight surgery on checkpoint...", flush=True)
    extended_checkpoint = extend_checkpoint_for_teacher(
        args_cli.checkpoint, standard_obs_dim, teacher_obs_dim
    )

    if "model_state_dict" in extended_checkpoint:
        runner.alg.actor_critic.load_state_dict(extended_checkpoint["model_state_dict"])
    else:
        runner.alg.actor_critic.load_state_dict(extended_checkpoint)
    print("[INFO] Extended checkpoint loaded successfully", flush=True)

    # Save config
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Train
    print(f"\n[TRAIN] Starting {robot.upper()} teacher training...", flush=True)
    start_time = time.time()

    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    elapsed = time.time() - start_time
    hours = elapsed / 3600
    print(f"\n{'='*70}", flush=True)
    print(f"  PHASE 2a TEACHER TRAINING COMPLETE — {robot.upper()}", flush=True)
    print(f"  Total time: {hours:.1f} hours", flush=True)
    print(f"  Checkpoints: {log_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)
