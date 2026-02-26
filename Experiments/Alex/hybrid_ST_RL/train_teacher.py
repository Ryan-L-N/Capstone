"""
Stage 2a: Teacher Training with Privileged Observations — H100 NVL 96GB
========================================================================

Trains a teacher policy with privileged observations (friction, terrain type,
contact forces) that are only available in simulation. The teacher learns to
leverage this extra information for better terrain adaptation.

Requires weight surgery: Stage 1 checkpoint has 235-dim input, teacher needs
254-dim input. The first layer is extended with zero-initialized columns for
the 19 privileged dimensions.

Usage (on H100 server):
    cd ~/IsaacLab
    ./isaaclab.sh -p /path/to/train_teacher.py --headless --num_envs 8192 \\
        --checkpoint /path/to/stage1_best_model.pt

Created for AI2C Tech Capstone — hybrid_ST_RL, February 2026
"""

# ── 0. Parse args BEFORE any Isaac imports ──────────────────────────────
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stage 2a: Teacher training with privileged obs")
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--max_iterations", type=int, default=20000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to Stage 1 best checkpoint")
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

from configs.teacher_env_cfg import SpotTeacherEnvCfg
from configs.teacher_ppo_cfg import SpotTeacherPPORunnerCfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ── 2. Weight Surgery ───────────────────────────────────────────────────

def extend_checkpoint_for_teacher(checkpoint_path: str, standard_obs_dim: int = 235):
    """Extend a Stage 1 checkpoint to accept teacher's privileged observations.

    The Stage 1 checkpoint has weights shaped for 235-dim input.
    The teacher network needs 254-dim input (235 + 19 privileged).

    This function:
    1. Loads the checkpoint state_dict
    2. Finds the first layer weights (shape [hidden, 235])
    3. Extends them to [hidden, 254] by appending zero-initialized columns
    4. Does the same for the critic
    5. Returns the modified state_dict

    Args:
        checkpoint_path: Path to Stage 1 model_XXXXX.pt
        standard_obs_dim: Expected observation dimension of the checkpoint (235)

    Returns:
        Modified state_dict compatible with the teacher network.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # RSL-RL saves model state as 'model_state_dict' inside the checkpoint
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    modified = {}
    for key, tensor in state_dict.items():
        # Find first layer of actor and critic (input layers)
        # RSL-RL naming: actor.0.weight, actor.0.bias, critic.0.weight, etc.
        if (key.endswith(".0.weight") and tensor.shape[1] == standard_obs_dim):
            # Extend: [hidden, 235] -> [hidden, 254]
            extra_cols = torch.zeros(tensor.shape[0], 254 - standard_obs_dim)
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
    env_cfg = SpotTeacherEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    agent_cfg = SpotTeacherPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    print(f"\n{'='*70}", flush=True)
    print(f"  HYBRID ST-RL — STAGE 2a: TEACHER TRAINING", flush=True)
    print(f"  {'='*66}", flush=True)
    print(f"  Checkpoint:       {args_cli.checkpoint}", flush=True)
    print(f"  Envs:             {env_cfg.scene.num_envs}", flush=True)
    print(f"  Max iterations:   {agent_cfg.max_iterations}", flush=True)
    print(f"  Learning rate:    {agent_cfg.algorithm.learning_rate}", flush=True)
    print(f"  Network:          {agent_cfg.policy.actor_hidden_dims}", flush=True)
    print(f"  Obs dims:         235 standard + 19 privileged = 254", flush=True)
    print(f"  Log dir:          {log_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Create environment
    gym.register(
        id="Isaac-Velocity-Teacher-Spot-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{SpotTeacherEnvCfg.__module__}:{SpotTeacherEnvCfg.__name__}",
        },
    )
    env = gym.make("Isaac-Velocity-Teacher-Spot-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Create runner (with 254-dim input)
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )

    # Weight surgery: extend Stage 1 checkpoint from 235 -> 254 dims
    print("[INFO] Performing weight surgery on checkpoint...", flush=True)
    extended_checkpoint = extend_checkpoint_for_teacher(args_cli.checkpoint)

    # Load the extended checkpoint
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
    print("\n[TRAIN] Starting teacher training...", flush=True)
    start_time = time.time()

    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    elapsed = time.time() - start_time
    hours = elapsed / 3600
    print(f"\n{'='*70}", flush=True)
    print(f"  STAGE 2a TEACHER TRAINING COMPLETE", flush=True)
    print(f"  Total time: {hours:.1f} hours", flush=True)
    print(f"  Checkpoints: {log_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
