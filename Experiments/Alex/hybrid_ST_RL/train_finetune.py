"""
Stage 1: Progressive Fine-Tuning from 48hr Rough Policy — H100 NVL 96GB
=========================================================================

Fine-tunes the working 48hr rough policy (model_27500.pt, [512,256,128])
on 12 terrain types with progressively expanding domain randomization.

Key innovations:
  - Warm start from checkpoint that already walks (not from scratch)
  - Progressive DR: friction starts [0.3,1.3] -> expands to [0.1,1.5] over 15K iters
  - Same [512,256,128] architecture — checkpoint loads directly
  - Conservative LR (1e-4) with adaptive KL to prevent catastrophic forgetting

Usage (on H100 server):
    cd ~/IsaacLab
    ./isaaclab.sh -p /path/to/train_finetune.py --headless --num_envs 16384 \\
        --checkpoint /path/to/model_27500.pt

Usage (local debug — RTX 2000 Ada):
    cd C:\\IsaacLab
    isaaclab.bat -p /path/to/train_finetune.py --headless --num_envs 64 \\
        --max_iterations 10 --dr_expansion_iters 5

Created for AI2C Tech Capstone — hybrid_ST_RL, February 2026
"""

# ── 0. Parse args BEFORE any Isaac imports ──────────────────────────────
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stage 1: Progressive fine-tuning")
parser.add_argument("--num_envs", type=int, default=16384,
                    help="Number of parallel environments (default 16384 for H100)")
parser.add_argument("--max_iterations", type=int, default=25000,
                    help="Max training iterations (default 25000)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to 48hr rough policy checkpoint (model_27500.pt)")
parser.add_argument("--dr_expansion_iters", type=int, default=15000,
                    help="Iterations over which DR expands from easy to hard (default 15000)")
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

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils import configclass

# Trigger standard gym registrations
import isaaclab_tasks  # noqa: F401

# Add our project to path for custom configs
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from configs.finetune_env_cfg import SpotFinetuneEnvCfg
from configs.finetune_ppo_cfg import SpotFinetunePPORunnerCfg

# Enable TF32 for faster matmul on H100
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# ── 2. Progressive Domain Randomization ─────────────────────────────────

def lerp(start: float, end: float, fraction: float) -> float:
    """Linear interpolation: start + (end - start) * clamp(fraction, 0, 1)."""
    fraction = max(0.0, min(1.0, fraction))
    return start + (end - start) * fraction


# DR schedule: (start_value, end_value) for each parameter
DR_SCHEDULE = {
    # Friction
    "static_friction_min":  (0.3,  0.1),
    "static_friction_max":  (1.3,  1.5),
    "dynamic_friction_min": (0.25, 0.08),
    "dynamic_friction_max": (1.1,  1.2),
    # Push robot
    "push_velocity":        (0.5,  1.0),
    "push_interval_min":    (10.0, 6.0),
    "push_interval_max":    (15.0, 13.0),
    # External forces
    "ext_force":            (3.0,  6.0),
    "ext_torque":           (1.0,  2.5),
    # Mass
    "mass_offset":          (5.0,  7.0),
    # Joint velocity reset
    "joint_vel_range":      (2.5,  3.0),
}


def update_dr_params(env, iteration: int, expansion_iters: int) -> dict:
    """Progressively expand domain randomization ranges.

    Args:
        env: The wrapped Isaac Lab environment.
        iteration: Current training iteration.
        expansion_iters: Number of iterations over which DR expands.

    Returns:
        Dict of current DR values for logging.
    """
    fraction = min(iteration / max(expansion_iters, 1), 1.0)
    cfg = env.unwrapped.cfg

    # --- Friction ---
    sf_min = lerp(*DR_SCHEDULE["static_friction_min"], fraction)
    sf_max = lerp(*DR_SCHEDULE["static_friction_max"], fraction)
    df_min = lerp(*DR_SCHEDULE["dynamic_friction_min"], fraction)
    df_max = lerp(*DR_SCHEDULE["dynamic_friction_max"], fraction)
    cfg.events.physics_material.params["static_friction_range"] = (sf_min, sf_max)
    cfg.events.physics_material.params["dynamic_friction_range"] = (df_min, df_max)

    # --- Push robot ---
    push_vel = lerp(*DR_SCHEDULE["push_velocity"], fraction)
    cfg.events.push_robot.params["velocity_range"] = {
        "x": (-push_vel, push_vel), "y": (-push_vel, push_vel)
    }
    push_min = lerp(*DR_SCHEDULE["push_interval_min"], fraction)
    push_max = lerp(*DR_SCHEDULE["push_interval_max"], fraction)
    cfg.events.push_robot.interval_range_s = (push_min, push_max)

    # --- External force / torque ---
    ext_force = lerp(*DR_SCHEDULE["ext_force"], fraction)
    ext_torque = lerp(*DR_SCHEDULE["ext_torque"], fraction)
    cfg.events.base_external_force_torque.params["force_range"] = (-ext_force, ext_force)
    cfg.events.base_external_force_torque.params["torque_range"] = (-ext_torque, ext_torque)

    # --- Mass offset ---
    mass_offset = lerp(*DR_SCHEDULE["mass_offset"], fraction)
    cfg.events.add_base_mass.params["mass_distribution_params"] = (-mass_offset, mass_offset)

    # --- Joint velocity reset range ---
    jv_range = lerp(*DR_SCHEDULE["joint_vel_range"], fraction)
    cfg.events.reset_robot_joints.params["velocity_range"] = (-jv_range, jv_range)

    return {
        "dr_fraction": fraction,
        "friction_range": f"[{sf_min:.2f}, {sf_max:.2f}]",
        "push_vel": push_vel,
        "ext_force": ext_force,
        "mass_offset": mass_offset,
    }


# ── 3. Main ─────────────────────────────────────────────────────────────

def main():
    # --- Environment config ---
    env_cfg = SpotFinetuneEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # --- Agent config ---
    agent_cfg = SpotFinetunePPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed

    # --- Logging ---
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    print(f"\n{'='*70}", flush=True)
    print(f"  HYBRID ST-RL — STAGE 1: PROGRESSIVE FINE-TUNING", flush=True)
    print(f"  {'='*66}", flush=True)
    print(f"  Checkpoint:       {args_cli.checkpoint}", flush=True)
    print(f"  Envs:             {env_cfg.scene.num_envs}", flush=True)
    print(f"  Max iterations:   {agent_cfg.max_iterations}", flush=True)
    print(f"  DR expansion:     {args_cli.dr_expansion_iters} iterations", flush=True)
    print(f"  Save interval:    {agent_cfg.save_interval}", flush=True)
    print(f"  Learning rate:    {agent_cfg.algorithm.learning_rate}", flush=True)
    print(f"  LR schedule:      adaptive KL (desired_kl={agent_cfg.algorithm.desired_kl})", flush=True)
    print(f"  Entropy coef:     {agent_cfg.algorithm.entropy_coef}", flush=True)
    print(f"  Mini-batches:     {agent_cfg.algorithm.num_mini_batches}", flush=True)
    print(f"  Steps per env:    {agent_cfg.num_steps_per_env}", flush=True)
    print(f"  Network:          {agent_cfg.policy.actor_hidden_dims}", flush=True)
    print(f"  Episode length:   {env_cfg.episode_length_s}s", flush=True)
    print(f"  Terrains:         12 types (ROBUST_TERRAINS_CFG)", flush=True)
    print(f"  Rewards:          18 terms (14 base + 4 custom)", flush=True)
    print(f"  Log dir:          {log_dir}", flush=True)
    steps_per_iter = env_cfg.scene.num_envs * agent_cfg.num_steps_per_env
    print(f"  Steps/iteration:  {steps_per_iter:,}", flush=True)
    print(f"  Est. total steps: {steps_per_iter * agent_cfg.max_iterations / 1e9:.1f}B", flush=True)
    print(flush=True)

    # DR schedule info
    print(f"  Progressive DR Schedule (over {args_cli.dr_expansion_iters} iterations):", flush=True)
    print(f"    Friction:  [{DR_SCHEDULE['static_friction_min'][0]}, {DR_SCHEDULE['static_friction_max'][0]}]"
          f" -> [{DR_SCHEDULE['static_friction_min'][1]}, {DR_SCHEDULE['static_friction_max'][1]}]", flush=True)
    print(f"    Push vel:  +/-{DR_SCHEDULE['push_velocity'][0]}"
          f" -> +/-{DR_SCHEDULE['push_velocity'][1]} m/s", flush=True)
    print(f"    Ext force: +/-{DR_SCHEDULE['ext_force'][0]}"
          f" -> +/-{DR_SCHEDULE['ext_force'][1]} N", flush=True)
    print(f"    Mass:      +/-{DR_SCHEDULE['mass_offset'][0]}"
          f" -> +/-{DR_SCHEDULE['mass_offset'][1]} kg", flush=True)
    print(f"{'='*70}\n", flush=True)

    # --- Create environment ---
    gym.register(
        id="Isaac-Velocity-Finetune-Spot-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{SpotFinetuneEnvCfg.__module__}:{SpotFinetuneEnvCfg.__name__}",
        },
    )
    env = gym.make("Isaac-Velocity-Finetune-Spot-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # --- Create runner ---
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )

    # --- Load 48hr rough policy checkpoint ---
    if args_cli.checkpoint:
        print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}", flush=True)
        runner.load(args_cli.checkpoint, load_optimizer=False)
        print(f"[INFO] Checkpoint loaded (optimizer reset for fine-tuning LR={agent_cfg.algorithm.learning_rate})", flush=True)
    else:
        print("[WARN] No checkpoint specified — training from scratch!", flush=True)
        print("[WARN] This is NOT recommended. Use --checkpoint /path/to/model_27500.pt", flush=True)

    # --- Save config ---
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Save training parameters
    with open(os.path.join(log_dir, "params", "training_params.txt"), "w") as f:
        f.write(f"stage: 1 (progressive fine-tuning)\n")
        f.write(f"checkpoint: {args_cli.checkpoint}\n")
        f.write(f"num_envs: {env_cfg.scene.num_envs}\n")
        f.write(f"max_iterations: {agent_cfg.max_iterations}\n")
        f.write(f"dr_expansion_iters: {args_cli.dr_expansion_iters}\n")
        f.write(f"learning_rate: {agent_cfg.algorithm.learning_rate}\n")
        f.write(f"seed: {args_cli.seed}\n")
        f.write(f"steps_per_iter: {steps_per_iter}\n")
        f.write(f"network: {agent_cfg.policy.actor_hidden_dims}\n")
        f.write(f"episode_length_s: {env_cfg.episode_length_s}\n")
        f.write(f"dr_schedule: {DR_SCHEDULE}\n")

    # --- Monkey-patch alg.update() for progressive DR ---
    print("\n[TRAIN] Starting training with progressive DR...", flush=True)
    start_time = time.time()

    original_update = runner.alg.update
    _iteration_counter = [0]
    _dr_log_interval = 500

    def update_with_progressive_dr(*args, **kwargs):
        """Wrapper that expands DR parameters before each PPO update."""
        it = _iteration_counter[0]

        # Update DR parameters
        dr_info = update_dr_params(env, it, args_cli.dr_expansion_iters)

        # Log DR progress periodically
        if it % _dr_log_interval == 0:
            elapsed = time.time() - start_time
            hours = elapsed / 3600
            fps = it * steps_per_iter / max(elapsed, 1) if it > 0 else 0
            print(
                f"[TRAIN] iter={it:6d}/{agent_cfg.max_iterations}  "
                f"dr={dr_info['dr_fraction']:.1%}  "
                f"friction={dr_info['friction_range']}  "
                f"push={dr_info['push_vel']:.2f}  "
                f"force={dr_info['ext_force']:.1f}  "
                f"elapsed={hours:.1f}h  fps={fps:.0f}",
                flush=True,
            )

        _iteration_counter[0] += 1
        return original_update(*args, **kwargs)

    runner.alg.update = update_with_progressive_dr

    # Run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    # --- Training complete ---
    elapsed = time.time() - start_time
    hours = elapsed / 3600
    total_steps = agent_cfg.max_iterations * steps_per_iter

    print(f"\n{'='*70}", flush=True)
    print(f"  STAGE 1 TRAINING COMPLETE", flush=True)
    print(f"  {'='*66}", flush=True)
    print(f"  Total time:       {hours:.1f} hours ({elapsed:.0f} seconds)", flush=True)
    print(f"  Total steps:      {total_steps / 1e9:.1f}B", flush=True)
    print(f"  Avg throughput:   {total_steps / elapsed:.0f} steps/sec", flush=True)
    print(f"  Checkpoints:      {log_dir}", flush=True)
    print(f"  Final DR:         friction={DR_SCHEDULE['static_friction_min'][1]:.2f}-"
          f"{DR_SCHEDULE['static_friction_max'][1]:.2f}, "
          f"push={DR_SCHEDULE['push_velocity'][1]:.1f}, "
          f"force={DR_SCHEDULE['ext_force'][1]:.1f}", flush=True)
    print(f"{'='*70}\n", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
