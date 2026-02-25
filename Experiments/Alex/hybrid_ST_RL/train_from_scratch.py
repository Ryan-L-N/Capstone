"""
Attempt 5: Train From Scratch with 235-dim Obs + Terrain Curriculum
====================================================================

Trains a fresh policy from random initialization on flat terrain with
terrain curriculum to gradually introduce harder terrains (boulders,
stairs, uneven ground, friction, vegetation drag).

No checkpoint, no freeze/unfreeze, no LR warmup — actor and critic
grow up together from iteration 0. This eliminates the catastrophic
forgetting that killed Attempts 1-4.

Key features:
  - 235-dim observations (48 proprio + 187 height scan) from day one
  - Terrain curriculum: all robots start flat, auto-promote as they learn
  - Progressive DR: friction/push/force expand over 15K iterations
  - Standard PPO hyperparameters (LR 1e-3, clip 0.2, entropy 0.005)
  - [512, 256, 128] architecture (compatible with Stage 2 distillation)

Usage (H100):
    cd ~/IsaacLab
    ./isaaclab.sh -p /path/to/train_from_scratch.py --headless --num_envs 16384

Usage (local debug — RTX 2000 Ada):
    cd C:\\IsaacLab
    isaaclab.bat -p /path/to/train_from_scratch.py --headless --num_envs 64 \\
        --max_iterations 10 --dr_expansion_iters 5

Created for AI2C Tech Capstone — hybrid_ST_RL Attempt 5, February 2026
"""

# ── 0. Parse args BEFORE any Isaac imports ──────────────────────────────
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Attempt 5: Train from scratch with terrain curriculum")
parser.add_argument("--num_envs", type=int, default=16384,
                    help="Number of parallel environments (default 16384 for H100)")
parser.add_argument("--max_iterations", type=int, default=15000,
                    help="Max training iterations (default 15000)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--dr_expansion_iters", type=int, default=15000,
                    help="Iterations over which DR expands from easy to hard (default 15000)")
parser.add_argument("--min_noise_std", type=float, default=0.3,
                    help="Minimum noise std floor (default 0.3)")
parser.add_argument("--max_noise_std", type=float, default=2.0,
                    help="Maximum noise std ceiling (default 2.0)")
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

# Trigger standard gym registrations
import isaaclab_tasks  # noqa: F401

# Add our project to path for custom configs
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from configs.scratch_env_cfg import SpotScratchEnvCfg
from configs.scratch_ppo_cfg import SpotScratchPPORunnerCfg

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

    Returns dict of current DR values for logging.
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


# ── 3. Noise Std Safety Clamp ───────────────────────────────────────────

def clamp_noise_std(policy, min_std: float, max_std: float):
    """Clamp noise std as a safety net — prevent collapse or explosion."""
    with torch.no_grad():
        if hasattr(policy, 'noise_std_type') and policy.noise_std_type == "log":
            log_min = torch.log(torch.tensor(min_std, device=policy.log_std.device))
            log_max = torch.log(torch.tensor(max_std, device=policy.log_std.device))
            policy.log_std.clamp_(min=log_min.item(), max=log_max.item())
        else:
            policy.std.clamp_(min=min_std, max=max_std)


# ── 4. Main ─────────────────────────────────────────────────────────────

def main():
    # --- Environment config ---
    env_cfg = SpotScratchEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # --- Agent config ---
    agent_cfg = SpotScratchPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed

    # --- Logging ---
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    steps_per_iter = env_cfg.scene.num_envs * agent_cfg.num_steps_per_env

    print(f"\n{'='*70}", flush=True)
    print(f"  ATTEMPT 6: TRAIN FROM SCRATCH + TERRAIN CURRICULUM", flush=True)
    print(f"  {'='*66}", flush=True)
    print(f"  Mode:             FROM SCRATCH (no checkpoint)", flush=True)
    print(f"  Envs:             {env_cfg.scene.num_envs}", flush=True)
    print(f"  Max iterations:   {agent_cfg.max_iterations}", flush=True)
    print(f"  DR expansion:     {args_cli.dr_expansion_iters} iterations", flush=True)
    print(f"  Save interval:    {agent_cfg.save_interval}", flush=True)
    print(f"  Learning rate:    {agent_cfg.algorithm.learning_rate}", flush=True)
    print(f"  LR schedule:      adaptive KL (desired_kl={agent_cfg.algorithm.desired_kl})", flush=True)
    print(f"  Entropy coef:     {agent_cfg.algorithm.entropy_coef}", flush=True)
    print(f"  Clip param:       {agent_cfg.algorithm.clip_param}", flush=True)
    print(f"  Mini-batches:     {agent_cfg.algorithm.num_mini_batches}", flush=True)
    print(f"  Learning epochs:  {agent_cfg.algorithm.num_learning_epochs}", flush=True)
    print(f"  Steps per env:    {agent_cfg.num_steps_per_env}", flush=True)
    print(f"  Network:          {agent_cfg.policy.actor_hidden_dims}", flush=True)
    print(f"  Init noise std:   {agent_cfg.policy.init_noise_std}", flush=True)
    print(f"  Noise std range:  [{args_cli.min_noise_std}, {args_cli.max_noise_std}]", flush=True)
    print(f"  Episode length:   {env_cfg.episode_length_s}s", flush=True)
    print(f"  Terrains:         7 types (SCRATCH_TERRAINS_CFG, flat start)", flush=True)
    print(f"  Rewards:          19 terms (14 base + 5 custom)", flush=True)
    print(f"  Log dir:          {log_dir}", flush=True)
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
        id="Isaac-Velocity-Scratch-Spot-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{SpotScratchEnvCfg.__module__}:{SpotScratchEnvCfg.__name__}",
        },
    )
    env = gym.make("Isaac-Velocity-Scratch-Spot-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # --- Create runner (from random initialization — no checkpoint) ---
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )
    print(f"[INFO] Training from RANDOM INITIALIZATION (no checkpoint)", flush=True)
    print(f"[INFO] Actor and critic train together from iteration 0", flush=True)

    # --- Save config ---
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    with open(os.path.join(log_dir, "params", "training_params.txt"), "w") as f:
        f.write(f"attempt: 5 (from scratch + terrain curriculum)\n")
        f.write(f"mode: from_scratch (no checkpoint, no freeze/unfreeze)\n")
        f.write(f"num_envs: {env_cfg.scene.num_envs}\n")
        f.write(f"max_iterations: {agent_cfg.max_iterations}\n")
        f.write(f"dr_expansion_iters: {args_cli.dr_expansion_iters}\n")
        f.write(f"learning_rate: {agent_cfg.algorithm.learning_rate}\n")
        f.write(f"clip_param: {agent_cfg.algorithm.clip_param}\n")
        f.write(f"entropy_coef: {agent_cfg.algorithm.entropy_coef}\n")
        f.write(f"num_learning_epochs: {agent_cfg.algorithm.num_learning_epochs}\n")
        f.write(f"desired_kl: {agent_cfg.algorithm.desired_kl}\n")
        f.write(f"init_noise_std: {agent_cfg.policy.init_noise_std}\n")
        f.write(f"seed: {args_cli.seed}\n")
        f.write(f"steps_per_iter: {steps_per_iter}\n")
        f.write(f"network: {agent_cfg.policy.actor_hidden_dims}\n")
        f.write(f"episode_length_s: {env_cfg.episode_length_s}\n")
        f.write(f"terrain: SCRATCH_TERRAINS_CFG (7 types, flat start, curriculum)\n")
        f.write(f"rewards: 19 terms (14 base + 5 custom)\n")
        f.write(f"dr_schedule: {DR_SCHEDULE}\n")

    # --- Monkey-patch for progressive DR + noise clamp ---
    print("\n[TRAIN] Starting from-scratch training with terrain curriculum...", flush=True)
    print(f"[TRAIN] All robots start on flat terrain (level 0)", flush=True)
    print(f"[TRAIN] Curriculum auto-promotes as robot learns to walk", flush=True)
    start_time = time.time()

    original_update = runner.alg.update
    _iteration_counter = [0]
    _log_interval = 500

    def update_with_dr(*args, **kwargs):
        """Wrapper: progressive DR + noise clamp (no freeze/unfreeze needed)."""
        it = _iteration_counter[0]

        # --- Progressive DR ---
        dr_info = update_dr_params(env, it, args_cli.dr_expansion_iters)

        # --- Run the actual PPO update ---
        result = original_update(*args, **kwargs)

        # --- Safety: clamp noise std ---
        clamp_noise_std(runner.alg.policy, args_cli.min_noise_std, args_cli.max_noise_std)

        # Log progress periodically
        if it % _log_interval == 0:
            elapsed = time.time() - start_time
            hours = elapsed / 3600
            fps = it * steps_per_iter / max(elapsed, 1) if it > 0 else 0
            noise = runner.alg.policy.std.mean().item() if hasattr(runner.alg.policy, 'std') else 0
            current_lr = runner.alg.optimizer.param_groups[0]["lr"]
            print(
                f"[TRAIN] iter={it:6d}/{agent_cfg.max_iterations}  "
                f"dr={dr_info['dr_fraction']:.1%}  "
                f"friction={dr_info['friction_range']}  "
                f"push={dr_info['push_vel']:.2f}  "
                f"noise={noise:.3f}  "
                f"lr={current_lr:.1e}  "
                f"elapsed={hours:.1f}h  fps={fps:.0f}",
                flush=True,
            )

        _iteration_counter[0] += 1
        return result

    runner.alg.update = update_with_dr

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
    print(f"  ATTEMPT 6 TRAINING COMPLETE", flush=True)
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
