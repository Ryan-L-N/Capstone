"""Unified S2R Expert Training Script — one script, six terrain experts.

Trains a single terrain-specialist expert from scratch with full S2R hardening
(action delay, sensor noise, torque limits, push forces, wider DR).

Usage (H100):
    cd ~/SIM_TO_REAL
    python scripts/train_expert.py --expert_type stairs_up \
        --headless --no_wandb --num_envs 4096 \
        --max_iterations 10000 --save_interval 100 --max_noise_std 0.5

Usage (local debug):
    python scripts/train_expert.py --expert_type friction \
        --headless --num_envs 64 --max_iterations 10 --no_wandb --max_noise_std 0.5

Expert types: friction, stairs_up, stairs_down, boulders, slopes, mixed_rough

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

# -- 0. Parse args BEFORE any Isaac imports --------------------------------
import argparse
import sys

from isaaclab.app import AppLauncher

EXPERT_TYPES = ["friction", "stairs_up", "stairs_down", "boulders", "slopes", "mixed_rough"]

parser = argparse.ArgumentParser(description="S2R terrain expert training")
parser.add_argument("--expert_type", type=str, required=True, choices=EXPERT_TYPES,
                    help="Which terrain expert to train")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=10000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--no_wandb", action="store_true", default=False)

# LR schedule
parser.add_argument("--lr_max", type=float, default=1e-3)
parser.add_argument("--lr_min", type=float, default=1e-5)
parser.add_argument("--warmup_iters", type=int, default=50)

# Noise
parser.add_argument("--min_noise_std", type=float, default=0.3)
parser.add_argument("--max_noise_std", type=float, default=0.5,
                    help="ALWAYS pass explicitly (Bug #28d: default is 1.0 in train_ppo.py)")

# S2R wrapper params
parser.add_argument("--action_delay_steps", type=int, default=2,
                    help="Action delay ring buffer steps (2 @ 50Hz = 40ms)")
parser.add_argument("--obs_delay_steps", type=int, default=1,
                    help="Observation delay steps (1 @ 50Hz = 20ms)")
parser.add_argument("--sensor_dropout_rate", type=float, default=0.05,
                    help="Height scan ray dropout rate (0.05 = 5%%)")
parser.add_argument("--sensor_drift_rate", type=float, default=0.002,
                    help="IMU OU-process drift rate")

# Resume
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--load_checkpoint", type=str, default="model_.*\\.pt")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- 1. Post-launch imports ------------------------------------------------
import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401

# Add multi_robot_training to path for reward/terrain imports
_MRT_SRC = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "multi_robot_training",
    "source", "quadruped_locomotion",
))
if _MRT_SRC not in sys.path:
    sys.path.insert(0, _MRT_SRC)

import quadruped_locomotion  # noqa: F401

# S2R imports
_S2R_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _S2R_ROOT not in sys.path:
    sys.path.insert(0, _S2R_ROOT)

from quadruped_locomotion.utils.lr_schedule import cosine_annealing_lr, set_learning_rate
from quadruped_locomotion.utils.training_utils import configure_tf32, clamp_noise_std, register_std_safety_clamp

from wrappers import ActionDelayWrapper, ObservationDelayWrapper, SensorNoiseWrapper


# -- 2. Expert config dispatch ---------------------------------------------

def load_expert_config(expert_type: str):
    """Load the environment config for the specified expert type."""
    if expert_type == "friction":
        from configs.expert_friction_cfg import SpotFrictionExpertEnvCfg
        return SpotFrictionExpertEnvCfg()
    elif expert_type == "stairs_up":
        from configs.expert_stairs_up_cfg import SpotStairsUpExpertEnvCfg
        return SpotStairsUpExpertEnvCfg()
    elif expert_type == "stairs_down":
        from configs.expert_stairs_down_cfg import SpotStairsDownExpertEnvCfg
        return SpotStairsDownExpertEnvCfg()
    elif expert_type == "boulders":
        from configs.expert_boulders_cfg import SpotBouldersExpertEnvCfg
        return SpotBouldersExpertEnvCfg()
    elif expert_type == "slopes":
        from configs.expert_slopes_cfg import SpotSlopesExpertEnvCfg
        return SpotSlopesExpertEnvCfg()
    elif expert_type == "mixed_rough":
        from configs.expert_mixed_rough_cfg import SpotMixedRoughExpertEnvCfg
        return SpotMixedRoughExpertEnvCfg()
    else:
        raise ValueError(f"Unknown expert type: {expert_type}")


# -- 3. Main training function --------------------------------------------

def main():
    expert_type = args_cli.expert_type
    print(f"\n{'='*70}", flush=True)
    print(f"  S2R EXPERT TRAINING: {expert_type.upper()}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # TF32 for H100
    configure_tf32()

    # Load expert-specific env config
    env_cfg = load_expert_config(expert_type)
    env_cfg.scene.num_envs = args_cli.num_envs

    # Load agent config
    from configs.agent_cfg import SpotS2RExpertPPORunnerCfg
    agent_cfg = SpotS2RExpertPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.save_interval = args_cli.save_interval
    agent_cfg.seed = args_cli.seed
    agent_cfg.experiment_name = f"spot_s2r_{expert_type}"

    # Logger
    if args_cli.no_wandb:
        agent_cfg.logger = "tensorboard"

    # Steps per iteration for logging
    steps_per_iter = env_cfg.scene.num_envs * agent_cfg.num_steps_per_env

    # -- Create environment -------------------------------------------------
    print(f"[ENV] Creating {expert_type} environment with {env_cfg.scene.num_envs} envs...", flush=True)

    # Register the expert env dynamically
    gym_env_id = f"S2R-{expert_type}-Spot-v0"
    gym.register(
        id=gym_env_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"cfg": env_cfg},
        disable_env_checker=True,
    )

    env = gym.make(gym_env_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=True)

    # -- Apply S2R wrappers (stacking order matters) ------------------------
    print(f"[S2R] Applying wrappers: ObsDelay({args_cli.obs_delay_steps}), "
          f"ActionDelay({args_cli.action_delay_steps}), "
          f"SensorNoise(dropout={args_cli.sensor_dropout_rate})", flush=True)

    env = ObservationDelayWrapper(env, delay_steps=args_cli.obs_delay_steps)
    env = ActionDelayWrapper(env, delay_steps=args_cli.action_delay_steps)
    env = SensorNoiseWrapper(
        env,
        dropout_rate=args_cli.sensor_dropout_rate,
        drift_rate=args_cli.sensor_drift_rate,
    )

    # -- Create runner ------------------------------------------------------
    log_root_path = os.path.join(
        os.path.dirname(__file__), "..", "logs", "rsl_rl", agent_cfg.experiment_name
    )
    log_root_path = os.path.abspath(log_root_path)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.set_log_dir(log_root_path)
    log_dir = runner.log_dir

    # Register NaN safety clamp (Bug #24)
    register_std_safety_clamp(runner.alg.policy, args_cli.min_noise_std, args_cli.max_noise_std)

    # -- Resume handling ----------------------------------------------------
    start_iteration = 0
    if args_cli.resume:
        from isaaclab_tasks.utils import get_checkpoint_path
        resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, args_cli.load_checkpoint)
        print(f"[INFO] Resuming from: {resume_path}", flush=True)
        runner.load(resume_path)
        try:
            start_iteration = int(os.path.basename(resume_path).split("_")[1].split(".")[0]) + 1
        except (IndexError, ValueError):
            pass

    # -- Save config --------------------------------------------------------
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    with open(os.path.join(log_dir, "params", "s2r_params.txt"), "w") as f:
        f.write(f"expert_type: {expert_type}\n")
        f.write(f"num_envs: {env_cfg.scene.num_envs}\n")
        f.write(f"max_iterations: {agent_cfg.max_iterations}\n")
        f.write(f"lr_max: {args_cli.lr_max}\n")
        f.write(f"lr_min: {args_cli.lr_min}\n")
        f.write(f"max_noise_std: {args_cli.max_noise_std}\n")
        f.write(f"action_delay_steps: {args_cli.action_delay_steps}\n")
        f.write(f"obs_delay_steps: {args_cli.obs_delay_steps}\n")
        f.write(f"sensor_dropout_rate: {args_cli.sensor_dropout_rate}\n")
        f.write(f"sensor_drift_rate: {args_cli.sensor_drift_rate}\n")
        f.write(f"network: {agent_cfg.policy.actor_hidden_dims}\n")

    # -- Training loop with LR schedule + watchdog --------------------------
    print(f"[TRAIN] Starting {expert_type} training from scratch...", flush=True)
    start_time = time.time()

    original_update = runner.alg.update
    _iteration = [start_iteration]
    _VL_THRESHOLD = 100.0
    _VL_COOLDOWN_ITERS = 50
    _vl_penalty = [1.0]
    _vl_cooldown = [0]

    def update_with_schedule(*args, **kwargs):
        it = _iteration[0]

        # Cosine LR annealing
        lr = cosine_annealing_lr(
            it, agent_cfg.max_iterations,
            args_cli.lr_max, args_cli.lr_min, args_cli.warmup_iters
        )

        # Value loss watchdog penalty
        if _vl_cooldown[0] > 0:
            lr *= _vl_penalty[0]
            _vl_cooldown[0] -= 1
            if _vl_cooldown[0] == 0:
                _vl_penalty[0] = 1.0

        set_learning_rate(runner, lr)

        # Run PPO update
        result = original_update(*args, **kwargs)

        # Value loss watchdog check
        vl = result.get("value_function", 0.0) if isinstance(result, dict) else 0.0
        if vl > _VL_THRESHOLD:
            _vl_penalty[0] = 0.5
            _vl_cooldown[0] = _VL_COOLDOWN_ITERS
            print(f"[GUARD] Value loss spike: {vl:.1f} at iter {it}. Halving LR.", flush=True)

        # Noise clamp
        clamp_noise_std(runner.alg.policy, args_cli.min_noise_std, args_cli.max_noise_std)

        # Periodic logging
        if it % 500 == 0:
            elapsed = time.time() - start_time
            hours = elapsed / 3600
            fps = (it - start_iteration) * steps_per_iter / max(elapsed, 1) if it > start_iteration else 0
            noise = runner.alg.policy.std.mean().item() if hasattr(runner.alg.policy, 'std') else 0
            print(
                f"[TRAIN] {expert_type} iter={it:6d}/{agent_cfg.max_iterations}  "
                f"lr={lr:.2e}  noise={noise:.3f}  elapsed={hours:.1f}h  fps={fps:.0f}",
                flush=True,
            )

        _iteration[0] += 1
        return result

    runner.alg.update = update_with_schedule

    # Run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations - start_iteration,
        init_at_random_ep_len=True,
    )

    # -- Training complete --------------------------------------------------
    elapsed = time.time() - start_time
    total_steps = (agent_cfg.max_iterations - start_iteration) * steps_per_iter

    print(f"\n{'='*70}", flush=True)
    print(f"  {expert_type.upper()} EXPERT TRAINING COMPLETE", flush=True)
    print(f"  Time: {elapsed/3600:.1f}h  Steps: {total_steps/1e9:.2f}B", flush=True)
    print(f"  Logs: {log_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)
