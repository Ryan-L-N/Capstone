"""Phase 1: Unified PPO Training Script for Spot and Vision60.

Single script parameterized by --robot spot|vision60:
  - Loads robot-specific EnvCfg and PPOCfg
  - Cosine LR annealing with linear warmup
  - Progressive domain randomization (Vision60) or fixed DR (Spot)
  - Weights & Biases integration for experiment tracking
  - Custom per-reward-term W&B logging

Usage (H100 — Spot):
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/multi_robot_training/train_ppo.py --headless \\
        --robot spot --num_envs 20480

Usage (H100 — Vision60):
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/multi_robot_training/train_ppo.py --headless \\
        --robot vision60 --num_envs 20480

Usage (local debug):
    isaaclab.bat -p /path/to/train_ppo.py --headless \\
        --robot spot --num_envs 64 --max_iterations 10 --no_wandb

Created for AI2C Tech Capstone — MS for Autonomy, February 2026
"""

# ── 0. Parse args BEFORE any Isaac imports ──────────────────────────────
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Multi-robot PPO training (Phase 1)")
parser.add_argument("--robot", type=str, required=True, choices=["spot", "vision60"],
                    help="Robot to train: spot or vision60")
parser.add_argument("--num_envs", type=int, default=20480,
                    help="Number of parallel environments (default 20480 for H100)")
parser.add_argument("--max_iterations", type=int, default=60000,
                    help="Max training iterations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--resume", action="store_true", default=False,
                    help="Resume from latest checkpoint")
parser.add_argument("--load_run", type=str, default=None,
                    help="Run directory to resume from")
parser.add_argument("--load_checkpoint", type=str, default="model_.*\\.pt",
                    help="Checkpoint regex to load")
parser.add_argument("--lr_max", type=float, default=1e-3,
                    help="Max learning rate for cosine annealing")
parser.add_argument("--lr_min", type=float, default=1e-5,
                    help="Min learning rate for cosine annealing")
parser.add_argument("--warmup_iters", type=int, default=50,
                    help="LR warmup iterations (default 50; was 500 which caused value explosion in Trial 6)")
parser.add_argument("--dr_expansion_iters", type=int, default=15000,
                    help="Iterations over which DR expands (Vision60 only)")
parser.add_argument("--no_wandb", action="store_true", default=False,
                    help="Disable Weights & Biases logging")
parser.add_argument("--terrain", type=str, default="robust", choices=["robust", "robust_easy", "flat", "transition"],
                    help="Terrain type: 'robust' (12-type, full difficulty), 'robust_easy' (12-type, capped difficulty), 'flat' (100%% flat), or 'transition' (50%% flat + gentle rough)")
parser.add_argument("--min_noise_std", type=float, default=0.3,
                    help="Minimum noise std floor")
parser.add_argument("--max_noise_std", type=float, default=1.0,
                    help="Maximum noise std ceiling (was 2.0, reduced to prevent flip spiral)")
parser.add_argument("--save_interval", type=int, default=None,
                    help="Override checkpoint save interval (default: use config value)")
parser.add_argument("--num_learning_epochs", type=int, default=None,
                    help="Override PPO learning epochs per iteration (default: use config value, typically 8)")
parser.add_argument("--actor_only_resume", action="store_true", default=False,
                    help="Load only actor weights from checkpoint (leave critic fresh for reward weight changes)")
parser.add_argument("--critic_warmup_iters", type=int, default=0,
                    help="Freeze actor for N iters so critic can calibrate to new reward scale")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 1. Imports (AFTER SimulationApp) ────────────────────────────────────
import os
import sys
import time
from datetime import datetime

# Path setup so `from configs...` (this Loco_Policy_1) and `from quadruped_locomotion...`
# (Loco_Shared) both resolve cleanly.
_LOCO1_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ALEX_ROOT = os.path.abspath(os.path.join(_LOCO1_ROOT, ".."))
for _p in (
    _LOCO1_ROOT,
    os.path.join(_ALEX_ROOT, "Loco_Shared"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Trigger standard gym registrations
import isaaclab_tasks  # noqa: F401
import configs  # noqa: F401  — registers Loco_Policy_1 (ARL Baseline) gym envs
from isaaclab_tasks.utils import get_checkpoint_path

from quadruped_locomotion.utils.lr_schedule import cosine_annealing_lr, set_learning_rate
from quadruped_locomotion.utils.dr_schedule import update_dr_params, DR_SCHEDULE
from quadruped_locomotion.utils.training_utils import configure_tf32, clamp_noise_std, register_std_safety_clamp

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

# Enable TF32
configure_tf32()

# ── 2. Robot-Specific Config Loading ───────────────────────────────────

def load_robot_configs(robot: str):
    """Load env and agent configs for the specified robot.

    Returns:
        (env_cfg, agent_cfg, env_id) tuple.
    """
    if robot == "spot":
        from configs.env_cfg import SpotLocomotionEnvCfg
        from configs.agents.rsl_rl_ppo_cfg import SpotPPORunnerCfg
        return SpotLocomotionEnvCfg(), SpotPPORunnerCfg(), "Locomotion-Robust-Spot-v0"
    else:
        from quadruped_locomotion.tasks.locomotion.config.vision60.env_cfg import Vision60LocomotionEnvCfg
        from quadruped_locomotion.tasks.locomotion.config.vision60.agents.rsl_rl_ppo_cfg import Vision60PPORunnerCfg
        return Vision60LocomotionEnvCfg(), Vision60PPORunnerCfg(), "Locomotion-Robust-Vision60-v0"


# ── 3. Main ─────────────────────────────────────────────────────────────

def main():
    robot = args_cli.robot
    env_cfg, agent_cfg, env_id = load_robot_configs(robot)

    # Apply CLI overrides
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed
    if args_cli.save_interval is not None:
        agent_cfg.save_interval = args_cli.save_interval
    if args_cli.num_learning_epochs is not None:
        agent_cfg.algorithm.num_learning_epochs = args_cli.num_learning_epochs

    # Flat terrain override for warmup training (Bug #14 fix: learn to walk before rough terrain)
    if args_cli.terrain == "flat":
        print("[TERRAIN] Overriding to 100% FLAT terrain for warmup phase", flush=True)
        env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=True,
            sub_terrains={
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
            },
        )
    elif args_cli.terrain == "transition":
        print("[TERRAIN] Overriding to TRANSITION terrain (50% flat + gentle slopes/rough)", flush=True)
        env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=5,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=True,
            sub_terrains={
                # 50% flat — safe zone, policy can still practice walking
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.50),
                # 15% gentle slopes (max 0.25 rad ~14 deg, easy curriculum)
                "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.15,
                    slope_range=(0.0, 0.25),
                    platform_width=2.0,
                    border_width=0.25,
                ),
                # 10% slight random roughness (very mild bumps)
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.10,
                    noise_range=(0.01, 0.06),
                    noise_step=0.01,
                    border_width=0.25,
                ),
                # 10% gentle stairs (max 0.10m step — half of robust)
                "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.10,
                    step_height_range=(0.03, 0.10),
                    step_width=0.3,
                    platform_width=3.0,
                    border_width=1.0,
                    holes=False,
                ),
                # 10% wave terrain (gentle undulations)
                "wave": terrain_gen.HfWaveTerrainCfg(
                    proportion=0.10,
                    amplitude_range=(0.02, 0.08),
                    num_waves=2,
                    border_width=0.25,
                ),
                # 5% friction plane (for vegetation drag training)
                "vegetation_plane": terrain_gen.MeshPlaneTerrainCfg(
                    proportion=0.05,
                ),
            },
        )
    elif args_cli.terrain == "robust_easy":
        print("[TERRAIN] Overriding to ROBUST_EASY terrain (all 12 types, 3 difficulty rows)", flush=True)
        from quadruped_locomotion.tasks.locomotion.mdp.terrains import ROBUST_TERRAINS_CFG
        import copy
        robust_easy = copy.deepcopy(ROBUST_TERRAINS_CFG)
        robust_easy.num_rows = 3   # cap difficulty: only easy/medium rows
        robust_easy.num_cols = 20  # fewer columns for faster terrain gen
        env_cfg.scene.terrain.terrain_generator = robust_easy

    # Disable body_height_tracking on non-flat terrain (Bug #22: uses world-frame Z,
    # not height above terrain surface — penalizes robot for standing on top of stairs)
    if args_cli.terrain in ("robust", "robust_easy", "transition"):
        if hasattr(env_cfg.rewards, "body_height_tracking"):
            env_cfg.rewards.body_height_tracking.weight = 0.0
            print("[REWARDS] Disabled body_height_tracking (world-frame Z is meaningless on rough terrain)", flush=True)

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
    print(f"  MULTI-ROBOT PPO TRAINING — PHASE 1", flush=True)
    print(f"  {'='*66}", flush=True)
    print(f"  Robot:            {robot.upper()}", flush=True)
    print(f"  Envs:             {env_cfg.scene.num_envs}", flush=True)
    print(f"  Max iterations:   {agent_cfg.max_iterations}", flush=True)
    print(f"  Save interval:    {agent_cfg.save_interval}", flush=True)
    print(f"  LR schedule:      cosine {args_cli.lr_max} -> {args_cli.lr_min}", flush=True)
    print(f"  Warmup iters:     {args_cli.warmup_iters}", flush=True)
    print(f"  Entropy coef:     {agent_cfg.algorithm.entropy_coef}", flush=True)
    print(f"  Mini-batches:     {agent_cfg.algorithm.num_mini_batches}", flush=True)
    print(f"  Steps per env:    {agent_cfg.num_steps_per_env}", flush=True)
    print(f"  Network:          {agent_cfg.policy.actor_hidden_dims}", flush=True)
    print(f"  Episode length:   {env_cfg.episode_length_s}s", flush=True)
    print(f"  Terrain:          {args_cli.terrain.upper()}", flush=True)
    print(f"  Logger:           {agent_cfg.logger}", flush=True)
    print(f"  Log dir:          {log_dir}", flush=True)
    print(f"  Steps/iteration:  {steps_per_iter:,}", flush=True)
    print(f"  Est. total steps: {steps_per_iter * agent_cfg.max_iterations / 1e9:.1f}B", flush=True)

    if robot == "vision60":
        print(f"  DR expansion:     {args_cli.dr_expansion_iters} iterations", flush=True)
        print(f"  Progressive DR Schedule:", flush=True)
        print(f"    Friction:  [{DR_SCHEDULE['static_friction_min'][0]}, "
              f"{DR_SCHEDULE['static_friction_max'][0]}] -> "
              f"[{DR_SCHEDULE['static_friction_min'][1]}, "
              f"{DR_SCHEDULE['static_friction_max'][1]}]", flush=True)

    print(f"{'='*70}\n", flush=True)

    # ── Create environment ──────────────────────────────────────────────
    env = gym.make(env_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── Create runner ───────────────────────────────────────────────────
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )

    # ── Safety: clamp std inside PPO update mini-batches ────────────────
    # Prevents RuntimeError: normal expects all elements of std >= 0.0
    # when optimizer step pushes std negative between mini-batches.
    register_std_safety_clamp(runner.alg.policy, args_cli.min_noise_std, args_cli.max_noise_std)
    print(f"[SAFETY] Registered std clamp on policy.act(): [{args_cli.min_noise_std}, {args_cli.max_noise_std}]", flush=True)

    # ── Resume if requested ─────────────────────────────────────────────
    start_iteration = 0
    if args_cli.actor_only_resume:
        # Actor-only resume: load actor weights only, leave critic fresh.
        # Used when reward weights change (critic's value estimates are invalid).
        resume_path = get_checkpoint_path(
            log_root_path, args_cli.load_run, args_cli.load_checkpoint
        )
        print(f"[INFO] Actor-only resume from: {resume_path}", flush=True)
        loaded_dict = torch.load(resume_path, weights_only=False)
        full_state = loaded_dict["model_state_dict"]

        # Filter to actor.* and std keys only
        actor_keys = {k: v for k, v in full_state.items()
                      if k.startswith("actor.") or k in ("std", "log_std")}
        print(f"[INFO] Loading {len(actor_keys)}/{len(full_state)} keys (actor + std only)", flush=True)
        runner.alg.policy.load_state_dict(actor_keys, strict=False)
        skipped = len(full_state) - len(actor_keys)
        print(f"[INFO] Skipped {skipped} critic keys (freshly initialized)", flush=True)

        # Do NOT load optimizer — new reward scale needs fresh Adam moments
        # Do NOT set start_iteration — new training starts from 0
        print("[INFO] Critic + optimizer freshly initialized (reward weights changed)", flush=True)

    elif args_cli.resume:
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
        f.write(f"robot: {robot}\n")
        f.write(f"num_envs: {env_cfg.scene.num_envs}\n")
        f.write(f"max_iterations: {agent_cfg.max_iterations}\n")
        f.write(f"lr_max: {args_cli.lr_max}\n")
        f.write(f"lr_min: {args_cli.lr_min}\n")
        f.write(f"warmup_iters: {args_cli.warmup_iters}\n")
        f.write(f"seed: {args_cli.seed}\n")
        f.write(f"steps_per_iter: {steps_per_iter}\n")
        f.write(f"network: {agent_cfg.policy.actor_hidden_dims}\n")
        f.write(f"terrain: {args_cli.terrain}\n")
        f.write(f"logger: {agent_cfg.logger}\n")

    # ── W&B custom logging setup ────────────────────────────────────────
    wandb_run = None
    if agent_cfg.logger == "wandb":
        try:
            import wandb
            # RSL-RL OnPolicyRunner initializes W&B internally,
            # but we grab the active run for custom metric logging
            wandb_run = wandb.run
            if wandb_run is not None:
                print(f"[W&B] Active run: {wandb_run.name} ({wandb_run.url})", flush=True)
        except ImportError:
            print("[WARN] wandb not installed — falling back to TensorBoard", flush=True)

    # ── Training loop ───────────────────────────────────────────────────
    print(f"\n[TRAIN] Starting {robot.upper()} training with cosine annealing LR...",
          flush=True)
    start_time = time.time()

    # Set initial learning rate
    initial_lr = cosine_annealing_lr(
        start_iteration, agent_cfg.max_iterations,
        args_cli.lr_max, args_cli.lr_min, args_cli.warmup_iters
    )
    set_learning_rate(runner, initial_lr)
    print(f"[TRAIN] Initial LR: {initial_lr:.2e}", flush=True)

    # Monkey-patch the PPO update with LR schedule + DR
    original_update = runner.alg.update
    _lr_log_interval = 1000
    _iteration_counter = [start_iteration]

    # ── Critic warmup (actor-only resume) ─────────────────────────────────
    # When reward weights change, the critic is fresh and its value estimates
    # are garbage. Freezing the actor lets the critic calibrate to the new
    # reward scale without destroying the learned locomotion policy.
    _critic_warmup_remaining = [args_cli.critic_warmup_iters]
    _actor_frozen = [False]

    if args_cli.critic_warmup_iters > 0:
        # Freeze actor parameters
        for name, param in runner.alg.policy.named_parameters():
            if name.startswith("actor.") or name in ("std", "log_std"):
                param.requires_grad = False
        _actor_frozen[0] = True
        print(f"[WARMUP] Actor frozen for {args_cli.critic_warmup_iters} iters (critic calibration)", flush=True)

    # ── Value loss watchdog (Bug #25 fix) ────────────────────────────────
    # Detects value function loss spikes and temporarily halves the LR
    # to break the oscillation → amplification → NaN cascade.
    _VL_THRESHOLD = 100.0       # value loss above this triggers the guard
    _VL_COOLDOWN_ITERS = 50     # how many iters to keep reduced LR
    _vl_penalty = [1.0]         # mutable: current LR multiplier (1.0 = normal)
    _vl_cooldown = [0]          # mutable: remaining cooldown iters
    _vl_spike_count = [0]       # mutable: total spike count for logging

    def update_with_schedule(*args, **kwargs):
        """Wrapper: cosine LR + progressive DR (if Vision60) + noise clamp + value loss guard."""
        it = _iteration_counter[0]

        # Cosine LR annealing
        lr = cosine_annealing_lr(
            it, agent_cfg.max_iterations,
            args_cli.lr_max, args_cli.lr_min, args_cli.warmup_iters
        )

        # Apply value loss watchdog penalty if active
        if _vl_cooldown[0] > 0:
            lr *= _vl_penalty[0]
            _vl_cooldown[0] -= 1
            if _vl_cooldown[0] == 0:
                _vl_penalty[0] = 1.0
                print(f"[GUARD] Value loss cooldown expired, restoring scheduled LR", flush=True)

        set_learning_rate(runner, lr)

        # Progressive DR for Vision60
        dr_info = None
        if robot == "vision60":
            dr_info = update_dr_params(env, it, args_cli.dr_expansion_iters)

        # Run the actual PPO update
        result = original_update(*args, **kwargs)

        # ── Critic warmup countdown ─────────────────────────────────
        if _actor_frozen[0] and _critic_warmup_remaining[0] > 0:
            _critic_warmup_remaining[0] -= 1
            if _critic_warmup_remaining[0] == 0:
                # Unfreeze actor parameters
                for name, param in runner.alg.policy.named_parameters():
                    if name.startswith("actor.") or name in ("std", "log_std"):
                        param.requires_grad = True
                _actor_frozen[0] = False
                print(f"[WARMUP] Critic warmup complete at iter {it}. Actor unfrozen.", flush=True)

        # ── Value loss watchdog check ────────────────────────────────
        vl = result.get("value_function", 0.0) if isinstance(result, dict) else 0.0
        if vl > _VL_THRESHOLD:
            _vl_penalty[0] = 0.5
            _vl_cooldown[0] = _VL_COOLDOWN_ITERS
            _vl_spike_count[0] += 1
            print(
                f"[GUARD] Value loss spike: {vl:.1f} > {_VL_THRESHOLD} at iter {it} "
                f"(spike #{_vl_spike_count[0]}). Halving LR for {_VL_COOLDOWN_ITERS} iters.",
                flush=True,
            )

        # Safety: clamp noise std
        clamp_noise_std(runner.alg.policy, args_cli.min_noise_std, args_cli.max_noise_std)

        # Custom W&B logging
        if wandb_run is not None and it % 100 == 0:
            import wandb
            log_data = {"lr": lr}
            if dr_info is not None:
                log_data["dr/fraction"] = dr_info["dr_fraction"]
                log_data["dr/push_vel"] = dr_info["push_vel"]
                log_data["dr/ext_force"] = dr_info["ext_force"]
            wandb.log(log_data, step=it)

        # Periodic console logging
        if it % _lr_log_interval == 0:
            elapsed = time.time() - start_time
            hours = elapsed / 3600
            fps = (it - start_iteration) * steps_per_iter / max(elapsed, 1) if it > start_iteration else 0
            noise = runner.alg.policy.std.mean().item() if hasattr(runner.alg.policy, 'std') else 0

            dr_str = ""
            if dr_info is not None:
                dr_str = f"  dr={dr_info['dr_fraction']:.1%}"

            guard_str = ""
            if _vl_cooldown[0] > 0:
                guard_str = f"  [GUARD active: LR×{_vl_penalty[0]}, {_vl_cooldown[0]} iters left]"

            warmup_str = ""
            if _actor_frozen[0]:
                warmup_str = f"  [WARMUP: actor frozen, {_critic_warmup_remaining[0]} iters left]"

            print(
                f"[TRAIN] iter={it:6d}/{agent_cfg.max_iterations}  "
                f"lr={lr:.2e}{dr_str}  "
                f"noise={noise:.3f}  "
                f"elapsed={hours:.1f}h  fps={fps:.0f}"
                f"{guard_str}{warmup_str}",
                flush=True,
            )

        _iteration_counter[0] += 1
        return result

    runner.alg.update = update_with_schedule

    # Run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations - start_iteration,
        init_at_random_ep_len=True,
    )

    # ── Training complete ───────────────────────────────────────────────
    elapsed = time.time() - start_time
    hours = elapsed / 3600
    total_steps = (agent_cfg.max_iterations - start_iteration) * steps_per_iter

    print(f"\n{'='*70}", flush=True)
    print(f"  {robot.upper()} TRAINING COMPLETE", flush=True)
    print(f"  {'='*66}", flush=True)
    print(f"  Total time:       {hours:.1f} hours ({elapsed:.0f} seconds)", flush=True)
    print(f"  Total steps:      {total_steps / 1e9:.1f}B", flush=True)
    print(f"  Avg throughput:   {total_steps / elapsed:.0f} steps/sec", flush=True)
    print(f"  Checkpoints:      {log_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    # Use os._exit(0) to avoid CUDA deadlock on close
    import os as _os
    _os._exit(0)
