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

EXPERT_TYPES = ["friction", "stairs_up", "stairs_down", "boulders", "slopes", "mixed_rough", "obstacle_parkour"]

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

# Resume / fine-tune from checkpoint
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--load_checkpoint", type=str, default="model_.*\\.pt")
parser.add_argument("--resume_path", type=str, default=None,
                    help="Direct path to checkpoint for full resume (actor + critic + optimizer)")
parser.add_argument("--actor_only_resume", type=str, default=None,
                    help="Path to checkpoint for actor-only resume (fresh critic + optimizer)")
parser.add_argument("--critic_warmup_iters", type=int, default=300,
                    help="Freeze actor for N iters while critic calibrates to new rewards")

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

# Add multi_robot_training to path (pip-installed on H100, fallback paths for local)
for _p in [
    os.path.join(os.path.dirname(__file__), "..", "..", "multi_robot_training",
                 "source", "quadruped_locomotion"),
    os.path.join(os.path.dirname(__file__), "..", "..", "multi_robot_training",
                 "multi_robot_training", "source", "quadruped_locomotion"),
    os.path.expanduser("~/multi_robot_training_new/source/quadruped_locomotion"),
]:
    _p = os.path.abspath(_p)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import quadruped_locomotion  # noqa: F401

# S2R imports
_S2R_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _S2R_ROOT not in sys.path:
    sys.path.insert(0, _S2R_ROOT)

from quadruped_locomotion.utils.lr_schedule import cosine_annealing_lr, set_learning_rate
from quadruped_locomotion.utils.training_utils import configure_tf32, clamp_noise_std, register_std_safety_clamp

from wrappers.progressive_s2r import ProgressiveS2RWrapper
from control_panel.hot_reload import HotReloader


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
    elif expert_type == "obstacle_parkour":
        from configs.expert_obstacle_parkour_cfg import SpotObstacleParkourExpertEnvCfg
        return SpotObstacleParkourExpertEnvCfg()
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
    try:
        env = RslRlVecEnvWrapper(env, clip_actions=True)
    except (TypeError, ValueError):
        env = RslRlVecEnvWrapper(env)

    # -- Apply Progressive S2R wrapper (scales with terrain level) -----------
    s2r_wrapper = ProgressiveS2RWrapper(
        env,
        max_action_delay_steps=args_cli.action_delay_steps,
        max_obs_delay_steps=args_cli.obs_delay_steps,
        max_dropout_rate=args_cli.sensor_dropout_rate,
        max_drift_rate=args_cli.sensor_drift_rate,
        s2r_start_terrain=0.2,   # Start S2R at terrain row ~2
        s2r_full_terrain=0.6,    # Full S2R by terrain row ~6
    )
    env = s2r_wrapper
    print(f"[S2R] Progressive wrapper: delay 0→{args_cli.action_delay_steps} steps, "
          f"dropout 0→{args_cli.sensor_dropout_rate:.0%}, "
          f"ramp terrain 0.2→0.6", flush=True)

    # -- Create runner ------------------------------------------------------
    log_root_path = os.path.join(
        os.path.dirname(__file__), "..", "logs", "rsl_rl", agent_cfg.experiment_name
    )
    log_root_path = os.path.abspath(log_root_path)

    os.makedirs(log_root_path, exist_ok=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_root_path, device="cuda:0")
    log_dir = runner.log_dir if hasattr(runner, 'log_dir') else log_root_path

    # Register NaN safety clamp (Bug #24)
    register_std_safety_clamp(runner.alg.policy, args_cli.min_noise_std, args_cli.max_noise_std)

    # -- Actor-only resume from base checkpoint (Bug #30 pattern) ------------
    start_iteration = 0
    _actor_frozen = [False]
    _warmup_remaining = [0]

    if args_cli.resume_path:
        # Direct full resume from explicit path (no get_checkpoint_path lookup)
        ckpt_path = args_cli.resume_path
        print(f"[RESUME] Full resume from: {ckpt_path}", flush=True)
        runner.load(ckpt_path)
        try:
            start_iteration = int(os.path.basename(ckpt_path).split("_")[1].split(".")[0]) + 1
        except (IndexError, ValueError):
            pass
        print(f"[RESUME] Continuing from iteration {start_iteration}", flush=True)

    elif args_cli.actor_only_resume:
        ckpt_path = args_cli.actor_only_resume
        print(f"[RESUME] Actor-only resume from: {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location="cuda:0", weights_only=False)
        full_state = ckpt.get("model_state_dict", ckpt)

        # Load ONLY actor weights + std (leave critic fresh)
        policy = runner.alg.policy
        policy_state = policy.state_dict()
        actor_keys = {k: v for k, v in full_state.items()
                      if k.startswith("actor.") or k in ("std", "log_std")}
        policy_state.update(actor_keys)
        policy.load_state_dict(policy_state)
        print(f"[RESUME] Loaded {len(actor_keys)} actor keys, critic freshly initialized", flush=True)

        # Gradual actor unfreeze — critic warmup then layer-by-layer actor unfreeze
        # Phase 1 (0 to warmup_iters): ALL actor frozen, critic calibrates
        # Phase 2 (warmup+1 to warmup+200): Unfreeze last actor layer (128->12)
        # Phase 3 (warmup+201 to warmup+400): Unfreeze middle layer (256->128)
        # Phase 4 (warmup+401+): Unfreeze all layers (full fine-tuning)
        # This prevents the first gradient update from destroying the gait.
        if args_cli.critic_warmup_iters > 0:
            # Freeze ALL actor params initially
            for name, param in policy.named_parameters():
                if name.startswith("actor.") or name in ("std", "log_std"):
                    param.requires_grad = False
            _actor_frozen[0] = True
            _warmup_remaining[0] = args_cli.critic_warmup_iters
            print(f"[WARMUP] Phase 1: ALL actor frozen for {args_cli.critic_warmup_iters} iters", flush=True)
            print(f"[WARMUP] Phase 2: Last layer unfreezes at iter {args_cli.critic_warmup_iters}", flush=True)
            print(f"[WARMUP] Phase 3: Middle layer unfreezes at iter {args_cli.critic_warmup_iters + 200}", flush=True)
            print(f"[WARMUP] Phase 4: Full unfreeze at iter {args_cli.critic_warmup_iters + 400}", flush=True)

    elif args_cli.resume:
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

    # -- Live control panel (file-based hot reload) --------------------------
    _lr_bounds = {"max": args_cli.lr_max, "min": args_cli.lr_min}
    _noise_bounds = {"max": args_cli.max_noise_std, "min": args_cli.min_noise_std}

    hot_reloader = HotReloader(
        log_dir=log_dir,
        env=env,
        runner=runner,
        s2r_wrapper=s2r_wrapper,
        frozen_weights={"stumble", "body_height_tracking"},
        noise_bounds=_noise_bounds,
        lr_bounds=_lr_bounds,
    )
    hot_reloader.write_initial_state(
        iteration=start_iteration,
        lr=args_cli.lr_max,
        noise_std=args_cli.max_noise_std,
        terrain_level=0.0,
    )

    # -- Training loop with LR schedule + watchdog --------------------------
    mode = "fine-tuning from distilled_6899" if args_cli.actor_only_resume else "from scratch"
    print(f"[TRAIN] Starting {expert_type} training ({mode})...", flush=True)
    start_time = time.time()

    original_update = runner.alg.update
    _iteration = [start_iteration]
    _VL_THRESHOLD = 100.0
    _VL_CATASTROPHIC = 10000.0  # Emergency stop threshold (Bug #36)
    _VL_COOLDOWN_ITERS = 50
    _vl_penalty = [1.0]
    _vl_cooldown = [0]
    _vl_consecutive_spikes = [0]  # Track consecutive spikes for stacking
    _best_checkpoint_iter = [start_iteration]  # Track last good checkpoint for rollback

    def update_with_schedule(*args, **kwargs):
        it = _iteration[0]

        # Cosine LR annealing (uses mutable _lr_bounds for live tuning)
        lr = cosine_annealing_lr(
            it, agent_cfg.max_iterations,
            _lr_bounds["max"], _lr_bounds["min"], args_cli.warmup_iters
        )

        # Value loss watchdog penalty (stacking — Bug #36 fix)
        if _vl_cooldown[0] > 0:
            lr *= _vl_penalty[0]
            _vl_cooldown[0] -= 1
            if _vl_cooldown[0] == 0:
                _vl_penalty[0] = 1.0
                _vl_consecutive_spikes[0] = 0

        set_learning_rate(runner, lr)

        # Run PPO update
        result = original_update(*args, **kwargs)

        # Gradual actor unfreeze countdown
        # Actor network: actor.0 (235->512), actor.2 (512->256), actor.4 (256->128), actor.6 (128->12)
        if _actor_frozen[0] and _warmup_remaining[0] > 0:
            _warmup_remaining[0] -= 1
            if _warmup_remaining[0] == 0:
                # Phase 2: Unfreeze LAST layer only (actor.6 = 128->12 output)
                for name, param in runner.alg.policy.named_parameters():
                    if name.startswith("actor.6") or name in ("std", "log_std"):
                        param.requires_grad = True
                print(f"[WARMUP] Phase 2 at iter {it}: Last layer unfrozen (actor.6 + std)", flush=True)

        # Phase 3: Unfreeze middle layer at warmup + 200
        if _actor_frozen[0] and _warmup_remaining[0] == -200:
            for name, param in runner.alg.policy.named_parameters():
                if name.startswith("actor.4"):
                    param.requires_grad = True
            print(f"[WARMUP] Phase 3 at iter {it}: Middle layer unfrozen (actor.4)", flush=True)

        # Phase 4: Full unfreeze at warmup + 400
        if _actor_frozen[0] and _warmup_remaining[0] == -400:
            for name, param in runner.alg.policy.named_parameters():
                if name.startswith("actor."):
                    param.requires_grad = True
            _actor_frozen[0] = False
            print(f"[WARMUP] Phase 4 at iter {it}: ALL layers unfrozen. Full fine-tuning.", flush=True)

        # Keep counting down past 0 for phase tracking
        if _warmup_remaining[0] <= 0 and _actor_frozen[0]:
            _warmup_remaining[0] -= 1

        # Value loss watchdog check (stacking + emergency stop — Bug #36 fix)
        vl = result.get("value_function", 0.0) if isinstance(result, dict) else 0.0
        if vl > _VL_THRESHOLD:
            _vl_consecutive_spikes[0] += 1
            # Stack penalty: 0.5 -> 0.25 -> 0.125 -> 0.0625 (min)
            _vl_penalty[0] = max(0.0625, 0.5 ** _vl_consecutive_spikes[0])
            _vl_cooldown[0] = _VL_COOLDOWN_ITERS
            print(f"[GUARD] Value loss spike #{_vl_consecutive_spikes[0]}: {vl:.1f} at iter {it}. "
                  f"LR penalty={_vl_penalty[0]:.4f}", flush=True)

            # Emergency stop: if value loss is catastrophic, save and halt
            if vl > _VL_CATASTROPHIC:
                print(f"[EMERGENCY] Value loss {vl:.1f} > {_VL_CATASTROPHIC} — STOPPING TRAINING.", flush=True)
                print(f"[EMERGENCY] Last good checkpoint: model_{_best_checkpoint_iter[0]}.pt", flush=True)
                runner.save(os.path.join(log_dir, f"model_{it}.pt"))
                print(f"[EMERGENCY] Saved emergency checkpoint model_{it}.pt (may contain NaN)", flush=True)
                print(f"[EMERGENCY] Use model_{_best_checkpoint_iter[0]}.pt for evaluation.", flush=True)
                os._exit(1)
        else:
            # Reset consecutive counter on healthy iteration
            if _vl_consecutive_spikes[0] > 0 and _vl_cooldown[0] == 0:
                _vl_consecutive_spikes[0] = 0

        # Track last good checkpoint (saves happen at save_interval boundaries)
        if vl <= _VL_THRESHOLD and it % args_cli.save_interval == 0 and it > 0:
            _best_checkpoint_iter[0] = it

        # Noise clamp (uses mutable _noise_bounds for live tuning)
        clamp_noise_std(runner.alg.policy, _noise_bounds["min"], _noise_bounds["max"])

        # Update progressive S2R scale from terrain curriculum level
        # Read terrain level directly from the env's terrain curriculum
        terrain_level_raw = 0.0
        try:
            _inner = env
            while hasattr(_inner, 'env'):
                _inner = _inner.env
            if hasattr(_inner, 'unwrapped'):
                _inner = _inner.unwrapped
            if hasattr(_inner, 'scene') and hasattr(_inner.scene, 'terrain'):
                tl = _inner.scene.terrain.terrain_levels
                if tl is not None:
                    terrain_level_raw = float(tl.float().mean().item())
        except Exception:
            pass
        # Fallback: try result dict keys
        if terrain_level_raw == 0.0 and isinstance(result, dict):
            for k, v in result.items():
                if "terrain" in k.lower() and isinstance(v, (int, float)) and v > 0:
                    terrain_level_raw = v
                    break
        # Normalize for S2R scaling: terrain_levels is typically 0-9 raw
        terrain_level_norm = terrain_level_raw / 9.0 if terrain_level_raw > 1.0 else terrain_level_raw
        s2r_wrapper.update_scale_from_terrain(terrain_level_norm)

        # Periodic logging
        if it % 500 == 0:
            elapsed = time.time() - start_time
            hours = elapsed / 3600
            fps = (it - start_iteration) * steps_per_iter / max(elapsed, 1) if it > start_iteration else 0
            noise = runner.alg.policy.std.mean().item() if hasattr(runner.alg.policy, 'std') else 0
            warmup_str = f"  [WARMUP: {_warmup_remaining[0]} left]" if _actor_frozen[0] else ""
            s2r_str = s2r_wrapper.log_status()
            print(
                f"[TRAIN] {expert_type} iter={it:6d}/{agent_cfg.max_iterations}  "
                f"lr={lr:.2e}  noise={noise:.3f}  terrain={terrain_level_raw:.2f}  "
                f"{s2r_str}  elapsed={hours:.1f}h  fps={fps:.0f}{warmup_str}",
                flush=True,
            )

        # Live control panel: poll for parameter changes from CLI/dashboard
        changes = hot_reloader.poll_and_apply(it, lr, terrain_level_raw)
        for change_type, detail in changes:
            print(f"[CONTROL] {change_type}: {detail}", flush=True)

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
