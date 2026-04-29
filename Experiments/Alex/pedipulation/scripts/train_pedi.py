"""Pedipulation Training Script — Transfer Learning from Hybrid No-Coach Baseline.

Key features:
  - Weight surgery: extends first layer 235 → 240 dims (zero-init new columns)
  - 3-phase curriculum: flat → flat+manipulation → stairs+forces
  - All safety features from train.py (NaN sanitizer, value loss watchdog, etc.)

Usage (H100):
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/pedipulation/scripts/train_pedi.py --headless \\
        --num_envs 5000 --max_iterations 8000 --phase flat \\
        --standing_fraction 0.6 \\
        --base_checkpoint ~/pedipulation/checkpoints/hybrid_nocoach_19999.pt \\
        --critic_warmup_iters 300 --lr_max 3e-5 --save_interval 100

Usage (local debug):
    isaaclab.bat -p pedipulation/scripts/train_pedi.py --headless \\
        --num_envs 64 --max_iterations 10 --phase flat \\
        --base_checkpoint checkpoints/hybrid_nocoach_19999.pt --no_wandb

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

# ── 0. Parse args BEFORE any Isaac imports ──────────────────────────────
import argparse
import os
import sys

# Add pedipulation root to sys.path so configs/ and mdp/ are importable
PEDI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PEDI_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pedipulation PPO training (transfer learning)")
parser.add_argument("--num_envs", type=int, default=5000,
                    help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=15000,
                    help="Max training iterations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--base_checkpoint", type=str, default=None,
                    help="Path to hybrid_nocoach_19999.pt for weight surgery")
parser.add_argument("--resume", action="store_true", default=False,
                    help="Resume from latest pedipulation checkpoint")
parser.add_argument("--load_run", type=str, default=None,
                    help="Run directory to resume from")
parser.add_argument("--load_checkpoint", type=str, default="model_.*\\.pt",
                    help="Checkpoint regex to load")
parser.add_argument("--phase", type=str, default="flat",
                    choices=["flat", "stairs", "full"],
                    help="Training phase: flat (Phase 1/2), stairs (Phase 3), full (all terrain)")
parser.add_argument("--standing_fraction", type=float, default=0.6,
                    help="Fraction of time in walking mode (0.6=Phase1, 0.3=Phase2/3)")
parser.add_argument("--critic_warmup_iters", type=int, default=0,
                    help="Freeze actor for N iters (critic calibration after weight surgery)")
parser.add_argument("--lr_max", type=float, default=3e-5,
                    help="Max learning rate (3e-5 for transfer, NOT 1e-3)")
parser.add_argument("--lr_min", type=float, default=1e-5,
                    help="Min learning rate for cosine annealing")
parser.add_argument("--warmup_iters", type=int, default=50,
                    help="LR warmup iterations")
parser.add_argument("--min_noise_std", type=float, default=0.3,
                    help="Minimum noise std floor")
parser.add_argument("--max_noise_std", type=float, default=0.5,
                    help="Maximum noise std ceiling (0.5 for transfer, NOT 1.0)")
parser.add_argument("--save_interval", type=int, default=100,
                    help="Checkpoint save interval")
parser.add_argument("--no_wandb", action="store_true", default=False,
                    help="Disable W&B, use TensorBoard only")
parser.add_argument("--ext_force_range", type=float, default=0.0,
                    help="External force range on body (N) — Phase 3: 5-15")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 1. Imports (AFTER SimulationApp) ────────────────────────────────────
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Trigger standard gym registrations
import isaaclab_tasks  # noqa: F401
import quadruped_locomotion  # noqa: F401

# Trigger pedipulation gym registration
import configs  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path

from configs.pedi_env_cfg import PedipulationSpotEnvCfg
from configs.pedi_ppo_cfg import PedipulationPPORunnerCfg

from quadruped_locomotion.utils.lr_schedule import cosine_annealing_lr, set_learning_rate
from quadruped_locomotion.utils.training_utils import (
    configure_tf32,
    clamp_noise_std,
    register_std_safety_clamp,
)

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

# Enable TF32
configure_tf32()


# ── 2. Weight Surgery ───────────────────────────────────────────────────

def perform_weight_surgery(
    checkpoint_path: str,
    old_obs_dim: int = 235,
    new_obs_dim: int = 240,
    insert_at: int = 48,
    new_dims: int = 5,
) -> dict:
    """Extend first layer from old_obs_dim to new_obs_dim by inserting zero columns.

    The new columns (foot_target + leg_flags) are zero-initialized so the policy
    initially ignores them and walks exactly like the baseline. Training gradually
    learns what the new dims mean.

    Args:
        checkpoint_path: Path to hybrid_nocoach_19999.pt.
        old_obs_dim: Original observation dimension (235).
        new_obs_dim: New observation dimension (240).
        insert_at: Index where new columns are inserted (48 = after proprio).
        new_dims: Number of new columns to insert (5 = foot_target + leg_flags).

    Returns:
        Modified state_dict with extended first-layer weights.
    """
    print(f"[SURGERY] Loading checkpoint: {checkpoint_path}", flush=True)
    loaded = torch.load(checkpoint_path, weights_only=False)
    state_dict = loaded["model_state_dict"]

    # Verify checkpoint integrity
    for key, val in state_dict.items():
        if torch.is_tensor(val) and torch.isnan(val).any():
            raise ValueError(f"NaN found in checkpoint key: {key}")

    for prefix in ["actor", "critic"]:
        key = f"{prefix}.0.weight"
        if key not in state_dict:
            print(f"[WARN] Key {key} not found in checkpoint, skipping", flush=True)
            continue

        old_w = state_dict[key]
        if old_w.shape[1] != old_obs_dim:
            raise ValueError(
                f"Expected {old_obs_dim} input dims for {key}, got {old_w.shape[1]}"
            )

        hidden = old_w.shape[0]
        new_w = torch.zeros(hidden, new_obs_dim, dtype=old_w.dtype, device=old_w.device)

        # Copy proprio columns (0:insert_at) unchanged
        new_w[:, :insert_at] = old_w[:, :insert_at]
        # New pedipulation columns (insert_at:insert_at+new_dims) stay zero
        # Copy height_scan columns shifted right by new_dims
        new_w[:, insert_at + new_dims:] = old_w[:, insert_at:]

        state_dict[key] = new_w
        print(
            f"[SURGERY] {key}: [{hidden}, {old_obs_dim}] -> [{hidden}, {new_obs_dim}] "
            f"(inserted {new_dims} zero columns at index {insert_at})",
            flush=True,
        )

        # Bias stays the same shape — no surgery needed
        bias_key = f"{prefix}.0.bias"
        if bias_key in state_dict:
            print(f"[SURGERY] {bias_key}: [{state_dict[bias_key].shape[0]}] (unchanged)", flush=True)

    # Log all deeper layers (no changes needed)
    for key, val in state_dict.items():
        if ".0." not in key and torch.is_tensor(val):
            print(f"[SURGERY] {key}: {list(val.shape)} (unchanged)", flush=True)

    return state_dict


# ── 3. Main ─────────────────────────────────────────────────────────────

def main():
    env_cfg = PedipulationSpotEnvCfg()
    agent_cfg = PedipulationPPORunnerCfg()

    # Apply CLI overrides
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed
    agent_cfg.save_interval = args_cli.save_interval

    # Override standing_fraction in leg_selection command
    env_cfg.commands.leg_selection.standing_fraction = args_cli.standing_fraction

    # ── Phase-based terrain override ────────────────────────────────────
    if args_cli.phase == "flat":
        print("[TERRAIN] Phase 1/2: 100% FLAT terrain", flush=True)
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
    elif args_cli.phase == "stairs":
        print("[TERRAIN] Phase 3: Staircase-heavy terrain", flush=True)
        # Uses default PEDI_STAIRCASE_TERRAINS_CFG from env config
    elif args_cli.phase == "full":
        print("[TERRAIN] Full terrain: staircase-heavy + all types", flush=True)
        # Uses default PEDI_STAIRCASE_TERRAINS_CFG from env config

    # External force on body (Phase 3: simulates pushing resistance)
    if args_cli.ext_force_range > 0:
        f = args_cli.ext_force_range
        env_cfg.events.base_external_force_torque.params["force_range"] = (-f, f)
        env_cfg.events.base_external_force_torque.params["torque_range"] = (-f / 3, f / 3)
        print(f"[EVENTS] External force: ±{f:.1f}N body, ±{f/3:.1f}Nm torque", flush=True)

    # W&B toggle
    if args_cli.no_wandb:
        agent_cfg.logger = "tensorboard"

    # Logging
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)

    steps_per_iter = env_cfg.scene.num_envs * agent_cfg.num_steps_per_env

    print(f"\n{'='*70}", flush=True)
    print(f"  PEDIPULATION PPO TRAINING — TRANSFER LEARNING", flush=True)
    print(f"  {'='*66}", flush=True)
    print(f"  Base checkpoint:  {args_cli.base_checkpoint or 'None (from scratch)'}", flush=True)
    print(f"  Phase:            {args_cli.phase.upper()}", flush=True)
    print(f"  Standing fraction:{args_cli.standing_fraction}", flush=True)
    print(f"  Envs:             {env_cfg.scene.num_envs}", flush=True)
    print(f"  Max iterations:   {agent_cfg.max_iterations}", flush=True)
    print(f"  Save interval:    {agent_cfg.save_interval}", flush=True)
    print(f"  LR schedule:      cosine {args_cli.lr_max} -> {args_cli.lr_min}", flush=True)
    print(f"  Noise std:        [{args_cli.min_noise_std}, {args_cli.max_noise_std}]", flush=True)
    print(f"  Network:          {agent_cfg.policy.actor_hidden_dims}", flush=True)
    print(f"  Obs dims:         240 (48 proprio + 5 pedi + 187 height)", flush=True)
    print(f"  Critic warmup:    {args_cli.critic_warmup_iters} iters", flush=True)
    print(f"  Ext force:        ±{args_cli.ext_force_range}N", flush=True)
    print(f"  Logger:           {agent_cfg.logger}", flush=True)
    print(f"  Log dir:          {log_dir}", flush=True)
    print(f"  Steps/iteration:  {steps_per_iter:,}", flush=True)
    print(f"  Est. total steps: {steps_per_iter * agent_cfg.max_iterations / 1e9:.2f}B", flush=True)
    print(f"{'='*70}\n", flush=True)

    # ── Create environment ──────────────────────────────────────────────
    env = gym.make("Pedipulation-Spot-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── Create runner ───────────────────────────────────────────────────
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )

    # ── Safety: clamp std ───────────────────────────────────────────────
    register_std_safety_clamp(
        runner.alg.policy, args_cli.min_noise_std, args_cli.max_noise_std
    )
    print(
        f"[SAFETY] Registered std clamp: [{args_cli.min_noise_std}, {args_cli.max_noise_std}]",
        flush=True,
    )

    # ── Weight surgery + load ───────────────────────────────────────────
    start_iteration = 0

    if args_cli.base_checkpoint and not args_cli.resume:
        # Transfer learning: weight surgery on base checkpoint
        surgered_state = perform_weight_surgery(args_cli.base_checkpoint)

        # Verify first-layer shape
        for prefix in ["actor", "critic"]:
            key = f"{prefix}.0.weight"
            shape = surgered_state[key].shape
            assert shape == (512 if prefix == "actor" else 512, 240), (
                f"Weight surgery failed: {key} shape {shape}, expected (512, 240)"
            )
            # Verify new columns are zero
            new_cols = surgered_state[key][:, 48:53]
            assert (new_cols == 0).all(), f"New columns in {key} are not zero!"

        print("[SURGERY] Verification passed: shapes correct, new columns zero", flush=True)

        # Load surgered weights into the policy
        runner.alg.policy.load_state_dict(surgered_state, strict=True)
        print(f"[INFO] Loaded surgered checkpoint into policy", flush=True)
        print("[INFO] Optimizer freshly initialized (new reward landscape)", flush=True)

    elif args_cli.resume:
        # Standard resume from pedipulation checkpoint (240-dim already)
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
        f.write(f"task: pedipulation\n")
        f.write(f"base_checkpoint: {args_cli.base_checkpoint}\n")
        f.write(f"phase: {args_cli.phase}\n")
        f.write(f"standing_fraction: {args_cli.standing_fraction}\n")
        f.write(f"num_envs: {env_cfg.scene.num_envs}\n")
        f.write(f"max_iterations: {agent_cfg.max_iterations}\n")
        f.write(f"lr_max: {args_cli.lr_max}\n")
        f.write(f"lr_min: {args_cli.lr_min}\n")
        f.write(f"warmup_iters: {args_cli.warmup_iters}\n")
        f.write(f"critic_warmup_iters: {args_cli.critic_warmup_iters}\n")
        f.write(f"noise_std: [{args_cli.min_noise_std}, {args_cli.max_noise_std}]\n")
        f.write(f"ext_force_range: {args_cli.ext_force_range}\n")
        f.write(f"seed: {args_cli.seed}\n")
        f.write(f"steps_per_iter: {steps_per_iter}\n")
        f.write(f"network: {agent_cfg.policy.actor_hidden_dims}\n")
        f.write(f"obs_dim: 240\n")

    # ── Training loop ───────────────────────────────────────────────────
    print(f"\n[TRAIN] Starting pedipulation training...", flush=True)
    start_time = time.time()

    # Set initial LR
    initial_lr = cosine_annealing_lr(
        start_iteration, agent_cfg.max_iterations,
        args_cli.lr_max, args_cli.lr_min, args_cli.warmup_iters
    )
    set_learning_rate(runner, initial_lr)
    print(f"[TRAIN] Initial LR: {initial_lr:.2e}", flush=True)

    # Monkey-patch PPO update with LR schedule + safety
    original_update = runner.alg.update
    _lr_log_interval = 500
    _iteration_counter = [start_iteration]

    # ── Critic warmup ───────────────────────────────────────────────────
    _critic_warmup_remaining = [args_cli.critic_warmup_iters]
    _actor_frozen = [False]

    if args_cli.critic_warmup_iters > 0:
        for name, param in runner.alg.policy.named_parameters():
            if name.startswith("actor.") or name in ("std", "log_std"):
                param.requires_grad = False
        _actor_frozen[0] = True
        print(
            f"[WARMUP] Actor frozen for {args_cli.critic_warmup_iters} iters "
            f"(critic calibration)",
            flush=True,
        )

    # ── Value loss watchdog ─────────────────────────────────────────────
    _VL_THRESHOLD = 100.0
    _VL_COOLDOWN_ITERS = 50
    _vl_penalty = [1.0]
    _vl_cooldown = [0]
    _vl_spike_count = [0]

    def update_with_schedule(*args, **kwargs):
        """Wrapper: cosine LR + critic warmup + value loss guard + noise clamp."""
        it = _iteration_counter[0]

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
                print(f"[GUARD] Value loss cooldown expired, restoring LR", flush=True)

        set_learning_rate(runner, lr)

        # Run PPO update
        result = original_update(*args, **kwargs)

        # Critic warmup countdown
        if _actor_frozen[0] and _critic_warmup_remaining[0] > 0:
            _critic_warmup_remaining[0] -= 1
            if _critic_warmup_remaining[0] == 0:
                for name, param in runner.alg.policy.named_parameters():
                    if name.startswith("actor.") or name in ("std", "log_std"):
                        param.requires_grad = True
                _actor_frozen[0] = False
                print(
                    f"[WARMUP] Critic warmup complete at iter {it}. Actor unfrozen.",
                    flush=True,
                )

        # Value loss watchdog
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

        # Clamp noise std
        clamp_noise_std(runner.alg.policy, args_cli.min_noise_std, args_cli.max_noise_std)

        # Console logging
        if it % _lr_log_interval == 0:
            elapsed = time.time() - start_time
            hours = elapsed / 3600
            fps = (
                (it - start_iteration) * steps_per_iter / max(elapsed, 1)
                if it > start_iteration
                else 0
            )
            noise = (
                runner.alg.policy.std.mean().item()
                if hasattr(runner.alg.policy, "std")
                else 0
            )

            guard_str = ""
            if _vl_cooldown[0] > 0:
                guard_str = f"  [GUARD: LR×{_vl_penalty[0]}, {_vl_cooldown[0]} left]"
            warmup_str = ""
            if _actor_frozen[0]:
                warmup_str = f"  [WARMUP: {_critic_warmup_remaining[0]} left]"

            print(
                f"[TRAIN] iter={it:6d}/{agent_cfg.max_iterations}  "
                f"lr={lr:.2e}  noise={noise:.3f}  "
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
    print(f"  PEDIPULATION TRAINING COMPLETE", flush=True)
    print(f"  {'='*66}", flush=True)
    print(f"  Total time:       {hours:.1f} hours ({elapsed:.0f} seconds)", flush=True)
    print(f"  Total steps:      {total_steps / 1e9:.2f}B", flush=True)
    print(f"  Avg throughput:   {total_steps / elapsed:.0f} steps/sec", flush=True)
    print(f"  Checkpoints:      {log_dir}", flush=True)
    print(f"  VL spikes:        {_vl_spike_count[0]}", flush=True)
    print(f"{'='*70}\n", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)
