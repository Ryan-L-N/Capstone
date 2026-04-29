"""
Play/Evaluate a trained 100hr Spot multi-terrain robust policy.
================================================================

Loads a trained checkpoint and runs the policy visually on the
ROBUST_TERRAINS_CFG terrain grid. Supports video recording,
real-time playback, and JIT/ONNX export for deployment.

Usage (headless video capture):
    cd C:\\IsaacLab
    isaaclab.bat -p /path/to/play_100hr.py --headless --video --video_length 500

Usage (interactive — GUI):
    cd C:\\IsaacLab
    isaaclab.bat -p /path/to/play_100hr.py --num_envs 16

Usage (real-time on RTX 2000 Ada):
    cd C:\\IsaacLab
    isaaclab.bat -p /path/to/play_100hr.py --num_envs 16 --real_time

Usage (specific checkpoint):
    cd C:\\IsaacLab
    isaaclab.bat -p /path/to/play_100hr.py --checkpoint /path/to/model_50000.pt

Usage (H100 server — large-scale eval):
    cd ~/IsaacLab
    ./isaaclab.sh -p /path/to/play_100hr.py --headless --num_envs 512 --video

Created for AI2C Tech Capstone — MS for Autonomy, February 2026
"""

# ── 0. Parse args BEFORE any Isaac imports ──────────────────────────────
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play trained 100hr Spot robust policy")
parser.add_argument("--num_envs", type=int, default=50,
                    help="Number of environments to simulate (default 50)")
parser.add_argument("--video", action="store_true", default=False,
                    help="Record video of the policy running")
parser.add_argument("--video_length", type=int, default=300,
                    help="Length of recorded video in steps (default 300 = 6s at 50Hz)")
parser.add_argument("--real_time", action="store_true", default=False,
                    help="Run in real-time (limit to wall-clock step rate)")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Explicit path to a model checkpoint (.pt file)")
parser.add_argument("--load_run", type=str, default=None,
                    help="Specific run directory under logs/ to load from")
parser.add_argument("--load_checkpoint", type=str, default="model_.*\\.pt",
                    help="Checkpoint regex pattern (default: latest model_*.pt)")
parser.add_argument("--export", action="store_true", default=False,
                    help="Export policy as JIT and ONNX after loading")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Enable cameras for video recording
if args_cli.video:
    args_cli.enable_cameras = True

# Clear argv for any downstream Hydra/omni parsing
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 1. Imports (AFTER SimulationApp) ────────────────────────────────────
import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path

# Add our project to path for custom configs
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from configs.env_cfg import Spot100hrEnvCfg_PLAY
from configs.ppo_cfg import Spot100hrPPORunnerCfg


def main():
    # --- Environment config (PLAY variant: 50 envs, no curriculum, no corruption) ---
    env_cfg = Spot100hrEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # --- Agent config (for loading the trained model) ---
    agent_cfg = Spot100hrPPORunnerCfg()
    agent_cfg.seed = args_cli.seed

    # --- Locate checkpoint ---
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.checkpoint:
        # Explicit checkpoint path provided
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        # Auto-detect from logs directory
        resume_path = get_checkpoint_path(
            log_root_path, args_cli.load_run, args_cli.load_checkpoint
        )

    log_dir = os.path.dirname(resume_path)

    print(f"\n{'='*70}", flush=True)
    print(f"  100-HOUR SPOT POLICY — PLAY / EVALUATE", flush=True)
    print(f"  {'='*66}", flush=True)
    print(f"  Checkpoint:   {resume_path}", flush=True)
    print(f"  Num envs:     {env_cfg.scene.num_envs}", flush=True)
    print(f"  Video:        {args_cli.video} (length={args_cli.video_length})", flush=True)
    print(f"  Real-time:    {args_cli.real_time}", flush=True)
    print(f"  Export:        {args_cli.export}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # --- Register and create environment ---
    gym.register(
        id="Isaac-Velocity-Robust-Spot-100hr-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{Spot100hrEnvCfg_PLAY.__module__}:{Spot100hrEnvCfg_PLAY.__name__}",
        },
    )

    env = gym.make(
        "Isaac-Velocity-Robust-Spot-100hr-Play-v0",
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video with settings:", flush=True)
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # --- Load trained model ---
    print(f"[INFO] Loading model from: {resume_path}", flush=True)
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    runner.load(resume_path)

    # Get inference policy
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Get the neural network module
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # Get the normalizer (for export)
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # --- Export policy if requested ---
    if args_cli.export:
        export_dir = os.path.join(os.path.dirname(resume_path), "exported")
        os.makedirs(export_dir, exist_ok=True)
        print(f"[INFO] Exporting policy to: {export_dir}", flush=True)
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.onnx")
        print("[INFO] Export complete: policy.pt (JIT) + policy.onnx (ONNX)", flush=True)

    # --- Run inference loop ---
    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0

    print("[INFO] Starting inference loop... (Ctrl+C or close window to stop)", flush=True)

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            # Reset recurrent states for terminated episodes
            policy_nn.reset(dones)

        timestep += 1

        # Video mode: stop after capturing enough frames
        if args_cli.video and timestep >= args_cli.video_length:
            print(f"[INFO] Video recording complete ({timestep} steps).", flush=True)
            break

        # Real-time pacing
        if args_cli.real_time:
            elapsed = time.time() - start_time
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Periodic status
        if timestep % 500 == 0:
            elapsed_total = timestep * dt
            print(
                f"[PLAY] step={timestep}  sim_time={elapsed_total:.1f}s",
                flush=True,
            )

    # --- Cleanup ---
    env.close()
    print(f"\n[INFO] Done. Ran {timestep} steps ({timestep * dt:.1f}s sim time).", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
