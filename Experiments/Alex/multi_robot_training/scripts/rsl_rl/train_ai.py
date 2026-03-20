"""AI-Guided PPO Training for Spot.

Wraps the standard training loop with an LLM coach that monitors metrics
and adjusts reward weights at runtime. Trains through all phases
(flat → transition → robust_easy → robust) automatically.

Architecture:
    Training Loop (every iter) → Metrics Collector (every N iters) →
    Emergency Check → AI Coach (Claude API) → Guardrails → Actuator → Live Env

Usage (H100):
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/multi_robot_training/train_ai.py --headless \\
        --robot spot --start_phase flat --num_envs 5000 \\
        --max_noise_std 0.5 --min_noise_std 0.3 \\
        --coach_interval 100 --no_wandb

    # Resume mid-phase with AI coach:
    ./isaaclab.sh -p ~/multi_robot_training/train_ai.py --headless \\
        --robot spot --start_phase robust --num_envs 5000 \\
        --load_run 2026-03-08_12-40-08 --load_checkpoint model_1000.pt \\
        --coach_interval 100 --no_wandb

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

# ── 0. Parse args BEFORE any Isaac imports ──────────────────────────────
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AI-guided PPO training for Spot")
parser.add_argument("--robot", type=str, default="spot",
                    choices=["spot", "vision60"])
parser.add_argument("--start_phase", type=str, default="flat",
                    choices=["flat", "transition", "robust_easy", "robust", "mason_hybrid"],
                    help="Which phase to start from")
parser.add_argument("--end_phase", type=str, default="robust",
                    choices=["flat", "transition", "robust_easy", "robust", "mason_hybrid"],
                    help="Which phase to end at")
parser.add_argument("--num_envs", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--lr_max", type=float, default=None,
                    help="Override phase-specific lr_max")
parser.add_argument("--lr_min", type=float, default=1e-5)
parser.add_argument("--warmup_iters", type=int, default=50)
parser.add_argument("--max_noise_std", type=float, default=0.5)
parser.add_argument("--min_noise_std", type=float, default=0.3)
parser.add_argument("--num_learning_epochs", type=int, default=4)
parser.add_argument("--no_wandb", action="store_true", default=False)

# AI Coach args
parser.add_argument("--coach_interval", type=int, default=100,
                    help="Consult AI coach every N iterations")
parser.add_argument("--coach_model", type=str, default="claude-sonnet-4-20250514",
                    help="Claude model for coach decisions")
parser.add_argument("--no_coach", action="store_true", default=False,
                    help="Disable AI coach (run plain training)")
parser.add_argument("--anthropic_api_key", type=str, default=None,
                    help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")

# VLM vision args
parser.add_argument("--enable_vision", action="store_true", default=False,
                    help="Enable VLM mode: render frames and send to coach for visual gait analysis. "
                         "Requires --enable_cameras in AppLauncher args.")

# Resume args
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--load_checkpoint", type=str, default=None)

# Deferred coach activation (mason_hybrid mode)
parser.add_argument("--coach_mode", type=str, default="immediate",
                    choices=["immediate", "deferred"],
                    help="immediate=coach active from iter 0, deferred=3-stage activation")
parser.add_argument("--activation_threshold", type=int, default=3000,
                    help="Iterations before coach activates (deferred mode only)")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 1. Imports (AFTER SimulationApp) ────────────────────────────────────
import json
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import quadruped_locomotion  # noqa: F401  — registers our gym envs
from isaaclab_tasks.utils import get_checkpoint_path

from quadruped_locomotion.utils.lr_schedule import cosine_annealing_lr, set_learning_rate
from quadruped_locomotion.utils.training_utils import configure_tf32, register_std_safety_clamp

from quadruped_locomotion.ai_trainer.config import PHASE_CONFIGS, PHASE_ORDER, CoachConfig
from quadruped_locomotion.ai_trainer.metrics import MetricsCollector
from quadruped_locomotion.ai_trainer.coach import Coach
from quadruped_locomotion.ai_trainer.guardrails import Guardrails
from quadruped_locomotion.ai_trainer.actuator import Actuator
from quadruped_locomotion.ai_trainer.decision_log import DecisionLog

configure_tf32()


# ── 1b. VLM Frame Capture ─────────────────────────────────────────────

def _compute_look_at_quat(eye, target=(0.0, 0.0, 0.0)):
    """Compute (w, x, y, z) quaternion for a camera looking from eye toward target.

    Uses OpenGL convention: camera looks along -Z, up is +Y, right is +X.
    """
    import numpy as np

    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        # forward is parallel to up — use alternative up
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    # Rotation matrix: columns = camera axes (right, up, -forward) in world frame
    R = np.column_stack([right, up, -forward])

    # Matrix to quaternion
    trace = R.trace()
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q  # normalize to positive w
    return tuple(q.tolist())


# Env indices to sample from (picks best non-black, non-resetting frame)
_SAMPLE_ENV_IDS = [0, 10, 50]
# Skip envs that reset fewer than this many steps ago (mid-reset = garbage frame)
_MIN_EPISODE_STEPS = 10
# Collect frames from this many iterations before each coach check
_FRAME_WINDOW = 5


def _capture_best_frame(env):
    """Capture a single RGB frame, sampling the best from multiple envs.

    Picks the brightest non-black frame from _SAMPLE_ENV_IDS, skipping any
    env that is mid-reset (episode_length < _MIN_EPISODE_STEPS).

    Returns:
        numpy array (H, W, 3) uint8 or None if no valid frame available.
    """
    try:
        import numpy as np

        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env

        # Get episode lengths to skip mid-reset envs
        ep_lengths = None
        if hasattr(unwrapped, "episode_length_buf"):
            ep_lengths = unwrapped.episode_length_buf

        # Method 1: Read from dedicated coach_camera sensor (works in headless)
        if hasattr(unwrapped, "scene") and hasattr(unwrapped.scene, "sensors"):
            sensors = unwrapped.scene.sensors
            if "coach_camera" in sensors:
                camera = sensors["coach_camera"]
                rgb = camera.data.output["rgb"]
                if rgb is not None and rgb.numel() > 0:
                    num_envs = rgb.shape[0]
                    best_frame = None
                    best_brightness = -1.0

                    for eid in _SAMPLE_ENV_IDS:
                        if eid >= num_envs:
                            continue
                        # Skip mid-reset envs
                        if ep_lengths is not None and ep_lengths[eid] < _MIN_EPISODE_STEPS:
                            continue

                        frame = rgb[eid].cpu().numpy()
                        if frame.shape[-1] == 4:  # RGBA -> RGB
                            frame = frame[..., :3]
                        if frame.dtype != np.uint8:
                            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)

                        brightness = float(frame.mean())
                        if brightness > best_brightness and brightness > 0:
                            best_brightness = brightness
                            best_frame = frame

                    return best_frame

        # Method 2: Fallback to env.render() (single env, works with display)
        frame = unwrapped.render()
        if frame is not None and isinstance(frame, np.ndarray) and frame.max() > 0:
            if frame.dtype != np.uint8:
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            return frame

        return None
    except Exception as e:
        print(f"[AI-TRAIN] Frame capture failed: {e}", flush=True)
        return None


def _average_frames_to_png(frame_buffer) -> bytes | None:
    """Average a list of numpy frames and encode as PNG bytes.

    Args:
        frame_buffer: list of numpy arrays (H, W, 3) uint8

    Returns:
        PNG bytes or None if buffer is empty.
    """
    if not frame_buffer:
        return None
    try:
        from io import BytesIO

        import numpy as np
        from PIL import Image

        # Average all frames (float -> uint8)
        stacked = np.stack(frame_buffer, axis=0).astype(np.float32)
        averaged = np.mean(stacked, axis=0).astype(np.uint8)

        img = Image.fromarray(averaged)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        print(f"[AI-TRAIN] Frame averaging failed: {e}", flush=True)
        return None


# ── 2. Config Loading ───────────────────────────────────────────────────

def load_robot_configs(robot: str, phase: str = "robust"):
    """Load env and PPO configs for the specified robot and phase."""
    if robot == "spot" and phase == "mason_hybrid_obstacle":
        from quadruped_locomotion.tasks.locomotion.config.spot.mason_hybrid_env_cfg import SpotMasonHybridEnvCfg
        from quadruped_locomotion.tasks.locomotion.config.spot.agents.rsl_rl_mason_hybrid_cfg import SpotMasonHybridPPORunnerCfg
        from quadruped_locomotion.tasks.locomotion.mdp.terrains import OBSTACLE_FOCUS_TERRAINS_CFG
        env_cfg = SpotMasonHybridEnvCfg()
        # Swap terrain to obstacle-heavy config
        env_cfg.scene.terrain.terrain_generator = OBSTACLE_FOCUS_TERRAINS_CFG
        # Boost foot_clearance and loosen joint_pos for obstacle traversal
        env_cfg.rewards.foot_clearance.weight = 1.5   # up from 0.5
        env_cfg.rewards.joint_pos.weight = -0.4        # loosened from -0.7
        return env_cfg, SpotMasonHybridPPORunnerCfg(), "Locomotion-MasonHybrid-Spot-v0"
    elif robot == "spot" and phase == "mason_hybrid":
        from quadruped_locomotion.tasks.locomotion.config.spot.mason_hybrid_env_cfg import SpotMasonHybridEnvCfg
        from quadruped_locomotion.tasks.locomotion.config.spot.agents.rsl_rl_mason_hybrid_cfg import SpotMasonHybridPPORunnerCfg
        return SpotMasonHybridEnvCfg(), SpotMasonHybridPPORunnerCfg(), "Locomotion-MasonHybrid-Spot-v0"
    elif robot == "spot":
        from quadruped_locomotion.tasks.locomotion.config.spot.env_cfg import SpotLocomotionEnvCfg
        from quadruped_locomotion.tasks.locomotion.config.spot.agents.rsl_rl_ppo_cfg import SpotPPORunnerCfg
        return SpotLocomotionEnvCfg(), SpotPPORunnerCfg(), "Locomotion-Robust-Spot-v0"
    elif robot == "vision60":
        from quadruped_locomotion.tasks.locomotion.config.vision60.env_cfg import Vision60LocomotionEnvCfg
        from quadruped_locomotion.tasks.locomotion.config.vision60.agents.rsl_rl_ppo_cfg import Vision60PPORunnerCfg
        return Vision60LocomotionEnvCfg(), Vision60PPORunnerCfg(), "Locomotion-Robust-Vision60-v0"
    else:
        raise ValueError(f"Unknown robot: {robot}")


def apply_phase_terrain(env_cfg, phase_name: str):
    """Apply terrain configuration for the given phase."""
    terrain = PHASE_CONFIGS[phase_name].terrain
    if terrain == "flat":
        if hasattr(env_cfg.scene, "terrain"):
            env_cfg.scene.terrain.terrain_type = "plane"
        # Disable terrain curriculum — terrain_levels_vel requires generated terrain
        if hasattr(env_cfg, "curriculum") and env_cfg.curriculum is not None:
            if hasattr(env_cfg.curriculum, "terrain_levels"):
                env_cfg.curriculum.terrain_levels = None
    elif terrain in ("transition", "robust_easy", "robust"):
        if hasattr(env_cfg, "terrain_cfg_name"):
            env_cfg.terrain_cfg_name = terrain


# ── 3. Single Phase Training ───────────────────────────────────────────

def train_phase(
    phase_name: str,
    env_cfg,
    agent_cfg,
    env_id: str,
    coach: Coach | None,
    coach_cfg: CoachConfig,
    resume_run: str | None = None,
    resume_checkpoint: str | None = None,
) -> tuple[str | None, bool]:
    """Train a single phase with optional AI coaching.

    Returns:
        (checkpoint_path, should_advance) — path to best checkpoint and
        whether go/no-go criteria were met for phase advancement.
    """
    phase_cfg = PHASE_CONFIGS[phase_name]

    # Apply CLI overrides
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    agent_cfg.save_interval = args_cli.save_interval
    agent_cfg.max_iterations = phase_cfg.max_iterations

    if args_cli.no_wandb:
        agent_cfg.logger = "tensorboard"

    lr_max = args_cli.lr_max if args_cli.lr_max else phase_cfg.lr_max
    lr_min = args_cli.lr_min

    # Noise bounds — mutable dict shared with safety clamp and actuator
    noise_bounds = {
        "min": max(args_cli.min_noise_std, phase_cfg.min_noise_std),
        "max": min(args_cli.max_noise_std, phase_cfg.max_noise_std),
    }

    # Logging
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)

    steps_per_iter = env_cfg.scene.num_envs * agent_cfg.num_steps_per_env

    print(f"\n{'='*70}", flush=True)
    print(f"  AI-GUIDED TRAINING — PHASE: {phase_name.upper()}", flush=True)
    print(f"  {'='*66}", flush=True)
    print(f"  Robot:            {args_cli.robot}", flush=True)
    print(f"  Terrain:          {phase_cfg.terrain}", flush=True)
    print(f"  Envs:             {env_cfg.scene.num_envs}", flush=True)
    print(f"  Max iterations:   {agent_cfg.max_iterations}", flush=True)
    print(f"  LR:               {lr_max:.1e} -> {lr_min:.1e}", flush=True)
    print(f"  Noise bounds:     [{noise_bounds['min']}, {noise_bounds['max']}]", flush=True)
    print(f"  AI Coach:         {'ENABLED' if coach else 'DISABLED'}", flush=True)
    if coach:
        print(f"  Coach interval:   every {coach_cfg.check_interval} iters", flush=True)
        print(f"  Coach model:      {coach_cfg.api_model}", flush=True)
    print(f"  Log dir:          {log_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # ── Add coach camera sensor for VLM vision mode ────────────────────
    if args_cli.enable_vision:
        from isaaclab.sensors import CameraCfg

        # 3/4 rear view: behind, to the side, above — looking at the robot
        cam_eye = (-3.0, 2.0, 1.5)
        cam_target = (0.0, 0.0, 0.3)  # slightly above body center
        cam_rot = _compute_look_at_quat(cam_eye, cam_target)

        env_cfg.scene.coach_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body/coach_cam",
            offset=CameraCfg.OffsetCfg(pos=cam_eye, rot=cam_rot),
            width=320,
            height=240,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=15.0,
                focus_distance=5.0,
                horizontal_aperture=30.0,
            ),
            update_period=10.0,  # render infrequently — only need 1 frame per coach check
        )
        print(f"[AI-TRAIN] Coach camera added: 320x240 RGB, update every 10s sim-time",
              flush=True)

    # ── Create environment ──────────────────────────────────────────────
    EnvCfgClass = type(env_cfg)
    try:
        gym.register(
            id=env_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{EnvCfgClass.__module__}:{EnvCfgClass.__name__}",
            },
        )
    except gym.error.Error:
        pass  # Already registered

    make_kwargs = {"cfg": env_cfg}
    if args_cli.enable_vision:
        make_kwargs["render_mode"] = "rgb_array"

    env = gym.make(env_id, **make_kwargs)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── Create runner ───────────────────────────────────────────────────
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )

    # Resume from checkpoint if specified
    if resume_run and resume_checkpoint:
        resume_path = get_checkpoint_path(log_root_path, resume_run, resume_checkpoint)
        print(f"[AI-TRAIN] Resuming from {resume_path}", flush=True)

        # Validate checkpoint for NaN/Inf before loading (Bug: corrupted model_14100.pt wasted Trial 11g)
        print(f"[AI-TRAIN] Validating checkpoint for NaN/Inf...", flush=True)
        ckpt_data = torch.load(resume_path, weights_only=False, map_location="cpu")
        state_dict = ckpt_data if isinstance(ckpt_data, dict) else {}
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        nan_keys = []
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    nan_keys.append(key)
        if nan_keys:
            raise RuntimeError(
                f"CORRUPTED checkpoint! {len(nan_keys)} tensors contain NaN/Inf: "
                f"{nan_keys[:5]}{'...' if len(nan_keys) > 5 else ''}. "
                f"Do NOT resume from this checkpoint."
            )
        print(f"[AI-TRAIN] Checkpoint validated OK ({len(state_dict)} tensors checked)", flush=True)
        del ckpt_data, state_dict  # free memory

        runner.load(resume_path)

    # Register safety clamp
    register_std_safety_clamp(
        runner.alg.policy,
        min_std=noise_bounds["min"],
        max_std=noise_bounds["max"],
    )

    # ── Save config ─────────────────────────────────────────────────────
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # ── Setup AI Coach components ───────────────────────────────────────
    metrics_collector = MetricsCollector(env, runner, phase_name)
    guardrails = Guardrails(coach_cfg, phase_cfg)
    actuator = Actuator(env, runner)
    actuator.register_noise_control(noise_bounds)
    decision_log = DecisionLog(os.path.join(log_dir, coach_cfg.decision_log_path))

    # ── TensorBoard writer for AI coach metrics ──────────────────────
    # RSL-RL creates runner.writer after learn() starts, so we grab it lazily
    _tb_writer = [None]  # lazy init from runner.writer

    def _get_tb_writer():
        if _tb_writer[0] is None and hasattr(runner, "writer") and runner.writer:
            _tb_writer[0] = runner.writer
        return _tb_writer[0]

    def _tb_log_coach(it, snapshot, decision, applied, latency, guardrail_msgs):
        """Log AI coach activity to TensorBoard."""
        w = _get_tb_writer()
        if w is None:
            return

        # Coach decision type as integer (for plotting)
        action_map = {
            "no_change": 0, "adjust_weights": 1, "adjust_noise": 2,
            "adjust_lr": 3, "advance_phase": 4, "emergency_stop": 5,
        }
        w.add_scalar("AI_Coach/action_code", action_map.get(decision.action, -1), it)
        w.add_scalar("AI_Coach/confidence", decision.confidence, it)
        w.add_scalar("AI_Coach/api_latency_ms", latency, it)
        w.add_scalar("AI_Coach/num_changes_applied", len(applied), it)
        w.add_scalar("AI_Coach/guardrail_blocks", len(guardrail_msgs), it)

        # Log each current weight so we can track drift over time
        for name, weight in snapshot.current_weights.items():
            w.add_scalar(f"Reward_Weights/{name}", weight, it)

        # Log per-reward contributions
        for name, val in snapshot.reward_breakdown.items():
            w.add_scalar(f"Reward_Contrib/{name}", val, it)

        # Trends
        w.add_scalar("AI_Coach/reward_trend", snapshot.reward_trend, it)
        w.add_scalar("AI_Coach/terrain_trend", snapshot.terrain_trend, it)
        w.add_scalar("AI_Coach/value_loss_trend", snapshot.value_loss_trend, it)

        # Specific weight changes applied this step
        for name, (old, new) in applied.items():
            w.add_scalar(f"Weight_Changes/{name}", new - old, it)

    def _tb_log_emergency(it, action, details):
        """Log emergency events to TensorBoard."""
        w = _get_tb_writer()
        if w is None:
            return
        emergency_map = {"nan_rollback": 3, "halve_lr": 2, "emergency_stop": 3}
        w.add_scalar("AI_Coach/emergency", emergency_map.get(action, 1), it)
        w.add_text("AI_Coach/emergency_log", f"{action}: {details}", it)

    # ── Training loop with AI coaching ──────────────────────────────────
    start_time = time.time()
    best_reward = -float("inf")
    best_checkpoint = None

    # When resuming, the runner's iteration counter starts at the checkpoint
    # iteration (e.g. 3900), but the coach counter was starting at 0.
    # This caused a mismatch: training shows iter 10400 but coach logs iter 6500.
    # Fix: offset the coach counter so it matches checkpoint/runner iterations.
    _resume_offset = getattr(runner, "current_learning_iteration", 0)

    # Detect if RSL-RL is using adaptive LR schedule (mason_hybrid mode)
    _use_adaptive_lr = (agent_cfg.to_dict().get("algorithm", {}).get("schedule") == "adaptive")

    if not _use_adaptive_lr:
        initial_lr = cosine_annealing_lr(
            0, agent_cfg.max_iterations, lr_max, lr_min, args_cli.warmup_iters
        )
        set_learning_rate(runner, initial_lr)

    # Monkey-patch the update function for LR schedule + AI coaching
    original_update = runner.alg.update
    _iteration_counter = [_resume_offset]
    _log_interval = 100
    _lr_cooldown = [0]  # iterations remaining on emergency LR cooldown

    # Storage for per-iteration reward info (captured from runner logging)
    _last_reward_info = [{}]

    _initial_weights_logged = [False]

    # VLM frame buffer: collect frames from the last N iters before each coach check
    # and average them for a more representative visual signal.
    _frame_buffer = []

    # ── Capture RSL-RL internal metrics via log() interception ────────
    # RSL-RL's learn() computes mean_reward, mean_episode_length, losses,
    # and noise_std as local variables, then passes them to self.log(locals()).
    # We intercept log() to capture these values for the AI coach.
    # (1-iteration delay is negligible for coach checks every 100 iters.)
    _runner_log_metrics = {
        "mean_reward": 0.0,
        "mean_episode_length": 0.0,
        "mean_value_loss": 0.0,
        "mean_surrogate_loss": 0.0,
        "mean_std": 0.0,
    }

    import statistics as _stats
    _original_runner_log = runner.log

    def _intercepting_log(locs, *a, **kw):
        """Intercept runner.log() to capture rewbuffer/lenbuffer/losses/noise."""
        try:
            if "rewbuffer" in locs and len(locs["rewbuffer"]) > 0:
                _runner_log_metrics["mean_reward"] = _stats.mean(locs["rewbuffer"])
            if "lenbuffer" in locs and len(locs["lenbuffer"]) > 0:
                _runner_log_metrics["mean_episode_length"] = _stats.mean(locs["lenbuffer"])
            # RSL-RL stores losses in loss_dict (not as separate vars)
            if "loss_dict" in locs and isinstance(locs["loss_dict"], dict):
                ld = locs["loss_dict"]
                if "value_function" in ld:
                    _runner_log_metrics["mean_value_loss"] = float(ld["value_function"])
                if "surrogate" in ld:
                    _runner_log_metrics["mean_surrogate_loss"] = float(ld["surrogate"])
            # Fallback: some RSL-RL versions use mean_value_loss directly
            if "mean_value_loss" in locs:
                v = locs["mean_value_loss"]
                _runner_log_metrics["mean_value_loss"] = float(v.item() if hasattr(v, "item") else v)
            if "mean_surrogate_loss" in locs:
                v = locs["mean_surrogate_loss"]
                _runner_log_metrics["mean_surrogate_loss"] = float(v.item() if hasattr(v, "item") else v)
            # mean_std is computed inside log(), not in learn() locals.
            # Read directly from policy instead.
            try:
                _runner_log_metrics["mean_std"] = float(
                    runner.alg.policy.action_std.mean().item())
            except Exception:
                pass
        except Exception:
            pass
        return _original_runner_log(locs, *a, **kw)

    runner.log = _intercepting_log

    def update_with_ai_coach(*args, **kwargs):
        it = _iteration_counter[0]
        elapsed = time.time() - start_time
        hours = elapsed / 3600

        # Log initial reward weights once (after runner.writer is created)
        if not _initial_weights_logged[0]:
            w = _get_tb_writer()
            if w is not None:
                try:
                    rm = env.unwrapped.reward_manager
                    if hasattr(rm, "_term_cfgs"):
                        for name, cfg in rm._term_cfgs.items():
                            w.add_scalar(f"Reward_Weights/{name}", cfg.weight, 0)
                    _initial_weights_logged[0] = True
                except Exception:
                    pass

        # -- LR schedule (skip if using adaptive KL schedule) --
        if not _use_adaptive_lr:
            if _lr_cooldown[0] > 0:
                _lr_cooldown[0] -= 1
            else:
                lr = cosine_annealing_lr(
                    it - _resume_offset, agent_cfg.max_iterations, lr_max, lr_min,
                    args_cli.warmup_iters
                )
                set_learning_rate(runner, lr)

        # -- Run the actual PPO update --
        result = original_update(*args, **kwargs)

        # -- Collect reward info from env extras --
        try:
            extras = env.unwrapped.extras if hasattr(env.unwrapped, "extras") else {}
            log_data = extras.get("log", {})
            # Merge with any ep_infos from the runner
            if hasattr(runner, "ep_infos") and runner.ep_infos:
                for info in runner.ep_infos:
                    log_data.update(info)
            _last_reward_info[0] = log_data
        except Exception:
            pass

        # -- Inject RSL-RL internal metrics (captured from runner.log) --
        # These are the keys that MetricsCollector.collect() looks up:
        _last_reward_info[0]["Mean reward"] = _runner_log_metrics["mean_reward"]
        _last_reward_info[0]["Mean episode length"] = _runner_log_metrics["mean_episode_length"]
        _last_reward_info[0]["Mean value_function loss"] = _runner_log_metrics["mean_value_loss"]
        _last_reward_info[0]["Mean surrogate loss"] = _runner_log_metrics["mean_surrogate_loss"]
        _last_reward_info[0]["Mean action noise std"] = _runner_log_metrics["mean_std"]

        # -- Periodic logging --
        if it % _log_interval == 0:
            current_lr = runner.alg.optimizer.param_groups[0]["lr"]
            total_iters = agent_cfg.max_iterations + _resume_offset
            fps = (it - _resume_offset) * steps_per_iter / max(elapsed, 1) if it > _resume_offset else 0
            print(
                f"[AI-TRAIN] iter={it:6d}/{total_iters}  "
                f"lr={current_lr:.2e}  elapsed={hours:.1f}h  fps={fps:.0f}",
                flush=True,
            )

        # -- VLM frame buffering (collect frames approaching coach check) --
        if args_cli.enable_vision and coach and it > 0 and coach_cfg.check_interval > 0:
            remainder = it % coach_cfg.check_interval
            # remainder=0 means this IS the coach iter (handled in _run_coach_check)
            iters_until_check = (coach_cfg.check_interval - remainder) if remainder else coach_cfg.check_interval
            if 0 < iters_until_check <= _FRAME_WINDOW:
                frame = _capture_best_frame(env)
                if frame is not None:
                    _frame_buffer.append(frame)

        # -- AI Coach check (every N iterations) --
        if coach and it > 0 and it % coach_cfg.check_interval == 0:
            _run_coach_check(it, hours)

        _iteration_counter[0] += 1
        return result

    # 3-stage activation state for deferred mode
    _coach_activated = [args_cli.coach_mode == "immediate"]
    _first_plateau_seen = [False]
    _activation_threshold = args_cli.activation_threshold if args_cli.coach_mode == "deferred" else 0

    def _run_coach_check(it: int, hours: float):
        """Run the AI coach check at the current iteration.

        3-stage activation (deferred mode):
          Stage 1 (silent):  it < activation_threshold → collect metrics only, no API
          Stage 2 (passive): it >= threshold, no plateau yet → API call, biased no_change
          Stage 3 (active):  plateau detected → full coach intervention
        """
        current_lr = runner.alg.optimizer.param_groups[0]["lr"]
        relative_it = it - _resume_offset  # iterations since this run started

        # Collect metrics (always — needed for plateau detection)
        snapshot = metrics_collector.collect(
            iteration=it,
            elapsed_hours=hours,
            reward_info=_last_reward_info[0],
            lr=current_lr,
        )

        # Stage 1: Silent mode — metrics only, no API call
        if _activation_threshold > 0 and relative_it < _activation_threshold:
            if relative_it % 500 == 0:
                print(f"[AI-COACH] Silent mode — {_activation_threshold - relative_it} "
                      f"iters until activation (terrain={snapshot.mean_terrain_level:.2f})",
                      flush=True)
            return

        # Track best reward
        nonlocal best_reward, best_checkpoint
        if snapshot.mean_reward > best_reward:
            best_reward = snapshot.mean_reward
            best_checkpoint = os.path.join(log_dir, f"model_{it}.pt")

        # 1. Emergency checks (override coach)
        emergency = guardrails.check_emergency(snapshot)
        if emergency:
            if emergency == "nan_rollback":
                msg = "NaN in policy parameters"
                print(f"[AI-COACH] EMERGENCY: NaN detected at iter {it}!",
                      flush=True)
                decision_log.log_emergency(it, phase_name, "nan_rollback", msg)
                _tb_log_emergency(it, "nan_rollback", msg)
                env.close()
                return

            elif emergency == "halve_lr":
                new_lr = actuator.emergency_halve_lr()
                _lr_cooldown[0] = 50  # hold for 50 iters
                msg = (f"Value loss {snapshot.value_loss:.1f} > 100, "
                       f"LR halved to {new_lr:.2e}")
                decision_log.log_emergency(it, phase_name, "halve_lr", msg)
                _tb_log_emergency(it, "halve_lr", msg)
                return

            elif emergency == "emergency_stop":
                msg = f"action_smoothness={snapshot.reward_breakdown.get('action_smoothness', 'N/A')}"
                print(f"[AI-COACH] EMERGENCY STOP at iter {it}!", flush=True)
                decision_log.log_emergency(it, phase_name, "emergency_stop", msg)
                _tb_log_emergency(it, "emergency_stop", msg)
                return

        # 2. Activate coach if in deferred mode
        if not _coach_activated[0]:
            _coach_activated[0] = True
            coach.set_passive_mode(True)
            print(f"[AI-COACH] Activated in PASSIVE mode at iter {it}", flush=True)

        # 2b. Consult AI coach
        if not coach.is_available:
            return

        plateau = metrics_collector.is_plateau()

        # Transition from passive to active on first plateau
        if plateau and not _first_plateau_seen[0]:
            _first_plateau_seen[0] = True
            coach.set_passive_mode(False)
            print(f"[AI-COACH] PLATEAU DETECTED — switching to ACTIVE mode at iter {it}",
                  flush=True)
        recent_history = list(metrics_collector.history)
        recent_decisions = decision_log.get_recent(coach_cfg.decision_history)

        # Capture frame for VLM visual analysis (averaged over recent window)
        frame_png = None
        if args_cli.enable_vision:
            # Also capture one more frame right now (in case buffer is sparse)
            frame = _capture_best_frame(env)
            if frame is not None:
                _frame_buffer.append(frame)
            n_buffered = len(_frame_buffer)
            frame_png = _average_frames_to_png(_frame_buffer)
            _frame_buffer.clear()
            if frame_png:
                print(f"[AI-COACH] VLM frame: averaged {n_buffered} frames "
                      f"from envs {_SAMPLE_ENV_IDS}", flush=True)

        decision, latency = coach.get_decision(
            snapshot, recent_history, recent_decisions, plateau,
            frame_png=frame_png)

        print(f"[AI-COACH] iter={it} action={decision.action} "
              f"confidence={decision.confidence:.2f} "
              f"latency={latency:.0f}ms", flush=True)
        if decision.reasoning:
            print(f"[AI-COACH] reason: {decision.reasoning}", flush=True)

        # 3. Apply through guardrails
        all_msgs = []
        applied = {}

        if decision.action == "adjust_weights" and decision.weight_changes:
            validated_weights, msgs = guardrails.validate_weight_changes(
                decision.weight_changes, snapshot.current_weights,
                current_terrain_level=snapshot.mean_terrain_level)
            all_msgs.extend(msgs)
            if validated_weights:
                applied = actuator.apply_weight_changes(validated_weights)
                for name, (old, new) in applied.items():
                    print(f"[AI-COACH] CHANGED {name}: {old:.4f} -> {new:.4f}",
                          flush=True)

        if decision.lr_change is not None:
            new_lr, msgs = guardrails.validate_lr_change(decision.lr_change)
            all_msgs.extend(msgs)
            if new_lr is not None:
                old_lr = actuator.apply_lr_change(new_lr)
                applied["lr"] = (old_lr, new_lr)
                print(f"[AI-COACH] CHANGED LR: {old_lr:.2e} -> {new_lr:.2e}",
                      flush=True)

        if decision.noise_change is not None:
            new_noise, msgs = guardrails.validate_noise_change(
                decision.noise_change)
            all_msgs.extend(msgs)
            if new_noise is not None:
                actuator.apply_noise_change(new_noise)
                applied["max_noise_std"] = (noise_bounds["max"], new_noise)
                print(f"[AI-COACH] CHANGED noise: {noise_bounds['max']:.2f} "
                      f"-> {new_noise:.2f}", flush=True)

        if all_msgs:
            for msg in all_msgs:
                print(f"[AI-COACH] guardrail: {msg}", flush=True)

        # 4. Log everything
        decision_log.log_decision(
            iteration=it,
            phase=phase_name,
            metrics=snapshot.to_dict(),
            decision=decision.to_dict(),
            guardrail_msgs=all_msgs,
            applied_changes={k: list(v) if isinstance(v, tuple) else v
                             for k, v in applied.items()},
            api_latency_ms=latency,
        )

        # 5. Log to TensorBoard
        _tb_log_coach(it, snapshot, decision, applied, latency, all_msgs)

    runner.alg.update = update_with_ai_coach

    # ── Run training ────────────────────────────────────────────────────
    print(f"\n[AI-TRAIN] Starting phase {phase_name}...", flush=True)
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    elapsed = time.time() - start_time
    print(f"\n{'='*70}", flush=True)
    print(f"  PHASE {phase_name.upper()} COMPLETE — {elapsed/3600:.1f} hours",
          flush=True)
    print(f"  Best reward: {best_reward:.1f}", flush=True)
    print(f"  Checkpoints: {log_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Check go/no-go for phase advancement
    should_advance = False
    if coach:
        go, failures = metrics_collector.go_no_go(phase_cfg)
        if go:
            print(f"[AI-TRAIN] GO for next phase! All criteria met.", flush=True)
            should_advance = True
        else:
            print(f"[AI-TRAIN] NO-GO for next phase: {failures}", flush=True)

    # Save final checkpoint
    final_ckpt = os.path.join(log_dir, f"model_{_iteration_counter[0]}.pt")
    actuator.save_checkpoint(final_ckpt)

    env.close()
    return final_ckpt, should_advance


# ── 4. Main ─────────────────────────────────────────────────────────────

def main():
    coach_cfg = CoachConfig(
        check_interval=args_cli.coach_interval,
        api_model=args_cli.coach_model,
    )

    # Mason hybrid modes — use tighter bounds, disable LR/noise changes
    if args_cli.start_phase in ("mason_hybrid", "mason_hybrid_obstacle"):
        coach_cfg.activation_threshold = args_cli.activation_threshold
        coach_cfg.lr_change_enabled = False   # Adaptive KL schedule manages LR
        coach_cfg.noise_change_enabled = False # Adaptive schedule manages noise
        # Use phase-specific bounds
        if args_cli.start_phase == "mason_hybrid_obstacle":
            coach_cfg.weight_bounds = coach_cfg.mason_hybrid_obstacle_bounds

    # Initialize AI coach
    coach = None
    if not args_cli.no_coach:
        api_key = args_cli.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            start_phase_cfg = PHASE_CONFIGS[args_cli.start_phase]
            coach = Coach(coach_cfg, start_phase_cfg, api_key=api_key,
                          vision_enabled=args_cli.enable_vision)
            print("[AI-TRAIN] AI Coach initialized", flush=True)
            if args_cli.enable_vision:
                print("[AI-TRAIN] VLM vision mode ENABLED — coach will receive rendered frames",
                      flush=True)
            if args_cli.coach_mode == "deferred":
                print(f"[AI-TRAIN] Coach mode: DEFERRED (silent for {args_cli.activation_threshold} iters)",
                      flush=True)
        else:
            print("[AI-TRAIN] WARNING: No API key, running without AI coach",
                  flush=True)

    # Determine phase range — mason_hybrid variants are single-phase runs
    if args_cli.start_phase in ("mason_hybrid", "mason_hybrid_obstacle"):
        phases = [args_cli.start_phase]
    else:
        start_idx = PHASE_ORDER.index(args_cli.start_phase)
        end_idx = PHASE_ORDER.index(args_cli.end_phase)
        phases = PHASE_ORDER[start_idx:end_idx + 1]

    print(f"[AI-TRAIN] Training phases: {' → '.join(phases)}", flush=True)

    resume_run = args_cli.load_run
    resume_ckpt = args_cli.load_checkpoint

    for i, phase_name in enumerate(phases):
        print(f"\n[AI-TRAIN] === PHASE {i+1}/{len(phases)}: {phase_name} ===",
              flush=True)

        # Update coach phase config
        if coach:
            coach.update_phase(PHASE_CONFIGS[phase_name])

        # Load fresh configs for each phase (terrain changes need new env)
        env_cfg, agent_cfg, env_id = load_robot_configs(args_cli.robot, phase_name)
        if phase_name != "mason_hybrid":
            apply_phase_terrain(env_cfg, phase_name)

        checkpoint, should_advance = train_phase(
            phase_name=phase_name,
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            env_id=env_id,
            coach=coach,
            coach_cfg=coach_cfg,
            resume_run=resume_run,
            resume_checkpoint=resume_ckpt,
        )

        if not should_advance and i < len(phases) - 1:
            print(f"[AI-TRAIN] Phase {phase_name} did not meet advancement "
                  f"criteria. Stopping.", flush=True)
            # Write state file for potential manual resume
            state = {
                "last_phase": phase_name,
                "checkpoint": checkpoint,
                "next_phase": phases[i + 1] if i + 1 < len(phases) else None,
            }
            with open("ai_train_state.json", "w") as f:
                json.dump(state, f, indent=2)
            break

        # Next phase resumes from this phase's checkpoint
        if checkpoint:
            resume_run = os.path.dirname(checkpoint).split("/")[-1]
            resume_ckpt = os.path.basename(checkpoint)
        else:
            resume_run = None
            resume_ckpt = None

    print("\n[AI-TRAIN] All phases complete!", flush=True)


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)
