"""
Teleop-enabled rough terrain policy player.
============================================

Runs the trained Spot rough-terrain PPO policy inside Isaac Lab's physics
pipeline (GPU PhysX, exact same solver as training) with keyboard control
for velocity commands.

This BYPASSES the standalone deployment physics entirely, using Isaac Lab's
ManagerBasedRLEnv which is guaranteed to match training.

Controls
--------
  W / S       Forward / backward
  A / D       Strafe left / right
  Q / E       Turn left / right
  SPACE       Emergency stop (zero all velocities)
  R           Reset environment
  ESC         Quit

Usage
-----
  cd C:\\IsaacLab
  python "PATH\\play_rough_teleop.py" --num_envs 1

Created for AI2C Tech Capstone - MS for Autonomy, February 2026
"""

# ── 0. Parse args BEFORE any Isaac imports ──────────────────────────────
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teleop rough-terrain Spot policy")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel environments (default 1 for teleop)")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to model checkpoint (.pt)")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
# Clear sys.argv so Hydra doesn't complain
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 1. Imports (AFTER SimulationApp) ────────────────────────────────────
import os
import time
import weakref

import carb
import omni
import torch

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Trigger gym registrations for all Isaac Lab tasks
import isaaclab_tasks  # noqa: F401

# Direct imports of our Spot rough-terrain configs
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.rough_env_cfg import (
    SpotRoughEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.agents.rsl_rl_ppo_cfg import (
    SpotRoughPPORunnerCfg,
)

# ── 2. Default checkpoint path ──────────────────────────────────────────
DEFAULT_CKPT = (
    r"C:\IsaacLab\logs\rsl_rl\spot_rough"
    r"\2026-02-09_15-18-50\model_4999.pt"
)

# ── 3. Keyboard teleop ──────────────────────────────────────────────────

class KeyboardTeleop:
    """Maps WASD/QE to velocity commands with smooth acceleration."""

    def __init__(self, max_lin=2.0, max_ang=1.5, accel=4.0, decay=0.85):
        self.max_lin = max_lin
        self.max_ang = max_ang
        self.accel = accel
        self.decay = decay
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self._held = set()
        self._sub = None

        # Subscribe to carb keyboard (requires windowed/GUI mode)
        try:
            import omni.appwindow
            appwin = omni.appwindow.get_default_app_window()
            inp = carb.input.acquire_input_interface()
            kb = appwin.get_keyboard()
            self._sub = inp.subscribe_to_keyboard_events(
                kb,
                lambda ev, *a, ref=weakref.proxy(self): ref._on_key(ev, *a),
            )
            print("[TELEOP] Keyboard input active")
        except Exception as e:
            print(f"[TELEOP] No keyboard (headless mode): {e}")
            print("[TELEOP] Using constant forward walk: vx=1.0")

    def _on_key(self, event, *args, **kwargs):
        name = event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._held.add(name)
            if name == "SPACE":
                self.vx = self.vy = self.wz = 0.0
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._held.discard(name)
        return True

    def update(self, dt):
        # In headless mode (no keyboard), use constant forward walk
        if self._sub is None:
            self.vx = 1.0
            return torch.tensor([[self.vx, self.vy, self.wz]])

        a = self.accel * dt

        # Forward / backward
        if "W" in self._held:
            self.vx = min(self.vx + a, self.max_lin)
        elif "S" in self._held:
            self.vx = max(self.vx - a, -self.max_lin)
        else:
            self.vx *= self.decay
            if abs(self.vx) < 0.02:
                self.vx = 0.0

        # Left / right strafe
        if "A" in self._held:
            self.vy = min(self.vy + a, self.max_lin)
        elif "D" in self._held:
            self.vy = max(self.vy - a, -self.max_lin)
        else:
            self.vy *= self.decay
            if abs(self.vy) < 0.02:
                self.vy = 0.0

        # Turn
        if "Q" in self._held:
            self.wz = min(self.wz + a, self.max_ang)
        elif "E" in self._held:
            self.wz = max(self.wz - a, -self.max_ang)
        else:
            self.wz *= self.decay
            if abs(self.wz) < 0.02:
                self.wz = 0.0

        return torch.tensor([[self.vx, self.vy, self.wz]])


# ── 4. Main ─────────────────────────────────────────────────────────────

def main():
    # --- Environment config ---
    env_cfg = SpotRoughEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs or 1
    env_cfg.seed = 42
    # Disable domain randomization for interactive play
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.push_robot = None
    env_cfg.events.base_external_force_torque = None

    # --- Agent config ---
    agent_cfg = SpotRoughPPORunnerCfg()

    # --- Checkpoint ---
    ckpt = args_cli.checkpoint or DEFAULT_CKPT
    if not os.path.exists(ckpt):
        print(f"[ERROR] Checkpoint not found: {ckpt}")
        return
    print(f"[INFO] Checkpoint: {ckpt}")

    # --- Create environment ---
    env = gym.make(
        "Isaac-Velocity-Rough-Spot-Play-v0",
        cfg=env_cfg,
    )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # --- Load trained policy ---
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device,
    )
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy

    print(f"[INFO] Policy loaded. Device: {env.unwrapped.device}")

    # --- Keyboard teleop ---
    teleop = KeyboardTeleop()

    print()
    print("=" * 60)
    print("  ROUGH TERRAIN TELEOP - Spot PPO Policy")
    print("-" * 60)
    print("  W / S   Forward / backward")
    print("  A / D   Strafe left / right")
    print("  Q / E   Turn left / right")
    print("  SPACE   Emergency stop")
    print("  ESC     Quit")
    print("=" * 60)
    print()

    # --- Play loop ---
    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    device = env.unwrapped.device
    step_count = 0
    is_headless = teleop._sub is None
    max_steps = 500 if is_headless else float("inf")  # 10s test in headless

    print(f"[INFO] Starting play loop (dt={dt:.4f}s, "
          f"{'headless 10s test' if is_headless else 'interactive'})")

    while simulation_app.is_running() and step_count < max_steps:
        t0 = time.time()

        # Update teleop command
        cmd = teleop.update(dt).to(device)

        with torch.inference_mode():
            # Run policy on observations
            actions = policy(obs)

            # Step environment (physics + rendering + observations)
            obs, _, dones, _ = env.step(actions)

            # Override velocity command AFTER step
            # (env.step resamples commands internally, so we overwrite)
            vel_term = env.unwrapped.command_manager.get_term("base_velocity")
            vel_term.vel_command_b[:] = cmd

            # Also patch the velocity command in the observation tensor
            # so the NEXT policy eval uses our command, not the random one
            # Obs layout: [0:3] lin_vel, [3:6] ang_vel, [6:9] gravity,
            #             [9:12] velocity_commands, ...
            obs[:, 9:12] = cmd

            # Handle episode resets
            policy_nn.reset(dones)

        # Status display
        step_count += 1
        if step_count % 25 == 0:
            # Get robot state for logging
            root_state = env.unwrapped.scene["robot"].data.root_pos_w
            body_z = float(root_state[0, 2])
            gravity_b = env.unwrapped.scene["robot"].data.projected_gravity_b[0]
            gz = float(gravity_b[2])
            status = "UPRIGHT" if gz < -0.7 else "TILTED!" if gz < -0.3 else "FALLEN!"
            print(
                f"  step={step_count:5d} | "
                f"vx={teleop.vx:+5.2f} vy={teleop.vy:+5.2f} wz={teleop.wz:+5.2f} | "
                f"z={body_z:.3f} gz={gz:+.3f} [{status}]"
            )

        # Real-time pacing (only in GUI mode)
        if not is_headless:
            elapsed = time.time() - t0
            if dt > elapsed:
                time.sleep(dt - elapsed)

    if is_headless:
        root_state = env.unwrapped.scene["robot"].data.root_pos_w
        body_z = float(root_state[0, 2])
        gz = float(env.unwrapped.scene["robot"].data.projected_gravity_b[0, 2])
        survived = body_z > 0.2 and gz < -0.5
        print(f"\n[RESULT] After {step_count} steps ({step_count*dt:.1f}s): "
              f"z={body_z:.3f}, gz={gz:+.3f} -> "
              f"{'SURVIVED!' if survived else 'FELL!'}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
