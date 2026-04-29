"""Hardware safety layer for real Spot deployment.

Wraps policy execution with safety watchdogs that trigger E-stop on:
  - Joint torque exceeding 120% of motor limits
  - Body orientation exceeding 60 degrees pitch/roll
  - Base velocity exceeding 4.0 m/s (runaway detection)
  - Communication timeout (no policy output for >100 ms)
  - Battery below 10% (sit down)

All thresholds are configurable. The safety layer is the LAST line of
defense — the policy itself is trained with torque/orientation rewards,
but hardware safety cannot rely on learned behavior alone.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SafetyConfig:
    """Safety threshold configuration."""

    # Joint torque limits (Nm)
    hip_torque_limit: float = 45.0
    knee_torque_limit: float = 100.0
    torque_safety_margin: float = 1.2      # E-stop at 120% of limit

    # Orientation limits (radians)
    max_pitch: float = math.radians(60)    # 60 degrees
    max_roll: float = math.radians(60)     # 60 degrees

    # Velocity limits (m/s)
    max_velocity: float = 4.0              # Runaway detection

    # Communication timeout (seconds)
    command_timeout: float = 0.100         # 100 ms

    # Battery
    battery_warn_pct: float = 20.0         # Warning at 20%
    battery_sit_pct: float = 10.0          # Sit at 10%

    # Action limits (radians from default)
    max_action_delta: float = 0.5          # Max joint command deviation


class SafetyLayer:
    """Hardware safety system for real Spot deployment.

    Usage:
        safety = SafetyLayer(SafetyConfig())

        while running:
            action = policy(obs)
            safe_action, status = safety.check(action, robot_state)
            if status.e_stop:
                robot.e_stop()
                break
            robot.send_command(safe_action)
    """

    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()
        self.last_command_time = time.time()
        self._violation_count = 0
        self._total_steps = 0

    def check(self, action: np.ndarray, robot_state: dict) -> tuple:
        """Check action + robot state against safety thresholds.

        Args:
            action: (12,) joint position targets from policy.
            robot_state: Dict with keys:
                - joint_torques: (12,) current motor torques (Nm)
                - body_pitch: float (radians)
                - body_roll: float (radians)
                - base_velocity: float (m/s magnitude)
                - battery_pct: float (0-100)

        Returns:
            (safe_action, status) where status has .e_stop, .warning, .message
        """
        self._total_steps += 1
        self.last_command_time = time.time()

        status = SafetyStatus()
        safe_action = action.copy()

        cfg = self.config

        # 1. Clamp action to safe range
        safe_action = np.clip(safe_action, -cfg.max_action_delta, cfg.max_action_delta)

        # 2. Check joint torques
        torques = robot_state.get("joint_torques")
        if torques is not None:
            hip_limit = cfg.hip_torque_limit * cfg.torque_safety_margin
            knee_limit = cfg.knee_torque_limit * cfg.torque_safety_margin

            limits = np.array([hip_limit] * 4 + [hip_limit] * 4 + [knee_limit] * 4)
            violations = np.abs(torques) > limits

            if violations.any():
                status.e_stop = True
                status.message = f"TORQUE LIMIT: joints {np.where(violations)[0].tolist()}"
                return safe_action, status

        # 3. Check orientation
        pitch = abs(robot_state.get("body_pitch", 0.0))
        roll = abs(robot_state.get("body_roll", 0.0))

        if pitch > cfg.max_pitch or roll > cfg.max_roll:
            status.e_stop = True
            status.message = f"ORIENTATION: pitch={math.degrees(pitch):.1f} roll={math.degrees(roll):.1f}"
            return safe_action, status

        # 4. Check velocity
        velocity = robot_state.get("base_velocity", 0.0)
        if velocity > cfg.max_velocity:
            status.e_stop = True
            status.message = f"VELOCITY: {velocity:.1f} m/s > {cfg.max_velocity}"
            return safe_action, status

        # 5. Check battery
        battery = robot_state.get("battery_pct", 100.0)
        if battery < cfg.battery_sit_pct:
            status.e_stop = True
            status.message = f"BATTERY CRITICAL: {battery:.0f}%"
            return safe_action, status
        elif battery < cfg.battery_warn_pct:
            status.warning = True
            status.message = f"Battery low: {battery:.0f}%"

        return safe_action, status

    def check_timeout(self) -> bool:
        """Check if command timeout has been exceeded.

        Call this in a separate watchdog thread.

        Returns:
            True if timeout exceeded (should E-stop).
        """
        elapsed = time.time() - self.last_command_time
        return elapsed > self.config.command_timeout

    @property
    def violation_rate(self) -> float:
        """Fraction of steps with safety violations."""
        if self._total_steps == 0:
            return 0.0
        return self._violation_count / self._total_steps


class SafetyStatus:
    """Result of a safety check."""

    def __init__(self):
        self.e_stop: bool = False
        self.warning: bool = False
        self.message: str = ""

    def __repr__(self):
        if self.e_stop:
            return f"SafetyStatus(E-STOP: {self.message})"
        elif self.warning:
            return f"SafetyStatus(WARNING: {self.message})"
        return "SafetyStatus(OK)"
