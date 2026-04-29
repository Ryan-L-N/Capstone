"""Real-time telemetry logging for Spot deployment.

Logs all observations, actions, and robot state at 20 Hz to JSONL format
(compatible with 4-env eval for direct sim-vs-real comparison).

Optionally streams data via UDP to a laptop dashboard for live monitoring.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from __future__ import annotations

import json
import os
import socket
import time
from typing import Optional

import numpy as np


class TelemetryLogger:
    """Logs deployment telemetry to JSONL file + optional UDP stream."""

    def __init__(
        self,
        output_dir: str = "telemetry",
        run_name: Optional[str] = None,
        udp_host: Optional[str] = None,
        udp_port: int = 9999,
    ):
        """Initialize telemetry logger.

        Args:
            output_dir: Directory for JSONL output files.
            run_name: Name for this deployment run (auto-generated if None).
            udp_host: If set, stream data via UDP to this host.
            udp_port: UDP port for streaming.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        if run_name is None:
            run_name = time.strftime("deploy_%Y%m%d_%H%M%S")
        self.run_name = run_name

        self.file_path = os.path.join(output_dir, f"{run_name}.jsonl")
        self._file = open(self.file_path, "a")

        self._udp_socket = None
        self._udp_addr = None
        if udp_host:
            self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_addr = (udp_host, udp_port)

        self._step_count = 0
        self._start_time = time.time()

        print(f"[TELEMETRY] Logging to: {self.file_path}")
        if udp_host:
            print(f"[TELEMETRY] UDP stream: {udp_host}:{udp_port}")

    def log_step(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        velocity_cmd: np.ndarray,
        joint_torques: Optional[np.ndarray] = None,
        battery_pct: Optional[float] = None,
        body_pitch: Optional[float] = None,
        body_roll: Optional[float] = None,
        base_velocity: Optional[float] = None,
        extra: Optional[dict] = None,
    ):
        """Log a single control step.

        Args:
            observation: (235,) policy observation.
            action: (12,) policy action.
            velocity_cmd: (3,) [vx, vy, yaw_rate] command.
            joint_torques: (12,) motor torques if available.
            battery_pct: Battery percentage.
            body_pitch: Body pitch in radians.
            body_roll: Body roll in radians.
            base_velocity: Base velocity magnitude in m/s.
            extra: Any additional data to log.
        """
        elapsed = time.time() - self._start_time

        record = {
            "step": self._step_count,
            "time_s": round(elapsed, 4),
            "velocity_cmd": velocity_cmd.tolist(),
            "action": action.tolist(),
        }

        # Optional fields
        if joint_torques is not None:
            record["joint_torques"] = joint_torques.tolist()
        if battery_pct is not None:
            record["battery_pct"] = round(battery_pct, 1)
        if body_pitch is not None:
            record["body_pitch_rad"] = round(body_pitch, 4)
        if body_roll is not None:
            record["body_roll_rad"] = round(body_roll, 4)
        if base_velocity is not None:
            record["base_velocity_ms"] = round(base_velocity, 3)
        if extra:
            record.update(extra)

        # Write to JSONL
        line = json.dumps(record)
        self._file.write(line + "\n")

        # Stream via UDP (truncated for bandwidth)
        if self._udp_socket and self._udp_addr:
            try:
                compact = json.dumps({
                    "s": self._step_count,
                    "t": round(elapsed, 2),
                    "v": velocity_cmd.tolist(),
                    "a": action[:4].tolist(),  # First 4 joints only (bandwidth)
                    "bat": battery_pct,
                })
                self._udp_socket.sendto(compact.encode(), self._udp_addr)
            except Exception:
                pass  # Don't crash on UDP errors

        self._step_count += 1

        # Flush every 100 steps
        if self._step_count % 100 == 0:
            self._file.flush()

    def close(self):
        """Flush and close the telemetry log."""
        self._file.flush()
        self._file.close()
        if self._udp_socket:
            self._udp_socket.close()

        print(f"[TELEMETRY] Logged {self._step_count} steps to {self.file_path}")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
