"""Hardware calibration toolkit for Spot deployment.

Provides utilities for verifying and tuning the sim-to-real interface:
  - Joint zero position verification
  - PD gain sweep
  - Friction coefficient estimation
  - Communication latency measurement

Run each calibration before the first deployment to identify hardware-specific
parameters that differ from simulation defaults.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationResult:
    """Result from a calibration procedure."""
    name: str
    measured_value: float
    expected_value: float
    unit: str
    within_tolerance: bool
    tolerance: float

    def __repr__(self):
        status = "PASS" if self.within_tolerance else "FAIL"
        return (f"[{status}] {self.name}: measured={self.measured_value:.4f} "
                f"expected={self.expected_value:.4f} ({self.unit}) "
                f"tolerance=+/-{self.tolerance:.4f}")


def verify_joint_zeros(spot_joint_positions: np.ndarray, training_defaults: np.ndarray,
                       tolerance_rad: float = 0.05) -> list:
    """Compare Spot SDK default joint positions with training defaults.

    Args:
        spot_joint_positions: (12,) from Spot SDK in standing pose.
        training_defaults: (12,) from training config (SPOT_DEFAULT_POSITIONS).
        tolerance_rad: Acceptable difference in radians.

    Returns:
        List of CalibrationResult for each joint.
    """
    from deploy.spot_sdk_wrapper import DOF_ORDER

    results = []
    for i, name in enumerate(DOF_ORDER):
        diff = abs(spot_joint_positions[i] - training_defaults[i])
        results.append(CalibrationResult(
            name=f"joint_zero_{name}",
            measured_value=spot_joint_positions[i],
            expected_value=training_defaults[i],
            unit="rad",
            within_tolerance=diff < tolerance_rad,
            tolerance=tolerance_rad,
        ))
    return results


def measure_latency(send_fn, receive_fn, num_samples: int = 100) -> CalibrationResult:
    """Measure round-trip command latency.

    Args:
        send_fn: Callable that sends a command and returns send timestamp.
        receive_fn: Callable that waits for response and returns receive timestamp.
        num_samples: Number of latency samples to collect.

    Returns:
        CalibrationResult with median latency.
    """
    latencies = []
    for _ in range(num_samples):
        t_send = send_fn()
        t_recv = receive_fn()
        latencies.append(t_recv - t_send)
        time.sleep(0.01)  # 10ms between samples

    median_ms = np.median(latencies) * 1000
    expected_ms = 40.0  # Training uses 40ms delay

    return CalibrationResult(
        name="command_latency",
        measured_value=median_ms,
        expected_value=expected_ms,
        unit="ms",
        within_tolerance=abs(median_ms - expected_ms) < 20.0,
        tolerance=20.0,
    )


def sweep_pd_gains(spot_client, test_gains: list = None) -> list:
    """Sweep PD gains and measure tracking error.

    Args:
        spot_client: Spot SDK client with joint command access.
        test_gains: List of (Kp, Kd) tuples to test.

    Returns:
        List of (Kp, Kd, tracking_error_rad) results.
    """
    if test_gains is None:
        test_gains = [
            (40.0, 1.0),
            (50.0, 1.0),
            (60.0, 1.5),  # Training default
            (70.0, 1.5),
            (80.0, 2.0),
        ]

    # Placeholder — actual implementation requires Spot SDK
    print("[CALIB] PD gain sweep requires Spot SDK integration")
    print(f"[CALIB] Test configs: {test_gains}")
    print(f"[CALIB] Training default: Kp=60.0, Kd=1.5")
    return []
