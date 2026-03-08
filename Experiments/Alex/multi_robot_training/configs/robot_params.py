"""Robot parameter abstraction for multi-robot training.

Captures all robot-specific differences in a single dataclass so env configs,
reward terms, and training scripts can be parameterized by robot identity.

Created for AI2C Tech Capstone — MS for Autonomy, February 2026
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class RobotParams:
    """Robot-specific parameters for training and evaluation.

    All robot-specific differences between Spot and Vision60 are captured
    here so the env configs and training scripts can be parameterized.
    """

    # --- Identity ---
    name: str                          # "spot" or "vision60"
    description: str                   # Human-readable description

    # --- Body names (for SceneEntityCfg) ---
    foot_body_names: str               # Regex for foot bodies: ".*_foot" or "lower.*"
    termination_body_names: list = field(default_factory=list)  # Bodies that trigger termination

    # --- Gait ---
    gait_pairs: tuple = ()             # Diagonal foot pairs for trot gait

    # --- Physical dimensions ---
    body_height: float = 0.42          # Nominal standing height (m)
    foot_clearance_target: float = 0.10  # Swing foot clearance (m)
    stumble_knee_height: float = 0.15  # Height above which contact = stumble (m)
    spawn_height: float = 0.5         # Initial spawn height (m)

    # --- PD gains ---
    kp: float = 60.0                   # Position gain (stiffness)
    kd: float = 1.5                    # Velocity gain (damping)

    # --- Joint names ---
    hip_joint_names: str = ".*_h[xy]"  # Regex for hip joints
    all_joint_names: str = ".*"        # Regex for all joints

    # --- Asset ---
    asset_cfg_name: str = ""           # Name of the asset config to import

    # --- Observation/Action ---
    obs_dim: int = 235                 # 48 proprioceptive + 187 height scan
    act_dim: int = 12                  # Joint position targets

    # --- Reward weight adjustments for heavier/lighter robots ---
    # Vision60 is heavier → reduce penalties that fight against inertia
    air_time_weight: float = 3.0
    foot_slip_weight: float = -3.0
    base_motion_weight: float = -4.0
    joint_vel_weight: float = -5.0e-2
    body_height_tracking_weight: float = -2.0
    contact_force_smoothness_weight: float = -0.5
    stumble_weight: float = -2.0
    velocity_modulation_weight: float = 2.0
    vegetation_drag_weight: float = -0.001
