"""Spot robot parameters.

Boston Dynamics Spot (Spirit 40):
  - 12 DOF: 4 abduction (hx), 4 hip (hy), 4 knee (kn)
  - Body mass: ~32 kg, standing height ~0.42m
  - PD gains: Kp=60, Kd=1.5
  - Foot bodies: fl_foot, fr_foot, hl_foot, hr_foot

Created for AI2C Tech Capstone — MS for Autonomy, February 2026
"""

from configs.robot_params import RobotParams

SPOT_PARAMS = RobotParams(
    name="spot",
    description="Boston Dynamics Spot (Spirit 40) — 12 DOF quadruped",

    # Body names
    foot_body_names=".*_foot",
    termination_body_names=["body", ".*leg"],

    # Gait (trot: FL+HR diagonal, FR+HL diagonal)
    gait_pairs=(("fl_foot", "hr_foot"), ("fr_foot", "hl_foot")),

    # Physical dimensions
    body_height=0.42,
    foot_clearance_target=0.10,
    stumble_knee_height=0.15,
    spawn_height=0.5,

    # PD gains
    kp=60.0,
    kd=1.5,

    # Joint names
    hip_joint_names=".*_h[xy]",
    all_joint_names=".*",

    # Asset
    asset_cfg_name="SPOT_CFG",

    # Reward weights (from 100hr_env_run/configs/env_cfg.py)
    air_time_weight=3.0,
    foot_slip_weight=-3.0,
    base_motion_weight=-4.0,
    joint_vel_weight=-5.0e-2,
    body_height_tracking_weight=-2.0,
    contact_force_smoothness_weight=-0.5,
    stumble_weight=-2.0,
    velocity_modulation_weight=2.0,
    vegetation_drag_weight=-0.001,
)
