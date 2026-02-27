"""Vision60 robot parameters.

Ghost Robotics Vision60:
  - 12 DOF: 4 hip (joint_0/2/4/6), 4 knee (joint_1/3/5/7), 4 abduction (joint_8/9/10/11)
  - Body mass: ~13.6 kg, standing height ~0.55m
  - PD gains: Kp=80, Kd=2.0
  - Foot bodies: lower0, lower1, lower2, lower3 (toe links merged)

Created for AI2C Tech Capstone — MS for Autonomy, February 2026
"""

from configs.robot_params import RobotParams

V60_PARAMS = RobotParams(
    name="vision60",
    description="Ghost Robotics Vision60 — 12 DOF quadruped (URDF)",

    # Body names
    foot_body_names="lower.*",
    termination_body_names=["body"],

    # Gait (trot: FL+RR diagonal, FR+RL diagonal)
    # Vision60 lower links: lower0=FL, lower1=RL, lower2=FR, lower3=RR
    gait_pairs=(("lower0", "lower3"), ("lower2", "lower1")),

    # Physical dimensions
    body_height=0.55,
    foot_clearance_target=0.08,
    stumble_knee_height=0.20,
    spawn_height=0.6,

    # PD gains
    kp=80.0,
    kd=2.0,

    # Joint names (Vision60 uses numeric joint names)
    hip_joint_names="joint_.*",
    all_joint_names=".*",

    # Asset
    asset_cfg_name="VISION60_CFG",

    # Reward weights — adjusted for heavier robot with more inertia
    air_time_weight=2.0,           # Reduced from 3.0 — heavier, less airtime
    foot_slip_weight=-2.0,         # Reduced from -3.0 — more weight = more traction
    base_motion_weight=-3.0,       # Reduced from -4.0 — heavier = more damped
    joint_vel_weight=-3.0e-2,      # Reduced from -5e-2 — more inertia
    body_height_tracking_weight=-2.0,
    contact_force_smoothness_weight=-0.5,
    stumble_weight=-2.0,
    velocity_modulation_weight=2.0,
    vegetation_drag_weight=-0.001,
)
