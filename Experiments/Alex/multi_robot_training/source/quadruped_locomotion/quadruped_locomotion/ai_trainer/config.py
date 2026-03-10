"""Phase configs and coach settings for AI-guided training.

Encodes all hard-won lessons from TRAINING_CURRICULUM.md and the Bug Museum
into machine-readable form. These are the guardrails that prevent the AI
coach from repeating the mistakes we already made.
"""

from dataclasses import dataclass, field


@dataclass
class PhaseConfig:
    """Configuration for a single training phase."""
    name: str
    terrain: str               # CLI --terrain value
    num_envs: int
    max_iterations: int
    lr_max: float
    lr_min: float = 1e-5
    warmup_iters: int = 50
    max_noise_std: float = 0.5
    min_noise_std: float = 0.3
    save_interval: int = 100
    num_learning_epochs: int = 4

    # Go/no-go criteria for advancing TO the next phase
    min_survival_rate: float = 0.90    # >90%
    max_flip_rate: float = 0.05        # <5%
    max_noise_std_advance: float = 0.6
    max_value_loss: float = 10.0
    min_terrain_level: float = 0.0     # varies by phase
    min_consecutive_iters: int = 100   # criteria must hold for N iters

    # Reward weight overrides for this phase (applied on top of base config)
    reward_overrides: dict = field(default_factory=dict)

    # Weights the AI must never touch
    frozen_weights: set = field(default_factory=lambda: {
        "stumble",              # Bug #28b: world-frame Z breaks on elevated terrain
        "body_height_tracking", # Bug #22: world-frame Z meaningless on rough terrain
    })


# Phase definitions — proven values from TRAINING_CURRICULUM.md run history
PHASE_CONFIGS = {
    "flat": PhaseConfig(
        name="flat",
        terrain="flat",
        num_envs=10000,
        max_iterations=500,
        lr_max=3e-4,
        max_noise_std=1.0,
        min_noise_std=0.3,
        min_terrain_level=0.0,
        min_survival_rate=0.95,
        max_flip_rate=0.03,
    ),
    "transition": PhaseConfig(
        name="transition",
        terrain="transition",
        num_envs=10000,
        max_iterations=1000,
        lr_max=3e-4,
        max_noise_std=0.7,
        min_noise_std=0.3,
        min_terrain_level=0.0,
        min_survival_rate=0.90,
        max_flip_rate=0.05,
    ),
    "robust_easy": PhaseConfig(
        name="robust_easy",
        terrain="robust_easy",
        num_envs=5000,
        max_iterations=10000,
        lr_max=5e-5,
        max_noise_std=0.5,
        min_noise_std=0.3,
        min_terrain_level=2.0,
        min_survival_rate=0.85,
        max_flip_rate=0.10,
    ),
    "robust": PhaseConfig(
        name="robust",
        terrain="robust",
        num_envs=5000,
        max_iterations=30000,
        lr_max=3e-5,
        max_noise_std=0.5,
        min_noise_std=0.3,
        min_terrain_level=5.0,
        min_survival_rate=0.85,
        max_flip_rate=0.12,
    ),
    "mason_hybrid": PhaseConfig(
        name="mason_hybrid",
        terrain="robust",
        num_envs=4096,
        max_iterations=20000,
        lr_max=1e-3,          # Mason's — managed by adaptive KL schedule
        lr_min=1e-5,
        warmup_iters=0,       # No warmup — adaptive schedule handles it
        max_noise_std=1.0,    # Mason's init_noise_std — adaptive schedule manages
        min_noise_std=0.2,
        save_interval=100,
        num_learning_epochs=5,
        min_terrain_level=6.0,
        min_survival_rate=0.80,   # Slightly lower — harder terrain
        max_flip_rate=0.15,
        max_value_loss=15.0,
        frozen_weights={
            "stumble",              # Bug #28b
            "body_height_tracking", # Bug #22
        },
    ),
}

# Ordered phase progression
PHASE_ORDER = ["flat", "transition", "robust_easy", "robust"]


@dataclass
class CoachConfig:
    """Settings for the AI coach."""
    check_interval: int = 100          # consult AI every N iterations
    api_model: str = "claude-sonnet-4-20250514"
    max_weight_changes: int = 3        # max simultaneous changes (11k lesson)
    max_weight_delta_pct: float = 0.20 # max 20% change per adjustment
    max_weight_delta_abs: float = 0.5  # absolute cap for near-zero weights
    emergency_value_loss: float = 100.0
    emergency_smoothness: float = -10000.0
    nan_rollback: bool = True
    decision_log_path: str = "ai_coach_decisions.jsonl"
    max_stall_iters: int = 500         # plateau detection threshold
    history_window: int = 200          # rolling metric history size
    decision_history: int = 5          # past decisions sent to coach

    # Deferred coach activation — silent for N iters, then passive, then active
    activation_threshold: int = 0      # 0 = immediate, >0 = deferred
    lr_change_enabled: bool = True     # False = coach cannot change LR
    noise_change_enabled: bool = True  # False = coach cannot change noise

    # Weight bounds — absolute limits the coach cannot exceed
    # Positive rewards: (min, max), Negative penalties: (max_negative, min_negative)
    weight_bounds: dict = field(default_factory=lambda: {
        # Positive rewards
        "air_time":                (1.0, 10.0),
        "base_angular_velocity":   (1.0, 15.0),
        "base_linear_velocity":    (1.0, 15.0),
        "foot_clearance":          (1.0, 5.0),
        "gait":                    (0.5, 15.0),
        "velocity_modulation":     (0.5, 5.0),
        # Negative penalties
        "action_smoothness":       (-5.0, -0.05),
        "air_time_variance":       (-5.0, -0.1),
        "base_motion":             (-5.0, -0.1),
        "base_orientation":        (-10.0, -0.5),
        "body_scraping":           (-5.0, -0.5),
        "contact_force_smoothness":(-0.1, -0.001),
        "foot_slip":               (-3.0, -0.1),
        "joint_acc":               (-1.0, -0.001),
        "joint_pos":               (-1.0, -0.05),
        "joint_torques":           (-1.0, -0.001),
        "joint_vel":               (-1.0, -0.01),
        "dof_pos_limits":          (-10.0, -0.5),
        "terrain_relative_height": (-5.0, -0.5),
        "undesired_contacts":      (-5.0, -0.5),
        "vegetation_drag":         (-1.0, -0.001),
    })

    # Tighter bounds for mason_hybrid — centered on Mason's proven values
    mason_hybrid_bounds: dict = field(default_factory=lambda: {
        # Positive rewards — tight range around Mason's 5.0/10.0
        "air_time":                (2.0, 8.0),
        "base_angular_velocity":   (3.0, 7.0),
        "base_linear_velocity":    (3.0, 7.0),
        "foot_clearance":          (0.2, 1.5),
        "gait":                    (5.0, 12.0),
        # Negative penalties — tight range around Mason's values
        "action_smoothness":       (-3.0, -0.3),
        "air_time_variance":       (-3.0, -0.3),
        "base_motion":             (-4.0, -1.0),
        "base_orientation":        (-5.0, -1.5),
        "foot_slip":               (-2.0, -0.2),
        "joint_acc":               (-5e-3, -5e-5),
        "joint_pos":               (-1.0, -0.3),  # NEVER loosen below -0.3
        "joint_torques":           (-5e-3, -5e-5),
        "joint_vel":               (-5e-2, -5e-3),
        "dof_pos_limits":          (-5.0, -1.0),
        "terrain_relative_height": (-4.0, -1.0),
    })

    # Phase-specific LR ceilings — NEVER exceed these
    phase_lr_limits: dict = field(default_factory=lambda: {
        "flat": 3e-4,
        "transition": 3e-4,
        "robust_easy": 5e-5,
        "robust": 3e-5,
        "mason_hybrid": 1e-3,  # Managed by adaptive KL schedule
    })
