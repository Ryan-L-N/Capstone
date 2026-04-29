"""Multi-expert distillation configuration.

All hyperparameters in one place. No Isaac Lab imports needed.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from dataclasses import dataclass, field


@dataclass
class DistillConfig:
    """Hyperparameters for multi-expert distillation."""

    # -- Expert checkpoints --
    friction_expert_path: str = ""
    obstacle_expert_path: str = ""

    # -- Student initialization --
    init_from: str = "friction"  # "friction", "obstacle", or "scratch"
    critic_warmup_iters: int = 300

    # -- Network architecture (must match experts) --
    actor_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    num_obs: int = 235          # 187 height_scan + 48 proprioceptive
    num_actions: int = 12
    height_scan_dims: int = 187

    # -- Distillation loss --
    alpha_start: float = 0.8    # initial distillation weight (high = trust experts)
    alpha_end: float = 0.2      # final distillation weight (low = trust PPO)
    kl_weight: float = 0.1      # KL vs MSE balance within distillation loss

    # -- Terrain routing --
    roughness_threshold: float = 0.005  # height_scan variance gate
    routing_temperature: float = 0.005  # sigmoid sharpness (lower = harder gate)

    # -- Training --
    max_iterations: int = 5000
    num_envs: int = 4096
    save_interval: int = 100
    experiment_name: str = "spot_distill"

    # -- Safety --
    min_noise_std: float = 0.3
    max_noise_std: float = 0.5

    # -- Distillation batch --
    distill_batch_size: int = 8192  # samples per distillation step
