"""Configuration for 6-expert distillation pipeline.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class S2RDistillConfig:
    """Hyperparameters for 6-expert terrain distillation."""

    # Expert checkpoint paths (populated from CLI args)
    expert_paths: Dict[str, str] = field(default_factory=lambda: {
        "friction": "",
        "stairs_up": "",
        "stairs_down": "",
        "boulders": "",
        "slopes": "",
        "mixed_rough": "",
    })

    # Network architecture (must match experts)
    num_obs: int = 235
    num_actions: int = 12
    actor_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])

    # Router
    router_hidden_dim: int = 64
    router_lr: float = 1e-4       # Separate, lower LR for gate_net
    num_experts: int = 6

    # Distillation
    alpha_start: float = 0.8      # Initial: trust experts
    alpha_end: float = 0.2        # Final: trust PPO
    kl_weight: float = 0.1        # KL divergence weight in loss
    distill_batch_size: int = 8192  # Samples per distillation step

    # Student initialization
    init_from: str = "friction"   # Best general expert to init from
    critic_warmup_iters: int = 300

    # Training
    max_iterations: int = 8000
    num_envs: int = 4096
    save_interval: int = 100

    # Noise
    min_noise_std: float = 0.3
    max_noise_std: float = 0.5

    # S2R wrappers (active during distillation)
    action_delay_steps: int = 2
    obs_delay_steps: int = 1
    sensor_dropout_rate: float = 0.03  # Lighter than expert training (0.05)
    sensor_drift_rate: float = 0.001   # Lighter than expert training (0.002)
