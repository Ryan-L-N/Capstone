"""RSL-RL PPO hyperparameters for Spot navigation training.

Uses isaaclab_rl wrapper classes for H100 compatibility.
The actual policy class is ActorCriticCNN, injected at runtime by train_nav.py.
"""

from __future__ import annotations

from isaaclab.utils import configclass

try:
    # Isaac Lab's RSL-RL wrapper (H100 version)
    from isaaclab_rl.rsl_rl import (
        RslRlOnPolicyRunnerCfg,
        RslRlPpoActorCriticCfg,
        RslRlPpoAlgorithmCfg,
    )
except ImportError:
    # Fallback for newer Isaac Lab versions
    from rsl_rl.runners import OnPolicyRunnerCfg as RslRlOnPolicyRunnerCfg
    from rsl_rl.algorithms import PPOCfg as RslRlPpoAlgorithmCfg
    from rsl_rl.modules import ActorCriticCfg as RslRlPpoActorCriticCfg


@configclass
class SpotNavPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for Spot navigation."""

    seed = 42
    num_steps_per_env = 128     # Longer rollouts for exploration
    max_iterations = 30000

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticCNN",  # Custom class injected at runtime
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=1e-4,  # Lower for CNN stability
        schedule="fixed",    # Cosine annealing handled externally
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    save_interval = 100
    experiment_name = "spot_nav_explore_ppo"
    logger = "tensorboard"
    empirical_normalization = False
