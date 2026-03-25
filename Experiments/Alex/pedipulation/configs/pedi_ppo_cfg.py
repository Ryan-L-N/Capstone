"""Pedipulation PPO Runner Config — matches hybrid_nocoach [512,256,128].

Architecture MUST match hybrid_nocoach_19999.pt for weight surgery:
  Actor:  240 → [512, 256, 128] → 12
  Critic: 240 → [512, 256, 128] → 1

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PedipulationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for pedipulation — [512,256,128] to match hybrid baseline."""

    num_steps_per_env = 32
    max_iterations = 15000
    save_interval = 100
    experiment_name = "spot_pedi_ppo"
    store_code_state = False
    seed = 42

    logger = "tensorboard"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=8,
        num_mini_batches=64,
        learning_rate=1.0e-3,
        schedule="fixed",         # Cosine annealing applied externally in train_pedi.py
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
