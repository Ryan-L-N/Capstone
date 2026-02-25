"""Attempt 5: PPO Configuration for From-Scratch Training.

Standard from-scratch hyperparameters — same as the original 48hr rough policy.
No ultra-conservative settings needed because there's no fine-tuning surgery.
Actor and critic train together from random initialization.

Architecture: [512, 256, 128] — same as 48hr policy for compatibility with
future teacher-student distillation (Stage 2).

Created for AI2C Tech Capstone — hybrid_ST_RL Attempt 5, February 2026
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class SpotScratchPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for training Spot from scratch on terrain curriculum.

    Standard from-scratch hyperparameters:
    - LR 1e-3 with adaptive KL (same as 48hr policy)
    - clip 0.2, entropy 0.005, 5 epochs (standard PPO)
    - init_noise_std 1.0 (converges naturally during training)
    - Architecture [512, 256, 128] (same as 48hr for Stage 2 compatibility)
    """

    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 500
    experiment_name = "spot_scratch_terrain"
    run_name = "attempt5"
    store_code_state = False
    seed = 42

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # From scratch — will converge naturally
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,            # Standard PPO
        entropy_coef=0.005,        # Standard exploration bonus
        num_learning_epochs=5,     # Standard
        num_mini_batches=4,        # 16,384 envs * 24 steps / 4 = 98K per mini-batch
        learning_rate=1.0e-3,      # Standard from-scratch
        schedule="adaptive",       # Adaptive KL — adjusts LR automatically
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,           # Standard — looser for from-scratch exploration
        max_grad_norm=1.0,
    )
