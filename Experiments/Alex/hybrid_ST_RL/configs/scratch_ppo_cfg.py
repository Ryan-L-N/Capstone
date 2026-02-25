"""Attempt 6: PPO Configuration for From-Scratch Training.

Standard from-scratch hyperparameters — same as the original 48hr rough policy.
No ultra-conservative settings needed because there's no fine-tuning surgery.
Actor and critic train together from random initialization.

Architecture: [512, 256, 128] — same as 48hr policy for compatibility with
future teacher-student distillation (Stage 2).

Changes from Attempt 5:
  - Enabled actor/critic observation normalization (was False — critical fix)
  - Reduced init_noise_std from 1.0 to 0.5 (less random flailing at start)

Created for AI2C Tech Capstone — hybrid_ST_RL Attempt 6, February 2026
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
    - init_noise_std 0.5 (reduced from 1.0 — less flailing, faster convergence)
    - Architecture [512, 256, 128] (same as 48hr for Stage 2 compatibility)
    """

    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 500
    experiment_name = "spot_scratch_terrain"
    run_name = "attempt6"
    store_code_state = False
    seed = 42

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,  # Reduced from 1.0 — less flailing, faster convergence
        actor_obs_normalization=True,   # ENABLED — critical for heterogeneous obs scales
        critic_obs_normalization=True,  # ENABLED — critic needs clean inputs too
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
