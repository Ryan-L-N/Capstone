"""PPO Runner Config for S2R expert and distillation training.

Based on Mason's proven [512, 256, 128] config with adaptive KL.
Expert training: 10000 iters. Distillation training: 8000 iters.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class SpotS2RExpertPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Mason PPO config for S2R expert training — 10000 iterations."""

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "spot_s2r_expert"
    store_code_state = False
    seed = 42

    logger = "tensorboard"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class SpotS2RDistillPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Mason PPO config for S2R distillation — 8000 iterations at 20 Hz."""

    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 100
    experiment_name = "spot_s2r_distill"
    store_code_state = False
    seed = 42

    logger = "tensorboard"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
