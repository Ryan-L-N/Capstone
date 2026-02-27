"""Spot PPO Runner Configuration — W&B enabled.

[1024, 512, 256] actor and critic, 20,480 envs, cosine LR annealing.
W&B integration via RSL-RL native logger field.

Template: 100hr_env_run/configs/ppo_cfg.py
Created for AI2C Tech Capstone — MS for Autonomy, February 2026
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class SpotPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for Spot on H100 NVL 96GB.

    W&B integration: set logger="wandb" for native RSL-RL W&B support.
    Cosine LR annealing is applied in the training script (not natively supported).
    """

    num_steps_per_env = 32
    max_iterations = 60000
    save_interval = 500
    experiment_name = "spot_robust_ppo"
    run_name = ""
    store_code_state = False
    seed = 42

    # W&B integration
    logger = "wandb"
    wandb_project = "capstone-quadruped-rl"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=[1024, 512, 256],
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
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
