"""Spot Navigation PPO Runner Configuration.

Smaller network than loco policy since nav decisions are higher-level.
Fewer envs (512-1024) due to camera rendering overhead.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class SpotNavPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for Spot navigation policy.

    Smaller network [256, 128, 64] since nav policy makes higher-level
    steering decisions, not low-level joint control.
    """

    num_steps_per_env = 64  # Longer rollouts for goal-reaching episodes
    max_iterations = 20000
    save_interval = 100
    experiment_name = "spot_nav_ppo"
    run_name = ""
    store_code_state = False
    seed = 42

    logger = "tensorboard"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=4,
        num_mini_batches=16,  # Fewer envs = fewer mini-batches
        learning_rate=3e-4,  # Higher LR for smaller network
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
