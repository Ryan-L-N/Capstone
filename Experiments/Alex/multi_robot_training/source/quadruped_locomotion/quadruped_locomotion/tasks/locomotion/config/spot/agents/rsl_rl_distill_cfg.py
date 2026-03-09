"""Shared Distillation PPO Configuration.

Used for Phase 2b student distillation. Lower learning rate, same architecture.
Parameterized by robot via experiment_name prefix.

Template: hybrid_ST_RL/train_distill.py
Created for AI2C Tech Capstone — MS for Autonomy, February 2026
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class DistillPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for student-teacher distillation (Phase 2b).

    Lower learning rate (5e-5) for fine-tuning the student with BC loss.
    """

    num_steps_per_env = 32
    max_iterations = 10000
    save_interval = 500
    experiment_name = "distill"  # Overridden per-robot in train_distill.py
    run_name = "stage2b_distill"
    store_code_state = False
    seed = 42

    # W&B integration
    logger = "wandb"
    wandb_project = "capstone-quadruped-rl"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
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
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=64,
        learning_rate=5.0e-5,  # Very low for distillation
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
