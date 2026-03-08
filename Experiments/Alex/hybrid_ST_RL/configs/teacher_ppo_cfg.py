"""Stage 2a: Teacher PPO Configuration.

Same [512, 256, 128] architecture as Stage 1, but the first layer accepts
254 dims (235 standard + 19 privileged) instead of 235.

LR is very low (5e-5) since we're starting from a well-trained Stage 1 policy.

Created for AI2C Tech Capstone — hybrid_ST_RL, February 2026
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class SpotTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for teacher training with privileged observations.

    Architecture [512, 256, 128] with 254-dim input (235 + 19 privileged).
    Weight surgery needed to initialize from Stage 1 checkpoint (235-dim).
    """

    num_steps_per_env = 24
    max_iterations = 20000  # ~24hr at 8K envs
    save_interval = 500
    experiment_name = "spot_hybrid_st_rl"
    run_name = "stage2a_teacher"
    store_code_state = False
    seed = 42

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.65,
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
        entropy_coef=0.003,        # Very low — teacher should exploit
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=5.0e-5,      # Very low for fine-tuning
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.006,          # Very tight constraint
        max_grad_norm=1.0,
    )
