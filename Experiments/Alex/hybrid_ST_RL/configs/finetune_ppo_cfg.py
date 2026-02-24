"""Stage 1: PPO Configuration for Progressive Fine-Tuning.

Architecture: [512, 256, 128] — MUST match the 48hr rough policy checkpoint.
Learning rate: 1e-4 (3x lower than 48hr's 3e-4 for fine-tuning stability).
Entropy: 0.005 (lower than 48hr's 0.008 — warm start needs less exploration).

Created for AI2C Tech Capstone — hybrid_ST_RL, February 2026
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class SpotFinetunePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for fine-tuning the 48hr rough policy on 12 terrain types.

    Architecture [512, 256, 128] matches model_27500.pt exactly.
    Hyperparameters are conservative for fine-tuning:
    - LR 1e-4 (not 3e-4) to avoid catastrophic forgetting
    - KL target 0.008 (not 0.01) for tighter policy updates
    - Entropy 0.005 (not 0.008) since the policy already knows how to walk
    - init_noise_std 0.65 matches the checkpoint's converged action noise
    """

    num_steps_per_env = 24
    max_iterations = 25000
    save_interval = 500
    experiment_name = "spot_hybrid_st_rl"
    run_name = "stage1_finetune"
    store_code_state = False
    seed = 42

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.65,  # Match checkpoint's converged noise (was 0.8 in 48hr)
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],   # MUST match 48hr checkpoint
        critic_hidden_dims=[512, 256, 128],  # MUST match 48hr checkpoint
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,        # Lower than 48hr's 0.008 — warm start
        num_learning_epochs=5,     # Same as 48hr
        num_mini_batches=8,        # 16,384 envs * 24 steps / 8 = 49K per mini-batch
        learning_rate=1.0e-4,      # Lower than 48hr's 3e-4 for fine-tuning
        schedule="adaptive",       # Adaptive KL — prevents destructive updates
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,          # Tighter than 48hr's 0.01
        max_grad_norm=1.0,
    )
