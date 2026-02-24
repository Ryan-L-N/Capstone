"""Stage 1: PPO Configuration for Progressive Fine-Tuning.

Architecture: [512, 256, 128] — MUST match the 48hr rough policy checkpoint.

Attempt #4 hyperparameters — ultra-conservative for fine-tuning:
  - LR 1e-5 (10× lower than Attempt #3's 1e-4 which caused catastrophic forgetting)
  - clip 0.1 (halved — limits per-update policy change)
  - entropy 0.0 (disabled — noise std is permanently frozen at 0.65)
  - 3 learning epochs (reduced from 5 — fewer gradient steps per iteration)
  - KL target 0.005 (tighter — prevents large policy divergence)

Attempt #3 failure: LR=1e-4, clip=0.2, entropy=0.005, epochs=5 caused the actor
to catastrophically forget within 30 iterations of unfreezing (episode length
206 → 2.5 steps). The PPO updates were too aggressive for fine-tuning.

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

    Attempt #4 hyperparameters — ultra-conservative:
    - LR 1e-5 (was 1e-4 in Attempt #3 — caused catastrophic forgetting at unfreeze)
    - clip 0.1 (was 0.2 — halved to limit per-update ratio change)
    - entropy 0.0 (was 0.005 — disabled because noise std is permanently frozen)
    - 3 learning epochs (was 5 — fewer gradient steps per PPO iteration)
    - KL target 0.005 (was 0.008 — tighter adaptive KL constraint)
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
        clip_param=0.1,            # Attempt #4: halved (was 0.2) — less per-update change
        entropy_coef=0.0,          # Attempt #4: disabled (was 0.005) — std is permanently frozen
        num_learning_epochs=3,     # Attempt #4: reduced (was 5) — fewer gradient steps
        num_mini_batches=8,        # 16,384 envs * 24 steps / 8 = 49K per mini-batch
        learning_rate=1.0e-5,      # Attempt #4: 10× lower (was 1e-4) — prevent catastrophic forgetting
        schedule="adaptive",       # Adaptive KL — prevents destructive updates
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,          # Attempt #4: tighter (was 0.008) — smaller policy steps
        max_grad_norm=1.0,
    )
