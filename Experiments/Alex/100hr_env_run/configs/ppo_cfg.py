"""PPO configuration for H100 NVL 96GB multi-terrain training.

Key changes from 48hr SpotRough48hPPORunnerCfg:
  - 65,536 envs — saturates H100 NVL 96GB (~57 GB VRAM)
  - 32 steps per env (was 24) — longer rollouts for terrain transitions
  - 30,000 max iterations — 3.2x more data/iter compensates for fewer iters
  - [1024, 512, 256] network (was [512, 256, 128]) — 4x more params
  - 1.0 init_noise_std (was 0.8) — more exploration for diverse terrain
  - 8 learning epochs (was 5) — more gradient updates per batch
  - 64 mini-batches — scaled for 65K envs (32K samples per mini-batch)

Learning rate: starts at 1e-3, cosine-anneals to 1e-5.
Note: RSL-RL doesn't natively support cosine annealing, so the training
script implements it by overriding lr each iteration. The config here
sets the starting LR.

Created for AI2C Tech Capstone — MS for Autonomy, February 2026
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class Spot100hrPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config tuned for H100 NVL 96GB with 65,536 envs.

    Estimated training scale:
      - Steps per iteration: 65,536 × 32 = 2,097,152
      - Estimated throughput: ~150,000+ steps/sec
      - Total timesteps at 30K iters: ~63 billion
      - Wall-clock estimate: ~5-6 days
    """

    num_steps_per_env = 32            # was 24 — longer rollouts
    max_iterations = 30000            # 30K iters × 3.2x data/iter
    save_interval = 500               # ~60 checkpoints total
    experiment_name = "spot_100hr_robust"
    run_name = "multi_terrain_v1"
    store_code_state = False
    seed = 42

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,           # was 0.8 — more exploration for diverse terrain
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[1024, 512, 256],   # was [512, 256, 128] — 4x capacity
        critic_hidden_dims=[1024, 512, 256],  # was [512, 256, 128]
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,            # was 0.008 — more exploration early
        num_learning_epochs=8,        # was 5 — more gradient updates
        num_mini_batches=64,          # was 8 — scaled for 65,536 envs
        learning_rate=1.0e-3,         # starting LR (cosine annealed in training script)
        schedule="adaptive",          # base schedule; training script overrides with cosine
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
