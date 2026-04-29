"""Mason Hybrid PPO Runner Config — Mason's network + hyperparams.

[512, 256, 128] actor/critic (3x smaller than our [1024, 512, 256]).
Adaptive KL-based LR schedule (not cosine annealing).
Proven to train effectively with 4096 envs.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class SpotMasonHybridPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Mason's proven PPO config — smaller network, adaptive LR."""

    num_steps_per_env = 24       # Mason's (ours was 32)
    max_iterations = 20000       # Mason's (ours was 60000)
    save_interval = 100          # Our lesson: 500 is too infrequent
    experiment_name = "spot_hybrid_ppo"
    store_code_state = False
    seed = 42

    # TensorBoard only (no wandb on H100)
    logger = "tensorboard"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,              # Mason's (ours was 0.5)
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],   # Mason's (ours was [1024, 512, 256])
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,             # Mason's (ours was 1.0)
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0025,             # Mason's (ours was 0.01)
        num_learning_epochs=5,           # Mason's (ours was 8)
        num_mini_batches=4,              # Mason's (ours was 64)
        learning_rate=1.0e-3,
        schedule="adaptive",             # Mason's KL-based (ours was fixed + cosine)
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
