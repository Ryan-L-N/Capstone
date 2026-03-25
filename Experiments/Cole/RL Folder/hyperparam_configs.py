"""
Hyperparameter Tuning Configurations for Spot PPO Training
===========================================================
Defines different hyperparameter sets to experiment with.

Author: Cole
Date: February 2026
"""

from training_config import PPOConfig, EnvironmentConfig, RewardConfig, TrainingConfig
from dataclasses import replace


# =============================================================================
# BASELINE CONFIGURATION (Current Best)
# =============================================================================
def get_baseline_config():
    """Baseline configuration - current working setup."""
    config = TrainingConfig()
    config.experiment_name = "baseline"
    return config


# =============================================================================
# LEARNING RATE EXPERIMENTS
# =============================================================================
def get_lr_low_config():
    """Lower learning rate for more stable learning."""
    config = TrainingConfig()
    config.experiment_name = "lr_1e4"
    config.ppo.learning_rate = 1e-4  # Lower than baseline (3e-4)
    return config


def get_lr_high_config():
    """Higher learning rate for faster learning."""
    config = TrainingConfig()
    config.experiment_name = "lr_1e3"
    config.ppo.learning_rate = 1e-3  # Higher than baseline (3e-4)
    return config


def get_lr_very_low_config():
    """Very low learning rate for maximum stability."""
    config = TrainingConfig()
    config.experiment_name = "lr_5e5"
    config.ppo.learning_rate = 5e-5  # Very conservative
    return config


# =============================================================================
# NETWORK ARCHITECTURE EXPERIMENTS
# =============================================================================
def get_small_network_config():
    """Smaller network - faster training, less capacity."""
    config = TrainingConfig()
    config.experiment_name = "small_net"
    config.ppo.actor_hidden_dims = (128, 128, 64)
    config.ppo.critic_hidden_dims = (128, 128, 64)
    return config


def get_large_network_config():
    """Larger network - more capacity, slower training."""
    config = TrainingConfig()
    config.experiment_name = "large_net"
    config.ppo.actor_hidden_dims = (512, 512, 256)
    config.ppo.critic_hidden_dims = (512, 512, 256)
    return config


def get_deep_network_config():
    """Deeper network with more layers."""
    config = TrainingConfig()
    config.experiment_name = "deep_net"
    config.ppo.actor_hidden_dims = (256, 256, 256, 128)
    config.ppo.critic_hidden_dims = (256, 256, 256, 128)
    return config


# =============================================================================
# PPO ALGORITHM EXPERIMENTS
# =============================================================================
def get_tight_clip_config():
    """Tighter clipping for more conservative updates."""
    config = TrainingConfig()
    config.experiment_name = "clip_0p1"
    config.ppo.clip_param = 0.1  # Tighter than baseline (0.2)
    return config


def get_loose_clip_config():
    """Looser clipping for larger policy updates."""
    config = TrainingConfig()
    config.experiment_name = "clip_0p3"
    config.ppo.clip_param = 0.3  # Looser than baseline (0.2)
    return config


def get_high_entropy_config():
    """Higher entropy bonus for more exploration."""
    config = TrainingConfig()
    config.experiment_name = "entropy_0p05"
    config.ppo.entropy_coef = 0.05  # Higher than baseline (0.01)
    return config


def get_low_entropy_config():
    """Lower entropy bonus - focus on exploitation."""
    config = TrainingConfig()
    config.experiment_name = "entropy_0p001"
    config.ppo.entropy_coef = 0.001  # Lower than baseline (0.01)
    return config


def get_more_ppo_epochs_config():
    """More PPO epochs per update."""
    config = TrainingConfig()
    config.experiment_name = "epochs_15"
    config.ppo.ppo_epoch = 15  # More than baseline (8)
    return config


# =============================================================================
# REWARD TUNING EXPERIMENTS
# =============================================================================
def get_high_distance_reward_config():
    """Emphasize distance reduction more."""
    config = TrainingConfig()
    config.experiment_name = "dist_reward_5"
    config.rewards.distance_reduction = 5.0  # Higher than baseline (2.0)
    return config


def get_low_time_penalty_config():
    """Reduce time penalty to allow longer exploration."""
    config = TrainingConfig()
    config.experiment_name = "time_penalty_0p5"
    config.rewards.time_penalty = 0.5  # Per second (lower than baseline 1.0)
    return config


def get_waypoint_focused_config():
    """Emphasize waypoint reaching."""
    config = TrainingConfig()
    config.experiment_name = "waypoint_focused"
    config.rewards.waypoint_reached = 200.0  # Higher than baseline (100.0)
    config.rewards.distance_reduction = 5.0  # Also increase this
    return config


def get_stability_focused_config():
    """Emphasize stable locomotion."""
    config = TrainingConfig()
    config.experiment_name = "stability_focused"
    config.rewards.stability_reward = 0.5  # Higher than baseline (0.2)
    config.rewards.height_deviation = 2.0  # Higher penalty for height deviation
    return config


# =============================================================================
# REWARD WEIGHT TUNING (Based on RL_Development.py scoring)
# =============================================================================
def get_reward_low_waypoint_bonus_config():
    """Lower waypoint bonus (10.0 vs baseline 15.0)."""
    config = TrainingConfig()
    config.experiment_name = "waypoint_10"
    
    # Based on simple_focused structure
    config.rewards.waypoint_reached = 10.0
    config.rewards.distance_reduction = 1.0
    config.rewards.time_penalty = 1.0  # Per second
    config.rewards.fall_penalty = 100.0
    
    # Disable auxiliary rewards
    config.rewards.forward_locomotion = 0.0
    config.rewards.lateral_penalty = 0.0
    config.rewards.backward_penalty = 0.0
    config.rewards.stability_reward = 0.0
    config.rewards.height_deviation = 0.0
    return config


def get_reward_high_waypoint_bonus_config():
    """Higher waypoint bonus (20.0 vs baseline 15.0)."""
    config = TrainingConfig()
    config.experiment_name = "waypoint_20"
    
    config.rewards.waypoint_reached = 20.0
    config.rewards.distance_reduction = 1.0
    config.rewards.time_penalty = 1.0  # Per second
    config.rewards.fall_penalty = 100.0
    
    # Disable auxiliary rewards
    config.rewards.forward_locomotion = 0.0
    config.rewards.lateral_penalty = 0.0
    config.rewards.backward_penalty = 0.0
    config.rewards.stability_reward = 0.0
    config.rewards.height_deviation = 0.0
    return config


def get_reward_very_high_waypoint_bonus_config():
    """Very high waypoint bonus (30.0 vs baseline 15.0)."""
    config = TrainingConfig()
    config.experiment_name = "waypoint_30"
    
    config.rewards.waypoint_reached = 30.0
    config.rewards.distance_reduction = 1.0
    config.rewards.time_penalty = 1.0  # Per second
    config.rewards.fall_penalty = 100.0
    
    # Disable auxiliary rewards
    config.rewards.forward_locomotion = 0.0
    config.rewards.lateral_penalty = 0.0
    config.rewards.backward_penalty = 0.0
    config.rewards.stability_reward = 0.0
    config.rewards.height_deviation = 0.0
    return config


def get_reward_low_time_penalty_config():
    """Lower time penalty (0.5/sec vs baseline 1.0/sec)."""
    config = TrainingConfig()
    config.experiment_name = "time_0p5"
    
    config.rewards.waypoint_reached = 15.0
    config.rewards.distance_reduction = 1.0
    config.rewards.time_penalty = 0.5  # Less pressure (per second)
    config.rewards.fall_penalty = 100.0
    
    # Disable auxiliary rewards
    config.rewards.forward_locomotion = 0.0
    config.rewards.lateral_penalty = 0.0
    config.rewards.backward_penalty = 0.0
    config.rewards.stability_reward = 0.0
    config.rewards.height_deviation = 0.0
    return config


def get_reward_high_time_penalty_config():
    """Higher time penalty (1.5/sec vs baseline 1.0/sec)."""
    config = TrainingConfig()
    config.experiment_name = "time_1p5"
    
    config.rewards.waypoint_reached = 15.0
    config.rewards.distance_reduction = 1.0
    config.rewards.time_penalty = 1.5  # More pressure (per second)
    config.rewards.fall_penalty = 100.0
    
    # Disable auxiliary rewards
    config.rewards.forward_locomotion = 0.0
    config.rewards.lateral_penalty = 0.0
    config.rewards.backward_penalty = 0.0
    config.rewards.stability_reward = 0.0
    config.rewards.height_deviation = 0.0
    return config


def get_reward_high_distance_shaping_config():
    """Higher distance shaping (2.0 vs baseline 1.0)."""
    config = TrainingConfig()
    config.experiment_name = "dist_2p0"
    
    config.rewards.waypoint_reached = 15.0
    config.rewards.distance_reduction = 2.0  # More dense shaping
    config.rewards.time_penalty = 1.0  # Per second
    config.rewards.fall_penalty = 100.0
    
    # Disable auxiliary rewards
    config.rewards.forward_locomotion = 0.0
    config.rewards.lateral_penalty = 0.0
    config.rewards.backward_penalty = 0.0
    config.rewards.stability_reward = 0.0
    config.rewards.height_deviation = 0.0
    return config


def get_reward_low_distance_shaping_config():
    """Lower distance shaping (0.5 vs baseline 1.0)."""
    config = TrainingConfig()
    config.experiment_name = "dist_0p5"
    
    config.rewards.waypoint_reached = 15.0
    config.rewards.distance_reduction = 0.5  # Less dense shaping
    config.rewards.time_penalty = 1.0  # Per second
    config.rewards.fall_penalty = 100.0
    
    # Disable auxiliary rewards
    config.rewards.forward_locomotion = 0.0
    config.rewards.lateral_penalty = 0.0
    config.rewards.backward_penalty = 0.0
    config.rewards.stability_reward = 0.0
    config.rewards.height_deviation = 0.0
    return config


def get_reward_low_fall_penalty_config():
    """Lower fall penalty (50.0 vs baseline 100.0)."""
    config = TrainingConfig()
    config.experiment_name = "fall_50"
    
    config.rewards.waypoint_reached = 15.0
    config.rewards.distance_reduction = 1.0
    config.rewards.time_penalty = 1.0  # Per second
    config.rewards.fall_penalty = 50.0  # Less harsh
    
    # Disable auxiliary rewards
    config.rewards.forward_locomotion = 0.0
    config.rewards.lateral_penalty = 0.0
    config.rewards.backward_penalty = 0.0
    config.rewards.stability_reward = 0.0
    config.rewards.height_deviation = 0.0
    return config


def get_reward_high_fall_penalty_config():
    """Higher fall penalty (150.0 vs baseline 100.0)."""
    config = TrainingConfig()
    config.experiment_name = "fall_150"
    
    config.rewards.waypoint_reached = 15.0
    config.rewards.distance_reduction = 1.0
    config.rewards.time_penalty = 1.0  # Per second
    config.rewards.fall_penalty = 150.0  # More harsh
    
    # Disable auxiliary rewards
    config.rewards.forward_locomotion = 0.0
    config.rewards.lateral_penalty = 0.0
    config.rewards.backward_penalty = 0.0
    config.rewards.stability_reward = 0.0
    config.rewards.height_deviation = 0.0
    return config


# =============================================================================
# COMBINED OPTIMIZED CONFIGURATIONS
# =============================================================================
def get_aggressive_learning_config():
    """Fast learning with higher LR and exploration."""
    config = TrainingConfig()
    config.experiment_name = "aggressive"
    config.ppo.learning_rate = 1e-3
    config.ppo.entropy_coef = 0.05
    config.ppo.clip_param = 0.3
    return config


def get_conservative_learning_config():
    """Stable learning with lower LR and tight clipping."""
    config = TrainingConfig()
    config.experiment_name = "conservative"
    config.ppo.learning_rate = 5e-5
    config.ppo.entropy_coef = 0.001
    config.ppo.clip_param = 0.1
    config.ppo.ppo_epoch = 15
    return config


def get_balanced_tuned_config():
    """Balanced configuration based on initial observations."""
    config = TrainingConfig()
    config.experiment_name = "balanced_tuned"
    config.ppo.learning_rate = 1e-4  # Slightly lower than baseline
    config.ppo.actor_hidden_dims = (256, 256, 128)  # Keep baseline
    config.ppo.critic_hidden_dims = (256, 256, 128)
    config.ppo.clip_param = 0.2  # Keep baseline
    config.ppo.entropy_coef = 0.02  # Slightly higher for exploration
    config.ppo.ppo_epoch = 10  # Slightly more epochs
    config.rewards.distance_reduction = 3.0  # Increased
    config.rewards.waypoint_reached = 150.0  # Increased
    config.rewards.time_penalty = 0.005  # Reduced
    return config


def get_simple_focused_config():
    """Simplified reward structure based on RL_Development.py.
    
    Focuses on core objective with minimal parameters:
    - Waypoint bonus: 15.0 (same as RL_Development.py)
    - Distance shaping: 1.0 (dense progress signal)
    - Time penalty: 1.0 per second (efficiency pressure)
    - Fall penalty: 100.0 (terminal, large negative)
    
    This reduces hyperparameter complexity from 11+ to 4 core terms.
    """
    config = TrainingConfig()
    config.experiment_name = "simple_focused"
    
    # Core reward components only
    config.rewards.waypoint_reached = 15.0      # Match RL_Development.py
    config.rewards.distance_reduction = 1.0     # Dense waypoint progress
    config.rewards.time_penalty = 1.0           # Per second (scaled by dt)
    config.rewards.fall_penalty = 100.0         # Terminal penalty
    
    # Disable all auxiliary rewards
    config.rewards.forward_locomotion = 0.0
    config.rewards.lateral_penalty = 0.0
    config.rewards.backward_penalty = 0.0
    config.rewards.stability_reward = 0.0
    config.rewards.height_deviation = 0.0
    config.rewards.successful_nudge = 0.0
    config.rewards.failed_nudge_penalty = 0.0
    config.rewards.smart_bypass = 0.0
    config.rewards.collision_penalty = 0.0
    config.rewards.action_smoothness = 0.0
    config.rewards.energy_penalty = 0.0
    config.rewards.timeout_penalty = 0.0
    
    return config


# =============================================================================
# CONFIGURATION REGISTRY
# =============================================================================
CONFIGS = {
    # Baseline
    "baseline": get_baseline_config,
    "simple_focused": get_simple_focused_config,  # Simplified reward structure
    
    # Learning rate
    "lr_low": get_lr_low_config,
    "lr_high": get_lr_high_config,
    "lr_very_low": get_lr_very_low_config,
    
    # Network architecture
    "small_net": get_small_network_config,
    "large_net": get_large_network_config,
    "deep_net": get_deep_network_config,
    
    # PPO parameters
    "tight_clip": get_tight_clip_config,
    "loose_clip": get_loose_clip_config,
    "high_entropy": get_high_entropy_config,
    "low_entropy": get_low_entropy_config,
    "more_epochs": get_more_ppo_epochs_config,
    
    # Reward tuning (legacy)
    "high_dist_reward": get_high_distance_reward_config,
    "low_time_penalty": get_low_time_penalty_config,
    "waypoint_focused": get_waypoint_focused_config,
    "stability_focused": get_stability_focused_config,
    
    # Reward weight tuning (systematic)
    "waypoint_10": get_reward_low_waypoint_bonus_config,
    "waypoint_20": get_reward_high_waypoint_bonus_config,
    "waypoint_30": get_reward_very_high_waypoint_bonus_config,
    "time_0p5": get_reward_low_time_penalty_config,   # 0.5 per second
    "time_1p5": get_reward_high_time_penalty_config,  # 1.5 per second
    "dist_2p0": get_reward_high_distance_shaping_config,
    "dist_0p5": get_reward_low_distance_shaping_config,
    "fall_50": get_reward_low_fall_penalty_config,
    "fall_150": get_reward_high_fall_penalty_config,
    
    # Combined configs
    "aggressive": get_aggressive_learning_config,
    "conservative": get_conservative_learning_config,
    "balanced_tuned": get_balanced_tuned_config,
}


def get_config(name: str):
    """Get configuration by name."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]()


def list_configs():
    """List all available configurations."""
    print("Available Configurations:")
    print("=" * 60)
    
    categories = {
        "Baseline": ["baseline", "simple_focused"],
        "Learning Rate": ["lr_low", "lr_high", "lr_very_low"],
        "Network Architecture": ["small_net", "large_net", "deep_net"],
        "PPO Parameters": ["tight_clip", "loose_clip", "high_entropy", 
                          "low_entropy", "more_epochs"],
        "Reward Tuning (Legacy)": ["high_dist_reward", "low_time_penalty", 
                                   "waypoint_focused", "stability_focused"],
        "Reward Weights - Waypoint Bonus": ["waypoint_10", "waypoint_20", "waypoint_30"],
        "Reward Weights - Time Penalty": ["time_0p5", "time_1p5"],
        "Reward Weights - Distance Shaping": ["dist_0p5", "dist_2p0"],
        "Reward Weights - Fall Penalty": ["fall_50", "fall_150"],
        "Combined": ["aggressive", "conservative", "balanced_tuned"],
    }
    
    for category, configs in categories.items():
        print(f"\n{category}:")
        for cfg_name in configs:
            print(f"  - {cfg_name}")
    
    print("=" * 60)


if __name__ == "__main__":
    list_configs()
    
    print("\n\nExample: Baseline vs Balanced Tuned")
    print("=" * 60)
    
    baseline = get_baseline_config()
    tuned = get_balanced_tuned_config()
    
    print(f"\nBaseline:")
    print(f"  LR: {baseline.ppo.learning_rate}")
    print(f"  Clip: {baseline.ppo.clip_param}")
    print(f"  Entropy: {baseline.ppo.entropy_coef}")
    print(f"  Epochs: {baseline.ppo.ppo_epoch}")
    
    print(f"\nBalanced Tuned:")
    print(f"  LR: {tuned.ppo.learning_rate}")
    print(f"  Clip: {tuned.ppo.clip_param}")
    print(f"  Entropy: {tuned.ppo.entropy_coef}")
    print(f"  Epochs: {tuned.ppo.ppo_epoch}")
    print(f"  Distance reward: {tuned.rewards.distance_reduction}")
    print(f"  Waypoint reward: {tuned.rewards.waypoint_reached}")
