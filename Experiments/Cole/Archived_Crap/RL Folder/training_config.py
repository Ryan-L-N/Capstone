"""
RSL-RL Training Configuration for Spot Obstacle-Aware Navigation
=================================================================

Hyperparameters for PPO training using RSL-RL framework.
Optimized for quadruped locomotion with obstacle interaction.

References:
- RSL-RL: https://github.com/leggedrobotics/rsl_rl
- Spot on Isaac Sim: NVIDIA Isaac Sim documentation
- PPO: Schulman et al. 2017

Author: Cole (MS for Autonomy Project)
Date: February 2026
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PPOConfig:
    """PPO algorithm hyperparameters."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # ALGORITHM
    # ─────────────────────────────────────────────────────────────────────────
    clip_param: float = 0.2                  # PPO clipping parameter
    ppo_epoch: int = 8                       # Number of PPO epochs per update
    num_mini_batches: int = 4                # Mini-batches per epoch
    value_loss_coef: float = 1.0             # Value function loss coefficient
    entropy_coef: float = 0.01               # Entropy regularization coefficient
    gamma: float = 0.99                      # Discount factor
    lam: float = 0.95                        # GAE lambda
    
    # ─────────────────────────────────────────────────────────────────────────
    # NETWORK ARCHITECTURE
    # ─────────────────────────────────────────────────────────────────────────
    actor_hidden_dims: tuple = (512, 256, 128)   # Actor MLP layers (match locomotion pre-training)
    critic_hidden_dims: tuple = (256, 256, 128)  # Critic MLP layers
    activation: str = "elu"                       # Activation function
    
    # ─────────────────────────────────────────────────────────────────────────
    # OPTIMIZATION
    # ─────────────────────────────────────────────────────────────────────────
    learning_rate: float = 5e-5              # Adam learning rate (conservative for stable learning)
    schedule: str = "adaptive"               # "adaptive" or "fixed"
    desired_kl: float = 0.01                 # Target KL divergence (adaptive)
    max_grad_norm: float = 1.0               # Gradient clipping
    
    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING
    # ─────────────────────────────────────────────────────────────────────────
    num_learning_iterations: int = 5000      # Total training iterations
    num_steps_per_env: int = 600             # Steps collected per env per iter (increased for longer episodes)
    max_episode_length: int = 3000           # Max steps per episode (60 sec @ 50Hz - enough time for 20m waypoints)
    
    # ─────────────────────────────────────────────────────────────────────────
    # NORMALIZATION
    # ─────────────────────────────────────────────────────────────────────────
    normalize_observations: bool = True      # Normalize observation inputs
    normalize_advantages: bool = True        # Normalize advantages
    normalize_values: bool = True            # Normalize value targets
    
    # ─────────────────────────────────────────────────────────────────────────
    # LOGGING
    # ─────────────────────────────────────────────────────────────────────────
    save_interval: int = 50                  # Save model every N iterations
    log_interval: int = 1                    # Log statistics every N iterations


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # SIMULATION
    # ─────────────────────────────────────────────────────────────────────────
    num_envs: int = 4096                     # Parallel environments (adjust for GPU)
    sim_device: str = "cuda:0"               # Simulation device
    rl_device: str = "cuda:0"                # RL computation device
    physics_dt: float = 1.0 / 500.0          # Physics timestep (500 Hz)
    control_dt: float = 1.0 / 50.0           # Control frequency (50 Hz)
    decimation: int = 10                     # Control decimation (500/50 = 10)
    
    # ─────────────────────────────────────────────────────────────────────────
    # ARENA
    # ─────────────────────────────────────────────────────────────────────────
    arena_radius: float = 25.0               # Arena radius (meters)
    num_obstacles: int = 40                  # Obstacles per episode
    num_small_obstacles: int = 100           # Small static obstacles
    num_waypoints: int = 25                  # Waypoints per episode
    waypoint_spacing_min: float = 40.0       # Min distance between waypoints
   
    # ─────────────────────────────────────────────────────────────────────────
    # SPOT CONFIGURATION
    # ─────────────────────────────────────────────────────────────────────────
    spot_mass: float = 32.7                  # Spot mass (kg)
    spot_start_height: float = 0.7           # Initial height (meters)
    fall_threshold: float = 0.3              # Fall detection height
    
    # ─────────────────────────────────────────────────────────────────────────
    # ACTION SPACE
    # ─────────────────────────────────────────────────────────────────────────
    action_scale: float = 1.0                # Scale factor for actions
    clip_actions: float = 1.0                # Clip actions to [-clip, clip]
    
    # Action limits
    max_forward_vel: float = 1.5             # Max forward velocity (m/s)
    max_lateral_vel: float = 0.5             # Max lateral velocity (m/s)
    max_angular_vel: float = 1.0             # Max angular velocity (rad/s)
    
    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVATION SPACE
    # ─────────────────────────────────────────────────────────────────────────
    num_nearest_obstacles: int = 5           # Nearest obstacles in observation
    observation_noise: float = 0.0           # Add noise to observations (disabled)
    
    # ─────────────────────────────────────────────────────────────────────────
    # EPISODE TERMINATION
    # ─────────────────────────────────────────────────────────────────────────
    max_episode_length_s: float = 300.0      # Max episode duration (seconds)
    terminate_on_fall: bool = True           # End episode if Spot falls
    terminate_on_completion: bool = True     # End episode when all waypoints reached


@dataclass
class RewardConfig:
    """Reward function weights (matches RewardWeights in spot_rl_env.py)."""
    
    # Progress
    waypoint_reached: float = 100.0
    distance_reduction: float = 2.0
    
    # Locomotion
    forward_locomotion: float = 1.0
    lateral_penalty: float = 0.5
    backward_penalty: float = 1.0
    
    # Stability
    stability_reward: float = 0.2
    height_deviation: float = 1.0
    
    # Obstacle interaction
    successful_nudge: float = 5.0
    failed_nudge_penalty: float = 2.0
    smart_bypass: float = 3.0
    collision_penalty: float = 0.5
    
    # Energy
    action_smoothness: float = 0.1
    energy_penalty: float = 0.05
    
    # Terminal
    fall_penalty: float = 100.0
    timeout_penalty: float = 10.0
    time_penalty: float = 1.0  # Penalty per second (scaled by dt each step)


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    
    # Sub-configurations
    ppo: PPOConfig = field(default_factory=PPOConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    
    # ─────────────────────────────────────────────────────────────────────────
    # EXPERIMENT TRACKING
    # ─────────────────────────────────────────────────────────────────────────
    experiment_name: str = "spot_obstacle_nav"
    run_name: Optional[str] = None           # Auto-generated if None
    seed: int = 42                           # Random seed
    
    # Directories
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    video_dir: str = "./videos"
    
    # ─────────────────────────────────────────────────────────────────────────
    # EVALUATION
    # ─────────────────────────────────────────────────────────────────────────
    eval_interval: int = 100                 # Evaluate every N iterations
    num_eval_episodes: int = 10              # Episodes per evaluation
    record_video: bool = True                # Record evaluation videos
    
    # ─────────────────────────────────────────────────────────────────────────
    # DEBUGGING
    # ─────────────────────────────────────────────────────────────────────────
    debug_mode: bool = False                 # Enable debug prints
    profile_training: bool = False           # Profile performance
    
    
# ═════════════════════════════════════════════════════════════════════════════
# PRE-CONFIGURED TRAINING PROFILES
# ═════════════════════════════════════════════════════════════════════════════

def get_default_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()


def get_debug_config() -> TrainingConfig:
    """Get configuration for debugging (fewer envs, more logging)."""
    config = TrainingConfig()
    config.env.num_envs = 16
    config.ppo.num_learning_iterations = 100
    config.ppo.save_interval = 10
    config.debug_mode = True
    return config


def get_fast_config() -> TrainingConfig:
    """Get configuration for fast prototyping (aggressive learning)."""
    config = TrainingConfig()
    config.ppo.learning_rate = 1e-3
    config.ppo.clip_param = 0.3
    config.ppo.ppo_epoch = 4
    config.env.num_envs = 1024
    return config


def get_stable_config() -> TrainingConfig:
    """Get configuration for stable, conservative training."""
    config = TrainingConfig()
    config.ppo.learning_rate = 1e-4
    config.ppo.clip_param = 0.15
    config.ppo.ppo_epoch = 10
    config.ppo.entropy_coef = 0.005
    config.env.num_envs = 8192
    return config


def get_high_performance_config() -> TrainingConfig:
    """Get configuration for high-performance training (GPU clusters)."""
    config = TrainingConfig()
    config.env.num_envs = 16384
    config.ppo.num_learning_iterations = 10000
    config.ppo.num_mini_batches = 8
    return config


# ═════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER SWEEP UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def create_sweep_configs(base_config: TrainingConfig,
                        param_name: str,
                        param_values: list) -> list:
    """
    Create multiple configs for hyperparameter sweep.
    
    Args:
        base_config: Base configuration
        param_name: Parameter to sweep (e.g., "ppo.learning_rate")
        param_values: List of values to try
    
    Returns:
        List of TrainingConfig objects
    """
    configs = []
    
    for value in param_values:
        import copy
        config = copy.deepcopy(base_config)
        
        # Parse nested parameter name
        parts = param_name.split('.')
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
        
        # Set run name to include parameter
        config.run_name = f"{param_name}_{value}"
        
        configs.append(config)
    
    return configs


# ═════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Default configuration
    config = get_default_config()
    print(f"Default config: {config.env.num_envs} envs, {config.ppo.num_learning_iterations} iterations")
    
    # Debug configuration
    debug_config = get_debug_config()
    print(f"Debug config: {debug_config.env.num_envs} envs")
    
    # Hyperparameter sweep example
    lr_sweep = create_sweep_configs(
        get_default_config(),
        "ppo.learning_rate",
        [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    )
    print(f"Created {len(lr_sweep)} configs for learning rate sweep")
