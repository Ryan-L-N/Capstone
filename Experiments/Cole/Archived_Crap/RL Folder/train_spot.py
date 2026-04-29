"""
Spot RL Training Script

Main training script integrating spot_rl_env with RSL-RL framework for parallel
multi-environment PPO training in Isaac Sim.

Usage:
    # Train with default configuration
    python train_spot.py
    
    # Train with specific profile
    python train_spot.py --config fast --num_envs 1024
    
    # Resume from checkpoint
    python train_spot.py --resume logs/checkpoints/model_2000.pt
    
    # Debug mode with rendering
    python train_spot.py --config debug --render

Author: Cole
Date: February 2026
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# RSL-RL imports (PPO framework from ETH Zurich)
try:
    from rsl_rl.algorithms import PPO
    from rsl_rl.modules import ActorCritic
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.env import VecEnv
except ImportError:
    print("ERROR: rsl_rl not found. Install with: pip install rsl-rl")
    sys.exit(1)

# Isaac Sim imports
from omni.isaac.kit import SimulationApp

# Import our modules (after Isaac Sim initialization)
from spot_rl_env import SpotRLEnv
from training_config import (
    get_default_config,
    get_config_by_name,
    TrainingConfig,
)


class SpotRLRunner:
    """
    Training runner for Spot RL that integrates Isaac Sim environments
    with RSL-RL's PPO algorithm.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        log_dir: str,
        checkpoint_path: Optional[str] = None,
        render: bool = False,
    ):
        """
        Initialize training runner.
        
        Args:
            config: Training configuration
            log_dir: Directory for logs and checkpoints
            checkpoint_path: Optional path to resume from checkpoint
            render: Whether to render environments (slow, use for debugging)
        """
        self.config = config
        self.log_dir = Path(log_dir)
        self.checkpoint_path = checkpoint_path
        self.render = render
        
        # Create log directories
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.tensorboard_dir = self.log_dir / "tensorboard"
        self.episode_log_dir = self.log_dir / "episodes"
        
        for dir_path in [self.checkpoint_dir, self.tensorboard_dir, self.episode_log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize components (set in setup())
        self.env = None
        self.ppo = None
        self.actor_critic = None
        self.start_iteration = 0
        
    def setup(self):
        """Initialize Isaac Sim environments and PPO algorithm."""
        print("\n" + "="*80)
        print("SPOT RL TRAINING SETUP")
        print("="*80)
        
        # Initialize Isaac Sim
        print(f"\n[1/4] Initializing Isaac Sim with {self.config.env.num_envs} environments...")
        self._init_simulation()
        
        # Create RL environments
        print(f"\n[2/4] Creating Spot RL environments...")
        self._init_environments()
        
        # Build actor-critic network
        print(f"\n[3/4] Building actor-critic network...")
        self._init_actor_critic()
        
        # Initialize PPO algorithm
        print(f"\n[4/4] Initializing PPO algorithm...")
        self._init_ppo()
        
        # Load checkpoint if provided
        if self.checkpoint_path:
            self._load_checkpoint()
        
        print("\n" + "="*80)
        print("SETUP COMPLETE - Ready to train!")
        print("="*80 + "\n")
    
    def _init_simulation(self):
        """Initialize Isaac Sim application."""
        # Launch Isaac Sim with appropriate settings
        config = {
            "headless": not self.render,
            "width": 1280 if self.render else 640,
            "height": 720 if self.render else 480,
        }
        self.simulation_app = SimulationApp(config)
        
        # Import Isaac Sim modules (after app initialization)
        from omni.isaac.core import World
        from omni.isaac.core.utils.extensions import enable_extension
        
        # Enable required extensions
        enable_extension("omni.isaac.debug_draw")
        
        # Create world
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.config.env.physics_dt,
            rendering_dt=self.config.env.control_dt,
        )
    
    def _init_environments(self):
        """Create parallel Spot RL environments."""
        # Create multiple environments in Isaac Sim
        from omni.isaac.core.utils.prims import create_prim
        from pxr import UsdGeom
        
        self.envs = []
        env_spacing = 10.0  # 10m between environment centers
        
        for i in range(self.config.env.num_envs):
            # Calculate grid position
            row = i // int(np.sqrt(self.config.env.num_envs))
            col = i % int(np.sqrt(self.config.env.num_envs))
            
            pos = np.array([row * env_spacing, col * env_spacing, 0.0])
            
            # Create environment instance
            env = SpotRLEnv(
                env_id=i,
                position=pos,
                config=self.config,
                world=self.world,
                log_dir=str(self.episode_log_dir),
            )
            self.envs.append(env)
        
        print(f"Created {len(self.envs)} environments in {int(np.sqrt(self.config.env.num_envs))}x grid")
        
        # Wrap in VecEnv interface for RSL-RL
        self.env = VecEnvWrapper(self.envs, self.device)
    
    def _init_actor_critic(self):
        """Build actor-critic neural network."""
        obs_dim = self.env.num_obs
        act_dim = self.env.num_acts
        
        print(f"Observation space: {obs_dim}")
        print(f"Action space: {act_dim}")
        print(f"Network architecture: {self.config.ppo.hidden_dims}")
        
        self.actor_critic = ActorCritic(
            num_actor_obs=obs_dim,
            num_critic_obs=obs_dim,
            num_actions=act_dim,
            actor_hidden_dims=self.config.ppo.hidden_dims,
            critic_hidden_dims=self.config.ppo.hidden_dims,
            activation=self.config.ppo.activation,
            init_noise_std=self.config.ppo.init_noise_std,
        ).to(self.device)
        
        # Print network summary
        total_params = sum(p.numel() for p in self.actor_critic.parameters())
        print(f"Total parameters: {total_params:,}")
    
    def _init_ppo(self):
        """Initialize PPO algorithm."""
        self.ppo = PPO(
            actor_critic=self.actor_critic,
            num_learning_epochs=self.config.ppo.num_learning_epochs,
            num_mini_batches=self.config.ppo.num_mini_batches,
            clip_param=self.config.ppo.clip_param,
            gamma=self.config.ppo.gamma,
            lam=self.config.ppo.lam,
            value_loss_coef=self.config.ppo.value_loss_coef,
            entropy_coef=self.config.ppo.entropy_coef,
            learning_rate=self.config.ppo.learning_rate,
            max_grad_norm=self.config.ppo.max_grad_norm,
            use_clipped_value_loss=self.config.ppo.use_clipped_value_loss,
            fixed_sigma=not self.config.ppo.adaptive_sigma,
            device=self.device,
        )
        
        print(f"PPO learning rate: {self.config.ppo.learning_rate}")
        print(f"PPO clip param: {self.config.ppo.clip_param}")
        print(f"PPO epochs per update: {self.config.ppo.num_learning_epochs}")
    
    def _load_checkpoint(self):
        """Load training checkpoint."""
        print(f"\nLoading checkpoint: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Load model weights
        self.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        if "optimizer_state_dict" in checkpoint:
            self.ppo.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Resume from iteration
        if "iteration" in checkpoint:
            self.start_iteration = checkpoint["iteration"] + 1
            print(f"Resuming from iteration {self.start_iteration}")
    
    def train(self):
        """Run main training loop."""
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
        print(f"Total iterations: {self.config.ppo.max_iterations}")
        print(f"Steps per iteration: {self.config.env.num_envs * self.config.env.max_episode_length}")
        print(f"Log directory: {self.log_dir}")
        print(f"Monitor training: tensorboard --logdir {self.tensorboard_dir}")
        print("="*80 + "\n")
        
        # Training statistics
        stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "waypoints_reached": [],
            "success_rate": [],
        }
        
        start_time = time.time()
        
        for iteration in range(self.start_iteration, self.config.ppo.max_iterations):
            iter_start = time.time()
            
            # Collect rollouts
            obs = self.env.reset() if iteration == self.start_iteration else obs
            
            rollout_data = []
            for step in range(self.config.ppo.num_steps_per_env):
                # Get actions from policy
                with torch.no_grad():
                    actions = self.actor_critic.act_inference(obs)
                
                # Step environments
                obs, rewards, dones, infos = self.env.step(actions)
                
                # Store transition
                rollout_data.append({
                    "obs": obs,
                    "actions": actions,
                    "rewards": rewards,
                    "dones": dones,
                    "infos": infos,
                })
                
                # Update statistics from completed episodes
                for info in infos:
                    if info.get("episode_finished", False):
                        stats["episode_rewards"].append(info["episode_reward"])
                        stats["episode_lengths"].append(info["episode_length"])
                        stats["waypoints_reached"].append(info["waypoints_reached"])
                        stats["success_rate"].append(float(info["success"]))
            
            # Compute returns and advantages
            self.ppo.compute_returns(rollout_data, self.config.ppo.gamma, self.config.ppo.lam)
            
            # Update policy
            update_info = self.ppo.update()
            
            # Learning rate schedule
            if self.config.ppo.schedule == "adaptive":
                self._update_learning_rate(iteration, update_info)
            
            # Log statistics
            iter_time = time.time() - iter_start
            elapsed_time = time.time() - start_time
            
            if iteration % self.config.ppo.log_interval == 0:
                self._log_iteration(iteration, stats, update_info, iter_time, elapsed_time)
                stats = {k: [] for k in stats}  # Reset stats
            
            # Save checkpoint
            if iteration % self.config.ppo.save_interval == 0:
                self._save_checkpoint(iteration)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        
        # Save final model
        self._save_checkpoint(self.config.ppo.max_iterations, is_final=True)
        
        # Cleanup
        self.world.stop()
        self.simulation_app.close()
    
    def _update_learning_rate(self, iteration: int, update_info: Dict):
        """Adaptive learning rate based on KL divergence."""
        kl_mean = update_info.get("kl_divergence", 0.0)
        
        if kl_mean > self.config.ppo.desired_kl * 2.0:
            # KL too high, reduce learning rate
            self.ppo.learning_rate *= 0.5
        elif kl_mean < self.config.ppo.desired_kl / 2.0:
            # KL too low, increase learning rate
            self.ppo.learning_rate = min(
                self.ppo.learning_rate * 1.5,
                self.config.ppo.learning_rate  # Cap at initial LR
            )
    
    def _log_iteration(
        self,
        iteration: int,
        stats: Dict,
        update_info: Dict,
        iter_time: float,
        elapsed_time: float,
    ):
        """Log training statistics."""
        # Compute statistics
        mean_reward = np.mean(stats["episode_rewards"]) if stats["episode_rewards"] else 0.0
        mean_length = np.mean(stats["episode_lengths"]) if stats["episode_lengths"] else 0.0
        mean_waypoints = np.mean(stats["waypoints_reached"]) if stats["waypoints_reached"] else 0.0
        success_rate = np.mean(stats["success_rate"]) if stats["success_rate"] else 0.0
        
        # Print to console
        print(f"Iteration {iteration}/{self.config.ppo.max_iterations}")
        print(f"  Mean Reward: {mean_reward:.2f}")
        print(f"  Mean Episode Length: {mean_length:.1f}")
        print(f"  Mean Waypoints: {mean_waypoints:.1f}/25")
        print(f"  Success Rate: {success_rate*100:.1f}%")
        print(f"  Policy Loss: {update_info.get('policy_loss', 0.0):.4f}")
        print(f"  Value Loss: {update_info.get('value_loss', 0.0):.4f}")
        print(f"  LR: {self.ppo.learning_rate:.6f}")
        print(f"  Iteration Time: {iter_time:.2f}s")
        print(f"  Total Time: {elapsed_time/3600:.1f}h")
        print()
        
        # TODO: Write to TensorBoard
        # - Log rewards, lengths, success rates
        # - Log policy/value losses
        # - Log learning rate
    
    def _save_checkpoint(self, iteration: int, is_final: bool = False):
        """Save training checkpoint."""
        suffix = "final" if is_final else f"{iteration:06d}"
        checkpoint_path = self.checkpoint_dir / f"model_{suffix}.pt"
        
        checkpoint = {
            "iteration": iteration,
            "model_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.ppo.optimizer.state_dict(),
            "config": self.config,
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")


class VecEnvWrapper:
    """
    Wrapper to make list of SpotRLEnv compatible with RSL-RL VecEnv interface.
    """
    
    def __init__(self, envs: list, device: torch.device):
        self.envs = envs
        self.device = device
        self.num_envs = len(envs)
        self.num_obs = envs[0].get_observations().shape[0]
        self.num_acts = 3  # [vx, vy, omega]
    
    def reset(self) -> torch.Tensor:
        """Reset all environments."""
        obs_list = []
        for env in self.envs:
            obs = env.reset()
            obs_list.append(obs)
        
        obs_tensor = torch.tensor(np.stack(obs_list), dtype=torch.float32, device=self.device)
        return obs_tensor
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """Step all environments."""
        actions_np = actions.cpu().numpy()
        
        obs_list = []
        rewards_list = []
        dones_list = []
        infos_list = []
        
        for i, env in enumerate(self.envs):
            obs, reward, done, info = env.step(actions_np[i])
            obs_list.append(obs)
            rewards_list.append(reward)
            dones_list.append(done)
            infos_list.append(info)
        
        obs_tensor = torch.tensor(np.stack(obs_list), dtype=torch.float32, device=self.device)
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones_list, dtype=torch.bool, device=self.device)
        
        return obs_tensor, rewards_tensor, dones_tensor, infos_list


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Spot RL policy")
    
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=["default", "debug", "fast", "stable", "high_performance"],
        help="Training configuration profile"
    )
    
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Override number of parallel environments"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override number of training iterations"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environments (slow, for debugging)"
    )
    
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory for logs and checkpoints"
    )
    
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()
    
    # Load configuration
    if args.config == "default":
        config = get_default_config()
    else:
        config = get_config_by_name(args.config)
    
    # Override config with command line args
    if args.num_envs is not None:
        config.env.num_envs = args.num_envs
    
    if args.iterations is not None:
        config.ppo.max_iterations = args.iterations
    
    # Set up log directory
    if args.log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/spot_rl_{args.config}_{timestamp}"
    else:
        log_dir = args.log_dir
    
    # Create runner
    runner = SpotRLRunner(
        config=config,
        log_dir=log_dir,
        checkpoint_path=args.resume,
        render=args.render,
    )
    
    # Setup and train
    runner.setup()
    runner.train()


if __name__ == "__main__":
    main()
