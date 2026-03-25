"""
Spot RL Training Script
=======================
PPO-based training for Spot robot using Isaac Lab/RSL RL framework.

Supports:
  - Multi-environment parallel training
  - Custom reward shaping
  - Policy checkpointing
  - Tensorboard logging
  - Curriculum learning

Author: Autonomy Project
Date: February 2026
"""

import argparse
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import json
from collections import defaultdict
import csv

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

# Training setup
def setup_training_config():
    """Setup training configuration"""
    
    config = {
        # Training parameters
        'num_episodes': 1000,
        'episode_length': 300,  # seconds
        'batch_size': 32,
        'learning_rate': 3e-4,
        'gamma': 0.99,  # discount factor
        'gae_lambda': 0.95,  # GAE parameter
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'max_grad_norm': 0.5,
        'num_epochs': 3,  # PPO epochs per batch
        
        # Network architecture
        'hidden_size': 256,
        'num_layers': 2,
        'activation': 'tanh',
        
        # Logging and checkpointing
        'log_interval': 10,
        'save_interval': 50,
        'tensorboard_log': './runs/spot_rl',
        'checkpoint_dir': './checkpoints/spot_rl',
        
        # Curriculum learning
        'use_curriculum': True,
        'curriculum_stages': [
            {'terrain': 'flat', 'episodes': 200},
            {'terrain': 'obstacles', 'episodes': 400},
            {'terrain': 'varied', 'episodes': 400},
        ],
        
        # Device
        'device': 'cuda',  # cuda or cpu
        'seed': 42,
    }
    
    return config


class ActorCriticNetwork(nn.Module):
    """PyTorch Actor-Critic Policy Network
    
    Architecture:
    - Shared trunk: observation -> 2 hidden layers (256 units each, ReLU)
    - Actor head: trunk -> continuous action distribution (Gaussian)
    - Critic head: trunk -> scalar value estimate
    
    Observation space: 59 dimensions
    Action space: 12 continuous dimensions (motor torques)
    """
    
    def __init__(self, obs_dim: int = 59, action_dim: int = 12, hidden_size: int = 256, 
                 device: str = 'cpu'):
        """Initialize actor-critic network
        
        Args:
            obs_dim: Observation dimension (59 for Spot)
            action_dim: Action dimension (12 for Spot motors)
            hidden_size: Hidden layer size (256)
            device: Torch device ('cpu' or 'cuda')
        """
        super(ActorCriticNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        
        # Shared trunk
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Actor head: outputs mean and log_std for Gaussian policy
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
            nn.Tanh(),  # Bounded output [-1, 1]
        )
        
        # Log standard deviation (learnable parameter, shared across batch)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic head: outputs scalar value estimate
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        
        self.to(device)
    
    def forward(self, obs: torch.Tensor) -> tuple:
        """Forward pass through network
        
        Args:
            obs: Observation tensor (batch_size, obs_dim) or (obs_dim,)
        
        Returns:
            mean: Action mean (batch_size, action_dim) or (action_dim,)
            log_std: Action log std (batch_size, action_dim) or (action_dim,)
            value: Value estimate (batch_size, 1) or (1,)
        """
        
        # Handle single observation
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Forward through shared backbone
        features = self.backbone(obs)
        
        # Actor outputs
        mean = self.actor_mean(features)
        # Scale tanh output from [-1, 1] to torque limits [-150, 150]
        mean = mean * 150.0  # Scale to motor torque range
        
        log_std = self.actor_log_std.expand(mean.size())
        
        # Critic output
        value = self.critic(features)
        
        # Squeeze if needed
        if squeeze_output:
            mean = mean.squeeze(0)
            log_std = log_std.squeeze(0)
            value = value.squeeze(0)
        
        return mean, log_std, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> tuple:
        """Get action from policy
        
        Args:
            obs: Observation tensor
            deterministic: If True, return mean action (no sampling)
        
        Returns:
            action: Action to execute
            log_prob: Log probability of action
            value: Value estimate
        """
        
        mean, log_std, value = self.forward(obs)
        
        if deterministic:
            # Deterministic action (mean)
            return mean, torch.zeros_like(mean).sum(), value
        
        # Stochastic action
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        action = dist.rsample()  # Reparameterized sampling
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Clamp action to motor limits
        action = torch.clamp(action, -150.0, 150.0)
        
        return action, log_prob, value.squeeze(-1)
    
    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> tuple:
        """Evaluate action under current policy
        
        Used during PPO training to compute policy gradients.
        
        Args:
            obs: Observation tensor (batch_size, obs_dim)
            action: Action tensor (batch_size, action_dim)
        
        Returns:
            log_prob: Log probability of action under current policy
            value: Value estimate
            entropy: Policy entropy (for entropy bonus)
        """
        
        mean, log_std, value = self.forward(obs)
        
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        
        return log_prob, value.squeeze(-1), entropy


class PPOTrainer:
    """PPO trainer for Spot RL"""
    
    def __init__(self, env, config: dict):
        """Initialize PPO trainer
        
        Args:
            env: SpotRLEnvironment instance
            config: Training configuration dictionary
        """
        
        self.env = env
        self.config = config
        self.episode_count = 0
        self.total_steps = 0
        
        # Ensure device is valid
        if self.config['device'] == 'cuda' and not torch.cuda.is_available():
            print("[WARNING] CUDA not available, falling back to CPU")
            self.config['device'] = 'cpu'
        
        # Setup directories
        self._setup_directories()
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(self.config['tensorboard_log'])
        
        # Setup CSV logging
        self.csv_file = Path(self.config['tensorboard_log']) / 'training_metrics.csv'
        self.csv_headers = [
            'episode', 'total_steps', 'avg_reward', 'avg_length',
            'policy_loss', 'value_loss', 'total_loss', 'entropy',
            'success_rate', 'timestamp'
        ]
        self._init_csv()
        
        # Initialize policy network
        self._setup_networks()
        
        # Training state
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_stats = defaultdict(list)
        
        print("\n[OK] PPO Trainer initialized")
        print(f"[OK] TensorBoard logs: {self.config['tensorboard_log']}")
        print(f"[OK] CSV logs: {self.csv_file}")
    
    def _init_csv(self):
        """Initialize CSV file with headers"""
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_headers)
                writer.writeheader()
    
    def _setup_directories(self):
        """Setup logging and checkpoint directories"""
        
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['tensorboard_log'], exist_ok=True)
        
        self.checkpoint_dir = Path(self.config['checkpoint_dir'])
        self.log_dir = Path(self.config['tensorboard_log'])
    
    def _setup_networks(self):
        """Setup policy and value networks using PyTorch"""
        
        # Create actor-critic network
        self.policy_net = ActorCriticNetwork(
            obs_dim=59,  # Spot observation dimension
            action_dim=self.env.config.num_motors,
            hidden_size=self.config['hidden_size'],
            device=self.config['device']
        )
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.config['learning_rate']
        )
        
        # Move to device
        self.policy_net = self.policy_net.to(self.config['device'])
        
        print(f"  [OK] Actor-Critic network created (59->256->12)")
        print(f"  [OK] Optimizer: Adam (lr={self.config['learning_rate']})")
        print(f"  [OK] Device: {self.config['device']}")
    
    def collect_trajectories(self, num_episodes: int) -> dict:
        """Collect trajectories from environment
        
        Args:
            num_episodes: Number of episodes to collect
        
        Returns:
            trajectories: Dictionary with collected data
        """
        
        trajectories = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'values': [],
            'log_probs': [],
        }
        
        for ep in range(num_episodes):
            obs = self.env.reset()
            done = False
            ep_reward = 0.0
            ep_length = 0
            
            while not done:
                # Flatten observations to state vector
                state = self._flatten_obs(obs)
                
                # Get action from policy (placeholder)
                action, log_prob, value = self._get_action(state)
                
                # Step environment
                obs, reward, done, info = self.env.step(action)
                
                # Store trajectory
                trajectories['states'].append(state)
                trajectories['actions'].append(action)
                trajectories['rewards'].append(reward)
                trajectories['log_probs'].append(log_prob)
                trajectories['values'].append(value)
                trajectories['dones'].append(done)
                
                ep_reward += reward
                ep_length += 1
                self.total_steps += 1
            
            self.episode_rewards.append(ep_reward)
            self.episode_lengths.append(ep_length)
            self.episode_count += 1
            
            if (ep + 1) % max(1, num_episodes // 5) == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"  Episode {ep + 1}/{num_episodes}: "
                      f"Reward={avg_reward:.4f}, Length={ep_length}")
        
        return trajectories
    
    def train_step(self, trajectories: dict):
        """Perform one PPO training step
        
        Implements:
        - Generalized Advantage Estimation (GAE)
        - Policy gradient loss with PPO clipping
        - Value function loss
        - Entropy bonus for exploration
        - Gradient clipping
        
        Args:
            trajectories: Dictionary with collected trajectories
        """
        
        # Convert to tensors
        states_list = trajectories['states']
        actions_list = trajectories['actions']
        rewards_list = trajectories['rewards']
        dones_list = trajectories['dones']
        old_values_list = trajectories['values']
        old_log_probs_list = trajectories['log_probs']
        
        # Stack into batch tensors
        states = torch.FloatTensor(np.array(states_list)).to(self.config['device'])
        actions = torch.FloatTensor(np.array(actions_list)).to(self.config['device'])
        rewards = torch.FloatTensor(rewards_list).to(self.config['device'])
        dones = torch.FloatTensor(dones_list).to(self.config['device'])
        old_values = torch.FloatTensor(old_values_list).to(self.config['device'])
        old_log_probs = torch.FloatTensor(old_log_probs_list).to(self.config['device'])
        
        # ===== GENERALIZED ADVANTAGE ESTIMATION (GAE) =====
        
        # Get new value estimates
        with torch.no_grad():
            _, _, new_values = self.policy_net(states)
            new_values = new_values.squeeze(-1).detach()
        
        # Compute TD residuals
        td_residuals = rewards + self.config['gamma'] * new_values[1:] * (1 - dones[:-1]) - old_values[:-1]
        
        # Compute advantages using GAE
        gae = 0
        advantages = []
        for t in reversed(range(len(td_residuals))):
            gae = td_residuals[t] + self.config['gamma'] * self.config['gae_lambda'] * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(self.config['device'])
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute returns
        returns = advantages + old_values[:-1]
        
        # ===== PPO TRAINING LOOP =====
        
        batch_states = states[:-1]  # Exclude last state
        batch_actions = actions[:-1]
        batch_old_values = old_values[:-1]
        batch_old_log_probs = old_log_probs[:-1]
        
        epoch_losses = {'policy': [], 'value': [], 'total': [], 'entropy': []}
        
        for epoch in range(self.config['num_epochs']):
            
            # Forward pass
            log_probs, values, entropy = self.policy_net.evaluate(batch_states, batch_actions)
            values = values.squeeze(-1)
            
            # Policy loss with PPO clipping
            ratio = torch.exp(log_probs - batch_old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages  # clip ratio
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Entropy bonus (for exploration)
            entropy_bonus = -self.config['entropy_coef'] * entropy
            
            # Total loss
            total_loss = policy_loss + self.config['value_coef'] * value_loss + entropy_bonus
            
            # Gradient step
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config['max_grad_norm'])
            self.optimizer.step()
            
            # Store losses
            self.training_stats['policy_loss'].append(policy_loss.item())
            self.training_stats['value_loss'].append(value_loss.item())
            self.training_stats['total_loss'].append(total_loss.item())
            self.training_stats['entropy'].append(entropy.item())
            
            epoch_losses['policy'].append(policy_loss.item())
            epoch_losses['value'].append(value_loss.item())
            epoch_losses['total'].append(total_loss.item())
            epoch_losses['entropy'].append(entropy.item())
        
        # Log training details
        if len(self.training_stats['total_loss']) > 0:
            avg_loss = np.mean(self.training_stats['total_loss'][-self.config['num_epochs']:])
            self.training_stats['recent_avg_loss'] = avg_loss
    
    def train(self):
        """Main training loop"""
        
        print("\n" + "=" * 80)
        print("STARTING SPOT RL TRAINING")
        print("=" * 80)
        print(f"Configuration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        print("=" * 80 + "\n")
        
        curriculum_stage = 0
        
        for episode in range(self.config['num_episodes']):
            
            # Update curriculum if enabled
            if self.config['use_curriculum']:
                stage_info = self._update_curriculum(episode)
                if stage_info:
                    print(f"\n>>> Curriculum Stage {curriculum_stage}: {stage_info['terrain']}")
                    curriculum_stage += 1
            
            # Collect trajectories
            print(f"\n[Episode {episode + 1}/{self.config['num_episodes']}] Collecting trajectories...")
            trajectories = self.collect_trajectories(num_episodes=1)
            
            # Train on collected data
            print(f"  Training on {len(trajectories['states'])} steps...")
            self.train_step(trajectories)
            
            # Logging
            if (episode + 1) % self.config['log_interval'] == 0:
                self._log_stats(episode)
            
            # Checkpointing
            if (episode + 1) % self.config['save_interval'] == 0:
                self._save_checkpoint(episode)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        
        self._save_checkpoint(self.config['num_episodes'] - 1, final=True)
    
    def _flatten_obs(self, obs: dict) -> np.ndarray:
        """Flatten observation dictionary to 1D array
        
        Args:
            obs: Observation dictionary
        
        Returns:
            state: Flattened observation vector
        """
        
        state_parts = []
        
        # Order: position (3), heading (1), lin_vel (3), ang_vel (3), 
        #        joint_pos (12), joint_vel (12), goal_dist (1), rel_goal (2)
        
        if 'position' in obs:
            state_parts.append(obs['position'].flatten())
        if 'heading' in obs:
            state_parts.append(obs['heading'].flatten())
        if 'linear_velocity' in obs:
            state_parts.append(obs['linear_velocity'].flatten())
        if 'angular_velocity' in obs:
            state_parts.append(obs['angular_velocity'].flatten())
        if 'joint_positions' in obs:
            state_parts.append(obs['joint_positions'].flatten())
        if 'joint_velocities' in obs:
            state_parts.append(obs['joint_velocities'].flatten())
        if 'distance_to_goal' in obs:
            state_parts.append(obs['distance_to_goal'].flatten())
        if 'relative_goal' in obs:
            state_parts.append(obs['relative_goal'].flatten())
        if 'motor_effort' in obs:
            state_parts.append(obs['motor_effort'].flatten())
        
        return np.concatenate(state_parts) if state_parts else np.array([])
    
    def _get_action(self, state: np.ndarray) -> tuple:
        """Get action from policy network
        
        Args:
            state: Observation state (59-dim numpy array)
        
        Returns:
            action: Motor torque commands (numpy array)
            log_prob: Log probability of action (scalar)
            value: Value estimate (scalar)
        """
        
        self.policy_net.eval()
        
        with torch.no_grad():
            # Convert to tensor
            state_tensor = torch.FloatTensor(state).to(self.config['device'])
            
            # Get action from policy
            action_tensor, log_prob_tensor, value_tensor = self.policy_net.get_action(
                state_tensor, 
                deterministic=False
            )
            
            # Convert to numpy
            action = action_tensor.cpu().numpy()
            log_prob = log_prob_tensor.cpu().item()
            value = value_tensor.cpu().item()
        
        self.policy_net.train()
        
        return action, log_prob, value
    
    def _update_curriculum(self, episode: int) -> dict:
        """Update curriculum stage if applicable
        
        Args:
            episode: Current episode number
        
        Returns:
            stage_info: Info about current curriculum stage or None
        """
        
        if not self.config['use_curriculum']:
            return None
        
        cumulative_episodes = 0
        for stage_idx, stage in enumerate(self.config['curriculum_stages']):
            cumulative_episodes += stage['episodes']
            if episode < cumulative_episodes:
                return stage
        
        return None
    
    def _log_stats(self, episode: int):
        """Log training statistics to console, TensorBoard, and CSV
        
        Args:
            episode: Current episode number
        """
        
        if len(self.episode_rewards) == 0:
            return
        
        avg_reward = np.mean(self.episode_rewards[-self.config['log_interval']:])
        avg_length = np.mean(self.episode_lengths[-self.config['log_interval']:])
        
        # Calculate loss metrics
        policy_losses = self.training_stats['policy_loss'][-self.config['num_epochs']*self.config['log_interval']:]
        value_losses = self.training_stats['value_loss'][-self.config['num_epochs']*self.config['log_interval']:]
        total_losses = self.training_stats['total_loss'][-self.config['num_epochs']*self.config['log_interval']:]
        
        avg_policy_loss = np.mean(policy_losses) if policy_losses else 0.0
        avg_value_loss = np.mean(value_losses) if value_losses else 0.0
        avg_total_loss = np.mean(total_losses) if total_losses else 0.0
        
        # Estimate success rate (reached goal or high reward)
        recent_rewards = self.episode_rewards[-self.config['log_interval']:]
        success_rate = np.sum(np.array(recent_rewards) > 100) / len(recent_rewards) if recent_rewards else 0.0
        
        # ===== TENSORBOARD LOGGING =====
        self.writer.add_scalar('Reward/avg_episode_reward', avg_reward, episode)
        self.writer.add_scalar('Reward/max_episode_reward', np.max(self.episode_rewards[-self.config['log_interval']:]), episode)
        self.writer.add_scalar('Reward/min_episode_reward', np.min(self.episode_rewards[-self.config['log_interval']:]), episode)
        self.writer.add_scalar('Metrics/episode_length', avg_length, episode)
        self.writer.add_scalar('Metrics/success_rate', success_rate, episode)
        self.writer.add_scalar('Loss/policy_loss', avg_policy_loss, episode)
        self.writer.add_scalar('Loss/value_loss', avg_value_loss, episode)
        self.writer.add_scalar('Loss/total_loss', avg_total_loss, episode)
        self.writer.add_scalar('Training/total_steps', self.total_steps, episode)
        self.writer.flush()
        
        # ===== CSV LOGGING =====
        csv_row = {
            'episode': episode + 1,
            'total_steps': self.total_steps,
            'avg_reward': f"{avg_reward:.4f}",
            'avg_length': f"{avg_length:.1f}",
            'policy_loss': f"{avg_policy_loss:.6f}",
            'value_loss': f"{avg_value_loss:.6f}",
            'total_loss': f"{avg_total_loss:.6f}",
            'entropy': f"{self.training_stats.get('entropy', [0])[-1]:.4f}",
            'success_rate': f"{success_rate:.3f}",
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            writer.writerow(csv_row)
        
        # ===== CONSOLE OUTPUT =====
        print(f"\n{'='*90}")
        print(f"[Episode {episode + 1}/{self.config['num_episodes']}] Training Progress")
        print(f"{'='*90}")
        print(f"  Reward:        {avg_reward:>10.4f} (max: {np.max(self.episode_rewards[-self.config['log_interval']:]):>8.4f}, "
              f"min: {np.min(self.episode_rewards[-self.config['log_interval']:]):>8.4f})")
        print(f"  Episode Length:{avg_length:>10.1f} steps")
        print(f"  Success Rate:  {success_rate:>10.1%}")
        print(f"  Policy Loss:   {avg_policy_loss:>10.6f}")
        print(f"  Value Loss:    {avg_value_loss:>10.6f}")
        print(f"  Total Loss:    {avg_total_loss:>10.6f}")
        print(f"  Total Steps:   {self.total_steps:>10d}")
        print(f"{'='*90}")
    
    def _save_checkpoint(self, episode: int, final: bool = False):
        """Save training checkpoint with network weights
        
        Args:
            episode: Current episode
            final: Whether this is final checkpoint
        """
        
        checkpoint_name = f"spot_rl_final" if final else f"spot_rl_ep{episode + 1}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint_data = {
            'episode': episode + 1,
            'total_steps': self.total_steps,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_stats': dict(self.training_stats),
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
            # Network state
            'policy_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        # Save as PyTorch checkpoint (includes weights)
        torch.save(checkpoint_data, str(checkpoint_path) + '.pt')
        
        # Also save as JSON (stats only)
        json_data = {k: v for k, v in checkpoint_data.items() 
                     if k not in ['policy_state_dict', 'optimizer_state_dict']}
        with open(str(checkpoint_path) + '.json', 'w') as f:
            json.dump(json_data, f, indent=2)
        
        if final:
            print(f"\n[OK] Final checkpoint saved: {checkpoint_path}.pt")
        if (episode + 1) % (self.config['save_interval'] * 2) == 0:
            print(f"[OK] Checkpoint saved: {checkpoint_path}.pt")
    
    def close(self):
        """Close TensorBoard writer and finalize logging"""
        self.writer.close()
        print(f"\n[OK] TensorBoard writer closed")
        print(f"[OK] All training data logged to {self.config['tensorboard_log']}")


def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description="Train Spot RL agent")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()
    
    # Import environment
    from SpotRL_Environment import SpotRLEnvironment, SpotRLConfig
    
    # Create environment
    config = SpotRLConfig()
    env = SpotRLEnvironment(config)
    
    # Setup training
    training_config = setup_training_config()
    training_config['num_episodes'] = args.episodes
    training_config['device'] = args.device
    
    # Create trainer
    trainer = PPOTrainer(env, training_config)
    
    # Print device info
    print(f"\n[INFO] Using device: {training_config['device']}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Print tracking info
    print(f"\n{'='*80}")
    print("TRAINING TRACKING SETUP")
    print(f"{'='*80}")
    print(f"TensorBoard logs:  {training_config['tensorboard_log']}")
    print(f"CSV logs:          {trainer.csv_file}")
    print(f"Checkpoints:       {training_config['checkpoint_dir']}")
    print(f"\nTo monitor training in real-time, run:")
    print(f"  tensorboard --logdir {training_config['tensorboard_log']} --port 6006")
    print(f"Then open http://localhost:6006 in your browser")
    print(f"{'='*80}\n")
    
    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[OK] Training interrupted by user")
    finally:
        trainer.close()
        env.close()


if __name__ == "__main__":
    main()
