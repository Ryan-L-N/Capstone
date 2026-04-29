"""
PPO Trainer
===========
Proximal Policy Optimization trainer for navigation policy.

Author: Cole (MS for Autonomy Project)
Date: March 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple


class PPOTrainer:
    """PPO algorithm trainer."""
    
    def __init__(self, policy, config: Dict):
        """
        Initialize PPO trainer.
        
        Args:
            policy: NavigationPolicy instance
            config: dictionary with PPO hyperparameters
        """
        self.policy = policy
        self.lr = config['learning_rate']
        self.clip_param = config['clip_param']
        self.value_coef = config['value_coef']
        self.entropy_coef = config['entropy_coef']
        self.gamma = config['gamma']
        self.gae_lambda = config['gae_lambda']
        self.max_grad_norm = config['max_grad_norm']
        self.target_kl = config.get('target_kl', None)  # Early stopping threshold
        self.ppo_epochs = config.get('ppo_epochs', 10)  # Multi-epoch training
        
        self.optimizer = optim.Adam(policy.parameters(), lr=self.lr)
    
    def update(self, rollout: Dict) -> Dict[str, float]:
        """
        Perform PPO update with multi-epoch training and early stopping.
        
        Args:
            rollout: dictionary containing:
                'observations': tensor (T, obs_dim)
                'actions': tensor (T, action_dim)
                'old_log_probs': tensor (T,)
                'returns': tensor (T,)
                'advantages': tensor (T,)
        
        Returns:
            stats: dictionary with training statistics
        """
        obs = rollout['observations']
        actions = rollout['actions']
        old_log_probs = rollout['old_log_probs']
        returns = rollout['returns']
        advantages = rollout['advantages']
        
        # Normalize advantages (important for stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Tracking statistics across epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        early_stop = False
        first_kl = None  # Track KL from first epoch
        
        # Multi-epoch training (full batch each epoch)
        for epoch in range(self.ppo_epochs):
            # Evaluate actions with current policy
            log_probs, values, entropy = self.policy.evaluate_actions(obs, actions)
            
            # Compute KL divergence for early stopping
            with torch.no_grad():
                ratio = torch.exp(log_probs - old_log_probs)
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
            
            # Track first KL for reporting
            if epoch == 0:
                first_kl = approx_kl
            
            # Check early stopping BEFORE updating (important!)
            if self.target_kl is not None and approx_kl > self.target_kl:
                early_stop = True
                break
            
            # Policy loss (clipped objective)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss (MSE)
            value_loss = ((values - returns) ** 2).mean()
            
            # Entropy bonus (for exploration)
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Accumulate statistics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += -entropy_loss.item()
            total_kl += approx_kl
        
        # Average statistics
        num_epochs = epoch if early_stop else self.ppo_epochs
        if num_epochs == 0:
            num_epochs = 1  # Avoid division by zero
        
        stats = {
            'policy_loss': total_policy_loss / num_epochs,
            'value_loss': total_value_loss / num_epochs,
            'entropy': total_entropy / num_epochs,
            'total_loss': (total_policy_loss + total_value_loss + total_entropy) / num_epochs,
            'approx_kl': total_kl / num_epochs if not early_stop else first_kl,  # Use first_kl if early stopped
            'epochs_completed': num_epochs,
            'early_stopped': early_stop
        }
        
        return stats


class RolloutBuffer:
    """Buffer for collecting rollout data."""
    
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def add(self, obs, action, reward, value, log_prob, done):
        """Add a transition to the buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get(self, next_value: float, gamma: float, gae_lambda: float) -> Dict:
        """
        Get rollout data with computed returns and advantages.
        
        Args:
            next_value: value estimate for the state after the last step
            gamma: discount factor
            gae_lambda: GAE lambda parameter
        
        Returns:
            rollout: dictionary with observations, actions, returns, advantages, etc.
        """
        # Convert to numpy
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        
        # Compute GAE
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        
        gae = 0
        next_val = next_value
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            if dones[t]:
                next_val = 0.0
            
            delta = rewards[t] + gamma * next_val - values[t]
            gae = delta + gamma * gae_lambda * gae * (1 - dones[t])
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Convert to tensors
        rollout = {
            'observations': torch.FloatTensor(np.array(self.observations)),
            'actions': torch.FloatTensor(np.array(self.actions)),
            'old_log_probs': torch.FloatTensor(np.array(self.log_probs)),
            'returns': torch.FloatTensor(returns),
            'advantages': torch.FloatTensor(advantages)
        }
        
        return rollout
    
    def clear(self):
        """Clear the buffer."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.observations)
