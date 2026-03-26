"""
Navigation Policy Network
=========================
High-level navigation policy for Spot waypoint navigation.
Outputs velocity commands [vx, vy, omega] to SpotFlatTerrainPolicy.

Pure PyTorch MLP - exportable via TorchScript or ONNX.

Author: Cole (MS for Autonomy Project)
Date: March 2026
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class NavigationPolicy(nn.Module):
    """
    High-level navigation policy MLP.
    
    Input: observation vector (75 dims for VS3 7-stage curriculum)
        - Base velocity: [vx, vy, omega] (3)
        - Heading: [sin(yaw), cos(yaw)] (2)
        - IMU data: [roll, pitch, accel_x, accel_y] (4)
        - Waypoint info: [dx, dy, distance] (3)
        - Multi-layer obstacle distances: 16 rays × 3 heights (48)
        - Foot contact feedback: 4 feet (4)
        - Leg joint summary: 4 legs (4)
        - Stage encoding: one-hot 7 stages (7)
    
    Output: action means [vx, vy, omega] (3)
    """
    
    def __init__(self, obs_dim: int = 75, action_dim: int = 3, 
                 hidden_dims: Tuple[int] = (256, 256, 128),
                 activation: str = "relu"):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Choose activation
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "silu":
            act_fn = nn.SiLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build actor network (policy)
        actor_layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            actor_layers.append(nn.Linear(prev_dim, hidden_dim))
            actor_layers.append(act_fn())
            prev_dim = hidden_dim
        actor_layers.append(nn.Linear(prev_dim, action_dim))
        actor_layers.append(nn.Tanh())  # Output in [-1, 1], will be scaled later
        
        self.actor = nn.Sequential(*actor_layers)
        
        # Build critic network (value function)
        critic_layers = []
        prev_dim = obs_dim
        critic_dims = (256, 128, 64)  # Slightly different architecture
        for hidden_dim in critic_dims:
            critic_layers.append(nn.Linear(prev_dim, hidden_dim))
            critic_layers.append(act_fn())
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        
        self.critic = nn.Sequential(*critic_layers)
        
        # Action log std (learned parameter, shared across all actions)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            obs: observation tensor, shape (batch, obs_dim)
        
        Returns:
            action_mean: mean of action distribution, shape (batch, action_dim)
            value: state value, shape (batch, 1)
        """
        action_mean = self.actor(obs)
        value = self.critic(obs)
        return action_mean, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            obs: observation tensor, shape (batch, obs_dim)
            deterministic: if True, return mean action without sampling
        
        Returns:
            action: sampled action, shape (batch, action_dim)
            log_prob: log probability of action, shape (batch,)
            value: state value, shape (batch,)
        """
        action_mean, value = self.forward(obs)
        
        if deterministic:
            return action_mean, None, value.squeeze()
        
        # Sample from Gaussian
        action_std = torch.exp(self.action_log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Note: No clipping applied here to maintain log_prob consistency
        # Actions are scaled to actual ranges by scale_action() function
        
        return action, log_prob, value.squeeze()
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions under current policy (for PPO update).
        
        Args:
            obs: observation tensor, shape (batch, obs_dim)
            actions: actions to evaluate, shape (batch, action_dim)
        
        Returns:
            log_probs: log probabilities, shape (batch,)
            values: state values, shape (batch,)
            entropy: action entropy, shape (batch,)
        """
        action_mean, values = self.forward(obs)
        
        action_std = torch.exp(self.action_log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, values.squeeze(), entropy


def scale_action(action: np.ndarray, vx_range: Tuple[float, float],
                 vy_range: Tuple[float, float], omega_range: Tuple[float, float]) -> np.ndarray:
    """
    Scale action from [-1, 1] to actual velocity command ranges.
    
    Args:
        action: normalized action in [-1, 1], shape (3,)
        vx_range: [min, max] for forward/back velocity
        vy_range: [min, max] for left/right strafe
        omega_range: [min, max] for turning rate
    
    Returns:
        scaled_action: actual velocity command [vx, vy, omega]
    """
    vx_min, vx_max = vx_range
    vy_min, vy_max = vy_range
    omega_min, omega_max = omega_range
    
    # Scale each component
    vx = action[0] * (vx_max if action[0] > 0 else -vx_min) if action[0] != 0 else 0.0
    vy = action[1] * (vy_max if action[1] > 0 else -vy_min) if action[1] != 0 else 0.0
    omega = action[2] * (omega_max if action[2] > 0 else -omega_min) if action[2] != 0 else 0.0
    
    return np.array([vx, vy, omega], dtype=np.float32)


if __name__ == "__main__":
    # Test network creation and export
    print("Testing NavigationPolicy network (75-dim observation)...")
    
    policy = NavigationPolicy(obs_dim=75, action_dim=3)
    print(f"Policy created with {sum(p.numel() for p in policy.parameters())} parameters")
    
    # Test forward pass
    obs = torch.randn(1, 75)
    action, log_prob, value = policy.get_action(obs)
    print(f"Action: {action}")
    print(f"Log prob: {log_prob}")
    print(f"Value: {value}")
    
    # Test TorchScript export
    print("\nTesting TorchScript export...")
    scripted = torch.jit.script(policy)
    print("Successfully exported to TorchScript!")
    
    # Test ONNX export
    print("\nTesting ONNX export...")
    torch.onnx.export(
        policy,
        obs,
        "navigation_policy.onnx",
        input_names=["observation"],
        output_names=["action_mean", "value"],
        dynamic_axes={"observation": {0: "batch"}, "action_mean": {0: "batch"}, "value": {0: "batch"}}
    )
    print("Successfully exported to ONNX!")
    print("\nNetwork is fully exportable and ready for deployment!")
