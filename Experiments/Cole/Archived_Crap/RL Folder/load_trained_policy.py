"""
LOAD TRAINED POLICY - Policy Deployment Utility
===============================================

This module loads trained RL policies for deployment in testing environments.

Usage Example:
--------------
from load_trained_policy import load_trained_policy

# Load the fully trained policy
actor_network, checkpoint_info = load_trained_policy(
    checkpoint_path="checkpoints/spot_rl_curriculum/FULLY_TRAINED_POLICY.pt"
)

# Or load a specific stage checkpoint
actor_network, checkpoint_info = load_trained_policy(
    checkpoint_path="checkpoints/spot_rl_curriculum/stage_3_complete.pt"
)

# Use in testing environment
observations = env.get_observations()  # Shape: (num_envs, obs_dim)
with torch.no_grad():
    actions = actor_network(observations)  # Shape: (num_envs, action_dim)

# Apply actions
env.apply_rl_action(actions)
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys


class ActorNetwork(nn.Module):
    """
    Actor network from PPO training.
    
    This is a simplified standalone version for inference only.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(512, 256, 128)):
        super().__init__()
        
        layers = []
        input_dim = obs_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ELU())
            input_dim = hidden_size
        
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())  # Actions normalized to [-1, 1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs):
        """
        Get actions from observations.
        
        Args:
            obs (torch.Tensor): Observations, shape (batch_size, obs_dim)
        
        Returns:
            torch.Tensor: Actions, shape (batch_size, action_dim)
        """
        return self.network(obs)


def load_trained_policy(checkpoint_path: str, device: str = "cuda"):
    """
    Load a trained policy from a checkpoint file.
    
    Args:
        checkpoint_path (str): Path to the checkpoint (.pt file)
        device (str): Device to load the model on ("cuda" or "cpu")
    
    Returns:
        tuple: (actor_network, checkpoint_info)
            - actor_network (ActorNetwork): Loaded actor network ready for inference
            - checkpoint_info (dict): Checkpoint metadata (iteration, stage, etc.)
    
    Example:
        >>> actor, info = load_trained_policy("checkpoints/spot_rl/FULLY_TRAINED_POLICY.pt")
        >>> print(f"Loaded policy from iteration {info['iteration']}, stage {info['stage']}")
        >>> with torch.no_grad():
        ...     actions = actor(observations)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    print(f"[LOAD] Loading policy from: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract metadata
    iteration = checkpoint.get('iteration', 'unknown')
    stage = checkpoint.get('stage', 'unknown')
    consecutive_successes = checkpoint.get('consecutive_successes', 'unknown')
    
    print(f"[INFO] Checkpoint metadata:")
    print(f"  - Iteration: {iteration}")
    print(f"  - Training Stage: {stage}")
    print(f"  - Consecutive Successes: {consecutive_successes}")
    
    # Get network dimensions
    if 'obs_dim' in checkpoint and 'action_dim' in checkpoint:
        obs_dim = checkpoint['obs_dim']
        action_dim = checkpoint['action_dim']
    else:
        # Try to infer from state dict
        actor_state = checkpoint['actor_state_dict']
        first_weight_key = [k for k in actor_state.keys() if 'weight' in k][0]
        first_weight = actor_state[first_weight_key]
        
        if 'network.0.weight' in actor_state:
            obs_dim = actor_state['network.0.weight'].shape[1]
        else:
            raise ValueError("Cannot determine obs_dim from checkpoint. Checkpoint may be incompatible.")
        
        last_weight_key = [k for k in actor_state.keys() if 'weight' in k][-1]
        action_dim = actor_state[last_weight_key].shape[0]
    
    print(f"  - Observation Dimension: {obs_dim}")
    print(f"  - Action Dimension: {action_dim}")
    
    # Create actor network
    actor = ActorNetwork(obs_dim=obs_dim, action_dim=action_dim).to(device)
    
    # Load weights
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()  # Set to evaluation mode
    
    print(f"[SUCCESS] Policy loaded and ready for inference!")
    print(f"[TIP] Use with torch.no_grad() for faster inference:")
    print(f"      with torch.no_grad():")
    print(f"          actions = actor(observations)")
    
    # Prepare info dict
    checkpoint_info = {
        'iteration': iteration,
        'stage': stage,
        'consecutive_successes': consecutive_successes,
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'checkpoint_path': str(checkpoint_path),
    }
    
    return actor, checkpoint_info


def list_available_checkpoints(checkpoint_dir: str = "checkpoints/spot_rl_curriculum"):
    """
    List all available checkpoint files in a directory.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoints
    
    Returns:
        list: List of checkpoint file paths
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"[WARNING] Checkpoint directory not found: {checkpoint_dir}")
        return []
    
    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    
    print(f"\n[INFO] Available checkpoints in {checkpoint_dir.name}:")
    print("=" * 80)
    
    if not checkpoints:
        print("  No checkpoints found.")
        return []
    
    for i, ckpt in enumerate(checkpoints, 1):
        print(f"  {i}. {ckpt.name}")
        
        # Try to load metadata without loading full model
        try:
            metadata = torch.load(ckpt, map_location='cpu')
            iteration = metadata.get('iteration', '?')
            stage = metadata.get('stage', '?')
            print(f"     → Iteration {iteration}, Stage {stage}")
        except Exception as e:
            print(f"     → (Could not read metadata)")
    
    print("=" * 80)
    
    return checkpoints


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example demonstrating how to load and use a trained policy.
    """
    
    print("\n" + "=" * 80)
    print("TRAINED POLICY LOADER - Example Usage")
    print("=" * 80)
    
    # 1. List available checkpoints
    checkpoint_dir = Path(__file__).parent / "checkpoints" / "spot_rl_curriculum"
    print(f"\nSearching for checkpoints in: {checkpoint_dir}")
    
    available = list_available_checkpoints(str(checkpoint_dir))
    
    if not available:
        print("\n[INFO] No checkpoints found yet. Train the model first using:")
        print("       python train_spot_ppo.py")
        sys.exit(0)
    
    # 2. Load the fully trained policy (if available)
    final_checkpoint = checkpoint_dir / "FULLY_TRAINED_POLICY.pt"
    
    if final_checkpoint.exists():
        print("\n" + "=" * 80)
        print("Loading FULLY TRAINED policy...")
        print("=" * 80)
        actor, info = load_trained_policy(str(final_checkpoint))
        
        # 3. Test with dummy data
        print("\n[TEST] Running inference test with random observations...")
        dummy_obs = torch.randn(16, info['obs_dim']).to('cuda')  # 16 parallel envs
        
        with torch.no_grad():
            actions = actor(dummy_obs)
        
        print(f"  Input shape: {dummy_obs.shape}")
        print(f"  Output shape: {actions.shape}")
        print(f"  Action range: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
        print(f"\n[SUCCESS] Inference test passed!")
        
    else:
        print("\n[INFO] FULLY_TRAINED_POLICY.pt not found yet.")
        print("        Training is still in progress or not started.")
        print("\nYou can load intermediate stage checkpoints:")
        
        # Load latest stage checkpoint
        stage_checkpoints = [c for c in available if 'stage_' in c.name]
        if stage_checkpoints:
            latest = stage_checkpoints[-1]
            print(f"\n[EXAMPLE] Loading latest stage checkpoint: {latest.name}")
            actor, info = load_trained_policy(str(latest))
    
    print("\n" + "=" * 80)
    print("To use in your testing environment:")
    print("=" * 80)
    print("""
from load_trained_policy import load_trained_policy
import torch

# Load trained policy
actor, info = load_trained_policy("checkpoints/spot_rl_curriculum/FULLY_TRAINED_POLICY.pt")

# In your environment loop:
observations = env.get_observations()  # Get current observations
with torch.no_grad():
    actions = actor(observations)  # Get actions from policy
env.apply_rl_action(actions)  # Apply actions to robots
    """)
    print("=" * 80 + "\n")
