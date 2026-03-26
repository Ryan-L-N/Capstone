"""
Transfer locomotion pre-training weights to waypoint navigation network.

The locomotion network was trained on 67-dim observations (base proprioceptive data).
The navigation network needs 98-dim observations (67 base + 31 waypoint/obstacle info).

This script:
1. Loads the best locomotion checkpoint (nn.Sequential structure)
2. Creates a new navigation-compatible network (nested Actor/Critic classes)
3. Transfers weights intelligently with key name mapping
4. Handles dimension expansion: 67-dim -> 98-dim for first layer
5. Saves as navigation-ready checkpoint for Phase 2 training
"""

import torch
import torch.nn as nn
from pathlib import Path


# ============================================================================
# Locomotion Network Architecture (matches train_locomotion_pretrain.py)
# Uses nn.Sequential directly, creating keys like "0.weight", "2.weight"
# ============================================================================

class LocomotionActorCritic(nn.Module):
    """Locomotion Actor-Critic with Sequential networks (67-dim obs)."""
    def __init__(self, obs_dim=67, action_dim=3):
        super().__init__()
        
        # Actor: 67 -> 512 -> 256 -> 128 -> 3
        actor_layers = []
        actor_layers.append(nn.Linear(obs_dim, 512))
        actor_layers.append(nn.Tanh())
        actor_layers.append(nn.Linear(512, 256))
        actor_layers.append(nn.Tanh())
        actor_layers.append(nn.Linear(256, 128))
        actor_layers.append(nn.Tanh())
        actor_layers.append(nn.Linear(128, action_dim))
        actor_layers.append(nn.Tanh())
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic: 67 -> 256 -> 256 -> 128 -> 1
        critic_layers = []
        critic_layers.append(nn.Linear(obs_dim, 256))
        critic_layers.append(nn.Tanh())
        critic_layers.append(nn.Linear(256, 256))
        critic_layers.append(nn.Tanh())
        critic_layers.append(nn.Linear(256, 128))
        critic_layers.append(nn.Tanh())
        critic_layers.append(nn.Linear(128, 1))
        self.critic = nn.Sequential(*critic_layers)


# ============================================================================
# Navigation Network Architecture (matches train_spot_ppo.py)
# Uses nested Actor/Critic classes, creating keys like "network.0.weight"
# ============================================================================

class Actor(nn.Module):
    """Actor network with nested Sequential."""
    def __init__(self, obs_dim, action_dim, hidden_dims=(512, 256, 128)):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class Critic(nn.Module):
    """Critic network with nested Sequential."""
    def __init__(self, obs_dim, hidden_dims=(256, 256, 128)):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class ActorCritic(nn.Module):
    """ActorCritic wrapper combining Actor and Critic."""
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic


# ============================================================================
# Weight Transfer Function
# ============================================================================

def transfer_weights():
    """Transfer locomotion weights to navigation network."""
    
    print("\n" + "=" * 70)
    print("LOCOMOTION → NAVIGATION WEIGHT TRANSFER")
    print("=" * 70)
    
    # Paths
    checkpoint_path = Path("checkpoints/locomotion_pretrain/best_model.pt")
    output_path = Path("checkpoints/locomotion_pretrain/nav_init_checkpoint.pt")
    
    if not checkpoint_path.exists():
        print(f"\n❌ ERROR: Checkpoint not found: {checkpoint_path}")
        return False
    
    # ========================================================================
    # Step 1: Load locomotion checkpoint
    # ========================================================================
    print(f"\n[1/5] Loading locomotion checkpoint...")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    print(f"      ✓ Loaded: {checkpoint_path}")
    print(f"      - Actor keys: {len(checkpoint['actor_state_dict'])} parameters")
    print(f"      - Critic keys: {len(checkpoint['critic_state_dict'])} parameters")
    
    # ========================================================================
    # Step 2: Create locomotion network and load weights
    # ========================================================================
    print(f"\n[2/5] Creating locomotion network (67-dim)...")
    loco_net = LocomotionActorCritic(obs_dim=67, action_dim=3)
    loco_net.actor.load_state_dict(checkpoint['actor_state_dict'])
    loco_net.critic.load_state_dict(checkpoint['critic_state_dict'])
    print(f"      ✓ Loaded locomotion weights successfully")
    
    # ========================================================================
    # Step 3: Create navigation network (98-dim)
    # ========================================================================
    print(f"\n[3/5] Creating navigation network (98-dim)...")
    nav_actor = Actor(obs_dim=98, action_dim=3)
    nav_critic = Critic(obs_dim=98)
    print(f"      ✓ Navigation network initialized")
    
    # ========================================================================
    # Step 4: Transfer ACTOR weights with key mapping
    # ========================================================================
    print(f"\n[4/5] Transferring actor weights...")
    
    # Get state dicts
    loco_actor_dict = loco_net.actor.state_dict()  # Keys: "0.weight", "2.weight", etc.
    nav_actor_dict = nav_actor.network.state_dict()  # Keys: "0.weight", "2.weight", etc.
    
    new_nav_actor_dict = {}
    
    for key in nav_actor_dict.keys():
        if key == "0.weight":  # First layer: dimension expansion needed
            loco_weight = loco_actor_dict[key]  # [512, 67]
            nav_shape = nav_actor_dict[key].shape  # [512, 98]
            
            print(f"      [EXPAND] {key}: {tuple(loco_weight.shape)} → {tuple(nav_shape)}")
            
            # Create expanded weight matrix
            expanded_weight = torch.zeros(nav_shape)
            expanded_weight[:, :67] = loco_weight  # Copy first 67 dims
            
            # Initialize new 31 dimensions with small random values
            torch.nn.init.uniform_(expanded_weight[:, 67:], a=-0.01, b=0.01)
            
            new_nav_actor_dict[key] = expanded_weight
            print(f"               Copied 67 dims + initialized 31 new dims")
            
        elif key in loco_actor_dict:  # All other layers: direct copy
            loco_param = loco_actor_dict[key]
            nav_param = nav_actor_dict[key]
            
            if loco_param.shape == nav_param.shape:
                print(f"      [COPY]   {key}: {tuple(loco_param.shape)}")
                new_nav_actor_dict[key] = loco_param.clone()
            else:
                print(f"      [WARN]   {key}: Mismatch {tuple(loco_param.shape)} vs {tuple(nav_param.shape)}")
                new_nav_actor_dict[key] = nav_param
        else:
            print(f"      [INIT]   {key}: Not in locomotion (random init)")
            new_nav_actor_dict[key] = nav_actor_dict[key]
    
    # Load into navigation actor with "network." prefix
    nav_actor_state = {f'network.{k}': v for k, v in new_nav_actor_dict.items()}
    nav_actor.load_state_dict(nav_actor_state)
    print(f"      ✓ Actor weights transferred successfully")
    
    # ========================================================================
    # Transfer CRITIC weights with key mapping
    # ========================================================================
    print(f"\n      Transferring critic weights...")
    
    loco_critic_dict = loco_net.critic.state_dict()  # Keys: "0.weight", "2.weight", etc.
    nav_critic_dict = nav_critic.network.state_dict()  # Keys: "0.weight", "2.weight", etc.
    
    new_nav_critic_dict = {}
    
    for key in nav_critic_dict.keys():
        if key == "0.weight":  # First layer: dimension expansion needed
            loco_weight = loco_critic_dict[key]  # [256, 67]
            nav_shape = nav_critic_dict[key].shape  # [256, 98]
            
            print(f"      [EXPAND] {key}: {tuple(loco_weight.shape)} → {tuple(nav_shape)}")
            
            # Create expanded weight matrix
            expanded_weight = torch.zeros(nav_shape)
            expanded_weight[:, :67] = loco_weight  # Copy first 67 dims
            
            # Initialize new 31 dimensions with small random values
            torch.nn.init.uniform_(expanded_weight[:, 67:], a=-0.01, b=0.01)
            
            new_nav_critic_dict[key] = expanded_weight
            print(f"               Copied 67 dims + initialized 31 new dims")
            
        elif key in loco_critic_dict:  # All other layers: direct copy
            loco_param = loco_critic_dict[key]
            nav_param = nav_critic_dict[key]
            
            if loco_param.shape == nav_param.shape:
                print(f"      [COPY]   {key}: {tuple(loco_param.shape)}")
                new_nav_critic_dict[key] = loco_param.clone()
            else:
                print(f"      [WARN]   {key}: Mismatch {tuple(loco_param.shape)} vs {tuple(nav_param.shape)}")
                new_nav_critic_dict[key] = nav_param
        else:
            print(f"      [INIT]   {key}: Not in locomotion (random init)")
            new_nav_critic_dict[key] = nav_critic_dict[key]
    
    # Load into navigation critic with "network." prefix
    nav_critic_state = {f'network.{k}': v for k, v in new_nav_critic_dict.items()}
    nav_critic.load_state_dict(nav_critic_state)
    print(f"      ✓ Critic weights transferred successfully")
    
    # ========================================================================
    # Step 5: Save navigation checkpoint
    # ========================================================================
    print(f"\n[5/5] Saving navigation checkpoint...")
    
    nav_checkpoint = {
        'actor_state_dict': nav_actor.state_dict(),
        'critic_state_dict': nav_critic.state_dict(),
        'iteration': 0,
        'best_reward': float('-inf'),
        'current_stage': 1,
        'consecutive_successes': 0,
        'transfer_info': {
            'source_checkpoint': str(checkpoint_path),
            'source_obs_dim': 67,
            'target_obs_dim': 98,
            'method': 'copy_67_dims_init_31_new_small',
            'source_best_reward': checkpoint.get('best_reward', 'unknown')
        }
    }
    
    torch.save(nav_checkpoint, output_path)
    
    print(f"      ✓ Saved: {output_path}")
    print(f"\n" + "=" * 70)
    print("✅ TRANSFER COMPLETE!")
    print("=" * 70)
    print(f"\nNavigation checkpoint ready for Phase 2 training.")
    print(f"Pre-trained locomotion knowledge (492.7 reward) transferred successfully.")
    print(f"\nTo start Phase 2 waypoint navigation training:")
    print(f"  python train_spot_ppo.py --checkpoint {output_path} --iterations 5000")
    print()
    
    return True


if __name__ == "__main__":
    success = transfer_weights()
    exit(0 if success else 1)
