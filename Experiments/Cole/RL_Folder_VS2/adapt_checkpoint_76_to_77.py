#!/usr/bin/env python3
"""
Adapt Run 6 checkpoint from 76-dim to 77-dim observation space
Adds contact_force dimension by initializing new weights to small random values
"""

import sys
import torch
import copy

def adapt_checkpoint(checkpoint_path, output_path):
    """
    Load a checkpoint trained with 76-dim observations and adapt it to 77-dim.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Find the actor and critic networks
    # Usually stored as 'policy' or 'actor' in the state dict
    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('policy_state_dict', checkpoint)
    
    print("\nOriginal state dict keys:")
    print([k for k in state_dict.keys() if 'weight' in k or 'bias' in k][:10])
    
    # Find first layer weights and bias (should map from observation to hidden)
    # Look for patterns like: actor.0.weight, policy.0.weight, mlp.0.weight, etc.
    
    first_layer_weight_key = None
    first_layer_bias_key = None
    
    for key in state_dict.keys():
        if '.0.weight' in key or ('fc' in key.lower() and '0' in key and 'weight' in key):
            # Check if this looks like it maps from 76 to something
            weight = state_dict[key]
            if weight.shape[1] == 76:  # Input dimension is 76
                first_layer_weight_key = key
                first_layer_bias_key = key.replace('weight', 'bias')
                print(f"\nFound first layer: {key}")
                print(f"  Weight shape: {weight.shape}")
                print(f"  Bias shape: {state_dict.get(first_layer_bias_key, 'N/A')}")
                break
    
    if first_layer_weight_key is None:
        print("\nERROR: Could not find first layer with input dimension 76")
        print(f"Available weight shapes:")
        for key in state_dict.keys():
            if 'weight' in key and len(state_dict[key].shape) == 2:
                print(f"  {key}: {state_dict[key].shape}")
        return False
    
    # Adapt the weights
    old_weight = state_dict[first_layer_weight_key]  # Shape: (hidden_dim, 76)
    old_bias = state_dict[first_layer_bias_key] if first_layer_bias_key in state_dict else None
    
    # Create new weight matrix with 77 input dimension
    hidden_dim = old_weight.shape[0]
    new_weight = torch.zeros(hidden_dim, 77, dtype=old_weight.dtype, device=old_weight.device)
    
    # Copy old weights for first 76 dimensions
    new_weight[:, :76] = old_weight
    
    # Initialize new contact_force dimension to small random values
    # Use Xavier uniform initialization scaled down
    new_weight[:, 76].uniform_(-0.01, 0.01)
    
    # Update state dict
    state_dict[first_layer_weight_key] = new_weight
    
    print(f"\nAdapted weight from {old_weight.shape} to {new_weight.shape}")
    if old_bias is not None:
        print(f"Bias shape remains: {old_bias.shape}")
    
    # Update checkpoint
    if 'model_state_dict' in checkpoint:
        checkpoint['model_state_dict'] = state_dict
    elif 'policy_state_dict' in checkpoint:
        checkpoint['policy_state_dict'] = state_dict
    else:
        checkpoint = state_dict
    
    # Save adapted checkpoint
    print(f"\nSaving adapted checkpoint to: {output_path}")
    torch.save(checkpoint, output_path)
    print("✓ Checkpoint adapted successfully!")
    
    return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        checkpoint_path = "checkpoints/run_6_aggressive/final_model.pt"
        output_path = "checkpoints/run_6_aggressive/final_model_adapted.pt"
    else:
        checkpoint_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else checkpoint_path.replace('.pt', '_adapted.pt')
    
    print("=" * 70)
    print("Checkpoint Dimension Adapter: 76-dim → 77-dim (add contact_force)")
    print("=" * 70)
    
    success = adapt_checkpoint(checkpoint_path, output_path)
    sys.exit(0 if success else 1)
