#!/usr/bin/env python3
"""
VS3 Moderate Run - Interactive GUI Viewer
Tests the trained moderate policy with explicit GUI window management
"""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=3)
args = parser.parse_args()

# CRITICAL: Initialize SimulationApp BEFORE any other Isaac Sim imports
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
})

# NOW import other modules after SimulationApp initialization
import torch
import numpy as np
import yaml
from omni.isaac.core import World
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
import omni

from navigation_policy import NavigationPolicy
from navigation_env import NavigationEnvironment

print("=" * 80)
print("VS3 MODERATE RUN - GUI VIEWER")
print("=" * 80)
print("Isaac Sim window should be opening now...\n")

try:
    # Load config
    config_path = Path("nav_config_moderate.yaml")
    with open(config_path, 'r') as f:
        nav_config = yaml.safe_load(f)
    
    # Create world
    physics_dt = nav_config['physics']['dt']
    rendering_dt = nav_config['physics']['rendering_dt']
    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=rendering_dt)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()
    world.reset()
    
    print("Isaac Sim world created")
    
    # Create Spot
    start_pos = np.array(nav_config['robot']['start_position'])
    start_ori = np.array(nav_config['robot']['start_orientation'])
    spot = SpotFlatTerrainPolicy(
        prim_path="/World/Spot",
        name="Spot",
        position=start_pos
    )
    spot.initialize()
    spot.robot.set_joints_default_state(spot.default_pos)
    print("Spot robot created in viewport")
    
    # Create environment
    env = NavigationEnvironment(world, stage, spot, str(config_path))
    env.current_stage = 0  # Stage 1 - Stability Foundation
    
    # Physics callback
    def on_physics_step(step_size: float):
        env.apply_command(step_size)
    
    world.add_physics_callback("spot_navigation_control", on_physics_step)
    
    # Load policy (VS3 uses 75-dimensional observations)
    policy = NavigationPolicy(
        obs_dim=75,  # VS3 full observation space
        action_dim=3,
        hidden_dims=tuple(nav_config['network']['hidden_dims']),
        activation=nav_config['network']['activation']
    )
    
    checkpoint = torch.load("VS3_checkpoints/moderate/final_model.pt", map_location='cpu')
    policy.load_state_dict(checkpoint['policy_state_dict'])
    print("Policy loaded successfully\n")
    
    print("=" * 80)
    print("RUNNING EPISODES IN GUI VIEWER")
    print("=" * 80)
    print(f"Episodes: {args.episodes}")
    print("Task: Maintain stability for 180 seconds (no falls)")
    print("=" * 80 + "\n")
    
    # Run episodes
    for episode in range(args.episodes):
        print(f"[EPISODE {episode + 1}/{args.episodes}] Starting...", flush=True)
        
        obs = env.reset(stage_id=0)
        # VS3 env returns full 75-dim observation directly
        
        step = 0
        done = False
        
        while not done and step < 1000:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action, _, _ = policy.get_action(obs_tensor, deterministic=True)
            action = action.squeeze(0).detach().cpu().numpy()  # Detach and move to CPU before numpy()
            
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            step += 1
            
            # Update GUI every frame
            world.step(render=True)
            
            if step % 100 == 0:
                print(f"  Step {step}: Time={info['episode_time']:.1f}s | Falls={info.get('falls', 0)}", flush=True)
        
        print(f"  ✓ Episode {episode + 1} complete - Score: {info['score']:.1f}, Success: {info['success']}", flush=True)
        print()
    
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    simulation_app.close()
    print("\nGUI window closed")
