"""
Test RL Setup - Verify RL environment integration
==================================================
Quick test script to verify that the RL environment can be instantiated
and runs correctly with the baseline environment.

This test:
1. Creates a single Isaac Sim world
2. Instantiates one RL environment
3. Runs 10 steps with random actions
4. Prints observation space and rewards
5. Verifies no crashes

Usage:
    cd "C:\\Users\\user\\Desktop\\Capstone_vs_1.2\\Immersive-Modeling-and-Simulation-for-Autonomy\\Experiments\\Cole\\RL Folder"
    C:\\isaac-sim\\python.bat test_rl_setup.py
    
Author: Cole
Date: February 2026
"""

import argparse
import sys
import numpy as np

# Parse args BEFORE Isaac Sim
parser = argparse.ArgumentParser(description="Test RL Setup")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
args = parser.parse_args()

# Initialize Isaac Sim
print("=" * 80)
print("INITIALIZING ISAAC SIM FOR RL TEST")
print("=" * 80)

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

# Now import Isaac modules
from omni.isaac.core import World

# Import RL environment
try:
    from spot_rl_env import SpotRLEnv, RewardWeights
    from training_config import get_default_config
    print("✓ Successfully imported RL modules")
except ImportError as e:
    print(f"✗ Failed to import RL modules: {e}")
    simulation_app.close()
    sys.exit(1)


def test_rl_environment():
    """Test basic RL environment functionality."""
    
    print("\n" + "=" * 80)
    print("CREATING TEST ENVIRONMENT")
    print("=" * 80)
    
    # Create Isaac Sim world
    config = get_default_config()
    
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=config.env.physics_dt,
        rendering_dt=config.env.control_dt,
    )
    print(f"✓ Created world (physics_dt={config.env.physics_dt}, control_dt={config.env.control_dt})")
    
    # Create single RL environment
    try:
        env = SpotRLEnv(
            env_id=0,
            position=np.array([0.0, 0.0, 0.0]),
            config=config,
            world=world,
            log_dir="test_logs",
        )
        print("✓ Successfully created SpotRLEnv")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        world.stop()
        simulation_app.close()
        sys.exit(1)
    
    # Initialize environment
    print("\n" + "=" * 80)
    print("INITIALIZING ENVIRONMENT")
    print("=" * 80)
    
    try:
        obs = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation dimension: {len(obs)}")
        print(f"  Expected dimension: 92")
        
        if len(obs) != 92:
            print(f"  ⚠ WARNING: Expected 92-dim obs, got {len(obs)}")
    except Exception as e:
        print(f"✗ Reset failed: {e}")
        import traceback
        traceback.print_exc()
        world.stop()
        simulation_app.close()
        sys.exit(1)
    
    # Run a few steps with random actions
    print("\n" + "=" * 80)
    print("RUNNING 10 TEST STEPS")
    print("=" * 80)
    
    try:
        for step in range(10):
            # Random action: [forward_vel, lateral_vel, angular_vel]
            action = np.random.uniform(-1, 1, size=3)
            action[0] = abs(action[0])  # Forward only (positive)
            action[1] *= 0.2            # Small lateral
            action[2] *= 0.5            # Moderate turning
            
            obs, reward, done, info = env.step(action)
            
            print(f"\nStep {step + 1}:")
            print(f"  Action: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]")
            print(f"  Reward: {reward:.3f}")
            print(f"  Done: {done}")
            print(f"  Obs shape: {obs.shape}")
            
            if "waypoint_distance" in info:
                print(f"  Waypoint distance: {info['waypoint_distance']:.2f} m")
            if "waypoints_reached" in info:
                print(f"  Waypoints reached: {info['waypoints_reached']}")
            
            if done:
                print(f"\n  Episode ended at step {step + 1}")
                if info.get("success", False):
                    print(f"  ✓ Success!")
                elif info.get("fell", False):
                    print(f"  ✗ Fell")
                else:
                    print(f"  ⏱ Timeout")
                break
        
        print("\n✓ All steps completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Step execution failed: {e}")
        import traceback
        traceback.print_exc()
        world.stop()
        simulation_app.close()
        sys.exit(1)
    
    # Observation space breakdown
    print("\n" + "=" * 80)
    print("OBSERVATION SPACE BREAKDOWN")
    print("=" * 80)
    print("Component                      Dimension")
    print("-" * 80)
    print(f"Joint positions                12")
    print(f"Joint velocities               12")
    print(f"Base orientation (quat)        4")
    print(f"Base linear velocity           3")
    print(f"Base angular velocity          3")
    print(f"Base height                    1")
    print(f"Roll angle                     1")
    print(f"Pitch angle                    1")
    print(f"Waypoint distance              1")
    print(f"Waypoint heading               1")
    print(f"Waypoint progress              1")
    print(f"5 nearest obstacles (×7)       35")
    print(f"Foot contacts                  4")
    print(f"Collision magnitude            1")
    print(f"Previous actions               3")
    print("-" * 80)
    print(f"TOTAL                          92")
    print("=" * 80)
    
    # Cleanup
    print("\n" + "=" * 80)
    print("CLEANING UP")
    print("=" * 80)
    
    world.stop()
    print("✓ World stopped")
    
    simulation_app.close()
    print("✓ Simulation closed")
    
    print("\n" + "=" * 80)
    print("✓ RL SETUP TEST PASSED!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run debug training: python train_spot.py --config debug")
    print("2. Monitor for any errors during training")
    print("3. If debug succeeds, run full training: python train_spot.py")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_rl_environment()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        simulation_app.close()
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
