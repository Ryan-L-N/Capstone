"""
Test Script: Find Maximum Safe Speed for Spot Robot
=====================================================
Tests Spot at various speeds to determine safe operating range.
"""

import numpy as np
import argparse

# Parse args before SimulationApp
parser = argparse.ArgumentParser(description="Spot Speed Test")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
args = parser.parse_args()

# Isaac Sim setup
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# Import Isaac modules
import omni
from omni.isaac.core import World
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
from pxr import UsdGeom, UsdLux, Gf

print("=" * 70)
print("SPOT ROBOT SPEED TEST - Finding Maximum Safe Speed")
print("=" * 70)

# World setup - EXACT same as Cole_vs1
world = World(
    physics_dt=1.0 / 500.0,
    rendering_dt=10.0 / 500.0,
    stage_units_in_meters=1.0
)
stage = omni.usd.get_context().get_stage()

# Add lighting
dome_light = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
dome_light.CreateIntensityAttr(1000.0)
dome_light.CreateColorAttr(Gf.Vec3f(0.95, 0.95, 1.0))

# Add ground plane
world.scene.add_default_ground_plane(
    z_position=0,
    name="default_ground_plane",
    prim_path="/World/defaultGroundPlane",
    static_friction=0.5,
    dynamic_friction=0.5,
    restitution=0.01,
)

# Test parameters
SPEEDS_TO_TEST = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
TEST_DURATION = 20.0  # seconds per test

# Start point
START_X = -45.0
START_Y = 0.0
START_Z = 0.7

# Goal position
GOAL = np.array([45.0, 0.0])

def get_robot_state(robot):
    pos, quat = robot.robot.get_world_pose()
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    heading = np.arctan2(siny_cosp, cosy_cosp)
    vel = robot.robot.get_linear_velocity()
    return np.array(pos), heading, vel

def test_speed(target_speed):
    """Test a specific speed for TEST_DURATION seconds"""
    print(f"\n{'='*70}")
    print(f"Testing Speed: {target_speed} m/s")
    print(f"{'='*70}")
    
    # Create Spot for this test
    spot = SpotFlatTerrainPolicy(
        prim_path="/World/Spot",
        name="Spot",
        position=np.array([START_X, START_Y, START_Z]),
    )
    
    # CRITICAL: Reset world BEFORE accessing spot properties
    world.reset()
    
    # Initialize the robot
    spot.initialize()
    spot.robot.set_joints_default_state(spot.default_pos)
    
    # Run physics steps for stability
    PHYSICS_DT = 1.0 / 500.0
    for _ in range(10):
        world.step(render=False)
    
    sim_time = 0.0
    crashed = False
    max_distance = 0.0
    final_pos = None
    
    start_pos, _, _ = get_robot_state(spot)
    
    try:
        while simulation_app.is_running() and sim_time < TEST_DURATION:
            pos, heading, vel = get_robot_state(spot)
            
            # Check for crash
            if pos[2] < 0.25:
                crashed = True
                print(f"  ❌ CRASHED at t={sim_time:.2f}s, Z={pos[2]:.3f}m")
                break
            
            # Simple go-to-goal with fixed speed
            to_goal = GOAL - pos[:2]
            dist_remaining = np.linalg.norm(to_goal)
            desired_heading = np.arctan2(to_goal[1], to_goal[0])
            heading_error = desired_heading - heading
            
            while heading_error > np.pi:
                heading_error -= 2 * np.pi
            while heading_error < -np.pi:
                heading_error += 2 * np.pi
            
            turn_rate = 2.0 * heading_error
            turn_rate = np.clip(turn_rate, -1.0, 1.0)
            
            command = np.array([target_speed, 0.0, turn_rate])
            spot.forward(PHYSICS_DT, command)
            
            world.step(render=not args.headless)
            sim_time += PHYSICS_DT
            
            distance_traveled = pos[0] - start_pos[0]
            max_distance = max(max_distance, distance_traveled)
            final_pos = pos
            
            # Print every 5 seconds
            if int(sim_time) % 5 == 0 and sim_time - int(sim_time) < PHYSICS_DT * 2:
                vel_magnitude = np.linalg.norm(vel)
                print(f"  t={sim_time:6.2f}s | X={pos[0]:7.2f}m | Distance:{distance_traveled:7.2f}m | "
                      f"Vel:{vel_magnitude:5.2f} m/s | Z: {pos[2]:.3f}m")
    
    except Exception as e:
        print(f"  ⚠️ Exception during test: {e}")
        crashed = True
    
    # Print results
    if crashed:
        status = "✗ CRASHED"
    else:
        status = "✓ STABLE"
    
    if final_pos is not None:
        print(f"\n  Results for {target_speed} m/s:")
        print(f"    Max Distance: {max_distance:.2f}m")
        print(f"    Status: {status}")
        print(f"    Final Position: X={final_pos[0]:.2f}m, Z={final_pos[2]:.3f}m")
    
    # Clean up for next test - remove Spot
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath("/World/Spot"):
        stage.RemovePrim("/World/Spot")
    
    return max_distance, not crashed, final_pos

# Main test loop
results = []
for speed in SPEEDS_TO_TEST:
    distance, stable, final_pos = test_speed(speed)
    results.append((speed, distance, stable))

# Print summary
print(f"\n{'='*70}")
print("SPEED TEST SUMMARY")
print(f"{'='*70}")
print(f"{'Speed (m/s)':15} {'Stable':15} {'Max Distance':20}")
print("-" * 70)

max_safe_speed = 0.0
for speed, distance, stable in results:
    status = "✓ YES" if stable else "✗ NO"
    print(f"{speed:<15.1f} {status:<15} {distance:<20.2f}m")
    if stable and speed > max_safe_speed:
        max_safe_speed = speed

print(f"\n📊 RECOMMENDATION:")
print(f"   Maximum Safe Speed: {max_safe_speed} m/s")
print(f"   Recommended Operating Range: [{0.5 if max_safe_speed > 0 else 0.5}, {max_safe_speed:.1f}] m/s")
if max_safe_speed > 0:
    print(f"   Default Training Speed: {min(1.5, max_safe_speed)} m/s")
else:
    print(f"   Default Training Speed: 1.5 m/s (needs investigation)")
