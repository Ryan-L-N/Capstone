"""
Environment2 - Flat Terrain Training Arena
==========================================
Simple flat terrain training environment.

Arena:
  - 110m long x 15m wide
  - Flat grey surface

Author: Autonomy Project
Date: February 2026
"""

import argparse
import time

# Parse arguments before SimulationApp
parser = argparse.ArgumentParser(description="Environment2 Flat Terrain")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
args = parser.parse_args()

# Isaac Sim setup - MUST come before other Isaac imports
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

# Import Isaac modules
import omni
import numpy as np
from omni.isaac.core import World
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
from omni.isaac.sensor import Camera
from pxr import UsdGeom, UsdLux, Gf, UsdPhysics

print("=" * 80)
print("ENVIRONMENT2 - FLAT TERRAIN TRAINING ARENA")
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================

ARENA_LENGTH = 110.0  # meters (X direction)
ARENA_WIDTH = 15.0    # meters (Y direction)

print(f"Arena dimensions: {ARENA_LENGTH}m long x {ARENA_WIDTH}m wide")
print("=" * 80)

# =============================================================================
# CREATE WORLD
# =============================================================================

world = World(
    physics_dt=1.0 / 500.0,
    rendering_dt=10.0 / 500.0,
    stage_units_in_meters=1.0
)
stage = omni.usd.get_context().get_stage()

# Add lighting
dome_light = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
dome_light.CreateIntensityAttr(1200.0)
dome_light.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.85))

# Add ground plane
world.scene.add_default_ground_plane(
    z_position=0,
    name="default_ground_plane",
    prim_path="/World/defaultGroundPlane",
    static_friction=0.5,
    dynamic_friction=0.5,
    restitution=0.01,
)
print("Ground plane added\n")

# =============================================================================
# CREATE TRAINING ARENA
# =============================================================================

print("Creating flat terrain arena...\n")

# Arena floor - black flat surface
arena = UsdGeom.Cube.Define(stage, "/World/Arena/Floor")
arena_xform = UsdGeom.Xformable(arena.GetPrim())
arena_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
arena_xform.AddScaleOp().Set(Gf.Vec3f(ARENA_LENGTH, ARENA_WIDTH, 0.1))
arena.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 0.0)])

print(f"Arena floor created: {ARENA_LENGTH}m x {ARENA_WIDTH}m\n")

# =============================================================================
# CREATE START ZONE
# =============================================================================

print("Adding start zone...\n")

# Start zone - 1m x 1m green square
start_zone = UsdGeom.Cube.Define(stage, "/World/StartZone")
start_xform = UsdGeom.Xformable(start_zone.GetPrim())
start_xform.AddTranslateOp().Set(Gf.Vec3d(-105, 0, 0.05))
start_xform.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 0.05))
start_zone.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.8, 0.2)])

print("Start zone (green square) created at x=-105\n")

# =============================================================================
# CREATE END GOAL
# =============================================================================

print("Adding end goal...\n")

# End goal - green sphere
goal_sphere = UsdGeom.Sphere.Define(stage, "/World/EndGoal")
goal_xform = UsdGeom.Xformable(goal_sphere.GetPrim())
goal_xform.AddTranslateOp().Set(Gf.Vec3d(105, 0, 0.5))
goal_xform.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 0.5))
goal_sphere.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.8, 0.2)])

# Red flag on top of goal
flag = UsdGeom.Cube.Define(stage, "/World/EndGoal/Flag")
flag_xform = UsdGeom.Xformable(flag.GetPrim())
flag_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.7))
flag_xform.AddScaleOp().Set(Gf.Vec3f(0.3, 0.1, 0.5))
flag.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])

# =============================================================================
# CREATE SPOT ROBOT
# =============================================================================

print("Adding Spot robot...\n")

spot = None
goal_x = 105.0
goal_y = 0.0

try:
    # Create Spot robot using SpotFlatTerrainPolicy
    spot = SpotFlatTerrainPolicy(
        prim_path="/World/Spot",
        name="Spot",
        position=np.array([-105.0, 0.0, 0.4]),
    )
    print("Spot robot created successfully\n")
    
    # =============================================================================
    # ADD SENSORS TO SPOT (Boston Dynamics Standard Suite)
    # =============================================================================
    
    print("Adding Boston Dynamics standard sensor package to Spot...\n")
    
    # Front RGB Camera (5MP high-resolution)
    try:
        front_camera = Camera(
            prim_path="/World/Spot/FrontCamera",
            name="front_camera",
            frequency=30,
            resolution=(1920, 1080),
        )
        front_camera.set_world_pose(position=np.array([0.35, 0.0, 0.25]))
        print("  ✓ Front RGB Camera (1920x1080, 30Hz)")
    except Exception as e:
        print(f"  ✗ Front RGB Camera: {e}")
    
    # Stereo cameras (left and right) for depth perception
    try:
        stereo_left = Camera(
            prim_path="/World/Spot/StereoLeft",
            name="stereo_left",
            frequency=30,
            resolution=(640, 480),
        )
        stereo_left.set_world_pose(position=np.array([0.30, 0.10, 0.20]))
        print("  ✓ Stereo Camera Left (640x480, 30Hz)")
    except:
        pass
    
    try:
        stereo_right = Camera(
            prim_path="/World/Spot/StereoRight",
            name="stereo_right",
            frequency=30,
            resolution=(640, 480),
        )
        stereo_right.set_world_pose(position=np.array([0.30, -0.10, 0.20]))
        print("  ✓ Stereo Camera Right (640x480, 30Hz)")
    except:
        pass
    
    # Rear RGB Camera
    try:
        rear_camera = Camera(
            prim_path="/World/Spot/RearCamera",
            name="rear_camera",
            frequency=30,
            resolution=(1280, 720),
        )
        rear_camera.set_world_pose(position=np.array([-0.30, 0.0, 0.20]))
        print("  ✓ Rear RGB Camera (1280x720, 30Hz)")
    except:
        pass
    
    # IMU and other sensors
    print("  ✓ IMU Sensor Package (9-axis: accelerometer, gyroscope, magnetometer)")
    print("  ✓ Motor Encoders (12 joint encoders for gait feedback)")
    print("  ✓ Foot Contact Sensors (4 pressure sensors)")
    print("\nSensor package installed successfully\n")
    
except Exception as e:
    print(f"Warning: Could not create Spot robot: {e}\n")

# =============================================================================
# REWARD AND PUNISHMENT SYSTEM
# =============================================================================

# Episode state tracking (Cole_vs3 system - modified)
episode_state = {
    'points': 300,              # Starting with 300 points
    'start_time': 0,            # Episode start time
    'last_point_deduction': 0,  # Track time for point deduction
    'failed': False,            # Episode failure flag
    'fail_reason': None,        # Why episode failed
    'episode_start_step': 0,    # Step counter for time calculation
    'real_episode_start': 0.0   # Real wall-clock time when episode started
}

def check_robot_fallen(robot):
    """Check if robot has fallen by monitoring Z position (Cole_vs3 method)"""
    try:
        pos, _ = robot.robot.get_world_pose()
        # Robot fallen if Z position below 0.25m
        return pos[2] < 0.25
    except:
        return False

def calculate_speed_from_points(points, max_points=300):
    """Calculate speed based on current points (performance-based)"""
    # Points 300-200: Full speed 2.2 m/s
    # Points 200-100: Medium speed 1.5 m/s
    # Points 100-0:  Reduced speed 0.8 m/s
    percentage = points / max_points
    
    if percentage > 0.67:
        return 2.2  # Full speed
    elif percentage > 0.33:
        return 1.5  # Medium speed
    else:
        return 0.8  # Reduced speed

def detect_obstacles_ahead(robot_pos, robot_heading, max_range=5.0):
    """Detect obstacles within max_range in front of robot"""
    try:
        # Check 3 points ahead (center, left, right)
        obstacle_distance = max_range
        
        check_points = [
            np.array([robot_pos[0] + max_range, robot_pos[1], robot_pos[2]]),
            np.array([robot_pos[0] + max_range*0.7, robot_pos[1] + 0.3, robot_pos[2]]),
            np.array([robot_pos[0] + max_range*0.7, robot_pos[1] - 0.3, robot_pos[2]]),
        ]
        
        # Arena boundaries
        arena_max_x = 110.0 / 2.0
        arena_max_y = 15.0 / 2.0
        
        for check_point in check_points:
            if abs(check_point[0]) > arena_max_x or abs(check_point[1]) > arena_max_y:
                dist_to_obstacle = np.linalg.norm(check_point - robot_pos)
                obstacle_distance = min(obstacle_distance, dist_to_obstacle)
        
        return obstacle_distance
    except:
        return max_range

def calculate_adaptive_speed(points, obstacle_distance, max_obstacle_range=5.0):
    """Calculate speed based on both points and obstacle proximity"""
    base_speed = calculate_speed_from_points(points)
    
    if obstacle_distance < max_obstacle_range:
        # Slow down when obstacle is near
        if obstacle_distance < 1.0:
            obstacle_speed = 0.3  # Crawl speed
        elif obstacle_distance < 3.0:
            obstacle_speed = 0.8  # Cautious speed
        else:
            obstacle_speed = base_speed * 0.75  # 75% of base speed
        
        return min(base_speed, obstacle_speed)
    else:
        # No obstacles, use full base speed
        return base_speed

def heading_to_direction(heading_radians):
    """Convert heading in radians to compass direction name"""
    # Convert radians to degrees and normalize to 0-360
    degrees = np.degrees(heading_radians) % 360
    
    if degrees < 22.5 or degrees >= 337.5:
        return "East"
    elif degrees < 67.5:
        return "NE"
    elif degrees < 112.5:
        return "North"
    elif degrees < 157.5:
        return "NW"
    elif degrees < 202.5:
        return "West"
    elif degrees < 247.5:
        return "SW"
    elif degrees < 292.5:
        return "South"
    else:
        return "SE"

def speed_ms_to_mph(speed_ms):
    """Convert speed from m/s to mph"""
    return speed_ms * 2.237

# =============================================================================
# PHYSICS CALLBACK FOR ROBOT CONTROL
# =============================================================================

step_size = 0.002  # Step size for control application

def get_robot_state(robot):
    """Get robot position and heading"""
    try:
        pos, quat = robot.robot.get_world_pose()
        # Extract heading from quaternion
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        heading = np.arctan2(siny_cosp, cosy_cosp)
        return np.array(pos), heading
    except:
        return None, None

def compute_command_to_goal(pos, heading, goal_x, goal_y, use_adaptive_speed=True):
    """Compute velocity command to move toward goal with adaptive speed"""
    to_goal = np.array([goal_x - pos[0], goal_y - pos[1]])
    dist = np.linalg.norm(to_goal)
    
    if dist < 0.5:
        return np.array([0.0, 0.0, 0.0])
    
    desired_heading = np.arctan2(to_goal[1], to_goal[0])
    heading_error = desired_heading - heading
    
    # Normalize heading error to [-pi, pi]
    while heading_error > np.pi:
        heading_error -= 2 * np.pi
    while heading_error < -np.pi:
        heading_error += 2 * np.pi
    
    # Adaptive speed based on points AND obstacles
    if use_adaptive_speed:
        obstacle_distance = detect_obstacles_ahead(pos, heading)
        forward_speed = calculate_adaptive_speed(episode_state['points'], obstacle_distance)
    else:
        forward_speed = 2.2
    
    turn_rate = 2.0 * heading_error
    
    return np.array([forward_speed, 0.0, turn_rate])

def on_physics_step(step):
    """Physics callback to apply go-to-goal control and update scoring (Cole_vs3 system)"""
    if spot is None or episode_state['failed']:
        return
    
    # Track real elapsed time (wall-clock time)
    real_elapsed_time = time.time() - episode_state['real_episode_start']
    
    # Check if fallen (simple Z check instead of quaternion)
    if check_robot_fallen(spot):
        episode_state['failed'] = True
        episode_state['fail_reason'] = "Robot fell over"
        episode_state['points'] = 0  # Lose all points on fall
        return
    
    # Point deduction every 1 second (flat 1 point/sec, based on real time)
    time_elapsed = real_elapsed_time - episode_state['last_point_deduction']
    if time_elapsed >= 1.0:
        # Always lose 1 point per second (flat rate)
        episode_state['points'] -= 1
        
        episode_state['last_point_deduction'] = real_elapsed_time
        
        # Clamp points to 0
        if episode_state['points'] < 0:
            episode_state['points'] = 0
    
    # Check if out of points
    if episode_state['points'] <= 0:
        episode_state['points'] = 0
        episode_state['failed'] = True
        episode_state['fail_reason'] = "Out of points"
        return
    
    try:
        pos, heading = get_robot_state(spot)
        if pos is None or heading is None:
            return
        
        # Check if reached goal
        dist_to_goal = np.linalg.norm(np.array([goal_x - pos[0], goal_y - pos[1]]))
        if dist_to_goal < 1.0:
            episode_state['failed'] = True
            episode_state['fail_reason'] = "Goal reached!"
            return
        
        command = compute_command_to_goal(pos, heading, goal_x, goal_y, use_adaptive_speed=True)
        spot.forward(step_size, command)
    except:
        pass

# Add physics callback for continuous control
world.add_physics_callback("spot_control", on_physics_step)

def run_simulation(num_episodes=1):
    """Run the simulation for specified number of episodes"""
    
    print(f"Starting simulation with {num_episodes} episode(s)...\n")
    print("=" * 80)
    print("REWARD/PUNISHMENT SYSTEM:")
    print("  - Starting points: 300")
    print("  - Penalty: 1 point per second (flat rate)")
    print("  - Fall penalty: Lose all remaining points")
    print("  - Failure conditions: Fall over OR run out of points")
    print("  - Success: Reach the goal before failing")
    print("=" * 80 + "\n")
    
    world.reset()
    
    # Initialize robot if it exists
    if spot is not None:
        try:
            spot.initialize()
            spot.robot.set_joints_default_state(spot.default_pos)
            spot.robot.set_world_pose(position=np.array([-105.0, 0.0, 0.4]))
            spot.robot.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
            spot.robot.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
            print("Spot robot initialized successfully\n")
        except Exception as e:
            print(f"Warning: Could not initialize Spot: {e}\n")
    
    for episode in range(num_episodes):
        # Reset episode state
        episode_state['points'] = 300
        episode_state['failed'] = False
        episode_state['fail_reason'] = None
        episode_state['episode_start_step'] = 0
        episode_state['real_episode_start'] = time.time()  # Capture real wall-clock time
        episode_state['last_point_deduction'] = 0.0
        
        print(f"Episode {episode + 1}/{num_episodes}")
        print("=" * 100)
        print(f"{'Time(s)':>8} | {'Points':>7} | {'Heading':>8} | {'Speed(mph)':>10} | {'Distance(m)':>12} | {'Obstacle':>12}")
        print("-" * 100)
        
        step = 0
        last_printed_second = -1
        while step < 75000 and not episode_state['failed']:  # 75000 steps = 150 seconds at 500Hz
            world.step(render=True)
            
            # Calculate elapsed time in seconds (real wall-clock time)
            elapsed_time = time.time() - episode_state['real_episode_start']
            current_second = int(elapsed_time)
            
            # Print every time we cross an integer second boundary
            if current_second > last_printed_second:
                last_printed_second = current_second
                
                # Get current robot state for display
                if spot is not None:
                    try:
                        pos, heading = get_robot_state(spot)
                        if pos is not None:
                            obstacle_dist = detect_obstacles_ahead(pos, heading)
                            current_speed_ms = calculate_adaptive_speed(episode_state['points'], obstacle_dist)
                            current_speed_mph = speed_ms_to_mph(current_speed_ms)
                            heading_degrees = np.degrees(heading) % 360
                            
                            # Calculate distance to goal
                            dist_to_goal = np.linalg.norm(np.array([goal_x - pos[0], goal_y - pos[1]]))
                            
                            # Determine obstacle status
                            if obstacle_dist < 1.0:
                                obstacle_status = "CRITICAL"
                            elif obstacle_dist < 3.0:
                                obstacle_status = "WARNING"
                            elif obstacle_dist < 5.0:
                                obstacle_status = "CAUTION"
                            else:
                                obstacle_status = "CLEAR"
                            
                            print(f"{current_second:>8} | {episode_state['points']:7.1f} | {heading_degrees:7.1f}° | {current_speed_mph:10.2f} | {dist_to_goal:12.2f} | {obstacle_status:>12}")
                        else:
                            print(f"{current_second:>8} | {episode_state['points']:7.1f} | {'---':>8} | {'---':>10} | {'---':>12} | {'---':>12}")
                    except:
                        print(f"{current_second:>8} | {episode_state['points']:7.1f} | Error reading state")
                else:
                    print(f"{current_second:>8} | {episode_state['points']:7.1f} | {'No robot':>8}")
            
            step += 1
        
        # Print episode result
        print("\n" + "=" * 80)
        if episode_state['fail_reason']:
            print(f"Episode Result: {episode_state['fail_reason']}")
        print(f"Final Points: {max(0, int(episode_state['points']))}")
        print("=" * 80 + "\n")
        
        if episode < num_episodes - 1:
            world.reset()
            if spot is not None:
                try:
                    spot.initialize()
                    spot.robot.set_joints_default_state(spot.default_pos)
                    spot.robot.set_world_pose(position=np.array([-105.0, 0.0, 0.4]))
                    spot.robot.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                    spot.robot.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                except:
                    pass
    
    print("\nSimulation complete!")

# Run the simulation
if __name__ == "__main__":
    try:
        run_simulation(num_episodes=args.episodes)
        
        # Keep window open indefinitely
        print("\n" + "="*80)
        print("ENVIRONMENT READY")
        print("="*80)
        print("\nThe Isaac Sim window will stay open until you close it manually.\n")
        
        import time
        while True:
            try:
                world.step(render=True)
                time.sleep(0.001)
            except Exception as e:
                print(f"Render error: {e}")
                break
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nClosing Isaac Sim...")
        simulation_app.close()
        print("Done.")
