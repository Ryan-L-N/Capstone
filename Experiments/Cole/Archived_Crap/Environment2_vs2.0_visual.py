"""
Environment2 v2.0 - 8 Side-by-Side Corridors
============================================
Training arena with 8 parallel corridors separated by small walls.

Arena Layout:
  - 8 corridors in a row (side by side)
  - Each corridor: 110m long x 10m wide
  - Dividing walls: 1m tall x 0.5m wide between corridors
  - Total width: ~90m (8×10m + 7×0.5m walls)
  - Total length: 110m

Author: Advanced Robotics
Date: February 2026
Version: 2.0 - Side-by-Side Corridors
"""

import numpy as np
import argparse
import math

# Parse arguments before SimulationApp
parser = argparse.ArgumentParser(description="Environment2 v2.0: 8 Side-by-Side Corridors")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
args = parser.parse_args()

# Set up device
device = "cuda" if True else "cpu"
print(f"Using device: {device}")

# Isaac Sim setup - MUST come before other Isaac imports
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

# Import Isaac modules
import omni
from omni.isaac.core import World
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
from pxr import UsdGeom, UsdLux, Gf, UsdPhysics, Sdf
import os

print("=" * 80)
print("ENVIRONMENT 2 v2.0 - 8 SIDE-BY-SIDE CORRIDORS")
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Corridor dimensions
CORRIDOR_LENGTH = 110.0  # meters (X direction)
CORRIDOR_WIDTH = 15.0    # meters (Y direction)
NUM_CORRIDORS = 6

# Terrain type labels for each corridor
TERRAIN_TYPES = [
    "flat_terrain",      # Corridor 0
    "beach_sand",        # Corridor 1
    "mud",               # Corridor 2
    "grass",             # Corridor 3
    "tall_grass",        # Corridor 4
    "gravel",            # Corridor 5
]

# Go-to-goal controller parameters
TURN_GAIN = 2.0
FORWARD_SPEED = 2.2  # 5 mph ≈ 2.2 m/s
HEADING_THRESHOLD = 0.1

# Wall dimensions (dividing walls between corridors)
WALL_HEIGHT = 1.0        # 1 meter tall
WALL_WIDTH = 0.5         # 0.5 meter wide
WALL_THICKNESS = 0.1     # thin dividers

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

def compute_command_to_goal(pos, heading, goal_x, goal_y):
    """Compute velocity command to move toward goal at constant speed"""
    to_goal = np.array([goal_x - pos[0], goal_y - pos[1]])
    dist = np.linalg.norm(to_goal)
    
    if dist < 0.5:
        # At goal - stop
        return np.array([0.0, 0.0, 0.0])
    
    desired_heading = np.arctan2(to_goal[1], to_goal[0])
    heading_error = desired_heading - heading
    
    # Normalize heading error to [-pi, pi]
    while heading_error > np.pi:
        heading_error -= 2 * np.pi
    while heading_error < -np.pi:
        heading_error += 2 * np.pi
    
    # Compute turn rate - gentle heading correction
    turn_rate = TURN_GAIN * heading_error
    turn_rate = np.clip(turn_rate, -1.0, 1.0)
    
    # Maintain constant forward speed regardless of heading error
    forward_speed = FORWARD_SPEED
    
    return np.array([forward_speed, 0.0, turn_rate])

print(f"Corridor dimensions: {CORRIDOR_LENGTH}m long x {CORRIDOR_WIDTH}m wide")
print(f"Number of corridors: {NUM_CORRIDORS}")
print(f"Dividing walls: 1.0m tall x 0.5m wide")
print("\nTerrain Types by Corridor:")
for i, terrain in enumerate(TERRAIN_TYPES):
    print(f"  Corridor {i}: {terrain}")
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
# CREATE CORRIDORS AND WALLS
# =============================================================================

print("Creating corridors...\n")

# Calculate total arena dimensions
total_width = (NUM_CORRIDORS * CORRIDOR_WIDTH) + ((NUM_CORRIDORS - 1) * WALL_WIDTH)
print(f"Arena dimensions: {CORRIDOR_LENGTH}m long x {total_width}m wide\n")

# Create each corridor
for i in range(NUM_CORRIDORS):
    # Get terrain type for this corridor
    terrain_type = TERRAIN_TYPES[i]
    
    # Calculate Y position for this corridor
    # Position corridors so they're centered at origin
    corridor_center_y = (i * (CORRIDOR_WIDTH + WALL_WIDTH)) - (total_width / 2.0) + (CORRIDOR_WIDTH / 2.0)
    
    # Corridor base
    corridor = UsdGeom.Cube.Define(stage, f"/World/{terrain_type}/{terrain_type}")
    corridor_xform = UsdGeom.Xformable(corridor.GetPrim())
    corridor_xform.AddTranslateOp().Set(Gf.Vec3d(0, corridor_center_y, 0))
    corridor_xform.AddScaleOp().Set(Gf.Vec3f(CORRIDOR_LENGTH, CORRIDOR_WIDTH, 0.1))
    corridor.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.2, 0.2)])

print(f"Created {NUM_CORRIDORS} corridors\n")

# Add obstacles to specific corridors
print("Adding obstacles...\n")

# Beach sand obstacle in corridor 1 (beach_sand)
# Corridor 1 center Y position
beach_sand_center_y = (1 * (CORRIDOR_WIDTH + WALL_WIDTH)) - (total_width / 2.0) + (CORRIDOR_WIDTH / 2.0)

# Create beach sand obstacle - fills center 100m of 110m corridor
beach_sand_obstacle = UsdGeom.Cube.Define(stage, f"/World/beach_sand/Obstacle")
obstacle_xform = UsdGeom.Xformable(beach_sand_obstacle.GetPrim())
obstacle_xform.AddTranslateOp().Set(Gf.Vec3d(0, beach_sand_center_y, 0.05))
# Use 7m width to stay within corridor boundaries, 0.1m height for ground-level appearance
obstacle_xform.AddScaleOp().Set(Gf.Vec3f(100, 7.0, 0.1))
# Beach sand color (tan/beige)
beach_sand_obstacle.GetDisplayColorAttr().Set([Gf.Vec3f(0.76, 0.7, 0.50)])

print("Created beach sand obstacle in beach_sand corridor\n")

# Mud obstacle in corridor 2 (mud)
# Corridor 2 center Y position
mud_center_y = (2 * (CORRIDOR_WIDTH + WALL_WIDTH)) - (total_width / 2.0) + (CORRIDOR_WIDTH / 2.0)

# Create mud obstacle - fills center 100m of 110m corridor
mud_obstacle = UsdGeom.Cube.Define(stage, f"/World/mud/Obstacle")
mud_xform = UsdGeom.Xformable(mud_obstacle.GetPrim())
mud_xform.AddTranslateOp().Set(Gf.Vec3d(0, mud_center_y, 0.05))
# Use 7m width to stay within corridor boundaries, 0.1m height for ground-level appearance
mud_xform.AddScaleOp().Set(Gf.Vec3f(100, 7.0, 0.15))
# Mud color (dark brown)
mud_obstacle.GetDisplayColorAttr().Set([Gf.Vec3f(0.25, 0.15, 0.08)])

print("Created mud obstacle in mud corridor\n")
# Corridor 3 center Y position
grass_center_y = (3 * (CORRIDOR_WIDTH + WALL_WIDTH)) - (total_width / 2.0) + (CORRIDOR_WIDTH / 2.0)

# Create grass obstacle - fills center 100m of 110m corridor
grass_obstacle = UsdGeom.Cube.Define(stage, f"/World/grass/Obstacle")
grass_xform = UsdGeom.Xformable(grass_obstacle.GetPrim())
grass_xform.AddTranslateOp().Set(Gf.Vec3d(0, grass_center_y, 0.05))
# Use 7m width to stay within corridor boundaries, 0.1m height for ground-level appearance
grass_xform.AddScaleOp().Set(Gf.Vec3f(100, 7.0, 0.15))
# Short grass color (green)
grass_obstacle.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.6, 0.2)])

print("Created grass obstacle in grass corridor\n")

# Tall grass obstacle in corridor 4 (tall_grass)
# Corridor 4 center Y position
tall_grass_center_y = (4 * (CORRIDOR_WIDTH + WALL_WIDTH)) - (total_width / 2.0) + (CORRIDOR_WIDTH / 2.0)

# Create tall grass obstacle - fills center 100m of 110m corridor, 1 meter tall
tall_grass_obstacle = UsdGeom.Cube.Define(stage, f"/World/tall_grass/Obstacle")
tall_grass_xform = UsdGeom.Xformable(tall_grass_obstacle.GetPrim())
tall_grass_xform.AddTranslateOp().Set(Gf.Vec3d(0, tall_grass_center_y, 0.5))
# Use 7m width to stay within corridor boundaries, 1.0m height for tall brush
tall_grass_xform.AddScaleOp().Set(Gf.Vec3f(100, 7.0, 1.0))
# Tall grass color (darker green)
tall_grass_obstacle.GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.35, 0.1)])

print("Created tall grass obstacle in tall_grass corridor\n")

# Gravel obstacle in corridor 5 (gravel)
# Corridor 5 center Y position
gravel_center_y = (5 * (CORRIDOR_WIDTH + WALL_WIDTH)) - (total_width / 2.0) + (CORRIDOR_WIDTH / 2.0)

# Create gravel with spheres ranging from 1/2" to 3" diameter (0.0127m to 0.0762m)
# For 0% ground coverage in the corridor (disabled for performance)
min_diameter = 0.0127  # 1/2 inch in meters
max_diameter = 0.0762  # 3 inches in meters
spacing = 1000  # Set to large value to create no gravel

gravel_count = 0
for x in np.arange(-50, 50, spacing):
    for y in np.arange(gravel_center_y - 7.25, gravel_center_y + 7.25, spacing):
        # Randomize diameter (no fixed seed - different each generation)
        diameter = np.random.uniform(min_diameter, max_diameter)
        z_offset = diameter / 2  # Sit on ground
        
        # Create gravel sphere
        gravel = UsdGeom.Sphere.Define(stage, f"/World/gravel/Gravel{gravel_count}")
        gravel_xform = UsdGeom.Xformable(gravel.GetPrim())
        gravel_xform.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(z_offset)))
        gravel_xform.AddScaleOp().Set(Gf.Vec3f(diameter/2, diameter/2, diameter/2))
        # Gravel color (grey)
        gravel.GetDisplayColorAttr().Set([Gf.Vec3f(0.5, 0.5, 0.5)])
        
        gravel_count += 1

print(f"Created {gravel_count} gravel spheres (10% ground coverage) in gravel corridor\n")

# Add Spot robots to each corridor
print("Adding Spot robots to each corridor...\n")

for i in range(NUM_CORRIDORS):
    terrain_type = TERRAIN_TYPES[i]
    corridor_center_y = (i * (CORRIDOR_WIDTH + WALL_WIDTH)) - (total_width / 2.0) + (CORRIDOR_WIDTH / 2.0)
    
    # Create 1m x 1m square start marker at corridor edge
    start_marker = UsdGeom.Cube.Define(stage, f"/World/{terrain_type}/StartMarker")
    start_xform = UsdGeom.Xformable(start_marker.GetPrim())
    # Position at edge of corridor (x=-105), on the ground (z=0.05 for thin square)
    start_xform.AddTranslateOp().Set(Gf.Vec3d(-105, corridor_center_y, 0.05))
    start_xform.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 0.05))  # 1m x 1m x 0.1m square
    start_marker.GetDisplayColorAttr().Set([Gf.Vec3f(0.8, 0.2, 0.2)])  # Red for visibility
    
    # Goal marker at end of corridor - green sphere with red flag
    # Green sphere base at far end (x=105)
    goal_sphere = UsdGeom.Sphere.Define(stage, f"/World/{terrain_type}/GoalSphere")
    goal_sphere_xform = UsdGeom.Xformable(goal_sphere.GetPrim())
    goal_sphere_xform.AddTranslateOp().Set(Gf.Vec3d(105, corridor_center_y, 0.5))
    goal_sphere_xform.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 0.5))  # 1m diameter sphere
    goal_sphere.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.8, 0.2)])  # Green
    
    # Red flag on top of sphere
    flag = UsdGeom.Cube.Define(stage, f"/World/{terrain_type}/Flag")
    flag_xform = UsdGeom.Xformable(flag.GetPrim())
    flag_xform.AddTranslateOp().Set(Gf.Vec3d(105, corridor_center_y, 1.2))  # On top of sphere
    flag_xform.AddScaleOp().Set(Gf.Vec3f(0.1, 0.3, 0.3))  # Thin flag
    flag.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])  # Red flag

print(f"Added start markers and goal spheres with flags to all {NUM_CORRIDORS} corridors\n")

# =============================================================================
# ADD SPOT ROBOTS
# =============================================================================

print("Adding Spot robots to each corridor...\n")

spot_robots = []  # Store robot references for movement control

for i in range(NUM_CORRIDORS):
    terrain_type = TERRAIN_TYPES[i]
    corridor_center_y = (i * (CORRIDOR_WIDTH + WALL_WIDTH)) - (total_width / 2.0) + (CORRIDOR_WIDTH / 2.0)
    
    # Create Spot robot using SpotFlatTerrainPolicy
    try:
        spot = SpotFlatTerrainPolicy(
            prim_path=f"/World/Spot_{i}",
            name=f"Spot_{i}",
            position=np.array([-105, corridor_center_y, 0.4]),
        )
        
        spot_robots.append({
            'robot': spot,
            'prim_path': f"/World/Spot_{i}",
            'start_x': -105,
            'end_x': 105,
            'corridor_y': corridor_center_y,
            'current_x': -105
        })
        print(f"Added Spot robot to {terrain_type} corridor at position (-105, {corridor_center_y:.2f})")
    except Exception as e:
        print(f"Warning: Could not create Spot in {terrain_type}: {e}")

print(f"Added {len(spot_robots)} Spot robot(s)\n")

print("Creating dividing walls...\n")

for i in range(NUM_CORRIDORS - 1):
    # Calculate wall position (between corridor i and i+1)
    corridor_i_center = (i * (CORRIDOR_WIDTH + WALL_WIDTH)) - (total_width / 2.0) + (CORRIDOR_WIDTH / 2.0)
    corridor_next_center = ((i + 1) * (CORRIDOR_WIDTH + WALL_WIDTH)) - (total_width / 2.0) + (CORRIDOR_WIDTH / 2.0)
    wall_center_y = (corridor_i_center + corridor_next_center) / 2.0
    
    # Create wall
    wall = UsdGeom.Cube.Define(stage, f"/World/Wall{i}")
    wall_xform = UsdGeom.Xformable(wall.GetPrim())
    wall_xform.AddTranslateOp().Set(Gf.Vec3d(0, wall_center_y, WALL_HEIGHT / 2.0))
    wall_xform.AddScaleOp().Set(Gf.Vec3f(CORRIDOR_LENGTH, WALL_WIDTH, WALL_HEIGHT))
    wall.GetDisplayColorAttr().Set([Gf.Vec3f(0.5, 0.5, 0.5)])

print(f"Created {NUM_CORRIDORS - 1} dividing walls\n")

print("=" * 80)
print(f"Arena created: {NUM_CORRIDORS} corridors with {NUM_CORRIDORS - 1} walls")
print("\nCorridor Layout (Left to Right):")
for i, terrain in enumerate(TERRAIN_TYPES):
    print(f"  {i + 1}. {terrain}")
print("=" * 80)

# =============================================================================
# PHYSICS CALLBACK FOR ROBOT CONTROL
# =============================================================================

step_size = 0.002  # Step size for control application

def on_physics_step(step):
    """Physics callback to apply go-to-goal control to each robot"""
    for robot_data in spot_robots:
        spot = robot_data['robot']
        goal_x = robot_data['end_x']
        goal_y = robot_data['corridor_y']
        
        try:
            # Get current robot state
            pos, heading = get_robot_state(spot)
            if pos is None or heading is None:
                continue
            
            # Compute command to move toward goal
            command = compute_command_to_goal(pos, heading, goal_x, goal_y)
            
            # Apply command to robot
            spot.forward(step_size, command)
        except Exception as e:
            pass  # Silently continue if command fails

# Add physics callback for continuous control
world.add_physics_callback("spot_control", on_physics_step)

# =============================================================================
# SIMULATION LOOP
# =============================================================================

def run_simulation(num_episodes=1):
    """Run the simulation for specified number of episodes"""
    
    print(f"\nStarting simulation with {num_episodes} episode(s)...")
    print("Press ESC or close window to stop\n")
    
    world.reset()
    
    # Initialize all robots
    for robot_data in spot_robots:
        spot = robot_data['robot']
        spot.initialize()
        spot.robot.set_joints_default_state(spot.default_pos)
        # Set initial position
        spot.robot.set_world_pose(position=np.array([-105, robot_data['corridor_y'], 0.4]))
        spot.robot.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
        spot.robot.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
    
    print(f"Initialized {len(spot_robots)} Spot robot(s)\n")
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        
        for step in range(500):  # 500 steps per episode
            # Render world (physics callback will automatically apply control)
            world.step(render=True)
            
            if step % 100 == 0:
                print(f"  Step {step}/500")
        
        if episode < num_episodes - 1:
            world.reset()
            # Re-initialize robots for next episode
            for robot_data in spot_robots:
                spot = robot_data['robot']
                spot.initialize()
                spot.robot.set_joints_default_state(spot.default_pos)
                spot.robot.set_world_pose(position=np.array([-105, robot_data['corridor_y'], 0.4]))
                spot.robot.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                spot.robot.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
    
    print("\nSimulation complete!")

# Run the simulation
if __name__ == "__main__":
    try:
        run_simulation(num_episodes=args.episodes)
        
        # Keep window open indefinitely - don't close the app
        print("\n" + "="*80)
        print("ENVIRONMENT READY FOR VIEWING")
        print("="*80)
        print("\nThe Isaac Sim window will stay open until you close it manually.")
        print("You can interact with the viewport (rotate, zoom, pan).")
        print("Close the Isaac Sim window when done.\n")
        print("="*80 + "\n")
        
        import time
        while True:
            try:
                world.step(render=True)
                time.sleep(0.001)  # Small sleep to prevent CPU pegging
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
