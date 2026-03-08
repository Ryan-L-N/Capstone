"""
Testing Environment 1
====================
100m x 100m enclosed testing field with walls
Centered at (0, 0)
- X range: -50m to +50m
- Y range: -50m to +50m
- Surrounded by walls for containment
- Start: Fixed at (-45, 0) - 5m from west end, centered Y
- Goal: Randomized within field (75m+ away from start)
- Obstacles: 360 total (100 large furniture + 250 small clutter + 5 cars + 5 trucks)
- Obstacle types: couch, chair, table, shelf, ottoman, bed, cabinet, small_clutter, car, truck
Purpose: Test RL-trained Spot robot navigation through ultra-dense cluttered environment
"""

import numpy as np
import argparse

# Parse args before SimulationApp
parser = argparse.ArgumentParser(description="Testing Environment 1")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
args = parser.parse_args()

# Isaac Sim setup - MUST come before other Isaac imports
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# Isaac modules
import omni
from omni.isaac.core import World
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
from pxr import UsdGeom, UsdLux, Gf, UsdPhysics

# Import SpotRobot module
import sys
sys.path.insert(0, r'c:\Users\user\Desktop\Capstone_vs_1.2\Immersive-Modeling-and-Simulation-for-Autonomy\Experiments\Cole')
from Spots.Spot_1 import SpotRobot

print("=" * 70)
print("TESTING ENVIRONMENT 1: RL AGENT EVALUATION")
print("100m x 100m Enclosed Field with Randomized Targets (75m+ away)")
print("=" * 70)

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

FIELD_LENGTH = 100.0  # X direction
FIELD_WIDTH = 100.0   # Y direction
FIELD_CENTER_X = 0.0
FIELD_CENTER_Y = 0.0

# Boundaries (centered at origin)
FIELD_MIN_X = FIELD_CENTER_X - FIELD_LENGTH / 2  # -50m
FIELD_MAX_X = FIELD_CENTER_X + FIELD_LENGTH / 2  # +50m
FIELD_MIN_Y = FIELD_CENTER_Y - FIELD_WIDTH / 2   # -25m
FIELD_MAX_Y = FIELD_CENTER_Y + FIELD_WIDTH / 2   # +25m

WALL_HEIGHT = 2.0
WALL_THICKNESS = 0.5

# Fixed start position: 5m from west end, centered on Y
START_X = FIELD_MIN_X + 5.0  # -50 + 5 = -45m
START_Y = 0.0                # Center Y
START_Z = 0.7

print(f"Field Dimensions: {FIELD_LENGTH}m (X) × {FIELD_WIDTH}m (Y)")
print(f"Field Boundaries:")
print(f"  X: [{FIELD_MIN_X}, {FIELD_MAX_X}]m")
print(f"  Y: [{FIELD_MIN_Y}, {FIELD_MAX_Y}]m")
print(f"  Center: ({FIELD_CENTER_X}, {FIELD_CENTER_Y})")
print(f"\nStart Position (Fixed): ({START_X}, {START_Y})")
print(f"End Position: Randomized within field")
print("=" * 70)

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
print("[OK] Ground plane added")

# =============================================================================
# CREATE WALLS (Surrounding boundaries)
# =============================================================================

def create_wall(stage, name, x_min, x_max, y_min, y_max, thickness, height):
    """Create a wall with given bounds"""
    wall = UsdGeom.Mesh.Define(stage, f"/World/{name}")
    
    # Create box vertices for the wall
    points = [
        Gf.Vec3f(x_min, y_min, 0),
        Gf.Vec3f(x_max, y_min, 0),
        Gf.Vec3f(x_max, y_max, 0),
        Gf.Vec3f(x_min, y_max, 0),
        Gf.Vec3f(x_min, y_min, height),
        Gf.Vec3f(x_max, y_min, height),
        Gf.Vec3f(x_max, y_max, height),
        Gf.Vec3f(x_min, y_max, height),
    ]
    
    wall.GetPointsAttr().Set(points)
    wall.GetFaceVertexCountsAttr().Set([4, 4, 4, 4, 4, 4])
    wall.GetFaceVertexIndicesAttr().Set([
        0, 1, 2, 3,  # bottom
        4, 7, 6, 5,  # top
        0, 4, 5, 1,  # front
        2, 6, 7, 3,  # back
        0, 3, 7, 4,  # left
        1, 5, 6, 2,  # right
    ])
    wall.GetDisplayColorAttr().Set([Gf.Vec3f(0.7, 0.7, 0.7)])

# Create four walls (North, South, East, West)
wall_offset = WALL_THICKNESS / 2

# North wall (top, Y = +25m)
create_wall(stage, "WallNorth",
    FIELD_MIN_X - wall_offset, FIELD_MAX_X + wall_offset,
    FIELD_MAX_Y - wall_offset, FIELD_MAX_Y + wall_offset,
    WALL_THICKNESS, WALL_HEIGHT)

# South wall (bottom, Y = -25m)
create_wall(stage, "WallSouth",
    FIELD_MIN_X - wall_offset, FIELD_MAX_X + wall_offset,
    FIELD_MIN_Y - wall_offset, FIELD_MIN_Y + wall_offset,
    WALL_THICKNESS, WALL_HEIGHT)

# East wall (right, X = +50m)
create_wall(stage, "WallEast",
    FIELD_MAX_X - wall_offset, FIELD_MAX_X + wall_offset,
    FIELD_MIN_Y - wall_offset, FIELD_MAX_Y + wall_offset,
    WALL_THICKNESS, WALL_HEIGHT)

# West wall (left, X = -50m)
create_wall(stage, "WallWest",
    FIELD_MIN_X - wall_offset, FIELD_MIN_X + wall_offset,
    FIELD_MIN_Y - wall_offset, FIELD_MAX_Y + wall_offset,
    WALL_THICKNESS, WALL_HEIGHT)

print(f"[OK] Walls created (height: {WALL_HEIGHT}m, thickness: {WALL_THICKNESS}m)")

# =============================================================================
# RANDOMIZED TARGET GENERATION
# =============================================================================

def generate_random_goal(min_distance=75.0):
    """
    Generate a randomized goal position within field boundaries
    min_distance: minimum distance from start position (75m required)
    """
    while True:
        goal_x = np.random.uniform(FIELD_MIN_X + 2, FIELD_MAX_X - 2)
        goal_y = np.random.uniform(FIELD_MIN_Y + 2, FIELD_MAX_Y - 2)
        
        # Ensure minimum distance from start (at least 75m away)
        dist_to_start = np.sqrt((goal_x - START_X)**2 + (goal_y - START_Y)**2)
        if dist_to_start >= min_distance:
            return np.array([goal_x, goal_y])

# =============================================================================
# RANDOMIZED OBSTACLE GENERATION
# =============================================================================

def is_valid_obstacle_position(obs_x, obs_y, obs_radius, goal_x, goal_y):
    """
    Check if obstacle position is valid:
    - At least 10m from start
    - At least 10m from goal
    """
    dist_to_start = np.sqrt((obs_x - START_X)**2 + (obs_y - START_Y)**2)
    dist_to_goal = np.sqrt((obs_x - goal_x)**2 + (obs_y - goal_y)**2)
    
    # Must be at least 10m from start and 10m from goal (accounting for obstacle radius)
    return dist_to_start >= (10.0 + obs_radius) and dist_to_goal >= (10.0 + obs_radius)

def generate_random_obstacles(stage, goal_x, goal_y, num_obstacles=100, allowed_types=None):
    """
    Generate random furniture-like obstacles with varying shapes and sizes
    Heights range from 0.1m to 2.0m
    Types: couches, chairs, tables, shelves, ottomans, beds, cabinets, small_clutter
    Constraints: 10m minimum from start and goal
    allowed_types: list of furniture types to use (None = use all)
    """
    obstacle_count = 0
    obstacle_index = 0
    max_attempts = 2000
    attempts = 0
    
    if allowed_types is None:
        allowed_types = ['couch', 'chair', 'table', 'shelf', 'ottoman', 'bed', 'cabinet', 'small_clutter', 'car', 'truck']
    
    furniture_types = allowed_types
    
    while obstacle_count < num_obstacles and attempts < max_attempts:
        attempts += 1
        
        # Random obstacle position
        obs_x = np.random.uniform(FIELD_MIN_X + 5, FIELD_MAX_X - 5)
        obs_y = np.random.uniform(FIELD_MIN_Y + 5, FIELD_MAX_Y - 5)
        
        # Random furniture type
        furniture_type = np.random.choice(furniture_types)
        
        if furniture_type == 'couch':
            # Long, low furniture (width: 2.5-4m, depth: 0.8-1.2m, height: 0.8-1.0m)
            width = np.random.uniform(2.5, 4.0)
            depth = np.random.uniform(0.8, 1.2)
            height = np.random.uniform(0.8, 1.0)
            if is_valid_obstacle_position(obs_x, obs_y, max(width, depth)/2, goal_x, goal_y):
                create_box_obstacle(stage, f"Obstacle_Couch_{obstacle_index}", obs_x, obs_y, width, depth, height)
                obstacle_count += 1
                obstacle_index += 1
                
        elif furniture_type == 'chair':
            # Medium chair (width: 0.7-1.0m, depth: 0.6-0.9m, height: 0.8-1.1m)
            width = np.random.uniform(0.7, 1.0)
            depth = np.random.uniform(0.6, 0.9)
            height = np.random.uniform(0.8, 1.1)
            if is_valid_obstacle_position(obs_x, obs_y, max(width, depth)/2, goal_x, goal_y):
                create_box_obstacle(stage, f"Obstacle_Chair_{obstacle_index}", obs_x, obs_y, width, depth, height)
                obstacle_count += 1
                obstacle_index += 1
                
        elif furniture_type == 'table':
            # Low table (width: 1.0-2.0m, depth: 0.8-1.4m, height: 0.4-0.6m)
            width = np.random.uniform(1.0, 2.0)
            depth = np.random.uniform(0.8, 1.4)
            height = np.random.uniform(0.4, 0.6)
            if is_valid_obstacle_position(obs_x, obs_y, max(width, depth)/2, goal_x, goal_y):
                create_box_obstacle(stage, f"Obstacle_Table_{obstacle_index}", obs_x, obs_y, width, depth, height)
                obstacle_count += 1
                obstacle_index += 1
                
        elif furniture_type == 'shelf':
            # Tall, narrow shelf/bookcase (width: 0.4-0.8m, depth: 0.3-0.5m, height: 1.5-2.0m)
            width = np.random.uniform(0.4, 0.8)
            depth = np.random.uniform(0.3, 0.5)
            height = np.random.uniform(1.5, 2.0)
            if is_valid_obstacle_position(obs_x, obs_y, max(width, depth)/2, goal_x, goal_y):
                create_box_obstacle(stage, f"Obstacle_Shelf_{obstacle_index}", obs_x, obs_y, width, depth, height)
                obstacle_count += 1
                obstacle_index += 1
                
        elif furniture_type == 'ottoman':
            # Small ottoman/stool (width: 0.5-0.8m, depth: 0.5-0.8m, height: 0.3-0.5m)
            size = np.random.uniform(0.5, 0.8)
            height = np.random.uniform(0.3, 0.5)
            if is_valid_obstacle_position(obs_x, obs_y, size/2, goal_x, goal_y):
                create_box_obstacle(stage, f"Obstacle_Ottoman_{obstacle_index}", obs_x, obs_y, size, size, height)
                obstacle_count += 1
                obstacle_index += 1
                
        elif furniture_type == 'bed':
            # Large bed (width: 1.4-2.0m, depth: 2.0-2.5m, height: 0.5-0.7m)
            width = np.random.uniform(1.4, 2.0)
            depth = np.random.uniform(2.0, 2.5)
            height = np.random.uniform(0.5, 0.7)
            if is_valid_obstacle_position(obs_x, obs_y, max(width, depth)/2, goal_x, goal_y):
                create_box_obstacle(stage, f"Obstacle_Bed_{obstacle_index}", obs_x, obs_y, width, depth, height)
                obstacle_count += 1
                obstacle_index += 1
                
        elif furniture_type == 'cabinet':
            # Cabinet/dresser (width: 0.9-1.4m, depth: 0.5-0.7m, height: 1.2-1.8m)
            width = np.random.uniform(0.9, 1.4)
            depth = np.random.uniform(0.5, 0.7)
            height = np.random.uniform(1.2, 1.8)
            if is_valid_obstacle_position(obs_x, obs_y, max(width, depth)/2, goal_x, goal_y):
                create_box_obstacle(stage, f"Obstacle_Cabinet_{obstacle_index}", obs_x, obs_y, width, depth, height)
                obstacle_count += 1
                obstacle_index += 1
                
        elif furniture_type == 'small_clutter':
            # Small clutter items (width: 0.25-1.0m, depth: 0.25-1.0m, height: 0.1-0.5m)
            width = np.random.uniform(0.25, 1.0)
            depth = np.random.uniform(0.25, 1.0)
            height = np.random.uniform(0.1, 0.5)
            if is_valid_obstacle_position(obs_x, obs_y, max(width, depth)/2, goal_x, goal_y):
                create_box_obstacle(stage, f"Obstacle_Clutter_{obstacle_index}", obs_x, obs_y, width, depth, height)
                obstacle_count += 1
                obstacle_index += 1
                
        elif furniture_type == 'car':
            # Car-sized obstacle (length: 4.0-5.0m, width: 1.8-2.0m, height: 1.5-1.7m)
            length = np.random.uniform(4.0, 5.0)
            width = np.random.uniform(1.8, 2.0)
            height = np.random.uniform(1.5, 1.7)
            if is_valid_obstacle_position(obs_x, obs_y, max(length, width)/2, goal_x, goal_y):
                create_box_obstacle(stage, f"Obstacle_Car_{obstacle_index}", obs_x, obs_y, length, width, height)
                obstacle_count += 1
                obstacle_index += 1
                
        elif furniture_type == 'truck':
            # Truck-sized obstacle (length: 5.0-7.0m, width: 2.0-2.5m, height: 2.0-2.5m)
            length = np.random.uniform(5.0, 7.0)
            width = np.random.uniform(2.0, 2.5)
            height = np.random.uniform(2.0, 2.5)
            if is_valid_obstacle_position(obs_x, obs_y, max(length, width)/2, goal_x, goal_y):
                create_box_obstacle(stage, f"Obstacle_Truck_{obstacle_index}", obs_x, obs_y, length, width, height)
                obstacle_count += 1
                obstacle_index += 1

def create_box_obstacle(stage, name, x, y, width, depth, height):
    """Create a box-shaped obstacle (furniture-like)"""
    box = UsdGeom.Mesh.Define(stage, f"/World/{name}")
    
    # Box vertices (centered at x, y)
    hw = width / 2
    hd = depth / 2
    points = [
        Gf.Vec3f(x - hw, y - hd, 0),
        Gf.Vec3f(x + hw, y - hd, 0),
        Gf.Vec3f(x + hw, y + hd, 0),
        Gf.Vec3f(x - hw, y + hd, 0),
        Gf.Vec3f(x - hw, y - hd, height),
        Gf.Vec3f(x + hw, y - hd, height),
        Gf.Vec3f(x + hw, y + hd, height),
        Gf.Vec3f(x - hw, y + hd, height),
    ]
    
    box.GetPointsAttr().Set(points)
    box.GetFaceVertexCountsAttr().Set([4, 4, 4, 4, 4, 4])
    box.GetFaceVertexIndicesAttr().Set([
        0, 1, 2, 3,  # bottom
        4, 7, 6, 5,  # top
        0, 4, 5, 1,  # front
        2, 6, 7, 3,  # back
        0, 3, 7, 4,  # left
        1, 5, 6, 2,  # right
    ])
    # Random color for furniture variety
    color = Gf.Vec3f(np.random.uniform(0.3, 0.8), np.random.uniform(0.3, 0.8), np.random.uniform(0.3, 0.8))
    box.GetDisplayColorAttr().Set([color])

# Generate initial goal (must be at least 75m from start)
current_goal = generate_random_goal(min_distance=75.0)
print(f"[OK] Initial goal generated: ({current_goal[0]:.2f}, {current_goal[1]:.2f})")

# Generate random obstacles
print("Generating random furniture obstacles...")
# 100 large furniture obstacles
large_furniture_types = ['couch', 'chair', 'table', 'shelf', 'ottoman', 'bed', 'cabinet']
generate_random_obstacles(stage, current_goal[0], current_goal[1], num_obstacles=100, allowed_types=large_furniture_types)
print("[OK] 100 Large furniture obstacles created")

# 250 small clutter obstacles
generate_random_obstacles(stage, current_goal[0], current_goal[1], num_obstacles=250, allowed_types=['small_clutter'])
print("[OK] 250 Small clutter obstacles created")

# 5 car obstacles
generate_random_obstacles(stage, current_goal[0], current_goal[1], num_obstacles=5, allowed_types=['car'])
print("[OK] 5 Car obstacles created")

# 5 truck obstacles
generate_random_obstacles(stage, current_goal[0], current_goal[1], num_obstacles=5, allowed_types=['truck'])
print("[OK] 5 Truck obstacles created")

print("[OK] All random obstacles created (360 total)")

# =============================================================================
# CREATE START AND GOAL MARKERS
# =============================================================================

# Green square at start position
start_square = UsdGeom.Mesh.Define(stage, "/World/StartMarker")
start_square.GetPointsAttr().Set([
    Gf.Vec3f(START_X - 0.5, START_Y - 0.5, 0.01),
    Gf.Vec3f(START_X + 0.5, START_Y - 0.5, 0.01),
    Gf.Vec3f(START_X + 0.5, START_Y + 0.5, 0.01),
    Gf.Vec3f(START_X - 0.5, START_Y + 0.5, 0.01)
])
start_square.GetFaceVertexCountsAttr().Set([4])
start_square.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
start_square.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.8, 0.2)])  # Green
print(f"[OK] Start marker created at ({START_X}, {START_Y})")

# Red sphere at goal position (will be updated each episode)
def create_goal_marker(x, y):
    """Create or update goal marker sphere"""
    goal_sphere = UsdGeom.Sphere.Define(stage, "/World/GoalMarker")
    goal_xform = UsdGeom.Xformable(goal_sphere.GetPrim())
    goal_xform.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.5))
    goal_xform.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 0.5))
    goal_sphere.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 0.0)])  # Black
    return goal_sphere

goal_marker = create_goal_marker(current_goal[0], current_goal[1])
print(f"[OK] Goal marker created at ({current_goal[0]:.2f}, {current_goal[1]:.2f})")

# =============================================================================
# RESET WORLD
# =============================================================================

world.reset()
print("[OK] World reset and physics enabled")

# =============================================================================
# CREATE SPOT ROBOT
# =============================================================================

spot = SpotRobot(world, stage, prim_path="/World/Spot", name="Spot", 
                 position=np.array([START_X, START_Y, START_Z]))
spot.stabilize(steps=10)
spot.print_config()

# =============================================================================
# TESTING ENVIRONMENT READY - RUN SIMULATION
# =============================================================================

print("=" * 70)
print("TESTING ENVIRONMENT 1 INITIALIZED")
print("Running simulation for 10 seconds...")
print("=" * 70)

# Run simulation for 10 seconds
start_time = 0.0
max_time = 10.0
frame_count = 0

try:
    while simulation_app.is_running() and start_time < max_time:
        # Compute go-to-goal command and apply to Spot BEFORE stepping physics
        command, distance_to_goal, heading_error = spot.compute_go_to_goal_command(current_goal)
        spot.set_command(command)
        
        # Step physics
        world.step(render=True)
        start_time += 1.0 / 500.0  # Physics timestep
        frame_count += 1
        
        # Print status every second
        if frame_count % 500 == 0:
            elapsed = start_time
            pos = spot.get_state()[0]  # Get current position
            print(f"  Simulation running... {elapsed:.1f}s / {max_time}s | Spot pos: ({pos[0]:.1f}, {pos[1]:.1f}) | Goal dist: {distance_to_goal:.1f}m")
except KeyboardInterrupt:
    print("\nSimulation interrupted by user")

print("=" * 70)
print(f"Simulation completed after {start_time:.2f} seconds")
print("=" * 70)

simulation_app.close()
print("Closing Isaac Sim...")

