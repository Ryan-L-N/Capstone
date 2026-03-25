"""
Spot Robot Test Environment
===========================
Simple test environment for Spot robot navigation.

Environment:
  - 100m long x 50m wide field surrounded by walls
  - Start point at X=-45m, Y=0m
  - End zone at X=+45m, Y=0m (1m wide)

Robot: 1 Spot robot with basic go-to-goal controller

Author: MS for Autonomy Project
Date: January 2026
"""

import numpy as np
import argparse
from collections import deque

# Parse args before SimulationApp
parser = argparse.ArgumentParser(description="Spot Test Environment with RL")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
args = parser.parse_args()

# Isaac Sim setup - MUST come before other Isaac imports
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

# Now import Isaac modules
import omni
from omni.isaac.core import World
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
from pxr import UsdGeom, UsdLux, Gf, UsdPhysics

print("=" * 60)
print("SPOT ROBOT TEST ENVIRONMENT")
print("=" * 60)

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

# Field dimensions
FIELD_LENGTH = 100.0  # meters (X direction)
FIELD_WIDTH = 100.0   # meters (Y direction)

# Start point
START_X = -45.0
START_Y = 0.0
START_Z = 0.7

# Training area boundaries
BOUNDARY_X_MIN = -50.0
BOUNDARY_X_MAX = 50.0
BOUNDARY_Y_MIN = -50.0
BOUNDARY_Y_MAX = 50.0
MIN_GOAL_DISTANCE = 75.0  # At least 75m from start

# Generate random end zone position at least 75m from start
def generate_random_goal(start_x, start_y, min_distance, x_min, x_max, y_min, y_max, max_attempts=100):
    """Generate random goal position at least min_distance away from start point."""
    for _ in range(max_attempts):
        goal_x = np.random.uniform(x_min, x_max)
        goal_y = np.random.uniform(y_min, y_max)
        distance = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
        if distance >= min_distance:
            return goal_x, goal_y
    # Fallback if max_attempts exceeded
    return x_max, y_max

END_X, END_Y = generate_random_goal(START_X, START_Y, MIN_GOAL_DISTANCE, 
                                     BOUNDARY_X_MIN, BOUNDARY_X_MAX,
                                     BOUNDARY_Y_MIN, BOUNDARY_Y_MAX)

# Goal position
GOAL = np.array([END_X, END_Y])

print(f"Field: {FIELD_LENGTH}m x {FIELD_WIDTH}m")
print(f"Start: X={START_X}m, Y={START_Y}m")
print(f"End Zone: X={END_X:.1f}m, Y={END_Y:.1f}m (random, {np.sqrt((END_X - START_X)**2 + (END_Y - START_Y)**2):.1f}m away)")
print("=" * 60)

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
print("Ground plane added")

# =============================================================================
# CREATE TRAINING AREA (BLACK GROUND)
# =============================================================================

# Large black rectangle covering entire training area
black_ground = UsdGeom.Cube.Define(stage, "/World/BlackGround")
ground_xform = UsdGeom.Xformable(black_ground.GetPrim())
ground_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))  # Slightly below ground to avoid z-fighting
ground_xform.AddScaleOp().Set(Gf.Vec3f(FIELD_LENGTH, FIELD_WIDTH, 0.1))
black_ground.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 0.0)])
print(f"Training area (black ground) created: {FIELD_LENGTH}m x {FIELD_WIDTH}m")

# =============================================================================
# CREATE START AND END MARKERS
# =============================================================================

# Green area at start position (5m x 5m)
start_area = UsdGeom.Cube.Define(stage, "/World/StartArea")
start_xform = UsdGeom.Xformable(start_area.GetPrim())
start_xform.AddTranslateOp().Set(Gf.Vec3d(START_X, START_Y, 0.1))
start_xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 0.01))
start_area.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.8, 0.2)])
print(f"Start area (green) created at X={START_X}m, Y={START_Y}m")

# Purple sphere at end position
end_sphere = UsdGeom.Sphere.Define(stage, "/World/EndMarker")
end_xform = UsdGeom.Xformable(end_sphere.GetPrim())
end_xform.AddTranslateOp().Set(Gf.Vec3d(END_X, END_Y, 0.5))
end_xform.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 0.5))
end_sphere.GetDisplayColorAttr().Set([Gf.Vec3f(0.8, 0.2, 0.8)])
print(f"End marker (purple sphere) created at X={END_X}m, Y={END_Y}m")

# =============================================================================
# CREATE GREY WALLS AROUND TRAINING AREA
# =============================================================================
WALL_THICKNESS = 2.0
WALL_HEIGHT = 2.0
WALL_COLOR = Gf.Vec3f(0.5, 0.5, 0.5)  # Grey

# Left wall (X = -50 - wall_thickness/2)
left_wall = UsdGeom.Cube.Define(stage, "/World/LeftWall")
left_xform = UsdGeom.Xformable(left_wall.GetPrim())
left_xform.AddTranslateOp().Set(Gf.Vec3d(-50 - WALL_THICKNESS/2, 0, WALL_HEIGHT/2))
left_xform.AddScaleOp().Set(Gf.Vec3f(WALL_THICKNESS, FIELD_WIDTH, WALL_HEIGHT))
left_wall.GetDisplayColorAttr().Set([WALL_COLOR])

# Right wall (X = 50 + wall_thickness/2)
right_wall = UsdGeom.Cube.Define(stage, "/World/RightWall")
right_xform = UsdGeom.Xformable(right_wall.GetPrim())
right_xform.AddTranslateOp().Set(Gf.Vec3d(50 + WALL_THICKNESS/2, 0, WALL_HEIGHT/2))
right_xform.AddScaleOp().Set(Gf.Vec3f(WALL_THICKNESS, FIELD_WIDTH, WALL_HEIGHT))
right_wall.GetDisplayColorAttr().Set([WALL_COLOR])

# Front wall (Y = -50 - wall_thickness/2)
front_wall = UsdGeom.Cube.Define(stage, "/World/FrontWall")
front_xform = UsdGeom.Xformable(front_wall.GetPrim())
front_xform.AddTranslateOp().Set(Gf.Vec3d(0, -50 - WALL_THICKNESS/2, WALL_HEIGHT/2))
front_xform.AddScaleOp().Set(Gf.Vec3f(FIELD_LENGTH + 2*WALL_THICKNESS, WALL_THICKNESS, WALL_HEIGHT))
front_wall.GetDisplayColorAttr().Set([WALL_COLOR])

# Back wall (Y = 50 + wall_thickness/2)
back_wall = UsdGeom.Cube.Define(stage, "/World/BackWall")
back_xform = UsdGeom.Xformable(back_wall.GetPrim())
back_xform.AddTranslateOp().Set(Gf.Vec3d(0, 50 + WALL_THICKNESS/2, WALL_HEIGHT/2))
back_xform.AddScaleOp().Set(Gf.Vec3f(FIELD_LENGTH + 2*WALL_THICKNESS, WALL_THICKNESS, WALL_HEIGHT))
back_wall.GetDisplayColorAttr().Set([WALL_COLOR])

print("Grey walls (2m thick, non-colliding) created around training area")

# =============================================================================
# OBSTACLE PARAMETERS
# =============================================================================
# Truck obstacles: 2.2m tall, 2.5m wide, 6m long
TRUCK_HEIGHT = 2.2
TRUCK_WIDTH = 2.5
TRUCK_LENGTH = 6.0
TRUCK_COLOR = Gf.Vec3f(0.8, 0.4, 0.0)  # Orange

# Large furniture: 1.2m tall, 1.8m wide, 2.5m long
LARGE_FURNITURE_HEIGHT = 1.2
LARGE_FURNITURE_WIDTH = 1.8
LARGE_FURNITURE_LENGTH = 2.5
LARGE_FURNITURE_COLOR = Gf.Vec3f(0.6, 0.3, 0.0)  # Brown

# Small furniture: 0.6m tall, 0.8m wide, 1.0m long
SMALL_FURNITURE_HEIGHT = 0.6
SMALL_FURNITURE_WIDTH = 0.8
SMALL_FURNITURE_LENGTH = 1.0
SMALL_FURNITURE_COLOR = Gf.Vec3f(0.7, 0.5, 0.3)  # Light brown

# Clutter: 0.1-0.5m tall, 0.5-1.0m long
CLUTTER_HEIGHT_MIN = 0.1
CLUTTER_HEIGHT_MAX = 0.5
CLUTTER_LENGTH_MIN = 0.5
CLUTTER_LENGTH_MAX = 1.0
CLUTTER_COLOR = Gf.Vec3f(0.4, 0.4, 0.4)  # Grey

def create_obstacle(stage, obstacle_type, position, rotation=None, obstacle_id=0):
    """Create a single obstacle of the specified type with collision detection."""
    x, y = position
    
    if obstacle_type == "truck":
        obstacle = UsdGeom.Cube.Define(stage, f"/World/Obstacles/Truck_{obstacle_id}")
        xform = UsdGeom.Xformable(obstacle.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, TRUCK_HEIGHT / 2))
        xform.AddScaleOp().Set(Gf.Vec3f(TRUCK_WIDTH, TRUCK_LENGTH, TRUCK_HEIGHT))
        obstacle.GetDisplayColorAttr().Set([TRUCK_COLOR])
        # Add collision
        UsdPhysics.CollisionAPI.Apply(obstacle.GetPrim())
        
    elif obstacle_type == "large_furniture":
        obstacle = UsdGeom.Cube.Define(stage, f"/World/Obstacles/LargeFurniture_{obstacle_id}")
        xform = UsdGeom.Xformable(obstacle.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, LARGE_FURNITURE_HEIGHT / 2))
        xform.AddScaleOp().Set(Gf.Vec3f(LARGE_FURNITURE_WIDTH, LARGE_FURNITURE_LENGTH, LARGE_FURNITURE_HEIGHT))
        obstacle.GetDisplayColorAttr().Set([LARGE_FURNITURE_COLOR])
        # Add collision
        UsdPhysics.CollisionAPI.Apply(obstacle.GetPrim())
        
    elif obstacle_type == "small_furniture":
        obstacle = UsdGeom.Cube.Define(stage, f"/World/Obstacles/SmallFurniture_{obstacle_id}")
        xform = UsdGeom.Xformable(obstacle.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, SMALL_FURNITURE_HEIGHT / 2))
        xform.AddScaleOp().Set(Gf.Vec3f(SMALL_FURNITURE_WIDTH, SMALL_FURNITURE_LENGTH, SMALL_FURNITURE_HEIGHT))
        obstacle.GetDisplayColorAttr().Set([SMALL_FURNITURE_COLOR])
        # Add collision
        UsdPhysics.CollisionAPI.Apply(obstacle.GetPrim())
        
    elif obstacle_type == "clutter":
        height = np.random.uniform(CLUTTER_HEIGHT_MIN, CLUTTER_HEIGHT_MAX)
        length = np.random.uniform(CLUTTER_LENGTH_MIN, CLUTTER_LENGTH_MAX)
        width = length * 0.6  # Make width proportional to length
        obstacle = UsdGeom.Cube.Define(stage, f"/World/Obstacles/Clutter_{obstacle_id}")
        xform = UsdGeom.Xformable(obstacle.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, height / 2))
        xform.AddScaleOp().Set(Gf.Vec3f(width, length, height))
        obstacle.GetDisplayColorAttr().Set([CLUTTER_COLOR])
        # Add collision
        UsdPhysics.CollisionAPI.Apply(obstacle.GetPrim())
    
    return obstacle

def place_random_obstacles(stage, num_trucks=2, num_large=3, num_small=5, num_clutter=10):
    """Place random obstacles in the training area, avoiding start and end zones (10m buffer)."""
    
    # Create obstacles directory if it doesn't exist
    obstacles_root = stage.DefinePrim("/World/Obstacles", "Xform")
    
    obstacle_id = 0
    
    # Place truck obstacles
    for i in range(num_trucks):
        while True:
            x = np.random.uniform(BOUNDARY_X_MIN + 10, BOUNDARY_X_MAX - 10)
            y = np.random.uniform(BOUNDARY_Y_MIN + 10, BOUNDARY_Y_MAX - 10)
            # Check 10m buffer from start position
            dist_to_start = np.sqrt((x - START_X)**2 + (y - START_Y)**2)
            # Check 10m buffer from end position
            dist_to_end = np.sqrt((x - END_X)**2 + (y - END_Y)**2)
            if dist_to_start > 10 and dist_to_end > 10:
                break
        create_obstacle(stage, "truck", (x, y), obstacle_id=obstacle_id)
        obstacle_id += 1
    
    # Place large furniture obstacles
    for i in range(num_large):
        while True:
            x = np.random.uniform(BOUNDARY_X_MIN + 8, BOUNDARY_X_MAX - 8)
            y = np.random.uniform(BOUNDARY_Y_MIN + 8, BOUNDARY_Y_MAX - 8)
            dist_to_start = np.sqrt((x - START_X)**2 + (y - START_Y)**2)
            dist_to_end = np.sqrt((x - END_X)**2 + (y - END_Y)**2)
            if dist_to_start > 10 and dist_to_end > 10:
                break
        create_obstacle(stage, "large_furniture", (x, y), obstacle_id=obstacle_id)
        obstacle_id += 1
    
    # Place small furniture obstacles
    for i in range(num_small):
        while True:
            x = np.random.uniform(BOUNDARY_X_MIN + 5, BOUNDARY_X_MAX - 5)
            y = np.random.uniform(BOUNDARY_Y_MIN + 5, BOUNDARY_Y_MAX - 5)
            dist_to_start = np.sqrt((x - START_X)**2 + (y - START_Y)**2)
            dist_to_end = np.sqrt((x - END_X)**2 + (y - END_Y)**2)
            if dist_to_start > 10 and dist_to_end > 10:
                break
        create_obstacle(stage, "small_furniture", (x, y), obstacle_id=obstacle_id)
        obstacle_id += 1
    
    # Place clutter obstacles
    for i in range(num_clutter):
        while True:
            x = np.random.uniform(BOUNDARY_X_MIN + 3, BOUNDARY_X_MAX - 3)
            y = np.random.uniform(BOUNDARY_Y_MIN + 3, BOUNDARY_Y_MAX - 3)
            dist_to_start = np.sqrt((x - START_X)**2 + (y - START_Y)**2)
            dist_to_end = np.sqrt((x - END_X)**2 + (y - END_Y)**2)
            if dist_to_start > 10 and dist_to_end > 10:
                break
        create_obstacle(stage, "clutter", (x, y), obstacle_id=obstacle_id)
        obstacle_id += 1
    
    print(f"Placed obstacles: {num_trucks} trucks, {num_large} large furniture, {num_small} small furniture, {num_clutter} clutter items")

# =============================================================================
# PLACE RANDOM OBSTACLES IN THE TRAINING AREA
# =============================================================================
num_obstacles_placed = 0  # Global counter for obstacle detection
place_random_obstacles(stage, num_trucks=5, num_large=25, num_small=50, num_clutter=125)
num_obstacles_placed = 5 + 25 + 50 + 125  # Total obstacles

# =============================================================================
# CREATE SPOT ROBOT
# =============================================================================

spot = SpotFlatTerrainPolicy(
    prim_path="/World/Spot",
    name="Spot",
    position=np.array([START_X, START_Y, START_Z]),
)
print(f"Spot created at position: ({START_X}, {START_Y}, {START_Z})")

# CRITICAL: Reset world BEFORE accessing spot properties
world.reset()

# Initialize the robot
spot.initialize()
spot.robot.set_joints_default_state(spot.default_pos)
print("Spot initialized")

# =============================================================================
# ADD SENSORS TO SPOT (Simplified approach for Isaac Sim 5.1)
# =============================================================================

sensors = {
    "camera": {"enabled": True, "resolution": (640, 480)},
    "lidar": {"enabled": True, "range": 10.0},
    "imu": {"enabled": True},
    "contact": {"enabled": True}
}

print("Sensors configured:")
if sensors["camera"]["enabled"]:
    print(f"  - Camera: {sensors['camera']['resolution']} RGB")
if sensors["lidar"]["enabled"]:
    print(f"  - Lidar: {sensors['lidar']['range']}m range")
if sensors["imu"]["enabled"]:
    print("  - IMU: acceleration, gyro, orientation")
if sensors["contact"]["enabled"]:
    print("  - Contact: ground detection")

# Run physics steps for stability
for _ in range(10):
    world.step(render=False)
print("Spot stable and ready")

# =============================================================================
# RL AGENT FOR TRAJECTORY OPTIMIZATION
# =============================================================================

class SimpleQLearningAgent:
    """
    Q-Learning agent for optimizing navigation with sensor feedback
    State: [distance_to_goal, heading_error]
    Action: 9 discrete actions - 3 directions × 3 speed levels
        0: Turn left, slow (0.4 m/s)
        1: Turn left, medium (2.0 m/s / 4.5 mph)
        2: Turn left, fast (6.7 m/s / 15.0 mph)
        3: Go straight, slow (0.4 m/s)
        4: Go straight, medium (2.0 m/s / 4.5 mph)
        5: Go straight, fast (6.7 m/s / 15.0 mph)
        6: Turn right, slow (0.4 m/s)
        7: Turn right, medium (2.0 m/s / 4.5 mph)
        8: Turn right, fast (6.7 m/s / 15.0 mph)
    """
    def __init__(self, state_size=2, action_size=9):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.learning_rate = 0.1
        self.gamma = 0.95
        
        self.q_table = {}
        self.episode = 0
        
    def discretize_state(self, state):
        """Quantize continuous state to discrete buckets"""
        state_key = tuple(np.round(state, 1))
        return state_key
    
    def get_action(self, state, training=True):
        """Epsilon-greedy action selection (returns action index 0-8)"""
        state_key = self.discretize_state(state)
        
        if training and np.random.random() < self.epsilon:
            # Explore: random action index
            return np.random.randint(0, self.action_size)
        else:
            # Exploit: best known action
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_size)
            
            q_values = self.q_table[state_key]
            return np.argmax(q_values)
    
    def action_to_command(self, action_idx):
        """Convert action index (0-8) to (direction, speed) command"""
        # Map action index to (direction, speed)
        # Directions: -1 (left), 0 (straight), 1 (right)
        # Speeds: 0.4 m/s (slow), 2.0 m/s (medium), 6.7 m/s (fast/15mph)
        speeds = [0.4, 2.0, 6.7]
        directions = [-1, 0, 1]  # Left, straight, right
        
        direction_idx = action_idx // 3  # 0-2
        speed_idx = action_idx % 3        # 0-2
        
        direction = directions[direction_idx]
        speed = speeds[speed_idx]
        
        return direction, speed
    
    def update_q_value(self, state, action_idx, reward, next_state):
        """Update Q-value based on reward"""
        state_key = self.discretize_state(state)
        next_state_key = self.discretize_state(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        current_q = self.q_table[state_key][action_idx]
        max_next_q = np.max(self.q_table[next_state_key])
        
        new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action_idx] = new_q
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def new_episode(self):
        """Mark start of new episode"""
        self.episode += 1
        self.decay_epsilon()

agent = SimpleQLearningAgent()
print("RL Agent initialized (Q-Learning for trajectory optimization)")

# =============================================================================
# GO-TO-GOAL CONTROLLER
# =============================================================================

FORWARD_SPEED = 1.5
TURN_GAIN = 2.0
HEADING_THRESHOLD = 0.1

sim_time = [0.0]
physics_ready = [False]
STABILIZE_TIME = 6.0  # 3s delay + 3s stability

# Reward tracking
points = [500]  # Start with 500 points
last_update_time = [STABILIZE_TIME]  # Track when points were last decremented
prev_state = [None]
prev_action_idx = [None]
prev_vel = [0.0]
prev_dist_to_goal = [90.0]  # Initial distance
episode_active = [True]
step_count = [0]  # Initialize step counter here

# Episode summary tracking
episode_start_time = [0.0]
episode_start_pos = [None]
max_speed = [0.0]  # Track maximum speed in m/s
episode_success = [False]

def get_robot_state():
    pos, quat = spot.robot.get_world_pose()
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    heading = np.arctan2(siny_cosp, cosy_cosp)
    return np.array(pos), heading

def get_sensor_data():
    """Simulate sensor data from robot state"""
    sensor_data = {}
    
    try:
        pos, heading = get_robot_state()
        vel, ang_vel = spot.robot.get_linear_velocity(), spot.robot.get_angular_velocity()
        
        # Simulated camera data (would be RGB image in real implementation)
        sensor_data['camera'] = (640, 480, 3)  # Shape tuple
        
        # Simulated lidar data (would be point cloud)
        sensor_data['lidar_points'] = max(0, int(360 / 0.25))  # 1440 points at 0.25° resolution
        
        # IMU data (actual values from robot)
        sensor_data['imu_accel'] = np.array(vel)  # Use velocity as proxy for acceleration
        sensor_data['imu_gyro'] = np.array(ang_vel)
        sensor_data['imu_heading'] = heading
        
        # Contact sensor (robot in contact if Z is reasonable)
        sensor_data['contact'] = pos[2] > 0.3
        
    except Exception as e:
        sensor_data['camera'] = (640, 480, 3)
        sensor_data['lidar_points'] = 1440
        sensor_data['imu_accel'] = np.array([0, 0, -9.81])
        sensor_data['imu_gyro'] = np.array([0, 0, 0])
        sensor_data['imu_heading'] = 0
        sensor_data['contact'] = False
    
    return sensor_data

def lidar_scan(pos, heading, max_range=10.0, num_beams=16):
    """
    Simulate LiDAR scan using raycasts to detect obstacles.
    Returns array of distances from robot to nearest obstacle in each beam direction.
    
    Args:
        pos: Robot position [x, y, z]
        heading: Robot heading in radians
        max_range: Maximum LiDAR range in meters (default 10m like real Spot)
        num_beams: Number of beams to scan (higher = more precise)
    
    Returns:
        Array of distances to obstacles in each beam direction
    """
    ranges = []
    
    # Scan in num_beams directions around the robot
    for beam_idx in range(num_beams):
        # Beam angle relative to robot heading
        beam_angle = (beam_idx / num_beams) * 2 * np.pi
        # Global angle
        global_angle = heading + beam_angle
        
        # Ray direction
        ray_dir = np.array([np.cos(global_angle), np.sin(global_angle), 0])
        
        # Simple distance-based detection to obstacle centroids
        # In production, would use physics engine raycast
        min_dist = max_range
        
        # Check distance to all obstacles (approximated)
        # We can estimate by checking nearby obstacle locations
        try:
            # Get stage and search for obstacles
            stage = omni.usd.get_context().get_stage()
            obstacles_prim = stage.GetPrimAtPath("/World/Obstacles")
            
            if obstacles_prim:
                # Iterate through obstacle children
                for child in obstacles_prim.GetChildren():
                    try:
                        # Get obstacle position
                        xform = UsdGeom.Xformable(child)
                        if xform:
                            world_transform = xform.ComputeLocalToWorldTransform(0)
                            obs_pos = world_transform.ExtractTranslation()
                            obs_pos_2d = np.array([obs_pos[0], obs_pos[1]])
                            
                            # Vector from robot to obstacle
                            to_obstacle = obs_pos_2d - pos[:2]
                            dist_to_obs = np.linalg.norm(to_obstacle)
                            
                            # Check if obstacle is roughly in this beam direction
                            if dist_to_obs < max_range:
                                obs_angle = np.arctan2(to_obstacle[1], to_obstacle[0])
                                angle_diff = obs_angle - global_angle
                                # Normalize angle difference
                                while angle_diff > np.pi:
                                    angle_diff -= 2 * np.pi
                                while angle_diff < -np.pi:
                                    angle_diff += 2 * np.pi
                                
                                # If obstacle is within ~20 degrees of beam direction
                                if abs(angle_diff) < np.radians(20):
                                    if dist_to_obs < min_dist:
                                        min_dist = dist_to_obs
                    except:
                        pass
        except:
            pass
        
        ranges.append(min_dist)
    
    return np.array(ranges)

def get_obstacle_type(prim_name):
    """
    Determine obstacle type from prim name.
    Returns: 'truck', 'large_furniture', 'small_furniture', 'clutter', or 'unknown'
    """
    if "Truck" in prim_name:
        return "truck"
    elif "LargeFurniture" in prim_name:
        return "large_furniture"
    elif "SmallFurniture" in prim_name:
        return "small_furniture"
    elif "Clutter" in prim_name:
        return "clutter"
    else:
        return "unknown"

def detect_obstacles_near_robot(pos, heading, max_distance=2.5):
    """
    Detect obstacles using LiDAR and identify their type.
    Returns: (min_dist, angle_diff_to_obstacle, obstacle_type)
    Where angle_diff_to_obstacle is the angle from robot heading to obstacle (-pi to pi)
    """
    min_dist = max_distance
    closest_angle_diff = 0.0
    closest_obstacle_type = "unknown"
    
    try:
        # Get stage and search for obstacles
        stage = omni.usd.get_context().get_stage()
        obstacles_prim = stage.GetPrimAtPath("/World/Obstacles")
        
        if obstacles_prim:
            # Iterate through obstacle children
            for child in obstacles_prim.GetChildren():
                try:
                    # Get obstacle position
                    xform = UsdGeom.Xformable(child)
                    if xform:
                        world_transform = xform.ComputeLocalToWorldTransform(0)
                        obs_pos = world_transform.ExtractTranslation()
                        obs_pos_2d = np.array([obs_pos[0], obs_pos[1]])
                        
                        # Vector from robot to obstacle
                        to_obstacle = obs_pos_2d - pos[:2]
                        dist_to_obs = np.linalg.norm(to_obstacle)
                        
                        # Check if obstacle is in detection range
                        if dist_to_obs < max_distance:
                            obs_angle = np.arctan2(to_obstacle[1], to_obstacle[0])
                            angle_diff = obs_angle - heading
                            
                            # Normalize angle difference
                            while angle_diff > np.pi:
                                angle_diff -= 2 * np.pi
                            while angle_diff < -np.pi:
                                angle_diff += 2 * np.pi
                            
                            # If obstacle is in front of robot (~90 degree cone)
                            if abs(angle_diff) < np.radians(90):
                                if dist_to_obs < min_dist:
                                    min_dist = dist_to_obs
                                    closest_angle_diff = angle_diff
                                    # Get obstacle type
                                    closest_obstacle_type = get_obstacle_type(child.GetName())
                except:
                    pass
    except:
        pass
    
    return min_dist, closest_angle_diff, closest_obstacle_type

def detect_obstacles_near_robot_old(pos, max_distance=2.5):
    """
    Detect obstacles near robot using raycasts in 8 directions
    Returns: min_distance to nearest obstacle, direction to obstacle
    """
    pos_2d = pos[:2]
    min_dist = max_distance
    obstacle_dir = 0.0
    
    # Check 8 directions around the robot
    directions = [0, 45, 90, 135, 180, 225, 270, 315]
    
    for angle in directions:
        rad = np.radians(angle)
        # Ray from robot position in this direction
        ray_end = pos_2d + np.array([np.cos(rad), np.sin(rad)]) * max_distance
        
        # Check distance to all obstacles using simple spatial heuristic
        # (In real implementation, would use physics raycast)
        min_dist_in_dir = max_distance
        nearest_obstacle_angle = angle
        
        # This is a simplified check - in production, use physics engine raycast
        # For now, estimate based on obstacle proximity
        if min_dist_in_dir < min_dist:
            min_dist = min_dist_in_dir
            obstacle_dir = angle
    
    return min_dist, obstacle_dir

def compute_command(pos, heading, rl_action_idx=None):
    """
    Compute robot command based on go-to-goal controller + RL action guidance
    
    Args:
        pos: Robot position [x, y, z]
        heading: Robot heading in radians
        rl_action_idx: RL action index (0-8) for direction and speed selection
    
    Returns:
        Command vector [forward_velocity, lateral_velocity, angular_velocity]
    """
    to_goal = GOAL - pos[:2]
    dist = np.linalg.norm(to_goal)
    desired_heading = np.arctan2(to_goal[1], to_goal[0])
    heading_error = desired_heading - heading
    while heading_error > np.pi:
        heading_error -= 2 * np.pi
    while heading_error < -np.pi:
        heading_error += 2 * np.pi
    
    # Base go-to-goal controller
    turn_rate = TURN_GAIN * heading_error
    turn_rate = np.clip(turn_rate, -1.0, 1.0)
    
    # Default base speed
    base_forward_speed = 0.6  # Default 0.6 m/s
    forward_speed = base_forward_speed
    
    # Apply RL action for direction and speed modulation
    if rl_action_idx is not None:
        direction, speed = agent.action_to_command(rl_action_idx)
        
        # Adjust turn rate based on direction preference
        if direction < 0:  # Left turn
            turn_rate = max(-1.0, turn_rate - 0.3)
        elif direction > 0:  # Right turn
            turn_rate = min(1.0, turn_rate + 0.3)
        # direction == 0 (straight): use turn_rate as-is
        
        # Use RL-specified speed instead of base speed
        forward_speed = speed
    else:
        # Fallback: slow down when turning sharply
        if abs(heading_error) < HEADING_THRESHOLD:
            forward_speed = base_forward_speed
        else:
            forward_speed = base_forward_speed * 0.4
    
    turn_rate = np.clip(turn_rate, -1.0, 1.0)
    return np.array([forward_speed, 0.0, turn_rate])

def get_rl_state(pos, heading, dist_to_goal):
    """Build RL state from navigation data"""
    state = np.array([
        dist_to_goal,
        heading  # Heading error proxy
    ])
    return state.astype(np.float32)

def calculate_reward(points, prev_points, prev_dist, curr_dist, prev_vel, curr_vel, fell=False, reached_goal=False, obstacle_proximity=3.0):
    """
    Calculate reward based on:
    - Distance to goal (reward for moving closer)
    - Velocity increase bonus
    - Goal reached bonus (500 points)
    - Fall penalty (all remaining points)
    
    NOTE: NO penalties for obstacle proximity - steering logic handles avoidance
    """
    reward = 0.0
    
    # Fall penalty - lose all remaining points
    if fell:
        reward = -points
        return reward
    
    # Goal reached bonus
    if reached_goal:
        reward += 500
        return reward
    
    return reward

def on_physics_step(step_size):
    if not physics_ready[0]:
        physics_ready[0] = True
        return
    
    if not episode_active[0]:
        return
    
    sim_time[0] += step_size
    
    # Keep spot still for first 3 seconds to allow camera adjustment
    if sim_time[0] < 3.0:
        spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
        return
    
    # Then stabilize for 3 more seconds
    if sim_time[0] < STABILIZE_TIME:
        spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
        return
    
    pos, heading = get_robot_state()
    dist_to_goal = np.linalg.norm(GOAL - pos[:2])
    curr_vel = np.linalg.norm(spot.robot.get_linear_velocity())
    
    # Track maximum speed
    if curr_vel > max_speed[0]:
        max_speed[0] = curr_vel
    
    # Get RL action (returns action index 0-8) - must be done first to use in reward system
    rl_state = get_rl_state(pos, heading, dist_to_goal)
    rl_action_idx = agent.get_action(rl_state, training=True)
    
    # Point decay system - lose points every second regardless of speed
    time_elapsed = sim_time[0] - last_update_time[0]
    if time_elapsed >= 1.0:
        # Always lose points based on points level
        if points[0] > 250:
            points[0] -= 1  # Lose 1 point per second
        elif points[0] > 100:
            points[0] -= 2  # Lose 2 points per second
        else:
            points[0] -= 3  # Lose 3 points per second
        
        last_update_time[0] = sim_time[0]
        
        # Clamp points to 0 (but allow going above 500 with fast speed bonus)
        if points[0] < 0:
            points[0] = 0
    
    # Check if fallen
    fell = pos[2] < 0.25
    if fell:
        # Lose all remaining points
        points[0] = 0
    
    # Check if reached goal
    reached_goal = dist_to_goal < 1.0
    
    # Detect obstacles near robot for reward calculation
    obs_dist, _, _ = detect_obstacles_near_robot(pos, heading, max_distance=3.0)
    
    # Update Q-value if we have previous state
    if prev_state[0] is not None and prev_action_idx[0] is not None:
        reward = calculate_reward(points[0], points[0], prev_dist_to_goal[0], dist_to_goal, prev_vel[0], curr_vel, fell, reached_goal, obs_dist)
        agent.update_q_value(prev_state[0], prev_action_idx[0], reward, rl_state)
    
    # Store current state and action for next iteration
    prev_state[0] = rl_state.copy()
    prev_action_idx[0] = rl_action_idx
    prev_vel[0] = curr_vel
    prev_dist_to_goal[0] = dist_to_goal
    
    # Compute command with RL action index (direction and speed selection)
    command = compute_command(pos, heading, rl_action_idx)
    spot.forward(step_size, command)

world.add_physics_callback("spot_control", on_physics_step)

# =============================================================================
# MAIN LOOP - MULTIPLE EPISODES
# =============================================================================

print(f"\nStarting {args.episodes} episode(s)...")
print("=" * 60)

episode_results = []

for episode_num in range(1, args.episodes + 1):
    print(f"\n[EPISODE {episode_num}/{args.episodes}]")
    print("-" * 60)
    
    # Reset episode variables
    sim_time[0] = 0.0
    physics_ready[0] = False
    episode_active[0] = True
    episode_success[0] = False
    points[0] = 500
    last_update_time[0] = STABILIZE_TIME
    prev_state[0] = None
    prev_action_idx[0] = None
    prev_vel[0] = 0.0
    prev_dist_to_goal[0] = 90.0
    max_speed[0] = 0.0
    last_print = 0.0
    step_count[0] = 0
    
    # Reset robot to start position
    spot.robot.set_world_pose(position=np.array([START_X, START_Y, START_Z]))
    spot.robot.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
    spot.robot.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
    
    # Run physics steps to stabilize
    for _ in range(10):
        world.step(render=False)
    
    start_pos, _ = get_robot_state()
    episode_start_pos[0] = start_pos.copy()
    episode_start_time[0] = sim_time[0]
    
    # Always print initial position
    print(f"Initial position: X={start_pos[0]:.2f}m, Y={start_pos[1]:.2f}m")
    
    try:
        while simulation_app.is_running() and episode_active[0]:
            try:
                world.step(render=not args.headless)
                step_count[0] += 1
            except Exception as e:
                print(f"ERROR in world.step(): {type(e).__name__}: {e}")
                break
            
            if sim_time[0] - last_print >= 1.0:
                last_print = sim_time[0]
                try:
                    pos, heading = get_robot_state()
                    sensors = get_sensor_data()
                except Exception as e:
                    print(f"ERROR getting robot state at t={sim_time[0]:.1f}s: {e}")
                    break
                
                dist_to_goal = np.linalg.norm(GOAL - pos[:2])
                progress = pos[0] - start_pos[0]
                curr_vel = np.linalg.norm(spot.robot.get_linear_velocity())
                curr_vel_mph = curr_vel * 2.237  # Convert m/s to mph
                contact_status = "Yes" if sensors['contact'] else "No"
                
                # Always print monitoring data (headless or not)
                print(f"  t={sim_time[0]:5.1f}s | Distance: {dist_to_goal:5.1f}m | Points: {points[0]:3d} | Contact: {contact_status} | Velocity: {curr_vel_mph:.2f}mph")
                
                # Check if Spot hit the end marker (within 1m)
                dist_to_marker = np.linalg.norm(np.array([pos[0], pos[1]]) - np.array([END_X, END_Y]))
                if dist_to_marker < 1.0:
                    episode_active[0] = False
                    episode_success[0] = True
                    print(f"\n*** END MARKER REACHED in {sim_time[0]:.1f} seconds! ***")
                    print(f"*** EPISODE SUCCESS - Points remaining: {points[0]} ***")
                    break
                
                if pos[2] < 0.25:
                    episode_active[0] = False
                    episode_success[0] = False
                    print(f"\n*** SPOT FELL! Z={pos[2]:.2f}m ***")
                    print(f"*** EPISODE FAILURE - Points: 0 ***")
                    break
                
                if points[0] <= 0:
                    episode_active[0] = False
                    episode_success[0] = False
                    print(f"\n*** POINTS DEPLETED! ***")
                    print(f"*** EPISODE FAILURE - Points: 0 ***")
                    break
    
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    
    # Episode summary
    final_pos, _ = get_robot_state()
    total_progress = final_pos[0] - start_pos[0]
    episode_duration = sim_time[0] - episode_start_time[0]
    max_speed_mph = max_speed[0] * 2.237
    status = "SUCCESS" if episode_success[0] else "FAILURE"
    
    # Store results
    episode_results.append({
        'episode': episode_num,
        'status': status,
        'distance': total_progress,
        'duration': episode_duration,
        'max_speed_mph': max_speed_mph,
        'max_speed_ms': max_speed[0],
        'final_points': points[0],
        'epsilon': agent.epsilon
    })
    
    # Print episode result
    print(f"Ep {episode_num:3d}: {status:7s} | Dist: {total_progress:6.2f}m | Time: {episode_duration:5.1f}s | Speed: {max_speed_mph:5.2f}mph | Points: {points[0]:4d} | ε: {agent.epsilon:.4f}")
    
    # Decay epsilon for next episode
    agent.new_episode()

# Print summary statistics
print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)

if episode_results:
    successes = sum(1 for r in episode_results if r['status'] == 'SUCCESS')
    avg_distance = np.mean([r['distance'] for r in episode_results])
    avg_time = np.mean([r['duration'] for r in episode_results])
    avg_speed = np.mean([r['max_speed_mph'] for r in episode_results])
    avg_points = np.mean([r['final_points'] for r in episode_results])
    
    print(f"Episodes: {args.episodes}")
    print(f"Successes: {successes}/{args.episodes} ({100*successes/args.episodes:.1f}%)")
    print(f"Avg Distance: {avg_distance:.2f}m")
    print(f"Avg Duration: {avg_time:.1f}s")
    print(f"Avg Max Speed: {avg_speed:.2f}mph")
    print(f"Avg Final Points: {avg_points:.1f}")
    print(f"Final Epsilon: {agent.epsilon:.4f}")
    print("=" * 60)

simulation_app.close()
print("Done.")
