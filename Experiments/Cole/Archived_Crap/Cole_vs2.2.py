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
simulation_app = SimulationApp({"headless": False})  # Always run with GUI

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
FIELD_WIDTH = 50.0    # meters (Y direction)

# Start point
START_X = -45.0
START_Y = 0.0
START_Z = 0.7

# End zone (1m wide at far end)
END_X = 45.0
END_Y = 0.0

# Goal position
GOAL = np.array([END_X, END_Y])

print(f"Field: {FIELD_LENGTH}m x {FIELD_WIDTH}m")
print(f"Start: X={START_X}m, Y={START_Y}m")
print(f"End Zone: X={END_X}m, Y={END_Y}m")
print(f"Distance to travel: {END_X - START_X:.1f}m")
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
# CREATE TRAINING AREA (BLACK BACKGROUND)
# =============================================================================

# Large black rectangle covering entire training area (extended 5m past start and end)
TRAINING_AREA_MIN_X = START_X - 5.0  # 5m before start point
TRAINING_AREA_MAX_X = END_X + 5.0    # 5m after end point
black_bg = UsdGeom.Mesh.Define(stage, "/World/TrainingAreaBG")
black_bg.GetPointsAttr().Set([
    Gf.Vec3f(TRAINING_AREA_MIN_X, -FIELD_WIDTH/2, 0.005),
    Gf.Vec3f(TRAINING_AREA_MAX_X, -FIELD_WIDTH/2, 0.005),
    Gf.Vec3f(TRAINING_AREA_MAX_X, FIELD_WIDTH/2, 0.005),
    Gf.Vec3f(TRAINING_AREA_MIN_X, FIELD_WIDTH/2, 0.005)
])
black_bg.GetFaceVertexCountsAttr().Set([4])
black_bg.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
black_bg.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 0.0)])
print(f"Training area background (black) created: X=[{TRAINING_AREA_MIN_X}, {TRAINING_AREA_MAX_X}]m")

# =============================================================================
# CREATE START AND END MARKERS
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
start_square.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.8, 0.2)])
print(f"Start marker (green square) created at X={START_X}m, Y={START_Y}m")

# Purple sphere at end position
end_sphere = UsdGeom.Sphere.Define(stage, "/World/EndMarker")
end_xform = UsdGeom.Xformable(end_sphere.GetPrim())
end_xform.AddTranslateOp().Set(Gf.Vec3d(END_X, END_Y, 0.5))
end_xform.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 0.5))
end_sphere.GetDisplayColorAttr().Set([Gf.Vec3f(0.8, 0.2, 0.8)])
print(f"End marker (purple sphere) created at X={END_X}m, Y={END_Y}m")

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

def compute_command(pos, heading, rl_action_idx=None):
    """
    Compute robot command based on go-to-goal controller + RL modulation
    
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
    
    return np.array([forward_speed, 0.0, turn_rate])

def get_rl_state(pos, heading, dist_to_goal):
    """Build RL state from navigation data"""
    state = np.array([
        dist_to_goal,
        heading  # Heading error proxy
    ])
    return state.astype(np.float32)

def calculate_reward(points, prev_points, prev_dist, curr_dist, prev_vel, curr_vel, fell=False, reached_goal=False):
    """
    Calculate reward based on:
    - Point decay (1-3 points per second depending on points level)
    - Velocity increase bonus (3 points per m/s increase)
    - Goal reached bonus (500 points)
    - Fall penalty (all remaining points)
    """
    reward = 0.0
    
    # Fall penalty - lose all remaining points
    if fell:
        reward = -points
        return reward
    
    # Goal reached bonus
    if reached_goal:
        reward += 500.0
    
    # Velocity increase bonus (3 points per m/s of velocity gain)
    vel_increase = curr_vel - prev_vel
    if vel_increase > 0:
        reward += vel_increase * 3.0
    
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
    
    # Point decay system with speed-based rewards
    time_elapsed = sim_time[0] - last_update_time[0]
    if time_elapsed >= 1.0:
        # Get current commanded speed from RL action
        direction, commanded_speed = agent.action_to_command(rl_action_idx)
        
        # Check if robot is at medium or fast speed
        if commanded_speed == 2.0:
            # Medium speed: no points deducted
            pass
        elif commanded_speed == 6.7:
            # Fast speed: add 5 points per second
            points[0] += 5
        else:
            # Slow speed: normal point decay based on points level
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
    
    # Update Q-value if we have previous state
    if prev_state[0] is not None and prev_action_idx[0] is not None:
        reward = calculate_reward(points[0], points[0], prev_dist_to_goal[0], dist_to_goal, prev_vel[0], curr_vel, fell, reached_goal)
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
    
    if not args.headless:
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
                
                if not args.headless:
                    print(f"  t={sim_time[0]:5.1f}s | Distance: {dist_to_goal:5.1f}m | Points: {points[0]:3d} | Contact: {contact_status} | Velocity: {curr_vel_mph:.2f}mph")
                
                # Check if Spot hit the end marker (within 1m)
                dist_to_marker = np.linalg.norm(np.array([pos[0], pos[1]]) - np.array([END_X, END_Y]))
                if dist_to_marker < 1.0:
                    episode_active[0] = False
                    episode_success[0] = True
                    if not args.headless:
                        print(f"\n*** END MARKER REACHED in {sim_time[0]:.1f} seconds! ***")
                        print(f"*** EPISODE SUCCESS - Points remaining: {points[0]} ***")
                    break
                
                if pos[2] < 0.25:
                    episode_active[0] = False
                    episode_success[0] = False
                    if not args.headless:
                        print(f"\n*** SPOT FELL! Z={pos[2]:.2f}m ***")
                        print(f"*** EPISODE FAILURE - Points: 0 ***")
                    break
                
                if points[0] <= 0:
                    episode_active[0] = False
                    episode_success[0] = False
                    if not args.headless:
                        print(f"\n*** POINTS DEPLETED! ***")
                        print(f"*** EPISODE FAILURE - Points: 0 ***")
                    break
                
                if sim_time[0] > 120:
                    episode_active[0] = False
                    if not args.headless:
                        print(f"\n*** TIMEOUT after 120s ***")
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