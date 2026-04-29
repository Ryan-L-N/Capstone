"""
Spot Navigation RL Training with Pre-trained Locomotion
========================================================
5 Spot robots in a 100m x 50m walled arena centered at (0,0).
Each robot starts on the left side and must reach its goal on the right side.

Uses pre-trained SpotFlatTerrainPolicy for locomotion.
RL only learns the navigation (velocity commands).

Action Space: 3 (forward velocity, lateral velocity, yaw rate)

Author: MS for Autonomy Project
Date: January 2026
"""

import numpy as np
import argparse
import os

# Parse args before SimulationApp
parser = argparse.ArgumentParser(description="Spot Navigation RL Training")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--max_iterations", type=int, default=100, help="Training iterations")
parser.add_argument("--load_checkpoint", type=str, default=None, help="Load existing policy")
args = parser.parse_args()

# Isaac Sim setup - MUST come before other Isaac imports
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

# Now import Isaac modules
import omni
from omni.isaac.core import World
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
from pxr import UsdGeom, UsdLux, Gf, UsdPhysics

print("=" * 70)
print("SPOT NAVIGATION RL - 5 ROBOTS WITH PRE-TRAINED LOCOMOTION")
print("=" * 70)

# =============================================================================
# ENVIRONMENT CONFIGURATION - CENTERED AT (0,0)
# =============================================================================

class EnvConfig:
    # Arena dimensions (100m x 50m) centered at origin
    ARENA_LENGTH = 100.0  # X: -50 to +50
    ARENA_WIDTH = 50.0    # Y: -25 to +25
    WALL_HEIGHT = 2.0
    WALL_THICKNESS = 0.5
    
    # Number of robots
    NUM_ROBOTS = 5
    
    # Starting positions (left side of arena, X = -45)
    START_X = -45.0
    START_Y_POSITIONS = [-20.0, -10.0, 0.0, 10.0, 20.0]  # Spread across Y
    START_Z = 0.7
    
    # Goal positions (right side of arena, X = +45)
    GOAL_X = 45.0
    # Goals have same Y as start positions (straight across)
    
    # Physics
    PHYSICS_DT = 1.0 / 500.0  # 500 Hz physics
    RENDERING_DT = 10.0 / 500.0  # 50 Hz rendering
    CONTROL_DT = 1.0 / 50.0  # 50 Hz control
    
    # Velocity limits
    MAX_FORWARD_VEL = 2.0  # m/s
    MAX_LATERAL_VEL = 0.5  # m/s (limit lateral to encourage forward motion)
    MAX_YAW_RATE = 1.5     # rad/s
    
    # Episode settings
    STABILIZE_TIME = 3.0   # Seconds before robot can move
    MAX_EPISODE_TIME = 90.0  # Max episode length (90m at 1m/s = 90s)
    
    # Reward settings
    PROGRESS_SCALE = 20.0      # Strong reward for forward progress
    VELOCITY_SCALE = 10.0      # Strong reward for moving toward goal
    ORIENTATION_SCALE = 5.0    # Reward for facing goal
    ALIVE_REWARD = 0.5
    GOAL_REWARD = 1000.0       # Big bonus for reaching goal
    FALL_PENALTY = -50.0
    ACTION_COST = 0.01

config = EnvConfig()

print(f"Arena: {config.ARENA_LENGTH}m x {config.ARENA_WIDTH}m (centered at origin)")
print(f"Number of robots: {config.NUM_ROBOTS}")
print(f"Start X: {config.START_X}m, Goal X: {config.GOAL_X}m")
print(f"Distance to travel: {config.GOAL_X - config.START_X:.0f}m")
for i, y in enumerate(config.START_Y_POSITIONS):
    print(f"  Robot {i}: Start ({config.START_X}, {y}) -> Goal ({config.GOAL_X}, {y})")
print("=" * 70)

# =============================================================================
# CREATE WORLD
# =============================================================================

world = World(
    physics_dt=config.PHYSICS_DT,
    rendering_dt=config.RENDERING_DT,
    stage_units_in_meters=1.0
)
stage = omni.usd.get_context().get_stage()

# Add lighting
dome_light = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
dome_light.CreateIntensityAttr(1500.0)
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

# =============================================================================
# CREATE ARENA CENTERED AT (0,0)
# =============================================================================

def create_walls():
    """Create walls around the arena centered at origin."""
    L = config.ARENA_LENGTH
    W = config.ARENA_WIDTH
    H = config.WALL_HEIGHT
    T = config.WALL_THICKNESS
    WALL_COLOR = Gf.Vec3f(0.5, 0.5, 0.5)
    
    # Arena spans X: [-L/2, L/2] and Y: [-W/2, W/2]
    walls = [
        # North wall (Y = +W/2)
        ("WallNorth", (0, W/2 + T/2, H/2), (L + 2*T, T, H)),
        # South wall (Y = -W/2)
        ("WallSouth", (0, -W/2 - T/2, H/2), (L + 2*T, T, H)),
        # East wall (X = +L/2) - goal side
        ("WallEast", (L/2 + T/2, 0, H/2), (T, W, H)),
        # West wall (X = -L/2) - start side
        ("WallWest", (-L/2 - T/2, 0, H/2), (T, W, H)),
    ]
    
    for name, pos, scale in walls:
        wall = UsdGeom.Cube.Define(stage, f"/World/Walls/{name}")
        xform = UsdGeom.Xformable(wall.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xform.AddScaleOp().Set(Gf.Vec3f(*scale))
        wall.GetDisplayColorAttr().Set([WALL_COLOR])
        UsdPhysics.CollisionAPI.Apply(wall.GetPrim())
    
    print(f"Created walled arena: {L}m x {W}m centered at (0,0)")

def create_visual_markers():
    """Create start zones and goal markers for each robot."""
    for i, y in enumerate(config.START_Y_POSITIONS):
        # Start zone (green square)
        start_zone = UsdGeom.Mesh.Define(stage, f"/World/StartZone_{i}")
        sx = config.START_X
        start_zone.GetPointsAttr().Set([
            Gf.Vec3f(sx - 1, y - 1, 0.02),
            Gf.Vec3f(sx + 1, y - 1, 0.02),
            Gf.Vec3f(sx + 1, y + 1, 0.02),
            Gf.Vec3f(sx - 1, y + 1, 0.02)
        ])
        start_zone.GetFaceVertexCountsAttr().Set([4])
        start_zone.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        start_zone.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.8, 0.2)])
        
        # Goal zone (blue sphere)
        goal_marker = UsdGeom.Sphere.Define(stage, f"/World/GoalMarker_{i}")
        xform = UsdGeom.Xformable(goal_marker.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(config.GOAL_X, y, 0.5))
        xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        goal_marker.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.2, 0.9)])
    
    print(f"Created {config.NUM_ROBOTS} start zones and goal markers")

create_walls()
create_visual_markers()

# =============================================================================
# CREATE 5 SPOT ROBOTS
# =============================================================================

spots = []
for i in range(config.NUM_ROBOTS):
    y = config.START_Y_POSITIONS[i]
    spot = SpotFlatTerrainPolicy(
        prim_path=f"/World/Spot_{i}",
        name=f"Spot_{i}",
        position=np.array([config.START_X, y, config.START_Z]),
    )
    spots.append(spot)
    print(f"Created Spot_{i} at ({config.START_X}, {y}, {config.START_Z})")

# CRITICAL: Reset world BEFORE accessing spot properties
world.reset()

# Initialize all robots
for i, spot in enumerate(spots):
    spot.initialize()
    spot.robot.set_joints_default_state(spot.default_pos)
print(f"All {config.NUM_ROBOTS} robots initialized")

# Run physics steps for stability
for _ in range(20):
    world.step(render=False)
print("Robots stable and ready")

# =============================================================================
# NAVIGATION POLICY
# =============================================================================

class NavigationPolicy:
    """Simple policy for goal-directed navigation.
    
    Uses a proportional controller plus learned adjustments.
    """
    
    def __init__(self):
        # Learning parameters
        self.forward_gain = 1.5   # Base forward speed when aligned
        self.turn_gain = 2.0      # Proportional gain for turning
        self.lateral_gain = 0.3   # Small lateral corrections
        
        # Learned adjustments (start at 0)
        self.forward_bias = 0.0
        self.turn_bias = 0.0
        
        # Experience for learning
        self.episode_progress = []
        self.lr = 0.01
    
    def get_action(self, goal_dir, heading_error, dist_to_goal):
        """Compute velocity command to reach goal.
        
        Args:
            goal_dir: Normalized direction to goal [x, y]
            heading_error: Angle error to goal (radians)
            dist_to_goal: Distance to goal (meters)
        
        Returns:
            [forward_vel, lateral_vel, yaw_rate]
        """
        # Proportional turn control - turn toward goal
        turn_rate = self.turn_gain * heading_error + self.turn_bias
        turn_rate = np.clip(turn_rate, -config.MAX_YAW_RATE, config.MAX_YAW_RATE)
        
        # Forward speed: full speed if aligned, reduced if turning
        alignment = np.cos(heading_error)  # 1 when aligned, -1 when backwards
        if abs(heading_error) < 0.2:  # Well aligned (< ~11 degrees)
            forward_speed = config.MAX_FORWARD_VEL * (self.forward_gain + self.forward_bias)
        elif abs(heading_error) < 0.5:  # Somewhat aligned
            forward_speed = config.MAX_FORWARD_VEL * 0.7
        else:  # Need to turn significantly
            forward_speed = config.MAX_FORWARD_VEL * 0.3
        
        forward_speed = np.clip(forward_speed, 0, config.MAX_FORWARD_VEL)
        
        # Small lateral correction toward goal
        lateral = self.lateral_gain * goal_dir[1] * alignment
        lateral = np.clip(lateral, -config.MAX_LATERAL_VEL, config.MAX_LATERAL_VEL)
        
        return np.array([forward_speed, lateral, turn_rate])
    
    def update(self, progress):
        """Update policy based on episode progress."""
        self.episode_progress.append(progress)
        if len(self.episode_progress) >= 5:
            recent = np.mean(self.episode_progress[-5:])
            if recent < 30:  # Not making good progress
                self.forward_gain = min(2.0, self.forward_gain + self.lr)
            self.episode_progress = self.episode_progress[-10:]
    
    def save(self, path):
        np.savez(path, forward_gain=self.forward_gain, turn_gain=self.turn_gain,
                 forward_bias=self.forward_bias, turn_bias=self.turn_bias)
        print(f"Policy saved to {path}")
    
    def load(self, path):
        if os.path.exists(path):
            data = np.load(path)
            self.forward_gain = float(data['forward_gain'])
            self.turn_gain = float(data['turn_gain'])
            print(f"Policy loaded from {path}")

# Create policy
policy = NavigationPolicy()
if args.load_checkpoint:
    policy.load(args.load_checkpoint)

# =============================================================================
# ROBOT STATE TRACKING
# =============================================================================

class RobotState:
    def __init__(self, robot_id, spot, start_y):
        self.id = robot_id
        self.spot = spot
        self.start_pos = np.array([config.START_X, start_y, config.START_Z])
        self.goal_pos = np.array([config.GOAL_X, start_y])
        self.prev_dist = config.GOAL_X - config.START_X
        self.episode_reward = 0.0
        self.episode_progress = 0.0
        self.goal_reached = False
        self.fell = False
    
    def get_state(self):
        """Get position, heading, velocity."""
        pos, quat = self.spot.robot.get_world_pose()
        vel = self.spot.robot.get_linear_velocity()
        
        # Extract yaw from quaternion
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        heading = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array(pos), heading, np.array(vel)
    
    def compute_goal_info(self):
        """Compute goal direction and heading error."""
        pos, heading, vel = self.get_state()
        
        to_goal = self.goal_pos - pos[:2]
        dist = np.linalg.norm(to_goal)
        goal_dir = to_goal / (dist + 1e-6)
        
        # Heading error (desired - actual)
        desired_heading = np.arctan2(to_goal[1], to_goal[0])
        heading_error = desired_heading - heading
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        return goal_dir, heading_error, dist, pos, heading, vel
    
    def compute_reward(self, dist, pos, heading, vel, action):
        """Compute reward for this step."""
        reward = 0.0
        
        # Progress toward goal (X direction)
        progress = self.prev_dist - dist
        reward += config.PROGRESS_SCALE * progress
        
        # Velocity toward goal
        goal_dir = (self.goal_pos - pos[:2]) / (np.linalg.norm(self.goal_pos - pos[:2]) + 1e-6)
        vel_toward_goal = np.dot(vel[:2], goal_dir)
        reward += config.VELOCITY_SCALE * max(0, vel_toward_goal)
        
        # Facing goal
        desired_heading = np.arctan2(goal_dir[1], goal_dir[0])
        heading_error = abs(desired_heading - heading)
        if heading_error > np.pi:
            heading_error = 2 * np.pi - heading_error
        facing_bonus = 1.0 - heading_error / np.pi
        reward += config.ORIENTATION_SCALE * facing_bonus
        
        # Alive
        if pos[2] > 0.4:
            reward += config.ALIVE_REWARD
        
        # Action cost
        reward -= config.ACTION_COST * np.sum(action**2)
        
        self.prev_dist = dist
        self.episode_reward += reward
        return reward
    
    def reset(self):
        """Reset robot to starting position."""
        self.spot.robot.set_world_pose(
            position=self.start_pos,
            orientation=np.array([1.0, 0.0, 0.0, 0.0])  # Facing +X
        )
        self.spot.robot.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
        self.spot.robot.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
        self.prev_dist = config.GOAL_X - config.START_X
        self.episode_progress = 0.0
        self.episode_reward = 0.0
        self.goal_reached = False
        self.fell = False

# Create robot states
robot_states = [
    RobotState(i, spots[i], config.START_Y_POSITIONS[i])
    for i in range(config.NUM_ROBOTS)
]

# =============================================================================
# TRAINING LOOP
# =============================================================================

sim_time = [0.0]
episode_time = [0.0]
physics_ready = [False]
episode_num = [1]

def on_physics_step(step_size):
    """Physics callback - controls all robots."""
    if not physics_ready[0]:
        physics_ready[0] = True
        return
    
    sim_time[0] += step_size
    episode_time[0] += step_size
    
    # Stabilization period - hold position
    if episode_time[0] < config.STABILIZE_TIME:
        for state in robot_states:
            state.spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
        return
    
    # Control each robot
    for state in robot_states:
        if state.goal_reached or state.fell:
            # Already done - just maintain stance
            state.spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
            continue
        
        # Get goal info
        goal_dir, heading_error, dist, pos, heading, vel = state.compute_goal_info()
        
        # Check termination
        if pos[2] < 0.25:
            state.fell = True
            state.episode_reward += config.FALL_PENALTY
            continue
        
        if dist < 2.0:
            state.goal_reached = True
            state.episode_reward += config.GOAL_REWARD
            continue
        
        # Get action from policy
        action = policy.get_action(goal_dir, heading_error, dist)
        
        # Send command to robot
        state.spot.forward(step_size, action)
        
        # Compute reward
        state.compute_reward(dist, pos, heading, vel, action)
        
        # Track progress
        state.episode_progress = config.GOAL_X - config.START_X - dist

# Register callback
world.add_physics_callback("spot_nav_control", on_physics_step)

# =============================================================================
# MAIN LOOP
# =============================================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print(f"Episodes: {args.max_iterations}")
print(f"Stabilization: {config.STABILIZE_TIME}s")
print(f"Max episode time: {config.MAX_EPISODE_TIME}s")
print("=" * 70 + "\n")

last_print = 0.0

try:
    while simulation_app.is_running() and episode_num[0] <= args.max_iterations:
        world.step(render=not args.headless)
        
        # Print progress every 5 seconds
        if sim_time[0] - last_print >= 5.0 and episode_time[0] > config.STABILIZE_TIME:
            last_print = sim_time[0]
            print(f"  t={episode_time[0]:.1f}s ", end="")
            for state in robot_states:
                _, _, dist, pos, _, _ = state.compute_goal_info()
                status = "G" if state.goal_reached else ("F" if state.fell else f"{dist:.0f}m")
                print(f"| R{state.id}:{status} ", end="")
            print()
        
        # Check for episode end
        all_done = all(s.goal_reached or s.fell for s in robot_states)
        timeout = episode_time[0] >= config.MAX_EPISODE_TIME
        
        if (all_done or timeout) and episode_time[0] > config.STABILIZE_TIME:
            # Episode summary
            print(f"\nEpisode {episode_num[0]} Complete:")
            goals = sum(1 for s in robot_states if s.goal_reached)
            falls = sum(1 for s in robot_states if s.fell)
            avg_progress = np.mean([s.episode_progress for s in robot_states])
            avg_reward = np.mean([s.episode_reward for s in robot_states])
            
            print(f"  Goals: {goals}/{config.NUM_ROBOTS} | "
                  f"Falls: {falls} | "
                  f"Avg Progress: {avg_progress:.1f}m | "
                  f"Avg Reward: {avg_reward:.1f}")
            
            for state in robot_states:
                status = "GOAL!" if state.goal_reached else ("FELL" if state.fell else "TIMEOUT")
                print(f"    Robot {state.id}: Progress={state.episode_progress:.1f}m, "
                      f"Reward={state.episode_reward:.1f}, Status={status}")
            
            # Update policy
            policy.update(avg_progress)
            
            # Reset all robots
            for state in robot_states:
                state.reset()
            
            episode_time[0] = 0.0
            episode_num[0] += 1
            
            # Physics steps to stabilize
            for _ in range(20):
                world.step(render=False)
            
            print()

except KeyboardInterrupt:
    print("\nTraining interrupted")

# =============================================================================
# SAVE AND CLEANUP
# =============================================================================

save_dir = os.path.dirname(os.path.abspath(__file__))
policy_path = os.path.join(save_dir, "spot_nav_policy.npz")
policy.save(policy_path)

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print(f"Policy saved to: {policy_path}")
print("=" * 70)

simulation_app.close()
print("Done.")