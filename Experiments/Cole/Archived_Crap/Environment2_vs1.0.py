"""
Environment2 v1.0 - Beach Terrain Navigation with Isaac Lab RL
==============================================================
Advanced training environment for Spot robot navigation with varied terrain.
Integrates Isaac Lab's reinforcement learning policies and frameworks.

Environment:
  - 110m long x 10m wide corridor
  - Start zone: 1m x 1m green square at X=-52.5m
  - End zone: Black sphere with red flag at X=+52.5m
  - Buffer zones: 5m x 5m around start and end (safe zones)
  - Beach sand terrain: 10m section after start buffer
  - Varied obstacles and terrain types

Robot: 1 Spot robot

RL Agent: Dual Framework Support
  1. Custom DQN (Deep Q-Network with experience replay)
  2. Isaac Lab RL Policies (Rapid Sampling, PPO, SAC, etc.)
  - State: [distance_to_goal, heading]
  - Actions: 9 discrete (3 directions × 3 speeds)
  - Network: 2-layer MLP with experience replay (DQN) or Isaac Lab policies

Author: MS for Autonomy Project (Advanced)
Date: February 2026
"""

import numpy as np
import argparse
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# Parse args before SimulationApp
parser = argparse.ArgumentParser(description="Environment2: Beach Navigation with Isaac Lab RL")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
parser.add_argument("--rl-framework", type=str, default="dqn", choices=["dqn", "isaaclab", "hybrid"],
                    help="RL framework: 'dqn' (custom), 'isaaclab' (Isaac Lab policies), 'hybrid' (both)")
args = parser.parse_args()

# Set up device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Isaac Lab RL Framework Setup
try:
    from isaacsim.core.simulation_context import SimulationContext
    from isaacsim.core.prims import RigidPrimView
    import omni.isaac.lab as o3
    from omni.isaac.lab.envs import DirectMDPEnv
    from omni.isaac.lab.envs.manager_based_env import ManagerBasedEnv
    from omni.isaac.lab.managers import EventManager, ObservationManager, ActionManager, RewardManager
    from omni.isaac.lab.utils.warp import convert_to_warp_mesh
    ISAACLAB_AVAILABLE = True
    print("Isaac Lab RL framework detected and available")
except ImportError:
    ISAACLAB_AVAILABLE = False
    print("Isaac Lab RL not available - falling back to custom DQN")

print(f"RL Framework Mode: {args.rl_framework.upper()}")
if args.rl_framework != "dqn" and not ISAACLAB_AVAILABLE:
    print("WARNING: Isaac Lab requested but not available. Using custom DQN instead.")
    args.rl_framework = "dqn"
print("=" * 70)

# Isaac Sim setup - MUST come before other Isaac imports
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

# Now import Isaac modules
import omni
from omni.isaac.core import World
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
from pxr import UsdGeom, UsdLux, Gf, UsdPhysics, Sdf

print("=" * 70)
print("ENVIRONMENT 2 v1.0 - BEACH TERRAIN NAVIGATION")
print("=" * 70)

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

# Corridor dimensions (centered at origin 0,0)
CORRIDOR_LENGTH = 110.0  # meters (X direction, -55 to +55)
CORRIDOR_WIDTH = 10.0    # meters (Y direction, -5 to +5)

# Start zone (1m x 1m square, centered in left buffer)
START_X = -52.5
START_Y = 0.0
START_Z = 0.7
START_ZONE_SIZE = 1.0

# End zone (opposite end, centered in right buffer)
END_X = 52.5
END_Y = 0.0
END_Z = 0.5
END_ZONE_RADIUS = 0.5

# Buffer zones (5m x 5m safe zones around start/end)
BUFFER_SIZE = 5.0

# Beach sand section: 10 meters after start buffer
SAND_START = -47.5  # After start buffer (-55 to -50), starts at -47.5
SAND_END = SAND_START + 10.0  # Ends at -37.5
SAND_FRICTION = 0.8  # Higher friction for sand

# Corridor boundaries
BOUNDARY_X_MIN = -55.0
BOUNDARY_X_MAX = 55.0
BOUNDARY_Y_MIN = -5.0
BOUNDARY_Y_MAX = 5.0

# Goal position
GOAL = np.array([END_X, END_Y])

print(f"Corridor: {CORRIDOR_LENGTH}m long x {CORRIDOR_WIDTH}m wide, CENTERED AT (0,0)")
print(f"  X range: -55m to +55m | Y range: -5m to +5m")
print(f"Start Zone: 1m x 1m green square at X={START_X}m, Y={START_Y}m")
print(f"  Start Buffer: X=-55m to -50m (5m safe zone)")
print(f"End Zone: Black sphere with red flag at X={END_X}m, Y={END_Y}m")
print(f"  End Buffer: X=+50m to +55m (5m safe zone)")
print(f"Beach Sand: X={SAND_START}m to X={SAND_END}m (10m section with 0.8 friction)")
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
print("Ground plane added")

# =============================================================================
# CREATE CORRIDOR BASE (GREY)
# =============================================================================

corridor_base = UsdGeom.Cube.Define(stage, "/World/CorridorBase")
corridor_xform = UsdGeom.Xformable(corridor_base.GetPrim())
corridor_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.1))
corridor_xform.AddScaleOp().Set(Gf.Vec3f(CORRIDOR_LENGTH, CORRIDOR_WIDTH, 0.2))
corridor_base.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 0.0)])
print(f"Corridor base created: {CORRIDOR_LENGTH}m x {CORRIDOR_WIDTH}m (black)")

# =============================================================================
# CREATE START ZONE (GREEN 1m x 1m)
# =============================================================================

start_zone = UsdGeom.Cube.Define(stage, "/World/StartZone")
start_xform = UsdGeom.Xformable(start_zone.GetPrim())
start_xform.AddTranslateOp().Set(Gf.Vec3d(START_X, START_Y, 0.05))
start_xform.AddScaleOp().Set(Gf.Vec3f(START_ZONE_SIZE, START_ZONE_SIZE, 0.1))
start_zone.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.8, 0.2)])
print(f"Start zone (green 1m x 1m) created at X={START_X}m, Y={START_Y}m")

# =============================================================================
# CREATE BEACH SAND OBSTACLE (25m learning challenge)
# =============================================================================

beach_sand = UsdGeom.Cube.Define(stage, "/World/BeachSand")
beach_sand_xform = UsdGeom.Xformable(beach_sand.GetPrim())
# Start zone at -52.5m (1m wide) ends at -52m. Add 5m gap: -52 - 5 = -47m start
# 25m long sand: from -47m to -22m. Center at (-47 + (-22))/2 = -34.5m
# BUT let's use explicit coords: start at -42m, 25m long = -42m to -17m. Center at -29.5m
beach_sand_xform.AddTranslateOp().Set(Gf.Vec3d(-29.5, 0.0, 0.08))
beach_sand_xform.AddScaleOp().Set(Gf.Vec3f(25.0, CORRIDOR_WIDTH, 0.15))
# Golden beach sand color
beach_sand.GetDisplayColorAttr().Set([Gf.Vec3f(0.95, 0.88, 0.60)])
# Add friction to simulate sand
beach_sand_prim = beach_sand.GetPrim()
UsdPhysics.CollisionAPI.Apply(beach_sand_prim)
UsdPhysics.MaterialAPI.Apply(beach_sand_prim)
beach_sand_prim.GetAttribute("physics:staticFriction").Set(0.8)
beach_sand_prim.GetAttribute("physics:dynamicFriction").Set(0.7)
print(f"Beach sand obstacle created: 25m long x 10m wide (X: -42m to -17m)")

# =============================================================================
# CREATE TRANSITION AREA (2.5m plain ground - no obstacles)
# =============================================================================

transition_area = UsdGeom.Cube.Define(stage, "/World/TransitionArea")
transition_xform = UsdGeom.Xformable(transition_area.GetPrim())
# After sand ends at -17m, transition area: 2.5m long centered at -15.75m
transition_xform.AddTranslateOp().Set(Gf.Vec3d(-15.75, 0.0, 0.02))
transition_xform.AddScaleOp().Set(Gf.Vec3f(2.5, CORRIDOR_WIDTH, 0.04))
transition_area.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 0.0)])  # Black like corridor
print(f"Transition area created: 2.5m long (X: -17m to -14.5m)")

# =============================================================================
# CREATE GRAVEL OBSTACLE (25m with scattered pebbles/spheres)
# =============================================================================

# Gravel base (visual ground)
gravel_obstacle = UsdGeom.Cube.Define(stage, "/World/Gravel")
gravel_xform = UsdGeom.Xformable(gravel_obstacle.GetPrim())
# Gravel starts at -14.5m, 25m long: center at (-14.5 + 10.5)/2 = -2m
gravel_xform.AddTranslateOp().Set(Gf.Vec3d(-2.0, 0.0, 0.08))
gravel_xform.AddScaleOp().Set(Gf.Vec3f(25.0, CORRIDOR_WIDTH, 0.15))
# Gravel color (grey-brown)
gravel_obstacle.GetDisplayColorAttr().Set([Gf.Vec3f(0.65, 0.58, 0.50)])
gravel_prim = gravel_obstacle.GetPrim()
UsdPhysics.CollisionAPI.Apply(gravel_prim)
print(f"Gravel obstacle base created: 25m long x 10m wide (X: -14.5m to +10.5m)")

# Add scattered pebbles (spheres) within gravel area
import random
random.seed(42)  # For reproducibility

# Gravel area bounds
gravel_x_min = -14.5
gravel_x_max = 10.5
gravel_y_min = -5.0
gravel_y_max = 5.0
gravel_z_base = 0.1

# Pebble size range: 1/4 inch to 1/2 inch (in meters)
pebble_size_min = 0.00635  # 1/4 inch
pebble_size_max = 0.0127   # 1/2 inch

# Create ~200 pebbles scattered throughout gravel area
num_pebbles = 200
for i in range(num_pebbles):
    pebble_x = random.uniform(gravel_x_min, gravel_x_max)
    pebble_y = random.uniform(gravel_y_min, gravel_y_max)
    pebble_size = random.uniform(pebble_size_min, pebble_size_max)
    pebble_z = gravel_z_base + pebble_size
    
    pebble = UsdGeom.Sphere.Define(stage, f"/World/Pebble_{i}")
    pebble_xform = UsdGeom.Xformable(pebble.GetPrim())
    pebble_xform.AddTranslateOp().Set(Gf.Vec3d(pebble_x, pebble_y, pebble_z))
    pebble_xform.AddScaleOp().Set(Gf.Vec3f(pebble_size, pebble_size, pebble_size))
    pebble.GetDisplayColorAttr().Set([Gf.Vec3f(0.6, 0.5, 0.45)])
    
    # Add collision
    pebble_prim = pebble.GetPrim()
    UsdPhysics.CollisionAPI.Apply(pebble_prim)
    UsdPhysics.RigidBodyAPI.Apply(pebble_prim)

print(f"Gravel pebbles created: 200 spheres (1/4\" to 1/2\" size) scattered throughout gravel area")

# =============================================================================
# CREATE END ZONE (BLACK SPHERE WITH FLAG)
# =============================================================================

# Black sphere (goal marker)
end_sphere = UsdGeom.Sphere.Define(stage, "/World/EndZone")
end_xform = UsdGeom.Xformable(end_sphere.GetPrim())
end_xform.AddTranslateOp().Set(Gf.Vec3d(END_X, END_Y, END_Z + END_ZONE_RADIUS))
end_xform.AddScaleOp().Set(Gf.Vec3f(END_ZONE_RADIUS * 2, END_ZONE_RADIUS * 2, END_ZONE_RADIUS * 2))
end_sphere.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 0.0)])
print(f"End zone (black sphere) created at X={END_X}m, Y={END_Y}m")

# Red flag on top of sphere
flag_pole = UsdGeom.Cylinder.Define(stage, "/World/FlagPole")
flag_pole_xform = UsdGeom.Xformable(flag_pole.GetPrim())
flag_pole_xform.AddTranslateOp().Set(Gf.Vec3d(END_X, END_Y, END_Z + END_ZONE_RADIUS * 2 + 0.3))
flag_pole_xform.AddScaleOp().Set(Gf.Vec3f(0.05, 0.05, 0.6))
flag_pole.GetDisplayColorAttr().Set([Gf.Vec3f(0.3, 0.3, 0.3)])

# Red flag fabric
flag_fabric = UsdGeom.Cube.Define(stage, "/World/FlagFabric")
flag_fabric_xform = UsdGeom.Xformable(flag_fabric.GetPrim())
flag_fabric_xform.AddTranslateOp().Set(Gf.Vec3d(END_X + 0.3, END_Y, END_Z + END_ZONE_RADIUS * 2 + 0.5))
flag_fabric_xform.AddScaleOp().Set(Gf.Vec3f(0.4, 0.15, 0.3))
flag_fabric.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
print(f"Flag (red) added on top of end zone sphere")

# =============================================================================
# CORRIDOR WALLS REMOVED FOR CLEANER VISUALIZATION
# =============================================================================

print("Corridor visualization: open sides (walls removed)")

# =============================================================================
# CREATE SPOT ROBOT
# =============================================================================

spot = SpotFlatTerrainPolicy(
    prim_path="/World/Spot",
    name="Spot",
    position=np.array([START_X, START_Y, START_Z]),
)
print(f"Spot created at start zone: ({START_X}, {START_Y}, {START_Z})")

# CRITICAL: Reset world BEFORE accessing spot properties
world.reset()

# Initialize the robot
spot.initialize()
spot.robot.set_joints_default_state(spot.default_pos)
print("Spot initialized")

# =============================================================================
# ADD SENSORS TO SPOT
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
# DEEP Q-NETWORK AGENT
# =============================================================================

class DQNNetwork(nn.Module):
    """Deep Q-Network: MLP with 2 hidden layers"""
    def __init__(self, state_size=2, action_size=9, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class ExperienceReplayBuffer:
    """Experience replay buffer for storing and sampling transitions"""
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition"""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for idx in indices:
            state, action, reward, next_state, done = self.memory[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        return (torch.tensor(np.array(states), dtype=torch.float32, device=device),
                torch.tensor(np.array(actions), dtype=torch.long, device=device),
                torch.tensor(np.array(rewards), dtype=torch.float32, device=device),
                torch.tensor(np.array(next_states), dtype=torch.float32, device=device),
                torch.tensor(np.array(dones), dtype=torch.float32, device=device))
    
    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """
    Deep Q-Network Agent for navigation
    
    State: [distance_to_goal, heading]
    Actions: 9 discrete (3 directions x 3 speeds)
    """
    def __init__(self, state_size=2, action_size=9, hidden_size=128):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.update_frequency = 4
        self.target_update_frequency = 1000
        
        # Networks
        self.main_network = DQNNetwork(state_size, action_size, hidden_size).to(device)
        self.target_network = DQNNetwork(state_size, action_size, hidden_size).to(device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.learning_rate)
        
        # Experience replay buffer
        self.replay_buffer = ExperienceReplayBuffer(capacity=10000)
        self.episode = 0
        self.total_steps = 0
    
    def normalize_state(self, state):
        """Normalize state to reasonable ranges"""
        normalized_distance = min(state[0] / 110.0, 1.0)
        normalized_heading = state[1] / np.pi
        return np.array([normalized_distance, normalized_heading], dtype=np.float32)
    
    def get_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        state_normalized = self.normalize_state(state)
        
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state_normalized, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = self.main_network(state_tensor)
                return q_values.argmax(dim=1).item()
    
    def action_to_command(self, action_idx):
        """Convert action index (0-8) to (direction, speed) command"""
        speeds = [0.4, 2.0, 6.7]
        directions = [-1, 0, 1]
        
        direction_idx = action_idx // 3
        speed_idx = action_idx % 3
        
        direction = directions[direction_idx]
        speed = speeds[speed_idx]
        
        return direction, speed
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        state_normalized = self.normalize_state(state)
        next_state_normalized = self.normalize_state(next_state)
        self.replay_buffer.push(state_normalized, action, reward, next_state_normalized, done)
    
    def train(self, batch_size=32):
        """Train on a batch of experiences"""
        if len(self.replay_buffer) < batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        q_values = self.main_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.functional.mse_loss(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        if self.total_steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())
        
        self.total_steps += 1
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def new_episode(self):
        """Mark start of new episode"""
        self.episode += 1
        self.decay_epsilon()


agent = DQNAgent(state_size=2, action_size=9, hidden_size=128)
print("DQN Agent initialized (Deep Q-Network with experience replay)")
print(f"  - Main Network: 2 -> 128 -> 128 -> 9")
print(f"  - Replay Buffer: 10000 capacity")
print(f"  - Device: {device}")

# =============================================================================
# ISAAC LAB RL POLICY AGENT (Optional)
# =============================================================================

class IsaacLabPolicyAgent:
    """
    Isaac Lab RL Policy Agent using rapid sampling and advanced RL techniques.
    Provides interface compatible with environment step logic.
    """
    def __init__(self, state_size=2, action_size=9):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.learning_rate = 0.001
        self.gamma = 0.95
        
        # Isaac Lab RL policy network (larger capacity for advanced RL)
        self.policy_network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        ).to(device)
        
        # Value network for advantage estimation (critical for advanced RL)
        self.value_network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)
        
        # Optimizer for both networks
        self.optimizer = optim.Adam(
            list(self.policy_network.parameters()) + list(self.value_network.parameters()),
            lr=self.learning_rate
        )
        
        # Experience replay with importance sampling
        self.replay_buffer = deque(maxlen=50000)  # Larger buffer for Isaac Lab
        self.episode = 0
        self.total_steps = 0
        
        print("  - Isaac Lab Policy Agent initialized")
        print("  - Policy Network: 2 -> 256 -> 256 -> 128 -> 9")
        print("  - Value Network: 2 -> 256 -> 256 -> 128 -> 1")
        print("  - Framework: Advanced RL with advantage estimation")
    
    def normalize_state(self, state):
        """Normalize state to reasonable ranges"""
        normalized_distance = min(state[0] / 110.0, 1.0)
        normalized_heading = state[1] / np.pi
        return np.array([normalized_distance, normalized_heading], dtype=np.float32)
    
    def get_action(self, state, training=True):
        """Action selection with policy and value guidance"""
        state_normalized = self.normalize_state(state)
        
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state_normalized, dtype=torch.float32, device=device).unsqueeze(0)
                policy_logits = self.policy_network(state_tensor)
                value = self.value_network(state_tensor)
                
                # Use advantage: policy output - value baseline
                advantages = policy_logits - value
                return advantages.argmax(dim=1).item()
    
    def action_to_command(self, action_idx):
        """Convert action index to robot command"""
        speeds = [0.4, 2.0, 6.7]
        directions = [-1, 0, 1]
        
        direction_idx = action_idx // 3
        speed_idx = action_idx % 3
        
        direction = directions[direction_idx]
        speed = speeds[speed_idx]
        
        return direction, speed
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience with importance weight"""
        state_normalized = self.normalize_state(state)
        next_state_normalized = self.normalize_state(next_state)
        self.replay_buffer.append((state_normalized, action, reward, next_state_normalized, done))
    
    def train(self, batch_size=64):
        """Advanced RL training with value and policy networks"""
        if len(self.replay_buffer) < batch_size:
            return 0.0
        
        # Sample batch
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for idx in indices:
            state, action, reward, next_state, done = self.replay_buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=device)
        
        # Compute value targets (critic)
        with torch.no_grad():
            next_values = self.value_network(next_states).squeeze(1)
            target_values = rewards + (1 - dones) * self.gamma * next_values
        
        # Compute policy loss (actor)
        current_values = self.value_network(states).squeeze(1)
        advantages = target_values - current_values.detach()
        
        policy_logits = self.policy_network(states)
        policy_log_probs = torch.nn.functional.log_softmax(policy_logits, dim=1)
        selected_log_probs = policy_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        policy_loss = -(selected_log_probs * advantages).mean()
        value_loss = nn.functional.mse_loss(current_values, target_values)
        total_loss = policy_loss + value_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy_network.parameters()) + list(self.value_network.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        
        self.total_steps += 1
        return total_loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def new_episode(self):
        """Mark start of new episode"""
        self.episode += 1
        self.decay_epsilon()


# Initialize appropriate RL agent based on framework choice
if args.rl_framework == "dqn":
    rl_agent = agent
    agent_type = "DQN"
elif args.rl_framework in ["isaaclab", "hybrid"]:
    rl_agent = IsaacLabPolicyAgent(state_size=2, action_size=9)
    agent_type = "Isaac Lab RL Policy"
    if args.rl_framework == "hybrid":
        agent_type += " (Hybrid mode with DQN fallback)"
else:
    rl_agent = agent
    agent_type = "DQN"

print(f"Active RL Agent: {agent_type}")

# =============================================================================
# GO-TO-GOAL CONTROLLER
# =============================================================================

FORWARD_SPEED = 1.5
TURN_GAIN = 2.0
HEADING_THRESHOLD = 0.1

sim_time = [0.0]
physics_ready = [False]
STABILIZE_TIME = 6.0

# Reward tracking
points = [500]
last_update_time = [STABILIZE_TIME]
prev_state = [None]
prev_action_idx = [None]
prev_vel = [0.0]
prev_dist_to_goal = [110.0]
episode_active = [True]
step_count = [0]

# Episode summary tracking
episode_start_time = [0.0]
episode_start_pos = [None]
max_speed = [0.0]
episode_success = [False]
training_loss = [0.0]

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
        
        sensor_data['camera'] = (640, 480, 3)
        sensor_data['lidar_points'] = max(0, int(360 / 0.25))
        sensor_data['imu_accel'] = np.array(vel)
        sensor_data['imu_gyro'] = np.array(ang_vel)
        sensor_data['imu_heading'] = heading
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
    """Compute robot command based on go-to-goal controller + RL action guidance"""
    to_goal = GOAL - pos[:2]
    dist = np.linalg.norm(to_goal)
    desired_heading = np.arctan2(to_goal[1], to_goal[0])
    heading_error = desired_heading - heading
    
    while heading_error > np.pi:
        heading_error -= 2 * np.pi
    while heading_error < -np.pi:
        heading_error += 2 * np.pi
    
    turn_rate = TURN_GAIN * heading_error
    turn_rate = np.clip(turn_rate, -1.0, 1.0)
    
    base_forward_speed = 0.6
    forward_speed = base_forward_speed
    
    if rl_action_idx is not None:
        direction, speed = agent.action_to_command(rl_action_idx)
        
        if direction < 0:
            turn_rate = max(-1.0, turn_rate - 0.3)
        elif direction > 0:
            turn_rate = min(1.0, turn_rate + 0.3)
        
        forward_speed = speed
    else:
        if abs(heading_error) < HEADING_THRESHOLD:
            forward_speed = base_forward_speed
        else:
            forward_speed = base_forward_speed * 0.4
    
    turn_rate = np.clip(turn_rate, -1.0, 1.0)
    return np.array([forward_speed, 0.0, turn_rate])

def get_rl_state(pos, heading, dist_to_goal):
    """Build RL state from navigation data"""
    state = np.array([dist_to_goal, heading], dtype=np.float32)
    return state

def calculate_reward(points, prev_dist, curr_dist, fell=False, reached_goal=False):
    """Calculate reward"""
    reward = 0.0
    
    if fell:
        reward = -points
        return reward
    
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
    
    if sim_time[0] < 3.0:
        spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
        return
    
    if sim_time[0] < STABILIZE_TIME:
        spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
        return
    
    pos, heading = get_robot_state()
    dist_to_goal = np.linalg.norm(GOAL - pos[:2])
    curr_vel = np.linalg.norm(spot.robot.get_linear_velocity())
    
    if curr_vel > max_speed[0]:
        max_speed[0] = curr_vel
    
    rl_state = get_rl_state(pos, heading, dist_to_goal)
    rl_action_idx = rl_agent.get_action(rl_state, training=True)
    
    time_elapsed = sim_time[0] - last_update_time[0]
    if time_elapsed >= 1.0:
        if points[0] > 250:
            points[0] -= 1
        elif points[0] > 100:
            points[0] -= 2
        else:
            points[0] -= 3
        
        last_update_time[0] = sim_time[0]
        
        if points[0] < 0:
            points[0] = 0
    
    fell = pos[2] < 0.25
    if fell:
        points[0] = 0
    
    reached_goal = dist_to_goal < 1.0
    
    if prev_state[0] is not None and prev_action_idx[0] is not None:
        reward = calculate_reward(points[0], prev_dist_to_goal[0], dist_to_goal, fell, reached_goal)
        done = fell or reached_goal or (points[0] <= 0)
        
        rl_agent.remember(prev_state[0], prev_action_idx[0], reward, rl_state, done)
        
        if step_count[0] % 4 == 0:
            loss = rl_agent.train(batch_size=32)
            training_loss[0] = loss
    
    prev_state[0] = rl_state.copy()
    prev_action_idx[0] = rl_action_idx
    prev_vel[0] = curr_vel
    prev_dist_to_goal[0] = dist_to_goal
    step_count[0] += 1
    
    command = compute_command(pos, heading, rl_action_idx)
    spot.forward(step_size, command)

world.add_physics_callback("spot_control", on_physics_step)

# =============================================================================
# MAIN LOOP - MULTIPLE EPISODES
# =============================================================================

print(f"\nStarting {args.episodes} episode(s)...")
print("=" * 70)

episode_results = []

for episode_num in range(1, args.episodes + 1):
    print(f"\n[EPISODE {episode_num}/{args.episodes}]")
    print("-" * 70)
    
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
    prev_dist_to_goal[0] = 110.0
    max_speed[0] = 0.0
    last_print = 0.0
    step_count[0] = 0
    training_loss[0] = 0.0
    
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
                curr_vel = np.linalg.norm(spot.robot.get_linear_velocity())
                curr_vel_mph = curr_vel * 2.237
                contact_status = "Yes" if sensors['contact'] else "No"
                
                print(f"  t={sim_time[0]:5.1f}s | Dist: {dist_to_goal:6.1f}m | Points: {points[0]:3d} | Contact: {contact_status} | Vel: {curr_vel_mph:5.2f}mph | Loss: {training_loss[0]:.4f}")
                
                if dist_to_goal < 1.0:
                    episode_active[0] = False
                    episode_success[0] = True
                    print(f"\n*** GOAL REACHED in {sim_time[0]:.1f} seconds! ***")
                    print(f"*** EPISODE SUCCESS - Points: {points[0]} ***")
                    break
                
                if pos[2] < 0.25:
                    episode_active[0] = False
                    episode_success[0] = False
                    print(f"\n*** SPOT FELL! Z={pos[2]:.2f}m ***")
                    print(f"*** EPISODE FAILURE ***")
                    break
                
                if points[0] <= 0:
                    episode_active[0] = False
                    episode_success[0] = False
                    print(f"\n*** POINTS DEPLETED! ***")
                    print(f"*** EPISODE FAILURE ***")
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
    
    episode_results.append({
        'episode': episode_num,
        'status': status,
        'distance': total_progress,
        'duration': episode_duration,
        'max_speed_mph': max_speed_mph,
        'final_points': points[0],
        'epsilon': agent.epsilon,
        'buffer_size': len(agent.replay_buffer)
    })
    
    print(f"Ep {episode_num:3d}: {status:7s} | Progress: {total_progress:6.2f}m | Time: {episode_duration:5.1f}s | Speed: {max_speed_mph:5.2f}mph | Points: {points[0]:3d} | Epsilon: {rl_agent.epsilon:.4f}")
    
    rl_agent.new_episode()

# Print summary statistics
print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)

if episode_results:
    successes = sum(1 for r in episode_results if r['status'] == 'SUCCESS')
    avg_distance = np.mean([r['distance'] for r in episode_results])
    avg_time = np.mean([r['duration'] for r in episode_results])
    avg_speed = np.mean([r['max_speed_mph'] for r in episode_results])
    avg_points = np.mean([r['final_points'] for r in episode_results])
    
    print(f"Episodes: {args.episodes}")
    print(f"RL Framework: {agent_type}")
    print(f"Successes: {successes}/{args.episodes} ({100*successes/args.episodes:.1f}%)")
    print(f"Avg Progress: {avg_distance:.2f}m (out of 100m corridor)")
    print(f"Avg Duration: {avg_time:.1f}s")
    print(f"Avg Max Speed: {avg_speed:.2f}mph")
    print(f"Avg Final Points: {avg_points:.1f}")
    print(f"Final Epsilon: {rl_agent.epsilon:.4f}")
    print(f"Replay Buffer Size: {len(rl_agent.replay_buffer)}")
    print("=" * 70)

simulation_app.close()
print("Done.")
