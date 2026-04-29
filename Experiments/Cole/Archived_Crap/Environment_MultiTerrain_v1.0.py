#!/usr/bin/env python3
"""
MULTI-TERRAIN MULTI-ROBOT TRAINING ENVIRONMENT v1.0
Training 8 Spot robots simultaneously across 8 different terrain types
Shared DQN policy learns from all robot experiences in parallel

Terrain Types:
1. Flat (baseline)
2. Beach Sand
3. Small Gravel (1/4" - 1/2")
4. Medium Gravel (1/2" - 1")
5. Large Gravel (1" - 2")
6. Mud
7. Thick Grass/Brush
8. Short Grass
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from isaacsim import SimulationApp

# Create simulation app
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage
from omni.isaac.core.utils.prims import create_prim
from pxr import Gf, UsdGeom, UsdPhysics, Sdf
from omni.isaac.core.world import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import math

# =============================================================================
# DEEP Q-NETWORK IMPLEMENTATION
# =============================================================================

class DQNNetwork(nn.Module):
    def __init__(self, state_dim=2, action_dim=9):
        super(DQNNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class ExperienceReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim=2, action_dim=9, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.main_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)
        self.replay_buffer = ExperienceReplayBuffer(capacity=50000)
        
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.target_update_frequency = 1000
        self.steps = 0
    
    def normalize_state(self, distance, heading):
        """Normalize state to [-1, 1] range"""
        normalized_distance = distance / 110.0
        normalized_heading = heading / math.pi
        return np.array([normalized_distance, normalized_heading], dtype=np.float32)
    
    def get_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.main_network(state_tensor)
        return q_values.argmax(dim=1).item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def train(self, batch_size=32):
        """Train the DQN on batch of experiences"""
        if len(self.replay_buffer) < batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        current_q = self.main_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        next_q = self.target_network(next_states_t).max(dim=1)[0]
        target_q = rewards_t + (1 - dones_t) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Stage and world setup
stage = create_new_stage()
world = World(stage_units_in_meters=1.0)

print("=" * 70)
print("MULTI-TERRAIN MULTI-ROBOT TRAINING ENVIRONMENT v1.0")
print("=" * 70)
print(f"Training 8 Spot robots across 8 terrain types")
print(f"Environment: 110m long, 8 corridors × 15m wide = 120m total width")
print("=" * 70)

# =============================================================================
# TERRAIN CREATION FUNCTIONS
# =============================================================================

def create_flat_corridor(world, start_y, end_y, corridor_index):
    """Create flat baseline corridor"""
    print(f"Corridor {corridor_index}: FLAT TERRAIN")
    # Ground is already black from base

def create_beach_sand(world, start_y, end_y, corridor_index):
    """Create beach sand obstacle"""
    print(f"Corridor {corridor_index}: BEACH SAND")
    sand = UsdGeom.Cube.Define(world.stage, f"/World/Sand_{corridor_index}")
    sand_xform = UsdGeom.Xformable(sand.GetPrim())
    sand_center_y = (start_y + end_y) / 2.0
    sand_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, sand_center_y, 0.08))
    sand_xform.AddScaleOp().Set(Gf.Vec3f(100.0, 15.0, 0.15))
    sand.GetDisplayColorAttr().Set([Gf.Vec3f(0.95, 0.88, 0.60)])
    
    sand_prim = sand.GetPrim()
    UsdPhysics.CollisionAPI.Apply(sand_prim)
    UsdPhysics.MaterialAPI.Apply(sand_prim)
    sand_prim.GetAttribute("physics:staticFriction").Set(0.8)
    sand_prim.GetAttribute("physics:dynamicFriction").Set(0.7)

def create_small_gravel(world, start_y, end_y, corridor_index):
    """Create small gravel (1/4" - 1/2") obstacle"""
    print(f"Corridor {corridor_index}: SMALL GRAVEL (1/4\" - 1/2\")")
    create_gravel_corridor(world, start_y, end_y, corridor_index, 
                          pebble_min=0.00635, pebble_max=0.0127, color=(0.65, 0.58, 0.50))

def create_medium_gravel(world, start_y, end_y, corridor_index):
    """Create medium gravel (1/2" - 1") obstacle"""
    print(f"Corridor {corridor_index}: MEDIUM GRAVEL (1/2\" - 1\")")
    create_gravel_corridor(world, start_y, end_y, corridor_index,
                          pebble_min=0.0127, pebble_max=0.0254, color=(0.6, 0.5, 0.45))

def create_large_gravel(world, start_y, end_y, corridor_index):
    """Create large gravel (1" - 2") obstacle"""
    print(f"Corridor {corridor_index}: LARGE GRAVEL (1\" - 2\")")
    create_gravel_corridor(world, start_y, end_y, corridor_index,
                          pebble_min=0.0254, pebble_max=0.0508, color=(0.55, 0.48, 0.42))

def create_gravel_corridor(world, start_y, end_y, corridor_index, pebble_min, pebble_max, color):
    """Helper function to create gravel with varying pebble sizes"""
    # Base gravel area (visual only, for ground texture)
    gravel_base = UsdGeom.Cube.Define(world.stage, f"/World/GravelBase_{corridor_index}")
    gravel_xform = UsdGeom.Xformable(gravel_base.GetPrim())
    gravel_center_y = (start_y + end_y) / 2.0
    gravel_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, gravel_center_y, 0.08))
    gravel_xform.AddScaleOp().Set(Gf.Vec3f(100.0, 15.0, 0.15))
    gravel_base.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    
    gravel_prim = gravel_base.GetPrim()
    UsdPhysics.CollisionAPI.Apply(gravel_prim)
    
    # Scatter pebbles (only highest detail pebbles for performance)
    random.seed(42 + corridor_index)
    num_pebbles = 80
    for i in range(num_pebbles):
        pebble_x = random.uniform(-50.0, 50.0)
        pebble_y = random.uniform(start_y + 1.0, end_y - 1.0)
        pebble_size = random.uniform(pebble_min, pebble_max)
        pebble_z = 0.1 + pebble_size
        
        pebble = UsdGeom.Sphere.Define(world.stage, f"/World/Pebble_{corridor_index}_{i}")
        pebble_xform = UsdGeom.Xformable(pebble.GetPrim())
        pebble_xform.AddTranslateOp().Set(Gf.Vec3d(pebble_x, pebble_y, pebble_z))
        pebble_xform.AddScaleOp().Set(Gf.Vec3f(pebble_size, pebble_size, pebble_size))
        pebble.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
        
        pebble_prim = pebble.GetPrim()
        UsdPhysics.CollisionAPI.Apply(pebble_prim)
        UsdPhysics.RigidBodyAPI.Apply(pebble_prim)

def create_mud(world, start_y, end_y, corridor_index):
    """Create mud obstacle (high friction)"""
    print(f"Corridor {corridor_index}: MUD")
    mud = UsdGeom.Cube.Define(world.stage, f"/World/Mud_{corridor_index}")
    mud_xform = UsdGeom.Xformable(mud.GetPrim())
    mud_center_y = (start_y + end_y) / 2.0
    mud_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, mud_center_y, 0.08))
    mud_xform.AddScaleOp().Set(Gf.Vec3f(100.0, 15.0, 0.15))
    mud.GetDisplayColorAttr().Set([Gf.Vec3f(0.4, 0.3, 0.2)])
    
    mud_prim = mud.GetPrim()
    UsdPhysics.CollisionAPI.Apply(mud_prim)
    UsdPhysics.MaterialAPI.Apply(mud_prim)
    mud_prim.GetAttribute("physics:staticFriction").Set(1.2)
    mud_prim.GetAttribute("physics:dynamicFriction").Set(1.0)

def create_thick_grass_brush(world, start_y, end_y, corridor_index):
    """Create tall grass and brush"""
    print(f"Corridor {corridor_index}: THICK GRASS & BRUSH")
    random.seed(42 + corridor_index)
    num_obstacles = 100
    for i in range(num_obstacles):
        obs_x = random.uniform(-50.0, 50.0)
        obs_y = random.uniform(start_y + 1.0, end_y - 1.0)
        obs_height = random.uniform(0.5, 1.2)
        obs_width = random.uniform(0.1, 0.3)
        
        grass = UsdGeom.Cylinder.Define(world.stage, f"/World/Grass_{corridor_index}_{i}")
        grass_xform = UsdGeom.Xformable(grass.GetPrim())
        grass_xform.AddTranslateOp().Set(Gf.Vec3d(obs_x, obs_y, obs_height / 2.0))
        grass_xform.AddScaleOp().Set(Gf.Vec3f(obs_width, obs_width, obs_height))
        grass.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.5, 0.2)])
        
        grass_prim = grass.GetPrim()
        UsdPhysics.CollisionAPI.Apply(grass_prim)
        UsdPhysics.RigidBodyAPI.Apply(grass_prim)

def create_short_grass(world, start_y, end_y, corridor_index):
    """Create short grass (low resistance, tall appearance)"""
    print(f"Corridor {corridor_index}: SHORT GRASS")
    random.seed(42 + corridor_index)
    num_obstacles = 150
    for i in range(num_obstacles):
        obs_x = random.uniform(-50.0, 50.0)
        obs_y = random.uniform(start_y + 1.0, end_y - 1.0)
        obs_height = random.uniform(0.15, 0.35)
        obs_width = random.uniform(0.05, 0.15)
        
        grass = UsdGeom.Cylinder.Define(world.stage, f"/World/ShortGrass_{corridor_index}_{i}")
        grass_xform = UsdGeom.Xformable(grass.GetPrim())
        grass_xform.AddTranslateOp().Set(Gf.Vec3d(obs_x, obs_y, obs_height / 2.0))
        grass_xform.AddScaleOp().Set(Gf.Vec3f(obs_width, obs_width, obs_height))
        grass.GetDisplayColorAttr().Set([Gf.Vec3f(0.3, 0.6, 0.2)])
        
        grass_prim = grass.GetPrim()
        UsdPhysics.CollisionAPI.Apply(grass_prim)
        UsdPhysics.RigidBodyAPI.Apply(grass_prim)

# =============================================================================
# CREATE WORLD SETUP
# =============================================================================

# Ground plane
ground_plane = UsdGeom.Cube.Define(world.stage, "/World/GroundPlane")
ground_xform = UsdGeom.Xformable(ground_plane.GetPrim())
ground_xform.AddTranslateOp().Set(Gf.Vec3d(0, 60, -0.1))
ground_xform.AddScaleOp().Set(Gf.Vec3f(110.0, 120.0, 0.2))
ground_plane.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 0.0)])
UsdPhysics.CollisionAPI.Apply(ground_plane.GetPrim())

# Create 8 parallel corridors with terrain
corridor_configs = [
    ("Flat", create_flat_corridor),
    ("Beach Sand", create_beach_sand),
    ("Small Gravel", create_small_gravel),
    ("Medium Gravel", create_medium_gravel),
    ("Large Gravel", create_large_gravel),
    ("Mud", create_mud),
    ("Thick Grass", create_thick_grass_brush),
    ("Short Grass", create_short_grass),
]

corridor_y_positions = []
for i in range(8):
    start_y = i * 15.0
    end_y = (i + 1) * 15.0
    corridor_y_positions.append((start_y, end_y))
    
    # Create corridor markers (clear lines of separation)
    marker_left = UsdGeom.Cube.Define(world.stage, f"/World/CorridorMarker_{i}_left")
    marker_xform = UsdGeom.Xformable(marker_left.GetPrim())
    marker_xform.AddTranslateOp().Set(Gf.Vec3d(-55.0, (start_y + end_y) / 2.0 - 0.25, 0.0))
    marker_xform.AddScaleOp().Set(Gf.Vec3f(110.0, 0.5, 0.02))
    marker_left.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 1.0, 1.0)])
    
    # Create terrain for this corridor
    terrain_func = corridor_configs[i][1]
    terrain_func(world, start_y, end_y, i)
    
    # Create start zone (green 1m × 1m square)
    start_zone = UsdGeom.Cube.Define(world.stage, f"/World/StartZone_{i}")
    start_xform = UsdGeom.Xformable(start_zone.GetPrim())
    start_xform.AddTranslateOp().Set(Gf.Vec3d(-52.5, (start_y + end_y) / 2.0, 0.05))
    start_xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 0.1))
    start_zone.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.8, 0.2)])
    
    # Create end zone (black sphere with red flag)
    end_sphere = UsdGeom.Sphere.Define(world.stage, f"/World/EndZone_{i}")
    end_xform = UsdGeom.Xformable(end_sphere.GetPrim())
    end_xform.AddTranslateOp().Set(Gf.Vec3d(52.5, (start_y + end_y) / 2.0, 1.0))
    end_xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    end_sphere.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 0.0)])
    
    # Red flag
    flag = UsdGeom.Cylinder.Define(world.stage, f"/World/Flag_{i}")
    flag_xform = UsdGeom.Xformable(flag.GetPrim())
    flag_xform.AddTranslateOp().Set(Gf.Vec3d(52.5, (start_y + end_y) / 2.0, 2.0))
    flag_xform.AddScaleOp().Set(Gf.Vec3f(0.1, 0.5, 0.1))
    flag.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])

print("Environment created with 8 parallel corridors")
print("=" * 70)

# =============================================================================
# SPAWN 8 SPOT ROBOTS
# =============================================================================

# Spawn 8 Spot robots (one per corridor)
spot_robots = []
for i in range(8):
    robot_name = f"Spot_{i}"
    start_y = corridor_y_positions[i][0]
    end_y = corridor_y_positions[i][1]
    spawn_y = (start_y + end_y) / 2.0
    
    # Add Spot robot
    spot_path = f"/World/{robot_name}"
    add_reference_to_stage(
        usd_path="omniverse://localhost/Projects/Robotics/Robots/Spot/spot.usd",
        prim_path=spot_path
    )
    
    robot = Robot(prim_path=spot_path, name=robot_name)
    robot.set_world_pose(position=[-52.5, spawn_y, 0.5], orientation=[0, 0, 0, 1])
    spot_robots.append(robot)

print(f"Spawned 8 Spot robots")

# =============================================================================
# TRAINING SETUP
# =============================================================================

world.initialize_physics()
agent = DQNAgent(state_dim=2, action_dim=9, learning_rate=0.001)

# Tracking variables
episode_steps = [0] * 8
episode_rewards = [0.0] * 8
episode_distances = [0.0] * 8
training_loss = [0.0] * 8

print("=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)
print(f"DQN Agent initialized")
print(f"  - State dim: 2 (distance to goal, heading)")
print(f"  - Action dim: 9 (motor commands)")
print(f"  - Replay buffer: 50,000 capacity")
print(f"  - Batch size: 32")
print(f"  - Learning rate: 0.001")
print(f"  - Gamma (discount): 0.95")
print(f"  - Epsilon (exploration): 1.0 -> 0.1")
print(f"All 8 robots share ONE policy")
print("=" * 70)

# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def run_simulation(num_episodes=1):
    for episode in range(num_episodes):
        print(f"\n[EPISODE {episode + 1}/{num_episodes}]")
        
        # Reset all robots
        for i, robot in enumerate(spot_robots):
            start_y = corridor_y_positions[i][0]
            end_y = corridor_y_positions[i][1]
            spawn_y = (start_y + end_y) / 2.0
            robot.set_world_pose(position=[-52.5, spawn_y, 0.5], orientation=[0, 0, 0, 1])
            episode_steps[i] = 0
            episode_rewards[i] = 500.0
            episode_distances[i] = 0.0
        
        agent.decay_epsilon()
        
        # Run episode
        for step in range(600):  # Max 600 steps (~60 seconds at 10Hz)
            world.step(render=True)
            
            for i, robot in enumerate(spot_robots):
                start_y = corridor_y_positions[i][0]
                end_y = corridor_y_positions[i][1]
                goal_y = (start_y + end_y) / 2.0
                
                # Get robot state
                position = robot.get_world_pose()[0]
                pos_x, pos_y = position[0], position[1]
                
                # Calculate distance to goal
                dist_to_goal = np.sqrt((52.5 - pos_x) ** 2 + (goal_y - pos_y) ** 2)
                episode_distances[i] = 105.0 - dist_to_goal
                
                # Heading (simplified - angle toward goal)
                dx = 52.5 - pos_x
                heading = math.atan2(goal_y - pos_y, dx)
                
                # Get normalized state
                state = agent.normalize_state(dist_to_goal, heading)
                
                # Get action and execute
                action = agent.get_action(state, training=True)
                
                # Simple motor command mapping (0-8 actions)
                velocity_cmd = [0.5, -0.5, 0.5, -0.5][action % 4] if action < 4 else 0.3
                angular_cmd = [-0.3, 0.3, 0.0, 0.0][action % 4] if action < 4 else 0.1
                
                # Calculate reward
                reward = max(0, episode_rewards[i] - 1.0)
                if dist_to_goal < 1.5:  # Goal reached
                    reward = 500.0
                    
                episode_rewards[i] = reward
                
                # Store experience and train
                prev_state = agent.normalize_state(dist_to_goal + 0.5, heading)
                next_state = state
                agent.remember(prev_state, action, reward, next_state, dist_to_goal < 1.5)
                
                if step % 4 == 0:
                    loss = agent.train(batch_size=32)
                    training_loss[i] = loss
                
                episode_steps[i] += 1
        
        # Print episode summary
        avg_distance = np.mean(episode_distances)
        avg_reward = np.mean(episode_rewards)
        avg_loss = np.mean([l for l in training_loss if l > 0])
        print(f"Episode {episode + 1}: Avg Distance: {avg_distance:.1f}m | Avg Reward: {avg_reward:.0f} | Loss: {avg_loss:.4f}")
        print(f"Replay Buffer Size: {len(agent.replay_buffer)}")
        print(f"Epsilon: {agent.epsilon:.4f}")

# Run training
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()
    
    try:
        run_simulation(num_episodes=args.episodes)
    finally:
        simulation_app.close()
