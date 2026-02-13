"""
Spot Robot RL Training - Cole_vs2
==================================
Reinforcement Learning agent trained to navigate 90m training area.

Agent learns:
- Variable forward speed (0.5-3.0 m/s)
- Turn rate optimization

Observations:
- Distance to goal
- Current points (score)
- Robot velocity

Reward System:
- Start: 500 points
- Per second: -1 point
- Fall: -500 (automatic failure)
- Success: +250 points
- Faster than moving average: +10 points per second saved
"""

import numpy as np
import argparse
from collections import deque
import json
from pathlib import Path

# Parse args before SimulationApp
parser = argparse.ArgumentParser(description="Spot RL Training")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to train")
args = parser.parse_args()

# Isaac Sim setup - MUST come before other Isaac imports
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

# Now import Isaac modules
import omni
from omni.isaac.core import World
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
from pxr import UsdGeom, UsdLux, Gf

# RL imports
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

print("=" * 70)
print("SPOT ROBOT RL TRAINING - Cole_vs2")
print("=" * 70)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Environment
START_X = -45.0
START_Y = 0.0
START_Z = 0.7
END_X = 45.0
END_Y = 0.0
GOAL = np.array([END_X, END_Y])
FIELD_WIDTH = 50.0
MAX_EPISODE_TIME = 120.0  # seconds

# Agent
MIN_SPEED = 1.0
MAX_SPEED = 2.0  # Conservative range - Cole_vs1 works at 1.5 m/s
MAX_TURN_RATE = 1.0

# Reward
INITIAL_POINTS = 500
TIME_PENALTY = 1.0  # per second
SUCCESS_BONUS = 250
FALL_PENALTY = 500
SPEED_BONUS = 10.0  # per second saved vs moving average
START_DELAY = 10.0  # seconds - extended for more physics settling before RL control

# =============================================================================
# GYM ENVIRONMENT
# =============================================================================

class SpotNavigationEnv(gym.Env):
    """OpenAI Gym environment for Spot navigation with RL"""
    
    metadata = {"render_modes": ["human"], "render_fps": 50}
    
    def __init__(self, headless=False):
        super().__init__()
        
        self.headless = headless
        self.world = None
        self.spot = None
        self.stage = None
        
        # Action space: [speed (-1 to 1), turn_rate (-1 to 1)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: [distance_to_goal, points, velocity_x, velocity_y]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -5.0, -5.0], dtype=np.float32),
            high=np.array([100, 500, 5.0, 5.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Metrics
        self.moving_avg_completion_time = deque(maxlen=10)
        self.episode_count = 0
        self.success_count = 0
        
        # Episode state
        self.sim_time = 0.0
        self.points = INITIAL_POINTS
        self.start_pos = None
        self.fallen = False
        self.succeeded = False
        
        self._setup_world()
    
    def _setup_world(self):
        """Initialize Isaac Sim world and Spot robot"""
        
        self.world = World(
            physics_dt=1.0 / 500.0,
            rendering_dt=10.0 / 500.0,
            stage_units_in_meters=1.0
        )
        self.stage = omni.usd.get_context().get_stage()
        
        # Add lighting
        dome_light = UsdLux.DomeLight.Define(self.stage, "/World/Lights/DomeLight")
        dome_light.CreateIntensityAttr(1000.0)
        dome_light.CreateColorAttr(Gf.Vec3f(0.95, 0.95, 1.0))
        
        # Add ground plane
        self.world.scene.add_default_ground_plane(
            z_position=0,
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.01,
        )
        
        # Create black background
        TRAINING_AREA_MIN_X = START_X - 5.0
        TRAINING_AREA_MAX_X = END_X + 5.0
        black_bg = UsdGeom.Mesh.Define(self.stage, "/World/TrainingAreaBG")
        black_bg.GetPointsAttr().Set([
            Gf.Vec3f(TRAINING_AREA_MIN_X, -FIELD_WIDTH/2, 0.005),
            Gf.Vec3f(TRAINING_AREA_MAX_X, -FIELD_WIDTH/2, 0.005),
            Gf.Vec3f(TRAINING_AREA_MAX_X, FIELD_WIDTH/2, 0.005),
            Gf.Vec3f(TRAINING_AREA_MIN_X, FIELD_WIDTH/2, 0.005)
        ])
        black_bg.GetFaceVertexCountsAttr().Set([4])
        black_bg.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        black_bg.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 0.0)])
        
        # Green square at start position
        start_square = UsdGeom.Mesh.Define(self.stage, "/World/StartMarker")
        start_square.GetPointsAttr().Set([
            Gf.Vec3f(START_X - 0.5, START_Y - 0.5, 0.01),
            Gf.Vec3f(START_X + 0.5, START_Y - 0.5, 0.01),
            Gf.Vec3f(START_X + 0.5, START_Y + 0.5, 0.01),
            Gf.Vec3f(START_X - 0.5, START_Y + 0.5, 0.01)
        ])
        start_square.GetFaceVertexCountsAttr().Set([4])
        start_square.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        start_square.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.8, 0.2)])
        
        # Purple sphere at end position
        end_sphere = UsdGeom.Sphere.Define(self.stage, "/World/EndMarker")
        end_xform = UsdGeom.Xformable(end_sphere.GetPrim())
        end_xform.AddTranslateOp().Set(Gf.Vec3d(END_X, END_Y, 0.5))
        end_xform.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 0.5))
        end_sphere.GetDisplayColorAttr().Set([Gf.Vec3f(0.8, 0.2, 0.8)])
        
        # Create Spot
        self.spot = SpotFlatTerrainPolicy(
            prim_path="/World/Spot",
            name="Spot",
            position=np.array([START_X, START_Y, START_Z]),
        )
        
        self.world.reset()
        self.spot.initialize()
        self.spot.robot.set_joints_default_state(self.spot.default_pos)
        
        # Stabilize robot before starting training - longer wait ensures physics settle
        for _ in range(500):
            self.world.step(render=False)
    
    def _get_observation(self):
        """Get current observation"""
        pos, _, vel = self._get_robot_state()
        dist_to_goal = np.linalg.norm(GOAL - pos[:2])
        
        obs = np.array([
            dist_to_goal,
            self.points,
            vel[0],
            vel[1]
        ], dtype=np.float32)
        
        return obs
    
    def _get_robot_state(self):
        """Get robot position, heading, and velocity"""
        pos, quat = self.spot.robot.get_world_pose()
        vel, _ = self.spot.robot.get_linear_velocity(), self.spot.robot.get_angular_velocity()
        
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        heading = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array(pos), heading, vel
    
    def _compute_action(self, action):
        """Convert normalized action to command"""
        speed = ((action[0] + 1.0) / 2.0) * (MAX_SPEED - MIN_SPEED) + MIN_SPEED
        speed = np.clip(speed, MIN_SPEED, MAX_SPEED)
        
        turn_rate = action[1] * MAX_TURN_RATE
        turn_rate = np.clip(turn_rate, -MAX_TURN_RATE, MAX_TURN_RATE)
        
        return np.array([speed, 0.0, turn_rate])
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        if seed is not None:
            np.random.seed(seed)
            
        if self.spot is not None:
            self.world.reset()
            self.spot.initialize()
            self.spot.robot.set_joints_default_state(self.spot.default_pos)
            # Longer stabilization to ensure robot is stable before episode starts
            for _ in range(500):
                self.world.step(render=False)
        
        self.sim_time = 0.0
        self.points = INITIAL_POINTS
        self.fallen = False
        self.succeeded = False
        self.start_pos, _, _ = self._get_robot_state()
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step of the environment"""
        
        # During startup delay, keep robot stationary for stability
        if self.sim_time < START_DELAY:
            command = np.array([0.0, 0.0, 0.0])  # Keep still
        else:
            command = self._compute_action(action)
        
        self.spot.forward(self.world.get_physics_dt(), command)
        self.world.step(render=not self.headless)
        
        self.sim_time += self.world.get_physics_dt()
        
        # Don't deduct points during startup grace period
        if self.sim_time > START_DELAY:
            self.points -= TIME_PENALTY * self.world.get_physics_dt()
        
        pos, heading, vel = self._get_robot_state()
        obs = self._get_observation()
        
        # Check failure conditions
        if pos[2] < 0.35:  # Increased threshold from 0.25 to avoid false positives
            self.fallen = True
            self.points = 0
            reward = -FALL_PENALTY
            done = True
            info = {"episode": self.episode_count, "reason": "fell"}
        
        elif self.points <= 0:
            reward = -self.points
            done = True
            info = {"episode": self.episode_count, "reason": "timeout"}
        
        else:
            # Check success
            dist_to_goal = np.linalg.norm(GOAL - pos[:2])
            if dist_to_goal < 1.0:
                self.succeeded = True
                self.points += SUCCESS_BONUS
                
                if len(self.moving_avg_completion_time) > 0:
                    avg_time = np.mean(list(self.moving_avg_completion_time))
                    time_saved = avg_time - self.sim_time
                    self.points += max(0, SPEED_BONUS * time_saved)
                
                self.moving_avg_completion_time.append(self.sim_time)
                self.success_count += 1
                
                reward = self.points
                done = True
                info = {
                    "episode": self.episode_count,
                    "reason": "success",
                    "completion_time": self.sim_time,
                    "final_points": self.points
                }
            
            else:
                reward = 0.0
                done = False
                info = {}
        
        return obs, reward, done, False, info
    
    def render(self, mode="human"):
        pass
    
    def close(self):
        if self.world is not None:
            simulation_app.close()

# =============================================================================
# TRAINING LOOP
# =============================================================================

def train():
    """Main training function"""
    
    print("\n" + "=" * 70)
    print("Initializing RL Training Environment")
    print("=" * 70)
    
    env = SpotNavigationEnv(headless=args.headless)
    
    print("\nEnvironment Created:")
    print(f"  Action Space: {env.action_space}")
    print(f"  Observation Space: {env.observation_space}")
    print(f"  Speed Range: [{MIN_SPEED}, {MAX_SPEED}] m/s")
    
    print("\nInitializing PPO Agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        n_steps=512,
        batch_size=64,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )
    
    print("Agent initialized!")
    print("\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70)
    
    episode_stats = {
        "episode_count": 0,
        "success_count": 0,
        "completion_times": [],
        "points_scores": [],
        "moving_avg_time": []
    }
    
    total_steps = 0
    for episode in range(args.episodes):
        print(f"\n--- Episode {episode + 1}/{args.episodes} ---")
        
        obs, _ = env.reset()
        env.episode_count = episode
        done = False
        
        while not done and simulation_app.is_running():
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            total_steps += 1
            
            if not args.headless:
                env.render()
        
        if env.succeeded:
            print(f"✓ SUCCESS in {env.sim_time:.2f}s | Points: {env.points:.0f}")
            episode_stats["success_count"] += 1
            episode_stats["completion_times"].append(env.sim_time)
        else:
            if env.fallen:
                print(f"✗ FELL | Points: 0")
            else:
                print(f"✗ TIMEOUT | Points: {env.points:.0f}")
        
        episode_stats["points_scores"].append(env.points)
        if len(env.moving_avg_completion_time) > 0:
            episode_stats["moving_avg_time"].append(np.mean(list(env.moving_avg_completion_time)))
        
        episode_stats["episode_count"] = episode + 1
        
        if (episode + 1) % 1 == 0:
            print(f"Training on {total_steps} steps...")
            model.learn(total_timesteps=512, reset_num_timesteps=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total Episodes: {episode_stats['episode_count']}")
    print(f"Successful Episodes: {episode_stats['success_count']}")
    print(f"Success Rate: {100 * episode_stats['success_count'] / max(1, episode_stats['episode_count']):.1f}%")
    
    if episode_stats["completion_times"]:
        print(f"Average Completion Time: {np.mean(episode_stats['completion_times']):.2f}s")
        print(f"Best Completion Time: {min(episode_stats['completion_times']):.2f}s")
    
    avg_points = np.mean(episode_stats["points_scores"])
    print(f"Average Points: {avg_points:.1f}")
    print(f"Max Points: {max(episode_stats['points_scores']):.1f}")
    
    model.save("spot_rl_policy")
    print(f"\nModel saved to: spot_rl_policy")
    
    with open("training_stats.json", 'w') as f:
        json.dump({
            "episode_stats": {k: (v if not isinstance(v, list) else [float(x) for x in v]) for k, v in episode_stats.items()},
            "config": {
                "initial_points": INITIAL_POINTS,
                "time_penalty": TIME_PENALTY,
                "success_bonus": SUCCESS_BONUS,
                "speed_bonus": SPEED_BONUS,
                "min_speed": MIN_SPEED,
                "max_speed": MAX_SPEED
            }
        }, f, indent=2)
    print(f"Stats saved to: training_stats.json")
    
    env.close()

if __name__ == "__main__":
    train()
