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


# Create module-level config instance
config = EnvConfig()

# Print configuration on import
print(f"Arena: {config.ARENA_LENGTH}m x {config.ARENA_WIDTH}m (centered at origin)")
print(f"Number of robots: {config.NUM_ROBOTS}")
print(f"Start X: {config.START_X}m, Goal X: {config.GOAL_X}m")
print(f"Distance to travel: {config.GOAL_X - config.START_X:.0f}m")
for i, y in enumerate(config.START_Y_POSITIONS):
    print(f"  Robot {i}: Start ({config.START_X}, {y}) -> Goal ({config.GOAL_X}, {y})")
print("=" * 70)



world = World(
physics_dt=config.PHYSICS_DT,
rendering_dt=config.RENDERING_DT,
stage_units_in_meters=1.0)

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