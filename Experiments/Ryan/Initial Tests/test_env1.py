"""
RAW Quadruped Training - Simple Go-To-Goal
===========================================
Clean implementation: Spot walks from start zone to end zone.

Environment:
  - 50m long x 25m wide room (centered at origin)
  - Start zone: 1m x 1m at X=-23.5, Y=0
  - End zone: opposite side at X=+24m

Author: MS for Autonomy Project
Date: January 2026
"""

import numpy as np
import argparse

# Parse args before SimulationApp
parser = argparse.ArgumentParser(description="RAW Quadruped Training")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
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
print("RAW QUADRUPED TRAINING - GO TO GOAL")
print("=" * 60)

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

# Room dimensions
ROOM_LENGTH = 50.0   # meters (X direction)
ROOM_WIDTH = 25.0    # meters (Y direction)

# Start zone (1m x 1m, centered at Y=0)
START_X = -23.5      # Center of start zone (1m from left wall, room centered at origin)
START_Y = 0.0
START_Z = 0.7        # Height for Spot

# End zone
END_X = 24.0         # Near the far wall (room centered at origin)
END_Y = 0.0

# Robot start position
ROBOT_START = np.array([START_X, START_Y, START_Z])

# Goal position
GOAL = np.array([END_X, END_Y])

print(f"Room: {ROOM_LENGTH}m x {ROOM_WIDTH}m")
print(f"Start: X={START_X}m, Y={START_Y}m")
print(f"Goal:  X={END_X}m, Y={END_Y}m")
print(f"Distance to travel: {END_X - START_X:.1f}m")
print("=" * 60)

# =============================================================================
# CREATE WORLD
# =============================================================================

# Create world with proper physics rate (500Hz for quadruped stability)
world = World(
    physics_dt=1.0 / 500.0,
    rendering_dt=10.0 / 500.0,  # Render every 10 physics steps
    stage_units_in_meters=1.0
)
stage = omni.usd.get_context().get_stage()

# Add lighting
dome_light = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
dome_light.CreateIntensityAttr(1000.0)
dome_light.CreateColorAttr(Gf.Vec3f(0.95, 0.95, 1.0))

# Add ground plane with physics - default ground plane is infinite
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
# CREATE VISUAL ZONES
# =============================================================================

# Start zone (green, 1m x 1m)
start_zone = UsdGeom.Mesh.Define(stage, "/World/StartZone")
start_zone.GetPointsAttr().Set([
    Gf.Vec3f(-ROOM_LENGTH/2, -0.5, 0.01),
    Gf.Vec3f(-ROOM_LENGTH/2 + 1, -0.5, 0.01),
    Gf.Vec3f(-ROOM_LENGTH/2 + 1, 0.5, 0.01),
    Gf.Vec3f(-ROOM_LENGTH/2, 0.5, 0.01)
])
start_zone.GetFaceVertexCountsAttr().Set([4])
start_zone.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
start_zone.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.8, 0.2)])  # Green

# End zone (blue, 1m x 1m at far end)
end_zone = UsdGeom.Mesh.Define(stage, "/World/EndZone")
end_zone.GetPointsAttr().Set([
    Gf.Vec3f(ROOM_LENGTH/2 - 1, -0.5, 0.01),
    Gf.Vec3f(ROOM_LENGTH/2, -0.5, 0.01),
    Gf.Vec3f(ROOM_LENGTH/2, 0.5, 0.01),
    Gf.Vec3f(ROOM_LENGTH/2 - 1, 0.5, 0.01)
])
end_zone.GetFaceVertexCountsAttr().Set([4])
end_zone.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
end_zone.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.2, 0.8)])  # Blue

print("Start zone (green) and end zone (blue) created")

# =============================================================================
# CREATE WALLS (Confined Space)
# =============================================================================

WALL_HEIGHT = 2.0    # 2 meter tall walls
WALL_THICKNESS = 0.5  # 0.5 meter thick walls
WALL_COLOR = Gf.Vec3f(0.5, 0.5, 0.5)  # Gray

# Left wall (X = -ROOM_LENGTH/2)
left_wall = UsdGeom.Cube.Define(stage, "/World/Walls/LeftWall")
left_xform = UsdGeom.Xformable(left_wall.GetPrim())
left_xform.AddTranslateOp().Set(Gf.Vec3d(-ROOM_LENGTH/2 - WALL_THICKNESS/2, 0, WALL_HEIGHT/2))
left_xform.AddScaleOp().Set(Gf.Vec3f(WALL_THICKNESS, ROOM_WIDTH, WALL_HEIGHT))
left_wall.GetDisplayColorAttr().Set([WALL_COLOR])
UsdPhysics.CollisionAPI.Apply(left_wall.GetPrim())

# Right wall (X = +ROOM_LENGTH/2)
right_wall = UsdGeom.Cube.Define(stage, "/World/Walls/RightWall")
right_xform = UsdGeom.Xformable(right_wall.GetPrim())
right_xform.AddTranslateOp().Set(Gf.Vec3d(ROOM_LENGTH/2 + WALL_THICKNESS/2, 0, WALL_HEIGHT/2))
right_xform.AddScaleOp().Set(Gf.Vec3f(WALL_THICKNESS, ROOM_WIDTH, WALL_HEIGHT))
right_wall.GetDisplayColorAttr().Set([WALL_COLOR])
UsdPhysics.CollisionAPI.Apply(right_wall.GetPrim())

# Top wall (Y = +ROOM_WIDTH/2)
top_wall = UsdGeom.Cube.Define(stage, "/World/Walls/TopWall")
top_xform = UsdGeom.Xformable(top_wall.GetPrim())
top_xform.AddTranslateOp().Set(Gf.Vec3d(0, ROOM_WIDTH/2 + WALL_THICKNESS/2, WALL_HEIGHT/2))
top_xform.AddScaleOp().Set(Gf.Vec3f(ROOM_LENGTH + WALL_THICKNESS*2, WALL_THICKNESS, WALL_HEIGHT))
top_wall.GetDisplayColorAttr().Set([WALL_COLOR])
UsdPhysics.CollisionAPI.Apply(top_wall.GetPrim())

# Bottom wall (Y = -ROOM_WIDTH/2)
bottom_wall = UsdGeom.Cube.Define(stage, "/World/Walls/BottomWall")
bottom_xform = UsdGeom.Xformable(bottom_wall.GetPrim())
bottom_xform.AddTranslateOp().Set(Gf.Vec3d(0, -ROOM_WIDTH/2 - WALL_THICKNESS/2, WALL_HEIGHT/2))
bottom_xform.AddScaleOp().Set(Gf.Vec3f(ROOM_LENGTH + WALL_THICKNESS*2, WALL_THICKNESS, WALL_HEIGHT))
bottom_wall.GetDisplayColorAttr().Set([WALL_COLOR])
UsdPhysics.CollisionAPI.Apply(bottom_wall.GetPrim())

print(f"Walls created: {ROOM_LENGTH}m x {ROOM_WIDTH}m enclosed space, {WALL_HEIGHT}m tall")

# =============================================================================
# CREATE SPOT ROBOT
# =============================================================================

spot = SpotFlatTerrainPolicy(
    prim_path="/World/Spot",
    name="Spot",
    position=ROBOT_START,
)
print(f"Spot created at position: {ROBOT_START}")

# CRITICAL: Reset world BEFORE accessing spot properties
world.reset()

# Initialize the robot
spot.initialize()
spot.robot.set_joints_default_state(spot.default_pos)
print("Spot initialized with default joint state")

# CRITICAL: Run a few physics steps to fully initialize the simulation view
for _ in range(10):
    world.step(render=False)

# =============================================================================
# GO-TO-GOAL CONTROLLER
# =============================================================================

# Control parameters
FORWARD_SPEED = 1.5      # m/s base forward speed
TURN_GAIN = 2.0          # Proportional gain for turning
HEADING_THRESHOLD = 0.1  # radians - when aligned enough to go full speed

# Simulation state
sim_time = [0.0]
physics_ready = [False]
STABILIZE_TIME = 3.0     # Wait 3 seconds before moving (time to position camera)

def get_robot_state():
    """Get robot position and heading"""
    pos, quat = spot.robot.get_world_pose()
    
    # Extract yaw from quaternion
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    heading = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array(pos), heading

def compute_go_to_goal_command(pos, heading):
    """Compute velocity command to reach goal"""
    # Vector to goal
    to_goal = GOAL - pos[:2]
    dist_to_goal = np.linalg.norm(to_goal)
    
    # Desired heading (angle to goal)
    desired_heading = np.arctan2(to_goal[1], to_goal[0])
    
    # Heading error (wrapped to [-pi, pi])
    heading_error = desired_heading - heading
    while heading_error > np.pi:
        heading_error -= 2 * np.pi
    while heading_error < -np.pi:
        heading_error += 2 * np.pi
    
    # Proportional turn control
    turn_rate = TURN_GAIN * heading_error
    turn_rate = np.clip(turn_rate, -1.0, 1.0)
    
    # Forward speed: full speed if aligned, reduced if turning
    if abs(heading_error) < HEADING_THRESHOLD:
        forward_speed = FORWARD_SPEED
    else:
        forward_speed = FORWARD_SPEED * 0.3  # Slow down while turning
    
    # Command: [forward, lateral, angular]
    return np.array([forward_speed, 0.0, turn_rate])

def on_physics_step(step_size):
    """Physics callback - runs at 500Hz"""
    if not physics_ready[0]:
        physics_ready[0] = True
        return
    
    sim_time[0] += step_size
    
    # Stabilization period - no movement
    if sim_time[0] < STABILIZE_TIME:
        spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
        return
    
    # Get robot state
    pos, heading = get_robot_state()
    
    # Compute and send command
    command = compute_go_to_goal_command(pos, heading)
    spot.forward(step_size, command)

# Register physics callback
world.add_physics_callback("spot_control", on_physics_step)

# =============================================================================
# MAIN SIMULATION LOOP
# =============================================================================

print(f"\nStabilizing for {STABILIZE_TIME}s, then walking to goal...")
print("-" * 60)

start_pos, _ = get_robot_state()
print(f"Initial position: X={start_pos[0]:.2f}m, Y={start_pos[1]:.2f}m")

last_print_time = 0.0

try:
    while simulation_app.is_running():
        world.step(render=not args.headless)
        
        # Print status every second
        if sim_time[0] - last_print_time >= 1.0:
            last_print_time = sim_time[0]
            
            pos, heading = get_robot_state()
            dist_to_goal = np.linalg.norm(GOAL - pos[:2])
            progress = pos[0] - start_pos[0]
            
            print(f"  t={sim_time[0]:5.1f}s | X={pos[0]:6.2f}m Y={pos[1]:5.2f}m | "
                  f"Progress: {progress:5.2f}m | Dist to goal: {dist_to_goal:5.1f}m")
            
            # Check if reached goal
            if pos[0] >= END_X - 1.0:
                print(f"\n*** GOAL REACHED in {sim_time[0]:.1f} seconds! ***")
                break
            
            # Check if fell
            if pos[2] < 0.25:
                print(f"\n*** SPOT FELL! Z={pos[2]:.2f}m ***")
                break
            
            # Timeout after 60 seconds
            if sim_time[0] > 60:
                print(f"\n*** TIMEOUT after 60s ***")
                break

except KeyboardInterrupt:
    print("\nExiting...")

# =============================================================================
# FINAL REPORT
# =============================================================================

final_pos, _ = get_robot_state()
total_progress = final_pos[0] - start_pos[0]

print("\n" + "=" * 60)
print("SESSION COMPLETE")
print("=" * 60)
print(f"Total X progress: {total_progress:.2f}m")
print(f"Final position: X={final_pos[0]:.2f}m, Y={final_pos[1]:.2f}m, Z={final_pos[2]:.2f}m")
if final_pos[0] >= END_X - 1.0:
    print("STATUS: SUCCESS!")
else:
    print(f"STATUS: Did not reach goal (needed {END_X - 1.0:.1f}m)")
print("=" * 60)

simulation_app.close()
print("Done.")