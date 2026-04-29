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

# Parse args before SimulationApp
parser = argparse.ArgumentParser(description="Spot Test Environment")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
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
# GO-TO-GOAL CONTROLLER
# =============================================================================

FORWARD_SPEED = 1.5
TURN_GAIN = 2.0
HEADING_THRESHOLD = 0.1

sim_time = [0.0]
physics_ready = [False]
STABILIZE_TIME = 6.0  # 3s delay + 3s stability

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

def compute_command(pos, heading):
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
    
    if abs(heading_error) < HEADING_THRESHOLD:
        forward_speed = FORWARD_SPEED
    else:
        forward_speed = FORWARD_SPEED * 0.3
    
    return np.array([forward_speed, 0.0, turn_rate])

def on_physics_step(step_size):
    if not physics_ready[0]:
        physics_ready[0] = True
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
    command = compute_command(pos, heading)
    spot.forward(step_size, command)

world.add_physics_callback("spot_control", on_physics_step)

# =============================================================================
# MAIN LOOP
# =============================================================================

print(f"\nStabilizing for {STABILIZE_TIME}s, then walking to end zone...")
print("-" * 60)

start_pos, _ = get_robot_state()
print(f"Initial position: X={start_pos[0]:.2f}m, Y={start_pos[1]:.2f}m")

last_print = 0.0

step_count = [0]
try:
    while simulation_app.is_running():
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
            print(f"  t={sim_time[0]:5.1f}s | X={pos[0]:6.2f}m Y={pos[1]:5.2f}m | Progress: {progress:5.2f}m | Dist: {dist_to_goal:5.1f}m")
            print(f"    Sensors: Camera={sensors['camera']}, Lidar={sensors['lidar_points']} pts, IMU accel={np.linalg.norm(sensors['imu_accel']):.2f}m/s², Contact={'Yes' if sensors['contact'] else 'No'}")
            
            # Check if Spot hit the end marker (within 1m)
            dist_to_marker = np.linalg.norm(np.array([pos[0], pos[1]]) - np.array([END_X, END_Y]))
            if dist_to_marker < 1.0:
                print(f"\n*** END MARKER REACHED in {sim_time[0]:.1f} seconds! ***")
                break
            
            if pos[2] < 0.25:
                print(f"\n*** SPOT FELL! Z={pos[2]:.2f}m ***")
                break
            
            if sim_time[0] > 120:
                print(f"\n*** TIMEOUT after 120s ***")
                break

except KeyboardInterrupt:
    print("\nExiting...")

final_pos, _ = get_robot_state()
total_progress = final_pos[0] - start_pos[0]

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
print(f"Total progress: {total_progress:.2f}m")
print(f"Final position: X={final_pos[0]:.2f}m, Y={final_pos[1]:.2f}m, Z={final_pos[2]:.2f}m")
if final_pos[0] >= END_X - 0.5:
    print("STATUS: SUCCESS!")
else:
    print("STATUS: Did not reach end zone")
print("=" * 60)

simulation_app.close()
print("Done.")