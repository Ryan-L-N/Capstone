"""
Test Run 2 Policy in Simple Baseline Environment
=================================================
Test the trained Run 2 navigation policy in a simpler environment
similar to Cole/Baseline_Folder to see how it performs.

This tests:
- Policy generalization to different environment setup
- Navigation performance without the training curriculum overhead
- Real-world applicability

Usage:
    python test_run2_simple.py --checkpoint checkpoints/run_2_fixed_v5/best_model.pt
    python test_run2_simple.py --checkpoint checkpoints/run_2_fixed_v5/stage_5_complete.pt --episodes 10
    python test_run2_simple.py --checkpoint checkpoints/run_2_fixed_v5/stage_6_checkpoint.pt --obstacles --headless

Author: Cole (MS for Autonomy Project)
Date: March 11, 2026
"""

import argparse
import os
import math
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Parse args BEFORE Isaac Sim import
parser = argparse.ArgumentParser(description="Test Run 2 policy in simple environment")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained policy checkpoint")
parser.add_argument("--episodes", type=int, default=3, help="Number of test episodes")
parser.add_argument("--max-time", type=float, default=600.0, help="Max time per episode in seconds")
parser.add_argument("--obstacles", action="store_true", help="Add light obstacles (10% coverage)")
parser.add_argument("--waypoints", type=int, default=25, help="Number of waypoints")
args = parser.parse_args()

# Initialize Isaac Sim
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

# Now import Isaac modules
from omni.isaac.core import World
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
from omni.isaac.core.objects import VisualCuboid, DynamicCuboid
import omni
from pxr import Gf, UsdGeom, UsdPhysics, Sdf

# Import our modules
from navigation_policy import NavigationPolicy, scale_action


# ============================================================================
# Constants
# ============================================================================

ARENA_RADIUS = 25.0  # meters (50m diameter, same as baseline)

# Waypoint configuration
WAYPOINT_DISTANCE_FIRST = 20.0  # meters to first waypoint
WAYPOINT_DISTANCE_SUBSEQUENT = 40.0  # meters between waypoints
WAYPOINT_CAPTURE_RADIUS = 2.0  # meters

# Colors
COLOR_WAYPOINT = Gf.Vec3f(1.0, 0.8, 0.0)  # gold
COLOR_NEXT_WAYPOINT = Gf.Vec3f(0.0, 1.0, 0.0)  # bright green
COLOR_OBSTACLE = Gf.Vec3f(0.3, 0.3, 0.3)  # dark gray

# Physics
PHYSICS_DT = 1.0 / 500.0  # 500 Hz physics
RENDERING_DT = 10.0 / 500.0  # 50 Hz rendering (20 Hz control)


# ============================================================================
# Helper Functions
# ============================================================================

def log(msg: str):
    """Print with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def quaternion_to_yaw(quat: np.ndarray) -> float:
    """Convert quaternion [w,x,y,z] to yaw angle in radians."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-π, π]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def random_inside_arena(margin: float = 3.0) -> np.ndarray:
    """Sample random position inside arena with margin."""
    r_limit = ARENA_RADIUS - margin
    while True:
        x = np.random.uniform(-r_limit, r_limit)
        y = np.random.uniform(-r_limit, r_limit)
        if x**2 + y**2 < r_limit**2:
            return np.array([x, y])


def spawn_waypoints(start_pos: np.ndarray, count: int) -> list:
    """
    Spawn waypoints similar to Stage 5 curriculum.
    First waypoint at 20m, subsequent at 40m spacing.
    """
    waypoints = []
    
    # First waypoint at 20m from start
    angle = np.random.uniform(0, 2 * math.pi)
    first_wp = start_pos[:2] + WAYPOINT_DISTANCE_FIRST * np.array([math.cos(angle), math.sin(angle)])
    
    # Keep in bounds
    if np.linalg.norm(first_wp) > ARENA_RADIUS - 3:
        first_wp = first_wp / np.linalg.norm(first_wp) * (ARENA_RADIUS - 3)
    
    waypoints.append(first_wp)
    
    # Subsequent waypoints at 40m spacing
    for i in range(1, count):
        prev_wp = waypoints[-1]
        
        # Try to place 40m from previous
        for _ in range(100):  # max attempts
            angle = np.random.uniform(0, 2 * math.pi)
            new_wp = prev_wp + WAYPOINT_DISTANCE_SUBSEQUENT * np.array([math.cos(angle), math.sin(angle)])
            
            # Check if in bounds
            if np.linalg.norm(new_wp) < ARENA_RADIUS - 3:
                waypoints.append(new_wp)
                break
        else:
            # Couldn't place, just use random position
            waypoints.append(random_inside_arena())
    
    return waypoints


def spawn_obstacles(stage, count: int, waypoints: list, start_pos: np.ndarray):
    """Spawn light obstacles (10% coverage like Stage 6)."""
    arena_area = math.pi * ARENA_RADIUS**2
    target_area = arena_area * 0.10  # 10% coverage
    
    obstacle_size = 2.0  # 2x2x2 meter cubes
    obstacle_area = obstacle_size ** 2
    num_obstacles = int(target_area / obstacle_area)
    
    log(f"  Spawning {num_obstacles} obstacles ({obstacle_size}m cubes)")
    
    obstacle_paths = []
    
    for i in range(num_obstacles):
        # Find valid position (not too close to waypoints or start)
        for _ in range(100):
            pos = random_inside_arena(margin=5.0)
            
            # Check distance to start and waypoints
            if np.linalg.norm(pos - start_pos[:2]) < 5.0:
                continue
            
            too_close = False
            for wp in waypoints:
                if np.linalg.norm(pos - wp) < 5.0:
                    too_close = True
                    break
            
            if not too_close:
                break
        else:
            continue  # Couldn't find valid position
        
        # Create obstacle
        obs_path = f"/World/Obstacle_{i}"
        obstacle = DynamicCuboid(
            prim_path=obs_path,
            name=f"Obstacle_{i}",
            position=np.array([pos[0], pos[1], obstacle_size/2]),
            scale=np.array([obstacle_size, obstacle_size, obstacle_size]),
            color=np.array([COLOR_OBSTACLE[0], COLOR_OBSTACLE[1], COLOR_OBSTACLE[2]]),
            mass=50.0  # 50 kg (heavy enough to resist pushes)
        )
        obstacle_paths.append(obs_path)
    
    return obstacle_paths


def create_waypoint_marker(world, stage, position: np.ndarray, idx: int, is_next: bool = False):
    """Create visual waypoint marker (pole + flag)."""
    color = COLOR_NEXT_WAYPOINT if is_next else COLOR_WAYPOINT
    
    # Create pole (thin cylinder)
    pole_path = f"/World/Waypoint_{idx}_pole"
    pole = VisualCuboid(
        prim_path=pole_path,
        name=f"WP_{idx}_pole",
        position=np.array([position[0], position[1], 1.25]),
        scale=np.array([0.1, 0.1, 2.5]),
        color=np.array([0.9, 0.9, 0.9])
    )
    
    # Create flag (wide flat box)
    flag_path = f"/World/Waypoint_{idx}_flag"
    flag = VisualCuboid(
        prim_path=flag_path,
        name=f"WP_{idx}_flag",
        position=np.array([position[0], position[1], 2.2]),
        scale=np.array([0.7, 0.06, 0.4]),
        color=np.array([color[0], color[1], color[2]])
    )
    
    return pole_path, flag_path


def update_waypoint_color(stage, pole_path: str, flag_path: str, color: Gf.Vec3f):
    """Update waypoint marker color."""
    flag_prim = stage.GetPrimAtPath(flag_path)
    if flag_prim.IsValid():
        geom = UsdGeom.Gprim(flag_prim)
        geom.GetDisplayColorAttr().Set([color])


def get_observation(spot, waypoint_pos: np.ndarray, stage_id: int = 5) -> np.ndarray:
    """
    Construct observation vector for policy (32 dims).
    Simplified version - raycasts set to max range (no obstacle detection in this test).
    """
    # Robot state
    robot_pos, robot_ori = spot.robot.get_world_pose()
    robot_vel = spot.robot.get_linear_velocity()
    robot_ang_vel = spot.robot.get_angular_velocity()
    
    yaw = quaternion_to_yaw(robot_ori)
    
    # Base velocity (3)
    obs = [robot_vel[0], robot_vel[1], robot_ang_vel[2]]
    
    # Heading (2)
    obs.extend([math.sin(yaw), math.cos(yaw)])
    
    # Waypoint info (3)
    dx = waypoint_pos[0] - robot_pos[0]
    dy = waypoint_pos[1] - robot_pos[1]
    distance = math.sqrt(dx**2 + dy**2)
    obs.extend([dx, dy, distance])
    
    # Obstacle distances (16) - set to max range (no raycasting in this simple test)
    obs.extend([5.0] * 16)  # 5m max range
    
    # Stage encoding (8) - one-hot, use stage 5 or 6 depending on obstacles
    stage_encoding = [0.0] * 8
    stage_encoding[stage_id] = 1.0
    obs.extend(stage_encoding)
    
    return np.array(obs, dtype=np.float32)


# ============================================================================
# Main Test Loop
# ============================================================================

def main():
    log("=" * 80)
    log("RUN 2 POLICY TEST — SIMPLE BASELINE ENVIRONMENT")
    log("=" * 80)
    log(f"Checkpoint: {args.checkpoint}")
    log(f"Episodes: {args.episodes}")
    log(f"Waypoints: {args.waypoints}")
    log(f"Max time: {args.max_time}s")
    log(f"Obstacles: {'Yes (10% coverage)' if args.obstacles else 'No'}")
    log("")
    
    # Load trained policy
    log("Loading trained navigation policy...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = NavigationPolicy(obs_dim=32, action_dim=3).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'policy_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['policy_state_dict'])
        log(f"  ✓ Loaded from iteration {checkpoint.get('iteration', 'unknown')}")
        log(f"  ✓ Training stage: {checkpoint.get('current_stage', 'unknown')}")
    else:
        policy.load_state_dict(checkpoint)
    
    policy.eval()
    log("✓ Policy loaded successfully")
    log("")
    
    # Create world
    log("Creating Isaac Sim world...")
    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()
    world.reset()
    log("✓ World created")
    
    # Create Spot
    log("Creating Spot robot...")
    spot = SpotFlatTerrainPolicy(
        prim_path="/World/Spot",
        name="Spot",
        position=np.array([0.0, 0.0, 0.7])
    )
    spot.initialize()
    spot.robot.set_joints_default_state(spot.default_pos)
    log("✓ Spot initialized")
    log("")
    
    # Run test episodes
    results = []
    
    for episode_idx in range(args.episodes):
        log("=" * 80)
        log(f"EPISODE {episode_idx + 1}/{args.episodes}")
        log("=" * 80)
        
        # Reset Spot
        spot.robot.set_world_pose(position=np.array([0.0, 0.0, 0.7]))
        spot.robot.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
        spot.robot.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
        spot.robot.set_joints_default_state(spot.default_pos)
        
        # Spawn waypoints
        start_pos = np.array([0.0, 0.0, 0.0])
        waypoints = spawn_waypoints(start_pos, count=args.waypoints)
        log(f"✓ Spawned {len(waypoints)} waypoints")
        
        # Spawn obstacles if requested
        obstacle_paths = []
        if args.obstacles:
            obstacle_paths = spawn_obstacles(stage, count=20, waypoints=waypoints, start_pos=start_pos)
            log(f"✓ Spawned {len(obstacle_paths)} obstacles")
        
        # Create waypoint markers
        marker_paths = []
        for i, wp in enumerate(waypoints):
            pole_path, flag_path = create_waypoint_marker(world, stage, wp, i, is_next=(i == 0))
            marker_paths.append((pole_path, flag_path))
        
        # Episode variables
        current_wp_idx = 0
        waypoints_captured = 0
        episode_time = 0.0
        control_step_count = 0
        dt = RENDERING_DT  # Control at rendering rate (20 Hz)
        
        log("Starting episode...")
        log("")
        
        # Episode loop
        while episode_time < args.max_time and current_wp_idx < len(waypoints):
            # Step simulation
            world.step(render=not args.headless)
            episode_time += dt
            control_step_count += 1
            
            # Control at 20 Hz (every rendering step)
            if True:  # Could add control_step_count % N if want slower control
                # Get current waypoint
                target_wp = waypoints[current_wp_idx]
                
                # Get stage ID (5 for no obstacles, 6 for obstacles)
                stage_id = 6 if args.obstacles else 5
                
                # Build observation
                obs = get_observation(spot, target_wp, stage_id=stage_id)
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
                
                # Get action from policy (deterministic for testing)
                with torch.no_grad():
                    action, _, _ = policy.get_action(obs_tensor, deterministic=True)
                    action = action.cpu().numpy()[0]
                
                # Scale action to velocity ranges
                action_scaled = scale_action(
                    action,
                    vx_range=(-0.5, 2.0),
                    vy_range=(-0.5, 0.5),
                    omega_range=(-1.0, 1.0)
                )
                
                # Apply command to Spot using forward() API
                spot.forward(dt, [action_scaled[0], action_scaled[1], action_scaled[2]])
            
            # Check waypoint capture
            robot_pos, _ = spot.robot.get_world_pose()
            dist_to_wp = math.sqrt((robot_pos[0] - target_wp[0])**2 + (robot_pos[1] - target_wp[1])**2)
            
            if dist_to_wp < WAYPOINT_CAPTURE_RADIUS:
                waypoints_captured += 1
                
                # Update marker colors
                if current_wp_idx < len(marker_paths):
                    pole_path, flag_path = marker_paths[current_wp_idx]
                    update_waypoint_color(stage, pole_path, flag_path, Gf.Vec3f(0.5, 0.5, 0.5))  # gray
                
                current_wp_idx += 1
                
                if current_wp_idx < len(marker_paths):
                    pole_path, flag_path = marker_paths[current_wp_idx]
                    update_waypoint_color(stage, pole_path, flag_path, COLOR_NEXT_WAYPOINT)  # green
                
                log(f"  ✓ Waypoint {waypoints_captured}/{args.waypoints} captured! (t={episode_time:.1f}s, dist={dist_to_wp:.2f}m)")
            
            # Progress logging every 30 seconds
            if control_step_count % 600 == 0:  # 600 steps at 20Hz = 30s
                log(f"  [{episode_time:.0f}s] WP: {waypoints_captured}/{args.waypoints}, dist to next: {dist_to_wp:.1f}m")
        
        # Episode complete
        success = waypoints_captured == args.waypoints
        
        log("")
        log(f"Episode {episode_idx + 1} Complete:")
        log(f"  Waypoints captured: {waypoints_captured}/{args.waypoints}")
        log(f"  Success: {'YES' if success else 'NO'}")
        log(f"  Time: {episode_time:.1f}s")
        log(f"  Completion rate: {waypoints_captured / args.waypoints * 100:.1f}%")
        log("")
        
        results.append({
            'episode': episode_idx + 1,
            'waypoints': waypoints_captured,
            'success': success,
            'time': episode_time
        })
        
        # Clean up markers and obstacles
        for pole_path, flag_path in marker_paths:
            stage.RemovePrim(pole_path)
            stage.RemovePrim(flag_path)
        
        for obs_path in obstacle_paths:
            stage.RemovePrim(obs_path)
    
    # Final summary
    log("=" * 80)
    log("TEST COMPLETE — RESULTS SUMMARY")
    log("=" * 80)
    
    total_waypoints = sum(r['waypoints'] for r in results)
    success_count = sum(r['success'] for r in results)
    avg_waypoints = total_waypoints / len(results)
    success_rate = success_count / len(results) * 100
    avg_time = sum(r['time'] for r in results) / len(results)
    
    log(f"Episodes: {len(results)}")
    log(f"Total waypoints: {total_waypoints}/{args.waypoints * len(results)}")
    log(f"Average per episode: {avg_waypoints:.1f} waypoints")
    log(f"Success rate: {success_rate:.1f}%")
    log(f"Average time: {avg_time:.1f}s")
    log("")
    
    log("Performance Assessment:")
    if success_rate >= 80:
        log("  ✓ EXCELLENT — Policy performing well in new environment")
    elif success_rate >= 50:
        log("  ⚠ GOOD — Policy working but some failures")
    elif avg_waypoints >= 15:
        log("  ⚠ MODERATE — Policy navigates but struggles to complete")
    else:
        log("  ✗ POOR — Policy not generalizing well")
    
    log("")
    log(f"Individual episode results:")
    for r in results:
        status = "✓" if r['success'] else "✗"
        log(f"  Episode {r['episode']}: {status} {r['waypoints']}/{args.waypoints} waypoints in {r['time']:.1f}s")
    
    simulation_app.close()


if __name__ == "__main__":
    main()
