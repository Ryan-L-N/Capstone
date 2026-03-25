"""
Test Policy Transfer: Run 2 Navigation Policy on Unitree A1
============================================================
Test the trained Run 2 navigation policy on a different quadruped robot (Unitree A1)
in a simple environment similar to Cole/Baseline_Folder.

This tests sim-to-sim transfer and policy generalization.

Usage:
    python test_policy_transfer.py --checkpoint checkpoints/run_2_fixed_v5/best_model.pt
    python test_policy_transfer.py --checkpoint checkpoints/run_2_fixed_v5/stage_5_complete.pt --headless

Author: Cole (MS for Autonomy Project)  
Date: March 2026
"""

import argparse
import os
import math
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Parse args BEFORE Isaac Sim import
parser = argparse.ArgumentParser(description="Test navigation policy on different robot")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained policy checkpoint")
parser.add_argument("--robot", type=str, default="a1", choices=["a1", "anymal"], help="Robot to test (a1 or anymal)")
parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")
parser.add_argument("--max-steps", type=int, default=6000, help="Max steps per episode (6000 = 300s at 20Hz)")
args = parser.parse_args()

# Initialize Isaac Sim
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

# Now import Isaac modules
from omni.isaac.core import World
from omni.isaac.quadruped.robots import Unitree
from omni.isaac.core.objects import VisualCuboid, VisualSphere
from omni.isaac.core.prims import XFormPrim
import omni
from pxr import Gf, UsdGeom

# Import our modules
from navigation_policy import NavigationPolicy, scale_action


# ============================================================================
# Constants
# ============================================================================

ARENA_RADIUS = 25.0  # meters
WAYPOINT_DISTANCE_FIRST = 20.0  # meters to first waypoint
WAYPOINT_DISTANCE_SUBSEQUENT = 40.0  # meters between waypoints
WAYPOINT_COUNT = 25
WAYPOINT_CAPTURE_RADIUS = 2.0  # meters

# Robot configs (Unitree A1 vs ANYmal)
ROBOT_CONFIGS = {
    "a1": {
        "class": Unitree,
        "name": "Unitree_A1",
        "prim_path": "/World/A1",
        "usd_path": None,  # Uses default
    },
    # Note: ANYmal requires Isaac Lab, so we'll focus on A1 which works with omni.isaac.quadruped
}

# Colors
COLOR_WAYPOINT = Gf.Vec3f(1.0, 0.8, 0.0)  # gold
COLOR_NEXT_WAYPOINT = Gf.Vec3f(0.0, 1.0, 0.0)  # green


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
    """Sample random position inside arena."""
    r_limit = ARENA_RADIUS - margin
    while True:
        x = np.random.uniform(-r_limit, r_limit)
        y = np.random.uniform(-r_limit, r_limit)
        if x**2 + y**2 < r_limit**2:
            return np.array([x, y])


def spawn_waypoints(start_pos: np.ndarray, count: int = 25) -> list:
    """
    Spawn waypoints similar to navigation training curriculum.
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


def create_waypoint_marker(stage, position: np.ndarray, idx: int, is_next: bool = False):
    """Create visual waypoint marker in the scene."""
    color = COLOR_NEXT_WAYPOINT if is_next else COLOR_WAYPOINT
    
    # Create sphere marker
    marker_path = f"/World/Waypoint_{idx}"
    sphere = VisualSphere(
        prim_path=marker_path,
        name=f"Waypoint_{idx}",
        position=np.array([position[0], position[1], 1.5]),
        radius=0.5,
        color=np.array([color[0], color[1], color[2]])
    )
    
    return marker_path


def get_observation(robot, waypoint_pos: np.ndarray, stage_id: int = 5) -> np.ndarray:
    """
    Construct observation vector for policy (32 dims).
    Simplified version without obstacles (raycasts set to max range).
    """
    # Robot state
    robot_pos, robot_ori = robot.get_world_pose()
    robot_vel = robot.get_linear_velocity()
    robot_ang_vel = robot.get_angular_velocity()
    
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
    
    # Obstacle distances (16) - set to max range since no obstacles
    obs.extend([5.0] * 16)  # 5m max range
    
    # Stage encoding (8) - one-hot
    stage_encoding = [0.0] * 8
    stage_encoding[stage_id] = 1.0
    obs.extend(stage_encoding)
    
    return np.array(obs, dtype=np.float32)


# ============================================================================
# Main Test Loop
# ============================================================================

def main():
    log("=" * 80)
    log("POLICY TRANSFER TEST: Run 2 → Unitree A1")
    log("=" * 80)
    log(f"Checkpoint: {args.checkpoint}")
    log(f"Robot: {args.robot}")
    log(f"Episodes: {args.episodes}")
    log("")
    
    # Load trained policy
    log("Loading trained navigation policy...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = NavigationPolicy(obs_dim=32, action_dim=3).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'policy_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['policy_state_dict'])
        log(f"  Loaded from iteration {checkpoint.get('iteration', 'unknown')}")
        log(f"  Stage: {checkpoint.get('current_stage', 'unknown')}")
    else:
        policy.load_state_dict(checkpoint)
    
    policy.eval()
    log("✓ Policy loaded successfully")
    log("")
    
    # Create world
    log("Creating Isaac Sim world...")
    world = World(stage_units_in_meters=1.0, physics_dt=1/500, rendering_dt=10/500)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()
    world.reset()
    log("✓ World created")
    
    # Create robot
    log(f"Creating {args.robot.upper()} robot...")
    robot_config = ROBOT_CONFIGS[args.robot]
    
    if args.robot == "a1":
        # Create Unitree A1
        robot = Unitree(
            prim_path=robot_config["prim_path"],
            name=robot_config["name"],
            position=np.array([0.0, 0.0, 0.5])
        )
        robot.initialize()
    else:
        log(f"ERROR: Robot '{args.robot}' not yet implemented")
        simulation_app.close()
        return
    
    log("✓ Robot created")
    log("")
    
    # Run test episodes
    results = []
    
    for episode_idx in range(args.episodes):
        log("=" * 80)
        log(f"EPISODE {episode_idx + 1}/{args.episodes}")
        log("=" * 80)
        
        # Reset robot position
        robot.set_world_pose(position=np.array([0.0, 0.0, 0.5]))
        robot.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
        robot.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
        
        # Spawn waypoints
        start_pos = np.array([0.0, 0.0, 0.0])
        waypoints = spawn_waypoints(start_pos, count=WAYPOINT_COUNT)
        log(f"Spawned {len(waypoints)} waypoints")
        
        # Create visual markers
        marker_paths = []
        for i, wp in enumerate(waypoints):
            marker_path = create_waypoint_marker(stage, wp, i, is_next=(i == 0))
            marker_paths.append(marker_path)
        
        # Episode variables
        current_wp_idx = 0
        waypoints_captured = 0
        episode_steps = 0
        
        # Episode loop
        while episode_steps < args.max_steps and current_wp_idx < len(waypoints):
            # Step simulation
            world.step(render=not args.headless)
            episode_steps += 1
            
            # Get current waypoint
            target_wp = waypoints[current_wp_idx]
            
            # Build observation
            obs = get_observation(robot, target_wp, stage_id=5)
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
            
            # Get action from policy
            with torch.no_grad():
                action, _, _ = policy.get_action(obs_tensor, deterministic=True)
                action = action.cpu().numpy()[0]
            
            # Scale action to velocity ranges (same as training)
            action_scaled = scale_action(
                action,
                vx_range=(-0.5, 2.0),
                vy_range=(-0.5, 0.5),
                omega_range=(-1.0, 1.0)
            )
            
            # Apply command to robot (Unitree uses different API than SpotFlatTerrainPolicy)
            # For now, let's just set velocities directly
            robot.set_linear_velocity(np.array([action_scaled[0], action_scaled[1], 0.0]))
            robot.set_angular_velocity(np.array([0.0, 0.0, action_scaled[2]]))
            
            # Check waypoint capture
            robot_pos, _ = robot.get_world_pose()
            dist_to_wp = math.sqrt((robot_pos[0] - target_wp[0])**2 + (robot_pos[1] - target_wp[1])**2)
            
            if dist_to_wp < WAYPOINT_CAPTURE_RADIUS:
                waypoints_captured += 1
                current_wp_idx += 1
                
                # Hide captured marker, show next
                if current_wp_idx > 0:
                    # TODO: Update marker colors
                    pass
                
                log(f"  ✓ Waypoint {waypoints_captured}/{WAYPOINT_COUNT} captured! ({episode_steps} steps)")
            
            # Progress logging every 50 steps
            if episode_steps % 1000 == 0:
                log(f"  Step {episode_steps}: WP {waypoints_captured}/{WAYPOINT_COUNT}, dist={dist_to_wp:.1f}m")
        
        # Episode complete
        time_elapsed = episode_steps * (1/20)  # 20Hz control
        success = waypoints_captured == WAYPOINT_COUNT
        
        log("")
        log(f"Episode {episode_idx + 1} Results:")
        log(f"  Waypoints captured: {waypoints_captured}/{WAYPOINT_COUNT}")
        log(f"  Success: {success}")
        log(f"  Time: {time_elapsed:.1f}s ({episode_steps} steps)")
        log(f"  Completion rate: {waypoints_captured / WAYPOINT_COUNT * 100:.1f}%")
        log("")
        
        results.append({
            'episode': episode_idx + 1,
            'waypoints': waypoints_captured,
            'success': success,
            'steps': episode_steps,
            'time': time_elapsed
        })
        
        # Clean up markers
        for marker_path in marker_paths:
            stage.RemovePrim(marker_path)
    
    # Final summary
    log("=" * 80)
    log("TRANSFER TEST COMPLETE")
    log("=" * 80)
    
    total_waypoints = sum(r['waypoints'] for r in results)
    success_count = sum(r['success'] for r in results)
    avg_waypoints = total_waypoints / len(results)
    success_rate = success_count / len(results) * 100
    
    log(f"Episodes: {len(results)}")
    log(f"Total waypoints captured: {total_waypoints}/{WAYPOINT_COUNT * len(results)}")
    log(f"Average waypoints per episode: {avg_waypoints:.1f}")
    log(f"Success rate: {success_rate:.1f}%")
    log("")
    
    log("Policy Transfer Assessment:")
    if success_rate >= 80:
        log("  ✓ EXCELLENT transfer - policy generalizes well to new robot")
    elif success_rate >= 50:
        log("  ⚠ MODERATE transfer - policy partially transfers but needs tuning")
    elif avg_waypoints >= 10:
        log("  ⚠ POOR transfer - basic navigation works but struggles to complete")
    else:
        log("  ✗ FAILED transfer - policy does not generalize to new robot")
    
    log("")
    log("Note: The Unitree A1 API doesn't support the SpotFlatTerrainPolicy")
    log("locomotion controller, so velocity commands are applied directly.")
    log("For full testing, consider using a robot with compatible control APIs.")
    
    simulation_app.close()


if __name__ == "__main__":
    main()
