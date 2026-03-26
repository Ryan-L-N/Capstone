"""
Minimal RL Integration Demo
============================
Shows how to add RL observation/reward on top of the existing baseline environment.

This replaces the complex spot_rl_env.py with a minimal example that:
1. Uses the existing CircularWaypointEnv from RL_Development.py
2. Adds get_observations() method for RL policy
3. Adds calculate_reward() method
4. Demonstrates one episode with random actions

Usage:
    cd "C:\\Users\\user\\Desktop\\Capstone_vs_1.2\\Immersive-Modeling-and-Simulation-for-Autonomy\\Experiments\\Cole\\RL Folder"
    C:\\isaac-sim\\python.bat minimal_rl_demo.py --headless

Author: Cole
Date: February 2026
"""

import argparse
import math
import numpy as np
import sys

# Action space bounds for velocity commands [vx, vy, omega_z]
# These are module-level constants safe to define here
ACTION_VX_MIN = -0.5  # m/s (backward)
ACTION_VX_MAX = 1.5   # m/s (forward)
ACTION_VY_MIN = -0.3  # m/s (left)
ACTION_VY_MAX = 0.3   # m/s (right)  
ACTION_OMEGA_MIN = -1.0  # rad/s (CCW)
ACTION_OMEGA_MAX = 1.0   # rad/s (CW)

# Global reward weights (will be initialized when needed)
reward_weights = None

def _ensure_reward_weights():
    """Lazy initialization of reward weights."""
    global reward_weights
    if reward_weights is None:
        try:
            from training_config import get_default_config
            config = get_default_config()
            reward_weights = config.rewards
        except ImportError:
            # Default reward weights if training_config not available
            class RewardWeights:
                waypoint_reached = 100.0
                distance_reduction = 2.0
                forward_locomotion = 1.0
                lateral_penalty = 0.5
                backward_penalty = 1.0
                stability_reward = 0.2
                height_deviation = 1.0
                time_penalty = 0.01
            reward_weights = RewardWeights()
    return reward_weights


def get_rl_observation(env, previous_action=None) -> np.ndarray:
    """
    Construct full RL observation from environment state.
    
    Full observation space (98 dimensions):
    - Base linear velocity (3) - body frame
    - Base angular velocity (3) - body frame
    - Projected gravity (3) - body frame
    - Command (3) - velocity command [vx, vy, omega_z]
    - Joint positions (12) - relative to default
    - Joint velocities (12)
    - Previous actions (3)
    - Waypoint relative position (2)
    - Waypoint direction (2) - unit vector to waypoint
    - Waypoint distance (1)
    - Progress ratio (1)
    - Nearest obstacles (5 obstacles × 5 features = 25)
    - Foot contacts (4)
    - Height scan (24) - simplified terrain sensing
    
    Total: 3+3+3+3+12+12+3+2+2+1+1+25+4+24 = 98 dimensions
    """
    obs = []
    
    # === PROPRIOCEPTIVE (48 dims) ===
    # Get velocities in world frame
    lin_vel_I = env.spot.robot.get_linear_velocity()
    ang_vel_I = env.spot.robot.get_angular_velocity()
    spot_pos, spot_quat = env.spot.robot.get_world_pose()
    
    # Transform to body frame
    from isaacsim.core.utils.rotations import quat_to_rot_matrix
    R_IB = quat_to_rot_matrix(spot_quat)
    R_BI = R_IB.transpose()
    lin_vel_b = np.matmul(R_BI, lin_vel_I)
    ang_vel_b = np.matmul(R_BI, ang_vel_I)
    gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))
    
    # Base velocities (3+3=6)
    obs.extend(lin_vel_b.tolist())
    obs.extend(ang_vel_b.tolist())
    
    # Projected gravity (3)
    obs.extend(gravity_b.tolist())
    
    # Command placeholder (3) - will be actual command during training
    obs.extend([0.0, 0.0, 0.0])
    
    # Joint states (12+12=24)
    joint_pos = env.spot.robot.get_joint_positions()
    joint_vel = env.spot.robot.get_joint_velocities()
    default_pos = env.spot.default_pos
    obs.extend((joint_pos - default_pos).tolist())
    obs.extend(joint_vel.tolist())
    
    # Previous action (3)
    if previous_action is not None:
        obs.extend(previous_action.tolist())
    else:
        obs.extend([0.0, 0.0, 0.0])
    
    # === NAVIGATION (6 dims) ===
    # Waypoint info
    if env.current_waypoint_idx < len(env.waypoints):
        wp = env.waypoints[env.current_waypoint_idx]
        wx, wy = wp["pos"][0], wp["pos"][1]
        
        # Relative position to waypoint (in body frame for better learning)
        rel_x_world = wx - spot_pos[0]
        rel_y_world = wy - spot_pos[1]
        rel_vec_world = np.array([rel_x_world, rel_y_world, 0.0])
        rel_vec_body = np.matmul(R_BI, rel_vec_world)
        
        obs.extend([
            rel_vec_body[0] / 50.0,  # normalized x
            rel_vec_body[1] / 50.0,  # normalized y
        ])
        
        # Distance to waypoint (1)
        dist = math.sqrt(rel_x_world**2 + rel_y_world**2)
        
        # Waypoint direction (2) - unit vector pointing to waypoint in body frame
        # This gives the agent explicit directional information
        if dist > 1e-6:  # Avoid division by zero
            direction_body_x = rel_vec_body[0] / dist
            direction_body_y = rel_vec_body[1] / dist
        else:
            direction_body_x = 0.0
            direction_body_y = 0.0
        
        obs.extend([
            direction_body_x,  # unit direction x
            direction_body_y,  # unit direction y
        ])
        
        obs.append(dist / 50.0)  # normalized distance
    else:
        obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])  # 2 (rel pos) + 2 (direction) + 1 (dist)
    
    # Progress ratio (1)
    progress = env.waypoints_reached / max(len(env.waypoints), 1)
    obs.append(progress)
    
    # === OBSTACLE SENSING (25 dims) ===
    # Get 5 nearest obstacles with features
    obstacles = []
    for obstacle in env.obstacle_mgr.obstacles[:100]:  # Limit search
        obs_pos = obstacle["pos"]
        dx = obs_pos[0] - spot_pos[0]
        dy = obs_pos[1] - spot_pos[1]
        dist_sq = dx*dx + dy*dy
        
        # Only consider obstacles within 15m
        if dist_sq < 225.0:
            obstacles.append((dist_sq, obstacle, dx, dy))
    
    # Sort by distance squared
    obstacles.sort(key=lambda x: x[0])
    
    # Extract features for 5 nearest obstacles
    for i in range(5):
        if i < len(obstacles):
            dist_sq, obstacle, dx, dy = obstacles[i]
            dist = math.sqrt(dist_sq)
            
            # Transform to body frame
            rel_vec_world = np.array([dx, dy, 0.0])
            rel_vec_body = np.matmul(R_BI, rel_vec_world)
            
            obs.extend([
                rel_vec_body[0] / 15.0,  # relative x (normalized)
                rel_vec_body[1] / 15.0,  # relative y (normalized)
                dist / 15.0,             # distance (normalized)
                obstacle["mass"] / 32.7, # mass (normalized by heavy mass)
                1.0 if obstacle["weight_class"] != "heavy" else 0.0,  # movable flag
            ])
        else:
            # Padding for missing obstacles
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    
    # === CONTACT SENSING (4 dims) ===
    # Foot contacts (FL, FR, HL, HR)
    # For now, use heuristic: if height is low and velocity is low, foot is in contact
    # TODO: Replace with actual contact sensors when available
    foot_contacts = [0.0, 0.0, 0.0, 0.0]
    if spot_pos[2] < 0.8:  # Close to ground
        if np.linalg.norm(lin_vel_b) < 0.5:  # Moving slowly
            foot_contacts = [1.0, 1.0, 1.0, 1.0]  # All feet in contact
    obs.extend(foot_contacts)
    
    # === HEIGHT SCAN (24 dims) ===
    # Simplified terrain height scanning in a grid around robot
    # For flat terrain, this will be zeros; for complex terrain, ray casting needed
    # 4x6 grid: 4 radial distances × 6 angular directions
    height_scan = [0.0] * 24  # Placeholder for flat terrain
    obs.extend(height_scan)
   
    return np.array(obs, dtype=np.float32)


def apply_rl_action(env, action):
    """Apply RL action to Spot robot.
    
    Action is 3-dim velocity command [vx, vy, omega_z].
    SpotFlatTerrainPolicy converts this to stable joint control.
    
    Args:
        env: CircularWaypointEnv with .spot and .world
        action: np.ndarray shape (3,) - [vx, vy, omega_z] in normalized range [-1, 1]
    """
    # Denormalize action from [-1, 1] to actual velocity ranges
    vx = np.interp(action[0], [-1, 1], [ACTION_VX_MIN, ACTION_VX_MAX])
    vy = np.interp(action[1], [-1, 1], [ACTION_VY_MIN, ACTION_VY_MAX])
    omega_z = np.interp(action[2], [-1, 1], [ACTION_OMEGA_MIN, ACTION_OMEGA_MAX])
    
    command = np.array([vx, vy, omega_z], dtype=np.float32)
    dt = env.world.get_physics_dt()
    
    # Apply command through SpotFlatTerrainPolicy
    env.spot.forward(dt, command)


def calculate_score_based_reward(dt, waypoint_reached=False, prev_dist=None, current_dist=None) -> float:
    """
    Calculate RL reward for current step using score-based system with distance shaping.
    
    Score system (matching RL_Development.py):
    - Start at 300 points
    - Subtract 1 point per second (-dt per step)
    - Add 15 points per waypoint reached
    - If Spot falls, score goes to 0 (failure)
    - If score reaches 0, episode fails
    
    Distance shaping (for learning guidance):
    - Add 10.0 × (prev_distance - current_distance) each step
    - Provides very strong gradient toward waypoint without changing score semantics
    - This is reward shaping, not part of the tracked score
    
    Args:
        dt: Actual simulation timestep in seconds (e.g., 0.02 for 50Hz control)
        waypoint_reached: Whether waypoint was just reached this step
        prev_dist: Previous distance to waypoint (meters)
        current_dist: Current distance to waypoint (meters)
    
    Returns:
        Step reward (score change + distance shaping)
    """
    reward = 0.0
    
    # Time penalty: -1 point per second
    reward -= dt
    
    # Waypoint bonus: +15 points
    if waypoint_reached:
        reward += 15.0
    
    # Distance shaping: reward progress toward waypoint (10.0 weight for very strong gradient)
    if prev_dist is not None and current_dist is not None:
        distance_reduction = prev_dist - current_dist
        reward += 10.0 * distance_reduction
    
    return reward


def run_rl_episode(simulation_app, args):
    """Run one episode with RL observations and rewards."""
    
    # Import Isaac modules and baseline environment
    # These must be imported AFTER SimulationApp is created
    from RL_Development import CircularWaypointEnv
    
    print("\n" + "=" * 80, flush=True)
    print("CREATING RL ENVIRONMENT", flush=True)
    print("=" * 80, flush=True)
    sys.stdout.flush()
    
    # Import Isaac Sim classes
    print("[DEBUG] Importing from RL_Development...", flush=True)
    from RL_Development import (_ensure_isaac_imports, build_world, setup_spot_sensors,
                                PHYSICS_DT, RENDERING_DT, SPOT_START_X, SPOT_START_Y, SPOT_START_Z,
                                WAYPOINT_COUNT)
    print("[DEBUG] Calling _ensure_isaac_imports...", flush=True)
    _ensure_isaac_imports()
    print("[DEBUG] Importing World and SpotFlatTerrainPolicy...", flush=True)
    from RL_Development import omni, World, SpotFlatTerrainPolicy
    print("[DEBUG] Imports complete!", flush=True)
    
    # Create world (same as in RL_Development main())
    print("[DEBUG] Creating World...", flush=True)
    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)
    print("[DEBUG] Getting stage...", flush=True)
    stage = omni.usd.get_context().get_stage()
    print("[DEBUG] Building world geometry...", flush=True)
    
    # Build environment geometry
    build_world(world, stage)
    print("[DEBUG] World geometry built!", flush=True)
    
    # Create RNG
    print("[DEBUG] Creating RNG...", flush=True)
    rng = np.random.default_rng(42)
    print("[DEBUG] RNG created!", flush=True)
    
    # Reset world FIRST (required to initialize physics engine)
    print("[DEBUG] Resetting world...", flush=True)
    world.reset()
    print("[DEBUG] World reset complete!", flush=True)
    
    # Create Spot robot (after world reset)
    print("[DEBUG] Creating Spot robot...", flush=True)
    spot_prim_path = "/World/Spot"
    spot = SpotFlatTerrainPolicy(
        prim_path=spot_prim_path,
        name="Spot",
        position=np.array([SPOT_START_X, SPOT_START_Y, SPOT_START_Z])
    )
    print(f"[DEBUG] SpotFlatTerrainPolicy created at {spot_prim_path}", flush=True)
    
    # Initialize Spot
    print("[DEBUG] Initializing Spot...", flush=True)
    spot.initialize()
    print("[DEBUG] Setting joint defaults...", flush=True)
    spot.robot.set_joints_default_state(spot.default_pos)
    print("[DEBUG] Spot initialized!", flush=True)
    
    # Add full sensor suite
    print("[DEBUG] Adding sensors...", flush=True)
    setup_spot_sensors(spot_prim_path)
    print("[DEBUG] Sensors added!", flush=True)
    
    # Create baseline environment
    print("[DEBUG] Creating CircularWaypointEnv...", flush=True)
    env = CircularWaypointEnv(world, stage, rng)
    env.spot = spot
    
    print("[OK] Environment created", flush=True)
    
    # Reset environment to generate waypoints and obstacles
    print("[DEBUG] Resetting environment to generate waypoints...", flush=True)
    episode_num = 1
    env.reset(episode_num)
    
    print(f"[OK] Environment reset for episode {episode_num}", flush=True)
    print(f"  Waypoints: {len(env.waypoints)}", flush=True)
    print(f"  Obstacles: {len(env.obstacle_mgr.obstacles)}", flush=True)
    
    # Get initial observation
    print("\n" + "=" * 80, flush=True)
    print("EPISODE START", flush=True)
    print("=" * 80, flush=True)
    print("[DEBUG] Getting initial observation...", flush=True)
    
    # Initialize previous action tracker
    previous_action = np.zeros(3, dtype=np.float32)
    
    obs = get_rl_observation(env, previous_action)
    print(f"[OK] Initial observation shape: {obs.shape}", flush=True)
    print(f"  Observation dimension: {len(obs)}", flush=True)
    print(f"  First 5 values: {obs[0]:.3f},{obs[1]:.3f},{obs[2]:.3f},{obs[3]:.3f},{obs[4]:.3f}", flush=True)
    
    # Episode loop
    print("[DEBUG] Setting up episode loop...", flush=True)
    total_reward = 0.0
    step_count = 0
    prev_dist = None  # Track distance for reward shaping
    max_steps = 3000  # 60 seconds @ 50Hz
    
    print(f"[DEBUG] Starting episode loop (max {max_steps} steps)...", flush=True)
    
    try:
        for step in range(max_steps):
            if step == 0:
                print(f"[DEBUG] Entering first loop iteration (step {step})...", flush=True)
            
            step_count += 1
            
            # Generate RL action (in real training, this comes from policy network)
            # For now: simple random policy in normalized range [-1, 1]
            action = np.random.uniform(-1, 1, size=3).astype(np.float32)
            
            if step == 0:
                print(f"[DEBUG] Generated action (raw): {action}", flush=True)
            
            # Apply RL action to Spot (converts to velocity command and applies via policy)
            apply_rl_action(env, action)
            
            if step == 0:
                print(f"[DEBUG] RL action applied via spot.forward()", flush=True)
                print(f"[DEBUG] About to call env.world.step()...", flush=True)
            
            # Step physics simulation
            env.world.step(render=not args.headless)
            
            if step == 0:
                print(f"[DEBUG] world.step() completed!", flush=True)
                print(f"[DEBUG] Getting post-step observation...", flush=True)
            
            # Get new observation with previous action
            obs = get_rl_observation(env, previous_action)
            
            # Update previous action for next step
            previous_action = action.copy()
            
            if step == 0:
                print(f"[DEBUG] Observation computed! Shape: {obs.shape}", flush=True)
            
            # Calculate distance to current waypoint
            if step == 0:
                print(f"[DEBUG] Calculating waypoint distance...", flush=True)
            
            waypoint_reached = False
            if env.current_waypoint_idx < len(env.waypoints):
                wp = env.waypoints[env.current_waypoint_idx]
                spot_pos, _ = env.spot.robot.get_world_pose()
                current_dist = math.sqrt(
                    (wp["pos"][0] - spot_pos[0])**2 + 
                    (wp["pos"][1] - spot_pos[1])**2
                )
                if current_dist < 0.5:
                    waypoint_reached = True
                    env.current_waypoint_idx += 1
                    env.waypoints_reached += 1
            else:
                current_dist = 0.0
            
            if step == 0:
                print(f"[DEBUG] Distance calculated: {current_dist:.2f}m", flush=True)
                print(f"[DEBUG] Calculating score-based reward with distance shaping...", flush=True)
            
            # Calculate step reward (score-based: -1 per second, +15 per waypoint, + distance shaping)
            # RENDERING_DT is the control frequency (0.02 sec = 50 Hz)
            reward = calculate_score_based_reward(RENDERING_DT, waypoint_reached, prev_dist, current_dist)
            total_reward += reward
            prev_dist = current_dist  # Update for next step
            
            if step == 0:
                print(f"[DEBUG] Reward calculated: {reward:.3f}, total={total_reward:.3f}", flush=True)
            
            # Print progress every 100 steps
            if step % 100 == 0:
                spot_pos, _ = env.spot.robot.get_world_pose()
                current_score = 300.0 + total_reward  # Score = 300 + accumulated reward
                print(f"  Step {step}: Pos=({spot_pos[0]:.1f}, {spot_pos[1]:.1f}), "
                      f"WP={env.current_waypoint_idx}, "
                      f"Score={current_score:.1f}, "
                      f"Step Reward={reward:.3f}", flush=True)
            
            # Check fall (0.25m threshold from RL_Development.py)
            if spot_pos[2] < 0.25:
                print(f"\n[FAIL] Spot fell at step {step}")
                total_reward = -total_reward  # Zero out score on fall
                break
            
            # Check termination (score depleted)
            current_score = 300.0 + total_reward
            if current_score <= 0:
                print(f"\n[FAIL] Episode terminated at step {step}")
                print(f"  Reason: Score reached 0")
                print(f"  Waypoints reached: {env.waypoints_reached}")
                break
            
            if env.waypoints_reached >= WAYPOINT_COUNT:
                print(f"\n[SUCCESS] Episode completed at step {step}")
                print(f"  Reason: All waypoints reached!")
                print(f"  Waypoints reached: {env.waypoints_reached}")
                break
    
    except KeyboardInterrupt:
        print("\n\n⚠ Episode interrupted by user")
    
    print("\n" + "=" * 80)
    print("EPISODE RESULTS")
    print("=" * 80)
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward per step: {total_reward/step_count:.3f}")
    print(f"Waypoints reached: {env.waypoints_reached} / {len(env.waypoints)}")
    final_score = 300.0 + total_reward  # Score = 300 + accumulated rewards
    print(f"Final score: {final_score:.1f} (started at 300)")
    print(f"  Time penalty: ~{step_count * RENDERING_DT:.1f} points ({step_count} steps × {RENDERING_DT} sec/step)")
    print(f"  Waypoint bonus: {env.waypoints_reached * 15} points ({env.waypoints_reached} waypoints × 15)")
    print("=" * 80)
    
    # Cleanup
    print("\nCleaning up...")
    simulation_app.close()
    print("[OK] Done!")


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser(description="Minimal RL Demo")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    args = parser.parse_args()
    
    # Initialize Isaac Sim
    print("=" * 80)
    print("MINIMAL RL INTEGRATION DEMO")
    print("=" * 80)
    
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": args.headless})
    
    print("\n[DEBUG] Entering main block...", flush=True)
    try:
        print("[DEBUG] About to call run_rl_episode()", flush=True)
        run_rl_episode(simulation_app, args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", flush=True)
        simulation_app.close()
    except Exception as e:
        print(f"\n\n[ERROR] Exception caught: {e}", flush=True)
        import traceback
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
