"""
LOCOMOTION PRE-TRAINING (BASELINE APPROACH)
============================================
Phase 1: Train Spot to walk stably using SpotFlatTerrainPolicy's built-in locomotion
Goal: Learn high-level command policies (forward speed, strafe, turn rate) rather than low-level joint control

This builds on the working Baseline_Environment.py approach where:
- SpotFlatTerrainPolicy handles all low-level balance and gait control
- RL policy learns to issue velocity commands: [vx, vy, omega]
- Robot uses built-in `spot.forward()` for stable locomotion

Reward Structure:
- Survival: +1 per second alive without falling
- Forward movement: +1.0 * forward_velocity (encourages forward walking)
- Stability bonus: +0.5 if moving without excessive tilt
- Fall penalty: Episode ends, reward goes to 0

Success Criteria:
- 10 consecutive iterations without any falls
- Saves locomotion_success.pt checkpoint for Stage 1 navigation training

Author: Cole (MS for Autonomy Project)
Date: March 2026
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import time
from datetime import datetime
import math

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Constants from Baseline_Environment.py
SPOT_START_X = 0.0
SPOT_START_Y = 0.0
SPOT_START_Z = 0.7
PHYSICS_DT = 1.0 / 500.0  # 500 Hz physics
RENDERING_DT = 10.0 / 500.0  # 50 Hz rendering
CONTROL_HZ = 20  # 20 Hz control frequency (same as Baseline)
CONTROL_DT = 1.0 / CONTROL_HZ
FALL_THRESHOLD = 0.25  # Height threshold for detecting falls

# Command ranges for spot.forward(step_size, command)
# command = [vx, vy, omega] where:
#   vx: forward/backward velocity (m/s), positive = forward
#   vy: left/right strafe velocity (m/s), positive = left
#   omega: turning rate (rad/s), positive = counter-clockwise
VX_MIN = -0.5  # Allow some backward but discourage it
VX_MAX = 2.0   # ~2 m/s forward (Spot's max is ~1.6 m/s, we give margin)
VY_MIN = -0.5  # Strafe right
VY_MAX = 0.5   # Strafe left
OMEGA_MIN = -1.5  # Turn right (rad/s)
OMEGA_MAX = 1.5   # Turn left (rad/s)


def log(msg, file_handle=None):
    """Log to console and file."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    if file_handle:
        file_handle.write(formatted + "\n")
        file_handle.flush()


class SimpleActorCritic(nn.Module):
    """
    Simple Actor-Critic network for locomotion command policy.
    Learns to output velocity commands for SpotFlatTerrainPolicy.
    """
    def __init__(self, obs_dim, action_dim, hidden_dims=(256, 256, 128)):
        super().__init__()
        
        # Actor network (policy)
        actor_layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            actor_layers.append(nn.Linear(prev_dim, hidden_dim))
            actor_layers.append(nn.Tanh())
            prev_dim = hidden_dim
        actor_layers.append(nn.Linear(prev_dim, action_dim))
        actor_layers.append(nn.Tanh())  # Actions in [-1, 1], scaled later
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic network (value function)
        critic_layers = []
        prev_dim = obs_dim
        for hidden_dim in (256, 128, 64):
            critic_layers.append(nn.Linear(prev_dim, hidden_dim))
            critic_layers.append(nn.Tanh())
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
    
    def forward(self, obs):
        return self.actor(obs), self.critic(obs)


class PPO:
    """Proximal Policy Optimization implementation."""
    def __init__(self, actor_critic, lr=3e-4, clip_param=0.2, value_coef=0.5, entropy_coef=0.01):
        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)
    
    def update(self, obs, actions, returns, advantages, old_log_probs):
        """Perform PPO update."""
        # Get current policy outputs
        action_preds, values = self.actor_critic(obs)
        
        # Calculate log probabilities (Gaussian with fixed std=0.3)
        std = 0.3
        dist = torch.distributions.Normal(action_preds, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Policy loss (PPO clipped objective)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = ((values.squeeze() - returns) ** 2).mean()
        
        # Entropy bonus (encourage exploration)
        entropy = dist.entropy().sum(dim=-1).mean()
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item()
        }


def get_simple_observation(spot):
    """
    Get simplified observation for locomotion (proprioceptive only).
    
    Observation space (18 dimensions):
    - Base position z-height (1)
    - Base linear velocity (3)
    - Base angular velocity (3)
    - Base orientation roll/pitch (2) - from quaternion
    - Joint positions (12 joints, normalized)
    
    Returns:
        np.ndarray: shape (21,) observation vector
    """
    obs = []
    
    # Get robot state
    base_pos, base_orientation = spot.robot.get_world_pose()
    base_vel = spot.robot.get_linear_velocity()
    base_angvel = spot.robot.get_angular_velocity()
    
    # Z-height (1)
    obs.append(base_pos[2])
    
    # Linear velocity (3)
    obs.extend([base_vel[0], base_vel[1], base_vel[2]])
    
    # Angular velocity (3)
    obs.extend([base_angvel[0], base_angvel[1], base_angvel[2]])
    
    # Orientation (roll and pitch from quaternion)
    # Quaternion is [w, x, y, z]
    w, x, y, z = base_orientation[0], base_orientation[1], base_orientation[2], base_orientation[3]
    # Roll (rotation about x-axis)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    # Pitch (rotation about y-axis)
    pitch = math.asin(2 * (w * y - z * x))
    obs.extend([roll, pitch])
    
    # Joint positions (12)
    joint_positions = spot.robot.get_joint_positions()
    if len(joint_positions) >= 12:
        # Normalize joint positions to roughly [-1, 1] range
        # Spot's joints typically range about ±1.5 rad
        normalized_joints = [jp / 1.5 for jp in joint_positions[:12]]
        obs.extend(normalized_joints)
    else:
        obs.extend([0.0] * 12)
    
    return np.array(obs, dtype=np.float32)


def scale_action(action):
    """
    Scale action from [-1, 1] to actual command ranges.
    
    Args:
        action: np.ndarray shape (3,) with values in [-1, 1]
    
    Returns:
        np.ndarray: shape (3,) scaled to [vx, vy, omega] ranges
    """
    vx = action[0] * (VX_MAX if action[0] > 0 else -VX_MIN) if action[0] != 0 else 0.0
    vy = action[1] * (VY_MAX if action[1] > 0 else -VY_MIN) if action[1] != 0 else 0.0
    omega = action[2] * (OMEGA_MAX if action[2] > 0 else -OMEGA_MIN) if action[2] != 0 else 0.0
    
    return np.array([vx, vy, omega], dtype=np.float32)


def calculate_reward(dt, fell, base_pos, base_vel, base_orientation):
    """
    Calculate reward for locomotion training.
    
    Rewards:
    - Survival: +1 per second alive
    - Forward velocity: +1.0 * vx (encourage forward movement)
    - Stability bonus: +0.5 if upright and not tilted excessively
    - Fall penalty: immediate termination (handled elsewhere)
    
    Args:
        dt: time step
        fell: bool, whether robot fell
        base_pos: position [x, y, z]
        base_vel: velocity [vx, vy, vz]
        base_orientation: quaternion [w, x, y, z]
    
    Returns:
        float: reward for this timestep
    """
    if fell:
        return -100.0  # Large penalty for falling
    
    reward = 0.0
    
    # Survival reward (+1 per second)
    reward += dt * (1.0 / dt)  # = 1.0 per second
    
    # Forward velocity reward (encourage forward walking)
    forward_vel = base_vel[0]
    reward += forward_vel * 1.0
    
    # Stability bonus (check if upright)
    w, x, y, z = base_orientation[0], base_orientation[1], base_orientation[2], base_orientation[3]
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - z * x))
    
    # Bonus if roll and pitch are small (robot is upright)
    if abs(roll) < 0.3 and abs(pitch) < 0.3:  # ~17 degrees
        reward += 0.5 * dt / CONTROL_DT
    
    return reward


def main():
    parser = argparse.ArgumentParser(description="Locomotion Pre-Training (Baseline Approach)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()
    
    # Setup
    checkpoint_dir = Path("checkpoints/locomotion_baseline")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(checkpoint_dir / "training_log.txt", "w")
    
    log("=" * 80, log_file)
    log("LOCOMOTION PRE-TRAINING (BASELINE APPROACH)", log_file)
    log("=" * 80, log_file)
    log(f"Iterations: {args.iterations}", log_file)
    log(f"Control frequency: {CONTROL_HZ} Hz", log_file)
    log(f"Headless: {args.headless}", log_file)
    log("", log_file)
    
    # Initialize Isaac Sim
    log("Initializing Isaac Sim...", log_file)
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": args.headless})
    
    # Import after SimulationApp init
    from omni.isaac.core import World
    from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
    import omni
    
    # Create world (physics at 500Hz, rendering at 50Hz like Baseline)
    log("Creating world...", log_file)
    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)
    world.scene.add_default_ground_plane()
    
    # Create Spot (following Baseline_Environment.py approach)
    log("Creating Spot robot...", log_file)
    spot = SpotFlatTerrainPolicy(
        prim_path="/World/Spot",
        name="Spot",
        position=np.array([SPOT_START_X, SPOT_START_Y, SPOT_START_Z])
    )
    
    # Reset world BEFORE initialization
    world.reset()
    log("World reset", log_file)
    
    # Initialize Spot and set default joint states
    spot.initialize()
    spot.robot.set_joints_default_state(spot.default_pos)
    log("Spot initialized with default joint positions", log_file)
    
    # Initialize PPO
    log("Initializing PPO...", log_file)
    obs_dim = 21  # Simple proprioceptive observation
    action_dim = 3  # [vx, vy, omega]
    
    actor_critic = SimpleActorCritic(obs_dim, action_dim)
    ppo = PPO(actor_critic, lr=5e-4)  # Slightly higher LR since we're learning high-level commands
    
    # Load checkpoint if specified
    start_iter = 0
    if args.checkpoint:
        log(f"Loading checkpoint: {args.checkpoint}", log_file)
        checkpoint = torch.load(args.checkpoint)
        actor_critic.load_state_dict(checkpoint['model_state_dict'])
        ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint.get('iteration', 0) + 1
        log(f"Resumed from iteration {start_iter}", log_file)
    
    log("Starting training...", log_file)
    log("", log_file)
    
    # Training loop
    best_mean_reward = -float('inf')
    steps_per_iteration = 400  # Collect 400 control steps (20 seconds at 20Hz)
    consecutive_no_fall_iterations = 0
    TARGET_CONSECUTIVE_SUCCESS = 10
    
    # Control step counter (for 20Hz control)
    last_control_time = 0.0
    
    for iteration in range(start_iter, args.iterations):
        iter_start_time = time.time()
        
        log(f"[ITER {iteration + 1}/{args.iterations}] Starting rollout collection...", log_file)
        
        # Collect rollout data
        observations = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs_list = []
        
        episode_rewards = []
        episode_lengths = []
        episode_falls = 0
        current_episode_reward = 0.0
        current_episode_steps = 0
        
        step_count = 0
        while step_count < steps_per_iteration:
            # Step physics simulation
            world.step(render=not args.headless)
            current_time = world.current_time
            
            # Control at 20Hz (only take actions every CONTROL_DT seconds)
            if current_time - last_control_time < CONTROL_DT:
                continue
            
            last_control_time = current_time
            
            # Get observation
            obs = get_simple_observation(spot)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            # Get action from policy
            with torch.no_grad():
                action_pred, value = actor_critic(obs_tensor)
                action_pred = action_pred.squeeze()
                value = value.squeeze()
                
                # Sample action with exploration noise
                std = 0.3
                dist = torch.distributions.Normal(action_pred, std)
                action_tensor = dist.sample()
                log_prob = dist.log_prob(action_tensor).sum()
                
                action_normalized = action_tensor.numpy()
                action_normalized = np.clip(action_normalized, -1, 1)
            
            # Scale action to command ranges
            command = scale_action(action_normalized)
            
            # Apply command using SpotFlatTerrainPolicy's forward method
            spot.forward(CONTROL_DT, command)
            
            # Get state after action
            base_pos, base_orientation = spot.robot.get_world_pose()
            base_vel = spot.robot.get_linear_velocity()
            fell = base_pos[2] < FALL_THRESHOLD
            
            # Calculate reward
            reward = calculate_reward(CONTROL_DT, fell, base_pos, base_vel, base_orientation)
            
            # Store transition
            observations.append(obs)
            actions.append(action_normalized)
            rewards.append(reward)
            dones.append(fell)
            values.append(value.item())
            log_probs_list.append(log_prob.item())
            
            current_episode_reward += reward
            current_episode_steps += 1
            step_count += 1
            
            # Handle episode end
            if fell or current_episode_steps >= 3000:  # Max 150 seconds (3000 steps at 20Hz)
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_steps)
                if fell:
                    episode_falls += 1
                
                # Reset Spot (following Baseline_Environment.py approach)
                spot.robot.set_world_pose(
                    position=np.array([SPOT_START_X, SPOT_START_Y, SPOT_START_Z]),
                    orientation=np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion [w, x, y, z]
                )
                spot.robot.set_joints_default_state(spot.default_pos)
                
                current_episode_reward = 0.0
                current_episode_steps = 0
                last_control_time = world.current_time  # Reset control timer
        
        # Convert to tensors
        observations = torch.FloatTensor(np.array(observations))
        actions = torch.FloatTensor(np.array(actions))
        rewards_tensor = torch.FloatTensor(rewards)
        values_tensor = torch.FloatTensor(values)
        old_log_probs = torch.FloatTensor(log_probs_list)
        
        # Calculate returns and advantages (GAE)
        returns = []
        advantages = []
        gae = 0
        gamma = 0.99
        lam = 0.95
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values_tensor[t + 1]
            
            delta = rewards_tensor[t] + gamma * next_value - values_tensor[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values_tensor[t])
        
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        update_stats = ppo.update(observations, actions, returns, advantages, old_log_probs)
        
        iter_time = time.time() - iter_start_time
        
        # Logging
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        mean_length = np.mean(episode_lengths) if episode_lengths else 0.0
        
        log("", log_file)
        log(f"Iteration {iteration + 1}/{args.iterations} completed in {iter_time:.2f}s", log_file)
        log(f"  Episodes: {len(episode_rewards)}", log_file)
        log(f"  Mean Episode Reward: {mean_reward:.3f}", log_file)
        log(f"  Mean Episode Length: {mean_length:.1f}", log_file)
        log(f"  Falls: {episode_falls}/{len(episode_rewards)}", log_file)
        log(f"  Entropy: {update_stats['entropy']:.4f}", log_file)
        log(f"  Policy Loss: {update_stats['policy_loss']:.4f}", log_file)
        log(f"  Value Loss: {update_stats['value_loss']:.4f}", log_file)
        
        # Check for early success (no falls in this iteration)
        if episode_falls == 0 and len(episode_rewards) > 0:
            consecutive_no_fall_iterations += 1
            log(f"  ✓ No falls this iteration! Consecutive: {consecutive_no_fall_iterations}/{TARGET_CONSECUTIVE_SUCCESS}", log_file)
            
            if consecutive_no_fall_iterations >= TARGET_CONSECUTIVE_SUCCESS:
                log("", log_file)
                log("=" * 80, log_file)
                log("🎉 LOCOMOTION TRAINING SUCCESS!", log_file)
                log(f"Spot completed {TARGET_CONSECUTIVE_SUCCESS} consecutive iterations without falling!", log_file)
                log(f"Training stopped early at iteration {iteration + 1}/{args.iterations}", log_file)
                log("=" * 80, log_file)
                
                # Save success checkpoint
                final_path = checkpoint_dir / "locomotion_success.pt"
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': actor_critic.state_dict(),
                    'optimizer_state_dict': ppo.optimizer.state_dict(),
                    'mean_reward': mean_reward,
                    'mean_length': mean_length,
                    'consecutive_success': consecutive_no_fall_iterations
                }, final_path)
                log(f"Success checkpoint saved: {final_path.name}", log_file)
                break
        else:
            if consecutive_no_fall_iterations > 0:
                log(f"  ✗ Falls detected. Resetting consecutive counter from {consecutive_no_fall_iterations} to 0", log_file)
            consecutive_no_fall_iterations = 0
        
        # Save checkpoint every 50 iterations
        if (iteration + 1) % 50 == 0:
            checkpoint_path = checkpoint_dir / f"model_{iteration + 1}.pt"
            torch.save({
                'iteration': iteration,
                'model_state_dict': actor_critic.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict(),
                'mean_reward': mean_reward,
                'mean_length': mean_length
            }, checkpoint_path)
            log(f"  Checkpoint saved: {checkpoint_path.name}", log_file)
        
        # Save best model
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_path = checkpoint_dir / "best_model.pt"
            torch.save({
                'iteration': iteration,
                'model_state_dict': actor_critic.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict(),
                'mean_reward': mean_reward,
                'mean_length': mean_length
            }, best_path)
            log(f"  * New best model saved! Reward: {mean_reward:.3f}", log_file)
    
    log("", log_file)
    log("=" * 80, log_file)
    if consecutive_no_fall_iterations >= TARGET_CONSECUTIVE_SUCCESS:
        log("TRAINING COMPLETE - EARLY SUCCESS!", log_file)
        log(f"Spot mastered locomotion after {iteration + 1} iterations", log_file)
    else:
        log("TRAINING COMPLETE!", log_file)
        log(f"Completed all {args.iterations} iterations", log_file)
    log(f"Best mean reward: {best_mean_reward:.3f}", log_file)
    log("=" * 80, log_file)
    
    log_file.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
