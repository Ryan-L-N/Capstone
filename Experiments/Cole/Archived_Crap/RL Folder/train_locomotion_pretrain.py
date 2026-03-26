"""
LOCOMOTION PRE-TRAINING
=======================
Phase 1: Train Spot to walk stably without falling
Goal: Learn basic locomotion before waypoint navigation

Reward Structure:
- Survival: +1 per second alive
- Forward movement: +0.5 * forward_velocity (encourages walking)
- Fall penalty: Episode ends, reward goes to 0
- Keep balanced: Small penalty for excessive lateral velocity

After training, checkpoint can be loaded for waypoint navigation training.
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import time
import random
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training_config import TrainingConfig

# Constants
SPOT_START_X = 0.0
SPOT_START_Y = 0.0
SPOT_START_Z = 0.72
CONTROL_DT = 0.02  # 50Hz control
FALL_THRESHOLD = 0.25  # Height threshold for detecting falls

# Action ranges (same as main training)
ACTION_VX_MIN = -0.5
ACTION_VX_MAX = 1.5
ACTION_VY_MIN = -0.3
ACTION_VY_MAX = 0.3
ACTION_OMEGA_MIN = -1.0
ACTION_OMEGA_MAX = 1.0


def log(msg, file_handle=None):
    """Log to console and file."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    if file_handle:
        file_handle.write(formatted + "\n")
        file_handle.flush()


class ActorCritic(nn.Module):
    """Simple Actor-Critic network for locomotion."""
    def __init__(self, obs_dim, action_dim, hidden_dims=(512, 256, 128)):
        super().__init__()
        
        # Actor network (policy)
        actor_layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            actor_layers.append(nn.Linear(prev_dim, hidden_dim))
            actor_layers.append(nn.Tanh())
            prev_dim = hidden_dim
        actor_layers.append(nn.Linear(prev_dim, action_dim))
        actor_layers.append(nn.Tanh())  # Actions in [-1, 1]
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic network (value function)
        critic_layers = []
        prev_dim = obs_dim
        for hidden_dim in (256, 256, 128):
            critic_layers.append(nn.Linear(prev_dim, hidden_dim))
            critic_layers.append(nn.Tanh())
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
    
    def forward(self, obs):
        return self.actor(obs), self.critic(obs)


class PPO:
    """Simplified PPO implementation."""
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
        
        # Calculate log probabilities (assuming Gaussian with fixed std=0.5)
        std = 0.5
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


def get_locomotion_observation(spot, prev_action):
    """
    Get observation for locomotion training (no waypoint info).
    
    Observation space (67 dimensions):
    - Base linear velocity (3)
    - Base angular velocity (3)
    - Projected gravity (3)
    - Commands from previous action (3)
    - Joint positions (12)
    - Joint velocities (12)
    - Previous actions (3)
    - Foot contacts (4)
    - Height scan (24) - not used for flat terrain but kept for compatibility
    
    Returns:
        np.ndarray: shape (67,) observation vector
    """
    obs = []
    
    # Get robot state
    base_vel = spot.robot.get_linear_velocity()
    base_angvel = spot.robot.get_angular_velocity()
    base_orientation = spot.robot.get_world_pose()[1]  # Quaternion
    
    # Linear velocity (3)
    obs.extend([base_vel[0], base_vel[1], base_vel[2]])
    
    # Angular velocity (3)
    obs.extend([base_angvel[0], base_angvel[1], base_angvel[2]])
    
    # Projected gravity (3)
    from scipy.spatial.transform import Rotation
    quat_xyzw = [base_orientation[1], base_orientation[2], base_orientation[3], base_orientation[0]]
    rotation = Rotation.from_quat(quat_xyzw)
    gravity_world = np.array([0, 0, -1])
    gravity_body = rotation.inv().apply(gravity_world)
    obs.extend([gravity_body[0], gravity_body[1], gravity_body[2]])
    
    # Commands (previous action) (3)
    obs.extend([prev_action[0], prev_action[1], prev_action[2]])
    
    # Joint positions (12)
    joint_positions = spot.robot.get_joint_positions()
    obs.extend(joint_positions[:12] if len(joint_positions) >= 12 else [0.0] * 12)
    
    # Joint velocities (12)
    joint_velocities = spot.robot.get_joint_velocities()
    obs.extend(joint_velocities[:12] if len(joint_velocities) >= 12 else [0.0] * 12)
    
    # Previous actions (3)
    obs.extend([prev_action[0], prev_action[1], prev_action[2]])
    
    # Foot contacts (4) - FL, FR, RL, RR
    # Use heuristic: if base is low and velocity is low, all feet are in contact
    foot_contacts = [0.0, 0.0, 0.0, 0.0]
    if base_vel[2] < 0.1 and abs(base_angvel[0]) < 0.5 and abs(base_angvel[1]) < 0.5:
        # Robot is relatively stable, assume all feet in contact
        foot_contacts = [1.0, 1.0, 1.0, 1.0]
    obs.extend(foot_contacts)
    
    # Height scan (24) - zeros for flat terrain
    obs.extend([0.0] * 24)
    
    return np.array(obs, dtype=np.float32)


def apply_action(spot, world, action):
    """Apply action to Spot (same as main training)."""
    vx = np.interp(action[0], [-1, 1], [ACTION_VX_MIN, ACTION_VX_MAX])
    vy = np.interp(action[1], [-1, 1], [ACTION_VY_MIN, ACTION_VY_MAX])
    omega_z = np.interp(action[2], [-1, 1], [ACTION_OMEGA_MIN, ACTION_OMEGA_MAX])
    
    command = np.array([vx, vy, omega_z], dtype=np.float32)
    dt = world.get_physics_dt()
    spot.forward(dt, command)


def calculate_locomotion_reward(dt, fell, forward_velocity, lateral_velocity):
    """
    Calculate reward for locomotion pre-training.
    
    Args:
        dt: Time step
        fell: Whether robot fell
        forward_velocity: Forward velocity (vx)
        lateral_velocity: Lateral velocity (vy)
    
    Returns:
        float: Reward value
    """
    if fell:
        return -100.0  # Large penalty for falling
    
    # Survival reward
    reward = 1.0  # +1 per second = +0.02 per step
    
    # Encourage forward movement
    if forward_velocity > 0:
        reward += 0.5 * forward_velocity  # Bonus for moving forward
    
    # Small penalty for excessive lateral movement (keep balanced)
    reward -= 0.1 * abs(lateral_velocity)
    
    return reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Load config
    config = TrainingConfig()
    
    # Setup
    checkpoint_dir = Path("checkpoints/locomotion_pretrain")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(checkpoint_dir / "training_log.txt", "w")
    
    log("=" * 80, log_file)
    log("LOCOMOTION PRE-TRAINING", log_file)
    log("=" * 80, log_file)
    log(f"Iterations: {args.iterations}", log_file)
    log(f"Learning rate: {config.ppo.learning_rate}", log_file)
    log(f"Headless: {args.headless}", log_file)
    log("", log_file)
    
    # Initialize Isaac Sim
    log("Initializing Isaac Sim...", log_file)
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": args.headless})
    
    # Import after SimulationApp init
    from RL_Development import (_ensure_isaac_imports, SPOT_START_X, SPOT_START_Y, SPOT_START_Z)
    _ensure_isaac_imports()
    from RL_Development import omni, World, SpotFlatTerrainPolicy
    
    from omni.isaac.core.utils.stage import get_current_stage
    stage = get_current_stage()
    
    # Setup sensors
    def setup_spot_sensors(prim_path):
        from omni.isaac.sensor import ContactSensor
        feet = ["front_left", "front_right", "rear_left", "rear_right"]
        for foot in feet:
            ContactSensor(
                prim_path=f"{prim_path}/{foot}_foot",
                name=f"{foot}_contact",
                min_threshold=0,
                max_threshold=10000000
            )
    
    # Create world
    log("Creating world...", log_file)
    world = World(stage_units_in_meters=1.0, physics_dt=CONTROL_DT, rendering_dt=CONTROL_DT)
    world.scene.add_default_ground_plane()
    
    # Reset world
    world.reset()
    
    # Create Spot
    log("Creating Spot robot...", log_file)
    spot = SpotFlatTerrainPolicy(
        prim_path="/World/Spot",
        name="Spot",
        position=np.array([SPOT_START_X, SPOT_START_Y, SPOT_START_Z])
    )
    spot.initialize()
    spot.robot.set_joints_default_state(spot.default_pos)
    setup_spot_sensors("/World/Spot")
    
    # Initialize PPO
    log("Initializing PPO...", log_file)
    obs_dim = 67  # Locomotion observation space (proprioceptive + height scan)
    action_dim = 3
    
    actor_critic = ActorCritic(obs_dim, action_dim)
    ppo = PPO(actor_critic, lr=config.ppo.learning_rate)
    
    # Load checkpoint if specified
    start_iter = 0
    if args.checkpoint:
        log(f"Loading checkpoint: {args.checkpoint}", log_file)
        checkpoint = torch.load(args.checkpoint)
        actor_critic.actor.load_state_dict(checkpoint['actor_state_dict'])
        actor_critic.critic.load_state_dict(checkpoint['critic_state_dict'])
        ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint.get('iteration', 0) + 1
        log(f"Resumed from iteration {start_iter}", log_file)
    
    log("Starting training...", log_file)
    log("", log_file)
    
    # Training loop
    best_mean_reward = -float('inf')
    steps_per_iteration = 600  # Steps to collect before update
    consecutive_no_fall_iterations = 0  # Track consecutive iterations with no falls
    TARGET_CONSECUTIVE_SUCCESS = 10  # Need 10 consecutive iterations without falls
    
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
        
        prev_action = np.zeros(3, dtype=np.float32)
        
        for step in range(steps_per_iteration):
            # Get observation
            obs = get_locomotion_observation(spot, prev_action)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            # Get action from policy
            with torch.no_grad():
                action_pred, value = actor_critic(obs_tensor)
                action_pred = action_pred.squeeze()
                value = value.squeeze()
                
                # Sample action with exploration noise
                std = 0.5
                dist = torch.distributions.Normal(action_pred, std)
                action_tensor = dist.sample()
                log_prob = dist.log_prob(action_tensor).sum()
                
                action = action_tensor.numpy()
                action = np.clip(action, -1, 1)
            
            # Apply action
            apply_action(spot, world, action)
            world.step(render=not args.headless)
            
            # Get next state
            spot_pos, _ = spot.robot.get_world_pose()
            base_vel = spot.robot.get_linear_velocity()
            fell = spot_pos[2] < FALL_THRESHOLD
            
            # Calculate reward
            reward = calculate_locomotion_reward(CONTROL_DT, fell, base_vel[0], base_vel[1])
            
            # Store transition
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(fell)
            values.append(value.item())
            log_probs_list.append(log_prob.item())
            
            current_episode_reward += reward
            current_episode_steps += 1
            prev_action = action
            
            # Handle episode end
            if fell or current_episode_steps >= 3000:  # Max 60 seconds
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_steps)
                if fell:
                    episode_falls += 1
                
                # Reset
                spot.robot.set_world_pose(np.array([SPOT_START_X, SPOT_START_Y, SPOT_START_Z]), np.array([1, 0, 0, 0]))
                spot.robot.set_linear_velocity(np.zeros(3))
                spot.robot.set_angular_velocity(np.zeros(3))
                spot.robot.set_joint_positions(spot.default_pos)
                spot.robot.set_joint_velocities(np.zeros(12))
                
                current_episode_reward = 0.0
                current_episode_steps = 0
                prev_action = np.zeros(3, dtype=np.float32)
        
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
                
                # Save final checkpoint
                final_path = checkpoint_dir / "locomotion_success.pt"
                torch.save({
                    'iteration': iteration,
                    'actor_state_dict': actor_critic.actor.state_dict(),
                    'critic_state_dict': actor_critic.critic.state_dict(),
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
                'actor_state_dict': actor_critic.actor.state_dict(),
                'critic_state_dict': actor_critic.critic.state_dict(),
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
                'actor_state_dict': actor_critic.actor.state_dict(),
                'critic_state_dict': actor_critic.critic.state_dict(),
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
    log(f"Final checkpoint: checkpoints/locomotion_pretrain/model_{min(iteration + 1, args.iterations)}.pt", log_file)
    log(f"Best checkpoint: checkpoints/locomotion_pretrain/best_model.pt", log_file)
    log("=" * 80, log_file)
    
    log_file.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
