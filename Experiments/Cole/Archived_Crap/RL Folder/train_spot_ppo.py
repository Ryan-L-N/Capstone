"""
Spot RL Training with PPO
==========================
Train Boston Dynamics Spot for obstacle-aware navigation using PPO.

This script implements:
- Actor-Critic neural networks
- PPO algorithm with GAE
- Episode rollout collection
- Checkpointing and Tensorboard logging

Usage:
    cd "C:\\Users\\user\\Desktop\\Capstone_vs_1.2\\Immersive-Modeling-and-Simulation-for-Autonomy\\Experiments\\Cole\\RL Folder"
    C:\\isaac-sim\\python.bat train_spot_ppo.py --iterations 1000 --headless

Author: Cole
Date: February 2026
"""

import argparse
import math
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path

# Parse args BEFORE Isaac Sim
parser = argparse.ArgumentParser(description="Train Spot with PPO")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--iterations", type=int, default=5000, help="Training iterations")
parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
parser.add_argument("--eval-only", action="store_true", help="Evaluation mode only")
parser.add_argument("--config", type=str, default="baseline", 
                   help="Hyperparameter config name (from hyperparam_configs.py)")
args, unknown = parser.parse_known_args()  # Use parse_known_args to ignore Isaac Sim args

# Initialize Isaac Sim
print("=" * 80)
print("SPOT PPO TRAINING")
print("=" * 80)

# Build Isaac Sim config from args
sim_config = {"headless": args.headless}

from isaacsim import SimulationApp
simulation_app = SimulationApp(sim_config)

# Import training config
from training_config import get_default_config
import hyperparam_configs

# Import RL components
from minimal_rl_demo import get_rl_observation, apply_rl_action, calculate_score_based_reward

# Get configuration from hyperparameter configs
print(f"[CONFIG] Loading configuration: {args.config}")
try:
    config = hyperparam_configs.get_config(args.config)
    print(f"[CONFIG] Loaded hyperparameter profile: {config.experiment_name}")
except ValueError as e:
    print(f"[WARNING] {e}")
    print(f"[CONFIG] Falling back to default configuration")
    config = get_default_config()

# Override iterations if specified
if args.iterations:
    config.ppo.num_learning_iterations = args.iterations

print(f"[CONFIG] Experiment: {config.experiment_name}")
print(f"[CONFIG] Training iterations: {config.ppo.num_learning_iterations}")
print(f"[CONFIG] Steps per iteration: {config.ppo.num_steps_per_env}")
print(f"[CONFIG] Max episode length: {config.ppo.max_episode_length}")
print(f"[CONFIG] Learning rate: {config.ppo.learning_rate}")
print(f"[CONFIG] PPO clip param: {config.ppo.clip_param}")
print(f"[CONFIG] Entropy coef: {config.ppo.entropy_coef}")


# =============================================================================
# ACTOR-CRITIC NETWORKS
# =============================================================================

class Actor(nn.Module):
    """Policy network (actor) that outputs actions."""
    
    def __init__(self, obs_dim, action_dim, hidden_dims=(512, 256, 128)):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())  # Match locomotion pre-training
            prev_dim = hidden_dim
        
        # Output layer (mean of action distribution)
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())  # Actions in [-1, 1]
        
        self.network = nn.Sequential(*layers)
        
        # Log standard deviation (learnable parameter for stochastic policy)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs):
        """Forward pass returns action mean."""
        return self.network(obs)
    
    def get_distribution(self, obs):
        """Get action distribution for sampling."""
        mean = self.forward(obs)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mean, std)
    
    def get_action(self, obs, deterministic=False):
        """Sample action from policy."""
        if deterministic:
            return self.forward(obs), None
        
        dist = self.get_distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate_actions(self, obs, actions):
        """Evaluate log probabilities and entropy for given actions."""
        dist = self.get_distribution(obs)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, entropy


class Critic(nn.Module):
    """Value network (critic) that estimates state values."""
    
    def __init__(self, obs_dim, hidden_dims=(256, 256, 128)):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())  # Match locomotion pre-training
            prev_dim = hidden_dim
        
        # Output single value
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, obs):
        """Forward pass returns estimated value."""
        return self.network(obs).squeeze(-1)


class ActorCritic(nn.Module):
    """Combined actor-critic network."""
    
    def __init__(self, obs_dim, action_dim, actor_hidden=(256, 256, 128), 
                 critic_hidden=(256, 256, 128)):
        super().__init__()
        
        self.actor = Actor(obs_dim, action_dim, actor_hidden)
        self.critic = Critic(obs_dim, critic_hidden)
        
    def forward(self, obs):
        """Forward pass for both actor and critic."""
        action_mean = self.actor(obs)
        value = self.critic(obs)
        return action_mean, value


# =============================================================================
# PPO ALGORITHM
# =============================================================================

class PPO:
    """Proximal Policy Optimization algorithm."""
    
    def __init__(self, actor_critic, config):
        self.actor_critic = actor_critic
        self.config = config.ppo
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=self.config.learning_rate
        )
        
        # Statistics
        self.mean_value_loss = 0.0
        self.mean_policy_loss = 0.0
        self.mean_entropy = 0.0
        
    def update(self, rollouts):
        """Perform PPO update using collected rollouts."""
        # Get rollout data
        obs = rollouts['observations']
        actions = rollouts['actions']
        old_log_probs = rollouts['log_probs']
        advantages = rollouts['advantages']
        returns = rollouts['returns']
        
        # Normalize advantages
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Reshape for mini-batches
        batch_size = obs.shape[0]
        mini_batch_size = batch_size // self.config.num_mini_batches
        
        value_losses = []
        policy_losses = []
        entropies = []
        
        # PPO epochs
        for epoch in range(self.config.ppo_epoch):
            # Shuffle indices
            indices = torch.randperm(batch_size)
            
            # Mini-batches
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_indices = indices[start:end]
                
                mb_obs = obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # Evaluate current policy
                log_probs, entropy = self.actor_critic.actor.evaluate_actions(mb_obs, mb_actions)
                values = self.actor_critic.critic(mb_obs)
                
                # Policy loss (PPO clip objective)
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_param, 
                                   1.0 + self.config.clip_param) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (MSE)
                value_loss = nn.MSELoss()(values, mb_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.config.value_loss_coef * value_loss +
                       self.config.entropy_coef * entropy_loss)
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 
                                        self.config.max_grad_norm)
                self.optimizer.step()
                
                # Track losses
                value_losses.append(value_loss.item())
                policy_losses.append(policy_loss.item())
                entropies.append(entropy.mean().item())
        
        # Update statistics
        self.mean_value_loss = np.mean(value_losses)
        self.mean_policy_loss = np.mean(policy_losses)
        self.mean_entropy = np.mean(entropies)
        
        return {
            'value_loss': self.mean_value_loss,
            'policy_loss': self.mean_policy_loss,
            'entropy': self.mean_entropy
        }


# =============================================================================
# ROLLOUT STORAGE
# =============================================================================

class RolloutStorage:
    """Store rollout data for PPO updates."""
    
    def __init__(self, num_steps, obs_dim, action_dim):
        self.num_steps = num_steps
        
        # Storage
        self.observations = torch.zeros(num_steps, obs_dim)
        self.actions = torch.zeros(num_steps, action_dim)
        self.rewards = torch.zeros(num_steps)
        self.log_probs = torch.zeros(num_steps)
        self.values = torch.zeros(num_steps)
        self.dones = torch.zeros(num_steps)
        
        self.step = 0
        
    def add(self, obs, action, reward, log_prob, value, done):
        """Add transition to storage."""
        self.observations[self.step] = obs
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.log_probs[self.step] = log_prob
        self.values[self.step] = value
        self.dones[self.step] = float(done)
        
        self.step += 1
        
    def compute_returns_and_advantages(self, last_value, gamma=0.99, lam=0.95):
        """Compute GAE advantages and returns."""
        advantages = torch.zeros_like(self.rewards)
        last_gae = 0
        
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            
            # TD error
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            
            # GAE
            advantages[t] = last_gae = delta + gamma * lam * (1 - self.dones[t]) * last_gae
        
        # Returns = advantages + values
        returns = advantages + self.values
        
        return returns, advantages
    
    def get_data(self, returns, advantages):
        """Get all stored data as dictionary."""
        return {
            'observations': self.observations,
            'actions': self.actions,
            'log_probs': self.log_probs,
            'advantages': advantages,
            'returns': returns
        }
    
    def reset(self):
        """Reset storage for next iteration."""
        self.step = 0


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train():
    """Main training loop."""
    
    print("\n" + "=" * 80)
    print("INITIALIZING TRAINING")
    print("=" * 80)
    
    # Import Isaac Sim classes (lazy import)
    from RL_Development import (_ensure_isaac_imports, build_world, setup_spot_sensors,
                                CircularWaypointEnv, PHYSICS_DT, RENDERING_DT, 
                                SPOT_START_X, SPOT_START_Y, SPOT_START_Z)
    _ensure_isaac_imports()
    from RL_Development import omni, World, SpotFlatTerrainPolicy
    
    # Create world
    print("[INFO] Creating world...")
    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)
    stage = omni.usd.get_context().get_stage()
    build_world(world, stage)
    
    # Each world.step() advances by RENDERING_DT (control frequency)
    # Use this for time-based rewards
    CONTROL_DT = RENDERING_DT  # 0.02 seconds per step (50 Hz)
    
    # Create RNG
    rng = np.random.default_rng(config.seed)
    
    # Reset world
    world.reset()
    
    # Create Spot
    print("[INFO] Creating Spot robot...")
    spot = SpotFlatTerrainPolicy(
        prim_path="/World/Spot",
        name="Spot",
        position=np.array([SPOT_START_X, SPOT_START_Y, SPOT_START_Z])
    )
    spot.initialize()
    spot.robot.set_joints_default_state(spot.default_pos)
    setup_spot_sensors("/World/Spot")
    
    # Create environment
    print("[INFO] Creating environment...")
    env = CircularWaypointEnv(world, stage, rng)
    env.spot = spot
    env.reset(episode=1)
    
    # Initialize networks
    print("[INFO] Initializing PPO networks...")
    obs_dim = 98  # Updated with waypoint direction (96 + 2)
    action_dim = 3  # [vx, vy, omega_z]
    
    actor_critic = ActorCritic(
        obs_dim,
        action_dim,
        actor_hidden=config.ppo.actor_hidden_dims,
        critic_hidden=config.ppo.critic_hidden_dims
    )
    
    ppo = PPO(actor_critic, config)
    
    # Create checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir) / config.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file for training progress (visible output)
    log_file = checkpoint_dir / "training_log.txt"
    def log(msg):
        """Write to both console and file."""
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')
            f.flush()
    
    # Resume from checkpoint if specified
    start_iteration = 0
    consecutive_successes = 0
    current_stage = 1
    
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            log(f"\n[RESUME] Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            
            # Load network weights (strict=False allows missing log_std for transferred weights)
            actor_critic.actor.load_state_dict(checkpoint['actor_state_dict'], strict=False)
            actor_critic.critic.load_state_dict(checkpoint['critic_state_dict'], strict=False)
            
            # Load optimizer state if available (not present in transferred checkpoints)
            if 'optimizer_state_dict' in checkpoint:
                ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                log("[RESUME] Optimizer state loaded")
            else:
                log("[INFO] No optimizer state in checkpoint (fresh training)")
            
            # Load training state
            start_iteration = checkpoint.get('iteration', 0) + 1
            current_stage = checkpoint.get('current_stage', 1)
            consecutive_successes = checkpoint.get('consecutive_successes', 0)
            
            log(f"[RESUME] Resumed from iteration {start_iteration}")
            log(f"[RESUME] Current stage: {current_stage}")
            log(f"[RESUME] Consecutive successes: {consecutive_successes}")
            log(f"[RESUME] Checkpoint loaded successfully!\n")
        else:
            log(f"[WARNING] Checkpoint not found: {checkpoint_path}")
            log(f"[WARNING] Starting training from scratch\n")
    
    log(f"[INFO] Training started at {datetime.now()}")
    log(f"[INFO] Checkpoint directory: {checkpoint_dir}")
    log(f"[INFO] Observation dim: {obs_dim}")
    log(f"[INFO] Action dim: {action_dim}")
    log(f"[INFO] Actor hidden layers: {config.ppo.actor_hidden_dims}")
    log(f"[INFO] Critic hidden layers: {config.ppo.critic_hidden_dims}")
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    
    log("\n" + "=" * 80)
    log("STARTING TRAINING" if not args.checkpoint else "RESUMING TRAINING")
    log("=" * 80)
    
    # Curriculum learning: Stage progression tracking
    required_successes = 10    # Need 10 consecutive successes to advance stage
    max_stage = 4              # 4 stages total
    
    log("\n[CURRICULUM] 4-Stage Training:")
    log("  Stage 1: No obstacles (0% coverage)")
    log("  Stage 2: +10% light/medium moveable obstacles")
    log("  Stage 3: +10% heavy immovable obstacles (20% total)")
    log("  Stage 4: +10% small static obstacles (30% total)")
    log(f"  Progression: {required_successes} consecutive successful episodes to advance")
    log(f"  Success criteria: Reach all 25 waypoints\n")
    
    # Set initial stage
    env.set_training_stage(current_stage)
    
    # Episode state tracking (persistent across rollouts)
    episode_reward = 0.0
    episode_length = 0
    previous_action = np.zeros(3, dtype=np.float32)
    
    # Score-based system tracking
    episode_score = 300.0
    episode_elapsed_time = 0.0
    episode_waypoints = 0
    prev_dist = None  # For distance shaping
    
    obs = get_rl_observation(env, previous_action)
    
    # Training loop
    for iteration in range(start_iteration, config.ppo.num_learning_iterations):
        iteration_start = time.time()
        log(f"\n[ITER {iteration+1}/{config.ppo.num_learning_iterations}] Starting rollout collection...")
        
        # Rollout storage
        rollouts = RolloutStorage(config.ppo.num_steps_per_env, obs_dim, action_dim)
        rollout_reward = 0.0  # Track reward during this rollout
        
        # Get initial observation tensor
        obs_tensor = torch.from_numpy(obs).float()
        
        for step in range(config.ppo.num_steps_per_env):
            # Get action from policy
            with torch.no_grad():
                action_tensor, log_prob = actor_critic.actor.get_action(obs_tensor)
                value = actor_critic.critic(obs_tensor)
            
            action = action_tensor.numpy()
            
            # Apply action
            apply_rl_action(env, action)
            
            # Step simulation
            world.step(render=not args.headless)
            
            # Get state after step
            spot_pos, _ = env.spot.robot.get_world_pose()
            
            # Calculate distance to current waypoint
            current_dist = None
            waypoint_reached = False
            if env.current_waypoint_idx < len(env.waypoints):
                wp = env.waypoints[env.current_waypoint_idx]
                current_dist = math.sqrt(
                    (wp["pos"][0] - spot_pos[0])**2 + 
                    (wp["pos"][1] - spot_pos[1])**2
                )
                # Check if waypoint reached (0.5m threshold from RL_Development.py)
                if current_dist < 0.5:
                    waypoint_reached = True
                    env.current_waypoint_idx += 1
                    env.waypoints_reached += 1
                    episode_waypoints += 1
            
            # Check fall (0.25m threshold from RL_Development.py)
            fell = spot_pos[2] < 0.25
            
            # Update episode time
            episode_elapsed_time += CONTROL_DT
            
            # Calculate step reward (score change + distance shaping)
            reward = calculate_score_based_reward(CONTROL_DT, waypoint_reached, prev_dist, current_dist)
            prev_dist = current_dist  # Update for next step
            
            # Handle fall: score goes to 0 (failure)
            if fell:
                # Penalize by remaining score to make final accumulated reward = 0
                current_score = 300.0 - episode_elapsed_time + (episode_waypoints * 15.0)
                reward = -current_score  # This zeroes out the accumulated reward
                episode_score = 0.0
            else:
                # Update episode score normally (only score components, not shaping)
                score_change = -CONTROL_DT + (15.0 if waypoint_reached else 0.0)
                episode_score += score_change
            
            next_obs = get_rl_observation(env, action)
            next_obs_tensor = torch.from_numpy(next_obs).float()
            
            # Check termination
            done = False
            episode_length += 1
            
            if fell:  # Fall - score is 0 (failure)
                done = True
            elif episode_score <= 0.0:  # Score reached 0 without fall (failure)
                done = True
            elif episode_length >= config.ppo.max_episode_length:  # Max steps reached
                done = True
            
            # Store transition
            rollouts.add(obs_tensor, action_tensor, reward, log_prob, value, done)
            
            # Update for next step
            obs_tensor = next_obs_tensor
            obs = next_obs  # Update persistent obs
            previous_action = action.copy()
            episode_reward += reward
            rollout_reward += reward  # Track rollout reward
            
            # Reset if done
            if done:
                # Log episode results
                status = "FALL" if fell else ("TIMEOUT" if episode_length >= config.ppo.max_episode_length else "SCORE_ZERO")
                
                # Check if episode was successful (all 25 waypoints reached)
                episode_success = (episode_waypoints >= 25)
                
                if episode_success:
                    consecutive_successes += 1
                    success_marker = " ✓ SUCCESS"
                else:
                    consecutive_successes = 0  # Reset counter on any failure
                    success_marker = ""
                
                print(f"  Episode ended: {status} | Final Score: {episode_score:.1f} | WPs: {episode_waypoints}/25 | Steps: {episode_length} | Time: {episode_elapsed_time:.1f}s{success_marker}")
                print(f"  [CURRICULUM] Stage {current_stage} | Consecutive Successes: {consecutive_successes}/{required_successes}")
                
                # Check for stage progression
                if consecutive_successes >= required_successes and current_stage < max_stage:
                    # Save checkpoint before advancing
                    stage_checkpoint = checkpoint_dir / f"stage_{current_stage}_complete.pt"
                    torch.save({
                        'iteration': iteration,
                        'actor_state_dict': actor_critic.actor.state_dict(),
                        'critic_state_dict': actor_critic.critic.state_dict(),
                        'optimizer_state_dict': ppo.optimizer.state_dict(),
                        'stage': current_stage,
                        'consecutive_successes': consecutive_successes,
                    }, stage_checkpoint)
                    log(f"\n{'='*80}")
                    log(f"[CHECKPOINT] Stage {current_stage} completed! Saved to {stage_checkpoint.name}")
                    log(f"{'='*80}")
                    
                    # Advance to next stage
                    current_stage += 1
                    env.set_training_stage(current_stage)
                    consecutive_successes = 0  # Reset counter for new stage
                    
                    log(f"\n[CURRICULUM] Advancing to Stage {current_stage}")
                    log(f"  Target: {required_successes} more consecutive successes to advance\n")
                
                # Check if training is complete (Stage 4 finished)
                elif consecutive_successes >= required_successes and current_stage == max_stage:
                    # Save final trained policy
                    final_checkpoint = checkpoint_dir / "FULLY_TRAINED_POLICY.pt"
                    torch.save({
                        'iteration': iteration,
                        'actor_state_dict': actor_critic.actor.state_dict(),
                        'critic_state_dict': actor_critic.critic.state_dict(),
                        'optimizer_state_dict': ppo.optimizer.state_dict(),
                        'stage': current_stage,
                        'consecutive_successes': consecutive_successes,
                        'obs_dim': obs_dim,
                        'action_dim': action_dim,
                        'config': config.__dict__,
                    }, final_checkpoint)
                    log(f"\n{'='*80}")
                    log(f"🎓 TRAINING COMPLETE! All 4 stages finished!")
                    log(f"{'='*80}")
                    log(f"Final policy saved: {final_checkpoint}")
                    log(f"Total iterations: {iteration+1}")
                    log(f"Stage {max_stage} consecutive successes: {consecutive_successes}")
                    log(f"\nTo use this policy in other environments:")
                    log(f"  1. Load checkpoint: torch.load('{final_checkpoint.name}')")
                    log(f"  2. Create actor network with obs_dim={obs_dim}, action_dim={action_dim}")
                    log(f"  3. Load weights: actor.load_state_dict(checkpoint['actor_state_dict'])")
                    log(f"{'='*80}\n")
                    
                    # Stop training
                    break
                
                # Record episode statistics
                episode_rewards.append(episode_score)  # Final score
                episode_lengths.append(episode_length)
                
                # Reset environment and tracking
                env.reset(episode=iteration + 2)
                episode_score = 300.0
                episode_elapsed_time = 0.0
                episode_waypoints = 0
                episode_reward = 0.0
                episode_length = 0
                prev_dist = None  # Reset distance tracking
                previous_action = np.zeros(3, dtype=np.float32)
                obs = get_rl_observation(env, previous_action)
                obs_tensor = torch.from_numpy(obs).float()
        
        # Compute returns and advantages
        with torch.no_grad():
            last_value = actor_critic.critic(obs_tensor)
        
        returns, advantages = rollouts.compute_returns_and_advantages(
            last_value, config.ppo.gamma, config.ppo.lam
        )
        
        # Get rollout data
        rollout_data = rollouts.get_data(returns, advantages)
        
        # PPO update
        update_stats = ppo.update(rollout_data)
        
        # Logging
        iteration_time = time.time() - iteration_start
        
        if iteration % config.ppo.log_interval == 0:
            mean_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            mean_length = np.mean(episode_lengths[-10:]) if episode_lengths else 0.0
            
            log(f"\n[ITER {iteration+1}/{config.ppo.num_learning_iterations}]")
            log(f"  Time: {iteration_time:.2f}s")
            log(f"  Rollout Reward: {rollout_reward:.3f}")  # Show immediate feedback
            log(f"  Mean Episode Reward: {mean_reward:.3f}")
            log(f"  Mean Episode Length: {mean_length:.1f}")
            log(f"  Total Episodes: {len(episode_rewards)}")
            log(f"  Value Loss: {update_stats['value_loss']:.4f}")
            log(f"  Policy Loss: {update_stats['policy_loss']:.4f}")
            log(f"  Entropy: {update_stats['entropy']:.4f}")
        
        # Save checkpoint
        if iteration % config.ppo.save_interval == 0 and iteration > 0:
            checkpoint_path = checkpoint_dir / f"model_{iteration}.pt"
            torch.save({
                'iteration': iteration,
                'actor_critic_state_dict': actor_critic.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict(),
                'config': config,
            }, checkpoint_path)
            log(f"[SAVE] Checkpoint saved: {checkpoint_path}")
    
    log("\n" + "=" * 80)
    log("TRAINING COMPLETE")
    log("=" * 80)
    log(f"Total episodes: {len(episode_rewards)}")
    if episode_rewards:
        log(f"Final mean reward: {np.mean(episode_rewards[-100:]):.3f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
