"""
Navigation Policy Training
==========================
Main training script for hierarchical RL navigation policy.

Usage:
    python train_navigation.py --headless --iterations 10000

Author: Cole (MS for Autonomy Project)
Date: March 2026
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Parse args BEFORE Isaac Sim import
parser = argparse.ArgumentParser(description="Train navigation policy with curriculum learning")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--iterations", type=int, default=10000, help="Training iterations")
parser.add_argument("--config", type=str, default="nav_config.yaml", help="Config file")
parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/navigation_policy", help="Directory for checkpoints")
parser.add_argument("--stage", type=int, default=1, help="Starting curriculum stage (0-7, default=1 for Stage 2)")
args = parser.parse_args()

# Initialize Isaac Sim
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

# Now import Isaac modules
from omni.isaac.core import World
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
import omni

# Import our modules
from navigation_policy import NavigationPolicy
from navigation_env import NavigationEnvironment
from ppo_trainer import PPOTrainer, RolloutBuffer


def log(msg: str, log_file=None):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    if log_file:
        log_file.write(formatted + "\n")
        log_file.flush()


def main():
    # Load config
    config_path = Path(__file__).parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(checkpoint_dir / "training_log.txt", "w", encoding="utf-8")
    
    log("=" * 80, log_file)
    log("NAVIGATION POLICY TRAINING - HIERARCHICAL RL", log_file)
    log("=" * 80, log_file)
    log(f"Config: {args.config}", log_file)
    log(f"Iterations: {args.iterations}", log_file)
    log(f"Starting stage: {args.stage}", log_file)
    log(f"Headless: {args.headless}", log_file)
    log("", log_file)
    
    # Create world
    log("Creating Isaac Sim world...", log_file)
    physics_dt = config['physics']['dt']
    rendering_dt = config['physics']['rendering_dt']
    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=rendering_dt)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()
    
    # Reset world
    world.reset()
    log("World created and reset", log_file)
    
    # Create Spot
    log("Creating Spot robot...", log_file)
    start_pos = np.array(config['robot']['start_position'])
    start_ori = np.array(config['robot']['start_orientation'])
    spot = SpotFlatTerrainPolicy(
        prim_path="/World/Spot",
        name="Spot",
        position=start_pos
    )
    spot.initialize()
    spot.robot.set_joints_default_state(spot.default_pos)
    log("Spot initialized", log_file)
    
    # Create environment
    log("Creating navigation environment...", log_file)
    env = NavigationEnvironment(world, stage, spot, str(config_path))
    log("Environment created", log_file)
    
    # Register physics callback for Spot control
    def on_physics_step(step_size: float):
        """Physics callback - applies current command to Spot at each physics step."""
        env.apply_command(step_size)
    
    world.add_physics_callback("spot_navigation_control", on_physics_step)
    log("Physics callback registered", log_file)
    
    # Create policy
    log("Creating navigation policy...", log_file)
    obs_dim = 32
    action_dim = 3
    policy = NavigationPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=tuple(config['network']['hidden_dims']),
        activation=config['network']['activation']
    )
    log(f"Policy created with {sum(p.numel() for p in policy.parameters())} parameters", log_file)
    
    # Create PPO trainer
    log("Creating PPO trainer...", log_file)
    trainer = PPOTrainer(policy, config['ppo'])
    log("PPO trainer created", log_file)
    
    # Load checkpoint if specified
    start_iter = 0
    if args.checkpoint:
        log(f"Loading checkpoint: {args.checkpoint}", log_file)
        checkpoint = torch.load(args.checkpoint)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint.get('iteration', 0) + 1
        checkpoint_stage = checkpoint.get('stage', args.stage)
        # Use command-line stage if explicitly provided (allows advancing stages)
        env.current_stage = args.stage
        log(f"Resumed from iteration {start_iter}", log_file)
        log(f"Checkpoint was at stage {checkpoint_stage}, advancing to stage {env.current_stage}", log_file)
    else:
        env.current_stage = args.stage
    
    log("", log_file)
    log("=" * 80, log_file)
    log(f"STARTING TRAINING FROM STAGE {env.current_stage}", log_file)
    stage_info = config['curriculum']['stages'][env.current_stage]
    log(f"Stage: {stage_info['name']}", log_file)
    log(f"Goal: {stage_info['success_criterion']}", log_file)
    log("=" * 80, log_file)
    log("", log_file)
    
    # Training loop
    rollout_buffer = RolloutBuffer()
    steps_per_iteration = config['training']['steps_per_iteration']
    checkpoint_freq = config['training']['checkpoint_frequency']
    
    best_success_rate = 0.0
    episode_count = 0
    step_count = 0
    iteration = start_iter - 1  # Initialize in case loop doesn't run
    
    # Reset environment for first episode
    obs = env.reset()
    
    for iteration in range(start_iter, start_iter + args.iterations):
        iter_start = datetime.now()
        
        log(f"[ITER {iteration + 1}/{args.iterations}] Stage {env.current_stage + 1}/8: {stage_info['name']}", log_file)
        
        # Collect rollout
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        episode_waypoints = []
        
        while len(rollout_buffer) < steps_per_iteration:
            # Get action from policy
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action, log_prob, value = policy.get_action(obs_tensor)
            action = action.squeeze(0).numpy()
            log_prob = log_prob.item()
            value = value.item()
            
            # Environment step
            next_obs, reward, done, info = env.step(action)
            
            # Store transition
            rollout_buffer.add(obs, action, reward, value, log_prob, done)
            
            obs = next_obs
            step_count += 1
            
            # Handle episode end
            if done:
                episode_rewards.append(info['score'])
                episode_lengths.append(info['episode_time'])
                episode_successes.append(info['success'])
                episode_waypoints.append(info['waypoints_captured'])
                episode_count += 1
                
                # Reset for next episode
                obs = env.reset()
        
        # Compute next value for GAE
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        _, _, next_value = policy.get_action(obs_tensor, deterministic=True)
        next_value = next_value.item()
        
        # Get rollout data with returns and advantages
        rollout = rollout_buffer.get(
            next_value=next_value,
            gamma=config['ppo']['gamma'],
            gae_lambda=config['ppo']['gae_lambda']
        )
        
        # PPO update
        train_stats = trainer.update(rollout)
        
        # Clear buffer
        rollout_buffer.clear()
        
        # Logging
        iter_time = (datetime.now() - iter_start).total_seconds()
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        mean_length = np.mean(episode_lengths) if episode_lengths else 0.0
        mean_waypoints = np.mean(episode_waypoints) if episode_waypoints else 0.0
        total_waypoints = len(env.waypoints) if env.waypoints else 0  # Get actual waypoint count from environment
        success_rate = env.get_success_rate()
        
        log("", log_file)
        log(f"Iteration {iteration + 1}/{args.iterations} completed in {iter_time:.2f}s", log_file)
        log(f"  Episodes this iter: {len(episode_rewards)}", log_file)
        log(f"  Total episodes: {episode_count}", log_file)
        log(f"  Mean score: {mean_reward:.1f}", log_file)
        log(f"  Mean waypoints captured: {mean_waypoints:.1f}/{total_waypoints}", log_file)
        log(f"  Mean length: {mean_length:.1f}s", log_file)
        log(f"  Success rate (last {env.success_window}): {success_rate:.1%}", log_file)
        log(f"  Policy loss: {train_stats['policy_loss']:.4f}", log_file)
        log(f"  Value loss: {train_stats['value_loss']:.4f}", log_file)
        log(f"  Entropy: {train_stats['entropy']:.4f}", log_file)
        log(f"  Approx KL: {train_stats['approx_kl']:.4f}", log_file)
        log(f"  PPO epochs: {train_stats['epochs_completed']}/{config['ppo']['ppo_epochs']}", log_file)
        if train_stats['early_stopped']:
            log(f"  [EARLY STOP] KL divergence exceeded target ({config['ppo']['target_kl']:.4f})", log_file)
        
        # Check curriculum advancement
        if env.should_advance_curriculum():
            log("", log_file)
            log("=" * 80, log_file)
            log(f"[SUCCESS] STAGE {env.current_stage + 1} COMPLETE!", log_file)
            log(f"Success rate: {success_rate:.1%} (threshold: {env.success_threshold:.1%})", log_file)
            
            # Save stage completion checkpoint
            stage_checkpoint_path = checkpoint_dir / f"stage_{env.current_stage + 1}_complete.pt"
            torch.save({
                'iteration': iteration,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'stage': env.current_stage,
                'success_rate': success_rate
            }, stage_checkpoint_path)
            log(f"  Stage checkpoint saved: {stage_checkpoint_path.name}", log_file)
            
            # Auto-advance to next stage if not at final stage
            if env.current_stage < len(env.curriculum_stages) - 1:
                env.current_stage += 1
                log(f"[AUTO-ADVANCING] Moving to Stage {env.current_stage + 1}...", log_file)
                log("=" * 80, log_file)
                # Reset environment to new stage
                obs = env.reset(stage_id=env.current_stage)
            else:
                log("[COMPLETE] All stages finished! Training complete.", log_file)
                log("=" * 80, log_file)
                break
        
        # Save checkpoint periodically
        if (iteration + 1) % checkpoint_freq == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{iteration + 1}.pt"
            torch.save({
                'iteration': iteration,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'stage': env.current_stage,
                'success_rate': success_rate
            }, checkpoint_path)
            log(f"  Checkpoint saved: {checkpoint_path.name}", log_file)
        
        # Save best model
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            best_path = checkpoint_dir / "best_model.pt"
            torch.save({
                'iteration': iteration,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'stage': env.current_stage,
                'success_rate': success_rate
            }, best_path)
            log(f"  * New best model! Success rate: {success_rate:.1%}", log_file)
    
    # Final checkpoint
    final_path = checkpoint_dir / "final_model.pt"
    torch.save({
        'iteration': iteration,
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'stage': env.current_stage,
        'success_rate': success_rate
    }, final_path)
    
    log("", log_file)
    log("=" * 80, log_file)
    log("TRAINING COMPLETE", log_file)
    log(f"Total iterations: {iteration + 1}", log_file)
    log(f"Total episodes: {episode_count}", log_file)
    log(f"Final stage: {env.current_stage + 1}/8", log_file)
    log(f"Best success rate: {best_success_rate:.1%}", log_file)
    log(f"Final model saved: {final_path}", log_file)
    log("=" * 80, log_file)
    
    log_file.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
