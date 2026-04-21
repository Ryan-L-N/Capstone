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
parser.add_argument("--stage", type=int, default=1, help="Starting curriculum stage (1-6, default=1 for Stage 1)")
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
    variant_dir = checkpoint_dir.parent  # Get variant root directory (conservative/moderate/aggressive)
    variant_name = variant_dir.name
    log_file = open(variant_dir / f"{variant_name}_log.txt", "w", encoding="utf-8")
    
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
    obs_dim = 34
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
        checkpoint_stage = checkpoint.get('stage', args.stage - 1)
        checkpoint_iter = checkpoint.get('iteration', 0)
        
        # Use command-line stage if explicitly provided (allows advancing stages)
        # Convert from 1-indexed (stages 1-6) to 0-indexed (index 0-5)
        env.current_stage = args.stage - 1
        
        # Reset iteration counter if advancing to a new stage
        if env.current_stage != checkpoint_stage:
            start_iter = 0
            log(f"Checkpoint was at stage {checkpoint_stage + 1}, advancing to stage {env.current_stage + 1}", log_file)
            log(f"Iteration counter reset to 0 for new stage", log_file)
        else:
            start_iter = 0
            log(f"Loaded checkpoint from stage {checkpoint_stage + 1}, starting fresh at iteration 0", log_file)
    else:
        # Convert from 1-indexed (stages 1-6) to 0-indexed (index 0-5)
        env.current_stage = args.stage - 1
    
    log("", log_file)
    log("=" * 80, log_file)
    log(f"STARTING TRAINING FROM STAGE {env.current_stage + 1}", log_file)
    stage_info = config['curriculum']['stages'][env.current_stage]
    log(f"Stage: {stage_info['name']}", log_file)
    log(f"Goal: {stage_info['success_criterion']}", log_file)
    log("=" * 80, log_file)
    log("", log_file)
    
    # Training loop
    rollout_buffer = RolloutBuffer()
    steps_per_iteration = config['training']['steps_per_iteration']
    checkpoint_freq = 25  # Save every 25 iterations for better stage tracking
    max_iterations_per_stage = 500  # Maximum iterations allowed per stage
    
    best_success_rate = 0.0
    episode_count = 0
    step_count = 0
    iteration = start_iter - 1  # Initialize in case loop doesn't run
    stage_start_iteration = start_iter  # Track when current stage started (for iteration reset per stage)
    
    # Reset environment for first episode
    obs = env.reset()
    obs = env.get_network_observation()  # Extract 34-dim network observation
    
    for iteration in range(start_iter, start_iter + args.iterations):
        # Check if maximum iterations per stage exceeded
        iterations_on_stage = iteration - stage_start_iteration + 1
        if iterations_on_stage > max_iterations_per_stage:
            log("", log_file)
            log("=" * 80, log_file)
            log(f"[STAGE ITERATION LIMIT] Maximum 500 iterations reached on Stage {env.current_stage + 1}", log_file)
            log(f"Iterations on stage: {iterations_on_stage - 1} (limit: {max_iterations_per_stage})", log_file)
            log(f"Stopping training. To advance to next stage, run with --stage {env.current_stage + 2}", log_file)
            log("=" * 80, log_file)
            break
        iter_start = datetime.now()
        
        log(f"[ITER {iterations_on_stage}/500] Stage {env.current_stage + 1}/6: {stage_info['name']}", log_file)
        
        # Collect rollout
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        episode_waypoints = []
        episode_reward_breakdowns = []  # NEW: collect breakdown data
        episode_score_breakdowns = []   # NEW: collect score breakdown data
        episode_failures = []    # NEW: track how each episode failed (timeout, fall, boundary, incomplete)
        episode_safety_layers = []  # Track safety layer activations per episode
        
        while len(rollout_buffer) < steps_per_iteration:
            # Get action from policy
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action, log_prob, value = policy.get_action(obs_tensor)
            action = action.squeeze(0).numpy()
            log_prob = log_prob.item()
            value = value.item()
            
            # Environment step
            next_obs, reward, done, info = env.step(action)
            next_obs = env.get_network_observation()  # Extract 34-dim network observation
            
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
                episode_reward_breakdowns.append(info.get('reward_breakdown', {}))  # NEW
                episode_score_breakdowns.append(info.get('score_breakdown', {}))    # NEW
                episode_safety_layers.append(info.get('safety_layer', {}))
                
                # NEW: Track failure reason
                if info['success']:
                    failure_reason = 'SUCCESS'
                elif info.get('fall', False):
                    failure_reason = 'FALL'
                elif info.get('boundary', False):
                    failure_reason = 'BOUNDARY'
                elif info.get('timeout', False):
                    failure_reason = 'TIMEOUT'
                else:
                    failure_reason = 'INCOMPLETE'
                episode_failures.append(failure_reason)
                
                episode_count += 1
                
                # Reset for next episode
                obs = env.reset()
                obs = env.get_network_observation()  # Extract 34-dim network observation
        
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
        log(f"Iteration {iteration + 1}/{args.iterations} | Stage {env.current_stage + 1} completed in {iter_time:.2f}s", log_file)
        log(f"  Iterations on this stage: {iterations_on_stage}/{max_iterations_per_stage}", log_file)
        log(f"  Episodes this iter: {len(episode_rewards)}", log_file)
        log(f"  Total episodes: {episode_count}", log_file)
        log(f"  Mean score: {mean_reward:.1f}", log_file)
        log(f"  Mean waypoints captured: {mean_waypoints:.1f}/{total_waypoints}", log_file)
        log(f"  Mean length: {mean_length:.1f}s", log_file)
        log(f"  Success rate (last {env.success_window}): {success_rate:.1%}", log_file)
        
        # Per-episode breakdown
        log("", log_file)
        log("  [EPISODE BREAKDOWN]", log_file)
        for ep_idx in range(len(episode_rewards)):
            ep_num = episode_count - len(episode_rewards) + ep_idx + 1
            status = "✓ SUCCESS" if episode_successes[ep_idx] else "✗ " + episode_failures[ep_idx]
            log(f"    Ep {ep_num:4d}: Score={episode_rewards[ep_idx]:6.1f}  Waypoints={episode_waypoints[ep_idx]:2.0f}/{total_waypoints}  Time={episode_lengths[ep_idx]:6.1f}s  [{status}]", log_file)
        
        # NEW: Log failure breakdown
        if episode_failures:
            failure_counts = {}
            for failure in episode_failures:
                failure_counts[failure] = failure_counts.get(failure, 0) + 1
            log("  [FAILURE BREAKDOWN]", log_file)
            for failure_type, count in sorted(failure_counts.items()):
                pct = (count / len(episode_failures)) * 100
                log(f"    {failure_type:12s}: {count:3d} ({pct:5.1f}%)", log_file)
        
        # NEW: Log reward component breakdown
        if episode_reward_breakdowns:
            avg_reward_breakdown = {}
            for component in episode_reward_breakdowns[0].keys():
                values = [bd.get(component, 0.0) for bd in episode_reward_breakdowns]
                avg_reward_breakdown[component] = np.mean(values)
            
            total_reward = sum(avg_reward_breakdown.values())
            log("  [REWARD BREAKDOWN]", log_file)
            log(f"    TOTAL REWARD: {total_reward:8.4f}", log_file)
            # Sort by absolute value (descending) for readability
            sorted_components = sorted(avg_reward_breakdown.items(), key=lambda x: abs(x[1]), reverse=True)
            for component, value in sorted_components:
                if abs(value) > 0.001:  # Only show significant components
                    log(f"    {component:20s}: {value:8.4f}", log_file)
        
        # NEW: Log score component breakdown
        if episode_score_breakdowns:
            avg_score_breakdown = {}
            for component in episode_score_breakdowns[0].keys():
                values = [bd.get(component, 0.0) for bd in episode_score_breakdowns]
                avg_score_breakdown[component] = np.mean(values)
            
            total_score = sum(avg_score_breakdown.values())
            log("  [SCORE BREAKDOWN]", log_file)
            log(f"    TOTAL SCORE: {total_score:8.1f}", log_file)
            # Sort by absolute value (descending) for readability
            sorted_components = sorted(avg_score_breakdown.items(), key=lambda x: abs(x[1]), reverse=True)
            for component, value in sorted_components:
                if abs(value) > 0.1:  # Only show significant components
                    log(f"    {component:20s}: {value:8.1f}", log_file)
        
        log(f"  Policy loss: {train_stats['policy_loss']:.4f}", log_file)
        log(f"  Value loss: {train_stats['value_loss']:.4f}", log_file)
        log(f"  Entropy: {train_stats['entropy']:.4f}", log_file)
        log(f"  Approx KL: {train_stats['approx_kl']:.4f}", log_file)
        log(f"  PPO epochs: {train_stats['epochs_completed']}/{config['ppo']['ppo_epochs']}", log_file)
        if train_stats['early_stopped']:
            log(f"  [EARLY STOP] KL divergence exceeded target ({config['ppo']['target_kl']:.4f})", log_file)
        
        # Safety layer activation summary
        if episode_safety_layers:
            avg_hard = np.mean([s.get('hard_stops', 0) for s in episode_safety_layers])
            avg_soft = np.mean([s.get('soft_brakes', 0) for s in episode_safety_layers])
            avg_nudge = np.mean([s.get('nudges', 0) for s in episode_safety_layers])
            log(f"  [SAFETY LAYER] hard_stops={avg_hard:.1f}/ep  soft_brakes={avg_soft:.1f}/ep  nudges={avg_nudge:.1f}/ep", log_file)
        
        # Check curriculum progression milestone (informational only - requires 50 iterations + success threshold)
        if env.should_advance_curriculum(iterations_on_stage):
            log("", log_file)
            log("=" * 80, log_file)
            log(f"[MILESTONE] STAGE {env.current_stage + 1} REACHED TARGET SUCCESS!", log_file)
            log(f"Success rate: {success_rate:.1%} (threshold: {env.success_threshold:.1%})", log_file)
            log(f"Iterations on stage: {iterations_on_stage}", log_file)
            
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
            log(f"  [MILESTONE REACHED] To advance to next stage, run with: --stage {env.current_stage + 2}", log_file)
            log("=" * 80, log_file)
        
        # Save checkpoint every 25 iterations
        if iterations_on_stage % checkpoint_freq == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_stage{env.current_stage + 1}_iter{iterations_on_stage:03d}.pt"
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
    log(f"Final stage: {env.current_stage + 1}/6", log_file)
    log(f"Best success rate: {best_success_rate:.1%}", log_file)
    log(f"Final model saved: {final_path}", log_file)
    log("=" * 80, log_file)
    
    log_file.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
