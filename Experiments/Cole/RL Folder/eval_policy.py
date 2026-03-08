"""
Spot RL Policy Evaluation

Script for evaluating trained Spot RL policies with visualization and
performance metrics.

Usage:
    # Evaluate policy with rendering
    python eval_policy.py --checkpoint logs/checkpoints/model_5000.pt --episodes 10 --render
    
    # Batch evaluation for statistics
    python eval_policy.py --checkpoint logs/checkpoints/model_5000.pt --episodes 100 --seed 42
    
    # Compare multiple checkpoints
    python eval_policy.py --compare logs/checkpoints/model_*.pt --episodes 50

Author: Cole
Date: February 2026
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Isaac Sim imports
from omni.isaac.kit import SimulationApp

# Import our modules (after Isaac Sim initialization)
from spot_rl_env import SpotRLEnv
from training_config import TrainingConfig


class PolicyEvaluator:
    """
    Evaluates trained RL policies with comprehensive metrics and visualization.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        render: bool = False,
        save_video: bool = False,
    ):
        """
        Initialize policy evaluator.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            render: Whether to render evaluation
            save_video: Whether to save video of episodes
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.render = render
        self.save_video = save_video
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Isaac Sim
        sim_config = {
            "headless": not render,
            "width": 1920 if render else 640,
            "height": 1080 if render else 480,
        }
        self.simulation_app = SimulationApp(sim_config)
        
        # Import Isaac Sim modules
        from omni.isaac.core import World
        
        # Load checkpoint
        print(f"Loading checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Extract config and model
        self.config = checkpoint["config"]
        
        # Create world
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.config.env.physics_dt,
            rendering_dt=self.config.env.control_dt if render else self.config.env.physics_dt * 10,
        )
        
        # Create single evaluation environment
        print("Creating evaluation environment...")
        self.env = SpotRLEnv(
            env_id=0,
            position=np.array([0.0, 0.0, 0.0]),
            config=self.config,
            world=self.world,
            log_dir="eval_logs",
        )
        
        # Load actor-critic model
        from rsl_rl.modules import ActorCritic
        
        self.actor_critic = ActorCritic(
            num_actor_obs=self.env.get_observations().shape[0],
            num_critic_obs=self.env.get_observations().shape[0],
            num_actions=3,
            actor_hidden_dims=self.config.ppo.hidden_dims,
            critic_hidden_dims=self.config.ppo.hidden_dims,
            activation=self.config.ppo.activation,
            init_noise_std=self.config.ppo.init_noise_std,
        ).to(self.device)
        
        self.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        self.actor_critic.eval()  # Set to evaluation mode
        
        print("Policy loaded successfully!")
    
    def evaluate(
        self,
        num_episodes: int,
        seed: Optional[int] = None,
        deterministic: bool = True,
    ) -> Dict:
        """
        Evaluate policy over multiple episodes.
        
        Args:
            num_episodes: Number of episodes to run
            seed: Random seed for reproducibility
            deterministic: Whether to use deterministic policy (no exploration)
        
        Returns:
            Dictionary with evaluation metrics
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        print(f"\nEvaluating policy for {num_episodes} episodes...")
        print("=" * 80)
        
        # Metrics to track
        episode_rewards = []
        episode_lengths = []
        waypoints_reached = []
        success_count = 0
        fall_count = 0
        timeout_count = 0
        
        # Detailed metrics
        collision_counts = []
        distance_traveled = []
        avg_velocities = []
        nudge_attempts = []
        successful_nudges = []
        bypass_counts = []
        
        # Run episodes
        for episode in tqdm(range(num_episodes), desc="Episodes"):
            obs = self.env.reset()
            done = False
            episode_reward = 0.0
            step_count = 0
            
            # Episode-specific trackers
            episode_collisions = 0
            episode_distance = 0.0
            episode_velocities = []
            prev_pos = None
            
            while not done:
                # Get action from policy
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                with torch.no_grad():
                    if deterministic:
                        action = self.actor_critic.act_inference(obs_tensor)[0].cpu().numpy()
                    else:
                        action = self.actor_critic.act(obs_tensor)[0].cpu().numpy()
                
                # Step environment
                obs, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                step_count += 1
                
                # Track additional metrics
                if prev_pos is not None:
                    episode_distance += np.linalg.norm(info["position"] - prev_pos)
                prev_pos = info["position"]
                
                episode_velocities.append(np.linalg.norm(action[:2]))  # Forward + lateral velocity
                
                if info.get("collision", False):
                    episode_collisions += 1
                
                # Render if enabled
                if self.render:
                    self.world.step(render=True)
            
            # Record episode results
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            waypoints_reached.append(info["waypoints_reached"])
            
            collision_counts.append(episode_collisions)
            distance_traveled.append(episode_distance)
            avg_velocities.append(np.mean(episode_velocities))
            
            # Track outcomes
            if info.get("success", False):
                success_count += 1
            elif info.get("fell", False):
                fall_count += 1
            else:
                timeout_count += 1
            
            # Contact statistics from environment
            if hasattr(self.env, "contact_tracker"):
                nudge_attempts.append(self.env.contact_tracker.nudge_attempts)
                successful_nudges.append(self.env.contact_tracker.successful_nudges)
                bypass_counts.append(self.env.contact_tracker.bypasses)
        
        # Compile results
        results = {
            "num_episodes": num_episodes,
            "checkpoint": str(self.checkpoint_path),
            
            # Primary metrics
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_episode_length": float(np.mean(episode_lengths)),
            "mean_waypoints_reached": float(np.mean(waypoints_reached)),
            "max_waypoints_reached": int(np.max(waypoints_reached)),
            
            # Success metrics
            "success_rate": success_count / num_episodes,
            "fall_rate": fall_count / num_episodes,
            "timeout_rate": timeout_count / num_episodes,
            
            # Performance metrics
            "mean_collisions_per_episode": float(np.mean(collision_counts)),
            "mean_distance_traveled": float(np.mean(distance_traveled)),
            "mean_velocity": float(np.mean(avg_velocities)),
            
            # Obstacle interaction metrics
            "mean_nudge_attempts": float(np.mean(nudge_attempts)) if nudge_attempts else 0.0,
            "mean_successful_nudges": float(np.mean(successful_nudges)) if successful_nudges else 0.0,
            "nudge_success_rate": (
                float(np.sum(successful_nudges) / np.sum(nudge_attempts))
                if nudge_attempts and np.sum(nudge_attempts) > 0 else 0.0
            ),
            "mean_bypasses": float(np.mean(bypass_counts)) if bypass_counts else 0.0,
            
            # Raw data
            "episode_rewards": [float(r) for r in episode_rewards],
            "waypoints_per_episode": [int(w) for w in waypoints_reached],
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Print evaluation results in human-readable format."""
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"Checkpoint: {results['checkpoint']}")
        print(f"Episodes: {results['num_episodes']}")
        print()
        
        print("PRIMARY METRICS:")
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Mean Episode Length: {results['mean_episode_length']:.1f} steps")
        print(f"  Mean Waypoints: {results['mean_waypoints_reached']:.2f} / 25")
        print(f"  Max Waypoints: {results['max_waypoints_reached']} / 25")
        print()
        
        print("SUCCESS RATES:")
        print(f"  Success: {results['success_rate']*100:.1f}%")
        print(f"  Falls: {results['fall_rate']*100:.1f}%")
        print(f"  Timeouts: {results['timeout_rate']*100:.1f}%")
        print()
        
        print("PERFORMANCE:")
        print(f"  Collisions per Episode: {results['mean_collisions_per_episode']:.1f}")
        print(f"  Distance Traveled: {results['mean_distance_traveled']:.1f} m")
        print(f"  Average Velocity: {results['mean_velocity']:.2f} m/s")
        print()
        
        print("OBSTACLE INTERACTION:")
        print(f"  Nudge Attempts: {results['mean_nudge_attempts']:.1f}")
        print(f"  Successful Nudges: {results['mean_successful_nudges']:.1f}")
        print(f"  Nudge Success Rate: {results['nudge_success_rate']*100:.1f}%")
        print(f"  Bypasses: {results['mean_bypasses']:.1f}")
        print("=" * 80 + "\n")
    
    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_path}")
    
    def plot_results(self, results: Dict, output_path: Optional[str] = None):
        """Generate visualization plots of evaluation results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Policy Evaluation: {self.checkpoint_path.name}", fontsize=16)
        
        # Plot 1: Episode rewards
        ax = axes[0, 0]
        ax.plot(results["episode_rewards"], alpha=0.7)
        ax.axhline(y=np.mean(results["episode_rewards"]), color='r', linestyle='--', label='Mean')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title("Episode Rewards")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Waypoints reached distribution
        ax = axes[0, 1]
        waypoints = results["waypoints_per_episode"]
        bins = range(0, max(waypoints) + 2)
        ax.hist(waypoints, bins=bins, edgecolor='black', alpha=0.7)
        ax.axvline(x=np.mean(waypoints), color='r', linestyle='--', label=f'Mean: {np.mean(waypoints):.1f}')
        ax.set_xlabel("Waypoints Reached")
        ax.set_ylabel("Frequency")
        ax.set_title("Waypoints Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Success metrics pie chart
        ax = axes[1, 0]
        sizes = [
            results["success_rate"] * 100,
            results["fall_rate"] * 100,
            results["timeout_rate"] * 100,
        ]
        labels = [
            f'Success ({sizes[0]:.1f}%)',
            f'Falls ({sizes[1]:.1f}%)',
            f'Timeouts ({sizes[2]:.1f}%)',
        ]
        colors = ['#4CAF50', '#F44336', '#FFC107']
        ax.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
        ax.set_title("Episode Outcomes")
        
        # Plot 4: Key metrics comparison
        ax = axes[1, 1]
        metrics = [
            results["mean_waypoints_reached"],
            results["mean_collisions_per_episode"],
            results["mean_nudge_attempts"],
            results["mean_successful_nudges"],
        ]
        metric_labels = ["Waypoints", "Collisions", "Nudge\nAttempts", "Successful\nNudges"]
        bars = ax.bar(metric_labels, metrics, color=['#2196F3', '#FF5722', '#9C27B0', '#4CAF50'])
        ax.set_ylabel("Count")
        ax.set_title("Performance Metrics")
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_path}")
        else:
            plt.show()
    
    def cleanup(self):
        """Clean up resources."""
        self.world.stop()
        self.simulation_app.close()


def compare_checkpoints(checkpoint_paths: List[str], num_episodes: int = 50):
    """
    Compare multiple checkpoints.
    
    Args:
        checkpoint_paths: List of checkpoint paths to compare
        num_episodes: Number of episodes to evaluate each
    """
    print(f"\nComparing {len(checkpoint_paths)} checkpoints...")
    print("=" * 80)
    
    all_results = []
    
    for checkpoint_path in checkpoint_paths:
        print(f"\nEvaluating: {checkpoint_path}")
        
        evaluator = PolicyEvaluator(checkpoint_path, render=False)
        results = evaluator.evaluate(num_episodes=num_episodes, deterministic=True)
        all_results.append(results)
        evaluator.cleanup()
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Checkpoint Comparison", fontsize=16)
    
    checkpoint_names = [Path(cp).stem for cp in checkpoint_paths]
    
    # Plot 1: Mean rewards
    ax = axes[0, 0]
    rewards = [r["mean_reward"] for r in all_results]
    stds = [r["std_reward"] for r in all_results]
    ax.bar(checkpoint_names, rewards, yerr=stds, capsize=5)
    ax.set_ylabel("Mean Reward")
    ax.set_title("Mean Episode Reward")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Waypoints reached
    ax = axes[0, 1]
    waypoints = [r["mean_waypoints_reached"] for r in all_results]
    ax.bar(checkpoint_names, waypoints, color='#2196F3')
    ax.axhline(y=25, color='r', linestyle='--', label='Max (25)')
    ax.set_ylabel("Waypoints")
    ax.set_title("Mean Waypoints Reached")
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Success rates
    ax = axes[1, 0]
    width = 0.25
    x = np.arange(len(checkpoint_names))
    success_rates = [r["success_rate"] * 100 for r in all_results]
    fall_rates = [r["fall_rate"] * 100 for r in all_results]
    timeout_rates = [r["timeout_rate"] * 100 for r in all_results]
    
    ax.bar(x - width, success_rates, width, label='Success', color='#4CAF50')
    ax.bar(x, fall_rates, width, label='Falls', color='#F44336')
    ax.bar(x + width, timeout_rates, width, label='Timeouts', color='#FFC107')
    
    ax.set_ylabel("Percentage")
    ax.set_title("Episode Outcomes")
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoint_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Obstacle interaction
    ax = axes[1, 1]
    nudge_success = [r["nudge_success_rate"] * 100 for r in all_results]
    bypasses = [r["mean_bypasses"] for r in all_results]
    
    ax2 = ax.twinx()
    
    bars1 = ax.bar(x - width/2, nudge_success, width, label='Nudge Success %', color='#9C27B0')
    bars2 = ax2.bar(x + width/2, bypasses, width, label='Bypasses', color='#FF9800')
    
    ax.set_ylabel("Nudge Success Rate (%)")
    ax2.set_ylabel("Mean Bypasses")
    ax.set_title("Obstacle Interaction Strategies")
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoint_names, rotation=45)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig("checkpoint_comparison.png", dpi=300, bbox_inches='tight')
    print("\nComparison plot saved to: checkpoint_comparison.png")
    
    plt.show()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Spot RL policy")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (or glob pattern for --compare)"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render evaluation episodes"
    )
    
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save video of episodes"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple checkpoints (use glob pattern in --checkpoint)"
    )
    
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots"
    )
    
    return parser.parse_args()


def main():
    """Main evaluation entry point."""
    args = parse_args()
    
    if args.compare:
        # Compare multiple checkpoints
        from glob import glob
        checkpoint_paths = sorted(glob(args.checkpoint))
        
        if not checkpoint_paths:
            print(f"No checkpoints found matching: {args.checkpoint}")
            return
        
        compare_checkpoints(checkpoint_paths, num_episodes=args.episodes)
    
    else:
        # Evaluate single checkpoint
        evaluator = PolicyEvaluator(
            checkpoint_path=args.checkpoint,
            render=args.render,
            save_video=args.save_video,
        )
        
        results = evaluator.evaluate(
            num_episodes=args.episodes,
            seed=args.seed,
            deterministic=True,
        )
        
        evaluator.print_results(results)
        
        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_name = Path(args.checkpoint).stem
        evaluator.save_results(
            results,
            output_dir / f"{checkpoint_name}_results.json"
        )
        
        # Generate plots
        if args.plot:
            evaluator.plot_results(
                results,
                output_dir / f"{checkpoint_name}_plots.png"
            )
        
        evaluator.cleanup()


if __name__ == "__main__":
    main()
