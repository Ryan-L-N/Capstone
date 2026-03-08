"""
Training Analysis and Hyperparameter Tuning Tool
=================================================
Analyzes training logs to help tune PPO hyperparameters.

Usage:
    python analyze_training.py --log training_log.txt
    python analyze_training.py --compare log1.txt log2.txt log3.txt

Author: Cole
Date: February 2026
"""

import argparse
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def parse_training_log(log_path: Path) -> Dict[str, List[float]]:
    """Parse training log file and extract metrics."""
    metrics = {
        'iteration': [],
        'time': [],
        'mean_reward': [],
        'mean_length': [],
        'value_loss': [],
        'policy_loss': [],
        'entropy': []
    }
    
    with open(log_path, 'r') as f:
        for line in f:
            # Match iteration lines: [ITER 1/100]
            if '[ITER' in line:
                match = re.search(r'\[ITER (\d+)/(\d+)\]', line)
                if match:
                    iteration = int(match.group(1))
                    metrics['iteration'].append(iteration)
            
            # Match metric lines
            if 'Time:' in line:
                match = re.search(r'Time: ([\d.]+)s', line)
                if match:
                    metrics['time'].append(float(match.group(1)))
            
            if 'Mean Reward:' in line:
                match = re.search(r'Mean Reward: ([-\d.]+)', line)
                if match:
                    metrics['mean_reward'].append(float(match.group(1)))
            
            if 'Mean Length:' in line:
                match = re.search(r'Mean Length: ([\d.]+)', line)
                if match:
                    metrics['mean_length'].append(float(match.group(1)))
            
            if 'Value Loss:' in line:
                match = re.search(r'Value Loss: ([\d.]+)', line)
                if match:
                    metrics['value_loss'].append(float(match.group(1)))
            
            if 'Policy Loss:' in line:
                match = re.search(r'Policy Loss: ([-\d.]+)', line)
                if match:
                    metrics['policy_loss'].append(float(match.group(1)))
            
            if 'Entropy:' in line:
                match = re.search(r'Entropy: ([\d.]+)', line)
                if match:
                    metrics['entropy'].append(float(match.group(1)))
    
    return metrics


def plot_training_curves(metrics: Dict[str, List[float]], title: str = "Training Progress"):
    """Plot training curves for analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title)
    
    # Reward curve
    ax = axes[0, 0]
    if metrics['mean_reward']:
        ax.plot(metrics['iteration'], metrics['mean_reward'], 'b-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Reward')
        ax.set_title('Reward Curve')
        ax.grid(True)
    
    # Episode length
    ax = axes[0, 1]
    if metrics['mean_length']:
        ax.plot(metrics['iteration'], metrics['mean_length'], 'g-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Episode Length')
        ax.set_title('Episode Length')
        ax.grid(True)
    
    # Value loss
    ax = axes[0, 2]
    if metrics['value_loss']:
        ax.plot(metrics['iteration'], metrics['value_loss'], 'r-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value Loss')
        ax.set_title('Value Function Loss')
        ax.grid(True)
    
    # Policy loss
    ax = axes[1, 0]
    if metrics['policy_loss']:
        ax.plot(metrics['iteration'], metrics['policy_loss'], 'm-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Policy Loss')
        ax.set_title('Policy Loss')
        ax.grid(True)
    
    # Entropy
    ax = axes[1, 1]
    if metrics['entropy']:
        ax.plot(metrics['iteration'], metrics['entropy'], 'c-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Entropy')
        ax.set_title('Policy Entropy')
        ax.grid(True)
    
    # Iteration time
    ax = axes[1, 2]
    if metrics['time']:
        ax.plot(metrics['iteration'], metrics['time'], 'orange')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (s)')
        ax.set_title('Iteration Time')
        ax.grid(True)
    
    plt.tight_layout()
    return fig


def compare_runs(log_paths: List[Path], labels: List[str]):
    """Compare multiple training runs."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training Run Comparison')
    
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    
    for i, (log_path, label) in enumerate(zip(log_paths, labels)):
        metrics = parse_training_log(log_path)
        color = colors[i % len(colors)]
        
        # Reward comparison
        ax = axes[0, 0]
        if metrics['mean_reward']:
            ax.plot(metrics['iteration'], metrics['mean_reward'], 
                   color=color, label=label, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Reward')
        ax.set_title('Reward Comparison')
        ax.legend()
        ax.grid(True)
        
        # Value loss comparison
        ax = axes[0, 1]
        if metrics['value_loss']:
            ax.plot(metrics['iteration'], metrics['value_loss'], 
                   color=color, label=label, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value Loss')
        ax.set_title('Value Loss Comparison')
        ax.legend()
        ax.grid(True)
        
        # Policy loss comparison
        ax = axes[1, 0]
        if metrics['policy_loss']:
            ax.plot(metrics['iteration'], metrics['policy_loss'], 
                   color=color, label=label, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Policy Loss')
        ax.set_title('Policy Loss Comparison')
        ax.legend()
        ax.grid(True)
        
        # Entropy comparison
        ax = axes[1, 1]
        if metrics['entropy']:
            ax.plot(metrics['iteration'], metrics['entropy'], 
                   color=color, label=label, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Entropy')
        ax.set_title('Entropy Comparison')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    return fig


def print_summary(metrics: Dict[str, List[float]], log_path: Path):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print(f"TRAINING SUMMARY: {log_path.name}")
    print("=" * 60)
    
    if metrics['mean_reward']:
        rewards = metrics['mean_reward']
        print(f"Reward Stats:")
        print(f"  Initial: {rewards[0]:.3f}")
        print(f"  Final: {rewards[-1]:.3f}")
        print(f"  Mean: {np.mean(rewards):.3f}")
        print(f"  Max: {np.max(rewards):.3f}")
        print(f"  Min: {np.min(rewards):.3f}")
        print(f"  Improvement: {rewards[-1] - rewards[0]:.3f}")
    
    if metrics['value_loss']:
        print(f"\nValue Loss:")
        print(f"  Initial: {metrics['value_loss'][0]:.4f}")
        print(f"  Final: {metrics['value_loss'][-1]:.4f}")
        print(f"  Mean: {np.mean(metrics['value_loss']):.4f}")
    
    if metrics['policy_loss']:
        print(f"\nPolicy Loss:")
        print(f"  Initial: {metrics['policy_loss'][0]:.4f}")
        print(f"  Final: {metrics['policy_loss'][-1]:.4f}")
        print(f"  Mean: {np.mean(metrics['policy_loss']):.4f}")
    
    if metrics['entropy']:
        print(f"\nEntropy:")
        print(f"  Initial: {metrics['entropy'][0]:.4f}")
        print(f"  Final: {metrics['entropy'][-1]:.4f}")
        print(f"  Mean: {np.mean(metrics['entropy']):.4f}")
    
    if metrics['time']:
        print(f"\nTiming:")
        print(f"  Mean iteration time: {np.mean(metrics['time']):.2f}s")
        print(f"  Total time: {sum(metrics['time']):.2f}s")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze PPO training logs")
    parser.add_argument('--log', type=str, help="Single log file to analyze")
    parser.add_argument('--compare', nargs='+', help="Multiple log files to compare")
    parser.add_argument('--output', type=str, default='training_analysis.png', 
                       help="Output plot filename")
    args = parser.parse_args()
    
    if args.log:
        # Single log analysis
        log_path = Path(args.log)
        if not log_path.exists():
            print(f"Error: Log file not found: {log_path}")
            return
        
        metrics = parse_training_log(log_path)
        print_summary(metrics, log_path)
        
        fig = plot_training_curves(metrics, f"Training: {log_path.name}")
        output_path = Path(args.output)
        fig.savefig(output_path, dpi=150)
        print(f"\nPlot saved: {output_path}")
        plt.show()
    
    elif args.compare:
        # Multiple log comparison
        log_paths = [Path(p) for p in args.compare]
        labels = [p.stem for p in log_paths]
        
        # Print summaries
        for log_path in log_paths:
            if log_path.exists():
                metrics = parse_training_log(log_path)
                print_summary(metrics, log_path)
        
        # Plot comparison
        fig = compare_runs(log_paths, labels)
        output_path = Path(args.output)
        fig.savefig(output_path, dpi=150)
        print(f"\nComparison plot saved: {output_path}")
        plt.show()
    
    else:
        print("Error: Must specify either --log or --compare")
        parser.print_help()


if __name__ == "__main__":
    main()
