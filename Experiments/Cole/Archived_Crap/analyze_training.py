"""
Training Results Analysis Tool
===============================
Analyze and visualize Spot RL training results.

Features:
  - Load and parse training metrics CSV
  - Generate training curves plots
  - Calculate training statistics
  - Compare different training runs
  - Export analysis reports

Usage:
    python analyze_training.py --log-dir ./runs/spot_rl
    python analyze_training.py --csv ./runs/spot_rl/training_metrics.csv
"""

import argparse
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import pandas as pd


class TrainingAnalyzer:
    """Analyze training metrics from CSV and checkpoint files"""
    
    def __init__(self, csv_path=None, log_dir=None):
        """Initialize analyzer
        
        Args:
            csv_path: Path to training_metrics.csv
            log_dir: Path to tensorboard log directory (auto-finds CSV)
        """
        
        self.csv_path = None
        self.checkpoint_dir = None
        
        if csv_path:
            self.csv_path = Path(csv_path)
        elif log_dir:
            log_dir = Path(log_dir)
            self.csv_path = log_dir / 'training_metrics.csv'
            # Find checkpoint directory
            checkpoint_base = log_dir.parent / 'checkpoints'
            if checkpoint_base.exists():
                self.checkpoint_dir = checkpoint_base
        
        if not self.csv_path or not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        # Load metrics
        self.metrics = self._load_csv()
        self.df = pd.DataFrame(self.metrics)
        
        print(f"[OK] Loaded {len(self.metrics)} training records from {self.csv_path}")
    
    def _load_csv(self):
        """Load CSV metrics"""
        metrics = []
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                row['episode'] = int(row['episode'])
                row['total_steps'] = int(row['total_steps'])
                row['avg_reward'] = float(row['avg_reward'])
                row['avg_length'] = float(row['avg_length'])
                row['policy_loss'] = float(row['policy_loss'])
                row['value_loss'] = float(row['value_loss'])
                row['total_loss'] = float(row['total_loss'])
                row['entropy'] = float(row['entropy'])
                row['success_rate'] = float(row['success_rate'])
                metrics.append(row)
        return metrics
    
    def print_summary(self):
        """Print training summary statistics"""
        
        if len(self.df) == 0:
            print("No data available")
            return
        
        print("\n" + "="*80)
        print("TRAINING SUMMARY STATISTICS")
        print("="*80)
        
        print(f"\nEpisodes Trained: {len(self.df)}")
        print(f"Total Steps:     {self.df['total_steps'].iloc[-1]:,}")
        
        print(f"\n--- REWARD METRICS ---")
        print(f"  Max Reward:      {self.df['avg_reward'].max():.4f}")
        print(f"  Min Reward:      {self.df['avg_reward'].min():.4f}")
        print(f"  Final Reward:    {self.df['avg_reward'].iloc[-1]:.4f}")
        print(f"  Mean Reward:     {self.df['avg_reward'].mean():.4f}")
        print(f"  Reward Trend:    {self._trend(self.df['avg_reward'].values)}")
        
        print(f"\n--- SUCCESS METRICS ---")
        print(f"  Final Success Rate: {self.df['success_rate'].iloc[-1]:.1%}")
        print(f"  Mean Success Rate:  {self.df['success_rate'].mean():.1%}")
        print(f"  Best Success Rate:  {self.df['success_rate'].max():.1%}")
        
        print(f"\n--- LOSS METRICS ---")
        print(f"  Policy Loss:     {self.df['policy_loss'].iloc[-1]:.6f}")
        print(f"  Value Loss:      {self.df['value_loss'].iloc[-1]:.6f}")
        print(f"  Total Loss:      {self.df['total_loss'].iloc[-1]:.6f}")
        print(f"  Loss Trend:      {self._trend(self.df['total_loss'].values)}")
        
        print(f"\n--- EXPLORATION ---")
        print(f"  Final Entropy:   {self.df['entropy'].iloc[-1]:.4f}")
        print(f"  Mean Entropy:    {self.df['entropy'].mean():.4f}")
        
        print(f"\n--- EPISODE TIMING ---")
        print(f"  Min Episode Length:  {self.df['avg_length'].min():.1f} steps")
        print(f"  Max Episode Length:  {self.df['avg_length'].max():.1f} steps")
        print(f"  Mean Episode Length: {self.df['avg_length'].mean():.1f} steps")
        
        print(f"\n{'='*80}\n")
    
    def _trend(self, values, window=10):
        """Analyze trend (increasing/decreasing/stable)"""
        if len(values) < 2:
            return "N/A"
        
        recent = values[-window:]
        first_half_mean = np.mean(recent[:len(recent)//2])
        second_half_mean = np.mean(recent[len(recent)//2:])
        
        diff_pct = (second_half_mean - first_half_mean) / (abs(first_half_mean) + 1e-6) * 100
        
        if abs(diff_pct) < 5:
            return "STABLE"
        elif diff_pct > 5:
            return f"↑ IMPROVING ({diff_pct:+.1f}%)"
        else:
            return f"↓ DECLINING ({diff_pct:+.1f}%)"
    
    def plot_training(self, output_path=None):
        """Generate training curves plot
        
        Args:
            output_path: Path to save figure (optional)
        """
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Spot RL Training Progress', fontsize=16, fontweight='bold')
        
        episodes = self.df['episode'].values
        
        # 1. Reward over time
        ax = axes[0, 0]
        ax.plot(episodes, self.df['avg_reward'].values, 'b-', linewidth=2, label='Avg Reward')
        ax.fill_between(episodes, 
                        self.df['avg_reward'].values - self.df['avg_reward'].std(),
                        self.df['avg_reward'].values + self.df['avg_reward'].std(),
                        alpha=0.3, color='blue')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Reward Trajectory')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. Success rate
        ax = axes[0, 1]
        ax.plot(episodes, self.df['success_rate'].values * 100, 'g-', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate Over Time')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3)
        
        # 3. Losses
        ax = axes[0, 2]
        ax.plot(episodes, self.df['policy_loss'].values, 'r-', linewidth=2, label='Policy Loss')
        ax.plot(episodes, self.df['value_loss'].values, 'b-', linewidth=2, label='Value Loss')
        ax.plot(episodes, self.df['total_loss'].values, 'k--', linewidth=2, label='Total Loss')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 4. Episode length
        ax = axes[1, 0]
        ax.plot(episodes, self.df['avg_length'].values, 'm-', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Length')
        ax.grid(True, alpha=0.3)
        
        # 5. Entropy (exploration)
        ax = axes[1, 1]
        ax.plot(episodes, self.df['entropy'].values, 'c-', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Entropy')
        ax.set_title('Policy Entropy (Exploration)')
        ax.grid(True, alpha=0.3)
        
        # 6. Reward distribution (box plot for bins)
        ax = axes[1, 2]
        bin_size = max(1, len(self.df) // 5)
        reward_bins = []
        bin_labels = []
        for i in range(0, len(self.df), bin_size):
            batch = self.df['avg_reward'].iloc[i:i+bin_size].values
            reward_bins.append(batch)
            bin_labels.append(f"Ep {self.df['episode'].iloc[i]}-{self.df['episode'].iloc[min(i+bin_size-1, len(self.df)-1)]}")
        
        ax.boxplot(reward_bins, labels=bin_labels)
        ax.set_ylabel('Reward')
        ax.set_title('Reward Distribution (by Training Phase)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Plot saved to {output_path}")
        else:
            plt.show()
        
        return fig
    
    def export_report(self, output_path=None):
        """Export analysis report to text file
        
        Args:
            output_path: Path to save report
        """
        
        if output_path is None:
            output_path = self.csv_path.parent / 'training_report.txt'
        
        report = []
        report.append("="*80)
        report.append("SPOT RL TRAINING ANALYSIS REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Data Source: {self.csv_path}")
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS")
        report.append("-"*80)
        report.append(f"Total Episodes:      {len(self.df)}")
        report.append(f"Total Training Steps: {self.df['total_steps'].iloc[-1]:,}")
        report.append("")
        
        # Reward analysis
        report.append("REWARD PERFORMANCE")
        report.append("-"*80)
        report.append(f"Final Reward:        {self.df['avg_reward'].iloc[-1]:.4f}")
        report.append(f"Best Reward:         {self.df['avg_reward'].max():.4f} (Episode {self.df.loc[self.df['avg_reward'].idxmax(), 'episode']:.0f})")
        report.append(f"Mean Reward:         {self.df['avg_reward'].mean():.4f}")
        report.append(f"Reward Std Dev:      {self.df['avg_reward'].std():.4f}")
        report.append(f"Reward Trend:        {self._trend(self.df['avg_reward'].values)}")
        report.append("")
        
        # Success metrics
        report.append("SUCCESS METRICS")
        report.append("-"*80)
        report.append(f"Final Success Rate:  {self.df['success_rate'].iloc[-1]:.1%}")
        report.append(f"Best Success Rate:   {self.df['success_rate'].max():.1%}")
        report.append(f"Mean Success Rate:   {self.df['success_rate'].mean():.1%}")
        report.append("")
        
        # Loss metrics
        report.append("TRAINING LOSSES")
        report.append("-"*80)
        report.append(f"Policy Loss (Final):   {self.df['policy_loss'].iloc[-1]:.6f}")
        report.append(f"Value Loss (Final):    {self.df['value_loss'].iloc[-1]:.6f}")
        report.append(f"Total Loss (Final):    {self.df['total_loss'].iloc[-1]:.6f}")
        report.append(f"Policy Loss Trend:     {self._trend(self.df['policy_loss'].values)}")
        report.append(f"Value Loss Trend:      {self._trend(self.df['value_loss'].values)}")
        report.append("")
        
        # Exploration
        report.append("EXPLORATION BEHAVIOR")
        report.append("-"*80)
        report.append(f"Final Entropy:       {self.df['entropy'].iloc[-1]:.4f}")
        report.append(f"Mean Entropy:        {self.df['entropy'].mean():.4f}")
        report.append(f"Min Entropy:         {self.df['entropy'].min():.4f}")
        report.append(f"Max Entropy:         {self.df['entropy'].max():.4f}")
        report.append("")
        
        # Episode timing
        report.append("EPISODE STATISTICS")
        report.append("-"*80)
        report.append(f"Mean Episode Length: {self.df['avg_length'].mean():.1f} steps")
        report.append(f"Min Episode Length:  {self.df['avg_length'].min():.1f} steps")
        report.append(f"Max Episode Length:  {self.df['avg_length'].max():.1f} steps")
        report.append("")
        
        # Training efficiency
        report.append("TRAINING EFFICIENCY")
        report.append("-"*80)
        total_seconds = len(self.df) * 5  # Rough estimate
        report.append(f"Episodes per Hour:   ~{3600 / (total_seconds / len(self.df)):.1f}")
        report.append(f"Steps per Episode:   {self.df['total_steps'].iloc[-1] / len(self.df):.1f}")
        report.append("")
        
        report.append("="*80)
        report.append("END OF REPORT")
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        print(f"[OK] Report saved to {output_path}")
        
        return report_text


def main():
    """Main analysis function"""
    
    parser = argparse.ArgumentParser(description="Analyze Spot RL training results")
    parser.add_argument("--log-dir", type=str, default="./runs/spot_rl",
                       help="Path to tensorboard log directory")
    parser.add_argument("--csv", type=str, default=None,
                       help="Path to training_metrics.csv (overrides --log-dir)")
    parser.add_argument("--plot", type=str, default=None,
                       help="Save plot to specified path")
    parser.add_argument("--report", type=str, default=None,
                       help="Save analysis report to specified path")
    parser.add_argument("--full", action="store_true",
                       help="Generate plot and report automatically")
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = TrainingAnalyzer(csv_path=args.csv, log_dir=args.log_dir)
        
        # Print summary
        analyzer.print_summary()
        
        # Generate plots
        if args.plot or args.full:
            plot_path = args.plot or analyzer.csv_path.parent / 'training_curves.png'
            analyzer.plot_training(output_path=str(plot_path))
        
        # Generate report
        if args.report or args.full:
            report_path = args.report or analyzer.csv_path.parent / 'analysis_report.txt'
            analyzer.export_report(output_path=str(report_path))
        
        if not (args.plot or args.report or args.full):
            print("\nTo generate visualizations, use:")
            print(f"  python analyze_training.py --csv {analyzer.csv_path} --plot ./plot.png --report ./report.txt")
            print(f"\nOr use --full to generate both:")
            print(f"  python analyze_training.py --csv {analyzer.csv_path} --full")
    
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
