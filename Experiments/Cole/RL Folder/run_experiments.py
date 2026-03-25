"""
Hyperparameter Experiment Runner
=================================
Runs multiple training experiments with different hyperparameter configurations.

Usage:
    python run_experiments.py --iterations 50 --configs lr_low lr_high baseline
    python run_experiments.py --quick  # Quick 20-iteration tests
    python run_experiments.py --full   # Full comparison suite

Author: Cole
Date: February 2026
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time


# Quick experiment sets (20 iterations each for fast testing)
QUICK_EXPERIMENTS = [
    "baseline",
    "lr_low",
    "lr_high",
    "balanced_tuned",
]

# Learning rate sweep
LR_EXPERIMENTS = [
    "baseline",      # 3e-4
    "lr_very_low",   # 5e-5
    "lr_low",        # 1e-4
    "lr_high",       # 1e-3
]

# Network architecture sweep
NETWORK_EXPERIMENTS = [
    "baseline",      # 256,256,128
    "small_net",     # 128,128,64
    "large_net",     # 512,512,256
    "deep_net",      # 256,256,256,128
]

# PPO parameter sweep
PPO_EXPERIMENTS = [
    "baseline",
    "tight_clip",
    "loose_clip",
    "high_entropy",
    "low_entropy",
]

# Reward tuning sweep
REWARD_EXPERIMENTS = [
    "baseline",
    "high_dist_reward",
    "waypoint_focused",
    "low_time_penalty",
]

# Combined configurations
COMBINED_EXPERIMENTS = [
    "baseline",
    "aggressive",
    "conservative",
    "balanced_tuned",
]

# Full comparison (all important configs)
FULL_EXPERIMENTS = [
    "baseline",
    "lr_low",
    "lr_high",
    "small_net",
    "large_net",
    "tight_clip",
    "high_entropy",
    "balanced_tuned",
    "aggressive",
    "conservative",
]


def run_experiment(config_name, iterations, headless=True):
    """Run a single training experiment."""
    print("\n" + "=" * 80)
    print(f"RUNNING EXPERIMENT: {config_name}")
    print("=" * 80)
    
    # Build command
    cmd = [
        r"C:\isaac-sim\python.bat",
        "train_spot_ppo.py",
        "--config", config_name,
        "--iterations", str(iterations),
    ]
    
    if headless:
        cmd.append("--headless")
    
    # Run training
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start_time
    
    print(f"\n[COMPLETE] {config_name} - {elapsed:.1f}s - Exit code: {result.returncode}")
    
    return result.returncode == 0


def run_experiment_suite(configs, iterations, headless=True):
    """Run a suite of experiments."""
    print("\n" + "=" * 80)
    print(f"EXPERIMENT SUITE: {len(configs)} configurations")
    print(f"Iterations per experiment: {iterations}")
    print("=" * 80)
    
    results = {}
    total_start = time.time()
    
    for i, config_name in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Starting: {config_name}")
        success = run_experiment(config_name, iterations, headless)
        results[config_name] = success
    
    total_elapsed = time.time() - total_start
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUITE COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print(f"\nResults:")
    
    for config_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {config_name:20s}: {status}")
    
    # Print analysis command
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("To compare results, run:")
    log_paths = " ".join([f"checkpoints/{cfg}/training_log.txt" for cfg in configs])
    print(f"\npython analyze_training.py --compare {log_paths}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter experiments")
    
    # Experiment selection
    parser.add_argument("--quick", action="store_true", 
                       help="Quick test (20 iterations, 4 configs)")
    parser.add_argument("--full", action="store_true",
                       help="Full comparison suite (50 iterations, 10 configs)")
    parser.add_argument("--lr", action="store_true",
                       help="Learning rate sweep")
    parser.add_argument("--network", action="store_true",
                       help="Network architecture sweep")
    parser.add_argument("--ppo", action="store_true",
                       help="PPO parameter sweep")
    parser.add_argument("--reward", action="store_true",
                       help="Reward tuning sweep")
    parser.add_argument("--combined", action="store_true",
                       help="Combined configurations")
    
    # Manual config selection
    parser.add_argument("--configs", nargs="+",
                       help="Specific configs to run")
    
    # Training parameters
    parser.add_argument("--iterations", type=int, default=50,
                       help="Iterations per experiment (default: 50)")
    parser.add_argument("--gui", action="store_true",
                       help="Show GUI (default: headless)")
    
    args = parser.parse_args()
    
    # Determine which experiments to run
    configs = None
    iterations = args.iterations
    
    if args.quick:
        configs = QUICK_EXPERIMENTS
        iterations = 20
        print("[MODE] Quick test mode (20 iterations)")
    elif args.full:
        configs = FULL_EXPERIMENTS
        iterations = 50
        print("[MODE] Full comparison mode (50 iterations)")
    elif args.lr:
        configs = LR_EXPERIMENTS
    elif args.network:
        configs = NETWORK_EXPERIMENTS
    elif args.ppo:
        configs = PPO_EXPERIMENTS
    elif args.reward:
        configs = REWARD_EXPERIMENTS
    elif args.combined:
        configs = COMBINED_EXPERIMENTS
    elif args.configs:
        configs = args.configs
    else:
        print("Error: Must specify experiment type or --configs")
        print("\nAvailable experiment types:")
        print("  --quick      : Quick test (baseline, lr_low, lr_high, balanced_tuned)")
        print("  --full       : Full comparison (10 configs)")
        print("  --lr         : Learning rate sweep")
        print("  --network    : Network architecture sweep")
        print("  --ppo        : PPO parameter sweep")
        print("  --reward     : Reward tuning sweep")
        print("  --combined   : Combined configurations")
        print("\nOr specify configs manually:")
        print("  --configs baseline lr_low lr_high")
        return 1
    
    # Run experiments
    run_experiment_suite(configs, iterations, headless=not args.gui)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
