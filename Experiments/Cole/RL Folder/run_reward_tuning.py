"""
Reward Weight Tuning Experiments
=================================
Systematic exploration of reward weights based on RL_Development.py scoring:
- Start at 300 points (episode budget)
- -1 point per second (time_penalty scaled by dt)
- +15 points for waypoint
- Points → 0 if fallen

This script runs systematic experiments varying one reward component at a time.

Author: Cole
Date: February 2026
"""

import subprocess
import time
import sys
from pathlib import Path

# Experiment suites
WAYPOINT_BONUS_EXPERIMENTS = [
    ("simple_focused", 20),  # Baseline: 15.0
    ("waypoint_10", 20),     # Lower: 10.0
    ("waypoint_20", 20),     # Higher: 20.0
    ("waypoint_30", 20),     # Very high: 30.0
]

TIME_PENALTY_EXPERIMENTS = [
    ("simple_focused", 20),  # Baseline: 1.0/sec
    ("time_0p5", 20),        # Lower: 0.5/sec (less pressure)
    ("time_1p5", 20),        # Higher: 1.5/sec (more pressure)
]

DISTANCE_SHAPING_EXPERIMENTS = [
    ("simple_focused", 20),  # Baseline: 1.0
    ("dist_0p5", 20),        # Lower: 0.5 (less dense)
    ("dist_2p0", 20),        # Higher: 2.0 (more dense)
]

FALL_PENALTY_EXPERIMENTS = [
    ("simple_focused", 20),  # Baseline: 100.0
    ("fall_50", 20),         # Lower: 50.0 (less harsh)
    ("fall_150", 20),        # Very high: 150.0 (more harsh)
]

ALL_REWARD_EXPERIMENTS = [
    ("simple_focused", 30),   # Baseline
    ("waypoint_10", 30),
    ("waypoint_20", 30),
    ("waypoint_30", 30),
    ("time_0p5", 30),
    ("time_1p5", 30),
    ("dist_0p5", 30),
    ("dist_2p0", 30),
    ("fall_50", 30),
    ("fall_150", 30),
]


def run_experiment(config_name, iterations):
    """Run a single experiment with specified config."""
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT: {config_name} ({iterations} iterations)")
    print(f"{'='*80}\n")
    
    cmd = [
        r"C:\isaac-sim\python.bat",
        "train_spot_ppo.py",
        "--headless",
        "--config", config_name,
        "--iterations", str(iterations)
    ]
    
    log_file = f"reward_tune_{config_name}.txt"
    
    start_time = time.time()
    
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).parent
            )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ {config_name} completed successfully in {elapsed:.1f}s")
            print(f"  Log: {log_file}")
            return True
        else:
            print(f"✗ {config_name} failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"✗ {config_name} crashed: {e}")
        return False


def run_experiment_suite(experiments, suite_name):
    """Run a suite of experiments."""
    print(f"\n{'#'*80}")
    print(f"# {suite_name}")
    print(f"{'#'*80}\n")
    
    results = []
    total_start = time.time()
    
    for config_name, iterations in experiments:
        success = run_experiment(config_name, iterations)
        results.append((config_name, success))
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUITE SUMMARY: {suite_name}")
    print(f"{'='*80}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"\nResults:")
    for config_name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {config_name:20s}: {status}")
    
    success_count = sum(1 for _, success in results if success)
    print(f"\nSuccess rate: {success_count}/{len(results)}")
    
    # Analysis command
    print(f"\n{'='*80}")
    print("NEXT STEPS:")
    print(f"{'='*80}")
    print("1. Analyze results:")
    for config_name, _ in experiments:
        log_path = f"checkpoints/{config_name}/training_log.txt"
        print(f"   python analyze_training.py --log {log_path}")
    
    print("\n2. Compare experiments:")
    log_paths = " ".join([f"checkpoints/{cfg}/training_log.txt" for cfg, _ in experiments])
    labels = " ".join([cfg for cfg, _ in experiments])
    print(f"   python analyze_training.py --compare {log_paths} --labels {labels}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run reward tuning experiments")
    parser.add_argument("--waypoint", action="store_true", help="Run waypoint bonus experiments")
    parser.add_argument("--time", action="store_true", help="Run time penalty experiments")
    parser.add_argument("--distance", action="store_true", help="Run distance shaping experiments")
    parser.add_argument("--fall", action="store_true", help="Run fall penalty experiments")
    parser.add_argument("--all", action="store_true", help="Run all reward tuning experiments")
    parser.add_argument("--config", type=str, help="Run single config")
    parser.add_argument("--iterations", type=int, default=30, help="Iterations per experiment")
    
    args = parser.parse_args()
    
    if args.config:
        run_experiment(args.config, args.iterations)
    elif args.waypoint:
        run_experiment_suite(WAYPOINT_BONUS_EXPERIMENTS, "WAYPOINT BONUS TUNING")
    elif args.time:
        run_experiment_suite(TIME_PENALTY_EXPERIMENTS, "TIME PENALTY TUNING")
    elif args.distance:
        run_experiment_suite(DISTANCE_SHAPING_EXPERIMENTS, "DISTANCE SHAPING TUNING")
    elif args.fall:
        run_experiment_suite(FALL_PENALTY_EXPERIMENTS, "FALL PENALTY TUNING")
    elif args.all:
        run_experiment_suite(ALL_REWARD_EXPERIMENTS, "COMPLETE REWARD WEIGHT TUNING")
    else:
        print("Reward Weight Tuning Experiments")
        print("="*80)
        print("\nUsage:")
        print("  python run_reward_tuning.py --waypoint    # Tune waypoint bonus (10, 15, 20, 30)")
        print("  python run_reward_tuning.py --time        # Tune time penalty (0.01, 0.02, 0.03)")
        print("  python run_reward_tuning.py --distance    # Tune distance shaping (0.5, 1.0, 2.0)")
        print("  python run_reward_tuning.py --fall        # Tune fall penalty (50, 100, 150)")
        print("  python run_reward_tuning.py --all         # Run all reward tuning experiments")
        print("  python run_reward_tuning.py --config simple_focused --iterations 30")
        print("\nReward Structure (based on RL_Development.py):")
        print("  - Start at 300 points (episode budget)")
        print("  - -1 point per second (-0.02 per step at 50Hz)")
        print("  - +15 points for waypoint reached")
        print("  - Points → 0 if fallen")
