"""
Central Launcher for MS for Autonomy Experiments
=================================================

Usage:
    python launch.py                      # Interactive menu
    python launch.py grass-p1             # Run specific experiment
    python launch.py --list               # List all experiments
    python launch.py --list grass         # List experiments by category
    python launch.py grass-p1 --headless  # Pass extra args to script
"""

import argparse
import subprocess
import sys
import os


def load_registry():
    """Import experiment registry."""
    # Add this directory to path so we can import
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)

    from experiment_registry import EXPERIMENTS, list_experiments, get_categories
    return EXPERIMENTS, list_experiments, get_categories


def show_interactive_menu(EXPERIMENTS, list_experiments, get_categories):
    """Display numbered menu of all experiments grouped by category."""
    print()
    print("=" * 70)
    print("  MS FOR AUTONOMY - EXPERIMENT LAUNCHER")
    print("=" * 70)

    categories = get_categories()
    all_keys = []

    category_labels = {
        "drone": "QUADCOPTER DRONE",
        "grass": "GRASS TERRAIN (SPOT)",
        "spot-training": "SPOT TRAINING (ORIGINAL)",
        "test": "TEST SCRIPTS",
        "vision60": "VISION 60",
    }

    for cat in categories:
        exps = list_experiments(cat)
        label = category_labels.get(cat, cat.upper())
        print(f"\n  [{label}]")
        for key in sorted(exps.keys()):
            exp = exps[key]
            idx = len(all_keys) + 1
            all_keys.append(key)
            exists = os.path.exists(exp["script"])
            status = "" if exists else " (missing)"
            print(f"    {idx:2d}. {exp['name']:45s} [{key}]{status}")

    print(f"\n  Enter number (1-{len(all_keys)}), experiment ID, or 'q' to quit:")

    try:
        choice = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

    if choice.lower() in ("q", "quit", "exit"):
        return None

    # Try as number
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(all_keys):
            return all_keys[idx]
        print(f"  Number out of range (1-{len(all_keys)})")
        return None
    except ValueError:
        pass

    # Try as experiment key
    if choice in EXPERIMENTS:
        return choice

    # Try partial match
    matches = [k for k in EXPERIMENTS if choice.lower() in k.lower()]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"  Ambiguous match: {', '.join(matches)}")
        return None

    print(f"  Unknown experiment: {choice}")
    return None


def run_experiment(key, EXPERIMENTS, extra_args=None):
    """Run an experiment by its registry key."""
    exp = EXPERIMENTS[key]
    script = exp["script"]

    if not os.path.exists(script):
        print(f"\n  ERROR: Script not found: {script}")
        return 1

    print(f"\n{'=' * 70}")
    print(f"  Running: {exp['name']}")
    print(f"  Script:  {script}")
    print(f"  Robot:   {exp['robot']}")
    print(f"{'=' * 70}\n")

    # Run from the script's directory so relative paths work
    script_dir = os.path.dirname(script)
    cmd = [sys.executable, script]
    if extra_args:
        cmd.extend(extra_args)

    return subprocess.call(cmd, cwd=script_dir)


def show_list(list_experiments, category_filter):
    """Print experiment list to stdout."""
    cat = None if category_filter == "all" else category_filter
    exps = list_experiments(cat)

    if not exps:
        print(f"  No experiments found for category: {category_filter}")
        return

    print(f"\n  {'ID':25s} {'NAME':45s} {'ROBOT':10s}")
    print(f"  {'-'*25} {'-'*45} {'-'*10}")
    for key in sorted(exps.keys()):
        exp = exps[key]
        print(f"  {key:25s} {exp['name']:45s} {exp['robot']:10s}")


def main():
    parser = argparse.ArgumentParser(
        description="MS for Autonomy Experiment Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch.py                    Interactive menu
  python launch.py grass-p1           Run grass phase 1
  python launch.py --list             List all experiments
  python launch.py --list vision60    List Vision 60 experiments
  python launch.py v60-nav --headless Pass args through to script
        """,
    )
    parser.add_argument("experiment", nargs="?", help="Experiment ID to run")
    parser.add_argument(
        "--list",
        nargs="?",
        const="all",
        metavar="CATEGORY",
        help="List experiments (optionally filter by category)",
    )

    args, extra = parser.parse_known_args()

    EXPERIMENTS, list_experiments, get_categories = load_registry()

    # List mode
    if args.list is not None:
        show_list(list_experiments, args.list)
        return 0

    # Direct run
    if args.experiment:
        if args.experiment not in EXPERIMENTS:
            print(f"  Unknown experiment: {args.experiment}")
            print(f"  Run 'python launch.py --list' to see available experiments")
            return 1
        return run_experiment(args.experiment, EXPERIMENTS, extra)

    # Interactive menu
    key = show_interactive_menu(EXPERIMENTS, list_experiments, get_categories)
    if key:
        return run_experiment(key, EXPERIMENTS, extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
