"""
watch_training.py
=================
Live dashboard for combined nav+loco training.
Reads TensorBoard event files and prints a clean summary table every 30s.

Run this in a SECOND terminal while training is happening:

    python Experiments/Colby/CombinedPolicyTraining/watch_training.py

Or point it at a specific log dir:

    python Experiments/Colby/CombinedPolicyTraining/watch_training.py \
        --logdir Experiments/Alex/NAV_ALEX/logs/spot_nav_explore_ppo/

Press Ctrl+C to stop.
"""

import argparse
import os
import sys
import time
import glob
from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_LOGDIR = Path(__file__).resolve().parent / "logs"
REFRESH_SECONDS = 30

# Metrics to display and their "good direction" (1 = higher is better, -1 = lower is better)
METRICS = [
    # (tensorboard_tag,               display_name,              good_dir, format)
    ("Train/mean_reward",             "Total Reward",            1,        ".2f"),
    ("Train/mean_episode_length",     "Episode Length (steps)",  1,        ".0f"),
    ("Nav/forward_distance",          "Forward Distance (m)",    1,        ".1f"),
    ("Nav/survival_rate",             "Survival Rate",           1,        ".1%"),
    ("Nav/flip_rate",                 "Flip Rate",               -1,       ".1%"),
    ("Nav/body_height",               "Body Height (m)",         1,        ".3f"),
    ("Curriculum/terrain_level",      "Terrain Level (1-6)",     1,        ".2f"),
    ("Train/value_loss",              "Value Loss",              -1,       ".3f"),
    ("Train/mean_noise_std",          "Action Noise Std",        -1,       ".3f"),
]

# Thresholds for color/status indicators
HEALTHY = {
    "Nav/survival_rate":     (0.5,  None),   # above 50% = good
    "Nav/forward_distance":  (5.0,  None),   # above 5m = making progress
    "Curriculum/terrain_level": (1.5, None), # above 1.5 = advancing
    "Train/value_loss":      (None, 500.0),  # below 500 = stable
    "Nav/flip_rate":         (None, 0.5),    # below 50% = ok
}


# ---------------------------------------------------------------------------
# TensorBoard reader (no TF dependency — uses raw protobuf via tensorboard pkg)
# ---------------------------------------------------------------------------

def load_scalars(event_file: str) -> dict[str, list[tuple[int, float]]]:
    """Read all scalar summaries from a TensorBoard event file.

    Returns dict of {tag: [(step, value), ...]} sorted by step.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator, STORE_EVERYTHING_SIZE_GUIDANCE,
        )
    except ImportError:
        print("ERROR: tensorboard package not found.")
        print("Install it: pip install tensorboard")
        sys.exit(1)

    ea = EventAccumulator(event_file, size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE)
    ea.Reload()

    result = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        result[tag] = [(e.step, e.value) for e in events]

    return result


def find_latest_event_file(logdir: str) -> str | None:
    """Find the most recently modified TensorBoard event file under logdir."""
    pattern = os.path.join(logdir, "**", "events.out.tfevents.*")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def get_latest_value(scalars: dict, tag: str) -> float | None:
    """Get the most recent value for a tag."""
    if tag not in scalars or not scalars[tag]:
        return None
    return scalars[tag][-1][1]


def get_recent_trend(scalars: dict, tag: str, window: int = 5) -> float | None:
    """Compute slope (positive = increasing) over the last `window` points."""
    if tag not in scalars or len(scalars[tag]) < 2:
        return None
    points = scalars[tag][-window:]
    if len(points) < 2:
        return None
    steps = [p[0] for p in points]
    values = [p[1] for p in points]
    # Simple linear regression slope
    n = len(steps)
    sx = sum(steps)
    sy = sum(values)
    sxy = sum(x * y for x, y in zip(steps, values))
    sxx = sum(x * x for x in steps)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-9:
        return 0.0
    return (n * sxy - sx * sy) / denom


def trend_arrow(slope: float | None, good_dir: int) -> str:
    """Return a trend indicator based on slope direction."""
    if slope is None:
        return "  ?"
    if abs(slope) < 1e-6:
        return "  →"
    going_up = slope > 0
    if (going_up and good_dir == 1) or (not going_up and good_dir == -1):
        return "  ↑" if going_up else "  ↓"  # good direction
    else:
        return "  ↑" if going_up else "  ↓"  # still show direction


def status_marker(tag: str, value: float | None) -> str:
    """Return ✓, ✗, or ~ based on health thresholds."""
    if value is None:
        return "~"
    if tag not in HEALTHY:
        return " "
    lo, hi = HEALTHY[tag]
    if lo is not None and value < lo:
        return "✗"
    if hi is not None and value > hi:
        return "✗"
    return "✓"


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_dashboard(scalars: dict, event_file: str):
    """Print the full dashboard table."""
    os.system("cls" if os.name == "nt" else "clear")

    current_iter = 0
    if "Train/mean_reward" in scalars and scalars["Train/mean_reward"]:
        current_iter = scalars["Train/mean_reward"][-1][0]

    print("=" * 62)
    print("  COMBINED NAV + LOCO TRAINING — LIVE DASHBOARD")
    print("=" * 62)
    print(f"  Log file : {os.path.basename(os.path.dirname(event_file))}")
    print(f"  Iteration: {current_iter:,}    (refreshes every {REFRESH_SECONDS}s)")
    print(f"  Time     : {time.strftime('%H:%M:%S')}")
    print("-" * 62)
    print(f"  {'Metric':<28} {'Value':>10}  {'Trend':>5}  {'OK?':>4}")
    print("-" * 62)

    for tag, name, good_dir, fmt in METRICS:
        value = get_latest_value(scalars, tag)
        slope = get_recent_trend(scalars, tag)
        arrow = trend_arrow(slope, good_dir)
        marker = status_marker(tag, value)

        if value is None:
            value_str = "    —"
        else:
            value_str = format(value, fmt).rjust(10)

        print(f"  {name:<28} {value_str}  {arrow}   {marker}")

    print("-" * 62)
    print()
    print("  PHASE GUIDE:")

    # Determine phase from forward_distance and survival
    fwd = get_latest_value(scalars, "Nav/forward_distance")
    sur = get_latest_value(scalars, "Nav/survival_rate")
    lvl = get_latest_value(scalars, "Curriculum/terrain_level")

    if fwd is None:
        print("  Phase: STARTING — nav metrics not yet available")
        print("         (coach must be active for nav metrics, OR iter > 0)")
    elif sur is not None and sur < 0.3:
        print("  Phase: 1 — Learning to walk (survival rate low, normal early on)")
    elif fwd < 5.0:
        print("  Phase: 2 — Learning to move forward (distance still low)")
    elif lvl is not None and lvl < 2.0:
        print("  Phase: 2 — Moving but not advancing terrain levels yet")
    elif fwd < 15.0:
        print("  Phase: 3 — Improving navigation (terrain level climbing)")
    else:
        print("  Phase: 4 — Strong navigation (routing around obstacles)")

    print()
    print("  TARGET BENCHMARKS:")
    print("    Iter  500: forward_dist > 5m,  survival > 40%")
    print("    Iter 2000: forward_dist > 15m, terrain_level > 2.0")
    print("    Iter 5000: forward_dist > 25m, terrain_level > 3.0")
    print()
    print("  TensorBoard: http://172.24.254.24:6006  (H100)")
    print("  Press Ctrl+C to stop this monitor.")
    print("=" * 62)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Live training dashboard")
    parser.add_argument(
        "--logdir", type=str, default=str(DEFAULT_LOGDIR),
        help=f"TensorBoard log directory (default: {DEFAULT_LOGDIR})"
    )
    parser.add_argument(
        "--interval", type=int, default=REFRESH_SECONDS,
        help=f"Refresh interval in seconds (default: {REFRESH_SECONDS})"
    )
    args = parser.parse_args()

    print(f"Watching: {args.logdir}")
    print(f"Refreshing every {args.interval}s — Ctrl+C to stop\n")

    iteration = 0
    while True:
        event_file = find_latest_event_file(args.logdir)

        if event_file is None:
            print(f"\r[{time.strftime('%H:%M:%S')}] No event files found in {args.logdir} — waiting...", end="")
            time.sleep(args.interval)
            iteration += 1
            continue

        try:
            scalars = load_scalars(event_file)
            print_dashboard(scalars, event_file)
        except (OSError, ValueError, RuntimeError) as e:
            print(f"[{time.strftime('%H:%M:%S')}] Read error: {e} — retrying...")

        time.sleep(args.interval)
        iteration += 1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
