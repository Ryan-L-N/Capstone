"""TensorBoard watcher — tails JSONL episode files and writes TensorBoard events.

Runs as a background process alongside the parallel eval. Polls for new JSONL
lines every N seconds and writes per-episode metrics as TensorBoard scalars.

Usage:
    python tb_watcher.py --results_dir /path/to/results --tb_dir /path/to/tb \
        --envs friction grass boulder stairs --poll_interval 10
"""

import argparse
import json
import os
import signal
import sys
import time

# Graceful shutdown
_running = True

def _handle_signal(signum, frame):
    global _running
    _running = False

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def main():
    parser = argparse.ArgumentParser(description="TensorBoard JSONL watcher")
    parser.add_argument("--results_dir", required=True, help="Directory with JSONL files")
    parser.add_argument("--tb_dir", required=True, help="TensorBoard log directory")
    parser.add_argument("--envs", nargs="+", default=["friction", "grass", "boulder", "stairs"])
    parser.add_argument("--poll_interval", type=int, default=10, help="Seconds between polls")
    args = parser.parse_args()

    # Wait for torch/tensorboard import (may take a moment on H100)
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("[TB_WATCHER] tensorboard not available, trying tensorboardX...")
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            print("[TB_WATCHER] No TensorBoard writer available. Install: pip install tensorboard")
            sys.exit(1)

    os.makedirs(args.tb_dir, exist_ok=True)

    # One writer per environment
    writers = {}
    for env in args.envs:
        env_dir = os.path.join(args.tb_dir, env)
        os.makedirs(env_dir, exist_ok=True)
        writers[env] = SummaryWriter(log_dir=env_dir)

    # Also a combined writer for cross-env comparison
    combined_dir = os.path.join(args.tb_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)
    combined_writer = SummaryWriter(log_dir=combined_dir)

    # Track how many lines we've read per file
    file_offsets = {}  # filepath -> lines_read
    ep_counters = {}   # env -> episode_count

    for env in args.envs:
        ep_counters[env] = 0

    print(f"[TB_WATCHER] Watching {args.results_dir} for JSONL files...")
    print(f"[TB_WATCHER] Writing TensorBoard events to {args.tb_dir}")
    print(f"[TB_WATCHER] Environments: {args.envs}")
    sys.stdout.flush()

    while _running:
        for env in args.envs:
            jsonl_path = os.path.join(args.results_dir, f"{env}_rough_episodes.jsonl")
            if not os.path.exists(jsonl_path):
                continue

            # Read new lines since last check
            offset = file_offsets.get(jsonl_path, 0)
            try:
                with open(jsonl_path, "r") as f:
                    lines = f.readlines()
            except (IOError, OSError):
                continue

            new_lines = lines[offset:]
            if not new_lines:
                continue

            file_offsets[jsonl_path] = len(lines)

            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    ep = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ep_num = ep_counters[env]
                ep_counters[env] += 1
                writer = writers[env]

                # Per-episode scalars
                if "progress" in ep:
                    writer.add_scalar("Episode/progress_m", ep["progress"], ep_num)
                    combined_writer.add_scalar(f"{env}/progress_m", ep["progress"], ep_num)

                if "zone_reached" in ep:
                    writer.add_scalar("Episode/zone_reached", ep["zone_reached"], ep_num)
                    combined_writer.add_scalar(f"{env}/zone_reached", ep["zone_reached"], ep_num)

                if "completion" in ep:
                    writer.add_scalar("Episode/completed", 1.0 if ep["completion"] else 0.0, ep_num)
                    combined_writer.add_scalar(f"{env}/completed", 1.0 if ep["completion"] else 0.0, ep_num)

                if "fall_detected" in ep:
                    writer.add_scalar("Episode/fell", 1.0 if ep["fall_detected"] else 0.0, ep_num)
                    combined_writer.add_scalar(f"{env}/fell", 1.0 if ep["fall_detected"] else 0.0, ep_num)

                if "stability_score" in ep:
                    writer.add_scalar("Episode/stability_score", ep["stability_score"], ep_num)

                if "mean_roll" in ep:
                    writer.add_scalar("Stability/mean_roll", ep["mean_roll"], ep_num)

                if "mean_pitch" in ep:
                    writer.add_scalar("Stability/mean_pitch", ep["mean_pitch"], ep_num)

                if "height_variance" in ep:
                    writer.add_scalar("Stability/height_variance", ep["height_variance"], ep_num)

                if "mean_ang_vel" in ep:
                    writer.add_scalar("Stability/mean_ang_vel", ep["mean_ang_vel"], ep_num)

                if "mean_velocity" in ep:
                    writer.add_scalar("Episode/mean_velocity", ep["mean_velocity"], ep_num)
                    combined_writer.add_scalar(f"{env}/mean_velocity", ep["mean_velocity"], ep_num)

                if "total_energy" in ep:
                    writer.add_scalar("Episode/total_energy", ep["total_energy"], ep_num)

                if "episode_length" in ep:
                    writer.add_scalar("Episode/episode_length_s", ep["episode_length"], ep_num)
                    combined_writer.add_scalar(f"{env}/episode_length_s", ep["episode_length"], ep_num)

                if ep.get("time_to_complete") is not None:
                    writer.add_scalar("Episode/time_to_complete_s", ep["time_to_complete"], ep_num)

                if ep.get("fall_location") is not None:
                    writer.add_scalar("Episode/fall_location_m", ep["fall_location"], ep_num)

            # Flush after processing new lines
            writer.flush()
            combined_writer.flush()

            print(f"[TB_WATCHER] {env}: {ep_counters[env]} episodes logged", flush=True)

        time.sleep(args.poll_interval)

    # Cleanup
    print("[TB_WATCHER] Shutting down...")
    for w in writers.values():
        w.close()
    combined_writer.close()


if __name__ == "__main__":
    main()
