"""
Training Environment 1 — Metrics Logger
Handles per-episode CSV export for both baseline and training runs.
No Isaac imports — safe to use anywhere.
"""

import csv
import os
import time
from pathlib import Path

_EPISODE_FIELDS = [
    "timestamp", "run_type", "generation", "episode",
    "robot_id", "surface", "friction",
    "fell", "out_of_bounds", "survival_time", "total_reward",
    "mean_vel_x", "mean_cmd_vx", "slip_ratio",
    "mean_roll_deg", "mean_pitch_deg",
    "std_roll_deg", "std_pitch_deg",
    "mean_roll_rate", "mean_pitch_rate",
    "total_joint_work", "cmd_smoothness", "step_count",
]


class MetricsLogger:
    """
    Appends one CSV row per episode to a log file.
    Creates the file and writes a header if it does not exist.
    Also provides a console print_summary helper.
    """

    def __init__(self, log_path: str, run_type: str = "baseline"):
        self.log_path = Path(log_path)
        self.run_type = run_type
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._episode_count = 0
        self._generation = 0

        write_header = not self.log_path.exists()
        self._file = open(self.log_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=_EPISODE_FIELDS)
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def set_generation(self, gen: int):
        """Call before each CEM generation to tag CSV rows correctly."""
        self._generation = gen

    def log_episode(self, episode_summary: dict, episode_global: int = None):
        """
        Write one row to the CSV.

        Args:
            episode_summary : dict from RobotState.get_episode_summary()
            episode_global  : optional override for episode index column
        """
        self._episode_count += 1
        row = {
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_type":   self.run_type,
            "generation": self._generation,
            "episode":    episode_global if episode_global is not None else self._episode_count,
            **episode_summary,
        }
        for field in _EPISODE_FIELDS:
            row.setdefault(field, "")
        self._writer.writerow(row)
        self._file.flush()

    def print_summary(self, episode_summaries: list, label: str = ""):
        """Print aggregated stats for a batch of episode summaries to console."""
        if not episode_summaries:
            return
        n          = len(episode_summaries)
        falls      = sum(int(s.get("fell", 0)) for s in episode_summaries)
        rewards    = [float(s.get("total_reward", 0)) for s in episode_summaries]
        vel_xs     = [float(s.get("mean_vel_x", 0)) for s in episode_summaries]
        rolls      = [float(s.get("mean_roll_deg", 0)) for s in episode_summaries]
        pitches    = [float(s.get("mean_pitch_deg", 0)) for s in episode_summaries]
        slips      = [float(s.get("slip_ratio", 0)) for s in episode_summaries]
        surface    = episode_summaries[0].get("surface", "?")

        print(
            f"  [{label or surface:>14s}] n={n:3d} | "
            f"falls={falls:3d} ({100*falls/n:4.1f}%) | "
            f"reward={sum(rewards)/n:7.1f} | "
            f"vel={sum(vel_xs)/n:.2f}m/s | "
            f"slip={sum(slips)/n:.2f} | "
            f"roll={sum(rolls)/n:.1f}° pitch={sum(pitches)/n:.1f}°"
        )

    def close(self):
        self._file.close()

    def __del__(self):
        try:
            self._file.close()
        except Exception:
            pass
