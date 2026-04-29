"""
Generic experiment data collector with JSON output.

Consolidates DataCollector from grass phases 1-5.
Experiment-specific fields are passed as **kwargs to start_run/end_run.

Usage:
    from core import DataCollector

    collector = DataCollector("phase_1_baseline", "1.1_point_to_point",
                              config={"start": (1,1), "target": (17,8)})
    collector.start_run(1)
    collector.end_run(nav_controller, sim_time, collision_count=0)
    collector.save("results/phase_1")
"""

import json
import os
import numpy as np
from datetime import datetime


class DataCollector:
    """Collects and saves experiment metrics to JSON."""

    def __init__(self, experiment_name, test_name, config=None):
        """
        Args:
            experiment_name: Name of the experiment (e.g. "grass_phase_1_baseline")
            test_name: Name of this specific test (e.g. "1.1_point_to_point")
            config: Optional dict of experiment configuration to save
        """
        self.experiment_name = experiment_name
        self.test_name = test_name
        self.config = config or {}
        self.runs = []
        self.current_run = None

    def start_run(self, run_id, **extra_fields):
        """
        Start recording a new run.

        Args:
            run_id: Numeric or string run identifier
            **extra_fields: Any experiment-specific fields (e.g. seed=42, height_level="H1")
        """
        self.current_run = {
            "run_id": run_id,
            "start_time": datetime.now().isoformat(),
            "success": False,
            "navigation_time": 0.0,
            "path_length": 0.0,
            "path_efficiency": 0.0,
            "fell": False,
            "timeout": False,
            "failure_reason": None,
        }
        self.current_run.update(extra_fields)

    def end_run(self, nav_controller, sim_time, **extra_fields):
        """
        End current run, auto-extract metrics from NavigationController.

        Args:
            nav_controller: NavigationController instance
            sim_time: Current simulation time
            **extra_fields: Any additional fields (e.g. fell=True, collision_count=3)
        """
        if self.current_run is None:
            return

        self.current_run["success"] = nav_controller.is_complete()
        self.current_run["navigation_time"] = nav_controller.get_navigation_time(sim_time)
        self.current_run["path_length"] = nav_controller.path_length
        self.current_run["path_efficiency"] = nav_controller.get_path_efficiency()

        if nav_controller.is_failed():
            self.current_run["failure_reason"] = getattr(
                nav_controller, "failure_reason", "Unknown"
            )

        self.current_run.update(extra_fields)
        self.runs.append(self.current_run)
        self.current_run = None

    def get_summary(self):
        """Calculate summary statistics across all recorded runs."""
        if not self.runs:
            return {}

        successful = [r for r in self.runs if r["success"]]

        summary = {
            "test_name": self.test_name,
            "total_runs": len(self.runs),
            "successful_runs": len(successful),
            "task_completion_rate": len(successful) / len(self.runs) * 100,
            "fall_count": sum(1 for r in self.runs if r.get("fell", False)),
            "timeout_count": sum(1 for r in self.runs if r.get("timeout", False)),
        }

        if successful:
            nav_times = [r["navigation_time"] for r in successful]
            path_effs = [r["path_efficiency"] for r in successful]
            summary["mean_navigation_time"] = float(np.mean(nav_times))
            summary["std_navigation_time"] = float(np.std(nav_times))
            summary["mean_path_efficiency"] = float(np.mean(path_effs))
            summary["std_path_efficiency"] = float(np.std(path_effs))

        return summary

    def save(self, output_dir):
        """
        Save results to timestamped JSON file.

        Args:
            output_dir: Directory to write the JSON file

        Returns:
            Path to the saved file
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.test_name}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        data = {
            "experiment": self.experiment_name,
            "test_name": self.test_name,
            "config": self.config,
            "runs": self.runs,
            "summary": self.get_summary(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Results saved to: {filepath}")
        return filepath
