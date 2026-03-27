"""JSONL audit trail for all live parameter changes.

Each line records what changed, when, old/new values, and training state.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

import json
import os
from datetime import datetime


class ChangeLog:
    """Append-only JSONL logger for control panel changes."""

    def __init__(self, log_dir: str):
        self.path = os.path.join(log_dir, "control_panel_changes.jsonl")

    def record(self, iteration: int, command_type: str, command: dict,
               applied: dict, validation_msgs: list = None,
               state: dict = None):
        """Append one change record to the JSONL file."""
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "iteration": iteration,
            "command_type": command_type,
            "command": command,
            "applied": applied,
        }
        if validation_msgs:
            entry["validation"] = validation_msgs
        if state:
            entry["state"] = state
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def read_recent(self, n: int = 20) -> list:
        """Read the last N entries from the log."""
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            lines = f.readlines()
        entries = []
        for line in lines[-n:]:
            try:
                entries.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
        return entries
