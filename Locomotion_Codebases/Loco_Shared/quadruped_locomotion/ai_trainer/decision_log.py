"""JSONL audit trail for all AI coach decisions.

Every coach interaction is logged with full context: metrics at the time,
the decision made, guardrail modifications, and what was actually applied.
Human-reviewable after training completes.
"""

import json
import os
from datetime import datetime


class DecisionLog:
    """Append-only JSONL log of coach decisions."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def log_decision(
        self,
        iteration: int,
        phase: str,
        metrics: dict,
        decision: dict,
        guardrail_msgs: list[str],
        applied_changes: dict,
        api_latency_ms: float = 0.0,
    ):
        """Log a single coach decision."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "phase": phase,
            "metrics": metrics,
            "decision": decision,
            "guardrail_messages": guardrail_msgs,
            "applied_changes": applied_changes,
            "api_latency_ms": api_latency_ms,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_emergency(self, iteration: int, phase: str, action: str, details: str):
        """Log an emergency action (NaN, value spike, etc.)."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "phase": phase,
            "emergency": True,
            "action": action,
            "details": details,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_phase_transition(self, iteration: int, from_phase: str,
                             to_phase: str, checkpoint: str):
        """Log a phase transition."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "phase_transition": True,
            "from_phase": from_phase,
            "to_phase": to_phase,
            "checkpoint": checkpoint,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def get_recent(self, n: int = 5) -> list[dict]:
        """Read the last N decisions from the log."""
        if not os.path.exists(self.log_path):
            return []
        entries = []
        with open(self.log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries[-n:]
