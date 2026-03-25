"""Decision log — append-only JSONL audit trail for AI coach decisions.

Every coach consultation is logged with full context: metrics, decision,
guardrail messages, applied changes, and API latency. Emergency events
and phase transitions are also logged.

Format: one JSON object per line (JSONL). No pretty-printing.

Adapted from multi_robot_training/ai_trainer/decision_log.py.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone


class DecisionLog:
    """Append-only JSONL audit trail for training decisions.

    Args:
        log_path: Path to the JSONL log file.
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def _append(self, entry: dict) -> None:
        """Append a single JSON entry to the log file."""
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_decision(
        self,
        iteration: int,
        metrics: dict,
        decision: object,
        guardrail_msgs: list[str],
        applied_changes: dict,
        api_latency_ms: float = 0.0,
    ) -> None:
        """Log a coach consultation.

        Args:
            iteration: Current training iteration.
            metrics: Dict of key metrics (from snapshot).
            decision: CoachDecision object.
            guardrail_msgs: List of guardrail messages.
            applied_changes: Dict of {term: (old, new)} applied changes.
            api_latency_ms: Claude API response time in milliseconds.
        """
        self._append({
            "type": "decision",
            "iteration": iteration,
            "metrics": metrics,
            "decision": {
                "action": decision.action,
                "reasoning": decision.reasoning,
                "weight_changes": decision.weight_changes,
                "lr_change": decision.lr_change,
                "confidence": decision.confidence,
            },
            "guardrail_messages": guardrail_msgs,
            "applied_changes": {
                k: {"old": v[0], "new": v[1]}
                for k, v in applied_changes.items()
            },
            "api_latency_ms": api_latency_ms,
        })

    def log_emergency(
        self, iteration: int, action: str, details: str
    ) -> None:
        """Log an emergency event.

        Args:
            iteration: Current training iteration.
            action: Emergency action taken (e.g., "halve_lr", "nan_halt").
            details: Description of the emergency.
        """
        self._append({
            "type": "emergency",
            "iteration": iteration,
            "action": action,
            "details": details,
        })

    def log_phase_transition(
        self, iteration: int, from_phase: str, to_phase: str, checkpoint: str
    ) -> None:
        """Log a phase transition event.

        Args:
            iteration: Current training iteration.
            from_phase: Previous phase name.
            to_phase: Next phase name.
            checkpoint: Checkpoint path saved at transition.
        """
        self._append({
            "type": "phase_transition",
            "iteration": iteration,
            "from_phase": from_phase,
            "to_phase": to_phase,
            "checkpoint": checkpoint,
        })

    def get_recent(self, n: int = 5) -> list[dict]:
        """Read the last N entries from the log.

        Args:
            n: Number of entries to return.

        Returns:
            List of the most recent log entries.
        """
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
