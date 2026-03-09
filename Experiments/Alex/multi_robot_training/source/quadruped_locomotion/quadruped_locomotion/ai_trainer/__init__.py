"""AI-guided RL training for quadruped locomotion.

Wraps the existing Isaac Lab + RSL-RL training loop with an LLM coach
that monitors metrics and adjusts reward weights at runtime.
"""

from quadruped_locomotion.ai_trainer.config import PhaseConfig, CoachConfig, PHASE_CONFIGS
from quadruped_locomotion.ai_trainer.metrics import MetricsCollector, MetricsSnapshot
from quadruped_locomotion.ai_trainer.coach import Coach, CoachDecision
from quadruped_locomotion.ai_trainer.guardrails import Guardrails
from quadruped_locomotion.ai_trainer.actuator import Actuator
from quadruped_locomotion.ai_trainer.decision_log import DecisionLog

__all__ = [
    "PhaseConfig", "CoachConfig", "PHASE_CONFIGS",
    "MetricsCollector", "MetricsSnapshot",
    "Coach", "CoachDecision",
    "Guardrails", "Actuator", "DecisionLog",
]
