"""AI Coach — Claude-powered decision engine for navigation training.

Calls Claude Sonnet API every N iterations to analyze training metrics
and recommend reward weight adjustments. Text-only (no VLM — H100 has no
Vulkan for rendering frames).

Decisions: no_change | adjust_weights | adjust_lr | emergency_stop

Adapted from multi_robot_training/ai_trainer/coach.py for navigation-specific
reward terms and anti-crawl monitoring.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from nav_locomotion.ai_coach.prompt_builder import build_system_prompt, build_user_message


@dataclass
class CoachDecision:
    """Structured decision from the AI coach.

    Attributes:
        action: Decision type — what to do.
        reasoning: Natural language explanation of why.
        weight_changes: Dict of {reward_term: new_weight} if action == "adjust_weights".
        lr_change: New learning rate if action == "adjust_lr".
        confidence: Coach's self-assessed confidence 0.0–1.0.
    """
    action: str = "no_change"
    reasoning: str = ""
    weight_changes: dict[str, float] = field(default_factory=dict)
    lr_change: float | None = None
    confidence: float = 0.5


# Default: no_change with explanation
NO_CHANGE_FALLBACK = CoachDecision(
    action="no_change",
    reasoning="API unavailable — defaulting to no changes",
    confidence=0.0,
)


class Coach:
    """LLM-powered training coach using Claude Sonnet API.

    Analyzes metrics snapshots and returns structured decisions about
    reward weight adjustments. Implements failure resilience: 3 consecutive
    API failures -> fallback to no_change for rest of session.

    Args:
        api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        model: Claude model ID. Default claude-sonnet-4-20250514.
        max_tokens: Max response tokens. Default 1024.
        timeout: API timeout in seconds. Default 60.
        weight_bounds: Dict of {term: (min, max)} for coach-adjustable weights.
        max_weight_changes: Max simultaneous weight changes (Trial 11k lesson). Default 3.
        max_weight_delta_pct: Max percentage change per weight. Default 0.20.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        timeout: float = 60.0,
        weight_bounds: dict[str, tuple[float, float]] | None = None,
        max_weight_changes: int = 3,
        max_weight_delta_pct: float = 0.20,
    ):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.weight_bounds = weight_bounds or {}
        self.max_weight_changes = max_weight_changes
        self.max_weight_delta_pct = max_weight_delta_pct

        # Build system prompt
        self.system_prompt = build_system_prompt(
            weight_bounds=self.weight_bounds,
            max_changes=self.max_weight_changes,
            max_delta_pct=self.max_weight_delta_pct,
        )

        # Failure tracking
        self._consecutive_failures = 0
        self._max_failures = 3

    @property
    def is_available(self) -> bool:
        """Whether the coach API is still available (< 3 consecutive failures)."""
        return self._consecutive_failures < self._max_failures

    def get_decision(
        self,
        snapshot: object,
        recent_history: list,
        recent_decisions: list[CoachDecision],
        plateau_detected: bool = False,
    ) -> tuple[CoachDecision, float]:
        """Query Claude for a training decision.

        Args:
            snapshot: Current MetricsSnapshot with all training metrics.
            recent_history: List of recent MetricsSnapshot objects.
            recent_decisions: List of recent CoachDecision objects.
            plateau_detected: Whether terrain level has stalled.

        Returns:
            (decision, api_latency_ms) tuple.
        """
        if not self.is_available:
            return NO_CHANGE_FALLBACK, 0.0

        # Build user message from metrics
        user_msg = build_user_message(
            snapshot=snapshot,
            recent_history=recent_history,
            recent_decisions=recent_decisions,
            plateau_detected=plateau_detected,
        )

        t0 = time.time()
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            latency_ms = (time.time() - t0) * 1000

            # Parse response
            text = response.content[0].text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            text = text.strip()

            data = json.loads(text)
            decision = CoachDecision(
                action=data.get("action", "no_change"),
                reasoning=data.get("reasoning", ""),
                weight_changes=data.get("weight_changes", {}),
                lr_change=data.get("lr_change"),
                confidence=data.get("confidence", 0.5),
            )

            self._consecutive_failures = 0
            return decision, latency_ms

        except json.JSONDecodeError as e:
            self._consecutive_failures += 1
            latency_ms = (time.time() - t0) * 1000
            print(f"[AI-COACH] JSON parse error: {e}")
            return CoachDecision(
                action="no_change",
                reasoning=f"JSON parse error: {e}",
                confidence=0.0,
            ), latency_ms

        except Exception as e:
            self._consecutive_failures += 1
            latency_ms = (time.time() - t0) * 1000
            print(f"[AI-COACH] API error ({self._consecutive_failures}/{self._max_failures}): {e}")
            return CoachDecision(
                action="no_change",
                reasoning=f"API error: {e}",
                confidence=0.0,
            ), latency_ms
