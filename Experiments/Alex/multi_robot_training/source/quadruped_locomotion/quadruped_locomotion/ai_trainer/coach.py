"""AI coach — LLM-powered training decision engine.

Calls Claude API with training metrics and returns structured decisions
about reward weight adjustments, LR changes, noise changes, or phase advances.
Supports optional VLM mode: when a rendered frame is provided, the coach
receives a multimodal message (image + text) for visual gait analysis.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import anthropic

from quadruped_locomotion.ai_trainer.prompt_builder import build_system_prompt, build_user_message

if TYPE_CHECKING:
    from quadruped_locomotion.ai_trainer.config import CoachConfig, PhaseConfig
    from quadruped_locomotion.ai_trainer.metrics import MetricsSnapshot


@dataclass
class CoachDecision:
    """Structured decision from the AI coach."""
    action: str = "no_change"      # no_change, adjust_weights, adjust_noise,
                                    # adjust_lr, advance_phase, emergency_stop
    reasoning: str = ""
    weight_changes: dict = field(default_factory=dict)
    lr_change: float | None = None
    noise_change: float | None = None
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "reasoning": self.reasoning,
            "weight_changes": self.weight_changes,
            "lr_change": self.lr_change,
            "noise_change": self.noise_change,
            "confidence": self.confidence,
        }


class Coach:
    """LLM-powered training coach that analyzes metrics and returns decisions."""

    def __init__(self, coach_cfg: CoachConfig, phase_cfg: PhaseConfig,
                 api_key: str | None = None, vision_enabled: bool = False):
        self.coach_cfg = coach_cfg
        self.phase_cfg = phase_cfg
        self.client = anthropic.Anthropic(api_key=api_key)
        self._passive_mode = False
        self._vision_enabled = vision_enabled
        self.system_prompt = build_system_prompt(
            coach_cfg, phase_cfg, passive_mode=False, vision_enabled=vision_enabled)
        self._consecutive_failures = 0
        self._max_failures = 3

    def set_passive_mode(self, passive: bool):
        """Enable/disable passive mode (biased toward no_change)."""
        self._passive_mode = passive
        self.system_prompt = build_system_prompt(
            self.coach_cfg, self.phase_cfg, passive_mode=passive,
            vision_enabled=self._vision_enabled)

    def update_phase(self, phase_cfg: PhaseConfig):
        """Update the coach's phase config (after phase transition)."""
        self.phase_cfg = phase_cfg
        self.system_prompt = build_system_prompt(
            self.coach_cfg, phase_cfg, passive_mode=self._passive_mode,
            vision_enabled=self._vision_enabled)

    def get_decision(
        self,
        snapshot: MetricsSnapshot,
        recent_history: list[MetricsSnapshot],
        recent_decisions: list[dict],
        plateau_detected: bool = False,
        frame_png: bytes | None = None,
    ) -> tuple[CoachDecision, float]:
        """Call the AI coach and get a training decision.

        Args:
            snapshot: Current metrics snapshot.
            recent_history: Recent MetricsSnapshot objects.
            recent_decisions: Recent decision log entries.
            plateau_detected: Whether terrain level has plateaued.
            frame_png: Optional PNG image bytes from simulation render.
                       When provided, the coach receives a multimodal
                       message with the frame for visual gait analysis.

        Returns:
            (CoachDecision, api_latency_ms)
        """
        user_message = build_user_message(
            snapshot, recent_history, recent_decisions, plateau_detected)

        # Build content: multimodal (image + text) when frame provided,
        # plain text otherwise
        if frame_png is not None:
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(frame_png).decode(),
                    },
                },
                {"type": "text", "text": user_message},
            ]
        else:
            content = user_message

        max_retries = 3
        last_error = None
        start = time.time()

        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.coach_cfg.api_model,
                    max_tokens=1024,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": content}],
                    timeout=60.0,
                )
                latency_ms = (time.time() - start) * 1000
                self._consecutive_failures = 0

                # Parse response
                text = response.content[0].text.strip()
                if not text:
                    if attempt < max_retries - 1:
                        print(f"[AI-COACH] Empty response, retry {attempt + 2}/{max_retries}...")
                        time.sleep(2)
                        continue
                    raise ValueError("Empty response from API after all retries")

                # Strip markdown code fences if present
                if text.startswith("```"):
                    text = text.split("\n", 1)[1]
                    if text.endswith("```"):
                        text = text[:-3]
                    text = text.strip()

                decision_dict = json.loads(text)
                decision = CoachDecision(
                    action=decision_dict.get("action", "no_change"),
                    reasoning=decision_dict.get("reasoning", ""),
                    weight_changes=decision_dict.get("weight_changes", {}),
                    lr_change=decision_dict.get("lr_change"),
                    noise_change=decision_dict.get("noise_change"),
                    confidence=decision_dict.get("confidence", 0.5),
                )
                return decision, latency_ms

            except anthropic.APITimeoutError as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"[AI-COACH] Timeout, retry {attempt + 2}/{max_retries}...")
                    time.sleep(2)
                    continue
                latency_ms = (time.time() - start) * 1000
                self._consecutive_failures += 1
                print(f"[AI-COACH] API TIMEOUT after {max_retries} attempts "
                      f"({self._consecutive_failures}/{self._max_failures})")
                return CoachDecision(
                    action="no_change",
                    reasoning=f"API timeout: {e}",
                ), latency_ms

            except (anthropic.APIError, json.JSONDecodeError, KeyError,
                    IndexError, ValueError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"[AI-COACH] {type(e).__name__}, retry {attempt + 2}/{max_retries}...")
                    time.sleep(2)
                    continue
                latency_ms = (time.time() - start) * 1000
                self._consecutive_failures += 1
                print(f"[AI-COACH] API error after {max_retries} attempts "
                      f"({self._consecutive_failures}/{self._max_failures}): {e}")

                if self._consecutive_failures >= self._max_failures:
                    print("[AI-COACH] Too many failures, falling back to "
                          "no_change for this session")

                return CoachDecision(
                    action="no_change",
                    reasoning=f"API error: {e}",
                ), latency_ms

    @property
    def is_available(self) -> bool:
        """Whether the coach is available (not in failure fallback)."""
        return self._consecutive_failures < self._max_failures
