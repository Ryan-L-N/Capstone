"""Guardrails — safety validation for AI coach decisions.

Final gatekeeper between Claude's suggestions and the live training run.
Implements all Bug Museum lessons as hard constraints:
    - Max 3 weight changes at a time (Trial 11k: 6 changes -> total collapse)
    - Max 20% delta per change
    - Sign constraints (positive stays positive, negative stays negative)
    - Terrain-gated penalty loosening (penalties can't loosen until terrain >= 3.0)
    - Absolute bounds per weight
    - Emergency checks: NaN -> halt, value_loss > 100 -> halve LR

Adapted from multi_robot_training/ai_trainer/guardrails.py for navigation rewards.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Default weight bounds for navigation reward terms
# (min, max) — coach cannot push weights outside these ranges
NAV_WEIGHT_BOUNDS: dict[str, tuple[float, float]] = {
    "forward_velocity": (3.0, 15.0),
    "survival": (0.5, 3.0),
    "terrain_traversal": (0.5, 5.0),
    "terrain_relative_height": (-5.0, -1.0),
    "drag_penalty": (-4.0, -0.5),
    "cmd_smoothness": (-3.0, -0.1),
    "lateral_velocity": (-1.5, -0.05),
    "angular_velocity": (-2.0, -0.1),
    "vegetation_drag": (-0.01, -0.0001),
}


@dataclass
class GuardrailResult:
    """Result of guardrail validation.

    Attributes:
        approved: Whether any changes were approved.
        modifications: List of messages about guardrail-applied modifications.
        rejections: List of messages about rejected changes.
        emergency: Emergency action string if detected, else None.
    """
    approved: bool = True
    modifications: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)
    emergency: str | None = None


class Guardrails:
    """Safety validation for AI coach decisions.

    Args:
        weight_bounds: Dict of {term: (min, max)} bounds. Defaults to NAV_WEIGHT_BOUNDS.
        max_weight_changes: Maximum simultaneous weight changes. Default 3.
        max_weight_delta_pct: Maximum percentage change per weight. Default 0.20 (20%).
        penalty_loosen_terrain: Minimum terrain level before penalties can be loosened. Default 3.0.
        emergency_value_loss: Value loss threshold for emergency LR halving. Default 100.0.
    """

    def __init__(
        self,
        weight_bounds: dict[str, tuple[float, float]] | None = None,
        max_weight_changes: int = 3,
        max_weight_delta_pct: float = 0.20,
        penalty_loosen_terrain: float = 3.0,
        emergency_value_loss: float = 100.0,
    ):
        self.weight_bounds = weight_bounds or NAV_WEIGHT_BOUNDS
        self.max_weight_changes = max_weight_changes
        self.max_weight_delta_pct = max_weight_delta_pct
        self.penalty_loosen_terrain = penalty_loosen_terrain
        self.emergency_value_loss = emergency_value_loss

    def validate_weight_changes(
        self,
        changes: dict[str, float],
        current_weights: dict[str, float],
        current_terrain_level: float,
    ) -> tuple[dict[str, float], list[str]]:
        """Validate and constrain proposed weight changes.

        Applies all safety rules in order:
        1. Count limit (max 3)
        2. Sign constraints
        3. Terrain-gated loosening
        4. Absolute bounds
        5. Delta limit (max 20%)

        Args:
            changes: Proposed {term: new_weight} changes from coach.
            current_weights: Current {term: weight} values.
            current_terrain_level: Current mean terrain difficulty level.

        Returns:
            (approved_changes, messages) tuple.
        """
        messages = []
        approved = {}

        # 1. Count limit
        items = list(changes.items())
        if len(items) > self.max_weight_changes:
            messages.append(
                f"GUARDRAIL: Trimmed {len(items)} changes to {self.max_weight_changes} "
                f"(Trial 11k lesson: max {self.max_weight_changes} at a time)"
            )
            items = items[:self.max_weight_changes]

        for term, new_weight in items:
            current = current_weights.get(term)
            if current is None:
                messages.append(f"REJECTED: '{term}' not in current weights")
                continue

            # 2. Sign constraint — positive must stay positive, negative must stay negative
            if current > 0 and new_weight <= 0:
                messages.append(
                    f"REJECTED: '{term}' sign flip ({current:.3f} -> {new_weight:.3f}). "
                    "Positive rewards cannot become penalties."
                )
                continue
            if current < 0 and new_weight >= 0:
                messages.append(
                    f"REJECTED: '{term}' sign flip ({current:.3f} -> {new_weight:.3f}). "
                    "Penalties cannot become rewards."
                )
                continue

            # 3. Terrain-gated loosening — penalties can't loosen below terrain threshold
            if current < 0 and new_weight > current:  # Loosening = less negative
                if current_terrain_level < self.penalty_loosen_terrain:
                    messages.append(
                        f"REJECTED: '{term}' loosening ({current:.3f} -> {new_weight:.3f}) "
                        f"blocked — terrain {current_terrain_level:.1f} < {self.penalty_loosen_terrain:.1f}. "
                        "Penalties locked until terrain advances."
                    )
                    continue

            # 4. Absolute bounds
            if term in self.weight_bounds:
                lo, hi = self.weight_bounds[term]
                if new_weight < lo:
                    messages.append(
                        f"CLAMPED: '{term}' {new_weight:.3f} -> {lo:.3f} (min bound)"
                    )
                    new_weight = lo
                elif new_weight > hi:
                    messages.append(
                        f"CLAMPED: '{term}' {new_weight:.3f} -> {hi:.3f} (max bound)"
                    )
                    new_weight = hi

            # 5. Delta limit (max 20% change)
            if abs(current) > 0.01:  # Avoid division by near-zero
                delta_pct = abs(new_weight - current) / abs(current)
                if delta_pct > self.max_weight_delta_pct:
                    # Clamp to max delta
                    direction = 1.0 if new_weight > current else -1.0
                    clamped = current + direction * abs(current) * self.max_weight_delta_pct
                    messages.append(
                        f"CLAMPED: '{term}' delta {delta_pct:.1%} -> {self.max_weight_delta_pct:.0%} "
                        f"({current:.3f} -> {clamped:.3f} instead of {new_weight:.3f})"
                    )
                    new_weight = clamped
            else:
                # Near-zero weight: allow max 0.5 absolute change
                if abs(new_weight - current) > 0.5:
                    direction = 1.0 if new_weight > current else -1.0
                    clamped = current + direction * 0.5
                    messages.append(
                        f"CLAMPED: '{term}' near-zero delta to ±0.5 "
                        f"({current:.3f} -> {clamped:.3f})"
                    )
                    new_weight = clamped

            # Re-check bounds after delta clamping
            if term in self.weight_bounds:
                lo, hi = self.weight_bounds[term]
                new_weight = max(lo, min(hi, new_weight))

            approved[term] = new_weight

        return approved, messages

    def validate_lr_change(
        self, new_lr: float, current_lr: float, phase_lr_max: float = 1e-4
    ) -> tuple[float | None, list[str]]:
        """Validate a learning rate change.

        Args:
            new_lr: Proposed new learning rate.
            current_lr: Current learning rate.
            phase_lr_max: Maximum LR for current phase.

        Returns:
            (validated_lr or None, messages) tuple.
        """
        messages = []
        if new_lr > phase_lr_max:
            messages.append(f"CLAMPED: LR {new_lr:.2e} -> {phase_lr_max:.2e} (phase max)")
            new_lr = phase_lr_max
        if new_lr < 1e-6:
            messages.append(f"CLAMPED: LR {new_lr:.2e} -> 1e-6 (minimum)")
            new_lr = 1e-6
        return new_lr, messages

    def check_emergency(self, snapshot) -> str | None:
        """Check for emergency conditions requiring immediate action.

        Args:
            snapshot: MetricsSnapshot with current training state.

        Returns:
            Emergency action string, or None if no emergency.
        """
        if getattr(snapshot, "has_nan", False):
            return "nan_halt"

        if getattr(snapshot, "value_loss", 0) > self.emergency_value_loss:
            return "halve_lr"

        return None
