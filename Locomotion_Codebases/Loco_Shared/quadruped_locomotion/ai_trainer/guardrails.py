"""Safety guardrails for AI coach decisions.

Every decision from the LLM must pass through these checks before
being applied. This is the final gatekeeper between AI judgment
and a multi-day GPU training run.

Encodes all Bug Museum lessons as hard constraints.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quadruped_locomotion.ai_trainer.config import CoachConfig, PhaseConfig

# Weights that must NEVER be changed from their fixed values
FROZEN_WEIGHTS = {
    "stumble": 0.0,              # Bug #28b: world-frame Z
    "body_height_tracking": 0.0, # Bug #22: world-frame Z
}

# Sign constraints — positive rewards stay positive, penalties stay negative
SIGN_POSITIVE = {
    "air_time", "base_angular_velocity", "base_linear_velocity",
    "foot_clearance", "gait", "velocity_modulation",
}
SIGN_NEGATIVE = {
    "action_smoothness", "air_time_variance", "base_motion",
    "base_orientation", "body_scraping", "contact_force_smoothness",
    "dof_pos_limits", "foot_slip", "joint_acc", "joint_pos",
    "joint_torques", "joint_vel", "terrain_relative_height",
    "undesired_contacts", "vegetation_drag",
}


@dataclass
class GuardrailResult:
    """Result of guardrail validation."""
    approved: bool
    modifications: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)
    emergency: str | None = None


class Guardrails:
    """Validates and bounds every AI coach decision."""

    def __init__(self, coach_cfg: CoachConfig, phase_cfg: PhaseConfig):
        self.coach_cfg = coach_cfg
        self.phase_cfg = phase_cfg

    def validate_weight_changes(
        self,
        changes: dict[str, float],
        current_weights: dict[str, float],
        current_terrain_level: float = 0.0,
    ) -> tuple[dict[str, float], list[str]]:
        """Validate and bound proposed weight changes.

        Args:
            current_terrain_level: Used for terrain-gated penalty loosening.
                Penalties cannot be loosened (made less negative) below the
                penalty_loosen_terrain threshold. The robot must learn clean
                gait under strict constraints before getting freedom.

        Returns:
            (approved_changes, list_of_messages)
        """
        approved = {}
        messages = []

        # Check count limit (11k lesson: max 3 at a time)
        if len(changes) > self.coach_cfg.max_weight_changes:
            messages.append(
                f"REJECTED: {len(changes)} changes exceeds max "
                f"{self.coach_cfg.max_weight_changes}. Keeping first "
                f"{self.coach_cfg.max_weight_changes}.")
            # Keep only the first N
            keys = list(changes.keys())[:self.coach_cfg.max_weight_changes]
            changes = {k: changes[k] for k in keys}

        for name, new_val in changes.items():
            # Frozen weights — never touch
            if name in FROZEN_WEIGHTS:
                messages.append(
                    f"REJECTED {name}: frozen at {FROZEN_WEIGHTS[name]} "
                    f"(Bug #22/#28b)")
                continue

            # Must be in phase's allowed set
            if name in self.phase_cfg.frozen_weights:
                messages.append(
                    f"REJECTED {name}: frozen for phase {self.phase_cfg.name}")
                continue

            # Sign constraint
            if name in SIGN_POSITIVE and new_val < 0:
                messages.append(
                    f"REJECTED {name}: positive reward cannot be negative")
                continue
            if name in SIGN_NEGATIVE and new_val > 0:
                messages.append(
                    f"REJECTED {name}: penalty cannot be positive")
                continue

            # Terrain-gated penalty loosening — penalties cannot be made
            # less negative until terrain exceeds threshold (default 4.0).
            # The robot must learn clean gait under strict constraints first.
            loosen_threshold = getattr(
                self.coach_cfg, "penalty_loosen_terrain", 4.0)
            if (name in SIGN_NEGATIVE
                    and current_terrain_level < loosen_threshold):
                current = current_weights.get(name)
                if current is not None and new_val > current:
                    messages.append(
                        f"REJECTED {name}: cannot loosen penalty "
                        f"({current:.4f} -> {new_val:.4f}) at terrain "
                        f"{current_terrain_level:.2f} < {loosen_threshold:.1f}. "
                        f"Robot must master clean gait first.")
                    continue

            # Absolute bounds — use tighter mason_hybrid bounds if available
            bounds = None
            if self.phase_cfg.name == "mason_hybrid" and hasattr(self.coach_cfg, "mason_hybrid_bounds"):
                bounds = self.coach_cfg.mason_hybrid_bounds.get(name)
            if bounds is None:
                bounds = self.coach_cfg.weight_bounds.get(name)
            if bounds:
                lo, hi = bounds
                if new_val < lo:
                    messages.append(
                        f"CLAMPED {name}: {new_val:.4f} -> {lo:.4f} (min)")
                    new_val = lo
                elif new_val > hi:
                    messages.append(
                        f"CLAMPED {name}: {new_val:.4f} -> {hi:.4f} (max)")
                    new_val = hi

            # Delta check — max 20% change or 0.5 absolute
            current = current_weights.get(name)
            if current is not None and current != 0.0:
                delta_pct = abs(new_val - current) / abs(current)
                if delta_pct > self.coach_cfg.max_weight_delta_pct:
                    # Clamp to max delta
                    direction = 1.0 if new_val > current else -1.0
                    max_delta = abs(current) * self.coach_cfg.max_weight_delta_pct
                    clamped = current + direction * max_delta
                    messages.append(
                        f"BOUNDED {name}: {new_val:.4f} -> {clamped:.4f} "
                        f"(max {self.coach_cfg.max_weight_delta_pct:.0%} change)")
                    new_val = clamped
            elif current is not None and current == 0.0:
                # Weight is currently 0 — only allow if it's not frozen
                if abs(new_val) > self.coach_cfg.max_weight_delta_abs:
                    direction = 1.0 if new_val > 0 else -1.0
                    new_val = direction * self.coach_cfg.max_weight_delta_abs
                    messages.append(
                        f"BOUNDED {name}: clamped to {new_val:.4f} "
                        f"(from zero, max abs {self.coach_cfg.max_weight_delta_abs})")

            approved[name] = new_val

        return approved, messages

    def validate_lr_change(
        self, new_lr: float | None
    ) -> tuple[float | None, list[str]]:
        """Validate proposed learning rate change."""
        if new_lr is None:
            return None, []

        # LR changes disabled (e.g. mason_hybrid uses adaptive KL schedule)
        if not self.coach_cfg.lr_change_enabled:
            return None, ["REJECTED LR change: disabled for this run (adaptive schedule manages LR)"]

        messages = []
        phase_max = self.coach_cfg.phase_lr_limits.get(
            self.phase_cfg.name, 3e-5)

        if new_lr > phase_max:
            messages.append(
                f"CLAMPED LR: {new_lr:.2e} -> {phase_max:.2e} "
                f"(phase {self.phase_cfg.name} max)")
            new_lr = phase_max

        if new_lr < 1e-6:
            messages.append(f"CLAMPED LR: {new_lr:.2e} -> 1e-6 (min)")
            new_lr = 1e-6

        return new_lr, messages

    def validate_noise_change(
        self, new_noise: float | None
    ) -> tuple[float | None, list[str]]:
        """Validate proposed noise std change."""
        if new_noise is None:
            return None, []

        # Noise changes disabled (e.g. mason_hybrid uses adaptive schedule)
        if not self.coach_cfg.noise_change_enabled:
            return None, ["REJECTED noise change: disabled for this run (adaptive schedule manages noise)"]

        messages = []
        if new_noise > self.phase_cfg.max_noise_std:
            messages.append(
                f"CLAMPED noise: {new_noise:.2f} -> "
                f"{self.phase_cfg.max_noise_std:.2f} (phase max)")
            new_noise = self.phase_cfg.max_noise_std

        if new_noise < 0.2:
            messages.append(f"CLAMPED noise: {new_noise:.2f} -> 0.2 (min)")
            new_noise = 0.2

        return new_noise, messages

    def check_emergency(self, snapshot) -> str | None:
        """Check for emergency conditions that override the coach.

        Returns emergency action string or None.
        """
        if snapshot.has_nan:
            return "nan_rollback"

        if snapshot.value_loss > self.coach_cfg.emergency_value_loss:
            return "halve_lr"

        # Action smoothness explosion precursor (seen in 11h/11i)
        smoothness = snapshot.reward_breakdown.get("action_smoothness", 0.0)
        if smoothness < self.coach_cfg.emergency_smoothness:
            return "emergency_stop"

        return None

    def validate_phase_advance(
        self, go: bool, failures: list[str]
    ) -> tuple[bool, list[str]]:
        """Final check on phase advancement decision."""
        messages = []
        if not go:
            messages.append(
                f"BLOCKED phase advance: {', '.join(failures)}")
        return go, messages
