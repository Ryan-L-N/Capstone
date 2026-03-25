"""Applies validated AI coach decisions to the live training system.

Modifies reward weights, learning rate, and noise bounds at runtime
without requiring training restart.
"""

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class Actuator:
    """Applies parameter changes to the live Isaac Lab env and RSL-RL runner."""

    def __init__(self, env, runner):
        self.env = env
        self.runner = runner
        self._noise_bounds = None  # set by register_noise_control()

    def apply_weight_changes(self, changes: dict[str, float]) -> dict[str, tuple[float, float]]:
        """Change reward weights at runtime via the reward manager.

        Args:
            changes: Dict of {term_name: new_weight}.

        Returns:
            Dict of {term_name: (old_weight, new_weight)} for logging.
        """
        applied = {}
        env_unwrapped = self.env.unwrapped
        rm = env_unwrapped.reward_manager

        for name, new_weight in changes.items():
            old_weight = None
            # Isaac Lab RewardManager: _term_cfgs is a list, names in _term_names
            if hasattr(rm, "_term_names") and hasattr(rm, "_term_cfgs"):
                if name in rm._term_names:
                    idx = rm._term_names.index(name)
                    old_weight = rm._term_cfgs[idx].weight
                    rm._term_cfgs[idx].weight = new_weight
            # Fallback: dict-style API (older versions)
            elif hasattr(rm, "_term_cfgs") and hasattr(rm._term_cfgs, "items"):
                if name in rm._term_cfgs:
                    old_weight = rm._term_cfgs[name].weight
                    rm._term_cfgs[name].weight = new_weight

            if old_weight is not None:
                applied[name] = (old_weight, new_weight)

        return applied

    def apply_lr_change(self, new_lr: float) -> float:
        """Change learning rate via optimizer param groups.

        Returns:
            The old learning rate.
        """
        old_lr = self.runner.alg.optimizer.param_groups[0]["lr"]
        for param_group in self.runner.alg.optimizer.param_groups:
            param_group["lr"] = new_lr
        return old_lr

    def apply_noise_change(self, new_max_noise: float):
        """Change the max noise std bound.

        This requires the noise bounds to be stored in a mutable container
        registered via register_noise_control().
        """
        if self._noise_bounds is not None:
            self._noise_bounds["max"] = new_max_noise

    def register_noise_control(self, bounds_dict: dict):
        """Register the mutable noise bounds dict used by the safety clamp.

        Args:
            bounds_dict: Dict with 'min' and 'max' keys, shared with
                         the safety clamp hook in the training loop.
        """
        self._noise_bounds = bounds_dict

    def save_checkpoint(self, path: str) -> bool:
        """Save and verify a checkpoint.

        Returns:
            True if checkpoint is clean (no NaN).
        """
        self.runner.save(path)
        # Verify checkpoint integrity
        loaded = torch.load(path, weights_only=False, map_location="cpu")
        state = loaded.get("model_state_dict", loaded)
        for key, tensor in state.items():
            if isinstance(tensor, torch.Tensor):
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"[AI-COACH] CORRUPTED checkpoint: {key} has NaN/Inf!")
                    return False
        return True

    def emergency_halve_lr(self) -> float:
        """Emergency: halve learning rate immediately."""
        old_lr = self.runner.alg.optimizer.param_groups[0]["lr"]
        new_lr = old_lr / 2.0
        for param_group in self.runner.alg.optimizer.param_groups:
            param_group["lr"] = new_lr
        print(f"[AI-COACH] EMERGENCY: LR halved {old_lr:.2e} -> {new_lr:.2e}")
        return new_lr
