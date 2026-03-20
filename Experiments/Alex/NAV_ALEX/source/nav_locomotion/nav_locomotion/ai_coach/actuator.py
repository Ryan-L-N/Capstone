"""Actuator — applies validated coach decisions to live training.

Modifies reward weights, learning rate, and noise at runtime without
restarting the training process. All changes take effect on the next
forward pass / optimization step.

Adapted from multi_robot_training/ai_trainer/actuator.py.
"""

from __future__ import annotations

import torch


class Actuator:
    """Applies guardrail-approved changes to the live env and runner.

    Args:
        env: Isaac Lab environment (or NavEnvWrapper).
        runner: RSL-RL OnPolicyRunner.
    """

    def __init__(self, env, runner):
        self.env = env
        self.runner = runner
        self._noise_bounds = None

    def apply_weight_changes(self, changes: dict[str, float]) -> dict[str, tuple[float, float]]:
        """Modify reward weights in the live environment.

        Args:
            changes: Dict of {term_name: new_weight}.

        Returns:
            Dict of {term_name: (old_weight, new_weight)} for logging.
        """
        applied = {}

        # Access reward manager — may be on wrapper or unwrapped env
        reward_manager = getattr(self.env, "reward_manager", None)
        if reward_manager is None:
            env_unwrapped = getattr(self.env, "unwrapped", self.env)
            reward_manager = getattr(env_unwrapped, "reward_manager", None)

        if reward_manager is None:
            print("[ACTUATOR] WARNING: No reward_manager found — cannot apply weight changes")
            return applied

        # Isaac Lab RewardManager: _term_cfgs list + _term_names list
        term_names = getattr(reward_manager, "_term_names", [])
        term_cfgs = getattr(reward_manager, "_term_cfgs", [])

        for term_name, new_weight in changes.items():
            try:
                idx = term_names.index(term_name)
                old_weight = term_cfgs[idx].weight
                term_cfgs[idx].weight = new_weight
                applied[term_name] = (old_weight, new_weight)
                print(f"[ACTUATOR] {term_name}: {old_weight:.4f} -> {new_weight:.4f}")
            except (ValueError, IndexError):
                print(f"[ACTUATOR] WARNING: term '{term_name}' not found in reward manager")

        return applied

    def apply_lr_change(self, new_lr: float) -> float:
        """Change the optimizer learning rate.

        Args:
            new_lr: New learning rate.

        Returns:
            Old learning rate.
        """
        old_lr = self.runner.alg.optimizer.param_groups[0]["lr"]
        for pg in self.runner.alg.optimizer.param_groups:
            pg["lr"] = new_lr
        print(f"[ACTUATOR] LR: {old_lr:.2e} -> {new_lr:.2e}")
        return old_lr

    def emergency_halve_lr(self) -> float:
        """Emergency: halve the learning rate immediately.

        Returns:
            New (halved) learning rate.
        """
        old_lr = self.runner.alg.optimizer.param_groups[0]["lr"]
        new_lr = old_lr / 2.0
        for pg in self.runner.alg.optimizer.param_groups:
            pg["lr"] = new_lr
        print(f"[ACTUATOR] EMERGENCY: LR halved {old_lr:.2e} -> {new_lr:.2e}")
        return new_lr

    def register_noise_control(self, bounds_dict: dict) -> None:
        """Register mutable noise bounds dict for runtime noise control.

        Args:
            bounds_dict: Dict with "min" and "max" keys (mutable, shared with training loop).
        """
        self._noise_bounds = bounds_dict

    def apply_noise_change(self, new_max_noise: float) -> None:
        """Update the maximum noise standard deviation.

        Args:
            new_max_noise: New max noise std.
        """
        if self._noise_bounds is not None:
            old = self._noise_bounds["max"]
            self._noise_bounds["max"] = new_max_noise
            print(f"[ACTUATOR] Max noise: {old:.3f} -> {new_max_noise:.3f}")

    def save_checkpoint(self, path: str) -> bool:
        """Save a checkpoint and verify it's not corrupted.

        Args:
            path: File path for the checkpoint.

        Returns:
            True if checkpoint is clean, False if corrupted.
        """
        self.runner.save(path)

        # Verify — load and check all tensors for NaN/Inf
        try:
            ckpt = torch.load(path, weights_only=False, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            for key, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        print(f"[ACTUATOR] CORRUPTED checkpoint: NaN/Inf in '{key}'")
                        return False
            return True
        except Exception as e:
            print(f"[ACTUATOR] Checkpoint verification failed: {e}")
            return False
