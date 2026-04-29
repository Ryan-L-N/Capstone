"""Cosine annealing learning rate schedule with linear warmup.

Extracted from 100hr_env_run/train_100hr.py lines 91-118.
Robot-agnostic — pure math utility.

Created for AI2C Tech Capstone — MS for Autonomy, February 2026
"""

import math

from rsl_rl.runners import OnPolicyRunner


def cosine_annealing_lr(
    iteration: int,
    max_iterations: int,
    lr_max: float,
    lr_min: float,
    warmup_iters: int,
) -> float:
    """Compute learning rate with linear warmup + cosine annealing.

    Args:
        iteration: Current training iteration.
        max_iterations: Total training iterations.
        lr_max: Peak learning rate (after warmup).
        lr_min: Minimum learning rate (end of cosine decay).
        warmup_iters: Number of warmup iterations (linear ramp).

    Returns:
        Learning rate for this iteration.
    """
    if iteration < warmup_iters:
        return lr_min + (lr_max - lr_min) * (iteration / warmup_iters)
    else:
        progress = (iteration - warmup_iters) / max(1, max_iterations - warmup_iters)
        return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def set_learning_rate(runner: OnPolicyRunner, lr: float):
    """Override the learning rate in the PPO optimizer."""
    for param_group in runner.alg.optimizer.param_groups:
        param_group["lr"] = lr
