"""Distillation loss: MSE on action means + KL on action distributions.

Transfers knowledge from a blended expert ensemble to the student policy.
The MSE term teaches the student WHAT to do. The KL term teaches HOW
CONFIDENT the expert is, so the student learns uncertainty too.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

import torch
import torch.nn.functional as F


class DistillationLoss:
    """Combined MSE + KL distillation loss between student and expert actions."""

    def __init__(self, kl_weight: float = 0.1):
        self.kl_weight = kl_weight

    def __call__(self, student_mean, student_std, expert_mean, expert_std):
        """Compute distillation loss.

        Args:
            student_mean: (N, 12) student action means.
            student_std:  (N, 12) or (12,) student action stds.
            expert_mean:  (N, 12) blended expert action means.
            expert_std:   (N, 12) or (12,) blended expert action stds.

        Returns:
            loss:     scalar combined distillation loss.
            mse_val:  float MSE component (for logging).
            kl_val:   float KL component (for logging).
        """
        # Expand scalar stds to match batch shape
        if student_std.dim() == 1:
            student_std = student_std.unsqueeze(0).expand_as(student_mean)
        if expert_std.dim() == 1:
            expert_std = expert_std.unsqueeze(0).expand_as(expert_mean)

        # MSE on mean actions — the primary signal
        mse_loss = F.mse_loss(student_mean, expert_mean.detach())

        # KL divergence between diagonal Gaussians
        # KL(student || expert) for each dimension, then mean across all
        var_s = student_std.pow(2)
        var_e = expert_std.detach().pow(2).clamp(min=1e-6)

        kl = (
            torch.log(expert_std.detach().clamp(min=1e-6) / student_std.clamp(min=1e-6))
            + (var_s + (student_mean - expert_mean.detach()).pow(2)) / (2 * var_e)
            - 0.5
        )
        kl_loss = kl.mean().clamp(max=10.0)  # Bug #29: clamp unbounded terms

        loss = mse_loss + self.kl_weight * kl_loss

        return loss, mse_loss.item(), kl_loss.item()
