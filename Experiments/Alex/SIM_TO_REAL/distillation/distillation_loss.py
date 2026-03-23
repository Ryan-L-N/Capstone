"""Distillation loss: MSE on action means + KL on action distributions.

Adapted from multi_expert_distillation/distillation_loss.py.
Clamped to prevent gradient explosion (Bug #29 pattern).

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

import torch


class DistillationLoss:
    """Combined MSE + KL distillation loss with numerical stability."""

    def __init__(self, kl_weight: float = 0.1):
        """
        Args:
            kl_weight: Relative weight of KL divergence vs MSE (default 0.1).
        """
        self.kl_weight = kl_weight

    def __call__(
        self,
        student_mean: torch.Tensor,
        student_std: torch.Tensor,
        expert_mean: torch.Tensor,
        expert_std: torch.Tensor,
    ) -> tuple:
        """Compute distillation loss.

        Args:
            student_mean: (N, 12) student policy action means.
            student_std: (N, 12) student policy action stds.
            expert_mean: (N, 12) router-blended expert action means.
            expert_std: (N, 12) router-blended expert action stds.

        Returns:
            (total_loss, mse_value, kl_value) tuple.
        """
        # MSE on action means (primary imitation signal)
        mse = torch.nn.functional.mse_loss(student_mean, expert_mean)

        # KL divergence between diagonal Gaussians
        # KL(student || expert) for diagonal Gaussians:
        #   sum(log(s_e/s_s) + (s_s^2 + (m_s - m_e)^2) / (2*s_e^2) - 0.5)
        expert_var = expert_std.pow(2).clamp(min=1e-6)  # Numerical stability
        student_var = student_std.pow(2).clamp(min=1e-6)

        kl = (
            torch.log(expert_std / student_std.clamp(min=1e-6))
            + (student_var + (student_mean - expert_mean).pow(2)) / (2 * expert_var)
            - 0.5
        )
        kl = kl.sum(dim=-1).mean()

        # Clamp KL to prevent gradient explosion (Bug #29 pattern)
        kl = torch.clamp(kl, 0.0, 10.0)

        total = mse + self.kl_weight * kl
        return total, mse.item(), kl.item()
