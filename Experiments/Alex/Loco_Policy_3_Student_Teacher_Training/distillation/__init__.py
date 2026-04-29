"""6-expert distillation system.

MultiExpertRouter: Learned attention MLP (235->64->6 softmax) for expert routing
DistillationLoss: MSE + KL with numerical stability clamping
S2RDistillConfig: Hyperparameters for the distillation pipeline
"""

from .multi_expert_router import MultiExpertRouter
from .distillation_loss import DistillationLoss
from .config import S2RDistillConfig

__all__ = ["MultiExpertRouter", "DistillationLoss", "S2RDistillConfig"]
