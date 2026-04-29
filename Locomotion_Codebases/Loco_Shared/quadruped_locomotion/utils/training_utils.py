"""Common training utilities shared across all training scripts.

Extracted from vision60_training/train_vision60.py lines 164-172.
Robot-agnostic utility functions.

Created for AI2C Tech Capstone — MS for Autonomy, February 2026
"""

import argparse

import torch


def configure_tf32():
    """Enable TF32 for faster matmul on H100/A100."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def clamp_noise_std(policy, min_std: float, max_std: float):
    """Clamp noise std as a safety net — prevent collapse or explosion."""
    with torch.no_grad():
        if hasattr(policy, 'noise_std_type') and policy.noise_std_type == "log":
            log_min = torch.log(torch.tensor(min_std, device=policy.log_std.device))
            log_max = torch.log(torch.tensor(max_std, device=policy.log_std.device))
            policy.log_std.clamp_(min=log_min.item(), max=log_max.item())
        else:
            policy.std.clamp_(min=min_std, max=max_std)


def _sanitize_std(param, min_val: float, max_val: float):
    """Replace NaN/Inf/negative values in a std parameter, then clamp.

    clamp_() alone does NOT fix NaN — NaN.clamp_(min=0.3) is still NaN.
    We must explicitly replace bad values first.
    """
    bad = torch.isnan(param.data) | torch.isinf(param.data) | (param.data < 0)
    if bad.any():
        param.data[bad] = min_val
    param.data.clamp_(min=min_val, max=max_val)


def register_std_safety_clamp(policy, min_std: float, max_std: float):
    """Monkey-patch policy.act() to sanitize std before every forward pass.

    The post-update clamp_noise_std only runs AFTER the full PPO update returns.
    But during the update's 8 epochs × 64 mini-batches, an optimizer step can
    push std to NaN or negative, causing the NEXT policy.act() call to crash with:
        RuntimeError: normal expects all elements of std >= 0.0

    Plain clamp_() does NOT fix NaN values. This fix replaces NaN/Inf/negative
    values with min_std, then clamps to [min_std, max_std].
    """
    original_act = policy.act

    def safe_act(*args, **kwargs):
        with torch.no_grad():
            if hasattr(policy, 'noise_std_type') and policy.noise_std_type == "log":
                _sanitize_std(policy.log_std,
                              torch.log(torch.tensor(min_std)).item(),
                              torch.log(torch.tensor(max_std)).item())
            else:
                _sanitize_std(policy.std, min_std, max_std)
        return original_act(*args, **kwargs)

    policy.act = safe_act


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common training arguments to an ArgumentParser.

    Args:
        parser: Existing argparser to extend.

    Returns:
        The same parser with added arguments.
    """
    parser.add_argument("--num_envs", type=int, default=20480,
                        help="Number of parallel environments (default 20480 for H100)")
    parser.add_argument("--max_iterations", type=int, default=60000,
                        help="Max training iterations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--use_wandb", action="store_true", default=True,
                        help="Enable Weights & Biases logging (default: True)")
    parser.add_argument("--no_wandb", action="store_true", default=False,
                        help="Disable Weights & Biases logging")
    return parser
