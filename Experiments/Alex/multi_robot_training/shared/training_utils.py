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
