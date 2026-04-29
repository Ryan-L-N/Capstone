"""Frozen locomotion policy wrapper for Phase C navigation.

Wraps a trained Phase B locomotion policy as a fixed velocity-to-joint-position
mapping. The nav policy outputs [vx, vy, wz] velocity commands, and this wrapper
converts them to joint position targets using the frozen loco checkpoint.

Architecture:
    Nav Policy (10 Hz) --> [vx, vy, wz] --> LocoWrapper --> [12 joint pos] --> Robot
                                                ^
                                          frozen Phase B
                                          checkpoint

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrozenLocoPolicy(nn.Module):
    """Frozen locomotion policy that maps observations + velocity commands to joint actions.

    Loads a Phase B checkpoint and runs inference without gradients.
    The nav policy's output velocity commands replace the env's velocity command
    in the loco policy's observation vector.

    Args:
        checkpoint_path: Path to Phase B model_XXXX.pt file.
        obs_dim: Loco policy observation dimension (default 235 for Spot).
        action_dim: Joint position action dimension (default 12 for Spot).
        vel_cmd_indices: Indices in obs vector where velocity commands live.
            For Spot: base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) = 9,
            then velocity_commands at indices [9, 10, 11].
        device: Torch device.
    """

    def __init__(
        self,
        checkpoint_path: str,
        obs_dim: int = 235,
        action_dim: int = 12,
        vel_cmd_indices: tuple[int, ...] = (9, 10, 11),
        device: str = "cuda",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.vel_cmd_indices = list(vel_cmd_indices)
        self.device = device

        # Load checkpoint
        loaded = torch.load(checkpoint_path, weights_only=False, map_location=device)
        state_dict = loaded["model_state_dict"]

        # Extract actor weights only (no critic needed for inference)
        actor_state = {k: v for k, v in state_dict.items() if k.startswith("actor.")}
        std_key = "std" if "std" in state_dict else "log_std"
        if std_key in state_dict:
            actor_state[std_key] = state_dict[std_key]

        # Build actor network matching Phase B architecture: [512, 256, 128]
        # Input: obs_dim, Output: action_dim
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
        )

        # Load weights
        actor_only = {k.replace("actor.", ""): v for k, v in actor_state.items()
                      if k.startswith("actor.")}
        self.actor.load_state_dict(actor_only, strict=True)

        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False

        self.to(device)
        self.eval()

        print(f"[LocoWrapper] Loaded frozen loco policy from {checkpoint_path}", flush=True)
        print(f"[LocoWrapper] obs_dim={obs_dim}, action_dim={action_dim}, vel_indices={vel_cmd_indices}", flush=True)

    @torch.no_grad()
    def forward(self, obs: torch.Tensor, vel_cmd: torch.Tensor) -> torch.Tensor:
        """Run loco policy with overridden velocity commands.

        Args:
            obs: Full loco observation vector (N, obs_dim) — from env.
            vel_cmd: Nav policy velocity output (N, 3) — [vx, vy, wz].

        Returns:
            Joint position actions (N, action_dim).
        """
        # Override velocity command slots in observation
        modified_obs = obs.clone()
        for i, idx in enumerate(self.vel_cmd_indices):
            modified_obs[:, idx] = vel_cmd[:, i]

        return self.actor(modified_obs)

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs) -> "FrozenLocoPolicy":
        """Convenience constructor."""
        return cls(checkpoint_path=path, **kwargs)
