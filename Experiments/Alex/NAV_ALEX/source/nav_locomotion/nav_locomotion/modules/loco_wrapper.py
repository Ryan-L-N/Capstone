"""Frozen Phase B locomotion policy wrapper.

Loads a trained locomotion checkpoint and runs inference-only forward passes
to convert velocity commands [vx, vy, wz] into 12-dim joint position targets.

Architecture auto-detection:
    Reads the first actor layer weight shape to determine the network architecture.
    Supports both [512, 256, 128] (hybrid no-coach) and [1024, 512, 256] (AI-coached v8).

Velocity command injection:
    Nav policy outputs are injected at obs indices [9, 10, 11] in the Spot loco
    observation vector (after base_lin_vel[3] + base_ang_vel[3] + projected_gravity[3]).

Usage:
    loco = FrozenLocoPolicy.from_checkpoint("checkpoints/model_10600.pt", device="cuda")
    joint_actions = loco(loco_obs, vel_cmd)  # vel_cmd shape (N, 3)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Spot observation/action dimensions from Phase B
SPOT_LOCO_OBS_DIM = 235  # 48 proprio + 187 height scan
SPOT_ACTION_DIM = 12  # 3 joints x 4 legs
SPOT_VEL_CMD_INDICES = (9, 10, 11)  # vx, vy, wz in loco obs vector

# Known architectures by first hidden layer size
KNOWN_ARCHITECTURES = {
    512: [512, 256, 128],   # Hybrid no-coach (MH-2a)
    1024: [1024, 512, 256],  # AI-coached v8 (Trial 11l)
}


class FrozenLocoPolicy(nn.Module):
    """Frozen Phase B locomotion policy for hierarchical navigation.

    Loads actor weights from a Phase B checkpoint, freezes all parameters,
    and provides inference-only forward pass. The critic is discarded entirely.

    Args:
        checkpoint_path: Path to Phase B .pt checkpoint file.
        obs_dim: Loco observation dimension. Default 235 (Spot).
        action_dim: Joint action dimension. Default 12 (Spot).
        vel_cmd_indices: Tuple of obs indices for velocity commands. Default (9, 10, 11).
        device: Torch device. Default "cuda".
    """

    def __init__(
        self,
        checkpoint_path: str,
        obs_dim: int = SPOT_LOCO_OBS_DIM,
        action_dim: int = SPOT_ACTION_DIM,
        vel_cmd_indices: tuple[int, ...] = SPOT_VEL_CMD_INDICES,
        device: str = "cuda",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.vel_cmd_indices = vel_cmd_indices

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Extract actor-only keys (strip "actor." prefix)
        actor_state = {}
        for key, value in state_dict.items():
            if key.startswith("actor."):
                actor_state[key[len("actor."):]] = value

        if not actor_state:
            raise ValueError(
                f"No 'actor.*' keys found in checkpoint {checkpoint_path}. "
                f"Available keys: {list(state_dict.keys())[:10]}..."
            )

        # Auto-detect architecture from first layer weight shape
        first_layer_key = "0.weight"
        if first_layer_key not in actor_state:
            raise ValueError(
                f"Expected key '{first_layer_key}' in actor state dict. "
                f"Available: {list(actor_state.keys())}"
            )

        first_hidden = actor_state[first_layer_key].shape[0]
        if first_hidden in KNOWN_ARCHITECTURES:
            hidden_dims = KNOWN_ARCHITECTURES[first_hidden]
            print(f"[LocoWrapper] Auto-detected architecture: {hidden_dims}")
        else:
            raise ValueError(
                f"Unknown architecture: first hidden dim = {first_hidden}. "
                f"Known: {list(KNOWN_ARCHITECTURES.keys())}"
            )

        # Build actor network matching checkpoint architecture
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.actor = nn.Sequential(*layers)

        # Load weights (strict=True — architecture must match exactly)
        self.actor.load_state_dict(actor_state, strict=True)

        # Freeze everything — inference only
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        self.to(device)

        print(
            f"[LocoWrapper] Loaded frozen loco policy from {checkpoint_path} "
            f"({sum(p.numel() for p in self.parameters()):,} params, "
            f"arch={hidden_dims}, device={device})"
        )

    @torch.no_grad()
    def forward(
        self, loco_obs: torch.Tensor, vel_cmd: torch.Tensor
    ) -> torch.Tensor:
        """Run frozen loco policy with injected velocity commands.

        Args:
            loco_obs: Full loco observation vector, shape (N, obs_dim).
            vel_cmd: Velocity commands [vx, vy, wz], shape (N, 3).

        Returns:
            Joint position actions, shape (N, action_dim).
        """
        obs = loco_obs.clone()
        for i, idx in enumerate(self.vel_cmd_indices):
            obs[:, idx] = vel_cmd[:, i]
        return self.actor(obs)

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs) -> "FrozenLocoPolicy":
        """Convenience constructor.

        Args:
            path: Path to Phase B checkpoint.
            **kwargs: Additional arguments passed to __init__.

        Returns:
            FrozenLocoPolicy instance.
        """
        return cls(checkpoint_path=path, **kwargs)
