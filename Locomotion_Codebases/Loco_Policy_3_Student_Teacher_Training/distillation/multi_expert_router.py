"""Learned attention-weighted router for 6 terrain experts.

For 2 experts, height-scan-variance sigmoid gating works (smooth vs rough).
For 6 experts, we need a learned MLP because height scan variance alone can't
distinguish friction/stairs/boulder/slope/rough specialists — they share terrains.

The gate_net (235 -> 64 -> 6 softmax) trains jointly with the student via
gradient through the differentiable softmax. Expert models remain frozen.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiExpertRouter(nn.Module):
    """Learned attention-weighted routing over 6 frozen terrain experts.

    Input: 235-dim observation
    Output: 6 softmax weights + blended expert action

    The gate_net parameters (~4K) are trained jointly with the student.
    All expert models are frozen (no gradient).
    """

    EXPERT_NAMES = ["friction", "stairs_up", "stairs_down", "boulders", "slopes", "mixed_rough"]

    def __init__(
        self,
        experts: dict,
        num_obs: int = 235,
        num_experts: int = 6,
        hidden_dim: int = 64,
    ):
        """Initialize the multi-expert router.

        Args:
            experts: Dict of {name: frozen ActorCritic model} for each expert.
            num_obs: Observation dimension (must match experts).
            num_experts: Number of experts (default 6).
            hidden_dim: Hidden layer size for gate MLP.
        """
        super().__init__()
        self.experts = experts  # Not nn.Module — just a dict of frozen models
        self.num_experts = num_experts
        self.expert_names = list(experts.keys())

        # Small routing MLP: 235 -> 64 -> 6
        self.gate_net = nn.Sequential(
            nn.Linear(num_obs, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, obs: torch.Tensor):
        """Compute attention-weighted expert blend.

        Args:
            obs: (N, 235) observation tensor.

        Returns:
            blended_mean: (N, 12) weighted action means.
            blended_std: (N, 12) weighted action stds.
            weights: (N, 6) routing weights (for logging/debugging).
        """
        # Compute routing weights via softmax
        logits = self.gate_net(obs)                    # (N, 6)
        weights = torch.softmax(logits, dim=-1)        # (N, 6)

        # Query all frozen experts (no gradient)
        expert_means = []
        expert_stds = []
        with torch.no_grad():
            for name in self.expert_names:
                expert = self.experts[name]
                mean = expert.actor(obs)                # (N, 12)
                std = expert.std.expand_as(mean)        # (N, 12)
                expert_means.append(mean)
                expert_stds.append(std)

        # Stack: (N, 6, 12)
        means_stack = torch.stack(expert_means, dim=1)
        stds_stack = torch.stack(expert_stds, dim=1)

        # Weighted blend: (N, 6, 1) * (N, 6, 12) -> sum -> (N, 12)
        w = weights.unsqueeze(-1)                       # (N, 6, 1)
        blended_mean = (w * means_stack).sum(dim=1)     # (N, 12)
        blended_std = (w * stds_stack).sum(dim=1)       # (N, 12)

        return blended_mean, blended_std, weights

    @staticmethod
    def load_expert(checkpoint_path, num_obs=235, num_actions=12,
                    hidden_dims=None, device="cuda"):
        """Load a frozen expert from a training checkpoint.

        Reuses the pattern from multi_expert_distillation/expert_router.py.
        Handles both old RSL-RL and new TensorDict API.

        Args:
            checkpoint_path: Path to model_XXXX.pt file.
            num_obs: Observation dimensions.
            num_actions: Action dimensions.
            hidden_dims: Network layer sizes.
            device: CUDA or CPU.

        Returns:
            Frozen ActorCritic model.
        """
        from rsl_rl.modules import ActorCritic

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Detect obs dim from first layer
        first_layer = state_dict.get("actor.0.weight", None)
        if first_layer is not None:
            detected_obs = first_layer.shape[1]
            if detected_obs != num_obs:
                print(f"[ROUTER] WARNING: checkpoint has {detected_obs}-dim obs, expected {num_obs}")
                num_obs = detected_obs

        # H100 RSL-RL uses TensorDict API
        try:
            from tensordict import TensorDict
            obs = TensorDict({"policy": torch.zeros(1, num_obs, device=device)}, batch_size=[1])
            obs_groups = {"policy": ["policy"], "critic": ["policy"]}
            model = ActorCritic(
                obs=obs, obs_groups=obs_groups, num_actions=num_actions,
                actor_hidden_dims=hidden_dims, critic_hidden_dims=hidden_dims,
                activation="elu",
            )
        except TypeError:
            # Fallback for older RSL-RL
            model = ActorCritic(
                num_actor_obs=num_obs, num_critic_obs=num_obs,
                num_actions=num_actions,
                actor_hidden_dims=hidden_dims, critic_hidden_dims=hidden_dims,
                activation="elu",
            )

        model.to(device)
        model.load_state_dict(state_dict)
        model.eval()

        for p in model.parameters():
            p.requires_grad = False

        return model

    @classmethod
    def load_all_experts(cls, checkpoint_paths: dict, num_obs=235, num_actions=12,
                         hidden_dims=None, device="cuda"):
        """Load all 6 frozen experts and create the router.

        Args:
            checkpoint_paths: Dict of {"friction": "path.pt", "stairs_up": "path.pt", ...}
            num_obs: Observation dimensions.
            num_actions: Action dimensions.
            hidden_dims: Network layer sizes.
            device: CUDA or CPU.

        Returns:
            MultiExpertRouter instance with all experts loaded.
        """
        experts = {}
        for name, path in checkpoint_paths.items():
            print(f"[ROUTER] Loading expert '{name}' from: {path}", flush=True)
            experts[name] = cls.load_expert(path, num_obs, num_actions, hidden_dims, device)
        print(f"[ROUTER] All {len(experts)} experts loaded and frozen.", flush=True)

        router = cls(experts, num_obs=num_obs, num_experts=len(experts),
                     hidden_dim=64)
        router.to(device)
        return router
