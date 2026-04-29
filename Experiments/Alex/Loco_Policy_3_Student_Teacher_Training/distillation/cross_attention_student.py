"""Option C: Cross-Attention Student — attends over 3 frozen expert latent representations.

Architecture:
  1. Each frozen expert [512,256,128] produces a 128-dim "token" (penultimate layer)
  2. Student encoder produces a query from the observation
  3. Cross-attention blends expert representations based on terrain context
  4. Action head maps attended representation to 12 joint targets

This allows the student to extract DIFFERENT features from each expert
depending on the situation — e.g., use the stair expert's foot placement
logic on ramps but the flat expert's gait timing on smooth ground.

The attention weights are interpretable: you can visualize which expert
the student is "listening to" at each timestep.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertLatentExtractor(nn.Module):
    """Extracts penultimate-layer activations from a frozen expert network.

    Given an RSL-RL actor with layers:
        actor.0: 235 → 512 (ELU)
        actor.2: 512 → 256 (ELU)
        actor.4: 256 → 128 (ELU)  ← we extract HERE
        actor.6: 128 → 12  (output)

    Returns the 128-dim activation after actor.4 + ELU.
    """

    def __init__(self, expert_state_dict: dict, num_obs: int = 235):
        super().__init__()

        # Rebuild the expert's actor layers (frozen)
        self.layers = nn.Sequential(
            nn.Linear(num_obs, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )

        # Load weights from checkpoint
        layer_map = {
            'actor.0.weight': 0, 'actor.0.bias': 0,
            'actor.2.weight': 2, 'actor.2.bias': 2,
            'actor.4.weight': 4, 'actor.4.bias': 4,
        }
        state = {}
        for ckpt_key, layer_idx in layer_map.items():
            param_type = 'weight' if 'weight' in ckpt_key else 'bias'
            state[f'{layer_idx}.{param_type}'] = expert_state_dict[ckpt_key]

        self.layers.load_state_dict(state)

        # Freeze — experts never update
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract 128-dim latent from observation. Shape: (N, 128)."""
        return self.layers(obs)


class ExpertActionHead(nn.Module):
    """The final layer of an expert — maps 128-dim latent to 12 actions.

    Used to compute expert actions for the distillation loss without
    running the full expert forward pass twice.
    """

    def __init__(self, expert_state_dict: dict):
        super().__init__()
        self.head = nn.Linear(128, 12)
        self.head.weight.data = expert_state_dict['actor.6.weight']
        self.head.bias.data = expert_state_dict['actor.6.bias']

        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.head(latent)


class CrossAttentionStudent(nn.Module):
    """Student policy that attends over 3 frozen expert latent representations.

    Architecture:
        obs (235) → encoder (235→256→128) → query (128)
        3 expert latents (3 × 128) → keys + values
        cross_attention(query, keys, values) → attended (128)
        action_head(attended) → actions (12)

    The attention mechanism learns WHICH expert's representation is most
    useful for the current terrain context. The query encodes "what do I
    need?", the keys encode "what does each expert know?", and the values
    encode "what should I do?".

    Attention weights are logged for interpretability.
    """

    def __init__(
        self,
        expert_checkpoints: dict,
        num_obs: int = 235,
        num_actions: int = 12,
        encoder_dims: tuple = (256, 128),
        latent_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        """
        Args:
            expert_checkpoints: Dict of {"flat": state_dict, "boulder": state_dict, "stair": state_dict}
            num_obs: Observation dimension (235 for Spot).
            num_actions: Action dimension (12 DOF).
            encoder_dims: Student encoder hidden dims.
            latent_dim: Expert latent dimension (128 for [512,256,128] networks).
            num_heads: Number of attention heads.
            dropout: Attention dropout (0 for RL).
        """
        super().__init__()
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.expert_names = list(expert_checkpoints.keys())
        self.num_experts = len(self.expert_names)

        # ── Frozen expert latent extractors ────────────────────────────
        self.expert_extractors = nn.ModuleDict()
        self.expert_heads = nn.ModuleDict()
        for name, state_dict in expert_checkpoints.items():
            self.expert_extractors[name] = ExpertLatentExtractor(state_dict, num_obs)
            self.expert_heads[name] = ExpertActionHead(state_dict)

        # ── Student observation encoder (trainable) ────────────────────
        encoder_layers = []
        in_dim = num_obs
        for dim in encoder_dims:
            encoder_layers.extend([nn.Linear(in_dim, dim), nn.ELU()])
            in_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)

        # ── Cross-attention (trainable) ────────────────────────────────
        # Query comes from student encoder, K/V from expert latents
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.key_proj = nn.Linear(latent_dim, latent_dim)
        self.value_proj = nn.Linear(latent_dim, latent_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ── Action head (trainable) ───────────────────────────────────
        self.action_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, num_actions),
        )

        # ── Action noise (for PPO exploration) ────────────────────────
        self.std = nn.Parameter(torch.ones(num_actions) * 0.5)

        # ── Critic (separate, trainable) ──────────────────────────────
        self.critic = nn.Sequential(
            nn.Linear(num_obs, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

        # ── Logging ──────────────────────────────────────────────────
        self._last_attention_weights = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass: obs → attended expert blend → action means.

        Returns action means (N, 12). Use self.std for distribution.
        """
        # Extract frozen expert latents: (N, 3, 128)
        expert_latents = torch.stack([
            self.expert_extractors[name](obs)
            for name in self.expert_names
        ], dim=1)  # (N, num_experts, latent_dim)

        # Student encoder → query: (N, 1, 128)
        query = self.encoder(obs).unsqueeze(1)  # (N, 1, 128)

        # Project Q, K, V
        q = self.query_proj(query)       # (N, 1, 128)
        k = self.key_proj(expert_latents)  # (N, 3, 128)
        v = self.value_proj(expert_latents)  # (N, 3, 128)

        # Cross-attention: student queries expert representations
        attended, attn_weights = self.attn(q, k, v)  # attended: (N, 1, 128), weights: (N, 1, 3)
        attended = attended.squeeze(1)  # (N, 128)

        # Store attention weights for logging
        self._last_attention_weights = attn_weights.squeeze(1).detach()  # (N, 3)

        # Action head
        action_mean = self.action_head(attended)  # (N, 12)
        return action_mean

    def act(self, obs: torch.Tensor) -> tuple:
        """Sample action for PPO. Returns (action, log_prob, value)."""
        action_mean = self.forward(obs)
        std = self.std.expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple:
        """Evaluate actions for PPO update. Returns (log_prob, entropy, value)."""
        action_mean = self.forward(obs)
        std = self.std.expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value

    def get_expert_actions(self, obs: torch.Tensor) -> dict:
        """Get each expert's action for distillation loss. Returns dict of (N, 12)."""
        actions = {}
        for name in self.expert_names:
            latent = self.expert_extractors[name](obs)
            actions[name] = self.expert_heads[name](latent)
        return actions

    def get_attention_weights(self) -> torch.Tensor:
        """Get last attention weights (N, 3) for logging/visualization."""
        return self._last_attention_weights

    def get_expert_weight_summary(self) -> dict:
        """Get mean attention weight per expert for TensorBoard logging."""
        if self._last_attention_weights is None:
            return {}
        weights = self._last_attention_weights.mean(dim=0)  # (3,)
        return {name: weights[i].item() for i, name in enumerate(self.expert_names)}


class AttentionDistillationLoss:
    """Cross-attention distillation loss.

    Combines:
    1. Per-expert MSE weighted by attention (student should match the expert it's attending to)
    2. KL divergence between student and attention-blended expert distribution
    3. Attention entropy bonus (encourage exploration of all experts early on)
    """

    def __init__(self, kl_weight: float = 0.1, entropy_weight: float = 0.01):
        self.kl_weight = kl_weight
        self.entropy_weight = entropy_weight

    def __call__(
        self,
        student_mean: torch.Tensor,
        student_std: torch.Tensor,
        expert_actions: dict,
        attention_weights: torch.Tensor,
        expert_names: list,
    ) -> tuple:
        """
        Args:
            student_mean: (N, 12)
            student_std: (N, 12)
            expert_actions: dict of {name: (N, 12)} expert action means
            attention_weights: (N, num_experts) from cross-attention
            expert_names: list of expert names matching attention_weights columns

        Returns:
            (total_loss, mse_value, kl_value, entropy_value)
        """
        # Attention-weighted expert blend
        expert_stack = torch.stack([expert_actions[n] for n in expert_names], dim=1)  # (N, 3, 12)
        weights = attention_weights.unsqueeze(2)  # (N, 3, 1)
        blended_expert = (expert_stack * weights).sum(dim=1)  # (N, 12)

        # MSE: student should match the blended expert
        mse = F.mse_loss(student_mean, blended_expert)

        # KL divergence: student distribution vs blended expert
        # Assume experts have same std as student (simplification)
        expert_std = student_std.detach()
        student_var = student_std.pow(2).clamp(min=1e-6)
        expert_var = expert_std.pow(2).clamp(min=1e-6)
        kl = (
            torch.log(expert_std / student_std.clamp(min=1e-6))
            + (student_var + (student_mean - blended_expert).pow(2)) / (2 * expert_var)
            - 0.5
        ).sum(dim=-1).mean()
        kl = torch.clamp(kl, 0.0, 10.0)

        # Attention entropy bonus: encourage using all experts (prevent collapse)
        attn_entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1).mean()

        total = mse + self.kl_weight * kl - self.entropy_weight * attn_entropy
        return total, mse.item(), kl.item(), attn_entropy.item()


def load_expert_state_dict(checkpoint_path: str) -> dict:
    """Load expert checkpoint and extract model_state_dict."""
    ckpt = torch.load(checkpoint_path, map_location="cuda:0", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    # Filter to only actor keys
    actor_keys = {k: v for k, v in state.items() if k.startswith("actor.")}
    return actor_keys
