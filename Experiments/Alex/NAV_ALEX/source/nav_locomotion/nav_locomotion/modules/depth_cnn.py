"""Depth CNN encoder and custom ActorCriticCNN policy for RSL-RL.

Architecture:
    Depth Image (N, 1, 64, 64)
        -> Conv2d(1, 32, 5, stride=2)  -> ELU  -> (N, 32, 30, 30)
        -> Conv2d(32, 64, 3, stride=2) -> ELU  -> (N, 64, 14, 14)
        -> Conv2d(64, 64, 3, stride=2) -> ELU  -> (N, 64, 6, 6)
        -> Flatten                              -> (N, 2304)
        -> Linear(2304, 128)           -> ELU  -> (N, 128)

    CNN features (128) + proprioception (12) = 140
        -> Actor MLP [256, 128] -> 3-dim action [vx, vy, wz]
        -> Critic MLP [256, 128] -> 1-dim value

The CNN is shared between actor and critic (shared visual backbone).
Proprioception includes: body_lin_vel (3), body_ang_vel (3), projected_gravity (3),
prev_action (3) = 12 total.

Implements the RSL-RL ActorCritic interface: act(), act_inference(), evaluate(), reset().
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


# Observation layout constants
DEPTH_RES = 64
DEPTH_DIMS = DEPTH_RES * DEPTH_RES  # 4096
PROPRIO_DIMS = 12  # lin_vel(3) + ang_vel(3) + gravity(3) + prev_action(3)
TOTAL_OBS_DIMS = DEPTH_DIMS + PROPRIO_DIMS  # 4108


class DepthCNN(nn.Module):
    """3-layer CNN encoder that converts a 64x64 depth image to a 128-dim feature vector.

    Args:
        depth_res: Spatial resolution of the square depth image. Default 64.
        feature_dim: Output feature dimension. Default 128.
    """

    def __init__(self, depth_res: int = DEPTH_RES, feature_dim: int = 128):
        super().__init__()
        self.depth_res = depth_res
        self.feature_dim = feature_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2),  # (N, 32, 30, 30)
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (N, 64, 14, 14)
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),  # (N, 64, 6, 6)
            nn.ELU(),
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, depth_res, depth_res)
            flat_size = self.conv(dummy).numel()

        self.fc = nn.Sequential(
            nn.Linear(flat_size, feature_dim),
            nn.ELU(),
        )

    def forward(self, depth_flat: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            depth_flat: Flattened depth images, shape (N, depth_res*depth_res).

        Returns:
            Feature vector, shape (N, feature_dim).
        """
        x = depth_flat.view(-1, 1, self.depth_res, self.depth_res)
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


class ActorCriticCNN(nn.Module):
    """Custom ActorCritic policy for RSL-RL that splits flat obs into depth image
    and proprioception, runs depth through a CNN encoder, then feeds the concatenated
    features through separate actor and critic MLP heads.

    Implements the RSL-RL ActorCritic interface:
        - act(obs) -> actions, log_prob, value, mean_actions
        - act_inference(obs) -> actions
        - evaluate(obs, actions) -> value, log_prob, entropy
        - reset(dones) -> None

    Args:
        num_obs: Total observation dimension (depth_flat + proprio). Default 4108.
        num_actions: Action dimension (vx, vy, wz). Default 3.
        depth_res: Depth image resolution. Default 64.
        cnn_feature_dim: CNN output feature dimension. Default 128.
        mlp_hidden_dims: MLP hidden layer sizes. Default [256, 128].
        init_noise_std: Initial action noise standard deviation. Default 0.5.
    """

    def __init__(
        self,
        obs,                    # TensorDict {"policy": tensor(N, obs_dim)} or int
        obs_groups=None,        # obs groups config (ignored)
        num_actions: int = 3,
        depth_res: int = DEPTH_RES,
        cnn_feature_dim: int = 128,
        mlp_hidden_dims: list[int] | None = None,
        init_noise_std: float = 0.5,
        **kwargs,  # Absorb extra RSL-RL config params (noise_std_type, etc.)
    ):
        super().__init__()
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [256, 128]

        # RSL-RL passes (obs_tensordict, obs_groups, num_actions, **policy_cfg).
        # Extract num_obs from the obs tensor dict.
        if isinstance(obs, dict):
            policy_obs = obs.get("policy", next(iter(obs.values())))
            num_obs = int(policy_obs.shape[-1])
        elif hasattr(obs, "shape"):
            num_obs = int(obs.shape[-1])
        else:
            num_obs = int(obs)
        num_actions = int(num_actions)
        depth_res = int(depth_res)
        cnn_feature_dim = int(cnn_feature_dim)

        self.num_obs = num_obs
        self.num_actions = num_actions
        self.depth_dims = depth_res * depth_res
        self.proprio_dims = num_obs - self.depth_dims

        # Shared CNN encoder for depth images
        self.cnn = DepthCNN(depth_res=depth_res, feature_dim=cnn_feature_dim)

        # Combined feature dim: CNN features + proprioception
        combined_dim = cnn_feature_dim + self.proprio_dims

        # Actor MLP head
        actor_layers = []
        in_dim = combined_dim
        for hidden_dim in mlp_hidden_dims:
            actor_layers.append(nn.Linear(in_dim, hidden_dim))
            actor_layers.append(nn.ELU())
            in_dim = hidden_dim
        actor_layers.append(nn.Linear(in_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # Critic MLP head
        critic_layers = []
        in_dim = combined_dim
        for hidden_dim in mlp_hidden_dims:
            critic_layers.append(nn.Linear(in_dim, hidden_dim))
            critic_layers.append(nn.ELU())
            in_dim = hidden_dim
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Learnable action noise (log-space for stability)
        self.std = nn.Parameter(torch.ones(num_actions) * init_noise_std)

        # Distribution placeholder
        self.distribution = None

    def _unwrap_obs(self, obs):
        """Unwrap obs to tensor from dict/TensorDict/tensor."""
        if isinstance(obs, torch.Tensor):
            return obs
        # Handle dict or TensorDict
        if hasattr(obs, "get"):
            val = obs.get("policy", None)
            if val is not None:
                return val if isinstance(val, torch.Tensor) else next(iter(obs.values()))
        if hasattr(obs, "values"):
            return next(iter(obs.values()))
        return obs

    def _split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split flat observation into depth and proprioception components.

        Args:
            obs: Flat observation tensor, shape (N, num_obs).
                 Layout: [depth_flat (4096) | proprio (12)]

        Returns:
            depth_flat: (N, depth_dims)
            proprio: (N, proprio_dims)
        """
        depth_flat = obs[:, :self.depth_dims]
        proprio = obs[:, self.depth_dims:]
        return depth_flat, proprio

    def _get_features(self, obs) -> torch.Tensor:
        """Extract combined feature vector from observations.

        Args:
            obs: Flat observations tensor or dict {"policy": tensor}.

        Returns:
            Combined features, shape (N, cnn_feature_dim + proprio_dims).
        """
        obs = self._unwrap_obs(obs)
        if not hasattr(self, '_debug_printed'):
            print(f"[DEBUG CNN] obs type={type(obs)}, shape={obs.shape if hasattr(obs, 'shape') else '?'}, depth_dims={self.depth_dims}, proprio_dims={self.proprio_dims}", flush=True)
            self._debug_printed = True
        depth_flat, proprio = self._split_obs(obs)
        depth_features = self.cnn(depth_flat)
        return torch.cat([depth_features, proprio], dim=-1)

    def _sanitize_std(self) -> torch.Tensor:
        """Get sanitized standard deviation — guards against NaN/Inf/negative.

        Bug #24: clamp_() does NOT fix NaN. Must explicitly detect and replace
        NaN/Inf/negative values before clamping.

        Returns:
            Safe std tensor, shape (num_actions,).
        """
        std = self.std
        std = torch.where(torch.isfinite(std), std, torch.ones_like(std) * 0.5)
        std = torch.clamp(std, min=0.01, max=2.0)
        return std

    # --- RSL-RL interface properties ---

    @property
    def is_recurrent(self) -> bool:
        return False

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def act(self, obs, **kwargs) -> torch.Tensor:
        """Sample actions. RSL-RL calls separate methods for log_prob/value."""
        features = self._get_features(obs)
        mean = self.actor(features)
        std = self._sanitize_std()
        self.distribution = Normal(mean, std)
        actions = self.distribution.sample()
        return actions.detach()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Log prob of actions under current distribution."""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, obs, **kwargs) -> torch.Tensor:
        """Return value estimates."""
        features = self._get_features(obs)
        return self.critic(features)

    @torch.no_grad()
    def act_inference(self, obs) -> torch.Tensor:
        """Deterministic action selection."""
        features = self._get_features(obs)
        return self.actor(features)

    def reset(self, dones=None) -> None:
        """No-op for feedforward policy."""
        pass

    def update_normalization(self, obs) -> None:
        """No-op — we don't use observation normalization."""
        pass

    def get_hidden_states(self):
        """No hidden states for feedforward policy."""
        return None
