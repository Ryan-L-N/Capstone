"""Adapter classes to bridge ActorCriticCNN (old rsl_rl API) to rsl_rl 5.0.1 MLPModel interface.

rsl_rl 5.0.1 replaced the combined ActorCritic class with separate actor and critic
MLPModel objects. Alex's ActorCriticCNN was written for the old API (rsl_rl < 4.0.0).

These wrappers let OnPolicyRunner use ActorCriticCNN without any changes to Alex's code.
They are instantiated by rsl_rl via resolve_callable("cnn_compat:ActorCNNWrapper") etc.

rsl_rl 5.0.1 actor interface that PPO calls:
    actor(obs, stochastic_output=True)    -> sampled actions (N, act_dim)
    actor.get_output_log_prob(actions)    -> log probs       (N,)
    actor.output_distribution_params      -> (mean, std)     tuple of (N, act_dim)
    actor.output_entropy                  -> per-env entropy (N,)
    actor.output_std                      -> action std      (N, act_dim) or (act_dim,)
    actor.get_kl_divergence(old, new)     -> per-env KL      (N,)
    actor.update_normalization(obs)       -> no-op
    actor.get_hidden_state()              -> None  (not recurrent)
    actor.detach_hidden_state(dones)      -> no-op
    actor.reset(dones, hidden_state)      -> no-op
    actor.is_recurrent                    -> False

rsl_rl 5.0.1 critic interface:
    critic(obs)                           -> values (N, 1)
    critic.get_hidden_state()             -> None
    critic.reset(dones, hidden_state)     -> no-op
    critic.detach_hidden_state(dones)     -> no-op
    critic.update_normalization(obs)      -> no-op
    critic.is_recurrent                   -> False

Note: ActorCNNWrapper and CriticCNNWrapper each hold a full ActorCriticCNN instance.
The CNN backbone is NOT shared between them (rsl_rl instantiates them separately with
no shared-state hook for our case). Memory overhead is ~2x CNN params (~2 MB extra).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as torch_kl_divergence


def _extract_policy_obs(obs) -> torch.Tensor:
    """Pull the flat 'policy' tensor out of a dict/TensorDict or return as-is."""
    if isinstance(obs, dict):
        return obs.get("policy", next(iter(obs.values())))
    return obs


class ActorCNNWrapper(nn.Module):
    """Actor adapter: wraps ActorCriticCNN actor head for rsl_rl 5.0.1.

    rsl_rl calls this as:
        actor_class(obs, obs_groups, obs_set, num_actions, **actor_cfg)
    """

    is_recurrent: bool = False

    def __init__(self, obs, obs_groups, obs_set, num_actions, **kwargs):
        super().__init__()
        from nav_locomotion.modules.depth_cnn import ActorCriticCNN

        if isinstance(obs, dict):
            policy_obs = obs.get("policy", next(iter(obs.values())))
            num_obs = int(policy_obs.shape[-1])
        elif hasattr(obs, "shape"):
            num_obs = int(obs.shape[-1])
        else:
            num_obs = int(obs)

        self._net = ActorCriticCNN(num_obs, num_actions=int(num_actions))
        self._dist: Normal | None = None

    # ------------------------------------------------------------------
    # Core forward pass — sets self._dist, returns actions
    # ------------------------------------------------------------------

    def forward(
        self,
        obs,
        masks=None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs_t = _extract_policy_obs(obs)
        features = self._net._get_features(obs_t)
        mean = self._net.actor(features)
        std = self._net._sanitize_std()
        self._dist = Normal(mean, std)
        if stochastic_output:
            return self._dist.sample()
        return mean  # deterministic

    # ------------------------------------------------------------------
    # Distribution-related interface
    # ------------------------------------------------------------------

    def get_output_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Log prob of actions under the most recently sampled distribution.

        Must call forward(stochastic_output=True) before calling this.
        Returns shape (N,) — PPO sums over action dims internally via storage.
        """
        if self._dist is None:
            raise RuntimeError(
                "[ActorCNNWrapper] Call forward(stochastic_output=True) before get_output_log_prob()"
            )
        return self._dist.log_prob(actions).sum(dim=-1)

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """(mean, std) for the most recent distribution — used for KL computation."""
        if self._dist is None:
            # Return dummy params; PPO will overwrite on first real forward pass
            dummy = torch.zeros(1, device=next(self._net.parameters()).device)
            return (dummy, dummy + 1.0)
        return (self._dist.mean, self._dist.stddev)

    @property
    def output_entropy(self) -> torch.Tensor:
        """Per-env entropy of the current distribution, shape (N,)."""
        if self._dist is None:
            return torch.zeros(1, device=next(self._net.parameters()).device)
        return self._dist.entropy().sum(dim=-1)

    @property
    def output_mean(self) -> torch.Tensor:
        return self._dist.mean

    @property
    def output_std(self) -> torch.Tensor:
        """Action std used by the logger (shape (N, act_dim) or (act_dim,))."""
        if self._dist is None:
            return self._net._sanitize_std().unsqueeze(0)
        return self._dist.stddev

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """KL(old || new), shape (N,)."""
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        p = Normal(old_mean, old_std)
        q = Normal(new_mean, new_std)
        return torch_kl_divergence(p, q).sum(dim=-1)

    # ------------------------------------------------------------------
    # Recurrent / housekeeping interface (all no-ops for feedforward)
    # ------------------------------------------------------------------

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones=None) -> None:
        pass

    def reset(self, dones=None, hidden_state=None) -> None:
        pass

    def update_normalization(self, obs) -> None:
        pass  # no observation normalization


class CriticCNNWrapper(nn.Module):
    """Critic adapter: wraps ActorCriticCNN critic head for rsl_rl 5.0.1.

    rsl_rl calls this as:
        critic_class(obs, obs_groups, obs_set, num_actions, **critic_cfg)
    """

    is_recurrent: bool = False

    def __init__(self, obs, obs_groups, obs_set, num_actions, **kwargs):
        super().__init__()
        from nav_locomotion.modules.depth_cnn import ActorCriticCNN

        if isinstance(obs, dict):
            policy_obs = obs.get("policy", next(iter(obs.values())))
            num_obs = int(policy_obs.shape[-1])
        elif hasattr(obs, "shape"):
            num_obs = int(obs.shape[-1])
        else:
            num_obs = int(obs)

        # Separate ActorCriticCNN instance from the actor (not shared).
        self._net = ActorCriticCNN(num_obs, num_actions=int(num_actions))

    def forward(
        self,
        obs,
        masks=None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Returns value estimates, shape (N, 1)."""
        obs_t = _extract_policy_obs(obs)
        features = self._net._get_features(obs_t)
        return self._net.critic(features)  # (N, 1)

    # ------------------------------------------------------------------
    # Recurrent / housekeeping interface (all no-ops)
    # ------------------------------------------------------------------

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones=None) -> None:
        pass

    def reset(self, dones=None, hidden_state=None) -> None:
        pass

    def update_normalization(self, obs) -> None:
        pass
