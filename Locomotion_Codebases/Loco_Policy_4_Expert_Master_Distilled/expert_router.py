"""Expert routing based on terrain roughness from height scan observations.

Routes each environment to the friction expert (smooth terrain) or obstacle
expert (rough terrain) using soft sigmoid gating on height scan variance.

The height scan is the first 187 dims of the 235-dim observation vector.
On flat/smooth terrain, variance is near zero. On boulders/stairs, it's high.
The sigmoid gate smoothly blends between experts at the transition boundary.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

import torch
from rsl_rl.modules import ActorCritic


class ExpertRouter:
    """Routes observations to terrain-specialized experts via height scan gating."""

    def __init__(self, friction_expert, obstacle_expert,
                 height_scan_dims=187, threshold=0.005, temperature=0.005):
        """
        Args:
            friction_expert: Frozen ActorCritic for smooth terrain.
            obstacle_expert: Frozen ActorCritic for rough terrain.
            height_scan_dims: Number of height scan dimensions (first N obs dims).
            threshold: Height scan variance threshold for routing.
            temperature: Sigmoid temperature (lower = sharper gate).
        """
        self.friction_expert = friction_expert
        self.obstacle_expert = obstacle_expert
        self.height_scan_dims = height_scan_dims
        self.threshold = threshold
        self.temperature = temperature

    def compute_roughness(self, obs):
        """Compute terrain roughness from height scan variance.

        Args:
            obs: (N, 235) observation tensor.

        Returns:
            roughness: (N,) per-environment roughness scores.
        """
        height_scan = obs[:, :self.height_scan_dims]
        return torch.var(height_scan, dim=1)

    def compute_gate(self, obs):
        """Soft gate: 0.0 = use friction expert, 1.0 = use obstacle expert."""
        roughness = self.compute_roughness(obs)
        return torch.sigmoid((roughness - self.threshold) / self.temperature)

    @torch.no_grad()
    def get_expert_actions(self, obs):
        """Query both experts and blend actions based on terrain routing.

        Args:
            obs: (N, 235) observation tensor.

        Returns:
            blended_mean: (N, 12) weighted action means.
            blended_std:  (N, 12) weighted action stds.
            gate:         (N,) routing weights for logging.
            friction_mean: (N, 12) friction expert's action means.
            obstacle_mean: (N, 12) obstacle expert's action means.
        """
        friction_mean = self.friction_expert.actor(obs)
        obstacle_mean = self.obstacle_expert.actor(obs)

        gate = self.compute_gate(obs).unsqueeze(1)  # (N, 1) for broadcasting

        blended_mean = (1 - gate) * friction_mean + gate * obstacle_mean

        # Blend stds
        friction_std = self.friction_expert.std.expand_as(friction_mean)
        obstacle_std = self.obstacle_expert.std.expand_as(obstacle_mean)
        blended_std = (1 - gate) * friction_std + gate * obstacle_std

        return blended_mean, blended_std, gate.squeeze(1), friction_mean, obstacle_mean

    @staticmethod
    def load_expert(checkpoint_path, num_obs=235, num_actions=12,
                    hidden_dims=None, device="cuda"):
        """Load a frozen expert from a training checkpoint.

        Args:
            checkpoint_path: Path to model_XXXX.pt file.
            num_obs: Observation dimensions (must match checkpoint).
            num_actions: Action dimensions.
            hidden_dims: Network layer sizes.
            device: CUDA or CPU.

        Returns:
            Frozen ActorCritic model.
        """
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Detect obs dim from first layer weights
        first_layer = state_dict.get("actor.0.weight", None)
        if first_layer is not None:
            detected_obs = first_layer.shape[1]
            if detected_obs != num_obs:
                print(f"[ROUTER] WARNING: checkpoint has {detected_obs}-dim obs, expected {num_obs}")
                num_obs = detected_obs

        # H100 RSL-RL requires TensorDict obs + obs_groups for ActorCritic init
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
            # Fallback for older RSL-RL versions
            model = ActorCritic(
                num_actor_obs=num_obs, num_critic_obs=num_obs,
                num_actions=num_actions,
                actor_hidden_dims=hidden_dims, critic_hidden_dims=hidden_dims,
                activation="elu",
            )
        model.to(device)
        model.load_state_dict(state_dict)
        model.eval()

        # Freeze all parameters
        for p in model.parameters():
            p.requires_grad = False

        return model
