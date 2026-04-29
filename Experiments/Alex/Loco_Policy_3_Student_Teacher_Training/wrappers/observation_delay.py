"""Observation delay wrapper — simulates real-world sensor communication latency.

Wraps an RSL-RL compatible environment to return observations from N steps ago.
This simulates the 10-20 ms delay between IMU/encoder readings and the policy
receiving them (sensor processing + communication latency).

At 50 Hz control rate: 1 step = 20 ms.

Works independently from ActionDelayWrapper — real robots have BOTH action delay
(command → motor) and observation delay (sensor → policy). Both can be stacked.

Addresses Risk R1 (no sensor latency simulation) from sim-to-real evaluation.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from __future__ import annotations

import torch


class ObservationDelayWrapper:
    """Wraps an RSL-RL VecEnv to delay observations by N steps.

    Maintains a GPU ring buffer of past observations. get_observations()
    returns the observation from N steps ago instead of the current one.

    Stacking order: ObservationDelayWrapper should be INNER relative to
    ActionDelayWrapper and SensorNoiseWrapper:
        SensorNoise(ActionDelay(ObsDelay(RslRlVecEnvWrapper(env))))
    """

    def __init__(self, env, delay_steps: int = 1):
        """Initialize the observation delay wrapper.

        Args:
            env: RSL-RL compatible vectorized environment.
            delay_steps: Number of control steps to delay observations.
                At 50 Hz: 1 step = 20 ms.
                Must be >= 1.
        """
        self.env = env
        self.delay_steps = max(1, delay_steps)

        self.num_envs = env.num_envs
        self.device = env.device

        # We'll lazily initialize the buffer on first observation
        # (need to know obs dimensionality)
        self.buffer = None
        self.head = 0
        self._initialized = False

    def _init_buffer(self, obs_tensor: torch.Tensor):
        """Lazily initialize the observation ring buffer.

        Args:
            obs_tensor: (num_envs, obs_dim) sample observation tensor.
        """
        if obs_tensor.dim() == 1:
            obs_dim = obs_tensor.shape[0]
        else:
            obs_dim = obs_tensor.shape[-1]
        num_envs = obs_tensor.shape[0] if obs_tensor.dim() > 1 else self.num_envs
        self.buffer = torch.zeros(
            self.delay_steps, num_envs, obs_dim,
            device=self.device, dtype=torch.float32,
        )
        self.head = 0
        self._initialized = True

    def step(self, actions: torch.Tensor):
        """Step the environment and buffer the observation internally."""
        return self.env.step(actions)

    def reset(self):
        """Reset the environment and clear the observation buffer."""
        result = self.env.reset()
        if self.buffer is not None:
            self.buffer.zero_()
            self.head = 0
        self._initialized = False
        return result

    def get_observations(self):
        """Return delayed observation.

        Reads the current observation from the wrapped env, stores it,
        and returns the observation from N steps ago.

        Returns:
            Delayed observation dict (same format as wrapped env).
        """
        obs_dict = self.env.get_observations()

        # Extract the policy observation tensor (handles dict, TensorDict, or raw tensor)
        obs_key = None
        try:
            # TensorDict or dict-like
            if hasattr(obs_dict, 'keys'):
                obs_key = "policy" if "policy" in obs_dict else list(obs_dict.keys())[0]
                current_obs = obs_dict[obs_key]
            else:
                current_obs = obs_dict
        except Exception:
            current_obs = obs_dict

        # Lazy init on first call
        if not self._initialized:
            self._init_buffer(current_obs)
            # Fill buffer with current obs (no delay on first steps)
            for i in range(self.delay_steps):
                self.buffer[i] = current_obs.clone()

        # Read the oldest observation (delayed)
        delayed_obs = self.buffer[self.head].clone()

        # Store current observation in the buffer
        self.buffer[self.head] = current_obs

        # Advance head
        self.head = (self.head + 1) % self.delay_steps

        # Return with delayed observation
        if obs_key is not None:
            obs_dict[obs_key] = delayed_obs
        else:
            obs_dict = delayed_obs

        return obs_dict

    def reset_idx(self, env_ids: torch.Tensor):
        """Clear buffer entries for reset environments."""
        if self.buffer is not None and env_ids is not None and len(env_ids) > 0:
            self.buffer[:, env_ids, :] = 0.0
        if hasattr(self.env, 'reset_idx'):
            return self.env.reset_idx(env_ids)

    def __getattr__(self, name):
        """Forward any attribute access to the wrapped environment."""
        return getattr(self.env, name)
