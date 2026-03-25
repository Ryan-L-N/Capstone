"""Action delay wrapper — simulates real-world actuator communication latency.

Wraps an RSL-RL compatible environment to buffer actions for N control steps
before applying them. This simulates the 40-60 ms delay between the policy
computing an action and the real Spot motors executing it.

At 50 Hz control rate: 2 steps = 40 ms, 3 steps = 60 ms.

The ring buffer lives on GPU for zero-copy operation with IsaacLab environments.
Buffer entries for reset environments are zeroed to prevent stale action carryover.

Addresses Risk R1 (no actuator latency simulation) from sim-to-real evaluation.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from __future__ import annotations

import torch


class ActionDelayWrapper:
    """Wraps an RSL-RL VecEnv to delay action execution by N steps.

    The wrapper maintains a GPU ring buffer of shape
    (delay_steps, num_envs, num_actions). Each call to step():
      1. Reads the oldest action from the buffer
      2. Writes the new action into the buffer
      3. Advances the ring buffer head
      4. Calls env.step() with the DELAYED action

    This preserves the full RslRlVecEnvWrapper interface expected by
    OnPolicyRunner (observation_space, action_space, num_envs, etc.).
    """

    def __init__(self, env, delay_steps: int = 2):
        """Initialize the action delay wrapper.

        Args:
            env: RSL-RL compatible vectorized environment.
            delay_steps: Number of control steps to delay actions.
                At 50 Hz: 2 steps = 40 ms, 3 steps = 60 ms.
                Must be >= 1.
        """
        self.env = env
        self.delay_steps = max(1, delay_steps)

        # Determine dimensions from wrapped environment
        self.num_envs = env.num_envs
        self.num_actions = env.action_space.shape[1]
        self.device = env.device

        # Ring buffer: (delay_steps, num_envs, num_actions) on GPU
        self.buffer = torch.zeros(
            self.delay_steps, self.num_envs, self.num_actions,
            device=self.device, dtype=torch.float32,
        )
        self.head = 0  # Points to the oldest entry (next to read)

    def step(self, actions: torch.Tensor):
        """Apply delayed action to environment.

        Args:
            actions: (num_envs, num_actions) tensor from the policy.

        Returns:
            Same as env.step() — (obs_dict, rewards, dones, extras).
        """
        # Read the oldest action (this is the one we'll execute)
        delayed_action = self.buffer[self.head].clone()

        # Write the new action into the buffer slot
        self.buffer[self.head] = actions

        # Advance head
        self.head = (self.head + 1) % self.delay_steps

        # Execute the delayed action
        return self.env.step(delayed_action)

    def reset(self):
        """Reset the environment and zero the action buffer."""
        self.buffer.zero_()
        self.head = 0
        return self.env.reset()

    def get_observations(self):
        """Pass through to wrapped environment."""
        return self.env.get_observations()

    def reset_idx(self, env_ids: torch.Tensor):
        """Zero buffer entries for reset environments.

        Called by RSL-RL when specific environments reset mid-episode.
        Prevents stale delayed actions from carrying over after reset.
        """
        if env_ids is not None and len(env_ids) > 0:
            self.buffer[:, env_ids, :] = 0.0
        if hasattr(self.env, 'reset_idx'):
            return self.env.reset_idx(env_ids)

    # -- Delegate all other attributes to the wrapped environment --

    def __getattr__(self, name):
        """Forward any attribute access to the wrapped environment."""
        return getattr(self.env, name)
