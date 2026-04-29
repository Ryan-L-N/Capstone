"""Sensor noise wrapper — adds realistic noise beyond Isaac Lab's uniform noise.

Wraps an RSL-RL compatible environment to inject:
  1. Height scan ray dropout (5% of 187 rays zeroed each step)
  2. Ornstein-Uhlenbeck IMU drift on base_lin_vel and base_ang_vel
  3. Rare spike noise (0.1% probability, large magnitude)

These go BEYOND the additive uniform noise in the env config, which handles
basic sensor noise. This wrapper adds failure modes that uniform noise cannot:
  - Dropout: simulates LiDAR/depth occlusion (reflective surfaces, out-of-range)
  - OU drift: simulates IMU calibration drift (correlated temporal noise)
  - Spikes: simulates electromagnetic interference or sensor glitches

Observation layout (235-dim):
  [0:187]   = height_scan (17x11 grid)
  [187:190] = base_lin_vel (3)
  [190:193] = base_ang_vel (3)
  [193:196] = projected_gravity (3)
  [196:199] = velocity_commands (3)  -- NOT noised (policy input)
  [199:211] = joint_pos (12)
  [211:223] = joint_vel (12)
  [223:235] = last_action (12)       -- NOT noised (own output)

Addresses Risks R5 (idealized sensors) and R10 (no sensor dropout).

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from __future__ import annotations

import torch


class SensorNoiseWrapper:
    """Wraps an RSL-RL VecEnv to inject realistic sensor noise.

    Stacking order: SensorNoiseWrapper should be OUTERMOST:
        SensorNoise(ActionDelay(ObsDelay(RslRlVecEnvWrapper(env))))

    This ensures noise is applied to the final observation the policy sees,
    after any observation delay has been applied.
    """

    # Observation layout indices (235-dim Mason hybrid)
    HEIGHT_SCAN_START = 0
    HEIGHT_SCAN_END = 187
    BASE_LIN_VEL_START = 187
    BASE_LIN_VEL_END = 190
    BASE_ANG_VEL_START = 190
    BASE_ANG_VEL_END = 193
    VEL_COMMANDS_START = 196
    VEL_COMMANDS_END = 199
    LAST_ACTION_START = 223
    LAST_ACTION_END = 235

    def __init__(
        self,
        env,
        dropout_rate: float = 0.05,
        drift_rate: float = 0.002,
        drift_revert: float = 0.01,
        spike_prob: float = 0.001,
        spike_magnitude: float = 1.0,
        height_scan_dims: int = 187,
    ):
        """Initialize the sensor noise wrapper.

        Args:
            env: RSL-RL compatible vectorized environment.
            dropout_rate: Fraction of height scan rays to zero each step (0.05 = 5%).
            drift_rate: OU process noise injection rate for IMU drift.
            drift_revert: OU process mean-reversion rate (higher = faster revert to 0).
            spike_prob: Probability of a large noise spike per channel per step.
            spike_magnitude: Magnitude of spike noise.
            height_scan_dims: Number of height scan dimensions (first N obs dims).
        """
        self.env = env
        self.dropout_rate = dropout_rate
        self.drift_rate = drift_rate
        self.drift_revert = drift_revert
        self.spike_prob = spike_prob
        self.spike_magnitude = spike_magnitude
        self.height_scan_dims = height_scan_dims

        self.num_envs = env.num_envs
        self.device = env.device

        # IMU drift state — persistent across steps (Ornstein-Uhlenbeck process)
        # Covers base_lin_vel (3) + base_ang_vel (3) = 6 channels
        self.imu_drift = torch.zeros(
            self.num_envs, 6, device=self.device, dtype=torch.float32
        )

    def step(self, actions: torch.Tensor):
        """Pass through to wrapped environment."""
        return self.env.step(actions)

    def reset(self):
        """Reset and clear IMU drift state."""
        self.imu_drift.zero_()
        return self.env.reset()

    def get_observations(self):
        """Return observations with injected sensor noise.

        Applies (in order):
          1. Height scan ray dropout
          2. OU-process IMU drift on base velocities
          3. Rare spike noise on sensor channels

        Does NOT modify velocity_commands or last_action.
        """
        obs_dict = self.env.get_observations()

        # Extract policy observation tensor (handles dict, TensorDict, or raw tensor)
        obs_key = None
        try:
            if hasattr(obs_dict, 'keys'):
                obs_key = "policy" if "policy" in obs_dict else list(obs_dict.keys())[0]
                obs = obs_dict[obs_key]
            else:
                obs = obs_dict
        except Exception:
            obs = obs_dict

        # Work on a clone to avoid modifying the environment's internal state
        obs = obs.clone()

        # 1. Height scan ray dropout
        obs = self._apply_dropout(obs)

        # 2. IMU drift (Ornstein-Uhlenbeck process)
        obs = self._apply_imu_drift(obs)

        # 3. Spike noise on sensor channels (not commands, not last_action)
        obs = self._apply_spike_noise(obs)

        # Write back
        if obs_key is not None:
            obs_dict[obs_key] = obs
        else:
            obs_dict = obs

        return obs_dict

    def _apply_dropout(self, obs: torch.Tensor) -> torch.Tensor:
        """Randomly zero height scan rays to simulate LiDAR/depth dropouts.

        Uses 0.0 fill value matching training convention for flat ground.
        """
        if self.dropout_rate <= 0.0:
            return obs

        # Create dropout mask: True = drop this ray
        dropout_mask = torch.rand(
            self.num_envs, self.height_scan_dims,
            device=self.device,
        ) < self.dropout_rate

        # Zero the dropped rays (0.0 = flat ground convention)
        obs[:, self.HEIGHT_SCAN_START:self.HEIGHT_SCAN_END][dropout_mask] = 0.0

        return obs

    def _apply_imu_drift(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply Ornstein-Uhlenbeck drift to base velocity observations.

        OU process: dx = -theta * x * dt + sigma * dW
        This creates correlated temporal noise that slowly drifts and reverts.
        Real IMUs exhibit this drift pattern from temperature and calibration changes.
        """
        if self.drift_rate <= 0.0:
            return obs

        # OU update: drift toward zero + random walk
        noise = torch.randn_like(self.imu_drift) * self.drift_rate
        self.imu_drift = (1.0 - self.drift_revert) * self.imu_drift + noise

        # Apply drift to base_lin_vel (dims 187-189)
        obs[:, self.BASE_LIN_VEL_START:self.BASE_LIN_VEL_END] += self.imu_drift[:, :3]

        # Apply drift to base_ang_vel (dims 190-192)
        obs[:, self.BASE_ANG_VEL_START:self.BASE_ANG_VEL_END] += self.imu_drift[:, 3:]

        return obs

    def _apply_spike_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply rare large-magnitude noise spikes to sensor channels.

        Simulates electromagnetic interference or sensor hardware glitches.
        Applied to height_scan + proprioceptive channels, NOT to commands or actions.
        """
        if self.spike_prob <= 0.0:
            return obs

        # Sensor channels: height_scan(187) + base_vel(3) + base_ang(3) + gravity(3)
        #                 + joint_pos(12) + joint_vel(12) = dims 0:223
        sensor_end = self.LAST_ACTION_START  # 223

        spike_mask = torch.rand(
            self.num_envs, sensor_end,
            device=self.device,
        ) < self.spike_prob

        spike_values = torch.randn(
            self.num_envs, sensor_end,
            device=self.device,
        ) * self.spike_magnitude

        obs[:, :sensor_end] += spike_mask.float() * spike_values

        return obs

    def reset_idx(self, env_ids: torch.Tensor):
        """Clear IMU drift for reset environments."""
        if env_ids is not None and len(env_ids) > 0:
            self.imu_drift[env_ids] = 0.0
        if hasattr(self.env, 'reset_idx'):
            return self.env.reset_idx(env_ids)

    def __getattr__(self, name):
        """Forward any attribute access to the wrapped environment."""
        return getattr(self.env, name)
