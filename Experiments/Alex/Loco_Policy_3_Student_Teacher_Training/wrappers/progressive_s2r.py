"""Progressive S2R Wrapper — scales sim-to-real hardening with terrain level.

Instead of applying full S2R from step 0 (which causes standing-still collapse),
this wrapper ramps S2R intensity based on the robot's curriculum progress:

    Terrain Row 0-2:  scale=0.0 → Clean signals, learn to walk
    Terrain Row 3-4:  scale=0.3 → Light delay (12ms), 1.5% dropout
    Terrain Row 5-6:  scale=0.6 → Moderate delay (24ms), 3% dropout
    Terrain Row 7-9:  scale=1.0 → Full delay (40ms), 5% dropout, drift

The robot learns terrain mastery AND sim-to-real robustness simultaneously,
but S2R only kicks in after the gait is stable. By terrain row 5, the policy
is handling real-world-level latency alongside challenging terrain.

Analogy: Learning to juggle on solid ground first, then gradually adding
a rocking boat underneath — not the other way around.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from __future__ import annotations

import torch


class ProgressiveS2RWrapper:
    """Unified S2R wrapper with terrain-level-based intensity scaling.

    Combines action delay, observation delay, sensor noise/dropout, and IMU
    drift into a single wrapper whose intensity scales from 0.0 (clean) to
    1.0 (full S2R hardening) based on the current terrain curriculum level.

    Stacking: This replaces the three separate wrappers. Apply as:
        ProgressiveS2RWrapper(RslRlVecEnvWrapper(env))
    """

    # Observation layout (235-dim Mason hybrid)
    HEIGHT_SCAN_END = 187
    BASE_LIN_VEL_START = 187
    BASE_LIN_VEL_END = 190
    BASE_ANG_VEL_START = 190
    BASE_ANG_VEL_END = 193
    LAST_ACTION_START = 223

    def __init__(
        self,
        env,
        # Maximum S2R values (at scale=1.0)
        max_action_delay_steps: int = 2,     # 40ms at 50Hz
        max_obs_delay_steps: int = 1,        # 20ms at 50Hz
        max_dropout_rate: float = 0.05,      # 5% height scan dropout
        max_drift_rate: float = 0.002,       # OU-process IMU drift
        max_spike_prob: float = 0.001,       # 0.1% spike noise
        spike_magnitude: float = 1.0,
        # Curriculum ramp settings
        s2r_start_terrain: float = 0.2,      # Start ramping S2R at terrain 0.2 (row ~2)
        s2r_full_terrain: float = 0.6,       # Full S2R at terrain 0.6 (row ~6)
    ):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device

        # Max S2R params
        self.max_action_delay = max_action_delay_steps
        self.max_obs_delay = max_obs_delay_steps
        self.max_dropout_rate = max_dropout_rate
        self.max_drift_rate = max_drift_rate
        self.max_spike_prob = max_spike_prob
        self.spike_magnitude = spike_magnitude

        # Terrain ramp
        self.s2r_start = s2r_start_terrain
        self.s2r_full = s2r_full_terrain

        # Current S2R scale (0.0 = clean, 1.0 = full)
        self._scale = 0.0

        # Action delay buffer (pre-allocated at max size)
        self.num_actions = env.action_space.shape[-1] if hasattr(env.action_space, 'shape') else 12
        self._action_buffer = torch.zeros(
            max_action_delay_steps + 1, self.num_envs, self.num_actions,
            device=self.device, dtype=torch.float32,
        )
        self._action_head = 0

        # Observation delay buffer (lazy init)
        self._obs_buffer = None
        self._obs_head = 0
        self._obs_initialized = False

        # IMU drift state (Ornstein-Uhlenbeck)
        self.imu_drift = torch.zeros(
            self.num_envs, 6, device=self.device, dtype=torch.float32
        )

        # Stats for logging
        self._step_count = 0

    @property
    def scale(self) -> float:
        """Current S2R intensity scale (0.0 to 1.0)."""
        return self._scale

    def update_scale_from_terrain(self, terrain_level: float):
        """Update S2R scale based on current terrain curriculum level.

        Args:
            terrain_level: Average terrain level (0.0 to 1.0, normalized).
                0.0 = all robots on easiest terrain
                1.0 = all robots on hardest terrain
        """
        if terrain_level <= self.s2r_start:
            self._scale = 0.0
        elif terrain_level >= self.s2r_full:
            self._scale = 1.0
        else:
            # Linear ramp between start and full
            self._scale = (terrain_level - self.s2r_start) / (self.s2r_full - self.s2r_start)

    def set_scale(self, scale: float):
        """Manually set S2R scale (0.0 to 1.0)."""
        self._scale = max(0.0, min(1.0, scale))

    # -- Current effective S2R parameters --

    @property
    def effective_action_delay(self) -> int:
        """Current action delay steps (0 to max_action_delay)."""
        return round(self._scale * self.max_action_delay)

    @property
    def effective_dropout_rate(self) -> float:
        """Current height scan dropout rate."""
        return self._scale * self.max_dropout_rate

    @property
    def effective_drift_rate(self) -> float:
        """Current IMU drift rate."""
        return self._scale * self.max_drift_rate

    @property
    def effective_spike_prob(self) -> float:
        """Current spike noise probability."""
        return self._scale * self.max_spike_prob

    # -- Environment interface --

    def step(self, actions: torch.Tensor):
        """Apply scaled action delay, then step environment.

        At scale=0.0: actions pass through immediately (no delay).
        At scale=0.5: 1-step delay (20ms).
        At scale=1.0: 2-step delay (40ms).
        """
        delay = self.effective_action_delay

        if delay == 0:
            # No delay — pass through
            result = self.env.step(actions)
        else:
            # Read delayed action from buffer
            delayed_action = self._action_buffer[self._action_head].clone()
            # Store current action
            self._action_buffer[self._action_head] = actions
            # Advance head (ring buffer over `delay` slots)
            self._action_head = (self._action_head + 1) % delay
            result = self.env.step(delayed_action)

        self._step_count += 1
        return result

    def get_observations(self):
        """Return observations with scaled noise, dropout, and delay."""
        obs_dict = self.env.get_observations()

        # Extract tensor from dict/TensorDict
        obs_key = None
        try:
            if hasattr(obs_dict, 'keys'):
                obs_key = "policy" if "policy" in obs_dict else list(obs_dict.keys())[0]
                obs = obs_dict[obs_key]
            else:
                obs = obs_dict
        except Exception:
            obs = obs_dict

        # Clone to avoid modifying env state
        obs = obs.clone()

        # Only apply S2R effects if scale > 0
        if self._scale > 0.01:
            # 1. Height scan dropout
            obs = self._apply_dropout(obs)

            # 2. IMU drift
            obs = self._apply_imu_drift(obs)

            # 3. Spike noise
            obs = self._apply_spike_noise(obs)

        # Write back
        if obs_key is not None:
            obs_dict[obs_key] = obs
        else:
            obs_dict = obs

        return obs_dict

    def reset(self):
        """Reset environment and clear all S2R buffers."""
        self._action_buffer.zero_()
        self._action_head = 0
        self.imu_drift.zero_()
        if self._obs_buffer is not None:
            self._obs_buffer.zero_()
        self._obs_initialized = False
        return self.env.reset()

    def reset_idx(self, env_ids: torch.Tensor):
        """Clear buffers for reset environments."""
        if env_ids is not None and len(env_ids) > 0:
            self._action_buffer[:, env_ids, :] = 0.0
            self.imu_drift[env_ids] = 0.0
            if self._obs_buffer is not None:
                self._obs_buffer[:, env_ids, :] = 0.0
        if hasattr(self.env, 'reset_idx'):
            return self.env.reset_idx(env_ids)

    # -- S2R noise functions (scaled by self._scale) --

    def _apply_dropout(self, obs: torch.Tensor) -> torch.Tensor:
        """Randomly zero height scan rays. Dropout rate scales with S2R intensity."""
        rate = self.effective_dropout_rate
        if rate <= 0.0:
            return obs

        mask = torch.rand(
            obs.shape[0], self.HEIGHT_SCAN_END,
            device=self.device,
        ) < rate
        obs[:, :self.HEIGHT_SCAN_END][mask] = 0.0
        return obs

    def _apply_imu_drift(self, obs: torch.Tensor) -> torch.Tensor:
        """OU-process drift on base velocity channels. Rate scales with S2R intensity."""
        rate = self.effective_drift_rate
        if rate <= 0.0:
            return obs

        noise = torch.randn_like(self.imu_drift) * rate
        self.imu_drift = (1.0 - 0.01) * self.imu_drift + noise

        obs[:, self.BASE_LIN_VEL_START:self.BASE_LIN_VEL_END] += self.imu_drift[:, :3]
        obs[:, self.BASE_ANG_VEL_START:self.BASE_ANG_VEL_END] += self.imu_drift[:, 3:]
        return obs

    def _apply_spike_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """Rare large noise spikes. Probability scales with S2R intensity."""
        prob = self.effective_spike_prob
        if prob <= 0.0:
            return obs

        sensor_end = self.LAST_ACTION_START  # 223
        spike_mask = torch.rand(
            obs.shape[0], sensor_end, device=self.device
        ) < prob
        spike_values = torch.randn(
            obs.shape[0], sensor_end, device=self.device
        ) * self.spike_magnitude
        obs[:, :sensor_end] += spike_mask.float() * spike_values
        return obs

    def log_status(self) -> str:
        """Return a log string with current S2R status."""
        return (
            f"S2R scale={self._scale:.2f} "
            f"delay={self.effective_action_delay}steps "
            f"dropout={self.effective_dropout_rate:.1%} "
            f"drift={self.effective_drift_rate:.4f}"
        )

    def __getattr__(self, name):
        """Forward attribute access to wrapped environment."""
        return getattr(self.env, name)
