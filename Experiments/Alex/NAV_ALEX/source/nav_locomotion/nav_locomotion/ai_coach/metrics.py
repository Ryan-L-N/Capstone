"""Metrics collector for navigation training — captures snapshots for coach analysis.

Collects training metrics every N iterations and packages them into
MetricsSnapshot dataclasses. Tracks trends, detects plateaus, and
monitors body height for anti-crawl enforcement.

Nav-specific metrics beyond Phase B:
    - mean_forward_distance: Average X-distance per episode
    - mean_body_height: Body height relative to local terrain (anti-crawl)
    - mean_drag_penalty: Drag penalty magnitude (belly-sliding detection)
    - terrain_level: Current curriculum terrain difficulty

Adapted from multi_robot_training/ai_trainer/metrics.py.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class MetricsSnapshot:
    """Complete training state at a single point in time.

    Attributes:
        iteration: Current training iteration.
        elapsed_hours: Hours since training started.
        mean_reward: Mean episode reward across all envs.
        mean_episode_length: Mean episode length in steps.
        survival_rate: Fraction of envs that survived the episode.
        flip_rate: Fraction of envs that flipped over.
        mean_terrain_level: Mean terrain difficulty level (0-5).
        value_loss: Critic value function loss.
        policy_loss: Actor policy loss.
        noise_std: Current action noise standard deviation.
        learning_rate: Current optimizer learning rate.
        reward_breakdown: Per-term reward contributions.
        current_weights: Current reward weights.
        mean_forward_distance: Mean X-distance traveled per episode.
        mean_body_height: Mean body height relative to terrain.
        mean_drag_penalty: Mean drag penalty per step.
        reward_trend: Slope of reward over recent window.
        terrain_trend: Slope of terrain level over recent window.
        value_loss_trend: Slope of value loss over recent window.
        has_nan: Whether NaN detected in policy parameters.
        value_loss_spike: Whether value loss exceeds emergency threshold.
    """
    iteration: int = 0
    elapsed_hours: float = 0.0
    mean_reward: float = 0.0
    mean_episode_length: float = 0.0
    survival_rate: float = 0.0
    flip_rate: float = 0.0
    mean_terrain_level: float = 0.0
    value_loss: float = 0.0
    policy_loss: float = 0.0
    noise_std: float = 0.0
    learning_rate: float = 0.0
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    current_weights: dict[str, float] = field(default_factory=dict)
    mean_forward_distance: float = 0.0
    mean_body_height: float = 0.0
    mean_drag_penalty: float = 0.0
    reward_trend: float = 0.0
    terrain_trend: float = 0.0
    value_loss_trend: float = 0.0
    has_nan: bool = False
    value_loss_spike: bool = False


class MetricsCollector:
    """Collects and tracks training metrics for AI coach analysis.

    Args:
        env: NavEnvWrapper or Isaac Lab env.
        runner: RSL-RL OnPolicyRunner.
        history_size: Maximum number of snapshots to retain. Default 200.
        emergency_value_loss: Value loss threshold for spike detection. Default 100.0.
    """

    def __init__(
        self,
        env,
        runner,
        history_size: int = 200,
        emergency_value_loss: float = 100.0,
    ):
        self.env = env
        self.runner = runner
        self.history: deque[MetricsSnapshot] = deque(maxlen=history_size)
        self.emergency_value_loss = emergency_value_loss

        # Extras buffer for per-iteration metrics from training loop
        self._extras: dict = {}

    def update_extras(self, extras: dict) -> None:
        """Update cached extras from the training loop.

        Args:
            extras: Dict of extra metrics (reward_info, etc.)
        """
        self._extras.update(extras)

    def collect(
        self,
        iteration: int,
        elapsed_hours: float,
        reward_info: dict | None = None,
        lr: float = 0.0,
    ) -> MetricsSnapshot:
        """Collect current training metrics into a snapshot.

        Args:
            iteration: Current training iteration number.
            elapsed_hours: Hours since training started.
            reward_info: Dict of reward metrics from RSL-RL logging.
            lr: Current learning rate.

        Returns:
            MetricsSnapshot with all current metrics.
        """
        reward_info = reward_info or self._extras

        snapshot = MetricsSnapshot(
            iteration=iteration,
            elapsed_hours=elapsed_hours,
            learning_rate=lr,
        )

        # Parse reward info
        snapshot.mean_reward = reward_info.get("Mean reward", 0.0)
        snapshot.mean_episode_length = reward_info.get("Mean episode length", 0.0)
        snapshot.value_loss = reward_info.get("Value function loss", 0.0)
        snapshot.policy_loss = reward_info.get("Surrogate loss", 0.0)

        # Termination metrics
        survival = reward_info.get("Episode_Termination/time_out", 0.0)
        flip = reward_info.get("Episode_Termination/body_flip", 0.0)
        total_terms = survival + flip + reward_info.get("Episode_Termination/out_of_bounds", 0.0)
        if total_terms > 0:
            snapshot.survival_rate = survival / total_terms
            snapshot.flip_rate = flip / total_terms

        # Terrain level
        snapshot.mean_terrain_level = reward_info.get("Curriculum/terrain_levels", 0.0)

        # Noise std — read from policy
        try:
            policy = self.runner.alg.policy
            if hasattr(policy, "std"):
                std_val = policy.std.data
                snapshot.noise_std = std_val.mean().item() if std_val.numel() > 1 else std_val.item()
        except Exception:
            pass

        # Reward breakdown
        for key, value in reward_info.items():
            if key.startswith("Episode_Reward/"):
                term_name = key.split("/", 1)[1]
                snapshot.reward_breakdown[term_name] = value

        # Current weights — read from reward manager
        snapshot.current_weights = self._read_weights()

        # Nav-specific metrics
        snapshot.mean_forward_distance = reward_info.get("Nav/forward_distance", 0.0)
        snapshot.mean_body_height = reward_info.get("Nav/body_height", 0.0)
        snapshot.mean_drag_penalty = reward_info.get("Nav/drag_penalty", 0.0)

        # NaN check
        snapshot.has_nan = self._check_nan()
        snapshot.value_loss_spike = snapshot.value_loss > self.emergency_value_loss

        # Compute trends
        self.history.append(snapshot)
        snapshot.reward_trend = self._compute_trend("mean_reward")
        snapshot.terrain_trend = self._compute_trend("mean_terrain_level")
        snapshot.value_loss_trend = self._compute_trend("value_loss")

        return snapshot

    def _read_weights(self) -> dict[str, float]:
        """Read current reward weights from the environment's reward manager."""
        weights = {}
        reward_manager = getattr(self.env, "reward_manager", None)
        if reward_manager is None:
            env_unwrapped = getattr(self.env, "unwrapped", self.env)
            reward_manager = getattr(env_unwrapped, "reward_manager", None)
        if reward_manager is None:
            return weights

        term_names = getattr(reward_manager, "_term_names", [])
        term_cfgs = getattr(reward_manager, "_term_cfgs", [])
        for name, cfg in zip(term_names, term_cfgs):
            weights[name] = cfg.weight

        return weights

    def _check_nan(self) -> bool:
        """Check all policy parameters for NaN or Inf values."""
        try:
            policy = self.runner.alg.policy
            for name, param in policy.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"[METRICS] NaN/Inf detected in parameter: {name}")
                    return True
        except Exception:
            pass
        return False

    def _compute_trend(self, attr: str, window: int = 50) -> float:
        """Compute linear trend slope over recent history.

        Args:
            attr: MetricsSnapshot attribute name.
            window: Number of recent snapshots to use.

        Returns:
            Slope of linear fit, or 0.0 if insufficient data.
        """
        if len(self.history) < 3:
            return 0.0

        recent = list(self.history)[-window:]
        values = [getattr(s, attr, 0.0) for s in recent]
        if len(values) < 3:
            return 0.0

        try:
            x = np.arange(len(values), dtype=np.float64)
            coeffs = np.polyfit(x, values, deg=1)
            return float(coeffs[0])
        except Exception:
            return 0.0

    def is_plateau(
        self, metric: str = "mean_terrain_level", min_snapshots: int = 300, threshold: float = 0.15
    ) -> bool:
        """Detect if a metric has stalled (range < threshold over window).

        Args:
            metric: MetricsSnapshot attribute to check.
            min_snapshots: Minimum history length before checking. Default 300.
            threshold: Maximum range to consider "stalled". Default 0.15.

        Returns:
            True if metric is plateaued.
        """
        if len(self.history) < min_snapshots:
            return False

        recent = list(self.history)[-min_snapshots:]
        values = [getattr(s, metric, 0.0) for s in recent]
        return (max(values) - min(values)) < threshold

    def get_recent(self, n: int = 5) -> list[MetricsSnapshot]:
        """Get the N most recent snapshots.

        Args:
            n: Number of snapshots to return.

        Returns:
            List of recent MetricsSnapshot objects.
        """
        return list(self.history)[-n:]
