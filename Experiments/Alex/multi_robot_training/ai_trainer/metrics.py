"""Metrics collection for AI-guided training.

Reads training metrics from RSL-RL runner and Isaac Lab environment,
produces MetricsSnapshot objects for the coach to analyze.
"""

from collections import deque
from dataclasses import dataclass, field
import numpy as np
import torch


@dataclass
class MetricsSnapshot:
    """Point-in-time training metrics for coach analysis."""
    iteration: int
    phase: str
    elapsed_hours: float

    # Core performance
    mean_reward: float
    mean_episode_length: float
    survival_rate: float         # fraction surviving to timeout
    flip_rate: float             # fraction terminated by flip

    # Terrain curriculum
    mean_terrain_level: float

    # Training health
    value_loss: float
    policy_loss: float
    noise_std: float
    learning_rate: float

    # Per-reward breakdown (term_name -> mean contribution)
    reward_breakdown: dict = field(default_factory=dict)

    # Current reward weights (term_name -> weight)
    current_weights: dict = field(default_factory=dict)

    # Trends (slope over recent window)
    reward_trend: float = 0.0
    terrain_trend: float = 0.0
    value_loss_trend: float = 0.0

    # Alert flags
    has_nan: bool = False
    value_loss_spike: bool = False

    def to_dict(self) -> dict:
        """Serialize for JSON logging."""
        return {k: v for k, v in self.__dict__.items()}


class MetricsCollector:
    """Collects and aggregates training metrics from live env/runner.

    Maintains a rolling window of snapshots for trend analysis.
    """

    def __init__(self, env, runner, phase: str, history_size: int = 200):
        self.env = env
        self.runner = runner
        self.phase = phase
        self.history_size = history_size
        self.history: deque[MetricsSnapshot] = deque(maxlen=history_size)
        self._last_extras = {}

    def update_extras(self, extras: dict):
        """Called from the training loop to capture per-iteration extras."""
        self._last_extras = extras

    def collect(self, iteration: int, elapsed_hours: float,
                reward_info: dict, lr: float) -> MetricsSnapshot:
        """Collect a full metrics snapshot at the current iteration.

        Args:
            iteration: Current training iteration.
            elapsed_hours: Wall-clock hours since training start.
            reward_info: Dict from RSL-RL's logging (ep_infos / extras).
            lr: Current learning rate.

        Returns:
            MetricsSnapshot with all metrics populated.
        """
        # -- Core metrics from reward_info --
        mean_reward = reward_info.get("Mean reward", 0.0)
        mean_ep_len = reward_info.get("Mean episode length", 0.0)

        # Termination rates
        survival = reward_info.get("Episode_Termination/time_out", 0.0)
        flip = reward_info.get("Episode_Termination/body_flip_over", 0.0)

        # Terrain
        terrain = reward_info.get("Curriculum/terrain_levels", 0.0)

        # Training health
        value_loss = reward_info.get("Mean value_function loss", 0.0)
        policy_loss = reward_info.get("Mean surrogate loss", 0.0)
        noise_std = reward_info.get("Mean action noise std", 0.0)

        # Per-reward breakdown
        reward_breakdown = {}
        for key, val in reward_info.items():
            if key.startswith("Episode_Reward/"):
                term_name = key.replace("Episode_Reward/", "")
                reward_breakdown[term_name] = val

        # Current weights from reward manager
        current_weights = self._read_weights()

        # NaN check on policy parameters
        has_nan = self._check_nan()

        # Value loss spike
        value_loss_spike = value_loss > 100.0

        snapshot = MetricsSnapshot(
            iteration=iteration,
            phase=self.phase,
            elapsed_hours=elapsed_hours,
            mean_reward=mean_reward,
            mean_episode_length=mean_ep_len,
            survival_rate=survival,
            flip_rate=flip,
            mean_terrain_level=terrain,
            value_loss=value_loss,
            policy_loss=policy_loss,
            noise_std=noise_std,
            learning_rate=lr,
            reward_breakdown=reward_breakdown,
            current_weights=current_weights,
            has_nan=has_nan,
            value_loss_spike=value_loss_spike,
        )

        # Compute trends from history
        self.history.append(snapshot)
        snapshot.reward_trend = self._compute_trend("mean_reward")
        snapshot.terrain_trend = self._compute_trend("mean_terrain_level")
        snapshot.value_loss_trend = self._compute_trend("value_loss")

        return snapshot

    def _read_weights(self) -> dict:
        """Read current reward weights from the environment's reward manager."""
        weights = {}
        try:
            env_unwrapped = self.env.unwrapped
            if hasattr(env_unwrapped, "reward_manager"):
                rm = env_unwrapped.reward_manager
                # Isaac Lab RewardManager stores terms with configs
                if hasattr(rm, "_term_cfgs"):
                    for name, cfg in rm._term_cfgs.items():
                        weights[name] = cfg.weight
                elif hasattr(rm, "active_terms"):
                    for name, (term, cfg) in rm.active_terms.items():
                        weights[name] = cfg.weight
        except Exception:
            pass
        return weights

    def _check_nan(self) -> bool:
        """Check policy parameters for NaN/Inf."""
        try:
            for name, param in self.runner.alg.policy.named_parameters():
                if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                    return True
        except Exception:
            pass
        return False

    def _compute_trend(self, attr: str, window: int = 50) -> float:
        """Compute linear regression slope over recent history."""
        if len(self.history) < 10:
            return 0.0

        values = [getattr(s, attr) for s in list(self.history)[-window:]]
        n = len(values)
        x = np.arange(n)
        try:
            slope = np.polyfit(x, values, 1)[0]
            return float(slope)
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    def is_plateau(self, metric: str = "mean_terrain_level",
                   window: int = 300, threshold: float = 0.01) -> bool:
        """Detect if a metric has stalled (near-zero slope)."""
        if len(self.history) < window:
            return False
        values = [getattr(s, metric) for s in list(self.history)[-window:]]
        try:
            slope = np.polyfit(np.arange(len(values)), values, 1)[0]
            return abs(slope) < threshold
        except (np.linalg.LinAlgError, ValueError):
            return False

    def go_no_go(self, phase_cfg) -> tuple[bool, list[str]]:
        """Check if all phase advancement criteria are met.

        Returns:
            (passed, list_of_failures)
        """
        if len(self.history) < phase_cfg.min_consecutive_iters:
            return False, ["not enough history"]

        recent = list(self.history)[-phase_cfg.min_consecutive_iters:]
        failures = []

        avg_survival = np.mean([s.survival_rate for s in recent])
        if avg_survival < phase_cfg.min_survival_rate:
            failures.append(
                f"survival {avg_survival:.1%} < {phase_cfg.min_survival_rate:.1%}")

        avg_flip = np.mean([s.flip_rate for s in recent])
        if avg_flip > phase_cfg.max_flip_rate:
            failures.append(
                f"flip {avg_flip:.1%} > {phase_cfg.max_flip_rate:.1%}")

        avg_noise = np.mean([s.noise_std for s in recent])
        if avg_noise > phase_cfg.max_noise_std_advance:
            failures.append(
                f"noise {avg_noise:.2f} > {phase_cfg.max_noise_std_advance}")

        avg_vloss = np.mean([s.value_loss for s in recent])
        if avg_vloss > phase_cfg.max_value_loss:
            failures.append(
                f"value_loss {avg_vloss:.1f} > {phase_cfg.max_value_loss}")

        avg_terrain = np.mean([s.mean_terrain_level for s in recent])
        if avg_terrain < phase_cfg.min_terrain_level:
            failures.append(
                f"terrain {avg_terrain:.2f} < {phase_cfg.min_terrain_level}")

        return len(failures) == 0, failures
