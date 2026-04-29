"""
Training Environment 1 — Locomotion Command Policy
Pure NumPy feedforward network + Cross-Entropy Method (CEM) trainer.
No PyTorch dependency required — runs inside the IsaacSim Python 3.11 venv.

Architecture:
  Input(11) -> Linear(64) -> Tanh -> Linear(64) -> Tanh -> Linear(3) -> Tanh
  Output scaled to [vx: 0..VX_MAX, vy: ±VY_LIMIT, yaw: ±YAW_LIMIT]
"""

import os
import numpy as np
from env_config import EnvConfig, config as default_config


class LocoPolicy:
    """
    Two-hidden-layer feedforward network mapping obs(11) -> cmd(3).
    All arithmetic is NumPy — no external ML framework needed.
    """

    OBS_DIM = 11
    HIDDEN   = 64
    ACT_DIM  = 3

    def __init__(self, cfg: EnvConfig = default_config, seed: int = None):
        self.cfg = cfg
        rng = np.random.default_rng(seed)

        def _xavier(fan_in: int, fan_out: int) -> np.ndarray:
            std = np.sqrt(2.0 / fan_in)
            return rng.standard_normal((fan_out, fan_in)) * std

        self.W1 = _xavier(self.OBS_DIM, self.HIDDEN)
        self.b1 = np.zeros(self.HIDDEN)
        self.W2 = _xavier(self.HIDDEN, self.HIDDEN)
        self.b2 = np.zeros(self.HIDDEN)
        self.W3 = _xavier(self.HIDDEN, self.ACT_DIM)
        self.b3 = np.zeros(self.ACT_DIM)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """
        Run one forward pass.

        Args:
            obs : shape (11,) observation vector

        Returns:
            cmd : shape (3,) velocity command [vx, vy, yaw], already scaled
        """
        x = np.tanh(self.W1 @ obs + self.b1)
        x = np.tanh(self.W2 @ x  + self.b2)
        raw = np.tanh(self.W3 @ x + self.b3)   # all values in [-1, 1]
        return self._scale(raw)

    def _scale(self, raw: np.ndarray) -> np.ndarray:
        """Map tanh outputs [-1, 1] to physical action ranges."""
        cfg = self.cfg
        # vx: [-1,1] -> [0, VX_MAX]  (non-negative only — always walk forward)
        vx  = (raw[0] + 1.0) / 2.0 * cfg.VX_MAX
        # vy: [-1,1] -> [-VY_LIMIT, +VY_LIMIT]
        vy  = raw[1] * cfg.VY_LIMIT
        # yaw: [-1,1] -> [-YAW_LIMIT, +YAW_LIMIT]
        yaw = raw[2] * cfg.YAW_LIMIT
        return np.array([vx, vy, yaw], dtype=np.float64)

    # ------------------------------------------------------------------
    # Parameter vector interface (used by CEM)
    # ------------------------------------------------------------------

    def get_params(self) -> np.ndarray:
        """Flatten all weights and biases into a single 1-D vector."""
        return np.concatenate([
            self.W1.flatten(), self.b1,
            self.W2.flatten(), self.b2,
            self.W3.flatten(), self.b3,
        ])

    def set_params(self, params: np.ndarray):
        """Unpack a flat parameter vector back into network weights."""
        cursor = 0

        def _take(n: int) -> np.ndarray:
            nonlocal cursor
            chunk = params[cursor:cursor + n]
            cursor += n
            return chunk

        self.W1 = _take(self.HIDDEN * self.OBS_DIM).reshape(self.HIDDEN, self.OBS_DIM)
        self.b1 = _take(self.HIDDEN)
        self.W2 = _take(self.HIDDEN * self.HIDDEN).reshape(self.HIDDEN, self.HIDDEN)
        self.b2 = _take(self.HIDDEN)
        self.W3 = _take(self.ACT_DIM * self.HIDDEN).reshape(self.ACT_DIM, self.HIDDEN)
        self.b3 = _take(self.ACT_DIM)

    @property
    def param_dim(self) -> int:
        return (
            self.HIDDEN * self.OBS_DIM + self.HIDDEN +
            self.HIDDEN * self.HIDDEN  + self.HIDDEN +
            self.ACT_DIM * self.HIDDEN + self.ACT_DIM
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save weights to a .npz file. Extension is added if missing."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)
        print(f"[LocoPolicy] Saved to {path}.npz")

    def load(self, path: str):
        """Load weights from a .npz file."""
        npz = path if path.endswith(".npz") else path + ".npz"
        if not os.path.exists(npz):
            print(f"[LocoPolicy] No checkpoint at {npz}, keeping current weights.")
            return
        data = np.load(npz)
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], data["b2"]
        self.W3, self.b3 = data["W3"], data["b3"]
        print(f"[LocoPolicy] Loaded from {npz}")


# ---------------------------------------------------------------------------

class CEMTrainer:
    """
    Cross-Entropy Method trainer for LocoPolicy.

    Maintains a Gaussian distribution over the policy parameter space.
    Each generation:
      1. Sample `population_size` parameter vectors.
      2. Caller evaluates each and returns a list of scalar rewards.
      3. Keep top `elite_frac` fraction (elites).
      4. Refit mean and std from elites.
      5. Load best params into the policy.

    Usage
    -----
        cem = CEMTrainer(policy)
        for gen in range(N_GENS):
            population = cem.sample_population()   # list of param arrays
            rewards = [evaluate(p) for p in population]
            best, mean = cem.update(rewards)
    """

    def __init__(
        self,
        policy: LocoPolicy,
        population_size: int = 20,
        elite_frac: float = 0.2,
        noise_std: float = 0.05,
        noise_decay: float = 0.999,
        min_noise: float = 0.01,
        seed: int = None,
    ):
        self.policy = policy
        self.population_size = population_size
        self.n_elite = max(1, int(population_size * elite_frac))
        self.noise_std = noise_std
        self.noise_decay = noise_decay
        self.min_noise = min_noise
        self.rng = np.random.default_rng(seed)

        # Initialize distribution from current policy
        self.mean = policy.get_params().copy()
        self.std  = np.full_like(self.mean, noise_std)

        self._population: list = []
        self._generation: int = 0

    def sample_population(self) -> list:
        """
        Draw `population_size` parameter vectors from N(mean, std).
        The first entry is always the current mean (exploitation).

        Returns list of np.ndarray, each shape (param_dim,).
        """
        self._population = [self.mean.copy()]
        for _ in range(self.population_size - 1):
            noise = self.rng.standard_normal(len(self.mean)) * self.std
            self._population.append(self.mean + noise)
        return self._population

    def update(self, rewards: list) -> tuple:
        """
        Refit distribution from elite subset and update the policy.

        Args:
            rewards : list of float, same length as last sample_population()

        Returns:
            best_reward  : float
            mean_reward  : float
        """
        assert len(rewards) == len(self._population), (
            f"Expected {len(self._population)} rewards, got {len(rewards)}"
        )

        indices = np.argsort(rewards)[::-1]           # descending by reward
        elites  = [self._population[i] for i in indices[:self.n_elite]]

        elite_arr = np.array(elites)
        self.mean = elite_arr.mean(axis=0)
        self.std  = elite_arr.std(axis=0) + self.min_noise

        # Decay exploration noise
        self.noise_std = max(self.min_noise, self.noise_std * self.noise_decay)

        # Apply best parameters to policy
        self.policy.set_params(self.mean.copy())

        self._generation += 1
        best_reward = float(max(rewards))
        mean_reward = float(np.mean(rewards))

        print(
            f"[CEM gen {self._generation:4d}] "
            f"best={best_reward:8.2f}  mean={mean_reward:8.2f}  "
            f"noise={self.noise_std:.4f}"
        )
        return best_reward, mean_reward
