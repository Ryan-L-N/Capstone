"""
MPPI Navigator
==============
Standalone Model Predictive Path Integral (MPPI) navigation controller
for Spot waypoint navigation in Isaac Sim.

Replaces the RL NavigationPolicy + FSM stack entirely.  No neural network
checkpoint required — MPPI computes velocity commands fresh at every 20 Hz
nav step by sampling K unicycle trajectories and returning the
information-theoretic weighted optimal command.

Interface mirrors SkillNavLiteNavigator so the arena driver can swap in
MPPINavigator with a single line change.

Author: Cole (MS for Autonomy Project)
MPPI implementation: April 2026
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Kinematic forward model (unicycle, body frame)
# ---------------------------------------------------------------------------
#   x_world += (vx * cos(yaw) - vy * sin(yaw)) * dt
#   y_world += (vx * sin(yaw) + vy * cos(yaw)) * dt
#   yaw     += omega * dt
# ---------------------------------------------------------------------------


class MPPINavigator:
    """
    MPPI-based waypoint navigator.

    At every call to ``solve()``, this class:
    1. Draws K independent Gaussian perturbation sequences around a nominal
       zero trajectory (warm-started from the previous step's solution).
    2. Clips all samples to the physical action bounds.
    3. Simulates each trajectory forward H steps with a unicycle model
       (vectorised NumPy, no GPU required).
    4. Evaluates a composite cost per trajectory:
          J = w_goal    * dist(pos_H, goal)            [terminal]
            + w_heading * sum_t heading_err(t)^2       [running]
            + w_obs     * sum_t sum_n max(0, r_safe - clearance(t,n))^2  [running]
            + w_bound   * sum_t I(outside_arena(t))    [running]
    5. Computes MPPI weights: exp(-J / temperature), normalised.
    6. Returns the weighted-optimal first command and warm-starts next step.

    Parameters (tuning guide)
    -------------------------
    horizon : int
        Lookahead steps.  horizon * dt = lookahead time.
        Default: 25 steps × 0.05 s = 1.25 s lookahead.
        Increase if the robot overshoots waypoints in open space.
        Decrease if compute is tight (target <15 ms per solve on CPU).

    num_samples : int
        Trajectory count K.  More samples → smoother planning but more CPU.
        512 runs in ~8 ms on a modern CPU.  Use 256 for headless batch eval.

    temperature : float
        MPPI temperature λ.  Controls sharpness of trajectory weighting.
        Lower λ selects the best trajectory more aggressively.
        Higher λ blends more trajectories (smoother but may ignore obstacles).
        Good starting range: 0.03 – 0.10.

    dt : float
        Control timestep in seconds.  Must match the nav-step period in the
        arena driver (default 20 Hz → dt = 0.05 s).

    sigma_vx / sigma_vy / sigma_omega : float
        Per-channel noise standard deviation for perturbation sampling.
        Larger sigma → more exploratory; smaller → tighter around nominal.

    r_safe : float
        Obstacle safety radius in metres.  Trajectories that pass within
        r_safe of any obstacle centre are penalised quadratically.
        Default 2.0 m gives a comfortable margin around the bounding radius.

    w_goal / w_heading / w_obs / w_bound : float
        Cost weights.  Raise w_obs if the robot clips obstacles.
        Raise w_goal if the robot meanders instead of heading toward waypoints.
    """

    def __init__(
        self,
        # MPPI hyperparameters
        horizon: int = 25,
        num_samples: int = 512,
        temperature: float = 0.03,
        dt: float = 0.05,
        # Noise std per control channel [vx, vy, omega]
        sigma_vx: float = 0.8,
        sigma_vy: float = 0.0,
        sigma_omega: float = 0.25,
        # Action bounds (must match arena driver)
        vx_min: float = 0.0,
        vx_max: float = 3.0,
        vy_min: float = -0.5,
        vy_max: float = 0.5,
        omega_min: float = -1.5,
        omega_max: float = 1.5,
        # Cost weights
        w_goal: float = 20.0,
        w_heading: float = 1.5,
        w_obs: float = 80.0,
        w_bound: float = 50.0,
        r_safe: float = 0.4,
        # Arena radius (used for boundary cost)
        arena_radius: float = 25.0,
        arena_boundary_margin: float = 1.5,
    ):
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.dt = dt

        self._sigma = np.array([sigma_vx, sigma_vy, sigma_omega], dtype=np.float32)

        self._act_min = np.array([vx_min,    vy_min,    omega_min],  dtype=np.float32)
        self._act_max = np.array([vx_max,    vy_max,    omega_max],  dtype=np.float32)

        self.w_goal    = w_goal
        self.w_heading = w_heading
        self.w_obs     = w_obs
        self.w_bound   = w_bound
        self.r_safe    = r_safe

        self._arena_radius = arena_radius
        self._arena_limit  = arena_radius - arena_boundary_margin

        # Warm-start nominal sequence: (H, 3) — zero at startup
        self._nominal = np.zeros((horizon, 3), dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Call at the start of every episode to clear warm-start state."""
        self._nominal[:] = 0.0

    def solve(
        self,
        pos: np.ndarray,
        yaw: float,
        target: np.ndarray,
        obstacles: list,
    ) -> np.ndarray:
        """
        Compute a collision-avoiding velocity command [vx, vy, omega].

        Args:
            pos:       Current xy position, shape (2,)
            yaw:       Current heading in radians
            target:    Goal xy position, shape (2,)
            obstacles: List of (ox, oy, radius) tuples — bounding circles

        Returns:
            np.ndarray shape (3,) — [vx, vy, omega] clipped to action bounds
        """
        # Sample K × H × 3 Gaussian perturbations
        eps = (np.random.randn(self.num_samples, self.horizon, 3)
               .astype(np.float32))
        eps *= self._sigma  # broadcast over (K, H, 3)

        # Perturbed control sequences: (K, H, 3)
        U = self._nominal[np.newaxis, :, :] + eps
        U = np.clip(U, self._act_min, self._act_max)

        # Evaluate trajectories → (K,) costs
        costs = self._rollout(pos, yaw, target, obstacles, U)

        # MPPI weights with numerical stability
        costs -= costs.min()
        weights = np.exp(-costs / (self.temperature + 1e-8))
        weights /= weights.sum() + 1e-8

        # Weighted sum → optimal sequence: (H, 3)
        optimal = np.einsum("k,kth->th", weights, U)
        optimal = np.clip(optimal, self._act_min, self._act_max)

        # Warm-start: shift left by one step, repeat last for continuity
        self._nominal = np.roll(optimal, -1, axis=0)
        self._nominal[-1] = optimal[-1]

        return optimal[0].copy()

    # ------------------------------------------------------------------
    # Internal rollout (vectorised)
    # ------------------------------------------------------------------

    def _rollout(
        self,
        pos0: np.ndarray,
        yaw0: float,
        target: np.ndarray,
        obstacles: list,
        U: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorised forward simulation over K trajectories.

        Args:
            pos0:      Initial position [x, y]
            yaw0:      Initial heading (radians)
            target:    Goal position [x, y]
            obstacles: List of (ox, oy, radius) tuples
            U:         Control samples (K, H, 3)

        Returns:
            costs: np.ndarray (K,) — total cost per trajectory
        """
        K = self.num_samples
        tx, ty = float(target[0]), float(target[1])

        # State: (K,) each
        x   = np.full(K, float(pos0[0]), dtype=np.float32)
        y   = np.full(K, float(pos0[1]), dtype=np.float32)
        yaw = np.full(K, float(yaw0),    dtype=np.float32)
        costs = np.zeros(K, dtype=np.float32)

        # Obstacle matrix (N, 3): [ox, oy, radius]
        if obstacles:
            obs_arr = np.array(obstacles, dtype=np.float32)   # (N, 3)
            ox_  = obs_arr[:, 0]
            oy_  = obs_arr[:, 1]
            or_  = obs_arr[:, 2]
        else:
            obs_arr = None

        h_scale = 1.0 / self.horizon  # normalise running costs by horizon

        for t in range(self.horizon):
            vx    = U[:, t, 0]    # (K,)
            vy    = U[:, t, 1]
            omega = U[:, t, 2]

            # Unicycle kinematics (body → world)
            cos_y = np.cos(yaw)
            sin_y = np.sin(yaw)
            x   += (vx * cos_y - vy * sin_y) * self.dt
            y   += (vx * sin_y + vy * cos_y) * self.dt
            yaw += omega * self.dt

            # --- Heading cost ---
            dx      = tx - x
            dy      = ty - y
            desired_yaw = np.arctan2(dy, dx)                         # (K,)
            h_err   = np.arctan2(np.sin(desired_yaw - yaw),
                                 np.cos(desired_yaw - yaw))          # (K,)
            costs  += self.w_heading * h_err ** 2 * h_scale

            # --- Obstacle cost ---
            if obs_arr is not None:
                # dist_c: (K, N) Euclidean to each obstacle centre
                dist_c = np.sqrt(
                    (x[:, np.newaxis] - ox_[np.newaxis, :]) ** 2 +
                    (y[:, np.newaxis] - oy_[np.newaxis, :]) ** 2
                )
                clearance   = dist_c - or_[np.newaxis, :]           # (K, N)
                penetration = np.maximum(0.0, self.r_safe - clearance)
                costs += self.w_obs * np.sum(penetration ** 2, axis=1)

            # --- Arena boundary cost ---
            dist_from_centre = np.sqrt(x ** 2 + y ** 2)
            outside = (dist_from_centre > self._arena_limit).astype(np.float32)
            costs += self.w_bound * outside

        # --- Terminal goal cost ---
        costs += self.w_goal * np.sqrt((tx - x) ** 2 + (ty - y) ** 2)

        return costs
