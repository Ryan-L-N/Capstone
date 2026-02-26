"""
Training Environment 1 — Robot State
Observation extraction, reward tracking, and per-robot episode state.

Does NOT import Isaac at module level. All Isaac objects are passed in
as parameters, so this file is safe to import before SimulationApp if needed.
"""

import math
import numpy as np
from env_config import EnvConfig, config as default_config
from reward_fn import compute_reward


def _quat_to_roll_pitch(quat) -> tuple:
    """
    Convert IsaacSim quaternion [w, x, y, z] to roll and pitch in radians.
    Uses the ZYX (aerospace) Euler convention.
    """
    w = float(quat[0])
    x = float(quat[1])
    y = float(quat[2])
    z = float(quat[3])
    roll  = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return roll, pitch


class RobotState:
    """
    Tracks per-robot episode state, builds observations, and computes rewards.

    Parameters
    ----------
    robot_id  : int     — index among parallel robots (0..NUM_ROBOTS-1)
    spot      : object  — SpotFlatTerrainPolicy instance from the main script
    start_y   : float   — Y spawn coordinate for this robot
    friction  : float   — current static friction coefficient (fed into obs)
    cfg       : EnvConfig
    """

    def __init__(
        self,
        robot_id: int,
        spot,
        start_y: float,
        friction: float = 0.75,
        cfg: EnvConfig = default_config,
    ):
        self.id = robot_id
        self.spot = spot
        self.cfg = cfg
        self.start_pos = np.array([cfg.START_X, start_y, cfg.START_Z])
        self.friction = friction

        # Mutable state (reset each episode)
        self.prev_cmd_vx:   float = 0.0
        self.episode_reward: float = 0.0
        self.step_count:     int   = 0
        self.fell:           bool  = False
        self.out_of_bounds:  bool  = False

        # Per-step history for episode-level metric aggregation
        self._vel_x_hist:      list = []
        self._roll_hist:       list = []
        self._pitch_hist:      list = []
        self._roll_rate_hist:  list = []
        self._pitch_rate_hist: list = []
        self._cmd_vx_hist:     list = []
        self._joint_work_acc:  float = 0.0

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_observation(self) -> np.ndarray:
        """
        Build the 11-dimensional observation vector.

        Index  Feature       Description
        -----  -----------   ------------------------------------
          0    vel_x         Actual forward body velocity (m/s)
          1    vel_y         Actual lateral body velocity (m/s)
          2    vel_z         Actual vertical body velocity (m/s)
          3    ang_vel_x     Roll rate (rad/s)
          4    ang_vel_y     Pitch rate (rad/s)
          5    ang_vel_z     Yaw rate (rad/s)
          6    roll          Body roll angle (rad)
          7    pitch         Body pitch angle (rad)
          8    height        Body Z position (m) — raw, not normalized
          9    friction      Current static friction coefficient
         10    cmd_vx_prev   Last commanded forward velocity (m/s)
        """
        pos, quat = self.spot.robot.get_world_pose()
        vel       = self.spot.robot.get_linear_velocity()
        ang_vel   = self.spot.robot.get_angular_velocity()
        roll, pitch = _quat_to_roll_pitch(quat)

        return np.array([
            float(vel[0]),
            float(vel[1]),
            float(vel[2]),
            float(ang_vel[0]),
            float(ang_vel[1]),
            float(ang_vel[2]),
            roll,
            pitch,
            float(pos[2]),
            self.friction,
            self.prev_cmd_vx,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Step recording and reward
    # ------------------------------------------------------------------

    def record_step(
        self,
        cmd_vx: float,
        cmd_vy: float,
        cmd_yaw: float,
        dt: float,
    ) -> tuple:
        """
        Collect state, compute reward, update accumulators.
        Call this once per control step (50 Hz).

        Returns
        -------
        reward      : float
        parts       : dict[str, float]  — individual reward components
        fell_now    : bool              — True if the fall just occurred this step
        """
        pos, quat = self.spot.robot.get_world_pose()
        vel       = self.spot.robot.get_linear_velocity()
        ang_vel   = self.spot.robot.get_angular_velocity()
        roll, pitch = _quat_to_roll_pitch(quat)
        pos_z = float(pos[2])

        # Detect fall (only fire once per episode)
        fell_now = (pos_z < self.cfg.FALL_Z) and not self.fell

        reward, parts = compute_reward(
            vel_x=float(vel[0]),
            ang_vel_x=float(ang_vel[0]),
            ang_vel_y=float(ang_vel[1]),
            roll=roll,
            pitch=pitch,
            height=pos_z,
            cmd_vx=cmd_vx,
            cmd_vx_prev=self.prev_cmd_vx,
            cmd_vy=cmd_vy,
            cmd_yaw=cmd_yaw,
            pos_z=pos_z,
            fell=fell_now,
            cfg=self.cfg,
        )

        # Update accumulators
        self.episode_reward += reward
        self.step_count += 1
        self.prev_cmd_vx = cmd_vx

        self._vel_x_hist.append(float(vel[0]))
        self._roll_hist.append(abs(math.degrees(roll)))
        self._pitch_hist.append(abs(math.degrees(pitch)))
        self._roll_rate_hist.append(abs(float(ang_vel[0])))
        self._pitch_rate_hist.append(abs(float(ang_vel[1])))
        self._cmd_vx_hist.append(cmd_vx)

        if fell_now:
            self.fell = True
        if abs(float(pos[1])) > self.cfg.LATERAL_LIMIT:
            self.out_of_bounds = True

        # Joint work (optional — requires get_measured_joint_efforts)
        try:
            torques = self.spot.robot.get_measured_joint_efforts()
            vels    = self.spot.robot.get_joint_velocities()
            if torques is not None and vels is not None:
                self._joint_work_acc += float(np.sum(np.abs(torques * vels)) * dt)
        except Exception:
            pass   # Not all IsaacSim versions expose this; silently skip

        return reward, parts, fell_now

    # ------------------------------------------------------------------
    # Episode state queries
    # ------------------------------------------------------------------

    def is_done(self, sim_time: float) -> tuple:
        """
        Returns (done: bool, reason: str).
        reason is one of: 'fall', 'out_of_bounds', 'timeout', or ''.
        sim_time should include the stabilization period.
        """
        if self.fell:
            return True, "fall"
        if self.out_of_bounds:
            return True, "out_of_bounds"
        if sim_time >= self.cfg.STABILIZE_TIME + self.cfg.MAX_EPISODE_TIME:
            return True, "timeout"
        return False, ""

    def get_episode_summary(self, surface: str, sim_time: float) -> dict:
        """
        Aggregate per-step data into episode-level metrics for CSV logging.
        Call this after the episode ends.
        """
        mean_cmd_vx = float(np.mean(self._cmd_vx_hist)) if self._cmd_vx_hist else 0.0
        mean_vel_x  = float(np.mean(self._vel_x_hist))  if self._vel_x_hist  else 0.0
        slip_ratio  = (mean_vel_x / mean_cmd_vx) if mean_cmd_vx > 1e-6 else 0.0

        if len(self._cmd_vx_hist) > 1:
            diffs = np.diff(self._cmd_vx_hist)
            cmd_smoothness = float(np.mean(diffs ** 2))
        else:
            cmd_smoothness = 0.0

        episode_time = max(0.0, sim_time - self.cfg.STABILIZE_TIME)

        return {
            "robot_id":        self.id,
            "surface":         surface,
            "friction":        self.friction,
            "fell":            int(self.fell),
            "out_of_bounds":   int(self.out_of_bounds),
            "survival_time":   round(episode_time, 2),
            "total_reward":    round(self.episode_reward, 3),
            "mean_vel_x":      round(mean_vel_x, 4),
            "mean_cmd_vx":     round(mean_cmd_vx, 4),
            "slip_ratio":      round(slip_ratio, 4),
            "mean_roll_deg":   round(float(np.mean(self._roll_hist))  if self._roll_hist  else 0.0, 4),
            "mean_pitch_deg":  round(float(np.mean(self._pitch_hist)) if self._pitch_hist else 0.0, 4),
            "std_roll_deg":    round(float(np.std(self._roll_hist))   if self._roll_hist  else 0.0, 4),
            "std_pitch_deg":   round(float(np.std(self._pitch_hist))  if self._pitch_hist else 0.0, 4),
            "mean_roll_rate":  round(float(np.mean(self._roll_rate_hist))  if self._roll_rate_hist  else 0.0, 4),
            "mean_pitch_rate": round(float(np.mean(self._pitch_rate_hist)) if self._pitch_rate_hist else 0.0, 4),
            "total_joint_work": round(self._joint_work_acc, 4),
            "cmd_smoothness":  round(cmd_smoothness, 6),
            "step_count":      self.step_count,
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, friction: float = None):
        """
        Teleport robot to start position and clear all episode accumulators.

        Note: Does NOT call world.reset() — the caller is responsible for
        the simulation-level reset strategy. Joint states reset naturally
        during the 3-second stabilization period.
        """
        self.spot.robot.set_world_pose(
            position=self.start_pos,
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),   # Facing +X
        )
        self.spot.robot.set_linear_velocity(np.zeros(3))
        self.spot.robot.set_angular_velocity(np.zeros(3))

        if friction is not None:
            self.friction = friction

        # Clear episode state
        self.prev_cmd_vx    = 0.0
        self.episode_reward = 0.0
        self.step_count     = 0
        self.fell           = False
        self.out_of_bounds  = False

        self._vel_x_hist.clear()
        self._roll_hist.clear()
        self._pitch_hist.clear()
        self._roll_rate_hist.clear()
        self._pitch_rate_hist.clear()
        self._cmd_vx_hist.clear()
        self._joint_work_acc = 0.0
