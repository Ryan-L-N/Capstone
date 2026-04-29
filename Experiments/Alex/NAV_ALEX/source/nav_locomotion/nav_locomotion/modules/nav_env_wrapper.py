"""Navigation environment wrapper — translates velocity commands to joint actions.

This wrapper sits between RSL-RL and the Isaac Lab environment. The navigation
policy outputs 3-dim velocity commands [vx, vy, wz] at 10 Hz. The wrapper:

1. Collects the loco observation vector from the Isaac Lab env
2. Injects the nav policy's velocity command into the loco obs at indices [9,10,11]
3. Runs the frozen loco policy 5x (loco at 50 Hz, nav at 10 Hz)
4. Returns nav observations (depth image + proprioception) and accumulated rewards

The wrapper exposes num_obs=4108, num_actions=3 to RSL-RL, hiding the internal
12-dim joint action space of the loco policy.

Architecture:
    RSL-RL OnPolicyRunner
        -> ActorCriticCNN: obs(4108) -> vel_cmd(3)
        -> NavEnvWrapper.step(vel_cmd)
            -> FrozenLocoPolicy: loco_obs(235) + vel_cmd(3) -> joints(12)
            -> Isaac Lab env.step(joints)
        -> nav_obs(4108), reward, done, info
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import numpy as np

from nav_locomotion.modules.loco_wrapper import FrozenLocoPolicy


@dataclass
class NavEnvInfo:
    """Minimal env info interface for RSL-RL compatibility."""
    num_obs: int
    num_actions: int
    num_envs: int
    max_episode_length: int
    device: str


class NavEnvWrapper:
    """Wraps an Isaac Lab ManagerBasedRLEnv to provide a velocity-command interface.

    The wrapper intercepts step() calls: instead of passing actions directly to the
    env (which expects 12-dim joint positions), it routes 3-dim velocity commands
    through the frozen loco policy to produce joint actions.

    The loco policy runs at 50 Hz (5 steps per nav step at 10 Hz). Each nav step
    accumulates rewards across all 5 loco sub-steps.

    Args:
        env: Isaac Lab ManagerBasedRLEnv instance.
        loco_policy: Frozen Phase B locomotion policy.
        loco_steps_per_nav: Number of loco steps per nav step. Default 5.
        vel_cmd_ranges: Dict of velocity command clipping ranges.
    """

    # Observation layout
    DEPTH_DIMS = 64 * 64  # 4096
    PROPRIO_DIMS = 12     # lin_vel(3) + ang_vel(3) + gravity(3) + prev_action(3)
    NAV_OBS_DIMS = DEPTH_DIMS + PROPRIO_DIMS  # 4108
    NAV_ACTION_DIMS = 3   # vx, vy, wz

    def __init__(
        self,
        env,
        loco_policy: FrozenLocoPolicy,
        loco_steps_per_nav: int = 5,
        vel_cmd_ranges: dict | None = None,
    ):
        self.env = env
        self.loco_policy = loco_policy
        self.loco_steps_per_nav = loco_steps_per_nav

        # Velocity command clipping ranges
        if vel_cmd_ranges is None:
            vel_cmd_ranges = {
                "vx": (-1.0, 3.0),
                "vy": (-1.5, 1.5),
                "wz": (-2.0, 2.0),
            }
        self.vel_cmd_ranges = vel_cmd_ranges

        # Unwrap gym wrappers to access Isaac Lab env attributes
        self._unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env

        # Expose to RSL-RL
        self.num_obs = self.NAV_OBS_DIMS
        self.num_actions = self.NAV_ACTION_DIMS
        self.num_envs = self._unwrapped.num_envs
        self.device = self._unwrapped.device
        self.max_episode_length = getattr(self._unwrapped, "max_episode_length", 300)

        # Previous velocity command for smoothness tracking
        self._prev_vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)

        # Reward manager reference (for AI coach to modify weights)
        self.reward_manager = getattr(self._unwrapped, "reward_manager", None)

        # Track episode X-distance
        self._episode_start_x = torch.zeros(self.num_envs, device=self.device)

        print(
            f"[NavEnvWrapper] Initialized: "
            f"num_envs={self.num_envs}, nav_obs={self.num_obs}, "
            f"nav_actions={self.num_actions}, loco_steps={self.loco_steps_per_nav}"
        )

    def _clip_vel_cmd(self, vel_cmd: torch.Tensor) -> torch.Tensor:
        """Clip velocity commands to safe ranges.

        Args:
            vel_cmd: Raw velocity commands, shape (N, 3).

        Returns:
            Clipped commands, shape (N, 3).
        """
        clipped = vel_cmd.clone()
        clipped[:, 0] = torch.clamp(clipped[:, 0], *self.vel_cmd_ranges["vx"])
        clipped[:, 1] = torch.clamp(clipped[:, 1], *self.vel_cmd_ranges["vy"])
        clipped[:, 2] = torch.clamp(clipped[:, 2], *self.vel_cmd_ranges["wz"])
        return clipped

    def _build_nav_obs(self) -> torch.Tensor:
        """Build navigation observation from env state.

        Collects depth image + proprioception and concatenates into flat vector.

        Returns:
            Nav observation tensor, shape (N, 4108).
        """
        # Get the policy observation group from the env
        obs_dict = self._unwrapped.observation_manager.compute()
        if isinstance(obs_dict, dict):
            # ManagerBasedRLEnv returns dict with group names
            policy_obs = obs_dict.get("policy", None)
            if policy_obs is None:
                # Fallback: first group
                policy_obs = next(iter(obs_dict.values()))
        else:
            policy_obs = obs_dict

        return policy_obs

    def _get_loco_obs(self) -> torch.Tensor:
        """Get the full 235-dim loco observation vector from the env.

        This reads directly from the robot's state — base velocity, angular velocity,
        gravity, joint positions, joint velocities, actions, and height scan.

        Returns:
            Loco observation tensor, shape (N, 235).
        """
        robot = self._unwrapped.scene["robot"]
        height_scanner = self._unwrapped.scene.sensors["height_scanner"]

        # Proprioception (48 dims)
        base_lin_vel = robot.data.root_lin_vel_b  # (N, 3) body-frame
        base_ang_vel = robot.data.root_ang_vel_b  # (N, 3) body-frame
        projected_gravity = robot.data.projected_gravity_b  # (N, 3)
        # Velocity commands placeholder (will be overwritten by loco_wrapper)
        vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos  # (N, 12)
        joint_vel = robot.data.joint_vel  # (N, 12)
        last_action = self._unwrapped.action_manager.action  # (N, 12) last joint action

        # Height scan (187 dims)
        scan_data = height_scanner.data.ray_hits_w[..., 2]
        scanner_z = height_scanner.data.pos_w[:, 2:3]
        height_scan = scanner_z - scan_data  # Relative heights
        height_scan = torch.nan_to_num(height_scan, nan=0.0)
        height_scan = torch.clamp(height_scan, -1.0, 1.0)

        loco_obs = torch.cat([
            base_lin_vel,       # 3
            base_ang_vel,       # 3
            projected_gravity,  # 3
            vel_cmd,            # 3 (placeholder)
            joint_pos,          # 12
            joint_vel,          # 12
            last_action,        # 12
            height_scan,        # 187
        ], dim=-1)

        return loco_obs

    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset all environments and return initial nav observations.

        Returns:
            nav_obs: Initial observation, shape (N, 4108).
            info: Additional info dict.
        """
        obs_dict, info = self.env.reset()

        # Track episode start X position
        robot = self._unwrapped.scene["robot"]
        self._episode_start_x = robot.data.root_pos_w[:, 0].clone()
        self._prev_vel_cmd.zero_()

        nav_obs = self._build_nav_obs()
        return nav_obs, info

    def step(self, vel_cmd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Execute one nav step (multiple loco sub-steps).

        Args:
            vel_cmd: Velocity commands [vx, vy, wz], shape (N, 3).

        Returns:
            nav_obs: Next observation, shape (N, 4108).
            reward: Accumulated reward over loco sub-steps, shape (N,).
            done: Episode termination flags, shape (N,).
            info: Additional info dict.
        """
        vel_cmd = self._clip_vel_cmd(vel_cmd)

        # Store for smoothness tracking
        self._prev_vel_cmd = vel_cmd.clone()

        # Get loco observation and run frozen loco policy
        loco_obs = self._get_loco_obs()
        joint_actions = self.loco_policy(loco_obs, vel_cmd)

        # Step the Isaac Lab env with joint actions
        # The env's decimation handles physics sub-stepping internally
        obs_dict, reward, terminated, truncated, info = self.env.step(joint_actions)

        done = terminated | truncated

        # Build nav observations (same format as get_observations)
        nav_obs = self._build_nav_obs()

        # Reset tracking for terminated envs
        if done.any():
            robot = self._unwrapped.scene["robot"]
            self._episode_start_x[done] = robot.data.root_pos_w[done, 0]
            self._prev_vel_cmd[done] = 0.0

        class _ObsDict(dict):
            def to(self, device):
                return _ObsDict({k: v.to(device) if hasattr(v, 'to') else v for k, v in self.items()})

        return _ObsDict({"policy": nav_obs}), reward, done, info

    def get_observations(self):
        """Get current nav observations (RSL-RL compatibility).

        Returns:
            TensorDict-like dict with "policy" key containing obs tensor.
        """
        obs = self._build_nav_obs()

        class _ObsDict(dict):
            """Dict that supports .to() for RSL-RL compatibility."""
            def to(self, device):
                return _ObsDict({k: v.to(device) if hasattr(v, 'to') else v for k, v in self.items()})

        return _ObsDict({"policy": obs})

    def close(self) -> None:
        """Close the underlying environment."""
        self.env.close()

    @property
    def unwrapped(self):
        """Access the underlying Isaac Lab env."""
        return self.env.unwrapped

    @property
    def extras(self) -> dict:
        """Extra info from the env (for metrics collection)."""
        return getattr(self._unwrapped, "extras", {})

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Episode length buffer (RSL-RL compatibility)."""
        return getattr(self._unwrapped, "episode_length_buf",
                       torch.zeros(self.num_envs, device=self.device, dtype=torch.long))

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        """Set episode length buffer on the underlying env."""
        if hasattr(self._unwrapped, "episode_length_buf"):
            self._unwrapped.episode_length_buf = value
