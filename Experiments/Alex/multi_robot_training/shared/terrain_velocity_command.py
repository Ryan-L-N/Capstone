"""Terrain-Scaled Velocity Command Generator.

Wraps Isaac Lab's CommandTerm to scale velocity ranges based on
each robot's current terrain curriculum level. Robots on easy terrain get
fast commands (sprint), robots on hard terrain get slow commands (careful walk).

This teaches the policy to map height-scan patterns to appropriate speeds
proactively — the robot is never asked to sprint on level 8 stairs.

Velocity scaling (linear interpolation):
  Level 0:   lin_vel_x in [vel_x_easy_min, vel_x_easy_max]  (e.g. 0.5 to 3.0)
  Level 9:   lin_vel_x in [vel_x_hard_min, vel_x_hard_max]  (e.g. 0.0 to 1.0)
  Intermediate: linear interpolation between easy and hard ranges

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TerrainScaledVelocityCommand(CommandTerm):
    """Velocity command generator that scales ranges by terrain difficulty.

    On easy terrain (low curriculum level), commands sample from a wide, fast
    velocity range. On hard terrain (high curriculum level), commands sample
    from a narrow, slow range. Each robot gets commands appropriate to its
    current terrain difficulty.
    """

    cfg: TerrainScaledVelocityCommandCfg

    def __init__(self, cfg: TerrainScaledVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Command buffer: [vx, vy, yaw_rate]
        self.vel_command = torch.zeros(self.num_envs, 3, device=self.device)

    def __str__(self) -> str:
        return (
            f"TerrainScaledVelocityCommand(\n"
            f"  vel_x_easy=[{self.cfg.vel_x_easy_min}, {self.cfg.vel_x_easy_max}],\n"
            f"  vel_x_hard=[{self.cfg.vel_x_hard_min}, {self.cfg.vel_x_hard_max}],\n"
            f"  vel_y=[{self.cfg.vel_y_min}, {self.cfg.vel_y_max}],\n"
            f"  yaw=[{self.cfg.yaw_min}, {self.cfg.yaw_max}],\n"
            f")"
        )

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample new velocity commands scaled by terrain level."""
        n = len(env_ids)
        if n == 0:
            return

        # Get terrain levels for these environments
        terrain: TerrainImporter = self._env.scene.terrain
        if hasattr(terrain, "terrain_levels") and terrain.terrain_levels is not None:
            levels = terrain.terrain_levels[env_ids].float()
            max_level = terrain.max_terrain_level
            t = torch.clamp(levels / max_level, 0.0, 1.0)  # [0, 1]
        else:
            # No curriculum — use mid-range
            t = torch.full((n,), 0.5, device=self.device)

        # Interpolate velocity ranges based on terrain level
        # Easy (t=0) → fast range, Hard (t=1) → slow range
        vx_min = self.cfg.vel_x_easy_min + (self.cfg.vel_x_hard_min - self.cfg.vel_x_easy_min) * t
        vx_max = self.cfg.vel_x_easy_max + (self.cfg.vel_x_hard_max - self.cfg.vel_x_easy_max) * t

        # Sample uniformly within per-robot ranges
        vx = vx_min + (vx_max - vx_min) * torch.rand(n, device=self.device)
        vy = torch.empty(n, device=self.device).uniform_(self.cfg.vel_y_min, self.cfg.vel_y_max)
        yaw = torch.empty(n, device=self.device).uniform_(self.cfg.yaw_min, self.cfg.yaw_max)

        self.vel_command[env_ids, 0] = vx
        self.vel_command[env_ids, 1] = vy
        self.vel_command[env_ids, 2] = yaw

    def _update_command(self):
        """No dynamic command updates needed."""
        pass

    def _update_metrics(self):
        """No custom metrics to track."""
        pass


@configclass
class TerrainScaledVelocityCommandCfg(CommandTermCfg):
    """Configuration for terrain-scaled velocity commands."""

    class_type: type = TerrainScaledVelocityCommand

    # Forward velocity ranges (interpolated by terrain level)
    vel_x_easy_min: float = 0.5    # Level 0: min forward speed
    vel_x_easy_max: float = 3.0    # Level 0: max forward speed (sprint)
    vel_x_hard_min: float = 0.0    # Level 9: min forward speed
    vel_x_hard_max: float = 1.0    # Level 9: max forward speed (careful)

    # Lateral and yaw (same for all terrain levels)
    vel_y_min: float = -1.5
    vel_y_max: float = 1.5
    yaw_min: float = -2.0
    yaw_max: float = 2.0

    # Resampling interval (seconds)
    resampling_time_range: tuple[float, float] = (10.0, 10.0)
