"""Pedipulation command generators — foot targets and leg selection.

FootTargetCommand: 3D body-frame target positions for the active front foot.
LegSelectionCommand: Binary flags selecting which front leg to use.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class FootTargetCommand(CommandTerm):
    """3D body-frame foot target positions for pedipulation.

    Samples target positions within the front leg workspace.
    When a specific leg is active (from leg_selection command),
    constrains y-axis to the correct side of the body.
    """

    cfg: FootTargetCommandCfg

    def __init__(self, cfg: FootTargetCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.target = torch.zeros(self.num_envs, 3, device=self.device)

    def __str__(self) -> str:
        return (
            f"FootTargetCommand(x={self.cfg.x_range}, "
            f"y={self.cfg.y_range}, z={self.cfg.z_range})"
        )

    @property
    def command(self) -> torch.Tensor:
        return self.target

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        if n == 0:
            return

        dev = self.device

        # Sample x, z uniformly from workspace
        x = torch.empty(n, device=dev).uniform_(self.cfg.x_range[0], self.cfg.x_range[1])
        z = torch.empty(n, device=dev).uniform_(self.cfg.z_range[0], self.cfg.z_range[1])

        # Sample y magnitude
        y = torch.empty(n, device=dev).uniform_(self.cfg.y_range[0], self.cfg.y_range[1])

        # Constrain y sign based on active leg for better training signal
        try:
            leg_cmd = self._env.command_manager.get_command("leg_selection")
            left_active = leg_cmd[env_ids, 0] > 0.5
            right_active = leg_cmd[env_ids, 1] > 0.5

            # Left leg workspace → positive y
            y[left_active] = y[left_active].abs()
            # Right leg workspace → negative y
            y[right_active] = -y[right_active].abs()
            # Walking mode: y stays as sampled (either side)
        except (KeyError, AttributeError):
            pass  # leg_selection not yet initialized, use unconstrained y

        self.target[env_ids, 0] = x
        self.target[env_ids, 1] = y
        self.target[env_ids, 2] = z

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass


class LegSelectionCommand(CommandTerm):
    """Binary leg selection flags for pedipulation.

    Outputs [a_left, a_right] where:
    - [0, 0] = walking mode (no manipulation, all four legs locomote)
    - [1, 0] = left front leg active (pedipulation mode)
    - [0, 1] = right front leg active (pedipulation mode)

    The standing_fraction controls how often the robot is in walking mode:
    - standing_fraction=0.6 → 60% walking, 20% left, 20% right
    - standing_fraction=0.3 → 30% walking, 35% left, 35% right
    """

    cfg: LegSelectionCommandCfg

    def __init__(self, cfg: LegSelectionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.flags = torch.zeros(self.num_envs, 2, device=self.device)

    def __str__(self) -> str:
        return f"LegSelectionCommand(standing_fraction={self.cfg.standing_fraction})"

    @property
    def command(self) -> torch.Tensor:
        return self.flags

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        if n == 0:
            return

        dev = self.device
        sf = self.cfg.standing_fraction
        rand = torch.rand(n, device=dev)

        left = torch.zeros(n, device=dev)
        right = torch.zeros(n, device=dev)

        # standing_fraction → [0, 0] (walking mode)
        # (1-sf)/2 → [1, 0] (left front leg active)
        # (1-sf)/2 → [0, 1] (right front leg active)
        left_mask = (rand >= sf) & (rand < (1.0 + sf) / 2.0)
        right_mask = rand >= (1.0 + sf) / 2.0

        left[left_mask] = 1.0
        right[right_mask] = 1.0

        self.flags[env_ids, 0] = left
        self.flags[env_ids, 1] = right

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass


# =============================================================================
# Config classes
# =============================================================================


@configclass
class FootTargetCommandCfg(CommandTermCfg):
    """Configuration for body-frame foot target positions."""

    class_type: type = FootTargetCommand
    asset_name: str = "robot"

    # Workspace bounds (body frame, meters)
    x_range: tuple[float, float] = (0.20, 0.55)   # forward reach
    y_range: tuple[float, float] = (-0.20, 0.20)   # lateral reach
    z_range: tuple[float, float] = (-0.35, 0.10)   # vertical reach

    resampling_time_range: tuple[float, float] = (3.0, 8.0)
    debug_vis: bool = False


@configclass
class LegSelectionCommandCfg(CommandTermCfg):
    """Configuration for leg selection flags."""

    class_type: type = LegSelectionCommand
    asset_name: str = "robot"

    # Fraction of time in walking mode (flags=[0,0])
    # Remaining time split 50/50 between left and right
    standing_fraction: float = 0.6

    resampling_time_range: tuple[float, float] = (5.0, 15.0)
    debug_vis: bool = False
