"""Mastery Curriculum — Promote-only + reward ratchet for stair climbing.

Two innovations:
1. Promote-only: After full_release_iter, robots can only move UP in terrain difficulty.
2. Mastery ratchet: Reward multiplier = current_level / highest_reached.
   Once you've been to row 6, row 4 gives only 67% reward.

Usage in train_expert.py:
    mastery = MasteryCurriculum(env, full_release_iter=700)
    # In update_with_schedule:
    mastery.step(iteration)
    reward_scale = mastery.get_reward_scale()  # (N,) tensor, 0.0-1.0

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MasteryCurriculum:
    """Promote-only curriculum + mastery reward ratchet.

    After full_release_iter:
    - Robots can only be promoted (never demoted) in terrain difficulty
    - Reward is scaled by current_level / highest_level_reached

    This forces the robot to keep pushing forward — retreating to easy
    terrain pays less, and the curriculum won't let it go back anyway.
    """

    def __init__(
        self,
        env,
        full_release_iter: int = 700,
        ratchet_min: float = 0.3,
        warmup_scale: float = 1.0,
    ):
        """
        Args:
            env: The IsaacLab environment (for terrain access).
            full_release_iter: Iteration after which promote-only activates.
            ratchet_min: Minimum reward scale (prevents zero reward on row 0).
            warmup_scale: Reward scale during warmup (before full release).
        """
        self.env = env
        self.full_release_iter = full_release_iter
        self.ratchet_min = ratchet_min
        self.warmup_scale = warmup_scale

        self._promote_only_active = False
        self._original_update_origins = None

        # Per-robot highest terrain level reached
        self._highest_level = None
        self._device = None

    def _get_terrain(self):
        """Navigate wrapper chain to get terrain importer."""
        e = self.env
        while hasattr(e, 'env'):
            e = e.env
        if hasattr(e, 'unwrapped'):
            e = e.unwrapped
        if hasattr(e, 'scene') and hasattr(e.scene, 'terrain'):
            return e.scene.terrain
        return None

    def step(self, iteration: int):
        """Call once per training iteration. Activates promote-only when ready."""
        terrain = self._get_terrain()
        if terrain is None:
            return

        # Initialize highest_level tracker
        if self._highest_level is None:
            self._device = terrain.terrain_levels.device
            self._highest_level = terrain.terrain_levels.float().clone()

        # Update highest level reached per robot
        current = terrain.terrain_levels.float()
        self._highest_level = torch.max(self._highest_level, current)

        # Activate promote-only after full actor release
        if iteration >= self.full_release_iter and not self._promote_only_active:
            self._activate_promote_only(terrain)

    def _activate_promote_only(self, terrain):
        """Monkey-patch terrain.update_env_origins to block demotion."""
        if self._original_update_origins is not None:
            return  # Already patched

        self._original_update_origins = terrain.update_env_origins
        outer = self  # Capture for closure

        def promote_only_update(env_ids, move_up, move_down):
            """Modified update: allow promotion, block demotion."""
            move_down_blocked = torch.zeros_like(move_down)
            outer._original_update_origins(env_ids, move_up, move_down_blocked)
            # Update highest level after promotion
            current = terrain.terrain_levels.float()
            if outer._highest_level is not None:
                outer._highest_level = torch.max(outer._highest_level, current)

        terrain.update_env_origins = promote_only_update
        self._promote_only_active = True
        print(f"[MASTERY] Promote-only curriculum ACTIVATED at iter {self.full_release_iter}", flush=True)

    def get_reward_scale(self) -> torch.Tensor:
        """Get per-robot reward multiplier based on mastery ratchet.

        Returns (N,) tensor where:
            scale = max(ratchet_min, current_level / highest_reached)
            = 1.0 if at highest level (full reward)
            = 0.67 if 2 rows below peak
            = ratchet_min floor to prevent zero reward

        Before full_release_iter, returns warmup_scale for all robots.
        """
        if self._highest_level is None or not self._promote_only_active:
            terrain = self._get_terrain()
            if terrain is None:
                return torch.ones(1)
            return torch.full(
                (terrain.terrain_levels.shape[0],),
                self.warmup_scale,
                device=terrain.terrain_levels.device,
            )

        terrain = self._get_terrain()
        current = terrain.terrain_levels.float()
        highest = self._highest_level.clamp(min=1.0)  # Avoid division by zero

        scale = current / highest
        scale = torch.clamp(scale, min=self.ratchet_min, max=1.0)

        return scale

    def get_stats(self) -> dict:
        """Get mastery stats for logging."""
        if self._highest_level is None:
            return {"promote_only": self._promote_only_active}

        terrain = self._get_terrain()
        if terrain is None:
            return {"promote_only": self._promote_only_active}

        current = terrain.terrain_levels.float()
        return {
            "promote_only": self._promote_only_active,
            "mean_terrain": current.mean().item(),
            "mean_highest": self._highest_level.mean().item(),
            "mean_reward_scale": self.get_reward_scale().mean().item(),
            "min_terrain": current.min().item(),
            "max_terrain": current.max().item(),
        }
