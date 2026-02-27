"""Custom reward functions for multi-robot robust locomotion training.

5 robot-agnostic reward terms that supplement the standard IsaacLab Spot MDP
reward functions. These use parameterized body names via SceneEntityCfg so
they work identically for Spot (.*_foot) and Vision60 (lower.*).

Terms:
  - VegetationDragReward: Velocity-dependent drag forces (grass/fluid/mud)
  - velocity_modulation_reward: Terrain-adaptive speed tracking
  - body_height_tracking_penalty: Prevent unnatural crouching/rising
  - contact_force_smoothness_penalty: Gentler foot placement
  - stumble_penalty: Penalize tripping on obstacles

Source: 100hr_env_run/rewards/reward_terms.py (identical — already robot-agnostic)
Created for AI2C Tech Capstone — MS for Autonomy, February 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


class VegetationDragReward(ManagerTermBase):
    """Applies velocity-dependent drag forces to feet -- simulates grass/fluid/mud.

    This is both a PHYSICS MODIFIER and a REWARD TERM. It:
    1. Applies F_drag = -drag_coeff * v_foot to each foot every control step
    2. Returns a small penalty proportional to the drag force magnitude

    Terrain-aware behavior (requires curriculum=True in terrain config):
    - Robots on "friction_plane" columns: drag = 0 always (pure friction training)
    - Robots on "vegetation_plane" columns: drag > 0 always (pure drag training)
    - Robots on all other terrain types: randomized drag (generalization)
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.drag_max: float = cfg.params.get("drag_max", 8.0)
        self.contact_threshold: float = cfg.params.get("contact_threshold", 1.0)

        # Resolve foot body IDs
        self.foot_body_ids = cfg.params["asset_cfg"].body_ids

        # Store env reference for terrain lookups at reset time
        self._env_ref = env

        # --- Build terrain-aware column masks ---
        self.terrain_aware = False
        terrain: TerrainImporter = env.scene.terrain
        terrain_gen_cfg = getattr(terrain.cfg, "terrain_generator", None)

        if terrain_gen_cfg is not None and terrain_gen_cfg.curriculum:
            sub_terrain_names = list(terrain_gen_cfg.sub_terrains.keys())
            proportions = np.array(
                [sc.proportion for sc in terrain_gen_cfg.sub_terrains.values()]
            )
            proportions = proportions / proportions.sum()

            num_cols = terrain_gen_cfg.num_cols
            cum_props = np.cumsum(proportions)
            col_to_idx = []
            for col in range(num_cols):
                idx = int(np.min(np.where(cum_props > col / num_cols)[0]))
                col_to_idx.append(idx)

            veg_name = cfg.params.get("vegetation_terrain_name", "vegetation_plane")
            fric_name = cfg.params.get("friction_terrain_name", "friction_plane")

            veg_idx = sub_terrain_names.index(veg_name) if veg_name in sub_terrain_names else -1
            fric_idx = sub_terrain_names.index(fric_name) if fric_name in sub_terrain_names else -1

            self.is_vegetation_col = torch.tensor(
                [col_to_idx[c] == veg_idx for c in range(num_cols)],
                dtype=torch.bool, device=env.device,
            )
            self.is_friction_col = torch.tensor(
                [col_to_idx[c] == fric_idx for c in range(num_cols)],
                dtype=torch.bool, device=env.device,
            )
            self.terrain_aware = True

            n_veg = int(self.is_vegetation_col.sum())
            n_fric = int(self.is_friction_col.sum())
            print(
                f"[VegetationDragReward] Terrain-aware: "
                f"{n_veg}/{num_cols} vegetation cols, "
                f"{n_fric}/{num_cols} friction cols",
                flush=True,
            )

        # Per-environment drag coefficient: sampled at reset, shape (num_envs, 1)
        self.drag_coeff = torch.zeros(env.num_envs, 1, device=env.device)
        self._resample_drag(torch.arange(env.num_envs, device=env.device))

    def reset(self, env_ids: torch.Tensor):
        """Resample drag coefficients for reset environments."""
        self._resample_drag(env_ids)

    def _resample_drag(self, env_ids: torch.Tensor):
        """Sample new drag coefficients with terrain-aware overrides."""
        n = len(env_ids)
        dev = self.drag_coeff.device

        tier_rand = torch.rand(n, device=dev)
        tier_1 = tier_rand < 0.25
        tier_2 = (tier_rand >= 0.25) & (tier_rand < 0.50)
        tier_3 = (tier_rand >= 0.50) & (tier_rand < 0.75)
        tier_4 = tier_rand >= 0.75

        drag_vals = torch.zeros(n, 1, device=dev)
        if tier_2.any():
            drag_vals[tier_2] = torch.empty(tier_2.sum(), 1, device=dev).uniform_(0.5, 5.0)
        if tier_3.any():
            drag_vals[tier_3] = torch.empty(tier_3.sum(), 1, device=dev).uniform_(5.0, 12.0)
        if tier_4.any():
            drag_vals[tier_4] = torch.empty(tier_4.sum(), 1, device=dev).uniform_(12.0, self.drag_max)

        if self.terrain_aware:
            terrain: TerrainImporter = self._env_ref.scene.terrain
            robot_cols = terrain.terrain_types[env_ids]

            on_friction = self.is_friction_col[robot_cols]
            drag_vals[on_friction] = 0.0

            on_vegetation = self.is_vegetation_col[robot_cols]
            n_veg = int(on_vegetation.sum())
            if n_veg > 0:
                drag_vals[on_vegetation] = torch.empty(n_veg, 1, device=dev).uniform_(0.5, self.drag_max)

        self.drag_coeff[env_ids] = drag_vals

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        drag_max: float = 8.0,
        contact_threshold: float = 1.0,
        vegetation_terrain_name: str = "vegetation_plane",
        friction_terrain_name: str = "friction_plane",
    ) -> torch.Tensor:
        """Compute and apply drag forces, return penalty."""
        num_envs = env.num_envs
        num_feet = len(self.foot_body_ids) if isinstance(self.foot_body_ids, list) else 4

        foot_vel = self.asset.data.body_lin_vel_w[:, self.foot_body_ids, :]

        net_forces = self.contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
        force_mags = torch.norm(net_forces, dim=-1)
        is_contact = force_mags > self.contact_threshold

        drag_force = -self.drag_coeff.unsqueeze(2) * foot_vel
        drag_force = drag_force * is_contact.unsqueeze(2).float()
        drag_force[:, :, 2] = 0.0

        all_forces = torch.zeros(num_envs, self.asset.num_bodies, 3, device=env.device)
        all_torques = torch.zeros_like(all_forces)
        all_forces[:, self.foot_body_ids, :] = drag_force

        self.asset.permanent_wrench_composer.set_forces_and_torques(
            forces=all_forces,
            torques=all_torques,
        )

        drag_magnitude = torch.norm(drag_force, dim=-1)
        total_drag = torch.sum(drag_magnitude, dim=1)
        return total_drag


def velocity_modulation_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    std: float = 0.5,
) -> torch.Tensor:
    """Reward the robot for moving at an appropriate speed given terrain difficulty."""
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    cmd_vel = env.command_manager.get_command("base_velocity")[:, :2]
    cmd_speed = torch.linalg.norm(cmd_vel, dim=1)

    actual_speed = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)

    net_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    force_magnitudes = torch.norm(net_forces, dim=-1)
    force_variance = torch.var(force_magnitudes, dim=1)

    difficulty_factor = torch.clamp(force_variance / 500.0, 0.0, 1.0)
    adaptive_target = cmd_speed * (1.0 - 0.5 * difficulty_factor)

    speed_error = torch.abs(actual_speed - adaptive_target)
    return torch.exp(-speed_error / std)


def body_height_tracking_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.42,
) -> torch.Tensor:
    """Penalize deviation from the target standing height.

    Args:
        target_height: Target body height above ground (meters).
                       0.42m for Spot, 0.55m for Vision60.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    body_height = asset.data.root_pos_w[:, 2]
    height_error = torch.square(body_height - target_height)
    return height_error


def contact_force_smoothness_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize sudden spikes in ground reaction forces."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    if contact_sensor.cfg.history_length < 2:
        return torch.zeros(env.num_envs, device=env.device)

    current_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    prev_forces = contact_sensor.data.net_forces_w_history[:, 1, sensor_cfg.body_ids]

    force_diff = torch.norm(current_forces - prev_forces, dim=-1)
    return torch.sum(force_diff, dim=1)


def stumble_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    knee_height: float = 0.15,
    force_threshold: float = 5.0,
) -> torch.Tensor:
    """Penalize when the robot's feet hit obstacles at shin/knee height.

    Args:
        knee_height: Height threshold (meters). 0.15m for Spot, 0.20m for Vision60.
        force_threshold: Minimum contact force to consider (N).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    net_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    force_mags = torch.norm(net_forces, dim=-1)

    is_stumble = (foot_heights > knee_height) & (force_mags > force_threshold)
    stumble_forces = is_stumble.float() * force_mags
    return torch.sum(stumble_forces, dim=1)
