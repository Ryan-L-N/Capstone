"""Custom reward functions for 100hr multi-terrain robust locomotion training.

These supplement the existing Spot reward terms in:
  isaaclab_tasks/.../config/spot/mdp/rewards.py

New terms address weaknesses found in the 48hr rough policy evaluation:
  - VegetationDragReward: Applies velocity-dependent drag to feet (grass/fluid sim)
  - velocity_modulation_reward: Accept slower speeds on hard terrain
  - body_height_tracking_penalty: Prevent unnatural crouching/rising
  - contact_force_smoothness_penalty: Gentler foot placement
  - stumble_penalty: Penalize tripping on obstacles

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

    If the terrain config doesn't use curriculum mode, falls back to global
    randomization across all environments.

    Implementation:
    - Uses `permanent_wrench_composer.set_forces_and_torques()` to apply
      persistent forces that act every physics sub-step (500Hz)
    - Forces are recomputed every control step (50Hz) based on current foot velocities
    - Drag only applies when feet are in contact with the ground
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
            # Replicate the column-to-terrain-type mapping from TerrainGenerator
            cum_props = np.cumsum(proportions)
            col_to_idx = []
            for col in range(num_cols):
                idx = int(np.min(np.where(cum_props > col / num_cols)[0]))
                col_to_idx.append(idx)

            # Find sub-terrain indices for friction and vegetation planes
            veg_name = cfg.params.get("vegetation_terrain_name", "vegetation_plane")
            fric_name = cfg.params.get("friction_terrain_name", "friction_plane")

            veg_idx = sub_terrain_names.index(veg_name) if veg_name in sub_terrain_names else -1
            fric_idx = sub_terrain_names.index(fric_name) if fric_name in sub_terrain_names else -1

            # Boolean mask per column: is this column vegetation / friction?
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
        # Initialize with random values
        self._resample_drag(torch.arange(env.num_envs, device=env.device))

    def reset(self, env_ids: torch.Tensor):
        """Resample drag coefficients for reset environments."""
        self._resample_drag(env_ids)

    def _resample_drag(self, env_ids: torch.Tensor):
        """Sample new drag coefficients with terrain-aware overrides.

        Base distribution (for non-plane terrains):
          - 25% of envs: c=0 (clean ground)
          - 25% of envs: c in [0.5, 5.0] (light fluid through medium lawn)
          - 25% of envs: c in [5.0, 12.0] (medium lawn through thick grass)
          - 25% of envs: c in [12.0, drag_max] (thick grass through dense brush)

        Terrain overrides (when terrain_aware=True):
          - friction_plane columns: c forced to 0.0 (no drag ever)
          - vegetation_plane columns: c forced to [0.5, drag_max] (always drag)
        """
        n = len(env_ids)
        dev = self.drag_coeff.device

        # --- Base tiered sampling for all envs ---
        tier_rand = torch.rand(n, device=dev)
        tier_1 = tier_rand < 0.25                          # clean (c=0)
        tier_2 = (tier_rand >= 0.25) & (tier_rand < 0.50)  # light [0.5, 5.0]
        tier_3 = (tier_rand >= 0.50) & (tier_rand < 0.75)  # medium [5.0, 12.0]
        tier_4 = tier_rand >= 0.75                          # heavy [12.0, drag_max]

        drag_vals = torch.zeros(n, 1, device=dev)
        if tier_2.any():
            drag_vals[tier_2] = torch.empty(tier_2.sum(), 1, device=dev).uniform_(0.5, 5.0)
        if tier_3.any():
            drag_vals[tier_3] = torch.empty(tier_3.sum(), 1, device=dev).uniform_(5.0, 12.0)
        if tier_4.any():
            drag_vals[tier_4] = torch.empty(tier_4.sum(), 1, device=dev).uniform_(12.0, self.drag_max)

        # --- Terrain-aware overrides ---
        if self.terrain_aware:
            terrain: TerrainImporter = self._env_ref.scene.terrain
            robot_cols = terrain.terrain_types[env_ids]

            # Friction plane: force drag = 0 (pure low-friction challenge)
            on_friction = self.is_friction_col[robot_cols]
            drag_vals[on_friction] = 0.0

            # Vegetation plane: force drag > 0 (pure drag challenge)
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
        """Compute and apply drag forces, return penalty.

        Called every control step (50Hz). The forces persist through
        all physics sub-steps until the next call.
        """
        num_envs = env.num_envs
        num_feet = len(self.foot_body_ids) if isinstance(self.foot_body_ids, list) else 4

        # --- 1. Get foot velocities in world frame ---
        # shape: (num_envs, num_feet, 3)
        foot_vel = self.asset.data.body_lin_vel_w[:, self.foot_body_ids, :]

        # --- 2. Check which feet are in contact with ground ---
        net_forces = self.contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
        force_mags = torch.norm(net_forces, dim=-1)  # (num_envs, num_feet)
        is_contact = force_mags > self.contact_threshold  # (num_envs, num_feet)

        # --- 3. Compute drag force: F = -c_drag * v_foot (only when in contact) ---
        # drag_coeff shape: (num_envs, 1) → broadcast to (num_envs, num_feet, 3)
        drag_force = -self.drag_coeff.unsqueeze(2) * foot_vel  # (num_envs, num_feet, 3)

        # Zero out drag for feet not in contact (airborne feet get no drag)
        drag_force = drag_force * is_contact.unsqueeze(2).float()

        # Zero out Z component — drag is horizontal only (XY plane)
        drag_force[:, :, 2] = 0.0

        # --- 4. Apply drag forces to foot bodies ---
        # Build full-body force tensor (zeros for non-foot bodies)
        all_forces = torch.zeros(num_envs, self.asset.num_bodies, 3, device=env.device)
        all_torques = torch.zeros_like(all_forces)
        all_forces[:, self.foot_body_ids, :] = drag_force

        self.asset.permanent_wrench_composer.set_forces_and_torques(
            forces=all_forces,
            torques=all_torques,
        )

        # --- 5. Return penalty proportional to drag force magnitude ---
        # This encourages the policy to minimize time spent in drag zones
        # and to use efficient gaits that reduce foot ground contact time
        drag_magnitude = torch.norm(drag_force, dim=-1)  # (num_envs, num_feet)
        total_drag = torch.sum(drag_magnitude, dim=1)  # (num_envs,)

        return total_drag


def velocity_modulation_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    std: float = 0.5,
) -> torch.Tensor:
    """Reward the robot for moving at an appropriate speed given terrain difficulty.

    On easy terrain (high contact stability), the robot should track the full
    commanded velocity. On hard terrain (frequent contacts, high forces), the
    robot gets credit for moving at a reduced but non-zero speed.

    This prevents the policy from learning to freeze on hard terrain (zero velocity)
    or charging recklessly (full velocity regardless of terrain).

    The "difficulty" is estimated from the variance of foot contact forces —
    higher variance means more uneven/challenging terrain.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Commanded velocity magnitude
    cmd_vel = env.command_manager.get_command("base_velocity")[:, :2]
    cmd_speed = torch.linalg.norm(cmd_vel, dim=1)

    # Actual velocity magnitude
    actual_speed = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)

    # Estimate terrain difficulty from contact force variance
    net_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    force_magnitudes = torch.norm(net_forces, dim=-1)  # (num_envs, num_feet)
    force_variance = torch.var(force_magnitudes, dim=1)  # (num_envs,)

    # Adaptive target: full speed on easy terrain, 50% on very hard terrain
    # difficulty_factor: 0 (easy) to 1 (very hard)
    difficulty_factor = torch.clamp(force_variance / 500.0, 0.0, 1.0)
    adaptive_target = cmd_speed * (1.0 - 0.5 * difficulty_factor)

    # Reward tracking the adaptive target
    speed_error = torch.abs(actual_speed - adaptive_target)
    return torch.exp(-speed_error / std)


def body_height_tracking_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.42,
) -> torch.Tensor:
    """Penalize deviation from the target standing height.

    The 48hr policy sometimes crouches unnaturally on hard terrain or rises
    too high when trying to clear obstacles. This penalty keeps the base
    at a consistent height relative to the terrain.

    Args:
        target_height: Target body height above ground (meters).
                       0.42m is Spot's nominal standing height.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    body_height = asset.data.root_pos_w[:, 2]

    # Simple L2 penalty on height deviation
    height_error = torch.square(body_height - target_height)
    return height_error


def contact_force_smoothness_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize sudden spikes in ground reaction forces.

    Encourages the policy to place feet gently rather than slamming them down.
    This improves stability on low-friction surfaces (where hard impacts cause
    slipping) and reduces wear on real hardware.

    Computed as the L2 norm of the temporal difference of foot contact forces.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Contact force history: (num_envs, history_len, num_bodies, 3)
    # We use the first two history frames to compute temporal difference
    if contact_sensor.cfg.history_length < 2:
        return torch.zeros(env.num_envs, device=env.device)

    current_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    prev_forces = contact_sensor.data.net_forces_w_history[:, 1, sensor_cfg.body_ids]

    # Temporal difference of force magnitudes
    force_diff = torch.norm(current_forces - prev_forces, dim=-1)  # (num_envs, num_feet)
    return torch.sum(force_diff, dim=1)


def stumble_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    knee_height: float = 0.15,
    force_threshold: float = 5.0,
) -> torch.Tensor:
    """Penalize when the robot's feet hit obstacles at shin/knee height.

    This indicates the robot is tripping over obstacles rather than stepping
    over them. Encourages higher foot clearance and better obstacle awareness.

    The penalty fires when:
    1. A foot has contact forces above the threshold AND
    2. The foot is above knee_height (i.e., hitting the side of an obstacle,
       not stepping on top of it)

    Args:
        knee_height: Height threshold (meters) above which contact indicates
                     a stumble rather than a step.
        force_threshold: Minimum contact force to consider (N).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Foot heights
    foot_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # (num_envs, num_feet)

    # Contact forces
    net_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    force_mags = torch.norm(net_forces, dim=-1)  # (num_envs, num_feet)

    # Stumble condition: foot is elevated AND has significant contact force
    is_stumble = (foot_heights > knee_height) & (force_mags > force_threshold)

    # Penalty = sum of stumble force magnitudes (weighted by how bad each stumble is)
    stumble_forces = is_stumble.float() * force_mags
    return torch.sum(stumble_forces, dim=1)
