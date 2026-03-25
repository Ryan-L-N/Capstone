"""Navigation reward terms for Phase C exploration training.

All rewards follow the Bug #29 clamping convention:
    1. Compute raw value
    2. torch.clamp() to bounded range
    3. torch.nan_to_num() to zero out NaN/Inf
    4. torch.where(torch.isfinite(), result, zeros) as final guard

9 reward terms total:
    1. forward_velocity (+10.0)   — world-frame +X speed
    2. survival (+1.0)            — per-step alive bonus
    3. terrain_traversal (+2.0)   — cumulative X-distance normalized
    4. terrain_relative_height (-2.0) — penalize deviation from standing height
    5. drag_penalty (-1.5)        — penalize low height + forward motion (anti-crawl)
    6. cmd_smoothness (-1.0)      — penalize jerky velocity commands
    7. lateral_velocity (-0.3)    — light penalty for excessive strafing
    8. angular_velocity (-0.5)    — penalize excessive spinning
    9. vegetation_drag (-0.001)   — physics drag forces on feet + small penalty
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def _safe_clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Clamp tensor to [lo, hi] with NaN/Inf safety (Bug #29 convention).

    Args:
        x: Input tensor.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Clamped tensor with NaN/Inf replaced by zeros.
    """
    result = torch.clamp(x, lo, hi)
    result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.where(torch.isfinite(result), result, torch.zeros_like(result))


# ---------------------------------------------------------------------------
# 1. Forward Velocity Reward (+10.0)
# ---------------------------------------------------------------------------

def forward_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward proportional to world-frame forward (+X) velocity.

    Encourages the robot to move forward while capping extreme speeds.
    Negative velocity (moving backward) is mildly punished (clamp at -1.0).

    Args:
        env: Isaac Lab environment.
        asset_cfg: Robot asset config.

    Returns:
        Reward tensor, shape (N,), range [-1.0, 3.0].
    """
    asset = env.scene[asset_cfg.name]
    # World-frame root linear velocity, X component
    vx = asset.data.root_lin_vel_w[:, 0]
    return _safe_clamp(vx, -1.0, 3.0)


# ---------------------------------------------------------------------------
# 2. Survival Reward (+1.0)
# ---------------------------------------------------------------------------

def survival_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant per-step reward for staying alive.

    Simple alive bonus — the robot gets +1.0 every step it hasn't been
    terminated. Combined with the forward velocity reward, this creates
    a trade-off: move fast for reward but don't die.

    Returns:
        Tensor of ones, shape (N,).
    """
    return torch.ones(env.num_envs, device=env.device)


# ---------------------------------------------------------------------------
# 3. Terrain Traversal Reward (+2.0)
# ---------------------------------------------------------------------------

def terrain_traversal_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_distance: float = 50.0,
) -> torch.Tensor:
    """Reward based on cumulative forward (X) distance traveled since episode start.

    Normalized by max_distance so the reward is in [0, 1]. Encourages sustained
    forward progress rather than just instantaneous speed.

    Args:
        env: Isaac Lab environment.
        asset_cfg: Robot asset config.
        max_distance: Distance for full reward normalization. Default 50m.

    Returns:
        Reward tensor, shape (N,), range [0.0, 1.0].
    """
    asset = env.scene[asset_cfg.name]
    # Current X position relative to episode start (env origin)
    x_pos = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    progress = x_pos / max_distance
    return _safe_clamp(progress, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 4. Terrain-Relative Height Penalty (-2.0)
# ---------------------------------------------------------------------------

def terrain_relative_height_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_easy: float = 0.42,
    height_hard: float = 0.35,
    variance_flat: float = 0.001,
    variance_rough: float = 0.02,
    terrain_scaled: bool = True,
) -> torch.Tensor:
    """Penalize deviation from target standing height relative to local terrain.

    Uses the height scanner's center ray to get ground-relative body height.
    When terrain_scaled=True, adapts the target height based on height scan
    variance (flat ground -> stand tall at height_easy, rough ground -> moderate
    at height_hard, interpolated in between).

    Adapted from Phase B rewards.py. Addresses Bug #22 (world-frame Z breaks
    on elevated terrain) and Bug #28c (unbounded squared error).

    Args:
        env: Isaac Lab environment.
        sensor_cfg: Height scanner sensor config.
        asset_cfg: Robot asset config.
        height_easy: Target height on flat terrain (m). Default 0.42.
        height_hard: Target height on rough terrain (m). Default 0.35.
        variance_flat: Height scan variance threshold for flat. Default 0.001.
        variance_rough: Height scan variance threshold for rough. Default 0.02.
        terrain_scaled: Whether to adapt target by terrain roughness. Default True.

    Returns:
        Penalty tensor, shape (N,), range [0.0, 1.0] (squared clamped error).
    """
    height_scanner = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    # Height scan data: (N, num_rays) — distances from scanner to ground
    scan_data = height_scanner.data.ray_hits_w[..., 2]  # Z-coordinate of hits
    scanner_z = height_scanner.data.pos_w[:, 2]  # Scanner origin Z

    # Center ray: index for the center of the scan grid
    num_rays = scan_data.shape[1]
    center_idx = num_rays // 2
    ground_z = scan_data[:, center_idx]

    # Handle missed rays (NaN)
    ground_z = torch.nan_to_num(ground_z, nan=0.0, posinf=0.0, neginf=0.0)

    # Body height relative to local ground
    body_z = asset.data.root_pos_w[:, 2]
    rel_height = body_z - ground_z

    if terrain_scaled:
        # Compute height scan variance as roughness proxy
        scan_relative = scanner_z.unsqueeze(-1) - scan_data
        scan_relative = torch.nan_to_num(scan_relative, nan=0.0, posinf=0.0, neginf=0.0)
        variance = torch.var(scan_relative, dim=-1)
        variance = torch.clamp(variance, 0.0, 1.0)

        # Interpolate target height: flat -> height_easy, rough -> height_hard
        t = (variance - variance_flat) / (variance_rough - variance_flat + 1e-8)
        t = torch.clamp(t, 0.0, 1.0)
        target_height = height_easy * (1.0 - t) + height_hard * t
    else:
        target_height = height_easy

    # Squared error with Bug #28c clamping
    height_error = torch.abs(rel_height - target_height)
    height_error = torch.clamp(height_error, 0.0, 1.0)
    penalty = height_error ** 2

    return _safe_clamp(penalty, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 5. Drag Penalty (-1.5) — Anti-crawl
# ---------------------------------------------------------------------------

def drag_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_threshold: float = 0.25,
) -> torch.Tensor:
    """Penalize low body height combined with forward velocity (anti-belly-crawl).

    If the robot is low AND moving forward, it's likely dragging its belly.
    If it's low but stopped, the penalty is weaker (it fell, not crawling).

    Bug #27 showed that disabling height tracking + reducing penalties lets the
    robot discover belly-crawling as an exploit. This term specifically catches it.

    penalty = clamp(max(0, threshold - body_height) * forward_vel, 0, 3)

    Args:
        env: Isaac Lab environment.
        sensor_cfg: Height scanner sensor config.
        asset_cfg: Robot asset config.
        height_threshold: Height below which drag penalty activates (m). Default 0.25.

    Returns:
        Penalty tensor, shape (N,), range [0.0, 3.0].
    """
    height_scanner = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    # Ground-relative height via center ray
    scan_data = height_scanner.data.ray_hits_w[..., 2]
    num_rays = scan_data.shape[1]
    center_idx = num_rays // 2
    ground_z = torch.nan_to_num(scan_data[:, center_idx], nan=0.0)

    body_z = asset.data.root_pos_w[:, 2]
    rel_height = body_z - ground_z

    # How far below threshold (0 if above threshold)
    height_deficit = torch.clamp(height_threshold - rel_height, min=0.0)

    # Forward velocity (positive = moving forward)
    vx = asset.data.root_lin_vel_w[:, 0]
    forward_speed = torch.clamp(vx, min=0.0)

    # Drag = low * fast = crawling
    raw_penalty = height_deficit * forward_speed
    return _safe_clamp(raw_penalty, 0.0, 3.0)


# ---------------------------------------------------------------------------
# 6. Command Smoothness Penalty (-1.0)
# ---------------------------------------------------------------------------

def cmd_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize jerky velocity command changes between steps.

    Uses the difference between current and previous actions (velocity commands).
    Encourages smooth, gradual steering rather than rapid oscillation.

    Args:
        env: Isaac Lab environment.

    Returns:
        Penalty tensor, shape (N,), range [0.0, 5.0].
    """
    # action_manager tracks current and previous actions
    current = env.action_manager.action
    previous = env.action_manager.prev_action

    diff = current - previous
    raw = torch.sum(diff ** 2, dim=-1)
    return _safe_clamp(raw, 0.0, 5.0)


# ---------------------------------------------------------------------------
# 7. Lateral Velocity Penalty (-0.3)
# ---------------------------------------------------------------------------

def lateral_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Light penalty for excessive sideways velocity.

    Kept intentionally light — lateral movement is needed for obstacle avoidance.
    Only penalizes extreme strafing that wastes energy without purpose.

    Args:
        env: Isaac Lab environment.
        asset_cfg: Robot asset config.

    Returns:
        Penalty tensor, shape (N,), range [0.0, 2.0].
    """
    asset = env.scene[asset_cfg.name]
    vy = asset.data.root_lin_vel_w[:, 1]
    raw = vy ** 2
    return _safe_clamp(raw, 0.0, 2.0)


# ---------------------------------------------------------------------------
# 8. Angular Velocity Penalty (-0.5)
# ---------------------------------------------------------------------------

def angular_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize excessive yaw spinning.

    Prevents the robot from spinning in circles to avoid obstacles rather
    than navigating around them. Moderate weight — some turning is necessary.

    Args:
        env: Isaac Lab environment.
        asset_cfg: Robot asset config.

    Returns:
        Penalty tensor, shape (N,), range [0.0, 3.0].
    """
    asset = env.scene[asset_cfg.name]
    wz = asset.data.root_ang_vel_w[:, 2]
    raw = wz ** 2
    return _safe_clamp(raw, 0.0, 3.0)


# ---------------------------------------------------------------------------
# 9. Vegetation Drag Reward (-0.001) — Physics modifier + penalty
# ---------------------------------------------------------------------------

class VegetationDragReward(ManagerTermBase):
    """Applies velocity-dependent drag forces to feet — simulates grass/fluid/mud.

    This is both a PHYSICS MODIFIER and a REWARD TERM. It:
    1. Applies F_drag = -drag_coeff * v_foot to each foot every control step
    2. Returns a small penalty proportional to the drag force magnitude

    Terrain-aware behavior (requires curriculum=True in terrain config):
    - Robots on "friction_plane" columns: drag = 0 always (pure friction training)
    - Robots on "vegetation_plane" columns: drag > 0 always (pure drag training)
    - Robots on all other terrain types: randomized drag (generalization)

    Ported from locomotion/mdp/rewards.py (Phase B AI Coach training).
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
            if hasattr(terrain, "terrain_types"):
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
