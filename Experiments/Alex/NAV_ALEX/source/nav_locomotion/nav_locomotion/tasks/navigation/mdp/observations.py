"""Custom observation functions for navigation — depth camera image retrieval.

Isaac Lab's standard mdp module doesn't include a RayCasterCamera image observation
function, so we define one here. The depth image is read from the RayCasterCamera
sensor, normalized to [0, 1] by dividing by max_distance, and flattened.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def depth_image_obs(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
    max_distance: float = 30.0,
) -> torch.Tensor:
    """Get normalized, flattened depth image from RayCasterCamera.

    Reads the 'distance_to_image_plane' data type from the RayCasterCamera sensor,
    normalizes to [0, 1] (0 = contact, 1 = max_distance / no obstacle), and flattens.

    Args:
        env: Isaac Lab environment.
        sensor_cfg: RayCasterCamera sensor config.
        max_distance: Maximum depth range in meters (for normalization). Default 30.0.

    Returns:
        Flattened depth image, shape (N, H*W). Values in [0, 1].
    """
    sensor = env.scene.sensors[sensor_cfg.name]

    # RayCasterCamera data: (N, H, W) or (N, H*W) depending on data type
    depth_data = sensor.data.output["distance_to_image_plane"]

    # Normalize to [0, 1]
    depth_normalized = depth_data / max_distance
    depth_normalized = torch.clamp(depth_normalized, 0.0, 1.0)

    # Handle NaN (missed rays -> max distance = clear)
    depth_normalized = torch.nan_to_num(depth_normalized, nan=1.0)

    # Flatten spatial dimensions: (N, H, W) or (N, H, W, C) -> (N, H*W)
    if depth_normalized.dim() >= 3:
        depth_normalized = depth_normalized.reshape(depth_normalized.shape[0], -1)

    return depth_normalized


def nav_prev_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return previous nav-level action (3-dim velocity commands).

    The underlying env has 12-dim joint actions, but the nav policy outputs
    3-dim velocity commands. This returns zeros until the NavEnvWrapper
    properly tracks nav-level actions.

    Returns:
        Tensor of shape (N, 3).
    """
    return torch.zeros(env.num_envs, 3, device=env.device)
