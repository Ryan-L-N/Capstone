"""Privileged observation functions for the asymmetric critic.

These query the physx view or sensor buffers directly at obs time, so they
are always up-to-date after randomization events fire. All outputs are
per-env tensors shaped [num_envs, N].

Teacher critic sees these in addition to clean proprio + clean height_scan;
the student (policy group) never sees them.

Functions:
  - friction_coefficient : [num_envs, 2]   (static, dynamic)
  - added_mass_base      : [num_envs, 1]   (trunk mass delta from default)
  - foot_contact_forces  : [num_envs, 12]  (4 feet x 3 axes, world frame)

Pattern cribbed from isaaclab.envs.mdp.observations; uses the same
`SceneEntityCfg` plumbing so terms slot into an ObservationGroupCfg.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_DEFAULT_MASS_CACHE: dict[int, torch.Tensor] = {}


def _ensure_default_mass(env: "ManagerBasedEnv", asset: Articulation) -> torch.Tensor:
    """Cache the default per-env trunk mass on first call.

    `root_physx_view.get_masses()` is cheap but we want the DELTA from default,
    which is the actual randomized quantity the critic should see.
    """
    key = id(asset)
    if key not in _DEFAULT_MASS_CACHE:
        masses = asset.root_physx_view.get_masses().to(env.device)
        _DEFAULT_MASS_CACHE[key] = masses.clone()
    return _DEFAULT_MASS_CACHE[key]


def friction_coefficient(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Per-env ground friction coefficient (static, dynamic), averaged over shapes.

    Returns shape [num_envs, 2].

    `randomize_rigid_body_material` writes to `root_physx_view` material slots;
    we read them back here. If the robot has multiple collision shapes we
    average over them — the critic doesn't need per-shape granularity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # shape: [num_envs, num_shapes, 3] where [:, :, 0]=static, [:, :, 1]=dynamic
    mat = asset.root_physx_view.get_material_properties().to(env.device)
    return mat[..., :2].mean(dim=1)


def added_mass_base(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["body"]),
) -> torch.Tensor:
    """Per-env trunk mass delta vs default. Shape [num_envs, 1].

    `add_base_mass` event adds in [0, 3] kg (parkour DR). The critic gets the
    current value so it can calibrate value-function targets for pushed-around
    heavier runs without the actor ever seeing it.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if body_ids == slice(None) or body_ids is None:
        body_ids = [0]

    default = _ensure_default_mass(env, asset)
    current = asset.root_physx_view.get_masses().to(env.device)
    delta = (current - default)[:, body_ids].sum(dim=1, keepdim=True)
    return delta


def foot_contact_forces(
    env: "ManagerBasedEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
) -> torch.Tensor:
    """Foot contact forces in world frame, flattened. Shape [num_envs, 4*3=12].

    Read from the ContactSensor that Mason base env registers on .*_foot bodies.
    Order matches SceneEntityCfg body_ids resolution (fl, fr, hl, hr for Spot).
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # net_forces_w has shape [num_envs, num_bodies_in_cfg, 3]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return forces.reshape(forces.shape[0], -1)
