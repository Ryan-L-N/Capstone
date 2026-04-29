"""Loco Policy 1 — ARL Baseline. Gym environment registrations.

Imported by `scripts/train.py` to register the 5 baseline Spot envs that
use `SpotLocomotionEnvCfg`. The registration entry-points reference the
configs by importable path (`configs.env_cfg:SpotLocomotionEnvCfg`),
which works because `train.py` adds Loco_Policy_1's root to `sys.path`
before this module is imported.
"""

import gymnasium as gym

from . import agents  # noqa: F401  — also registers any agent-cfg side effects


gym.register(
    id="Locomotion-Flat-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "configs.env_cfg:SpotLocomotionEnvCfg",
    },
)

gym.register(
    id="Locomotion-Robust-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "configs.env_cfg:SpotLocomotionEnvCfg",
    },
)

gym.register(
    id="Locomotion-Play-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "configs.env_cfg:SpotLocomotionEnvCfg_PLAY",
    },
)

gym.register(
    id="Locomotion-Teacher-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "configs.env_cfg:SpotTeacherEnvCfg",
    },
)

gym.register(
    id="Locomotion-Distill-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "configs.env_cfg:SpotLocomotionEnvCfg",
    },
)
