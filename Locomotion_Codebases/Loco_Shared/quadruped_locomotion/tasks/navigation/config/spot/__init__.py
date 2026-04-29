"""Spot navigation task — gym environment registration."""

import gymnasium as gym

from . import agents  # noqa: F401

##
# Register Gym environments.
##

gym.register(
    id="Navigation-Sparse-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SpotNavEnvCfg",
    },
)

gym.register(
    id="Navigation-Dense-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SpotNavEnvCfg",
    },
)

gym.register(
    id="Navigation-Corridor-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SpotNavEnvCfg",
    },
)

gym.register(
    id="Navigation-Mixed-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SpotNavEnvCfg",
    },
)
