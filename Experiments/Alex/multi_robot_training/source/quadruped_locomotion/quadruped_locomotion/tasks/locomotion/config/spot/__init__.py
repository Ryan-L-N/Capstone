"""Spot locomotion task — gym environment registration."""

import gymnasium as gym

from . import agents  # noqa: F401

##
# Register Gym environments.
##

gym.register(
    id="Locomotion-Flat-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SpotLocomotionEnvCfg",
    },
)

gym.register(
    id="Locomotion-Robust-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SpotLocomotionEnvCfg",
    },
)

gym.register(
    id="Locomotion-Play-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SpotLocomotionEnvCfg_PLAY",
    },
)

gym.register(
    id="Locomotion-Teacher-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SpotTeacherEnvCfg",
    },
)

gym.register(
    id="Locomotion-Distill-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SpotLocomotionEnvCfg",
    },
)
