"""Vision60 locomotion task — gym environment registration."""

import gymnasium as gym

from . import agents  # noqa: F401

##
# Register Gym environments.
##

gym.register(
    id="Locomotion-Robust-Vision60-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Vision60LocomotionEnvCfg",
    },
)

gym.register(
    id="Locomotion-Play-Vision60-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Vision60LocomotionEnvCfg",
    },
)

gym.register(
    id="Locomotion-Teacher-Vision60-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Vision60TeacherEnvCfg",
    },
)

gym.register(
    id="Locomotion-Distill-Vision60-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Vision60LocomotionEnvCfg",
    },
)
