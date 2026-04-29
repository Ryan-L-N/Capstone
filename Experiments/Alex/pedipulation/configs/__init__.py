"""Pedipulation task — gym environment registration."""

import gymnasium as gym

gym.register(
    id="Pedipulation-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "configs.pedi_env_cfg:PedipulationSpotEnvCfg",
    },
)

gym.register(
    id="Pedipulation-Spot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "configs.pedi_env_cfg:PedipulationSpotEnvCfg_PLAY",
    },
)
