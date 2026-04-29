"""Gym environment registration for Spot navigation tasks."""

import gymnasium as gym

gym.register(
    id="Navigation-Explore-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nav_env_cfg:SpotNavExploreCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:SpotNavPPORunnerCfg",
    },
)

gym.register(
    id="Navigation-Explore-Spot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nav_env_cfg:SpotNavExploreCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:SpotNavPPORunnerCfg",
    },
)
