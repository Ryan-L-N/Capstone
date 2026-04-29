"""Loco Policy 2 — ARL Hybrid. Gym environment registrations.

Imported by `scripts/train_hybrid.py` to register the 2 ARL Hybrid Spot
envs that use `SpotARLHybridEnvCfg`. The registration entry-points
reference the configs by importable path
(`configs.arl_hybrid_env_cfg:SpotARLHybridEnvCfg`), which works because
`train_hybrid.py` adds Loco_Policy_2's root to `sys.path` before this
module is imported.
"""

import gymnasium as gym

from . import agents  # noqa: F401  — also registers any agent-cfg side effects


gym.register(
    id="Locomotion-ARLHybrid-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "configs.arl_hybrid_env_cfg:SpotARLHybridEnvCfg",
        "rsl_rl_cfg_entry_point": "configs.agents.rsl_rl_arl_hybrid_cfg:SpotARLHybridPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-ARLHybrid-Spot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "configs.arl_hybrid_env_cfg:SpotARLHybridEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "configs.agents.rsl_rl_arl_hybrid_cfg:SpotARLHybridPPORunnerCfg",
    },
)
