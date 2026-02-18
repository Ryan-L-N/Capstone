"""Isaac Lab environment configuration for capstone evaluation.

Inherits observation groups, height scanner, and physics settings from
rough_env_cfg.py (ARL_DELIVERY/05_Training_Package/isaac_lab_spot_configs/).

Key settings:
  - Physics: 500 Hz (dt=0.002s), decimation=10 (50 Hz control)
  - Solver: 4 position iterations, 0 velocity iterations
  - Backend: GPU PhysX (torch tensors on cuda:0)
  - Robot: SPOT_CFG from Isaac Lab
  - Height scanner: RayCaster 17x11 grid, 0.1m resolution

Reuses patterns from:
- ARL_DELIVERY/05_Training_Package/isaac_lab_spot_configs/rough_env_cfg.py
- ARL_DELIVERY/05_Training_Package/isaac_lab_spot_configs/agents/rsl_rl_ppo_cfg.py
"""

# TODO: Implementation
# - EvalEnvCfg dataclass (physics params, observation groups, episode settings)
# - EPISODE_TIMEOUT = 120.0  (seconds)
# - MAX_CONTROL_STEPS = 6000  (120s / 0.02s)
# - FALL_THRESHOLD = 0.15  (meters — base height below this = fall)
