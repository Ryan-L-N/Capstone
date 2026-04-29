"""SIM_TO_REAL environment and agent configurations.

Expert env configs (each subclasses SpotS2RBaseEnvCfg):
  - SpotFrictionExpertEnvCfg     (expert_friction_cfg.py)
  - SpotStairsUpExpertEnvCfg     (expert_stairs_up_cfg.py)
  - SpotStairsDownExpertEnvCfg   (expert_stairs_down_cfg.py)
  - SpotBouldersExpertEnvCfg     (expert_boulders_cfg.py)
  - SpotSlopesExpertEnvCfg       (expert_slopes_cfg.py)
  - SpotMixedRoughExpertEnvCfg   (expert_mixed_rough_cfg.py)

Distillation env:
  - SpotS2RDistillEnvCfg         (distillation_env_cfg.py)

Agent configs:
  - SpotS2RExpertPPORunnerCfg    (agent_cfg.py)
  - SpotS2RDistillPPORunnerCfg   (agent_cfg.py)
"""
