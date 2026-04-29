# Loco Policy 2 — ARL Hybrid

**Public name:** ARL Hybrid
**Internal class:** `SpotARLHybridEnvCfg` (in `configs/arl_hybrid_env_cfg.py`)
**Production checkpoint:** `mason_hybrid_best_33200.pt` (legacy filename, see "Checkpoint legacy" in checkpoints/README.md)

## What this policy is

The second locomotion baseline trained for the Capstone. Adopts Mason's
proven 11-term reward weights (gait=10, vel=5, joint_pos=-0.7,
orientation=-3.0) on a `[512, 256, 128]` actor MLP with lighter DR
(mass ±2.5 kg, friction 0.3-1.0). The "Hybrid" name reflects the design
goal: combine Mason's reward structure with our 12-terrain
`ROBUST_TERRAINS_CFG` and 3 surgical additions (terrain_relative_height,
dof_pos_limits, clamped action_smoothness).

This policy is the parent class for the Student Teacher Training base
config (`Loco_Policy_3/configs/base_s2r_env_cfg.py` inherits from
`SpotARLHybridEnvCfg`), so changes here propagate to Loco 3 and Loco 5
(Final Capstone Policy, which inherits from S2R).

## Layout

```
Loco_Policy_2_ARL_Hybrid/
├── configs/
│   ├── __init__.py                   — gym.register Locomotion-ARLHybrid-Spot-v0 + Play-v0
│   ├── arl_hybrid_env_cfg.py         — SpotARLHybridEnvCfg + SpotARLHybridEnvCfg_PLAY
│   │                                   + inner Hybrid{Observations,Actions,Commands,Event,Rewards,Terminations,Curriculum,Scene}Cfg
│   └── agents/
│       └── rsl_rl_arl_hybrid_cfg.py  — SpotARLHybridPPORunnerCfg
├── scripts/
│   └── train_hybrid.py               — entry point (cosine LR + AI Coach hooks)
├── checkpoints/
│   └── README.md                     — where the production .pt files live
└── README.md                         — this file
```

## How to train

```bash
# H100 production with AI Coach
./isaaclab.sh -p Loco_Policy_2_ARL_Hybrid/scripts/train_hybrid.py --headless \
    --num_envs 20480 --enable_ai_coach

# Local debug smoke
isaaclab.bat -p Loco_Policy_2_ARL_Hybrid/scripts/train_hybrid.py --headless \
    --num_envs 64 --max_iterations 10 --no_wandb
```

## Imports + path setup

`scripts/train_hybrid.py` adds two paths to `sys.path` before importing:

1. `Loco_Policy_2_ARL_Hybrid/` (this directory) — for `from configs.arl_hybrid_env_cfg`
2. `Loco_Shared/` — for `from quadruped_locomotion...` shared utilities

## Inner classes — kept "Hybrid" prefix

`HybridObservationsCfg`, `HybridActionsCfg`, `HybridCommandsCfg`,
`HybridEventCfg`, `HybridRewardsCfg`, `HybridTerminationsCfg`,
`HybridCurriculumCfg`, `HybridSceneCfg` keep their original short names —
they live inside the `SpotARLHybridEnvCfg` namespace, so the "Hybrid"
prefix reads naturally and doesn't collide with anything else. Renaming
all of them to `ARLHybrid*` would have meant ~30 cascading edits across
the codebase for no readability gain.

## Differences from Loco Policy 1 (ARL Baseline)

| Axis | ARL Baseline (Loco 1) | ARL Hybrid (Loco 2) |
|---|---|---|
| Reward weights | 19-term project stack | Mason's 11 + 3 additions = 14 terms |
| Network | `[1024, 512, 256]` | `[512, 256, 128]` |
| DR mass range | ±5 kg | ±2.5 kg |
| Friction range | 0.3–1.5 | 0.3–1.0 |
| Body contact | soft penalty | hard termination |
| AI Coach | not integrated | activates after silent period |
| Velocity threshold | 0.4 | 0.5 |

## Inheritance chain (downstream consumers)

```
SpotARLHybridEnvCfg (this class)
    └── SpotS2RBaseEnvCfg                      (Loco_Policy_3/configs/base_s2r_env_cfg.py)
            └── SpotS2RExpertFrictionEnvCfg    (Loco_Policy_3/configs/expert_friction_env_cfg.py)
            └── SpotS2RExpertStairsEnvCfg      (Loco_Policy_3/configs/expert_stairs_*.py)
            └── ... 24 more expert configs
                    └── FinalCapstonePolicyEnvCfg  (Loco_Policy_5/configs/final_capstone_policy_env_cfg.py)
```

Changes to `SpotARLHybridEnvCfg` cascade through Loco 3 (Student Teacher
Training) and Loco 5 (Final Capstone Policy). Be careful editing this
file — verify with the per-policy smoke tests after any change.
