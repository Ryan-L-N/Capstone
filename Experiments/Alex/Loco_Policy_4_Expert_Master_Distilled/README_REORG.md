# Loco Policy 4 — Expert Master Distilled

**Public name:** Expert Master Distilled
**Original directory:** `multi_expert_distillation/` (renamed)
**Inherits from:** `SpotARLHybridEnvCfg` (Loco_Policy_2)

## What this policy is

A focused 2-expert distillation pipeline. Trains a **single generalist
student** to imitate two specialist "master" experts:

- **Friction expert** — handles low-friction surfaces (friction/grass)
- **Obstacle expert** — handles obstacle/discrete surfaces (boulders/stairs)

A height-scan-roughness gate routes between the two experts at each
control step (sigmoid soft routing, with `routing_temperature` controlling
sharpness). The student learns to match this blended teacher action via
MSE + KL distillation loss, combined with PPO reward maximization.

This sits **next to** Loco_Policy_3 (Student Teacher Training)'s
6-master + DAGGER pipeline. Where Loco_3 distills 6 experts into a
generalist with full DAGGER, Loco_4 is the focused 2-master variant —
simpler, faster, and matches Mason's published distillation pattern.

## Layout

```
Loco_Policy_4_Expert_Master_Distilled/
├── README.md                    — original multi_expert_distillation README (renamed in place)
├── README_REORG.md              — this file (post-reorg pointer)
├── config.py                    — distillation hyperparameters
├── distill.py                   — entry point
├── distillation_loss.py         — MSE + KL loss combiner
└── expert_router.py             — soft routing on height-scan roughness
```

## How to train

```bash
# H100 production
python Loco_Policy_4_Expert_Master_Distilled/distill.py --headless \
    --friction_expert /path/to/friction_master.pt \
    --obstacle_expert /path/to/obstacle_master.pt \
    --num_envs 4096 --max_iterations 5000 --no_wandb

# Local debug
python Loco_Policy_4_Expert_Master_Distilled/distill.py --headless \
    --friction_expert checkpoints/mason_hybrid_best_33200.pt \
    --obstacle_expert checkpoints/obstacle_best.pt \
    --num_envs 64 --max_iterations 10 --no_wandb
```

## Imports + path setup

`distill.py` adds three paths to `sys.path` at module load:

1. `Loco_Policy_4_Expert_Master_Distilled/` — for `from expert_router`,
   `from distillation_loss`
2. `Loco_Policy_2_ARL_Hybrid/configs/` and `.../configs/agents/` — for
   `from arl_hybrid_env_cfg import SpotARLHybridEnvCfg` and
   `from rsl_rl_arl_hybrid_cfg import SpotARLHybridPPORunnerCfg`
3. `Loco_Shared/` — for `import quadruped_locomotion` (the shared
   utility package)

Gym registrations are pulled in indirectly via the
`Locomotion-ARLHybrid-Spot-v0` ID, which `Loco_Policy_2/configs/__init__.py`
registers. distill.py imports that module by importing the env_cfg.

## Where the experts come from

The two-master inputs to this distillation are produced by Loco_Policy_3
(Student Teacher Training). Specifically:

- `friction_expert` ← Loco_3's `expert_friction_final.pt` or `expert_grass_final.pt`
- `obstacle_expert` ← Loco_3's `boulder_v6_final.pt`, `boulder_v7_final.pt`,
  or one of the stair experts (`stair_v18_final.pt`, etc.)

The output `distilled_*.pt` is the unified master student. This is the
checkpoint used in mid-stage handoffs to Cole (nav integration) and Ryan
(ONNX delivery for hardware deployment).
