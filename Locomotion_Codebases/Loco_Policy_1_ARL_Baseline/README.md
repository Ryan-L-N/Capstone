# Loco Policy 1 — ARL Baseline

**Public name:** ARL Baseline
**Internal class:** `SpotLocomotionEnvCfg` (in `configs/env_cfg.py`)
**Production checkpoint:** `mason_baseline_final_19999.pt` (legacy filename, not renamed to keep external eval-script paths working — see "Checkpoint legacy" below)

## What this policy is

The first locomotion baseline trained for the Capstone. A single PPO Spot
policy trained on the 12-terrain `ROBUST_TERRAINS_CFG` curriculum with the
project's canonical 19-term reward stack and 235-dim observation
(48 proprioceptive + 187-point height scan).

This is the reference point against which Loco Policies 2 (ARL Hybrid),
3 (Student Teacher Training), 4 (Expert Master Distilled), and 5 (Final
Capstone Policy) are compared.

## Layout

```
Loco_Policy_1_ARL_Baseline/
├── configs/
│   ├── __init__.py             — gym.register Locomotion-{Flat,Robust,Play,Teacher,Distill}-Spot-v0
│   ├── env_cfg.py              — SpotLocomotionEnvCfg + SpotLocomotionEnvCfg_PLAY + SpotTeacherEnvCfg
│   └── agents/
│       └── rsl_rl_ppo_cfg.py   — SpotPPORunnerCfg (PPO hyperparameters)
├── scripts/
│   └── train.py                — entry point (cosine LR + value-loss watchdog + clamped DR)
├── checkpoints/
│   └── README.md               — where the production .pt files actually live
└── README.md                   — this file
```

## How to train

```bash
# H100 production (20480 envs, 60k iters)
./isaaclab.sh -p Loco_Policy_1_ARL_Baseline/scripts/train.py --headless \
    --robot spot --num_envs 20480

# Local debug smoke
isaaclab.bat -p Loco_Policy_1_ARL_Baseline/scripts/train.py --headless \
    --robot spot --num_envs 64 --max_iterations 10 --no_wandb
```

## Imports + path setup

`scripts/train.py` adds two paths to `sys.path` before importing:

1. `Loco_Policy_1_ARL_Baseline/` (this directory) — for `from configs...`
2. `Loco_Shared/` — for `from quadruped_locomotion...` shared utilities

If you run train.py from a different working directory, ensure both paths
are still resolvable (the script uses absolute path computation from
`__file__`, so the working dir does not matter).

## Checkpoint legacy

The production checkpoint name `mason_baseline_final_19999.pt` is preserved
as-is for backwards compatibility with external eval scripts. The "Mason"
prefix is the historical project codename; the public-facing name is
"ARL Baseline." Future re-trains can adopt the `arl_baseline_*.pt` naming.

## Differences from Loco Policy 2 (ARL Hybrid)

ARL Hybrid (Loco 2) is a separate baseline that uses Mason's proven
reward weights (`gait=10`, `vel=5`, `joint_pos=-0.7`) on a `[512, 256, 128]`
network with lighter DR. ARL Baseline (this policy) uses the project's
own 19-term reward stack on the wider `[1024, 512, 256]` network with
heavier DR. Both train on `ROBUST_TERRAINS_CFG`. See
`Loco_Policy_2_ARL_Hybrid/README.md` for the side-by-side comparison.
