# Locomotion Codebases

The Capstone team's five locomotion-policy projects, organized as one
deliverable. Each `Loco_Policy_N_*/` directory is a standalone training
pipeline; `Loco_Shared/` holds the common Python package; `4_env_test/`
is the eval framework used by all five; `docs/`, `archive/` are
project-wide docs and historical experiments.

## The five Loco Policies

| # | Public name | Source label | What it is |
|---|---|---|---|
| 1 | **ARL Baseline** | (was: Mason Baseline) | Single-PPO Spot trained on 12-terrain `ROBUST_TERRAINS_CFG` with the project's 19-term reward stack. The reference baseline that everything else is compared against. |
| 2 | **ARL Hybrid** | (was: Mason Hybrid) | Mason's proven 11-reward weights on `[512, 256, 128]` net with lighter DR + AI-Coach hooks + 3 surgical reward additions. Parent class for Loco 3, 4, 5 — changes here cascade. |
| 3 | **Student Teacher Training** | (was: SIM_TO_REAL) | 6-master expert pipeline + DAGGER distillation. Each expert is a sim-to-real-hardened specialist (action delay, sensor dropout, motor torque limits); the student is a single proprio-only generalist that imitates the 6-master ensemble. |
| 4 | **Expert Master Distilled** | (was: multi_expert_distillation) | Focused 2-expert distillation: one friction master + one obstacle master into a single generalist student via height-scan-roughness routing + MSE/KL loss. |
| 5 | **Final Capstone Policy** | (was: PARKOUR_NAV / "Project Hail Mary") | Asymmetric-critic teacher + proprio-only student distill with privileged terrain observation. The unified parkour-nav policy + Final World stair handoff. |

## Inheritance chain

The five policies are NOT independent — they share a class hierarchy:

```
Loco_Policy_2 (ARL Hybrid) — SpotARLHybridEnvCfg
    └── Loco_Policy_3 (S2R) — SpotS2RBaseEnvCfg
            └── 26 expert subclasses (friction/grass/boulder/stair/...)
                    └── Loco_Policy_5 (Final Capstone) — FinalCapstonePolicyEnvCfg
```

Loco_Policy_1 (ARL Baseline) is the standalone baseline; it doesn't
inherit from ARL Hybrid, but they're sibling configs that share
`Loco_Shared/quadruped_locomotion/` utilities.

Loco_Policy_4 (Expert Master Distilled) imports `SpotARLHybridEnvCfg`
directly to build its student env, parallel to Loco_3's SIM_TO_REAL but
with a focused 2-expert routing pattern.

## Layout

```
Locomotion_Codebases/
├── README.md                                    — this file
├── Loco_Shared/                                 — shared Python package + READMEs
│   ├── quadruped_locomotion/                   — utils, mdp/rewards, robots, ai_trainer
│   ├── pyproject.toml
│   └── setup.py
├── Loco_Policy_1_ARL_Baseline/
├── Loco_Policy_2_ARL_Hybrid/
├── Loco_Policy_3_Student_Teacher_Training/
├── Loco_Policy_4_Expert_Master_Distilled/
├── Loco_Policy_5_Final_Capstone_Policy/
├── 4_env_test/                                  — eval framework (friction/grass/boulder/stairs)
├── docs/                                        — project-wide training docs
│   ├── ARL_Hybrid_Tech_Manual.md
│   ├── ARL_Hybrid_Training.md
│   ├── HOW_TO_TRAIN_YOUR_RAWDOG.md             — bug compendium with §25 watchdog, §29 clamped penalties
│   ├── TRAINING_CURRICULUM.md
│   ├── Train_Spot_with_AI.md
│   └── AI_Training_Tech_Manual.md
└── archive/                                     — historical / non-canonical experiments
    ├── 100hr_env_run/
    ├── 48h_training/
    ├── H-100_Stress_Testing/
    ├── multi_robot_training/                   — residue (legacy eval, AI Coach train_ai.py)
    └── README.md
```

## Outside this wrapper (under `Experiments/Alex/`)

The following Capstone work is NOT locomotion-specific and stays under
`Experiments/Alex/` rather than moving here:

- `NAV_ALEX/` — navigation codebase (skill_nav_lite, depth-CNN, online obstacle tracker)
- `pedipulation/` — pedipulation experiments
- `Robust_RL/` — separate robustness initiative
- `5_ring_test/` — alternate eval arena (FiveRing)
- `final_world_stairs_test/` — FW USD stair eval framework (used by Loco_5)
- `AI_Trainer/` — git submodule (AI Coach development)
- `boulder_nav/`, `hybrid_ST_RL/`, `gravel_slope/`, `vision60_training/` — exploratory
- `ARL_DELIVERY/` — top-level ARL handoff package (sibling of this wrapper)
- `BD_SPOT_GIT/` — Boston Dynamics Spot SDK clone
- `TrainingResults/`, `FORTHERAWDOGS.md`, `auto_gait_switching_plan.md`,
  `dream_dojo_motion_prior_plan.md`, presentation .pptx — top-level docs/results

Where this matters: `Loco_Policy_5/scripts/eval.py` references
`Experiments/Alex/NAV_ALEX/scripts/cole_arena_skillnav_lite.py` for the
Cole-arena Skill-Nav-Lite eval. The path is computed by walking up to the
repo root, then down into `Experiments/Alex/NAV_ALEX/`.

## How each Loco_Policy resolves its imports

Every entry-point script (`Loco_Policy_N/scripts/*.py`,
`Loco_Policy_4/distill.py`) does the same `sys.path` setup at module
load:

```python
_LOCO_N_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_LOCO_CODEBASES_ROOT = os.path.abspath(os.path.join(_LOCO_N_ROOT, ".."))
for _p in (
    _LOCO_N_ROOT,                                                      # this policy's own configs
    os.path.join(_LOCO_CODEBASES_ROOT, "Loco_Policy_2_ARL_Hybrid", "configs"),  # parent ARLHybrid (when Loco 3/4/5)
    os.path.join(_LOCO_CODEBASES_ROOT, "Loco_Shared"),                 # shared quadruped_locomotion
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
```

The two non-trivial parts:
1. `Loco_Shared/` is on `sys.path`, so `from quadruped_locomotion.utils...`
   resolves to `Loco_Shared/quadruped_locomotion/utils/...`.
2. `Loco_Policy_2_ARL_Hybrid/configs/` is on `sys.path` (for Loco 3, 4, 5),
   so `from arl_hybrid_env_cfg import SpotARLHybridEnvCfg` resolves there
   without package-name collisions with each policy's own `configs/`.

## Where production checkpoints live

Every `Loco_Policy_N/checkpoints/` directory is empty in the source tree;
binary `.pt` and `.onnx` files are gitignored. Each one has a
`README.md` pointing at where the production artifacts actually live
(H100 paths, handoff dirs in `Experiments/Cole/Final_Capstone_Policy_handoff/`,
`Experiments/Ryan/Final_Capstone_Policy_22100/`, etc.).

The team-facing Final Policy ONNX deliverables for hardware deployment
live at the repo top level under `Final Policies/Locomotion Policies/`.

## Defense layers (Bug #25 + Bug #29)

All five Loco_Policy training pipelines inherit (or copy) the same
defense stack from `docs/HOW_TO_TRAIN_YOUR_RAWDOG.md`:

1. **Value-loss watchdog** — if `vf_loss > 100`, halve LR for 50 iters.
   Compound multiplicative penalty on consecutive spikes.
2. **Clamped L2 penalty wrappers** —
   `clamped_action_smoothness_penalty`, `clamped_joint_acceleration_penalty`,
   `clamped_joint_torques_penalty`, `clamped_joint_velocity_penalty` —
   bound the penalty outputs so a value-function instability doesn't get
   amplified into a NaN cascade.
3. **`nan_to_num` pre+post hooks** on actor/critic/normalizers.
4. **`_sanitize_std`** on `policy.act()` so the noise std parameter
   never escapes [min_noise_std, max_noise_std].
5. **Cosine LR annealing** with explicit warmup, replacing RSL-RL's
   adaptive scheduler.
6. **Critic warmup** for actor-only-resume: freeze actor + std for N
   iters with a zero-init critic head, so the critic calibrates to the
   new value scale before PPO updates the resumed actor.

These layers are why each `train.py` has the `[GUARD] / [SCHED] /
[WARMUP]` startup banners — they confirm the defense stack is armed
before training begins.
