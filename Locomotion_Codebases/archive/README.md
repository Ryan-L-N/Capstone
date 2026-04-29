# archive/ — Historical Experiments

This directory holds experiments that were exploratory or superseded by
the canonical 5-Loco-Policy ship matrix. They're preserved for git
history + reproducibility but are **not** part of the capstone
deliverable.

## Contents

| Directory | Notes |
|---|---|
| `100hr_env_run/` | Early 100-hour Spot multi-terrain robust training (20,480 envs / 60k iters). Predates the ARL Baseline + ARL Hybrid split. Superseded by Loco_Policy_1 (ARL Baseline). |
| `48h_training/` | First 48-hour Spot + ANYmal-C runs from the project bring-up phase. Predates the unified `quadruped_locomotion` shared package. Superseded by Loco_Policy_1 / Loco_Policy_2. |
| `H-100_Stress_Testing/` | One-off compute-stress experiments (max-env tests, thermal probes, GPU memory load profiling). Not a training pipeline. Kept for cluster operational reference. |
| `multi_robot_training/` | Original parent directory holding the shared `quadruped_locomotion` package + reward terms + scripts. Most contents migrated to `Loco_Shared/` (utils + MDP) and the per-policy `Loco_Policy_N/` directories (env_cfgs + train scripts). What remains here is the residue: legacy `eval/` arenas, the AI Coach `train_ai.py` integration, `train_distill.py`, and `MASON_README.md`. Kept here rather than purged so the project's evolution is bisectable. |

## Why these are archived rather than deleted

1. **Git blame survives** — moving rather than deleting preserves authorship and history.
2. **Bisect-friendly** — if a regression appears in any of the 5 canonical
   policies and it traces back to a refactor here, the original code is
   still inspectable.
3. **Eval-script paths** — a few external scripts (notably under
   `Experiments/Cole/...` and `Experiments/Ryan/...`) still reference
   legacy checkpoint filenames like `mason_baseline_final_19999.pt` and
   `mason_hybrid_best_33200.pt`. Until those are re-trained under the
   ARL Baseline / ARL Hybrid public names, the original script residue
   is useful for debugging.

## What's still active under `Experiments/Alex/`

- `Loco_Shared/` — shared `quadruped_locomotion` Python package
- `Loco_Policy_1_ARL_Baseline/` — Mason Baseline → ARL Baseline
- `Loco_Policy_2_ARL_Hybrid/` — Mason Hybrid → ARL Hybrid
- `Loco_Policy_3_Student_Teacher_Training/` — was SIM_TO_REAL/
- `Loco_Policy_4_Expert_Master_Distilled/` — was multi_expert_distillation/
- `Loco_Policy_5_Final_Capstone_Policy/` — was PARKOUR_NAV/
- `4_env_test/` — eval framework (shared by all 5 policies)
- `docs/` — top-level docs (TRAINING_CURRICULUM, RAWDOG bug compendium, ARL_Hybrid_*)
- `NAV_ALEX/` — navigation (skill_nav_lite, depth-CNN, online obstacle tracker)
- `pedipulation/` — pedipulation experiments (active, not part of locomotion ship matrix)
- `Robust_RL/` — separate robustness initiative (active, sibling of locomotion)

## When to dig in here

- Reproducing a published result that pre-dates the reorg.
- Tracing a regression that may have come from the move (use
  `git log --follow` on the moved file).
- Pulling the AI Coach `train_ai.py` flow when wiring the same coach
  hooks into one of the Loco_Policy_N pipelines.
