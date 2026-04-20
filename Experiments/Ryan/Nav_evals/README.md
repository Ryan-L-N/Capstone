# Nav Evals — 100-seed battery for Ryan

Gabriel already ran each of these policies once at each density on his machine (seed=42). This folder runs the same configs across **100 seeds** so we get statistical confidence intervals before the April 26 deliverable.

## What this runs

For each combination below, `run_nav_evals.py` launches `cole_arena_skillnav_lite.py` headless for one episode:

| Policies (4) | Densities (2) | Seeds | Total runs |
|---|---|---|---|
| `stock_flat`, `boulder_v6`, `mason_baseline`, `mason_hybrid` | `full` (6.1% coverage), `30pct` | 100 each | **800** |

Each run takes roughly **0.5–6 min** headless (flat runs settle fast, mason hybrid can go 5+ min). Budget ~24–36 wall-hours on a single machine. You can parallelize by partitioning seeds (see below) if you have multiple Isaac-Sim-capable GPUs/machines, but one episode per process is the simplest and safest.

Gabriel's single-seed (42) results you're replicating:

**Full density (6.1% coverage)**
| Policy | Result | WPs | Dist | Time |
|---|---|---|---|---|
| stock_flat | FELL | 5/25 | 363.6m | 321.6s |
| boulder_v6 | FELL | 0/25 | 15.4m | 32.8s |
| mason_baseline | FELL | 0/25 | 26.1m | 37.3s |
| mason_hybrid | FELL | 0/25 | 33.1m | 41.5s |

**30% density**
| Policy | Result | WPs | Dist | Time |
|---|---|---|---|---|
| boulder_v6 | FELL | 0/25 | 25.9m | 40.8s |
| stock_flat | SCORE_DEPLETED | 1/25 | 215.7m | 315.0s |
| mason_baseline | SCORE_DEPLETED | 2/25 | 261.2m | 330.0s |
| mason_hybrid | SCORE_DEPLETED | 3/25 | 294.6m | 345.0s |

Key finding we want to stress-test: **at 30% density, `mason_hybrid` (3/25) beats `stock_flat` (1/25)**, which reverses the full-density ranking. With n=1 this might be noise. 100 seeds will tell us whether rough-terrain training really helps at moderate density.

---

## Instructions for Ryan's Claude

You are running this on Ryan's machine, not Gabriel's. The paths in `run_nav_evals.py` will not work as-is. Before starting:

### 1. Verify environment

Ryan's machine needs the same Isaac Lab / Isaac Sim install that Alex's NAV_ALEX scripts expect:
- Isaac Sim 5.1.0 + Isaac Lab 2.3.0
- Python 3.11 conda env (Alex's is `isaaclab311`)
- PyTorch with CUDA

From the Capstone repo root, confirm Alex's script is present:

```bash
ls Experiments/Alex/NAV_ALEX/scripts/cole_arena_skillnav_lite.py
```

### 2. Locate the Python interpreter and checkpoints

Run these to find Ryan's equivalents (paths WILL differ from Gabriel's):

```bash
# Python interpreter — whichever conda env has Isaac Lab
which python                      # Linux/Mac
where python                      # Windows (pick the isaaclab one)

# Checkpoints — should live under Alex/ inside the Capstone repo
ls Experiments/Alex/NAV_ALEX/checkpoints/boulder_v6_expert_4500.pt
ls Experiments/Alex/multi_robot_training/checkpoints/mason_baseline_final_19999.pt
ls Experiments/Alex/multi_robot_training/checkpoints/mason_hybrid_best_33200.pt
```

If any checkpoint is missing, pull-LFS or re-sync the Alex submodule — they live in the shared repo. `stock_flat` needs no checkpoint (it uses the built-in Isaac Lab flat policy).

### 3. Edit `CONFIG` in `run_nav_evals.py`

Open `run_nav_evals.py` and update the four paths in the `CONFIG` dict:
- `PYTHON` — absolute path to the Python interpreter from step 2
- `NAV_ALEX_DIR` — absolute path to `Experiments/Alex/NAV_ALEX`
- `CKPT_BOULDER_V6`, `CKPT_MASON_BASELINE`, `CKPT_MASON_HYBRID` — absolute paths from step 2

Keep `PER_RUN_TIMEOUT_SEC` at 900s unless runs on Ryan's machine consistently exceed that.

### 4. Smoke-test one seed first

Before committing to 800 runs, verify the wiring with a single-seed single-policy pass:

```bash
cd Experiments/Ryan/Nav_evals
python run_nav_evals.py --seeds 1 --seed_start 42 --policies stock_flat --densities 30pct
```

This should take ~6 min and produce `results/run_<ts>/results.csv` with one row. If it prints `SCORE_DEPLETED wp=1/25 dist=215m` that matches Gabriel's seed-42 number exactly. If it prints `PARSE_FAIL` or times out, fix before proceeding.

### 5. Launch the full battery

```bash
cd Experiments/Ryan/Nav_evals
python run_nav_evals.py --seeds 100
```

This runs all 800 combos sequentially. The runner is resumable — if it crashes or Ryan stops it partway, re-running with the same `--out_dir` will skip any (policy, density, seed) combo whose log already contains a `Result:` line.

To **split across two machines**, run with disjoint seed windows:
```bash
# machine A
python run_nav_evals.py --seeds 50 --seed_start 0 --out_dir results/split_a
# machine B
python run_nav_evals.py --seeds 50 --seed_start 50 --out_dir results/split_b
```
Then concatenate `results.csv` files afterwards.

### 6. Aggregate

```bash
python aggregate_results.py results/run_<ts>/results.csv
```

Prints a table with fell/complete/score_depleted counts, mean/median/max WPs, mean distance and time per (policy, density).

### 7. Hand back

Commit `results/run_<ts>/results.csv` and the aggregate printout to this folder, then ping Gabriel. Do **not** commit the per-seed `logs/` directory — it's many hundreds of megabytes. It's fine to zip and drop on Drive if the full logs are ever needed.

---

## Common failure modes

- **Isaac Sim headless fails to open display** — make sure `OMNI_KIT_ACCEPT_EULA=YES` is exported; Alex's scripts do this themselves but the env sometimes needs it set shell-wide.
- **CUDA OOM** — only one Isaac Sim instance per GPU. Don't try to parallelize on a single machine.
- **`Loop exit: ... status=<blank>`** / PARSE_FAIL — check the log's tail for a Python traceback. Usually a checkpoint path typo.
- **Mason runs hang >15 min** — bump `PER_RUN_TIMEOUT_SEC` in `run_nav_evals.py` and re-run. Expected worst-case is ~6 min, timeout exists to catch true hangs.
- **Different Cole obstacle layout** than Gabriel got — the layout is seeded by the `--seed` arg, so same seed = same obstacles, as long as Alex's scripts haven't changed since this doc was written. If seed=42 doesn't reproduce Gabriel's numbers within ~5%, ask Gabriel before continuing.

## Files

- `run_nav_evals.py` — batch runner with resume + per-run timeout
- `aggregate_results.py` — CSV → summary table
- `results/` — per-run output dirs (gitignored except results.csv)
- `logs/` (inside each run dir) — raw stdout per run, for debugging
