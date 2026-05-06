# How to Run the 4-Environment Evaluation

A complete, copy-paste-able runbook for getting the friction / grass /
boulder / stairs eval working on **Linux**, **Windows**, or **WSL**.

If you're hitting a wall, jump straight to the [Common Issues](#common-issues)
section first — it covers the four problems 90% of fresh installs hit.

---

## TL;DR — fastest possible smoke

The **direct python invocation** is the most reliable cross-platform path
— it bypasses Isaac Lab's `isaaclab.bat` / `isaaclab.sh` wrapper, which
has known cmd.exe path-quoting bugs on Windows when the repo lives under
a path with spaces (e.g. `OneDrive\Desktop\Capstone Project\`).

```bash
# Cross-platform (Git Bash, WSL, Linux) — VERIFIED WORKING on Windows + Linux
cd "Locomotion_Codebases/4_env_test"

# Set Isaac Sim EULA + use the conda env's Python directly:
OMNI_KIT_ACCEPT_EULA=YES /c/miniconda3/envs/isaaclab311/python.exe \
    src/run_capstone_eval.py --headless \
    --num_episodes 1 --policy flat --env friction \
    --output_dir results/smoke
```

**Linux / H100** — the same pattern with the env_isaaclab Python:

```bash
cd ~/Capstone/Locomotion_Codebases/4_env_test
OMNI_KIT_ACCEPT_EULA=YES ~/miniconda3/envs/env_isaaclab/bin/python \
    src/run_capstone_eval.py --headless \
    --num_episodes 1 --policy flat --env friction \
    --output_dir results/smoke
```

**Windows PowerShell** — same direct-python idea:

```powershell
$env:OMNI_KIT_ACCEPT_EULA = "YES"
cd "Locomotion_Codebases\4_env_test"
& "C:\miniconda3\envs\isaaclab311\python.exe" `
    src\run_capstone_eval.py --headless `
    --num_episodes 1 --policy flat --env friction `
    --output_dir results\smoke
```

If you see `Saved N episodes to results/smoke/friction_flat_episodes.jsonl`
and `Evaluation complete!`, you're set. Skip to
[Production Runs](#production-runs).

> **Why not `isaaclab.bat -p ...` on Windows?** Because `cmd.exe` (which
> the `.bat` wrapper invokes) chokes on the space in
> `OneDrive\Desktop\Capstone Project\` even with quoted args — the
> standard "X was unexpected at this time" error. The direct-python
> approach uses Git Bash / PowerShell argument parsing instead, which
> handles quoted paths correctly.

---

## Prerequisites

| Component | Required version | Where to find it |
|---|---|---|
| Isaac Sim | 5.1.0 | NVIDIA Omniverse Launcher OR `pip install isaacsim==5.1.0 --index-url https://pypi.nvidia.com` |
| Isaac Lab | 2.3.0 | `git clone https://github.com/isaac-sim/IsaacLab.git`, checkout v2.3.0 tag |
| Python | 3.11.x | Conda env named `isaaclab311` (local) or `env_isaaclab` (H100) |
| GPU | RTX 20-series or newer (compute capability ≥ 7.5) | — |
| Disk | ~30 GB for Isaac Sim assets | — |

### Conda environment

The eval requires the same conda env that Isaac Lab was installed into.
Different machines use different names — that's fine, the scripts try
both:

```bash
# H100 server (ai2ct2):
conda activate env_isaaclab

# Local Windows / Linux dev box:
conda activate isaaclab311
```

The shipped scripts (`scripts/debug_5iter.sh`, etc.) try `env_isaaclab`
first then `isaaclab311`, so you don't have to edit them per-machine.

### Where Isaac Lab lives

Find your `isaaclab.sh` / `isaaclab.bat` path:

| Platform | Typical path |
|---|---|
| H100 (Linux) | `~/IsaacLab/isaaclab.sh` |
| Local Linux | `~/IsaacLab/isaaclab.sh` |
| Local Windows | `C:\IsaacLab\isaaclab.bat` (forward-slash form: `/c/IsaacLab/isaaclab.bat`) |
| WSL2 | `~/IsaacLab/isaaclab.sh` (Linux side) |

If yours is somewhere else, set this once in your shell rc:

```bash
# ~/.bashrc or ~/.zshrc
export ISAACLAB="$HOME/IsaacLab"   # or wherever yours lives
alias isaaclab="$ISAACLAB/isaaclab.sh"
```

---

## Setup

### 1. Pull this repo's main branch

```bash
git clone https://github.com/Ryan-L-N/Capstone.git
cd Capstone
```

The eval lives at `Locomotion_Codebases/4_env_test/`.

### 2. Verify your conda env has the right packages

```bash
conda activate isaaclab311   # or env_isaaclab on H100
python -c "import isaacsim; print('Isaac Sim:', isaacsim.__file__)"
python -c "import isaaclab; print('Isaac Lab:', isaaclab.__file__)"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'device count:', torch.cuda.device_count())"
```

Expected:
- Isaac Sim path → `.../site-packages/isaacsim/__init__.py`
- Isaac Lab path → `.../IsaacLab/source/isaaclab/isaaclab/__init__.py`
- CUDA: `True`, device count `1`+

If any of those fail, fix conda before going further.

### 3. Place policy checkpoints

The eval ships with several checkpoints under `policies/` and
`checkpoints/`. Confirm the ones you need:

```bash
ls Locomotion_Codebases/4_env_test/policies/
# Expect:
#   distilled_6899.pt
#   hybrid_nocoach_19999.pt
#   mason_baseline_final_19999.pt   ← Loco_Policy_1 / ARL Baseline
#   mason_hybrid_best_33200.pt      ← Loco_Policy_2 / ARL Hybrid
#   obstacle_best_44400.pt
```

The `flat` policy uses Isaac Sim's built-in NVIDIA flat-baseline (not a
file in this repo) — no setup needed for that one.

The `rough` policy needs a 235-dim checkpoint. Use any of:
- `mason_baseline_final_19999.pt` (ARL Baseline, the project reference)
- `mason_hybrid_best_33200.pt` (ARL Hybrid, AI-Coach-trained)
- `policies/distilled_6899.pt` (Expert Master Distilled, 2-expert routing)
- `Final Policies/Locomotion Policies/Locomotion Policy 1.pt` through `Policy 6.pt` (top-level deliverable ckpts on `main`)

Pass the checkpoint path with `--checkpoint`:

```bash
isaaclab.sh -p src/run_capstone_eval.py --headless \
    --policy rough --env friction --num_episodes 5 \
    --checkpoint policies/mason_baseline_final_19999.pt \
    --output_dir results/baseline_friction
```

### 4. Fix line endings (Windows → Linux only)

If you cloned on Windows and are pushing to H100:

```bash
sed -i "s/\r$//" Locomotion_Codebases/4_env_test/scripts/*.sh
chmod +x Locomotion_Codebases/4_env_test/scripts/*.sh
```

---

## Running the Eval

### Quick debug (1-5 episodes, 1-3 minutes)

For first-time verification:

```bash
cd Locomotion_Codebases/4_env_test

# Linux / WSL
bash scripts/debug_5iter.sh                # default: friction + flat
bash scripts/debug_5iter.sh stairs rough   # specify env + policy

# Windows (Git Bash)
bash scripts/debug_5iter.sh                # works the same

# Windows (PowerShell, no bash) — invoke the Python directly
"C:\IsaacLab\isaaclab.bat" -p .\src\run_capstone_eval.py --headless `
    --num_episodes 5 --policy flat --env friction `
    --output_dir .\results\debug
```

Expected runtime: 1-3 minutes (mostly Isaac Sim startup).
Output: `results/debug/friction_flat_<timestamp>.log` + `<env>_<policy>_episodes.jsonl`

### Production run (1000 episodes per combo, ~1-1.5h on H100)

```bash
cd Locomotion_Codebases/4_env_test
bash scripts/run_full_eval.sh
```

Runs all 8 combinations: 2 policies × 4 environments × 1000 episodes.

Output: `results/{env}_{policy}_episodes.jsonl` for each combo.

### Rendered visualization (10 ep × 8 combos, ~3-4h)

```bash
bash scripts/run_rendered_viz.sh
```

Captures video + keyframe PNGs for inspection. Output: `results/rendered/`.

### Single-environment teleop walkthrough

```bash
bash scripts/run_teleop.sh friction   # or grass / boulder / stairs
```

Xbox controller (or keyboard fallback). RB switches gait (FLAT/ROUGH);
LB toggles FPV camera.

### Analyze results

```bash
python src/metrics/reporter.py --input results/
```

Generates `results/summary.csv` and per-environment plots in
`results/plots/`.

---

## Production Runs (per platform)

### H100 (preferred — has 95 GB GPU and OS conda config baked in)

```bash
ssh t2user@172.24.254.24
cd ~/Locomotion_Codebases/4_env_test                           # if cloned here
# OR:
cd ~/Capstone/Locomotion_Codebases/4_env_test                  # if cloned full repo

screen -S eval_full -dm bash -c '
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab &&
  cd ~/Capstone/Locomotion_Codebases/4_env_test &&
  export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 &&
  bash scripts/run_full_eval.sh > ~/eval_full.log 2>&1
'

# Monitor:
ssh t2user@172.24.254.24 "tail -f ~/eval_full.log"
```

### Local Linux / WSL

Same as H100, just with `isaaclab311` instead of `env_isaaclab` and
your local IsaacLab path. The shell scripts auto-detect.

### Local Windows (Git Bash)

```bash
cd "/c/Users/Gabriel Santiago/OneDrive/Desktop/Capstone Project/Capstone/Locomotion_Codebases/4_env_test"
bash scripts/debug_5iter.sh
```

The bash scripts work in Git Bash. The internal `./isaaclab.sh` call
breaks though — see [Common Issues](#common-issues) below.

### Local Windows (PowerShell, no bash)

Invoke the Python directly:

```powershell
$repo = "C:\Users\Gabriel Santiago\OneDrive\Desktop\Capstone Project\Capstone"
$isaaclab = "C:\IsaacLab\isaaclab.bat"
$env:OMNI_KIT_ACCEPT_EULA = "YES"

& $isaaclab -p "$repo\Locomotion_Codebases\4_env_test\src\run_capstone_eval.py" `
    --headless --num_episodes 5 --policy flat --env friction `
    --output_dir "$repo\Locomotion_Codebases\4_env_test\results\debug"
```

---

## Canonical per-env eval invocation

After the Apr 29 rendered-smoke battery (22100 ckpt, 2-iter rendered),
these are the verified-working `--target_vx` + `--zone_slowdown_cap`
combinations for each environment. All 4 envs hit the "+1m past
zone 3" depth target on 2/2 episodes.

| Env | `--target_vx` | `--zone_slowdown_cap` | Verified result (2-iter, 22100) |
|---|---|---|---|
| **friction** | **3.0** | **1.0** | 1/2 COMPLETE 49.5m / 85.6s (best 48.6s in 3-iter) — project speed record |
| **grass** | **3.0** | **3.0** | 2/2 COMPLETE 49.5m / 74.5-76.1s |
| **boulder** | **2.0** | **0.67** | 2/2 reach zone 4 at 31.3m (TIMEOUT) |
| **stairs** | **2.0** | **1.0** | 2/2 reach zone 4 at 31.8-33.0m (FELL) |

### One-liner per-env command (Windows / Git Bash)

```bash
CKPT="Experiments/Ryan/Final_Capstone_Policy_22100/parkour_phasefwplus_22100.pt"
PYTHON="/c/miniconda3/envs/isaaclab311/python.exe"
COMMON="--policy rough --rendered --num_episodes 5 --max_episode_time 180 --checkpoint $CKPT --mason --action_scale 0.3 --output_dir results/canonical_eval"

OMNI_KIT_ACCEPT_EULA=YES $PYTHON src/run_capstone_eval.py --env friction $COMMON --target_vx 3.0 --zone_slowdown_cap 1.0
OMNI_KIT_ACCEPT_EULA=YES $PYTHON src/run_capstone_eval.py --env grass    $COMMON --target_vx 3.0 --zone_slowdown_cap 3.0
OMNI_KIT_ACCEPT_EULA=YES $PYTHON src/run_capstone_eval.py --env boulder  $COMMON --target_vx 2.0 --zone_slowdown_cap 0.67
OMNI_KIT_ACCEPT_EULA=YES $PYTHON src/run_capstone_eval.py --env stairs   $COMMON --target_vx 2.0 --zone_slowdown_cap 1.0
```

### Or use the canonical battery script

```bash
bash scripts/run_canonical_eval.sh                      # default 5 eps, rendered, 22100 ckpt
bash scripts/run_canonical_eval.sh 100 --headless       # production: 100 eps headless
```

Why these specific values:
- **friction tgt=3.0, cap=1.0** — push speed in zones 1-3 (sandpaper, dry rubber, wet
  concrete); cap to 1.0 in zones 4-5 to avoid slip-induced flips on wet ice + oil.
- **grass tgt=3.0, cap=3.0** — grass drag does the natural slowdown organically. No
  artificial cap needed; let the policy push.
- **boulder tgt=2.0, cap=0.67** — 3.0 m/s in zones 1-2 caused zone-2 flips at high
  speed on rocks. Toned to 2.0 to enter cleanly. Phase-9-locked 0.67 m/s past x≥20m
  for the dense-boulder zones 3+.
- **stairs tgt=2.0, cap=1.0** — 3.0 + cap=2.0 had 0/3 pass rate (all flipped at
  ~28-29m). Toned to 2.0 + cap=1.0 → 2/2 reach 31.8-33.0m. Tall risers (17-23cm)
  don't tolerate >1.0 m/s commands in zones 3+.

22100's training cmd_vel range was (-1.0, 1.5). Targets above 1.5 are
extrapolation — the cap is the safety net.

## Common Issues

### Issue 1 — "isaaclab.sh: command not found" on Windows

**Cause:** the bash scripts in `scripts/` call `./isaaclab.sh -p ...`
but Windows uses `isaaclab.bat`.

**Fix (option A, recommended):** invoke the Python entry point directly:

```bash
"C:\IsaacLab\isaaclab.bat" -p src/run_capstone_eval.py --headless --num_episodes 1 --policy flat --env friction
```

**Fix (option B):** install WSL2 + Ubuntu and run from the Linux side.
The bash scripts work as-is.

### Issue 2 — "conda: command not found" inside the script

**Cause:** SSH non-interactive shells don't load `~/.bashrc`, so conda's
shell-init isn't sourced.

**Fix:** the scripts already handle this via `eval "$(conda shell.bash hook)"`
and a fallback to `/home/t2user/miniconda3/bin/conda`. If you're on a
different conda install path, edit `scripts/debug_5iter.sh:31-37` to
add your path to the fallback list.

### Issue 3 — "Failed to import isaacsim" or shape mismatch error

**Cause:** wrong conda env activated, or Isaac Sim installed at a
different path than the script expects.

**Fix:** verify with the 3-line check in [Setup → step 2](#2-verify-your-conda-env-has-the-right-packages). Both isaacsim and
isaaclab must import from the SAME conda env.

### Issue 4 — Vulkan / display errors on H100 (headless)

**Cause:** Isaac Sim tries to enumerate Vulkan devices for rendering.
On an H100 with no display, you'll see `ERROR_INCOMPATIBLE_DRIVER`
warnings on startup.

**Fix:** the warnings are *cosmetic* in `--headless` mode — Isaac Sim
falls back to compute-only correctly. Ignore them. If the simulation
itself fails (no `[INFO]: Time taken for simulation start`), then the
Vulkan stack is genuinely broken — file an issue with the H100 admin.

### Issue 5 — eval starts but every episode terminates instantly

**Cause:** typically (a) the policy and the environment have a
proprio/obs-dim mismatch, or (b) the checkpoint was trained with a
different `action_scale` or DOF order.

**Fix:** verify with
```
python -c "
import torch
ckpt = torch.load('policies/mason_baseline_final_19999.pt', map_location='cpu', weights_only=False)
sd = ckpt['model_state_dict']
print('Actor input dim:', list(sd['actor.0.weight'].shape))
"
```
The first dim should be 235 (rough) or 48 (flat). If it's 485, that's
an *asymmetric critic* dim (Loco_Policy_5 / Final Capstone Policy)
which the eval doesn't handle directly — use the
`Locomotion_Codebases/Loco_Policy_5_Final_Capstone_Policy/scripts/eval.py`
launcher instead.

### Issue 6 — "ARL_DELIVERY/..." path not found

**Cause:** stale README references. The reorg moved files but some
docstrings still point to old paths.

**Fix:** ARL_DELIVERY/ stayed at `Experiments/Alex/ARL_DELIVERY/` (it's
not under Locomotion_Codebases/). Walk up two levels from
`Locomotion_Codebases/4_env_test/`:

```
../../Experiments/Alex/ARL_DELIVERY/05_Training_Package/...
```

This path is for *reference* only — the eval doesn't import from
ARL_DELIVERY at runtime, so the stale references are documentation rot,
not a hard dependency.

### Issue 7 — "Address already in use" or hanging at startup

**Cause:** a previous Isaac Sim process is still holding GPU memory or
a TCP port.

**Fix (Linux):**
```bash
pkill -9 -f run_capstone_eval
nvidia-smi --query-gpu=memory.used --format=csv
# If memory > 0 and no Python procs visible, you have zombies
# (defunct Python procs holding GPU contexts). H100 admin can
# `sudo reboot` or use the BMC ForceRestart skill if that's set up.
```

**Fix (Windows):**
```powershell
Get-Process | Where-Object { $_.ProcessName -like "*python*" -or $_.ProcessName -like "*kit*" } | Stop-Process -Force
```

### Issue 8 — H100 only: SSH disconnects mid-run

**Fix:** always wrap long runs in `screen` or `tmux`. The shipped
scripts assume this for production runs:

```bash
screen -S eval -dm bash -c 'cd ~/Capstone/Locomotion_Codebases/4_env_test && bash scripts/run_full_eval.sh > ~/eval.log 2>&1'
screen -ls                                 # verify it's running
ssh t2user@172.24.254.24 "tail -f ~/eval.log"   # tail from another terminal
```

---

## Outputs to expect

After a successful run, you'll have:

```
results/
├── debug/                                       (debug runs)
│   ├── friction_flat_<timestamp>.log           Isaac Sim log
│   └── friction_flat_episodes.jsonl            One JSON line per episode
├── friction_flat_episodes.jsonl                (production runs)
├── friction_rough_episodes.jsonl
├── grass_flat_episodes.jsonl
├── grass_rough_episodes.jsonl
├── boulder_flat_episodes.jsonl
├── boulder_rough_episodes.jsonl
├── stairs_flat_episodes.jsonl
├── stairs_rough_episodes.jsonl
├── summary.csv                                 (after metrics/reporter.py)
└── plots/                                      (after metrics/reporter.py)
    ├── friction_*.png
    ├── grass_*.png
    ├── boulder_*.png
    └── stairs_*.png
```

Each `.jsonl` line has the schema in `episode_schema.json`. Example
fields: `episode_id`, `policy`, `env`, `final_x`, `final_zone`, `falls`,
`time`, `mean_speed`, `total_path_length`, `terminated_via`.

---

## Quick reference — what each policy is

| `--policy` value | Internal name | Loco Policy # | Checkpoint default |
|---|---|---|---|
| `flat` | NVIDIA Spot Flat-Baseline | n/a | (built-in to Isaac Sim) |
| `rough` (default to `mason_baseline_final_19999.pt`) | ARL Baseline | 1 | `policies/mason_baseline_final_19999.pt` |
| `rough --checkpoint policies/mason_hybrid_best_33200.pt` | ARL Hybrid | 2 | — |
| `rough --checkpoint policies/distilled_6899.pt` | Expert Master Distilled | 4 | — |

For Loco_Policy_5 (Final Capstone Policy) evals, use the dedicated
launcher:

```bash
python Locomotion_Codebases/Loco_Policy_5_Final_Capstone_Policy/scripts/eval.py \
    --target 4_env --headless --num_episodes 3 --seeds 42,123,7 \
    --checkpoint <path-to-final-capstone-policy.pt>
```

It internally calls `4_env_test/src/run_capstone_eval.py` with the
right `--mason --action_scale 0.3` flags for that asymmetric policy.

---

## Where to file issues

If you hit something not covered here, check first:

1. `LESSONS_LEARNED.md` — accumulated debug journal
2. `capstone_test.md` — full master test plan
3. `Locomotion_Codebases/docs/HOW_TO_TRAIN_YOUR_RAWDOG.md` — training-side bug compendium

Then ping Alex or Gabriel with:
- The exact command you ran
- The full log file (`results/debug/*.log`)
- `python -c "import isaacsim; print(isaacsim.__file__); import isaaclab; print(isaaclab.__file__)"` output
- `nvidia-smi` snapshot
