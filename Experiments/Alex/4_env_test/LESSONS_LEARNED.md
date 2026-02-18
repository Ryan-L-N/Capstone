# Lessons Learned — 4-Environment Capstone Evaluation

**Project:** Comparative Policy Evaluation (Flat vs. Rough Terrain)
**Started:** February 2026
**Server:** H100 NVL (ai2ct2)

---

## Debug Iteration Log

Run `scripts/debug_5iter.sh` before each production run. Document every issue found.

### Iteration 1 — [DATE]
- **Environment tested:**
- **Policy tested:**
- **Issue found:**
- **Root cause:**
- **Fix applied:**
- **Verified fix:** [ ] Yes / [ ] No

### Iteration 2 — [DATE]
- **Environment tested:**
- **Policy tested:**
- **Issue found:**
- **Root cause:**
- **Fix applied:**
- **Verified fix:** [ ] Yes / [ ] No

### Iteration 3 — [DATE]
- **Environment tested:**
- **Policy tested:**
- **Issue found:**
- **Root cause:**
- **Fix applied:**
- **Verified fix:** [ ] Yes / [ ] No

### Iteration 4 — [DATE]
- **Environment tested:**
- **Policy tested:**
- **Issue found:**
- **Root cause:**
- **Fix applied:**
- **Verified fix:** [ ] Yes / [ ] No

### Iteration 5 — [DATE]
- **Environment tested:**
- **Policy tested:**
- **Issue found:**
- **Root cause:**
- **Fix applied:**
- **Verified fix:** [ ] Yes / [ ] No

---

## Critical Deployment Notes

These are hard-won lessons from ARL_DELIVERY training and deployment. **Read before every run.**

### Height Scan Fill Value = 0.0 (NOT 1.0)

When deploying the rough terrain policy without Isaac Lab's RayCaster (standalone mode),
the 187 height_scan dimensions must be filled with **0.0** for flat ground.

- Training environment values: `height_scan range: [-0.000002, 0.148083], mean: 0.003959`
- **Wrong (hs=1.0):** action norm 7.42 — robot collapses immediately
- **Correct (hs=0.0):** action norm 3.08 — robot walks normally

Source: `ARL_DELIVERY/08_Lessons_Learned/h100_training_lessons.md` Section 7

### GPU PhysX Required

Policies trained on GPU PhysX produce different dynamics under CPU PhysX.

- Use `backend="torch"`, `device="cuda:0"` in standalone deployment
- GPU PhysX silently ignores numpy arrays — use CUDA tensors for `set_joint_efforts()` / `set_joint_positions()`

### SimulationApp Before omni Imports

```python
# CORRECT ORDER — SimulationApp FIRST
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
# NOW import omni modules
from omni.isaac.core import World
```

Importing `omni.isaac` before creating `SimulationApp` causes silent failures.

### Observation Space: 235 Dimensions

| Term | Shape | Indices |
|------|-------|---------|
| base_lin_vel | (3,) | [0:3] |
| base_ang_vel | (3,) | [3:6] |
| projected_gravity | (3,) | [6:9] |
| velocity_commands | (3,) | [9:12] |
| joint_pos | (12,) | [12:24] |
| joint_vel | (12,) | [24:36] |
| actions | (12,) | [36:48] |
| height_scan | (187,) | [48:235] |
| **Total** | **235** | |

Height scan grid: `resolution=0.1, size=[1.6, 1.0]` = 17x11 = 187 points (includes endpoints).

### Deployment Checklist

- [ ] Observation vector: 235 dims (48 proprio + 187 height_scan)
- [ ] Height scan fill: 0.0 (flat ground)
- [ ] GPU PhysX: `backend="torch"`, `device="cuda:0"`
- [ ] PD gains: Kp=60, Kd=1.5
- [ ] Action scale: 0.25
- [ ] Decimation: 10 (50 Hz control, 500 Hz physics)
- [ ] Solver iterations: 4 position, 0 velocity
- [ ] Quaternion format: [w, x, y, z] scalar-first
- [ ] DOF order: type-grouped (all hx, all hy, all kn)
- [ ] CUDA tensors for ArticulationView setters
- [ ] Friction combine mode: "multiply"

---

## Local Development Setup (Windows)

### Installation Paths
- **Miniconda:** `C:\miniconda3\`
- **Isaac Lab conda env:** `C:\miniconda3\envs\isaaclab311\` (Python 3.11.14)
- **IsaacLab repo:** `C:\IsaacLab\` (launcher: `isaaclab.bat` / `isaaclab.sh`)
- **Isaac Sim version:** 5.1.0.0 (pip-installed in isaaclab311)
- **Isaac Lab version:** 0.54.2
- **PyTorch:** 2.7.0+cu128
- **GPU:** NVIDIA RTX 2000 Ada Generation Laptop GPU
- **pygame:** 2.6.1 (Xbox controller support)

### How to Run Locally

**Step 1: Activate the conda environment**
```
conda activate isaaclab311
```

**Step 2: Run teleop (friction example)**
```
cd "C:\Users\Gabriel Santiago\OneDrive\Desktop\Capstone Project\Capstone\Experiments\Alex\4_env_test"
python src/run_capstone_teleop.py --env friction --device xbox
```

Or using the IsaacLab launcher:
```
C:\IsaacLab\isaaclab.bat -p "C:\Users\Gabriel Santiago\OneDrive\Desktop\Capstone Project\Capstone\Experiments\Alex\4_env_test\src\run_capstone_teleop.py" --env friction --device xbox
```

**Step 3: Run evaluation (rendered, small batch)**
```
python src/run_capstone_eval.py --env friction --policy flat --num_episodes 5 --rendered --output_dir results/debug/
```

### Available Environments
```
--env friction    # Decreasing friction zones (easiest to visualize)
--env grass       # Increasing stalk density + drag
--env boulder     # Mixed polyhedra fields
--env stairs      # Ascending steps
```

### Xbox Controller
- Plug in Xbox controller via USB before launching
- pygame 2.6.1 detects it automatically
- Falls back to keyboard (WASD) if no controller found
- 12% deadzone applied to joystick inputs

### EULA for Local Headless Runs
```
set OMNI_KIT_ACCEPT_EULA=YES
```
Or in PowerShell:
```
$env:OMNI_KIT_ACCEPT_EULA="YES"
```

---

## H100 Server Notes

### CRLF Line Endings
- **Problem:** Windows `\r\n` breaks Linux bash scripts
- **Fix:** `sed -i "s/\r$//" scripts/*.sh` on server after upload
- **Prevention:** Run `dos2unix` or set `git config core.autocrlf input`

### SSH Non-Interactive Shell
- `conda` unavailable in non-interactive SSH (`.bashrc` not sourced)
- **Fix:** Explicitly source in scripts: `eval "$(/home/t2user/miniconda3/bin/conda shell.bash hook)"`
- Use `screen` for long runs — `.bashrc` IS sourced in screen sessions

### EULA Acceptance
- `export OMNI_KIT_ACCEPT_EULA=YES` required for headless execution
- Already in `.bashrc` but not available in non-interactive SSH

### Correct Conda Environment
- Use `isaaclab311` (Python 3.11) — NOT `isaaclab` (Python 3.13)
- Isaac Sim 5.1.0 requires Python 3.11

### GPU Performance
- GPU temp: 34C idle -> ~49C at 8,192 envs
- VRAM: ~10 GB at 8,192 envs (96 GB available)
- Physics clean at 8,192 envs, marginal at 32,768
- Expected throughput: ~36,000 steps/s at 8,192 envs

---

## Environment-Specific Issues

### Friction Environment
_(To be filled during debug iterations)_

### Grass / Fluid Resistance Environment
_(To be filled during debug iterations)_

### Boulder Field Environment
_(To be filled during debug iterations)_

### Staircase Environment
_(To be filled during debug iterations)_

---

## Performance Notes

_(Record timing data, memory usage, and throughput observations here)_

| Run | Envs | Batch Time | Steps/s | VRAM | Notes |
|-----|------|-----------|---------|------|-------|
| | | | | | |
