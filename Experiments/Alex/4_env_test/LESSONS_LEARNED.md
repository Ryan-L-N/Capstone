# Lessons Learned — 4-Environment Capstone Evaluation

**Project:** Comparative Policy Evaluation (Flat vs. Rough Terrain)
**Started:** February 2026
**Server:** H100 NVL (ai2ct2)

---

## Debug Iteration Log

Run `scripts/debug_5iter.sh` before each production run. Document every issue found.

### Iteration 1 — 2026-02-18
- **Environment tested:** Friction
- **Policy tested:** Flat
- **Issue found:** `AttributeError: 'SpotFlatTerrainPolicy' object has no attribute 'name'`
- **Root cause:** `world.scene.add(spot)` — SpotFlatTerrainPolicy is not a scene object
- **Fix applied:** Removed `world.scene.add()`, use `np.array()` for position instead of `Gf.Vec3d()`
- **Verified fix:** [x] Yes

### Iteration 2 — 2026-02-18
- **Environment tested:** Friction
- **Policy tested:** Flat
- **Issue found:** Robot legs don't move (idle collapse)
- **Root cause:** `initialize()` called before physics was running; needed to follow official `quadruped_example.py` pattern (init inside first physics callback)
- **Fix applied:** Moved `initialize()` + `post_reset()` into first physics callback
- **Verified fix:** [x] Yes

### Iteration 3 — 2026-02-18
- **Environment tested:** Friction
- **Policy tested:** Flat
- **Issue found:** `UnboundLocalError: cannot access local variable 'drive_mode_idx'`
- **Root cause:** Missing `nonlocal drive_mode_idx` in `on_physics_step()` closure
- **Fix applied:** Added `drive_mode_idx` to `nonlocal` declaration
- **Verified fix:** [x] Yes — robot walks, Xbox responds, 5.5min session with 0 errors

### Iteration 4 — 2026-02-19 (H100 Full Debug)
- **Environment tested:** ALL 4 (friction, grass, boulder, stairs)
- **Policy tested:** ALL 2 (flat, rough) — 8 combos total
- **Result:** All 8 passed (5 episodes each, ~34 min total)
- **Timing per combo:** 239-292s (includes Isaac Sim startup + 5 episodes)
- **Issue found:** None — all combos ran cleanly in headless mode
- **Notes:**
  - H100 conda env = `env_isaaclab` (not `isaaclab311`)
  - ~41s per episode (flat policies TIMEOUT at zone 2, ~12m progress)
  - Rough policies also TIMEOUT at zone 2, ~10-11m progress
  - GPU VRAM: ~4 GB per run (96 GB available)

### Iteration 5 — 2026-02-19 (Production Attempt)
- **Environment tested:** friction (flat) — first combo of 8
- **Policy tested:** flat
- **Issue found:** Isaac Sim startup hangs after killing screen session mid-run
- **Root cause:** Killing `screen -X quit` left zombie (PID 941461, Z state) and D-state processes. Zombie held 4GB GPU memory. New Isaac Sim processes hang on `nvidia-smi -q` during startup.
- **Fix applied:** Added "Never Kill Isaac Sim Mid-Run" lesson. Server required physical power cycle.
- **Prevention:** Always use Ctrl-C inside screen, never `screen -X quit`

### Iteration 6 — 2026-02-19 (Zombie Process Fix)
- **Environment tested:** friction (flat) — revalidation after fix
- **Policy tested:** flat
- **Issue found:** `simulation_app.close()` causes D-state zombie after every run
- **Root cause:** PhysX/CUDA driver teardown deadlocks in kernel — unkillable even by SIGKILL
- **Fix applied:** 3-layer defense:
  1. `os._exit(0)` replaces `simulation_app.close()` in `run_capstone_eval.py`
  2. Signal handler (SIGINT/SIGTERM) saves metrics then `os._exit(0)`
  3. Shell scripts: `timeout` + `pkill -f` safety nets in all 3 scripts
- **Files modified:**
  - `src/run_capstone_eval.py` — signal handler + os._exit(0)
  - `scripts/debug_5iter.sh` — timeout 300 + pkill cleanup
  - `scripts/run_full_eval.sh` — timeout 600 + pkill cleanup
  - `scripts/run_h100_master.sh` — timeout 300/7200 + pkill cleanup
- **Verified fix:** [ ] Pending H100 revalidation

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

### SpotFlatTerrainPolicy Is NOT a Scene Object

`SpotFlatTerrainPolicy` inherits `PolicyController → BaseController`, NOT a scene object.

- **WRONG:** `world.scene.add(spot)` → `AttributeError: no attribute 'name'`
- **CORRECT:** Just instantiate and initialize — no `scene.add()` needed

The constructor (`__init__`) already places the USD on stage via `define_prim()` + `AddReference()`.

### Initialize Robot Inside First Physics Step

Per the official `quadruped_example.py`, the robot must be initialized AFTER physics is running:

```python
spot = SpotFlatTerrainPolicy(prim_path="/World/Spot", name="Spot", position=np.array([0, 0, 0.6]))
world.reset()  # Start physics timeline

# Option A: Initialize on first physics callback (official pattern)
def on_physics_step(step_size):
    if not physics_ready:
        spot.initialize()
        spot.post_reset()
        physics_ready = True
        return
    spot.forward(step_size, cmd)

# Option B: Initialize after one physics step
world.step(render=False)
spot.initialize()
spot.post_reset()
```

- **Must call `post_reset()` after `initialize()`**
- Position uses `np.array([x,y,z])`, NOT `Gf.Vec3d()`
- `spot.robot` is the `SingleArticulation` (access pose, velocities, etc.)

### Python Closure Nonlocal in Physics Callbacks

Variables modified inside `on_physics_step()` that are defined in the outer scope
MUST be declared `nonlocal`. Otherwise Python treats them as local and raises:
`UnboundLocalError: cannot access local variable 'X' where it is not associated with a value`

### Deployment Checklist

- [ ] Observation vector: 235 dims (48 proprio + 187 height_scan)
- [ ] Height scan fill: 0.0 (flat ground)
- [ ] GPU PhysX: `backend="torch"`, `device="cuda:0"`
- [ ] PD gains: Kp=60, Kd=1.5
- [ ] Action scale: 0.2 (all configs use 0.2, NOT 0.25)
- [ ] Decimation: 10 (50 Hz control, 500 Hz physics)
- [ ] Solver iterations: 4 position, 0 velocity
- [ ] Quaternion format: [w, x, y, z] scalar-first
- [ ] DOF order: type-grouped (all hx, all hy, all kn)
- [ ] CUDA tensors for ArticulationView setters
- [ ] Friction combine mode: "multiply"
- [ ] No `world.scene.add()` for SpotFlatTerrainPolicy
- [ ] `post_reset()` after `initialize()`
- [ ] All `nonlocal` declarations in physics callbacks

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
- **Local (Windows):** `isaaclab311` (Python 3.11)
- **H100 server:** `env_isaaclab` (Python 3.11)
- Scripts use fallback: `conda activate env_isaaclab 2>/dev/null || conda activate isaaclab311`
- Isaac Sim 5.1.0 requires Python 3.11

### Never Call `simulation_app.close()` — Use `os._exit(0)` Instead
- **Problem:** `SimulationApp.close()` triggers GPU driver cleanup that enters a kernel-level D-state (uninterruptible sleep). The process becomes unkillable — `kill -9`, `Ctrl-C`, `SIGTERM` all fail. The zombie holds GPU memory (~4.5 GB), blocks `nvidia-smi --gpu-reset`, and even `sudo reboot` can hang indefinitely.
- **Root cause:** PhysX/CUDA driver teardown sequence deadlocks in the kernel. This is an NVIDIA driver bug, not application code.
- **Fix (3-layer defense):**
  1. **Code:** Replace `simulation_app.close()` with `os._exit(0)` at the end of `run_capstone_eval.py`. All data is already saved at this point, so skipping Python cleanup is safe.
  2. **Signal handler:** `SIGINT`/`SIGTERM` handler saves pending metrics via `_metrics_collector_ref.save()` then calls `os._exit(0)`. This makes `Ctrl-C` and `kill` safe.
  3. **Shell scripts:** `timeout` wrapper (300s debug, 7200s production) + `pkill -f` cleanup between combos. Belt-and-suspenders safety net.
- **Recovery:** If D-state processes are stuck, the only fix is a **physical power cycle** (IPMI/BMC or power button). Software reboot will hang.
- **Discovered:** 2026-02-19, friction_flat debug run. PID 3091 entered D-state after 5 episodes completed successfully. Required physical server reboot.

### Never Kill Isaac Sim Mid-Run via Screen Quit
- **Problem:** Killing a `screen` session (`screen -X quit`) while Isaac Sim is running leaves zombie (Z) and D-state processes that hold GPU memory.
- **Impact:** Same D-state issue as above — unkillable, blocks GPU reset and reboot.
- **Fix:** Always use `Ctrl-C` inside screen (now safe with signal handler), or let the run complete naturally. If you must abort: attach to the screen (`screen -r`), send `Ctrl-C`, wait for "[SHUTDOWN] Force-exiting", then exit.

### GPU Performance
- GPU temp: 34C idle -> ~49C at 8,192 envs
- VRAM: ~10 GB at 8,192 envs (96 GB available)
- Physics clean at 8,192 envs, marginal at 32,768
- Expected throughput: ~36,000 steps/s at 8,192 envs

---

## Bugs Fixed — Mason Hybrid Eval (2026-03-12)

### Bug E-1: Decimation Mismatch — Policy Running at 5 Hz Instead of 50 Hz

- **Symptom:** Robot spasms violently, action norms escalate (10 → 31) within 10 steps, falls immediately.
- **Root cause:** The main loop calls `forward()` once per `world.step()`. But `world.step()` advances `rendering_dt / physics_dt = 10` physics substeps. With `DECIMATION=10` in `SpotRoughTerrainPolicy.forward()`, the policy only evaluates every 10th `forward()` call = every 100 physics steps = **5 Hz**. Training ran at 50 Hz.
- **Why it worked in lava arenas:** The lava arenas use `world.add_physics_callback()` which fires at every physics step (500 Hz). With decimation=10, the policy evaluates at 50 Hz. The 4_env_test uses a simple `for` loop that calls `forward()` once per `world.step()`.
- **Fix:** Set `robot_policy._decimation = 1` after `initialize()` in `run_capstone_eval.py`. Since the main loop already runs at 50 Hz (control rate), decimation of 1 gives the correct policy frequency.
- **Impact:** Without fix, robot survives < 1 second. With fix, robot completes full 50m friction course.
- **Files modified:** `run_capstone_eval.py` (lines 227, 271)

### Bug E-2: Waypoint Follower Premature Termination

- **Symptom:** Episode ends at exactly 39.5m with status "TIMEOUT" (or "COMPLETE" falsely). Robot was still walking fine. Console showed 39.5m but JSONL showed episode_length of only 44.52s.
- **Root cause:** When robot reaches x=39.5m, it passes waypoint 4's threshold (40.0 - 0.5 = 39.5), which increments `current_wp` to 5. Then `is_done` checks `current_wp >= len(waypoints) - 1` → `5 >= 5 = True`. The episode terminates before the robot reaches waypoint 5 (x=50m).
- **Fix:** Changed `is_done` from `>=` to `>` (strictly greater than). Changed advance condition from `< len - 1` to `< len` to allow incrementing past the last waypoint only when actually reaching it.
- **Impact:** Without fix, every run terminates at 39.5m regardless of policy quality. With fix, robot can complete the full 50m course.
- **Files modified:** `navigation/waypoint_follower.py` (lines 53, 93)

### Bug E-3: Missing Scene Lighting

- **Symptom:** "The lights are off" — dark/black scene in rendered mode.
- **Root cause:** `run_capstone_eval.py` only created a ground plane and environment objects. No lights were added to the USD stage.
- **Fix:** Added dome light (ambient sky, intensity=500) and distant light (sun, intensity=3000) after ground plane creation.
- **Files modified:** `run_capstone_eval.py` (lines 172-187), added `UsdLux` import

### Bug E-4: `apply_gains()` CUDA Tensor Error (Non-Fatal)

- **Symptom:** `TypeError: can't convert cuda:0 device type tensor to numpy` in `apply_gains()`.
- **Root cause:** `SpotRoughTerrainPolicy.apply_gains()` passes CUDA tensors to `ArticulationView.set_gains()`, but the standalone Isaac Sim API uses numpy backend (not GPU ArticulationView). The function was written for the Isaac Lab GPU pipeline.
- **Impact:** Non-fatal — the gains were already correct (Kp=60, Kd=1.5) and solver iterations already 4/0, set by the flat policy. The `apply_gains()` call is still useful as a safety check (prints BEFORE values to verify).
- **Future fix:** Use numpy arrays instead of CUDA tensors in `apply_gains()` for standalone mode.

### Bug E-5: `__pycache__` Stale Bytecode

- **Symptom:** Changed `EPISODE_TIMEOUT` from 600 to 1800 in `eval_cfg.py`, but Python kept using old 600s value. Episode completed in same wall time with same progress.
- **Root cause:** Python's `.pyc` cache in `configs/__pycache__/` had stale bytecode. On some filesystems (OneDrive sync?), the `.py` modification time didn't properly invalidate the cache.
- **Fix:** `rm -rf configs/__pycache__/` before re-running.
- **Prevention:** Always clear `__pycache__` after editing config files if behavior doesn't change.

### Mason Hybrid `--mason` Flag Integration

Added `--mason` CLI flag to `run_capstone_eval.py` that:
1. Passes `mason_baseline=True` to `SpotRoughTerrainPolicy` constructor
2. Uses Mason's observation order: height_scan(187) first, then proprioception(48)
3. Uses `ACTION_SCALE_MASON = 0.2` (vs our 0.25)
4. Network architecture [512, 256, 128] (matches Mason's training)

### Bug E-6: Height Scan Blind to Terrain (PhysX Raycasting Fix)

- **Symptom:** Robot could not see boulders or stairs — height scan was all 0.0 (flat ground assumption). Boulder progress: 20.6m. Policy had no terrain awareness despite being trained with height scan.
- **Root cause:** Original `_cast_height_rays()` only supported an analytical `ground_height_fn` (stairs-only) or returned zeros. No raycasting against actual USD geometry.
- **Fix:** Implemented PhysX scene query raycasting via `omni.physx.get_physx_scene_query_interface()`:
  - Cast 187 rays downward from `body_z + 20.0m` through a 17×11 grid (1.6m × 1.0m, 0.1m resolution)
  - Filter self-hits on `/World/Robot/*` by re-casting from `body_z - 0.1`
  - Formula: `height = body_z - hit_z - 0.5`, clipped to [-1.0, 1.0]
  - Works for ALL environments (boulders, stairs, friction, grass)
- **Result:** Boulder improved 20.6m → 31.6m (+11m, +1 zone). Stairs: raycast confirmed hitting `/World/Staircase/zone_1/step_0,1,2` + `/World/GroundPlane`.
- **Split `ground_fn`:** `ground_fn_metrics` (analytical for stairs fall detection) vs `ground_fn_scanner` (always None → PhysX raycast for height scan).
- **Files modified:** `spot_rough_terrain_policy.py` (`_cast_height_rays()`), `run_capstone_eval.py` (split ground_fn)

### Bug E-7: `apply_gains()` Silent Failure — CUDA Tensors for Articulation Setters

- **Symptom:** Robot's leg stuck in the floor on stairs. Gains appeared correct in `apply_gains()` BEFORE printout, but the set calls silently failed.
- **Root cause:** `SpotRoughTerrainPolicy.apply_gains()` used `device=dev` (CUDA) for all tensors passed to `ArticulationView.set_gains()`, `set_solver_position_iteration_counts()`, `set_solver_velocity_iteration_counts()`, and `set_max_depenetration_velocity()`. In standalone Isaac Sim, these setters internally use numpy indexing — CUDA tensors cause `TypeError: can't convert cuda:0 device type tensor to numpy`. The try/except blocks swallowed the errors.
- **Impact:** PD gains, solver iterations, and depenetration velocity were NOT being set. The robot used whatever defaults the flat policy had left behind.
- **Fix:** Changed ALL articulation property setters from `device=dev` to `device="cpu"`:
  ```python
  kps = torch.full((1, n_dof), TRAINING_STIFFNESS, device="cpu")
  kds = torch.full((1, n_dof), TRAINING_DAMPING, device="cpu")
  av.set_gains(kps=kps, kds=kds)
  ```
- **Files modified:** `spot_rough_terrain_policy.py` (apply_gains method)

### Bug E-8: ACTION_SCALE Mismatch — 0.25 vs 0.2

- **Symptom:** Robot "folded up" immediately after spawning — legs overdriven, violent collapse. Happened for AI-coached model (non-mason mode).
- **Root cause:** `spot_rough_terrain_policy.py` used `ACTION_SCALE = 0.25` for non-mason configs, but the actual training config uses `scale=0.2` for ALL policies (confirmed at `env_cfg.py` line 111). The extra 25% action scale caused joint targets beyond safe range.
- **Fix:** Changed `ACTION_SCALE = 0.2` and `ACTION_SCALE_MASON = 0.2` (both configs use 0.2).
- **Files modified:** `spot_rough_terrain_policy.py`

### Bug E-9: Actor Architecture Mismatch — Hardcoded [512,256,128]

- **Symptom:** AI-coached v8 model loaded but produced garbage actions. Robot fell over immediately, not even attempting to walk.
- **Root cause:** `SpotRoughTerrainPolicy` hardcoded the actor network as `[512, 256, 128]` (Mason's architecture). The AI-coached v8 model uses `[1024, 512, 256]` (our config). Loading a 2.4M-param checkpoint into an 800K-param network silently truncated weights.
- **Fix:** Implemented `_build_actor()` static method that auto-detects hidden layer sizes from checkpoint weight shapes:
  ```python
  @staticmethod
  def _build_actor(checkpoint_path):
      ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
      state = ckpt["model_state_dict"]
      actor_state = {k.replace("actor.", ""): v for k, v in state.items() if k.startswith("actor.")}
      weight_keys = sorted(k for k in actor_state if k.endswith(".weight"))
      layers = []
      for wk in weight_keys:
          out_dim, in_dim = actor_state[wk].shape
          layers.append(nn.Linear(in_dim, out_dim))
          if out_dim != ACT_DIM:
              layers.append(nn.ELU())
      actor = nn.Sequential(*layers)
      actor.load_state_dict(actor_state)
      return actor
  ```
- **Impact:** Now supports any checkpoint architecture without code changes. Prints detected sizes: `[ROUGH] Auto-detected architecture: 235 -> [1024, 512, 256] -> 12`
- **Files modified:** `spot_rough_terrain_policy.py`

### Leg Penetration Through Terrain — Joint Clamping + Depenetration Fix

- **Symptom:** Robot's legs clip through terrain geometry (especially stairs). Legs get "stuck in the floor" and the robot can't recover.
- **Root cause:** AI-coached model produces larger joint excursions than Mason's conservative network. Combined with Bug E-7 (depenetration velocity not being set), PhysX couldn't resolve the penetration fast enough.
- **Fix (3 parts):**
  1. **Joint position clamping to URDF limits** in `forward()`:
     ```python
     joint_lower = [-0.785, -0.785, ..., -2.793, -2.793, ...]  # URDF min
     joint_upper = [ 0.785,  0.785, ..., -0.254, -0.254, ...]  # URDF max
     target_pos = np.clip(target_pos, joint_lower, joint_upper)
     ```
  2. **Depenetration velocity** increased from 1.0 → 10.0 m/s for faster collision resolution
  3. **CPU tensors** (Bug E-7 fix) — ensures depenetration velocity is actually applied
- **Note:** Solver iterations 16/4 was attempted but froze the simulation. 4/1 is the working setting.
- **Files modified:** `spot_rough_terrain_policy.py`

### Deployment Checklist — External Policy in 4_env_test

- [ ] `--mason` flag if using Mason's obs order (height_scan first)
- [ ] `--checkpoint` path to the correct `.pt` file
- [ ] `robot_policy._decimation = 1` (loop is already at 50 Hz)
- [ ] `robot_policy.apply_gains()` called after `initialize()` (verifies settings)
- [ ] Clear `__pycache__` after config changes
- [ ] Scene lighting exists (dome + sun)
- [ ] Waypoint follower `is_done` uses strict `>` (not `>=`)
- [ ] PhysX raycasting enabled (ground_height_fn=None → real terrain geometry)
- [ ] ACTION_SCALE = 0.2 (not 0.25 — ALL configs use 0.2)
- [ ] All articulation setters use `device="cpu"` tensors (not CUDA)
- [ ] Actor architecture auto-detected from checkpoint (no hardcoded sizes)

---

## Environment-Specific Issues

### Friction Environment

**Mason Hybrid No-Coach (model_13000.pt) — 2026-03-12:**
- COMPLETE — 49.5m, all 5 zones, 273.9s wall time
- Mean roll: 0.044 rad (very stable), mean velocity: 0.88 m/s
- No falls, no issues — clean traversal

**AI-Coached v8 (model_10600.pt) — 2026-03-13:**
- COMPLETE — 49.5m, all 5 zones, 50.2s wall time
- Faster than Mason hybrid (50s vs 274s) — more aggressive gait

**100-Episode Mason Baseline Sweep (model_19999.pt) — 2026-03-13:**
- 100/100 episodes completed (headless)
- Mean progress: 33.5m ± 10.0m
- Fall rate: 83% (high variance — some episodes very good, some early falls)

### Grass / Fluid Resistance Environment

**Mason Hybrid No-Coach (model_13000.pt) — 2026-03-12:**
- Results pending (raycasting now enabled)

**AI-Coached v8 (model_10600.pt) — 2026-03-13:**
- FELL — 41.2m, 5/5 zones reached, 121.0s wall time
- Made it through all zones but fell near the end

**100-Episode Mason Baseline Sweep (model_19999.pt) — 2026-03-13:**
- 50/100 episodes completed (interrupted)
- Mean progress: 28.9m ± 6.7m
- Fall rate: 22%

### Boulder Field Environment

**Mason Hybrid No-Coach (model_13000.pt) — 2026-03-12:**
- WITHOUT raycasting: 20.6m progress, fell (robot blind to boulders)
- WITH PhysX raycasting: 31.6m progress, zone 3, fell — +11m improvement
- Raycasting allows policy to detect boulder geometry ahead

**AI-Coached v8 (model_10600.pt) — 2026-03-13:**
- FELL — 23.0m, 3/5 zones, 34.8s wall time

### Staircase Environment

**AI-Coached v8 (model_10600.pt) — 2026-03-13:**
- FELL — 12.7m, 2/5 zones, 37.1s wall time
- Leg penetration issue (Bug E-7 + joint clamping fix applied)

**Zone boundary walls from platforms (FIXED 2026-02-18)**
- **Problem:** 2m "recovery platforms" between zones created visible walls. Each platform was a solid cube from z=0 to the zone's full cumulative height, but its leading edge overlapped with steps that were much shorter — creating a ~20cm+ vertical wall face at each zone boundary.
- **Fix:** Removed platforms entirely. Replaced with 5 transition steps at the start of zones 2-5 that linearly interpolate riser heights from the previous zone to the current zone. Also added 0.1m fill cubes at zone ends (33 steps × 0.30m = 9.9m < 10.0m zone width).
- **Key numbers:**
  - `TRANSITION_STEPS = 5`, fractions 1/6 through 5/6
  - Max riser increment at boundaries: `(curr_h - prev_h) / 6` ≈ 8.3mm
  - Total elevation: 20.95m (was 21.45m without transitions)
  - Both USD geometry (`stairs_env.py`) and elevation function (`zone_params.py:get_stair_elevation()`) use the same transition logic

**Floating-point precision in elevation calculations**
- **Problem:** Adding step_height in a loop (0.03 × 33 = 0.9900000000000007) vs multiplication (0.03 * 33 = 0.99) caused `test_monotonically_increasing` to fail — elevation appeared to decrease at zone boundaries.
- **Fix:** Use multiplication (`step_height * steps_climbed`) for non-transition steps, not a per-step accumulation loop.

**Mason Hybrid No-Coach (model_13000.pt) — 2026-03-12:**
- WITH PhysX raycasting: 12.7m progress, zone 2, fell — 95.2s wall time
- Raycast confirmed hitting stair geometry: `/World/Staircase/zone_1/step_0,1,2`
- Height scan range at spawn: [-0.019, 0.071] (correct — flat ground near first steps)
- Stairs is the hardest environment — zone 2 is a reasonable baseline for single-episode test

---

## Performance Notes

_(Record timing data, memory usage, and throughput observations here)_

| Run | Envs | Batch Time | Steps/s | VRAM | Notes |
|-----|------|-----------|---------|------|-------|
| | | | | | |
