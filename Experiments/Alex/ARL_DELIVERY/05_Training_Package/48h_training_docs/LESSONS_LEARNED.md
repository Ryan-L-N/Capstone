# Lessons Learned — 10-Iteration Debug Run

**Date:** February 13, 2026
**Server:** H100 NVL (ai2ct2)
**Run:** `2026-02-13_16-22-03_48h_proprioception` (10 iterations, 4096 envs)

---

## 1. Setup Issues Encountered & Fixed

### CRLF Line Endings
- **Problem:** Shell scripts written on Windows have `\r\n` line endings. Linux bash chokes on `\r`.
- **Fix:** `sed -i "s/\r$//" *.sh` on the server after upload.
- **Prevention:** All `.sh` files must be converted before upload.

### SSH Non-Interactive Shell
- **Problem:** `conda` is not available in non-interactive SSH because `.bashrc` is not sourced.
- **Fix:** Explicitly source conda: `eval "$(/home/t2user/miniconda3/bin/conda shell.bash hook)"`
- **Note:** The `debug_10iter.sh` script's `eval "$(conda shell.bash hook)"` works when run
  interactively in a screen session, but fails via direct `ssh ... 'bash script.sh'`.

### EULA Acceptance
- **Problem:** `import isaacsim` prompts for EULA in non-interactive mode, causing EOF error.
- **Fix:** `export OMNI_KIT_ACCEPT_EULA=YES` (already in `.bashrc` but not sourced non-interactively).
- **Note:** When running via `screen`, `.bashrc` IS sourced, so this is only a problem for
  direct SSH command execution.

### Missing Spot Rough Terrain Configs
- **Problem:** The H100's Isaac Lab installation only has `SpotFlatEnvCfg` — no rough terrain config.
- **Fix:** Uploaded 3 files from local Isaac Lab:
  - `rough_env_cfg.py` — Spot rough terrain environment config
  - `__init__.py` — Updated gym registrations (added `Isaac-Velocity-Rough-Spot-v0`)
  - `agents/rsl_rl_ppo_cfg.py` — Added `SpotRoughPPORunnerCfg`
- **Note:** The Spot-specific `mdp/rewards.py` and `mdp/events.py` already existed on the H100.

---

## 2. Observation Space Discovery

**CRITICAL FINDING:** The observation space is **235 dimensions**, not 208 as documented.

| Observation Term   | Shape  | Notes                                      |
|--------------------|--------|--------------------------------------------|
| base_lin_vel       | (3,)   |                                            |
| base_ang_vel       | (3,)   |                                            |
| projected_gravity  | (3,)   |                                            |
| velocity_commands  | (3,)   |                                            |
| joint_pos          | (12,)  |                                            |
| joint_vel          | (12,)  |                                            |
| actions            | (12,)  |                                            |
| height_scan        | **(187,)** | Was assumed 160. GridPattern(0.1, [1.6, 1.0]) = 17x11 = 187 |
| **Total**          | **235**|                                            |

**Impact on deployment:** The standalone policy wrapper must build a 235-dim observation vector,
with the height scan occupying indices [48:235] (187 values). The previous code assumed 208 dims
with height_scan at [48:208] (160 values). **This mismatch would cause the policy to fail.**

The height scan grid: `resolution=0.1, size=[1.6, 1.0]` produces a 17x11 = 187 point grid
(not 16x10 = 160 as previously calculated — the grid includes endpoints).

---

## 3. Training Metrics Analysis

### Throughput
- **16,200 steps/s at 4,096 envs** on H100
- Iteration time: ~6.0 seconds
- With 8,192 envs, expect ~36,000 steps/s (from stress test data)
- **48h estimate at 8,192 envs:** ~28,800 iterations (within our 30,000 target)

### Reward Progression (10 iterations)
| Iter | Mean Reward | Episode Length | Body Contact Term | Terrain Level |
|------|------------|----------------|-------------------|---------------|
| 0    | -0.90      | 19.68          | 22.4%             | 3.18          |
| 4    | -1.29      | 32.60          | 93.4%             | 1.35          |
| 6    | -1.52      | 33.02          | 96.2%             | 0.84          |

**Observations:**
- Mean reward starts negative and gets more negative — **this is expected** at the beginning
  because the stronger penalties (our overrides) dominate before the policy learns positive rewards.
- Episode length increased from 19.7 to 33.0 steps — the robot is surviving longer.
- Body contact termination is very high (96%) — most episodes end with the robot falling.
  This will decrease as training progresses.
- Terrain levels are decreasing (3.18 → 0.84) — the curriculum is moving robots to EASIER
  terrain because they're failing on harder terrain. This is correct behavior.

### Reward Overrides Verified
All 14 reward terms appeared in the training output with correct weights:
- `base_linear_velocity: 7.0` (override from 5.0)
- `foot_clearance: 2.5` (override from 2.0)
- `action_smoothness: -2.0` (override from -1.0)
- `base_motion: -3.0` (override from -2.0)
- `base_orientation: -5.0` (override from -3.0)
- `foot_slip: -1.0` (override from -0.5)
- `joint_acc: -0.0005` (override from -0.0001)
- `joint_pos: -1.0` (override from -0.7)
- `joint_torques: -0.002` (override from -0.0005)
- `joint_vel: -0.02` (override from -0.01)

### External Force Perturbation
Confirmed enabled: `base_external_force_torque` appears in active event terms (reset mode).

---

## 4. Network Architecture

Confirmed actor and critic:
```
Actor:  235 → 512 → 256 → 128 → 12 (ELU activations)
Critic: 235 → 512 → 256 → 128 → 1  (ELU activations)
```

Initial noise std: 0.79 (close to configured 0.8), decreasing during training.

---

## 5. Action Items for Full 48h Run

1. **Update deployment code** — Change observation vector from 208 to 235 dimensions,
   height scan from 160 to 187 values.
2. **Run with 8,192 envs** — Debug used 4,096, full run should use 8,192 for optimal throughput.
3. **Use `screen` session** — Direct SSH execution works but screen is more reliable for 48h.
4. **Monitor body_contact termination** — Should decrease from 96% to <10% by iteration 10,000.
   If still >50% at iteration 5,000, the penalty weights may be too aggressive.
5. **Clean up debug run** — `rm -rf ~/IsaacLab/logs/rsl_rl/spot_rough/2026-02-13_16-22-03_48h_proprioception/`

---

## 6. H100 Performance Notes

- GPU temperature: 34°C idle → ~49°C at 8,192 envs (from stress test)
- VRAM: ~10 GB at 8,192 envs (96 GB available — plenty of headroom)
- Physics: Clean at 8,192 envs, marginal at 32,768
- The H100 has **1 TB of system RAM** and **120 logical cores** — no CPU bottleneck

---

## 7. Deployment Lessons (Post-Training)

### Height Scan = 0.0, NOT 1.0

**CRITICAL**: When deploying the trained policy in standalone Isaac Sim (without Isaac Lab's
RayCaster), the 187 height_scan observation dimensions must be filled with **0.0**, not 1.0.

The original source code analysis concluded height_scan clips to 1.0 due to the RayCaster's
20m Z-offset. **This was wrong.** Actual runtime values from the training environment:
```
height_scan range: [-0.000002, 0.148083]
height_scan mean:  0.003959
```

Impact of wrong value: `hs=1.0` → action norm 7.42 (robot falls), `hs=0.0` → action norm 3.08 (walks).

**Lesson:** Always print actual observation values from the training environment. Never rely
on source code tracing alone for understanding observation ranges.

### GPU PhysX Required for Standalone Deployment

The standalone World() defaults to CPU PhysX. Policies trained on GPU PhysX produce different
dynamics. Use `backend="torch"`, `device="cuda:0"` in standalone deployment.

GPU PhysX silently ignores numpy arrays in `set_joint_efforts()` and `set_joint_positions()`.
Use CUDA tensors or wrap the robot API with NumpyRobotWrapper.

### Deployment Checklist (Verified Working)

- [x] Observation vector: 235 dims (48 proprio + 187 height_scan)
- [x] Height scan fill: **0.0** (flat ground assumption)
- [x] GPU PhysX: `backend="torch"`, `device="cuda:0"`
- [x] PD gains: Kp=60, Kd=1.5
- [x] Action scale: 0.25
- [x] Decimation: 10 (50 Hz control, 500 Hz physics)
- [x] Solver iterations: 4 position, 0 velocity
- [x] Quaternion: [w, x, y, z] scalar-first
- [x] DOF order: type-grouped (all hx, all hy, all kn)
- [x] CUDA tensors for ArticulationView setters
