# Capstone Project Memory — AI2C Tech Capstone (MS for Autonomy)

*Carnegie Mellon University — Last updated: March 2026*

---

## Instructions for Claude (Memory Maintenance)

When asked to update this memory doc or when reviewing new git changes:

### What to track
- **Spot training only** — ignore Alex's Vision 60 / Iris drone work entirely
- New training runs, results, reward weight changes, or curriculum updates
- New bugs discovered and their fixes (add to Bug Museum)
- Changes to key file locations or script names
- H100 server config changes (IP, env names, paths)
- Shifts in active work (e.g., if Mason Hybrid finishes and a new phase starts)

### What to ignore
- Alex's tangent experiments (Vision 60, drone, Eureka, etc.)
- Library/package changelogs inside `isaacSim_env/`
- Anything under `isaacSim_env/Lib/site-packages/` — that's installed packages, not project code

### How to update
1. Check `git log` or recent commits for what changed
2. Read only the relevant `.md` files and changed `.py` files in `Experiments/Alex/`, `Experiments/Cole/`, `Experiments/Dylan/`, `Experiments/Colby/`
3. Update the relevant section in-place — do NOT append dated journal entries. Keep this doc semantic (topic-based), not chronological
4. If a bug was fixed, move it to Bug Museum with the fix noted
5. Keep total file under ~200 lines — trim stale info if something is superseded

### Companion documents (always keep in sync)

**`Plan.md`** — `Experiments/Colby/ClaudeDocs/Plan.md`
- Update Section 4 (curriculum) if new LR limits or phase exit criteria are discovered
- Update Section 9 (Lessons Learned) whenever a training run teaches something new
- Update Section 7 (Debugging table) when a new failure mode is found and fixed
- Do NOT change the environment template (Section 1) unless there's a hard breaking change

**`Results.md`** — `Experiments/Colby/ClaudeDocs/Results.md`
- Add a new Run Card for every training run (use the template at the bottom of the file)
- Update the Summary Table at the top whenever a card is added or finalized
- Mark active runs 🔄, completed runs ✅, failed runs ❌
- Add to the Failures table whenever a run is retired with a failure
- Fill in Benchmark Targets table as ARL delivery tests complete
- Numbers here go directly into PowerPoints — keep them clean, no log dumps

### Environment preferences (do not contradict)
- **Local:** always use `isaacSim_env` venv — never suggest conda/miniconda locally
- **H100:** conda `env_isaaclab` is correct there — that's server-side only
- Do not recommend `isaaclab311` — that is someone else's machine (Gabriel Santiago's)

### Tone / scope
- Colby is focused on Spot training programs. Keep entries tight and practical.
- Prefer concrete facts (file paths, numbers, commands) over narrative explanations.
- Alex writes very thorough docs but goes on tangents — extract only Spot-relevant facts.

---

## Project Overview

Training **Spot** (Boston Dynamics quadruped) to navigate difficult terrain inside NVIDIA Isaac Sim — rough terrain locomotion, obstacle courses, cluttered indoor navigation.

**Deliverable:** ARL (Army Research Lab) delivery package with trained policies, teleop, and obstacle course.

*Note: Alex's folder also contains Vision 60 and Iris drone experiments — those are his tangents, not our focus.*

---

## Team Members

- **Alex** — RL training lead, obstacle course, AI coach system
- **Colby** — H100 server setup, network admin, navigation work
- **Cole** — Hierarchical navigation policy (high-level nav on top of SpotFlatTerrainPolicy)
- **Dylan** — Navigation environments (obstacles, world building)
- **Ryan** — (see `Experiments/Ryan/`)

---

## Technical Stack

| Component | Version |
|-----------|---------|
| Isaac Sim | 5.1.0 |
| Isaac Lab | 0.54.3 (H100) / 0.54.2 (local) |
| Python | 3.11 |
| PyTorch | 2.7.0+cu128 (local) / cu129 (H100, CUDA 12.9) |
| RL Framework | RSL-RL PPO |
| Local Python env | `isaacSim_env` (venv at `MainCap/isaacSim_env/`) |
| H100 conda env | `env_isaaclab` |

---

## H100 Server

- **IP:** `172.24.254.24`
- **SSH:** `ssh t2user@172.24.254.24`
- **Password:** `!QAZ@WSX3edc4rfv`
- **GPU:** NVIDIA H100 NVL, 96 GB VRAM, Driver **575 (latest)** — downgraded from 580.126.16 for stability
- **CUDA:** **12.9** — downgraded from 13.1 for stability
- **OS:** Ubuntu 22.04.5 LTS
- **Isaac Lab path:** `/home/t2user/IsaacLab`
- **Miniconda (server-side only):** `/home/t2user/miniconda3` — H100 uses conda, but locally we use `isaacSim_env`
- **CRITICAL: Only ONE SSH session at a time** — multiple parallel sessions cause the server to become unresponsive, requiring physical reboot.
- EULA: `export OMNI_KIT_ACCEPT_EULA=YES` (already in `.bashrc`, but not in non-interactive SSH)
- **TensorBoard:** `http://172.24.254.24:6006` — live visual training metrics (reward curves, terrain level, gait quality). Looks great, use it.

### Running Training on H100
```bash
# Source conda in non-interactive SSH (bashrc not sourced automatically)
eval "$(/home/t2user/miniconda3/bin/conda shell.bash hook)"
conda activate env_isaaclab
# Use screen for long runs
screen -S training
```

### Local Environment (Windows)
Use `isaacSim_env` venv at `MainCap/isaacSim_env/` — NOT conda/miniconda locally.
```bash
# Activate local env
source isaacSim_env/Scripts/activate   # bash
# or
isaacSim_env\Scripts\activate          # cmd/PowerShell
```

---

## Critical Isaac Sim Rules (Will Break Everything If Missed)

1. **SimulationApp FIRST** — `SimulationApp` MUST be created before any `omni.isaac` import. Importing `omni.isaac.*` first causes cryptic Carbonite plugin crash.
   ```python
   from isaacsim import SimulationApp
   app = SimulationApp({})       # FIRST
   from omni.isaac.core import World  # THEN omni imports
   ```

2. **Physics callbacks at 500 Hz** — Robot control goes in `world.add_physics_callback()`, NOT the render loop. Physics = 500 Hz, render = 50 Hz.

3. **GPU PhysX required** — Use `backend="torch"`, `device="cuda:0"` in World(). GPU PhysX silently ignores numpy arrays — convert to CUDA tensors for `set_joint_efforts()` / `set_joint_positions()`.

4. **Use `os._exit(0)` not `simulation_app.close()`** — `SimulationApp.close()` causes PhysX/CUDA driver teardown deadlock (D-state, unkillable, blocks `nvidia-smi --gpu-reset`, hangs reboot). Always exit via `os._exit(0)`.

5. **Height scan fill = 0.0 (not 1.0)** — For flat ground deployment without Isaac Lab's RayCaster, fill 187 height_scan dims with 0.0. Using 1.0 produces action norm 7.42 → instant robot collapse. Training mean ≈ 0.003959.

6. **Quaternion convention** — Isaac Sim uses [w, x, y, z] (scalar-first). Pegasus/SciPy uses [x, y, z, w].

7. **CRLF trap** — Shell scripts written on Windows must have line endings fixed: `sed -i "s/\r$//" *.sh` after upload to Linux server.

8. **SpotFlatTerrainPolicy is NOT a scene object** — Don't call `world.scene.add(spot)`. Just instantiate. Initialize inside first physics callback, call `post_reset()` after `initialize()`. Position = `np.array([x,y,z])`, NOT `Gf.Vec3d()`.

9. **`nonlocal` in physics callbacks** — Variables modified inside `on_physics_step()` closures must be declared `nonlocal`, or Python raises `UnboundLocalError`.

10. **`conda activate` in non-interactive SSH (H100 only)** — `.bashrc` not sourced; use `eval "$(/home/t2user/miniconda3/bin/conda shell.bash hook)"` first. Locally, use `isaacSim_env` venv instead of conda.

---

## Spot Rough Terrain Policy (Trained)

- **Observation space:** 235 dims
  - [0:3] base_lin_vel, [3:6] base_ang_vel, [6:9] projected_gravity
  - [9:12] velocity_commands, [12:24] joint_pos, [24:36] joint_vel
  - [36:48] actions, [48:235] height_scan (17×11 = 187 pts, grid 1.6×1.0m at 0.1m res)
- **Action space:** 12 joint position offsets, scale = 0.25
- **Network:** 235 → 512 → 256 → 128 → 12 (ELU), ~350K params
- **PD gains:** Kp=60, Kd=1.5
- **Control rate:** 50 Hz (decimation=10, physics=500 Hz)
- **Solver iterations:** 4 position, 0 velocity
- **Checkpoint:** `C:\IsaacLab\logs\rsl_rl\spot_rough\48h_run\model_29999.pt`
- **Training:** 30,000 iters, H100 NVL, 8,192 parallel envs, ~53 hrs, 5.9B timesteps
- **Final reward:** +143.74 (from -0.90), episode length 573 steps
- **DOF order:** type-grouped (all hx → all hy → all kn)

---

## AI Coach System (Current Active Work — March 2026)

An LLM (Claude API / Anthropic SDK) that adjusts Isaac Lab reward weights in real-time during training:
- Monkey-patches RSL-RL `OnPolicyRunner.update()` to intercept every PPO iteration
- Every N iterations: collect metrics → emergency check (NaN/explosion) → call Claude → validate via guardrails → apply approved weight changes
- **Key insight:** `RewardManager._term_cfgs["term"].weight` is mutable at runtime — changes take effect on next step, no restart needed
- Train entry point: `scripts/rsl_rl/train_ai.py` in `Experiments/Alex/multi_robot_training/`
- Install: `pip install -e source/quadruped_locomotion/` then `pip install anthropic`

### Mason Hybrid Training (MH-2, active March 2026)
- Combined Mason's 11 proven reward terms + our 12-type terrain + AI coach (deferred mode)
- Previous config (22 terms, 2.4M-param network) hit terrain level 4.83 ceiling; Mason's config reached ~6
- Mason's setup: [512,256,128] network (800K params), adaptive KL LR, fewer reward terms
- Added 3 surgical fixes: `terrain_relative_height` (-2.0), `dof_pos_limits` (-3.0), `clamped_action_smoothness`
- Coach activation: Silent → Passive (after 300-iter plateau) → Active (tighter bounds)
- MH-1 failed (coach boosted velocity rewards → gait destroyed); MH-2 launched with VLM + gait-quality-first
- Run dir: `logs/rsl_rl/spot_hybrid_ppo/<timestamp>/`

---

## Obstacle Course (ARL Delivery)

100-meter course, 12 terrain segments:
```
START → Warm-Up → Grass+Stones → Break → STAIRS(0.75m) →
Flat → RUBBLE POOL(-0.5m) → Flat → LARGE BLOCKS →
Flat → INSTABILITY FIELD(120 bricks) → FINISH
```
- Key: No default ground plane (rubble pool at Z=-0.5m). Custom ground segments with 0.01m overlaps.
- Teleop: WASD + Xbox controller, 4 drive modes, FPV camera, dual gait (FLAT/ROUGH toggle with G/RB)

---

## Cole's Hierarchical Navigation Policy

- High-level NavigationPolicy (learned) outputs velocity commands, low-level SpotFlatTerrainPolicy (frozen) handles locomotion
- Obs: 32 dims (velocity, heading, waypoint info, 16 raycasts for obstacles, stage one-hot)
- Action: vx [-0.5,2.0], vy [-0.5,0.5], omega [-1.5,1.5]
- 8-stage curriculum: random walking → 5m waypoints → 10m → 20m → 40m → light obstacles → heavy obstacles → mixed
- Success criterion: 80% over last 100 episodes

---

## Project Plan (Future Phases — ARL Delivery)

Spot clutter navigation in indoor rooms:
- Phase 1-6: Single room 10 objects → 40 mixed movable/immovable → two rooms → stairs → beam obstacles (crawl/jump)
- Tech stack same: Isaac Sim 5.1.0, Isaac Lab, RSL-RL PPO
- Key reward terms: progress_to_goal (+2.0), goal_reached (+50), fall_penalty (-10)

---

## Key File Locations

| Purpose | Path |
|---------|------|
| ARL Delivery package | `Experiments/Alex/ARL_DELIVERY/` |
| Obstacle course main script | `ARL_DELIVERY/02_Obstacle_Course/spot_obstacle_course.py` |
| Teleop system (1142 lines) | `ARL_DELIVERY/04_Teleop_System/spot_teleop.py` |
| AI coach training | `Experiments/Alex/multi_robot_training/scripts/rsl_rl/train_ai.py` |
| AI coach source | `Experiments/Alex/multi_robot_training/source/quadruped_locomotion/` |
| 4-env eval (H100) | `Experiments/Alex/4_env_test/` |
| Cole's nav policy | `Experiments/Cole/RL_Folder_VS2/` |
| H100 setup guide | `Experiments/Colby/NetworkSetup/H100Setup.md` |
| Lessons learned | `Experiments/Alex/ARL_DELIVERY/08_Lessons_Learned/` |
| Project plan (full) | `Experiments/Alex/ARL_DELIVERY/01_Documentation/PROJECT_PLAN.md` |
| FORTHETEAM.md (overview) | `Experiments/Alex/ARL_DELIVERY/01_Documentation/FORTHETEAM.md` |

---

## Common Gotchas / Bug Museum

- **`simulation_app.close()` D-state:** Never call it. Use `os._exit(0)`. If stuck, only physical power cycle works.
- **Never kill Isaac Sim mid-run via `screen -X quit`:** Always `Ctrl-C` inside screen, wait for clean shutdown.
- **Height scan 1.0 vs 0.0:** 1.0 = instant collapse (action norm 7.42). Always 0.0 for flat.
- **235 not 208 dims:** GridPattern(resolution=0.1, size=[1.6,1.0]) = 17×11 = 187 (includes both endpoints). Off-by-one in mental model.
- **Friction combine mode:** Must be `"multiply"` to match training.
- **`Gf.Quatd` vs `Gf.Quatf`:** Use `Gf.Quatf` for orientation ops.
- **`vy ≠ 0` in `spot.forward()`:** Always pass `[vx, 0.0, wz]`.
- **OneDrive + `.git/worktrees/`:** Permission denied on git ops → `git worktree prune`.
