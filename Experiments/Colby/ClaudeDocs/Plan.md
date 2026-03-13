# Colby's RL Training & Environment Planning Guide

*Reference for creating new Spot environments and training pipelines.*
*Update this doc whenever we learn something new about training.*

---

## How to Use This Doc

- **Before creating a new environment:** read Sections 1–3
- **Before launching a training run:** read Section 4 + the pre-flight checklist (Section 6)
- **When debugging a broken run:** go straight to Section 7

---

## 1. Environment Architecture Pattern

Every Isaac Sim environment we build follows this structure. Do not deviate.

### File Layout
```
my_experiment/
├── env.py               # Environment class (world setup, spawning, reset, step)
├── policy.py            # MLP policy network (if custom; otherwise use RSL-RL's)
├── trainer.py           # PPO training loop (or reuse train_ai.py)
├── config.yaml          # All tunable parameters (reward weights, curriculum, arena)
├── train.sh             # H100 launch script
└── debug.sh             # 10-iteration smoke test
```

### Environment Class Template

```python
# RULE: SimulationApp MUST be created before ANY omni.isaac import
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})   # headless for training

from omni.isaac.core import World
import numpy as np

class MySpotEnv:
    def __init__(self, config):
        # World: always GPU PhysX
        self.world = World(
            physics_dt=1/500,
            rendering_dt=10/500,        # 50Hz render, 500Hz physics
            backend="torch",
            device="cuda:0"             # GPU PhysX required to match training dynamics
        )
        self._physics_ready = False
        self._setup_scene()

    def _setup_scene(self):
        # SpotFlatTerrainPolicy: do NOT call world.scene.add(spot)
        # Just instantiate — it places the USD on stage itself
        self.spot = SpotFlatTerrainPolicy(
            prim_path="/World/Spot",
            name="Spot",
            position=np.array([0, 0, 0.6])   # np.array, NOT Gf.Vec3d
        )
        self.world.reset()
        self.world.add_physics_callback("control", self._on_physics_step)

    def _on_physics_step(self, step_size):
        if not self._physics_ready:
            self.spot.initialize()
            self.spot.post_reset()      # always post_reset() after initialize()
            self._physics_ready = True
            return
        # Control logic at 500Hz here
        self.spot.forward(step_size, self._current_cmd)

    def reset(self):
        self._physics_ready = False
        self.world.reset()

    def step(self, action):
        # vy must always be 0.0 — undefined behavior otherwise
        vx = float(action[0])
        wz = float(action[1])
        self._current_cmd = np.array([vx, 0.0, wz])

        for _ in range(10):             # decimation=10 → 50Hz control
            self.world.step(render=False)

        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_done()
        return obs, reward, done, {}

    # NEVER call app.close() — causes D-state kernel deadlock
    def close(self):
        import os
        os._exit(0)
```

### Hard Rules
- Robot control goes in `add_physics_callback()`, never in the render loop
- `post_reset()` must follow `initialize()` on every episode reset
- All `ArticulationView` setters need CUDA tensors (not numpy) under GPU PhysX
- `os._exit(0)` at shutdown — `simulation_app.close()` deadlocks the kernel
- Don't add `SpotFlatTerrainPolicy` to `world.scene`
- Variables modified inside `on_physics_step()` closures must be declared `nonlocal`

---

## 2. Observation & Action Space (Spot)

### High-level nav on top of SpotFlatTerrainPolicy (Cole's pattern)
- **Action out:** `[vx, 0.0, wz]` — vy is always 0.0
- **Obs in:** design your own. Cole's: 32 dims — velocity(3), heading(2), waypoint(3), 16 raycasts, stage one-hot(8)

### Direct rough terrain policy (235-dim obs)
| Term | Dims | Indices |
|------|------|---------|
| base_lin_vel | 3 | [0:3] |
| base_ang_vel | 3 | [3:6] |
| projected_gravity | 3 | [6:9] |
| velocity_commands | 3 | [9:12] |
| joint_pos | 12 | [12:24] |
| joint_vel | 12 | [24:36] |
| actions | 12 | [36:48] |
| height_scan | 187 | [48:235] |

- Height scan fill = **0.0** for flat ground (1.0 causes action norm 7.42 → instant collapse)
- GridPattern: `resolution=0.1, size=[1.6, 1.0]` = 17×11 = **187 points** (both endpoints included)
- PD gains: Kp=60, Kd=1.5 | Action scale: 0.25 | Decimation: 10

---

## 3. Reward Engineering Guidelines

Start simple. Add terms only when there is a clear behavioral problem to fix.

### Proven locomotion rewards
```python
rewards = {
    # Positive — what we WANT
    "gait_pattern":         +10.0,  # MOST IMPORTANT — trot diagonal pairs in sync
    "forward_velocity":     +7.0,   # track commanded vx
    "yaw_tracking":         +5.0,   # track commanded wz
    "air_time":             +5.0,   # foot swing — prevents dragging
    "foot_clearance":       +2.5,   # step high enough for terrain

    # Negative — what we DON'T want
    "base_orientation":     -5.0,   # don't tilt
    "base_lateral_motion":  -3.0,   # no bounce/sway
    "action_smoothness":    -2.0,   # no jerking — MUST BE CLAMPED, unbounded L2 → NaN
    "joint_torque":         -0.002, # energy efficiency
    "joint_pos_deviation":  -1.0,   # stay near default stance
    "foot_slip":            -1.0,
}
```

### Navigation/waypoint rewards (Cole's pattern)
```python
rewards = {
    "waypoint_capture":   +10.0,
    "progress_shaping":   +/- delta_distance,  # only in stages 2+
    "time_penalty":       -0.01,   # per step
    "fall_penalty":       -50.0,   # terminates episode
    "boundary_penalty":   -5.0,
}
```

### Rules
- **`gait_pattern` at +10.0 is non-negotiable** — without it the policy finds degenerate gaits
- **Always cap `action_smoothness`** — raw L2 → infinity → NaN explosion
- Keep total terms ≤ 12. Mason's 11-term config reached terrain 6; our 22-term config plateaued at 4.83
- Change one weight at a time — cascading interactions make multi-term changes undebuggable

---

## 4. Training Curriculum (Proven 4-Phase Pattern)

```
Phase A          →  Phase A.5        →  Phase B-easy      →  Phase B
100% flat           50% flat +           12 types,             12 types,
                    gentle rough         3 difficulty rows      10 difficulty rows
500 iters ~1.7hr   1000 iters ~2hr      30K iters ~21hr        30K iters ~16hr
```

**Do NOT skip phases.** Jumping A.5 → full robust caused action_smoothness to explode to -103 trillion in 15 iterations (Trial 10).

### Critical LR limits
| lr_max | Phase | Result |
|--------|-------|--------|
| 3e-4 | A (flat only) | Fine |
| 3e-4 | B-easy | Instant crash (value_loss 4,670+) |
| 1e-4 | B-easy | Recovers, re-explodes at iter ~1134 |
| **5e-5** | **B-easy and beyond** | **Stable — use this** |

### Phase exit criteria
| Phase | Move on when |
|-------|-------------|
| A | time_out > 95%, flip_over < 2% |
| A.5 | time_out > 90%, flip_over < 5% |
| B-easy | time_out > 80%, flip_over < 15%, terrain_levels climbing |
| B | terrain_level > 4.5, gait_quality > 4.0 |

### Expected Phase A metrics (from Trial 7b, flat terrain)
| Iter | Reward | Ep Length | Flip Over | Time Out |
|------|--------|-----------|-----------|----------|
| 100 | ~140 | ~1150 | ~28% | ~72% |
| 200 | ~375 | 1500 | ~5% | ~95% |
| 500 | ~567 | 1500 | <1% | >99% |

---

## 5. Environment Setup

### Local — Windows, RTX 4070
- **Env:** `isaacSim_env` venv at `MainCap/isaacSim_env/` — never use conda locally
- **Activate:**
  ```bash
  source isaacSim_env/Scripts/activate    # Git Bash
  isaacSim_env\Scripts\activate           # cmd / PowerShell
  ```
- **EULA:** `set OMNI_KIT_ACCEPT_EULA=YES` (cmd) or `$env:OMNI_KIT_ACCEPT_EULA="YES"` (PS)
- **Use for:** environment development, debug runs (10–100 iters), teleop testing
- **num_envs on 4070:** start at 512. Max ~2048 before VRAM pressure.

### H100 — production training, no miniconda
H100 currently has conda (`env_isaaclab`) but we want a clean venv. One-time setup:

```bash
ssh t2user@172.24.254.24      # ONE session only

python3.11 -m venv ~/isaacSim_env
source ~/isaacSim_env/bin/activate

pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
cd ~/IsaacLab && pip install -e .
echo 'export OMNI_KIT_ACCEPT_EULA=YES' >> ~/.bashrc
```

**Every training session:**
```bash
source ~/isaacSim_env/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
screen -S training
python train.py --headless --num_envs 8192 ... 2>&1 | tee run.log
# Detach: Ctrl+A, D  |  Reattach: screen -r training
```

**Monitor training visually — TensorBoard:**
Open `http://172.24.254.24:6006` in a browser while training runs.
Shows reward curves, terrain level, gait quality, value loss, and more in real time.
This is the primary way to know if a run is healthy or needs to be killed early.

---

## 6. Pre-Flight Checklist

### Always run a 10-iteration debug first
```bash
# Local (4070)
python train.py --headless --num_envs 512 --max_iterations 10

# H100
python train.py --headless --num_envs 4096 --max_iterations 10
```

- [ ] No crash on startup
- [ ] Print and verify observation shape — never trust the mental model
- [ ] All reward terms firing at iter 1 (print the reward dict)
- [ ] No NaN in obs or reward
- [ ] Action norms < 5.0
- [ ] GPU VRAM is reasonable (`nvidia-smi`)
- [ ] CRLF fixed before H100 upload: `sed -i "s/\r$//" *.sh`
- [ ] `OMNI_KIT_ACCEPT_EULA=YES` set

---

## 7. Debugging

| Symptom | Cause | Fix |
|---------|-------|-----|
| Robot collapses in < 2s | Height scan filled with 1.0 | Change fill to 0.0 |
| Robot collapses in < 2s | Undertrained checkpoint | Run more iterations |
| `$'\r': command not found` | CRLF in shell script | `sed -i "s/\r$//"` |
| `AttributeError: no attribute 'name'` on Spot | `world.scene.add(spot)` called | Remove that call |
| Robot legs don't move at start | `initialize()` called before physics | Move into first physics callback |
| `UnboundLocalError` in physics callback | Missing `nonlocal` | Add `nonlocal var` in closure |
| Isaac Sim hangs on startup (H100) | Previous zombie holds GPU memory | Physical reboot |
| D-state zombie after run ends | Called `simulation_app.close()` | Use `os._exit(0)` only |
| value_loss explodes to millions | LR too high for phase | Use lr_max=5e-5 for Phase B+ |
| action_smoothness → -infinity / NaN | Unbounded L2 penalty | Cap at 10.0 |
| Terrain plateaus at level ~4.8 | Too many reward terms or network too large | ≤12 terms, [512,256,128] network |
| Gait destroyed mid-training | AI Coach over-boosted velocity reward | Deferred coach mode; velocity bounds 3.0–7.0 |
| value_loss spike ~1000 at iter 1025 | Normal at lr=5e-5 Phase B-easy | Wait — it recovers. Kill only if >5000 and still rising at iter 1050 |

---

## 8. Run Naming & Logging

```
spot_<env>_<phase>_<attempt>
spot_flat_A_01
spot_clutter_phaseA_01
spot_nav_waypoint_01
```

- Log dir: `logs/rsl_rl/<run_name>/<timestamp>/`
- Checkpoints: `model_<iter>.pt` — save every 50–100 iters for long runs (~21MB each)
- Always tee output: `python train.py ... 2>&1 | tee run.log`

---

## 9. Lessons Learned (Update When We Learn Something New)

- **Mason's 11-term config > our 22-term config** — simpler reward = clearer gradient signal
- **Never skip Phase A.5** — direct A → robust terrain causes instant explosion (Trial 10)
- **lr_max = 5e-5 is the safe ceiling for Phase B+** — anything higher re-explodes
- **AI Coach must run in deferred mode** — 300+ silent iterations before first intervention
- **MH-1 failure:** coach positive-feedback-looped velocity reward 5→14.26 → gait destroyed. Fix: velocity bounds 3–7, disable LR changes from coach entirely
- **RTX 4070:** fine for env dev and smoke tests; not enough VRAM for production scale
- **H100 SSH:** one session only — parallel sessions cause unresponsive server, physical reboot required
