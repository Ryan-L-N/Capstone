# 48-Hour Training — Completion & Test Plan

**Date:** February 16, 2026
**Status:** TRAINING COMPLETE (30,000 / 30,000 iterations)
**Server:** H100 NVL (ai2ct2) — 172.24.254.24

---

## 1. Training Summary

### Final Metrics (Iteration 29,996)

| Metric | Start (Iter 0) | Final (Iter 29,996) | Change |
|--------|---------------:|--------------------:|--------|
| Mean Reward | -0.90 | +143.74 | +144.64 |
| Episode Length | 20 steps (0.4s) | 573 steps (11.5s) | 28.6x |
| Gait Reward | 0.06 | 5.28 | 88x |
| Terrain Level | 3.18 | 4.42 | +1.24 |
| Body Contact (falls) | 22% | 57.5% | — |
| Timeout (survived) | 0.9% | 42.2% | 47x |
| Base Lin Vel Tracking | 0.02 | 4.39 | 220x |
| Foot Clearance | 0.01 | 0.79 | stair-ready |
| Action Noise Std | 0.80 | 0.65 | converged |

### Full Progress Matrix

| Metric | Iter 0 | 2,476 | 8,561 | 14,616 | 21,421 | 24,611 | 27,781 | 29,996 |
|-----:|---:|---:|---:|---:|---:|---:|---:|---:|
| % Complete | 0% | 8% | 29% | 49% | 71% | 82% | 93% | 100% |
| Mean Reward | -0.90 | +117 | +137 | **+173** | +152 | +165 | +156 | +144 |
| Ep Length | 20 | 572 | 611 | 708 | 619 | 669 | 625 | 573 |
| Terrain Lvl | 3.18 | 2.74 | 3.72 | 4.41 | 4.39 | 4.41 | **4.45** | 4.42 |
| Gait | 0.06 | 4.28 | 4.33 | 5.32 | 5.46 | **5.56** | 5.36 | 5.28 |
| Body Contact | 22% | 66% | 60% | 60% | 60% | 59% | **56%** | 57% |
| Timeout | 0.9% | 34% | 39% | 39% | 40% | 40% | **44%** | 42% |

### Training Phases Observed

- **Phase 1 (Iter 0–10k):** Learned to stand and walk. Reward jumped from -0.90 to +137.
- **Phase 2 (Iter 10k–20k):** Rough terrain mastery. Peak reward +173 at iter 14,616.
- **Phase 3 (Iter 20k–30k):** Robustness & efficiency. Reward oscillated +135–165 as curriculum pushed harder terrain. Body contact hit all-time low of 56% at iter 27,781.

### Checkpoints Available

60 checkpoints saved at 500-iteration intervals on the H100:
- Path: `~/IsaacLab/logs/rsl_rl/spot_rough/2026-02-13_16-26-37_48h_proprioception/`
- Files: `model_0.pt` through `model_29999.pt`
- Key checkpoints of interest:
  - `model_14500.pt` — near peak reward (+173)
  - `model_27500.pt` — best survival rate (56% body contact, 44% timeout)
  - `model_29999.pt` — final model

---

## 2. Download Plan

### Step 1: Download Final Checkpoint

```bash
# Create local directory
mkdir -p C:/IsaacLab/logs/rsl_rl/spot_rough/48h_run/

# Download final model
scp t2user@172.24.254.24:~/IsaacLab/logs/rsl_rl/spot_rough/2026-02-13_16-26-37_48h_proprioception/model_29999.pt \
    C:/IsaacLab/logs/rsl_rl/spot_rough/48h_run/model_29999.pt

# Also download peak reward model for comparison
scp t2user@172.24.254.24:~/IsaacLab/logs/rsl_rl/spot_rough/2026-02-13_16-26-37_48h_proprioception/model_14500.pt \
    C:/IsaacLab/logs/rsl_rl/spot_rough/48h_run/model_14500.pt

# Download best survival model
scp t2user@172.24.254.24:~/IsaacLab/logs/rsl_rl/spot_rough/2026-02-13_16-26-37_48h_proprioception/model_27500.pt \
    C:/IsaacLab/logs/rsl_rl/spot_rough/48h_run/model_27500.pt
```

### Step 2: Download Training Params (for reference)

```bash
scp -r t2user@172.24.254.24:~/IsaacLab/logs/rsl_rl/spot_rough/2026-02-13_16-26-37_48h_proprioception/params/ \
    C:/IsaacLab/logs/rsl_rl/spot_rough/48h_run/params/
```

### Step 3: Download TensorBoard Events (optional)

```bash
scp t2user@172.24.254.24:~/IsaacLab/logs/rsl_rl/spot_rough/2026-02-13_16-26-37_48h_proprioception/events.out.* \
    C:/IsaacLab/logs/rsl_rl/spot_rough/48h_run/
```

---

## 3. Test Plan

### Test 1: Isaac Lab play.py (Guaranteed Baseline)

This should work out of the box — same framework that trained the policy:

```bash
cd C:\IsaacLab
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task=Isaac-Velocity-Rough-Spot-Play-v0 \
    --num_envs 50 \
    --load_run 48h_run \
    --load_checkpoint model_29999.pt
```

**Expected:** Spot walks on rough terrain, follows velocity commands, survives on terrain levels 0–4.

### Test 2: Compare Checkpoints

Test the three downloaded checkpoints side by side:

| Checkpoint | Expected Behavior |
|------------|------------------|
| `model_14500.pt` | Peak reward — smoothest walking on medium terrain |
| `model_27500.pt` | Best survival — most robust on hard terrain |
| `model_29999.pt` | Final — balanced performance |

### Test 3: Standalone Deployment (The Real Test)

Deploy in our standalone obstacle course / teleop environment. This requires:

1. **Observation vector:** Must be exactly 235 dimensions in the correct order:
   - `[base_lin_vel(3), base_ang_vel(3), projected_gravity(3), velocity_commands(3), joint_pos_rel(12), joint_vel_rel(12), last_action(12), height_scan(187)]`

2. **Physics match:**
   - `sim.dt = 0.002` (500 Hz physics)
   - `decimation = 10` (50 Hz control)
   - GPU PhysX: 4 position / 0 velocity solver iterations
   - PD gains: Kp=60, Kd=1.5

3. **Action processing:**
   - Policy output (12-dim) scaled by 0.25
   - Added to default joint positions
   - Sent as joint position targets

4. **No observation normalization** — `actor_obs_normalization=False`

5. **Height scan:** RayCaster with GridPattern(0.1, [1.6, 1.0]) = 17x11 = 187 points
   - Mounted on body prim with 20.0m Z-offset
   - Clipped to [-1.0, 1.0]
   - If height scan is broken (all 1.0), try flat ground first

---

## 4. Deployment Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Height scan returns all 1.0 | Medium | High — blind to terrain | Test on flat ground first; verify RayCaster Z-offset |
| Physics solver mismatch | Low | Medium — wobbly joints | Explicitly set GPU PhysX with 4/0 solver iters |
| Joint ordering wrong | Low | Critical — instant crash | Verify against SPOT_CFG URDF joint order |
| Action scale wrong | Low | Medium — over/under shoots | Hardcode 0.25 to match training |
| Quaternion convention | Low | Medium — wrong orientation | Isaac Sim = [w,x,y,z] scalar-first |

---

## 5. Success Criteria

| Test | Pass Condition |
|------|---------------|
| Isaac Lab play.py | Spot walks and follows commands on rough terrain |
| Flat ground standalone | Spot walks forward without falling for 10+ seconds |
| Rough terrain standalone | Spot navigates stairs/obstacles with <50% fall rate |
| Teleop control | WASD/Xbox commands produce responsive movement |
| FPV camera | First-person view works while policy is active |

---

## 6. Fallback Plan

If the policy doesn't walk in standalone deployment:

1. **Verify in Isaac Lab play.py first** — if it works there, the bug is in our wrapper
2. **Test with flat terrain only** — remove height scan from obs, fill with zeros
3. **Try model_14500.pt** — peak reward checkpoint, may be more stable
4. **Check observation order** — print obs vector shapes and compare to training
5. **Log raw actions** — compare policy output magnitude to training logs
