# Rough Terrain Policy Deployment Debug Handoff

**Date**: February 11, 2026  
**Environment**: Isaac Sim 5.1.0 + Isaac Lab 2.3.0  
**Python**: `C:\miniconda3\envs\isaaclab311\python.exe`

---

## 1. THE PROBLEM

We trained a rough terrain locomotion policy for Boston Dynamics Spot using Isaac Lab's RSL-RL framework. The policy **works perfectly** when tested with Isaac Lab's `play.py` script — the robot walks, handles rough ground, and is stable.

However, when we deploy the same policy in our custom obstacle course simulation (using Isaac Sim's standalone Python API), the robot **falls over within 1.5-2 seconds**, even when standing still with zero velocity command.

**Goal**: Deploy the trained rough terrain policy in `spot_obstacle_course.py` with Xbox controller teleop, switchable between flat (NVIDIA's pre-trained) and rough (our trained) gaits.

---

## 2. KEY FILES

### Training
- **Checkpoint**: `C:\IsaacLab\logs\rsl_rl\spot_rough\2026-02-09_15-18-50\model_4999.pt`
- **JIT Export**: `C:\IsaacLab\logs\rsl_rl\spot_rough\2026-02-09_15-18-50\exported\policy.pt`
- **Training Config**: `C:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\spot\rough_env_cfg.py`

### Deployment
- **Policy Wrapper**: `MS_for_autonomy\experimental_design_grass\code\spot_rough_terrain_policy.py`
- **Obstacle Course**: `MS_for_autonomy\experimental_design_grass\code\spot_obstacle_course.py`
- **Standalone Test**: `MS_for_autonomy\experimental_design_grass\code\test_rough_standalone.py`

---

## 3. VERIFIED WORKING

### ✅ Isaac Lab play.py
```powershell
cd C:\IsaacLab
python source\isaaclab\isaaclab\app\runners\play.py --task Isaac-Velocity-Rough-Spot-v0 --num_envs 1 --checkpoint "C:\IsaacLab\logs\rsl_rl\spot_rough\2026-02-09_15-18-50\model_4999.pt"
```
**Result**: Robot walks perfectly, handles rough terrain, stable gait.

### ✅ Model Loading Verified
We compared our manual MLP reconstruction against the JIT-exported model:
```python
# Test outputs
manual_output = actor(test_input)      # Our nn.Sequential MLP
jit_output = jit_model(test_input)     # torch.jit.load(policy.pt)
max_diff = (manual_output - jit_output).abs().max()  # Result: 0.0
```
**Conclusion**: Model weights are loaded correctly. Outputs are IDENTICAL.

### ✅ Quaternion Convention
Checked gravity projection in body frame:
```
gravity_b = R_BI @ [0, 0, -1]  →  [0.014, 0.012, -0.999]  (robot upright)
```
**Conclusion**: Quaternion convention is correct.

### ✅ Height Scan = 1.0 is Correct
Traced through Isaac Lab source:
```python
# observations.py line 292-300
height_scan = sensor.data.pos_w[:, 2] - sensor.data.ray_hits_w[..., 2] - 0.5
# where sensor.data.pos_w[:, 2] = body_z + 20.0 (RayCaster 20m Z-offset)
# Result: (body_z + 20) - ground_z - 0.5 ≈ 20.0, clipped to 1.0
```
**Conclusion**: Training ALWAYS had height_scan = 1.0. Our constant 1.0 is correct.

### ✅ DOF Ordering
```python
['fl_hx', 'fr_hx', 'hl_hx', 'hr_hx', 'fl_hy', 'fr_hy', 'hl_hy', 'hr_hy', 'fl_kn', 'fr_kn', 'hl_kn', 'hr_kn']
```
Grouped by joint type, not by leg. Verified consistent between training and deployment.

### ✅ Default Joint Positions Match
Training config default positions:
```python
TRAINING_DEFAULTS = {
    "fl_hx":  0.1, "fr_hx": -0.1, "hl_hx":  0.1, "hr_hx": -0.1,
    "fl_hy":  0.9, "fr_hy":  0.9, "hl_hy":  1.1, "hr_hy":  1.1,
    "fl_kn": -1.5, "fr_kn": -1.5, "hl_kn": -1.5, "hr_kn": -1.5,
}
```

---

## 4. TRAINING CONFIGURATION

### Policy Architecture
```
Input:  235 dims (48 proprioceptive + 187 height scan)
Hidden: 512 → 256 → 128 (ELU activations)
Output: 12 dims (joint position offsets)
```

### Key Parameters
```python
ACTION_SCALE = 0.25           # action * scale + default_pos = target
DECIMATION = 10               # Policy at 50 Hz, physics at 500 Hz
PHYSICS_DT = 1.0 / 500.0
```

### Observation Layout
```python
obs[0:3]    = lin_vel_b       # Linear velocity in body frame
obs[3:6]    = ang_vel_b       # Angular velocity in body frame  
obs[6:9]    = projected_gravity  # Gravity in body frame
obs[9:12]   = velocity_command   # [vx, vy, omega_z]
obs[12:24]  = joint_pos - default_pos  # Relative joint positions
obs[24:36]  = joint_vel       # Joint velocities
obs[36:48]  = last_action     # Previous policy output (RAW, not smoothed)
obs[48:235] = height_scan     # Always 1.0 (see above)
```

### Actuator Model (CRITICAL)
Training uses **explicit actuator models** that compute PD torques in Python:

**Hips** (8 DOFs: *_hx, *_hy): `DelayedPDActuatorCfg`
- Kp = 60.0, Kd = 1.5
- effort_limit = 45.0 N·m
- Action delay: 0-4 physics steps

**Knees** (4 DOFs: *_kn): `RemotizedPDActuatorCfg`  
- Kp = 60.0, Kd = 1.5
- Angle-dependent effort limit (30.6 – 113.2 N·m)
- Uses 103-row lookup table in `spot.py`

### Isaac Lab PhysX Settings (from articulation.py lines 1766-1782)
For explicit actuator models, Isaac Lab sets:
```python
write_joint_stiffness_to_sim(0.0)      # Disable PhysX PD
write_joint_damping_to_sim(0.0)        # Disable PhysX PD
write_joint_effort_limit_to_sim(1e9)   # Don't interfere
write_joint_velocity_limit_to_sim(12.0)
write_joint_armature_to_sim(0.0)
write_joint_friction_coefficient_to_sim(0.0)
```
Then computes PD torques in Python and applies via `set_dof_actuation_forces()`.

---

## 5. WHAT WE TRIED (ALL FAILED)

### Attempt 1: PhysX Position Drive with Training Gains
```python
av.set_gains(kps=np.full((1, 12), 60.0), kds=np.full((1, 12), 1.5))
robot.set_joint_positions(target_positions)
```
**Result**: Falls within 2 seconds. Gains too weak for PhysX position drive.

### Attempt 2: Reduced ACTION_SCALE
Changed from 0.25 to 0.15 to reduce position target magnitude.
**Result**: Still falls.

### Attempt 3: EMA Action Smoothing
```python
alpha = 0.7
smoothed_action = alpha * new_action + (1 - alpha) * prev_action
```
**Result**: Still falls.

### Attempt 4: Manual PD Torque (First Try)
```python
av.set_gains(kps=np.zeros((1, 12)), kds=np.zeros((1, 12)))
torques = Kp * (target - current) - Kd * velocity
robot.set_joint_efforts(torques)
```
**Result**: Still falls.

### Attempt 5: Target Smoothing at 500 Hz
Applied EMA smoothing to position targets (not actions):
```python
alpha = 0.15  # Very smooth
smoothed_target = alpha * target + (1 - alpha) * prev_target
```
With hip position error clamping (max 0.75 rad).
**Result**: Still falls. Analysis showed error clamping never activated.

### Attempt 6: Full PhysX Property Match + Manual PD
Replicated ALL Isaac Lab settings:
```python
av.set_gains(kps=np.zeros((1, n_dof)), kds=np.zeros((1, n_dof)))
av.set_friction_coefficients(np.zeros((1, n_dof)))
av.set_armatures(np.zeros((1, n_dof)))
av.set_max_efforts(np.full((1, n_dof), 1e9))
av.set_max_joint_velocities(np.full((1, n_dof), 12.0))

# Then compute PD torques manually
torques = KP * (target - current) - KD * velocity
# Clamp hips at 45 N·m, knees at angle-dependent limit
robot.set_joint_efforts(torques)
```
**Result**: STILL FALLS. Robot rolls within 1.5-2 seconds.

---

## 6. VERIFIED THE SAME API

Traced through Isaac Sim source:
```python
# isaacsim/core/prims/impl/articulation.py lines 1142-1213
def set_joint_efforts(self, efforts, ...):
    ...
    self._physics_view.set_dof_actuation_forces(new_dof_efforts, indices)
```
This is the **same underlying API** that Isaac Lab uses.

---

## 7. STANDALONE TEST RESULTS

Created `test_rough_standalone.py` to isolate gait-switching from policy deployment.

### Test A: Manual PD hold default pose (no policy)
- PhysX Kp=0, Kd=0
- Python computes PD torques: `Kp*(default-current) - Kd*vel`
- **Result**: Robot stands stably for 5 seconds ✅

### Test B: Policy + Manual PD torques
- Same PhysX settings as Test A
- Policy outputs position targets
- **Result**: Falls within ~0.5-1 second ❌

### Test C: Policy + PhysX Position Drive (USD default gains)
- Uses SpotFlatTerrainPolicy's default USD gains (much higher)
- Policy outputs position targets via `set_joint_positions()`
- **Result**: Falls within ~0.5-1 second ❌

**Conclusion**: The POLICY outputs are causing instability, not the torque mode.

---

## 8. REMAINING HYPOTHESES

### Hypothesis 1: Simulation Pipeline Difference
Isaac Lab's `play.py` might use GPU pipeline while our standalone uses CPU.
PhysX behavior can differ between pipelines.

### Hypothesis 2: Solver Iterations
Isaac Lab may use different PhysX solver iteration counts (position/velocity).
This affects constraint solving accuracy.

### Hypothesis 3: Stepping Order
Isaac Lab's decimation loop:
```python
for _ in range(decimation):
    apply_action()           # Same action all 10 substeps
    write_data_to_sim()      # Recompute PD with FRESH joint data
    sim.step()
    scene.update()           # Invalidate buffers for next iteration
```
Our deployment might have subtle timing differences.

### Hypothesis 4: Initial State Sensitivity
The policy might be extremely sensitive to initial observations.
Even small differences in starting state could cause divergence.

### Hypothesis 5: Missing Domain Randomization
Training used domain randomization (mass, friction, etc.).
Deployment uses nominal values. Policy might rely on averaged dynamics.

---

## 9. NEXT STEPS TO TRY

1. **Run play.py with verbose logging** to capture exact observations and actions, then compare against our deployment frame-by-frame.

2. **Check PhysX pipeline**: Verify whether play.py uses GPU or CPU pipeline.

3. **Match solver settings**: Find Isaac Lab's PhysX solver config and replicate in deployment.

4. **Disable action delay**: The DelayedPD actuator has 0-4 step delay. We don't implement this — try adding it.

5. **Print first 10 observations side-by-side** from play.py vs deployment to find any discrepancies.

6. **Alternative approach**: Modify Isaac Lab's play.py to add teleop controls instead of trying to replicate its physics in standalone.

---

## 10. CODE STATE

### spot_rough_terrain_policy.py (~477 lines)
Current implementation:
- Manual PD torque computation at 500 Hz
- All PhysX properties matched to training (Kp=0, Kd=0, friction=0, etc.)
- Hip effort limit: 45 N·m
- Knee effort limit: angle-dependent lookup table
- `_previous_action` = RAW policy output (no smoothing)
- Height scan = constant 1.0

### spot_obstacle_course.py (~1530 lines)
- Saves all PhysX properties from flat policy at startup
- Restores them when switching back from rough to flat
- Calls `spot_rough.apply_gains()` when switching to rough
- Calls `spot_rough.post_reset()` on gait transition

### test_rough_standalone.py (~320 lines)
- Three-phase test: A) manual PD only, B) policy+manual PD, C) policy+position drive
- Full robot reset between tests (body pose + joints + velocities)
- Result: A passes, B and C fail

---

## 11. REFERENCE: Isaac Lab Decimation Loop

From `manager_based_rl_env.py` lines 182-197:
```python
for _ in range(self.cfg.decimation):
    self._sim_step_counter += 1
    self.action_manager.apply_action()      # Same action all substeps
    self.scene.write_data_to_sim()          # PD computed with FRESH data
    self.sim.step(render=False)
    self.scene.update(dt=self.physics_dt)   # Invalidates buffers
```

The key insight: `scene.update()` is INSIDE the loop, which invalidates `TimestampedBuffer`s, causing the next `write_data_to_sim()` to read fresh `joint_pos`/`joint_vel` from PhysX. So PD torques ARE recomputed at every 500 Hz substep (same as our deployment).

---

## 12. HOW TO VERIFY POLICY WORKS

```powershell
cd C:\IsaacLab
python source\isaaclab\isaaclab\app\runners\play.py --task Isaac-Velocity-Rough-Spot-v0 --num_envs 1 --checkpoint "C:\IsaacLab\logs\rsl_rl\spot_rough\2026-02-09_15-18-50\model_4999.pt"
```
Robot runs around, handles slopes and rough terrain, never falls.

---

## 13. HOW TO RUN OBSTACLE COURSE (to see the bug)

```powershell
cd "C:\Users\Gabriel Santiago\OneDrive\Desktop\Nvidia Omniverse\AI2C_Tech_Capstone_MS_for_Autonomy\MS_for_autonomy\experimental_design_grass\code"
C:\miniconda3\envs\isaaclab311\python.exe spot_obstacle_course.py
```
- Press `G` to cycle gaits (FLAT → ROUGH)
- In ROUGH gait, robot falls within 1.5-2 seconds even standing still

---

## 14. HOW TO RUN STANDALONE TEST

```powershell
cd "C:\Users\Gabriel Santiago\OneDrive\Desktop\Nvidia Omniverse\AI2C_Tech_Capstone_MS_for_Autonomy\MS_for_autonomy\experimental_design_grass\code"
C:\miniconda3\envs\isaaclab311\python.exe test_rough_standalone.py
```
Runs three tests A/B/C automatically with results summary.

---

## SUMMARY

| What | Status |
|------|--------|
| Policy works in Isaac Lab play.py | ✅ |
| Model weights loaded correctly | ✅ |
| Observations match training layout | ✅ |
| Height scan = 1.0 is correct | ✅ |
| DOF ordering correct | ✅ |
| Quaternion convention correct | ✅ |
| Manual PD holds default pose | ✅ |
| Policy + any control mode (standalone) | ❌ Falls |
| **Policy + Isaac Lab teleop (GPU PhysX)** | **✅ SURVIVED 10s+** |

---

## 15. ROOT CAUSE FOUND (Feb 12, 2026)

**The root cause is GPU vs CPU PhysX pipeline mismatch.**

### Investigation Summary

1. **Solver iterations are NOT the issue** — The Spot USD already has
   `solver_position_iteration_count=4, solver_velocity_iteration_count=0`,
   matching SPOT_CFG. Verified by reading the values at runtime.

2. **Isaac Lab trains on GPU physics** (`device="cuda:0"`, GPU broadphase,
   fabric enabled). Standalone `World()` defaults to **CPU physics** (MBP
   broadphase, no fabric).

3. GPU and CPU PhysX implementations produce subtly different dynamics for
   articulated bodies. The policy learned actions that work with GPU PhysX's
   specific constraint solver behavior. In CPU PhysX, the same actions produce
   different joint responses, causing the feedback loop to diverge.

4. **Proof**: Running the SAME policy inside Isaac Lab's `ManagerBasedRLEnv`
   (GPU PhysX) with teleop velocity commands → robot walks stably for 10s+.
   Running in standalone `World()` (CPU PhysX) → falls within 1.6s.

### The Fix: `play_rough_teleop.py`

Instead of trying to match GPU PhysX behavior in standalone, we run the policy
inside Isaac Lab's environment framework where it already works.

**File**: `MS_for_autonomy/experimental_design_grass/code/play_rough_teleop.py`

```powershell
cd C:\IsaacLab
python "PATH\play_rough_teleop.py" --num_envs 1
```

Controls: W/S (forward/back), A/D (strafe), Q/E (turn), SPACE (stop)

Features:
- Uses Isaac Lab's ManagerBasedRLEnv with SpotRoughEnvCfg_PLAY
- GPU PhysX pipeline (exact same solver as training)
- Keyboard teleop overrides velocity commands
- Headless mode with constant forward walk for testing
- Real-time status display (body height, gravity vector)

### Why Standalone Deployment Fails

The standalone `isaacsim.core.api.World()` class uses CPU PhysX by default.
Even though we matched:
- Solver iterations (4/0) ✅
- Self-collisions (True) ✅
- Joint properties (Kp=0, Kd=0, friction=0, armature=0) ✅
- Manual PD torques (Kp=60, Kd=1.5) ✅
- Effort clamping (hips 45Nm, knees angle-dependent) ✅

The CPU PhysX solver produces different constraint resolution than GPU PhysX.
The policy's learned actions are tuned to GPU dynamics and diverge in CPU.

---

## 16. FINAL ROOT CAUSE FOUND (Feb 16, 2026)

### The REAL Bug: Height Scan Fill Value

**Section 5 ("Height Scan = 1.0 is Correct") was WRONG.**

Running the actual Isaac Lab training environment and printing raw observations proved it:
```
height_scan range: [-0.000002, 0.148083]
height_scan mean:  0.003959
```

The height scan is approximately **0.0** on flat ground, NOT 1.0. Our source code tracing
was incorrect — the RayCaster's `combine_frame_transforms` resolves ray hits differently
than what the manual trace predicted.

### Impact of the Bug

A parameter sweep of the trained actor showed extreme sensitivity:
```
height_scan = 0.0  →  action norm = 3.08  (normal walking)
height_scan = 0.2  →  action norm = 2.37  (optimal)
height_scan = 1.0  →  action norm = 7.42  (CATASTROPHIC)
```

With hs=1.0, the policy produced 2.4x larger action commands, immediately destabilizing
the robot. This was the PRIMARY cause of deployment failure — not GPU vs CPU PhysX
(Section 15 was a contributing factor but not the main issue).

### The Fix

In `spot_rough_terrain_policy.py`:
```python
# BEFORE (caused falls):
SCAN_CLIP_HI = 1.0
obs[48:235] = SCAN_CLIP_HI

# AFTER (robot walks):
SCAN_FILL_VAL = 0.0
obs[48:235] = SCAN_FILL_VAL
```

### Combined Fix: GPU PhysX + Correct Height Scan

The obstacle course now uses BOTH fixes:
1. **GPU PhysX** (`backend="torch"`, `device="cuda:0"`) — matches training physics
2. **Height scan = 0.0** — matches actual training observations
3. **NumpyRobotWrapper** — handles CUDA tensor ↔ numpy conversions for GPU PhysX

### Result

- Action norms: 7.20 → 3.08 (57% reduction)
- Robot behavior: instant fall → **stable walking**
- Obstacle course: fully functional with FLAT ↔ ROUGH gait switching
- Controls: WASD keyboard + Xbox controller both work in ROUGH gait

### Meta-Lesson

**Never trust source code tracing alone.** The 2-minute diagnostic of printing actual
observation values from the training environment would have caught this immediately.
Instead, we spent days debugging GPU/CPU PhysX, solver iterations, tensor types,
and domain randomization — none of which were the actual problem.

---

## UPDATED STATUS TABLE

| What | Status |
|------|--------|
| Policy works in Isaac Lab play.py | ✅ |
| Model weights loaded correctly | ✅ |
| Observations match training layout | ✅ |
| ~~Height scan = 1.0 is correct~~ | ❌ **WRONG — should be 0.0** |
| Height scan = 0.0 (flat ground) | ✅ FIXED |
| DOF ordering correct | ✅ |
| Quaternion convention correct | ✅ |
| Manual PD holds default pose | ✅ |
| GPU PhysX in standalone | ✅ FIXED |
| **Policy + obstacle course (GPU PhysX + hs=0.0)** | **✅ WORKS** |
| Dual gait switching (FLAT ↔ ROUGH) | ✅ |
| Xbox controller in ROUGH gait | ✅ |
