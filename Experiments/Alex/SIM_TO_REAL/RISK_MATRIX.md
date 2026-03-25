# Sim-to-Real Risk Matrix

Third-party evaluation of the Capstone quadruped locomotion RL codebase, performed
March 23 2026 prior to CMU PhD review meeting. This document captures every identified
risk, its severity, and the specific mitigation built into the SIM_TO_REAL pipeline.

---

## Risk Register

| ID | Risk | Severity | Likelihood | Impact | Mitigation | Status |
|----|------|----------|------------|--------|------------|--------|
| R1 | **No actuator latency simulation** — Policy trained with 0 ms delay; real Spot has 40-60 ms | CRITICAL | High | Policy oscillation, gait instability on hardware | `ActionDelayWrapper` (2 steps = 40 ms) + `ObservationDelayWrapper` (1 step = 20 ms) active in ALL expert training | MITIGATED |
| R2 | **No real-robot SDK integration** — No Boston Dynamics Spot SDK wrapper exists | CRITICAL | Certain | Cannot deploy without it | `deploy/spot_sdk_wrapper.py` — observation builder + action executor + 20 Hz control loop | MITIGATED |
| R3 | **Control frequency mismatch** — Trained at 50 Hz, Spot SDK runs at ~20 Hz | HIGH | High | Over-responsive policy, potential resonance | Distilled student trains at 20 Hz (decimation = 25); experts train at 50 Hz for learning efficiency | MITIGATED |
| R4 | **No temporal modeling (MLP only)** — No LSTM/GRU, policy can't do online system ID | HIGH | Medium | Can't adapt to mass/friction shifts on real robot | Accepted — MLP chosen for deployment simplicity (<10 ms inference). Mitigated by aggressive domain randomization (mass ±5 kg, friction 0.15-1.3) | ACCEPTED |
| R5 | **Height scan is idealized raycasting** — Real LiDAR/depth has noise, occlusions, dropouts | HIGH | High | Terrain perception failure on real sensors | Observation noise increased (±0.2 m Gaussian), `SensorNoiseWrapper` adds 5 % ray dropout + OU-process IMU drift | MITIGATED |
| R6 | **No motor torque limits during training** — Only enforced in eval wrappers | MEDIUM | Medium | Policy commands unachievable torques | `torque_limit_penalty` (hip 45 Nm, knee 100 Nm) + increased `joint_torques` weight (-1e-3) in ALL expert training | MITIGATED |
| R7 | **No energy/power reward** — No penalty for motor power consumption | MEDIUM | Medium | Wasteful gaits that drain battery fast | `motor_power_penalty` (weight -0.005) = `sum(|torque * vel|)` in ALL expert training | MITIGATED |
| R8 | **External push forces disabled** — `force_range=(0.0, 0.0)` in Mason baseline | MEDIUM | Medium | Reduced disturbance rejection on real terrain | External forces ±3.0 N, mass DR ±5.0 kg, friction 0.15-1.3, pushes every 7-12 s in ALL expert training | MITIGATED |
| R9 | **Observation normalization OFF** — Raw 235-dim obs, no standardization | LOW | Low | Gradient imbalance (mitigated by ELU + careful tuning) | Accepted — Mason baseline trains stably without normalization; ELU handles scale variation | ACCEPTED |
| R10 | **No sensor dropout simulation** — Never sees NaN/missing height scan data | MEDIUM | Medium | Policy brittle to real sensor failures | `SensorNoiseWrapper` randomly zeros 5 % of 187 height scan rays each step | MITIGATED |

---

## Detailed Risk Analysis

### R1: No Actuator Latency Simulation (CRITICAL)

**What happens without mitigation:**
The policy learns to respond instantaneously to observations. On real hardware,
there is 40-60 ms of delay between the policy computing an action and the motors
executing it (network communication + servo loop). This causes:
- High-frequency oscillation as the policy overreacts to stale observations
- Gait resonance that amplifies small perturbations into falls
- Inability to maintain balance during dynamic maneuvers

**Our mitigation:**
- `wrappers/action_delay.py`: GPU ring buffer that holds actions for 2 control
  steps before applying them. At 50 Hz this is 40 ms — matching real Spot latency.
- `wrappers/observation_delay.py`: Returns observations from 1 step ago (20 ms),
  simulating IMU/encoder communication delay.
- Both wrappers active from training step 0 in ALL 6 expert trainings.
- During distillation at 20 Hz, delay is maintained (2 steps = 100 ms at 20 Hz, reduced
  to 1 step = 50 ms to stay within real-world bounds).

**Validation:** Run `eval_student.py` with and without delay wrappers; performance
should degrade <20 % with wrappers active.

---

### R2: No Real-Robot SDK Integration (CRITICAL)

**What happens without mitigation:**
The trained policy exists only as a PyTorch checkpoint. There is no code to:
- Read Spot's joint states and IMU data
- Convert policy actions into Spot SDK `JointCommand` messages
- Handle the Spot SDK connection lifecycle (lease, E-stop, power)

**Our mitigation:**
- `deploy/spot_sdk_wrapper.py`: Full bridge between policy I/O and Spot SDK
  - Observation builder: Maps Spot state → 235-dim tensor
  - Action executor: Maps 12-dim action → Spot JointCommand (scale 0.2, Kp=60, Kd=1.5)
  - 20 Hz control loop matching distillation training rate
- `deploy/safety_layer.py`: Hardware watchdogs (torque, orientation, velocity, timeout)
- `deploy/height_scan_builder.py`: Depth camera → 187-dim elevation grid
- `deploy/export_onnx.py`: ONNX export for edge deployment

**Validation:** Unit test with mocked Spot SDK; verify 235-dim observation matches training format.

---

### R3: Control Frequency Mismatch (HIGH)

**What happens without mitigation:**
Policy trained at 50 Hz outputs an action every 20 ms. Real Spot SDK accepts commands
at ~20 Hz (every 50 ms). Running a 50 Hz policy at 20 Hz causes:
- Policy receives fewer observations per gait cycle
- Action timing misaligned with physical dynamics
- Learned gait frequencies may not transfer

**Our mitigation:**
- 6 expert trainings run at 50 Hz for maximum learning signal quality
- Distilled student trains at 20 Hz (decimation = 25) on all terrains
- The student inherits terrain expertise from experts but learns its own 20 Hz timing
- Deployment runs at 20 Hz matching the student's training rate

**Validation:** `eval_student.py` runs at 20 Hz (decimation = 25); must pass all terrains.

---

### R4: No Temporal Modeling (HIGH — ACCEPTED)

**What happens without mitigation:**
MLP policy processes only the current observation + last action. It cannot:
- Estimate latent parameters (current friction, actual mass)
- Track sensor drift over time
- Compensate for time-varying delays

**Why we accept this risk:**
- MLP inference is <10 ms (vs ~15-20 ms for LSTM) — critical for 20 Hz control
- All existing eval infrastructure is MLP-compatible
- Aggressive domain randomization (mass ±5 kg, friction 0.15-1.3, sensor noise)
  forces the policy to be robust without explicit system ID
- Adding LSTM would require fundamental architecture change and full retraining
- Industry precedent: ETH Zurich's ANYmal deployed MLP policies successfully

**Future option:** If real-world performance is insufficient, add 3-frame observation
history stack (+96 dims proprioceptive) as a lightweight temporal feature.

---

### R5: Idealized Height Scan (HIGH)

**What happens without mitigation:**
Training uses PhysX raycasting — perfect ray intersection with no noise, no occlusion,
no sensor-specific artifacts. Real depth cameras/LiDAR have:
- Gaussian noise (±5-20 cm depending on range and surface)
- Missing data from reflective/transparent surfaces
- Temporal latency (frame acquisition + processing)
- Occlusion from robot body and legs

**Our mitigation:**
- Observation noise increased from Mason's ±0.1 to ±0.2 m on height scan
- `SensorNoiseWrapper` adds:
  - 5 % ray dropout (randomly zeros individual rays each step)
  - Ornstein-Uhlenbeck drift on IMU channels (correlated temporal noise)
- `enable_corruption = True` (Mason had False)
- `deploy/height_scan_builder.py` handles real sensor → grid conversion with
  0.0 fill for missing data (matching training convention)

**Validation:** Run eval with 10 % dropout + ±0.3 m noise; verify graceful degradation.

---

### R6: No Motor Torque Limits During Training (MEDIUM)

**What happens without mitigation:**
PhysX applies large effort limits (1e9 Nm) allowing the policy to learn gaits that
require torques exceeding real motor capacity:
- Spot hip motors: ~45 Nm max
- Spot knee motors: 30-100 Nm (angle-dependent, RemotizedPD)
The policy may command high-torque maneuvers that stall real motors.

**Our mitigation:**
- `rewards/torque_limit.py`: Soft penalty `ReLU(|torque| - limit)` per joint
  - Weight -0.3 in all expert training
  - Hip limit: 45 Nm, Knee limit: 100 Nm (conservative)
- `rewards/motor_power.py`: `sum(|torque * vel|)` penalty (weight -0.005)
- `joint_torques` weight increased from -5e-4 to -1e-3
- `deploy/safety_layer.py`: Hard clamp at 120 % of rated limits as last resort

**Validation:** Log joint torques during eval; 95 % must be within Spot limits.

---

### R7: No Energy/Power Reward (MEDIUM)

**What happens without mitigation:**
Policy optimizes for gait quality and velocity tracking with no incentive to minimize
energy use. Real Spot has ~9 Wh battery for ~1 hour runtime at ~100-200 W continuous.
Wasteful gaits (spinning joints, high-frequency oscillation) drain battery rapidly.

**Our mitigation:**
- `rewards/motor_power.py`: Penalizes mechanical power `sum(|torque_i * vel_i|)`
- Weight -0.005 in all expert training (light enough to not compromise terrain performance)
- Clamped to [0, 500] for gradient safety

**Validation:** Compare joint power integral (J) between S2R student and baseline policy.

---

### R8: External Push Forces Disabled (MEDIUM)

**What happens without mitigation:**
Mason baseline has `force_range=(0.0, 0.0)` — no external disturbances during training.
Real robots encounter:
- Wind gusts
- Terrain-induced impulses (stumbling on rocks)
- Payload shifts
- Human interaction (accidental bumps)

**Our mitigation:**
In `base_s2r_env_cfg.py` (active for ALL experts):
- External forces: ±3.0 N (up from 0.0)
- External torques: ±1.0 Nm (up from 0.0)
- Mass randomization: ±5.0 kg (up from ±2.5)
- Friction range: 0.15-1.3 (up from 0.3-1.0)
- Push velocity: ±0.5 m/s every 7-12 s (more frequent than Mason's 10-15 s)

**Validation:** `eval_student.py` with 5 N pushes; must recover within 2 seconds.

---

### R9: Observation Normalization OFF (LOW — ACCEPTED)

**What happens without mitigation:**
Different observation channels have different scales (height scan ±1.0 m, joint vel
±10 rad/s, gravity ~1.0). Without normalization, large-scale channels can dominate
gradients. However, Mason's [512,256,128] with ELU activation trains stably without
normalization — demonstrated across Trials 12, 12b, and hybrid no-coach (20K+ iters each).

**Why we accept:** Proven stable in practice. Adding normalization mid-pipeline
would require revalidation of all reward weights.

---

### R10: No Sensor Dropout Simulation (MEDIUM)

**What happens without mitigation:**
Real LiDAR/depth cameras occasionally return invalid readings:
- Reflective surfaces (water, glass)
- Out-of-range targets
- Sensor hardware glitches
Training never encounters missing data, so the policy may catastrophically
misinterpret terrain when rays fail.

**Our mitigation:**
- `SensorNoiseWrapper`: 5 % of 187 height scan rays randomly set to 0.0 each step
- Fill value 0.0 matches the training convention for flat ground
- Policy learns that some rays may be unreliable

**Validation:** Run eval with 10 % dropout; verify no catastrophic failures.

---

## Risk Mitigation Coverage Map

| Mitigation Feature | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 |
|--------------------|----|----|----|----|----|----|----|----|----|-----|
| ActionDelayWrapper | X  |    |    |    |    |    |    |    |    |     |
| ObservationDelayWrapper | X |  |    |    |    |    |    |    |    |     |
| SensorNoiseWrapper |    |    |    |    | X  |    |    |    |    | X   |
| 20 Hz distillation |    |    | X  |    |    |    |    |    |    |     |
| motor_power reward |    |    |    |    |    | X  | X  |    |    |     |
| torque_limit reward|    |    |    |    |    | X  |    |    |    |     |
| Wider DR (mass/friction) | | |    |    |    |    |    | X  |    |     |
| External push forces |   |    |    |    |    |    |    | X  |    |     |
| Observation noise increase | | |  |    | X  |    |    |    |    |     |
| enable_corruption=True |  |  |    |    | X  |    |    |    |    |     |
| spot_sdk_wrapper   |    | X  |    |    |    |    |    |    |    |     |
| safety_layer       |    | X  |    |    |    | X  |    |    |    |     |
| height_scan_builder|    | X  |    |    | X  |    |    |    |    |     |
| export_onnx        |    | X  |    |    |    |    |    |    |    |     |
| DEPLOYMENT_CHECKLIST |   | X  |    |    |    |    |    |    |    |     |

---

## Codebase Strengths Identified During Evaluation

These existing features already contribute to sim-to-real readiness:

1. **35 documented bug fixes** — Including 5 sim-exploitable reward hacks
   (world-frame Z on slopes, unbounded penalties, belly-crawl exploit)
2. **All penalties clamped** (Bug #29) — Prevents NaN/Inf gradient explosions
3. **NaN sanitizer** (Bug #24) — Explicit NaN/Inf detection on every forward pass
4. **terrain_relative_height_penalty** (Bug #22) — Height measured relative to
   local terrain, not world-frame Z
5. **Value loss watchdog** (Bug #25) — Halves LR when value_loss > 100
6. **12 terrain types** in ROBUST_TERRAINS_CFG — Stairs, slopes, boulders, friction,
   rough, stepping stones, waves, obstacles
7. **Domain randomization** — Friction 0.3-1.0, mass ±2.5 kg, joint noise, push perturbations
8. **Contact-safe termination** — Soft penalty for body contact (not hard termination)
9. **Gait enforcement** — weight 10.0 diagonal trot reward
10. **Observation noise** — Additive uniform noise on all sensor channels

---

## Evaluation Scores (Pre-S2R Baseline)

| Terrain | Best Policy | Distance | Zone | Gap to 5/5 |
|---------|-------------|----------|------|------------|
| Friction | mason_hybrid_best_33200 | 49.5 m | 5/5 | None |
| Grass | mason_hybrid_best_33200 | 49.5 m | 5/5 | None |
| Boulder | obstacle_best_44400 | 30.4 m | 4/5 | 19.1 m |
| Stairs | obstacle_best_44400 | 15.7 m | 2/5 | 33.8 m |

**Target after S2R pipeline:** 49.5 m (5/5) on ALL terrains.

---

*Generated by third-party codebase evaluation, March 23 2026*
*AI2C Tech Capstone — MS for Autonomy*
