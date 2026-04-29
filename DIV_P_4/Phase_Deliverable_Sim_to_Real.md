# Deployment Document: Simulation to Real Transfer

## 1. Overview

This document captures the infrastructure specifications, sim-to-real transfer requirements, safety systems, and deployment protocol for transferring trained RL locomotion policies from simulation to the physical Boston Dynamics Spot quadruped. It covers the full pipeline from the NVIDIA Isaac Lab training environment through ONNX export to on-robot execution.

## 2. Training Infrastructure

### 2.1 Hardware

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA H100 NVL, 95,830 MiB (96 GB) HBM3 VRAM |
| System | Supermicro SYS-E403-13E-FRN2T |
| BMC | Supermicro Redfish API at 172.24.254.25 |
| OS access | SSH to t2user@172.24.254.24 |
| Recovery | BMC ForceRestart via Redfish when OS hangs |

**Training capacity:**

| Configuration | Environments | GPU Temp | Power | Status |
|--------------|-------------|----------|-------|--------|
| Sustained safe | 8,192 | 49C | 171W | Recommended |
| Maximum tested | 65,536 | 65C | 299W | Thermal limit |

### 2.2 Software Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| NVIDIA Isaac Lab | 0.54.2 | Physics simulation framework |
| PhysX | GPU-accelerated | Rigid body physics engine |
| RSL-RL | 5.0.1 | PPO implementation |
| PyTorch | 2.7.0+cu128 | Neural network training |
| Isaac Sim | 5.1.0 | Rendering and visualization |
| Python | 3.11 | Runtime |
| Conda env | `env_isaaclab` (H100), `isaaclab311` (local) | Package management |

### 2.3 Physics Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Physics DT | 0.002s (500 Hz) | Stable contact resolution |
| Control DT | 0.02s (50 Hz) | Matches Spot SDK command rate |
| Decimation | 10 | 500/50 = 10 physics steps per control step |
| Gravity | -9.81 m/s^2 | Standard Earth gravity |
| Solver type | TGS (GPU) | Temporal Gauss-Seidel for parallel solving |
| Contact offset | 0.02m | Collision detection margin |
| Friction combine | Multiply | Product of surface friction coefficients |

## 3. Robot Specifications

### 3.1 Boston Dynamics Spot

| Property | Value |
|----------|-------|
| Mass | 32 kg (nominal) |
| Degrees of freedom | 12 (4 legs x 3 joints) |
| Joint types | HX (abduction), HY (hip flexion), KN (knee) |
| Standing height | ~0.42m (body center) |

### 3.2 Joint Configuration

**DOF Ordering (Type-Grouped):**
```
Index 0-3:  HX joints [fl_hx, fr_hx, hl_hx, hr_hx]  (abduction)
Index 4-7:  HY joints [fl_hy, fr_hy, hl_hy, hr_hy]  (hip flexion)
Index 8-11: KN joints [fl_kn, fr_kn, hl_kn, hr_kn]  (knee)
```

**Default Standing Positions (radians):**

| Joint | FL | FR | HL | HR |
|-------|-----|-----|-----|-----|
| HX | 0.1 | -0.1 | 0.1 | -0.1 |
| HY | 0.9 | 0.9 | 1.1 | 1.1 |
| KN | -1.5 | -1.5 | -1.5 | -1.5 |

### 3.3 PD Control Gains

| Parameter | Value |
|-----------|-------|
| Kp (stiffness) | 60.0 |
| Kd (damping) | 1.5 |

These gains must match exactly between simulation and deployment. Mismatched PD gains are a primary source of sim-to-real failure.

### 3.4 Joint Effort Limits

| Joint Group | Torque Limit |
|------------|-------------|
| HX (abduction) | 45.0 Nm |
| HY (hip flexion) | 45.0 Nm |
| KN (knee) | 100.0 Nm |

### 3.5 Action Specification

| Property | Value |
|----------|-------|
| Action dimensions | 12 |
| Action type | Joint position offset from default |
| Action scale | 0.2 (network output x 0.2 = radians offset) |
| Control rate | 50 Hz |

## 4. Observation Specification (235 dimensions)

### 4.1 Height Scan (indices 0-186, 187 dimensions)

| Property | Value |
|----------|-------|
| Grid layout | 17 columns x 11 rows |
| Resolution | 0.1m between rays |
| Coverage | 1.6m forward/back x 1.0m left/right |
| Mount | Body link, 20.0m vertical offset (rays point down) |
| Alignment | Yaw-aligned (rotates with body heading) |
| Value range | [-1.0, 1.0] relative height from scanner |
| Fill value | **0.0 for flat ground** (NOT 1.0 — critical) |
| Noise | Uniform +/-0.1m |

### 4.2 Proprioception (indices 187-234, 48 dimensions)

| Index | Component | Dims | Noise |
|-------|-----------|------|-------|
| 187-189 | Base linear velocity [vx, vy, vz] | 3 | +/-0.1 m/s |
| 190-192 | Base angular velocity [wx, wy, wz] | 3 | +/-0.1 rad/s |
| 193-195 | Projected gravity [gx, gy, gz] | 3 | +/-0.05 |
| 196-198 | Velocity commands [vx_cmd, vy_cmd, wz_cmd] | 3 | None |
| 199-210 | Joint positions (relative to default) | 12 | +/-0.05 rad |
| 211-222 | Joint velocities | 12 | +/-0.5 rad/s |
| 223-234 | Previous actions | 12 | None |

## 5. Sim-to-Real Hardening

### 5.1 Three Core Wrappers

Sim-to-real transfer requires training the policy under conditions that approximate real-world imperfections. Three wrappers progressively degrade the simulation to match hardware:

**1. Action Delay Wrapper (40ms)**

| Property | Value |
|----------|-------|
| Maximum delay | 2 control steps = 40ms at 50 Hz |
| Implementation | GPU ring buffer, zero-copy |
| Scaling | Progressive: 0% (terrain rows 0-2), 30% (rows 3-4), 60% (rows 5-6), 100% (rows 7+) |
| Real-world source | Actuator communication latency (40-60ms) + motor response time |

**2. Observation Delay Wrapper (20ms)**

| Property | Value |
|----------|-------|
| Maximum delay | 1 control step = 20ms at 50 Hz |
| Scaling | Progressive ramp tied to terrain curriculum |
| Real-world source | IMU/encoder bus latency (10-20ms) + processing pipeline |

**3. Sensor Noise Wrapper**

| Noise Type | Parameters | Real-world Source |
|-----------|-----------|-------------------|
| Height scan dropout | 5% rays zeroed per step | LiDAR missed returns |
| IMU drift (Ornstein-Uhlenbeck) | Rate 0.002, reversion 0.01 | Gyroscope drift |
| Spike noise | 0.1% probability, magnitude 1.0 | Electromagnetic interference |
| Dropout fill value | 0.0 | Consistent with flat-ground assumption |
| Progressive scaling | Rows 0-2: 0x, Rows 3-4: 0.3x, Rows 5-6: 0.6x, Rows 7-9: 1.0x | |

### 5.2 Domain Randomization

Physics parameters are randomized during training to cover the range of conditions the robot may encounter:

| Parameter | Range | Rationale |
|-----------|-------|-----------|
| Static friction | 0.3 - 1.0 | Covers wet concrete to dry rubber |
| Dynamic friction | 0.3 - 0.8 | Sliding friction variation |
| Friction combine mode | Multiply | Consistent with PhysX behavior |
| Mass offset | +/-2.5 kg | Payload variation, component wear |
| Push velocity | +/-0.5 m/s every 10-15s | Human contact, wind gusts |
| Restitution | 0.0 | No bouncing (conservative) |

**Locked parameters (proven safe):** These ranges were established through extensive testing. Values outside these ranges caused training instability:
- Friction below 0.15 caused unrecoverable falls
- Mass offset above +/-5.0 kg was too aggressive
- Push forces above +/-3.0 N caused falls

### 5.3 Staged Actor Unfreeze

When fine-tuning a pre-trained policy with sim-to-real wrappers, network layers are unfrozen progressively to prevent catastrophic forgetting:

```
Iteration 0-300:    ALL layers frozen     (fresh critic calibrates to new reward landscape)
Iteration 300-500:  Layer 4 unfrozen      (output layer: 128 -> 12 adapts first)
Iteration 500-700:  Layer 3 unfrozen      (middle layer: 256 -> 128 adapts)
Iteration 700+:     ALL layers unfrozen   (full fine-tuning with stabilized critic)
```

### 5.4 Additional S2R Reward Penalties

Beyond the standard locomotion rewards, S2R training adds hardware-protective penalties:

| Term | Weight | Description |
|------|--------|-------------|
| `motor_power` | -0.005 | sum(\|torque_i x velocity_i\|), clamped [0, 500] |
| `torque_limit` | -0.3 | ReLU(abs(torque) - limit), per-joint |
| `undesired_contacts` | -1.5 | Soft penalty replacing hard body-contact termination |

## 6. ONNX Export for Deployment

### 6.1 Export Specification

| Property | Value |
|----------|-------|
| Format | ONNX (Open Neural Network Exchange) |
| Opset version | 17 |
| Model size | ~1.1 MB (actor only) |
| Input | (1, 235) float32 observation vector |
| Output | (1, 12) float32 joint position offsets |
| Dynamic batch | Supported |
| Verification | Max diff vs PyTorch < 1e-5 |
| Inference target | < 10ms per forward pass |

### 6.2 Export Process

1. Load trained PyTorch checkpoint (actor weights only)
2. Trace model with representative input tensor
3. Export to ONNX with dynamic batch axes
4. Verify output matches PyTorch within 1e-5 tolerance
5. Benchmark inference latency on target hardware

## 7. Safety Systems

### 7.1 Software Safety Layer

| Threshold | Value | Action |
|-----------|-------|--------|
| Joint torque E-stop | 120% of motor limits (HX/HY: 54 Nm, KN: 120 Nm) | Immediate motor shutdown |
| Body pitch limit | +/-60 degrees | Sit-down command |
| Body roll limit | +/-60 degrees | Sit-down command |
| Max velocity | 4.0 m/s | Runaway detection, E-stop |
| Command timeout | 100ms | If no new command in 100ms, E-stop |
| Battery warning | 20% remaining | Audio/visual alert |
| Battery sit-down | 10% remaining | Automatic sit-down |

### 7.2 Hardware Safety

- **Physical E-stop button:** Hardware kill switch that cuts motor power immediately, bypassing all software
- **Tether:** Physical tether during initial deployment stages (Stages 1-2)
- **Operator proximity:** Human operator within arm's reach during all tethered testing

### 7.3 Training Safety Constraints

These constraints are enforced during simulation training to prevent the policy from learning unsafe behaviors:

- **Episode termination on flip:** Body inverts past 90 degrees
- **Bad orientation termination:** Body pitch/roll exceeds safe limits
- **Joint position limits:** Hard mechanical limits enforced in physics
- **Body contact penalty:** -10.0 weight prevents body-ground contact learning

## 8. Four-Environment Evaluation Gauntlet

### 8.1 Overview

All trained policies are evaluated in a standardized 4-environment test that measures performance across distinct terrain challenges. Each arena is a 50m linear course divided into 5 zones of increasing difficulty.

| Property | Value |
|----------|-------|
| Arena length | 50.0m |
| Arena width | 30.0m |
| Zones | 5 (10m each) |
| Episode timeout | 600 seconds (10 minutes) |
| Fall threshold | 0.15m body height |
| Completion criterion | X position >= 49.0m |

### 8.2 Friction Arena

Tests traction limits as surface friction decreases:

| Zone | Range | Static Friction | Dynamic Friction | Description |
|------|-------|----------------|-----------------|-------------|
| 1 | 0-10m | 0.90 | 0.80 | 60-grit sandpaper |
| 2 | 10-20m | 0.60 | 0.50 | Dry rubber on concrete |
| 3 | 20-30m | 0.35 | 0.25 | Wet concrete |
| 4 | 30-40m | 0.15 | 0.08 | Wet ice |
| 5 | 40-50m | 0.05 | 0.02 | Oil on polished steel |

### 8.3 Grass Arena

Tests locomotion under increasing drag forces simulating vegetation:

| Zone | Range | Drag Coefficient | Description |
|------|-------|-----------------|-------------|
| 1 | 0-10m | 0.5 | Light fluid resistance |
| 2 | 10-20m | 2.0 | Thin grass |
| 3 | 20-30m | 5.0 | Medium lawn |
| 4 | 30-40m | 10.0 | Thick grass |
| 5 | 40-50m | 20.0 | Dense brush |

### 8.4 Boulder Arena

Tests obstacle negotiation with increasing obstacle size:

| Zone | Range | Boulder Edge Size | Density | Count | Description |
|------|-------|--------------------|---------|-------|-------------|
| 1 | 0-10m | 0.03-0.05m | 15/m^2 | 4,500 | Gravel |
| 2 | 10-20m | 0.10-0.15m | 8/m^2 | 2,400 | River rocks |
| 3 | 20-30m | 0.25-0.35m | 4/m^2 | 1,200 | Large rocks |
| 4 | 30-40m | 0.50-0.70m | 2/m^2 | 600 | Small boulders |
| 5 | 40-50m | 0.80-1.20m | 1/m^2 | 300 | Large boulders |

Boulder shapes are distributed evenly across four polyhedra types: D8 (octahedron), D10 (trapezohedron), D12 (dodecahedron), D20 (icosahedron) — 25% each.

### 8.5 Stairs Arena

Tests step-climbing ability with increasing riser height:

| Zone | Range | Step Height | Step Depth | Steps | Description |
|------|-------|------------|-----------|-------|-------------|
| 1 | 0-10m | 0.03m | 0.30m | 33 | Access ramp |
| 2 | 10-20m | 0.08m | 0.30m | 33 | Low residential |
| 3 | 20-30m | 0.13m | 0.30m | 33 | Standard residential |
| 4 | 30-40m | 0.18m | 0.30m | 33 | Steep commercial |
| 5 | 40-50m | 0.23m | 0.30m | 33 | Maximum challenge |

Zone transitions use 5 interpolation steps to smooth the height change between zones.

### 8.6 Evaluation Metrics

Each episode records 17 metrics:

| Metric | Description |
|--------|-------------|
| completion | Boolean: reached 49.0m |
| progress | Maximum X position achieved (meters) |
| zone_reached | Highest zone entered (1-5) |
| time_to_complete | Seconds to reach 49.0m (null if incomplete) |
| stability_score | Composite of roll/pitch variance |
| mean_roll | Average body roll angle |
| mean_pitch | Average body pitch angle |
| height_variance | Body height stability |
| mean_ang_vel | Average angular velocity magnitude |
| fall_detected | Boolean: body height below 0.15m |
| flip_detected | Boolean: body inverted |
| fall_location | X position of fall (null if no fall) |
| fall_zone | Zone where fall occurred |
| mean_velocity | Average forward speed |
| total_energy | Cumulative energy consumption |
| episode_length | Total episode duration (seconds) |

## 9. Five-Stage Deployment Testing Protocol

### 9.1 Pre-Deployment Checklist

Before any physical testing, verify:

- [ ] Policy passes 4-env gauntlet (target: 5/5 zones all terrains)
- [ ] S2R stress test: 40ms delay + 20 Hz control + 5% sensor dropout
- [ ] ONNX export verification: max diff < 1e-5 vs PyTorch
- [ ] Spot SDK connection: lease acquisition, power on, stand, sit
- [ ] Joint name mapping: training order matches SDK order exactly
- [ ] PD gains configured: Kp=60, Kd=1.5
- [ ] Safety layer tested: mock data triggers for all E-stop conditions
- [ ] Height scan pipeline: depth camera -> 187-dim observation -> policy
- [ ] Telemetry logging: JSONL output functional
- [ ] Hardware E-stop button: tested and confirmed functional
- [ ] Battery: fully charged

### 9.2 Stage 1: Tethered Flat Ground (Lab)

| Property | Value |
|----------|-------|
| Duration | ~30 minutes |
| Environment | Indoor flat floor |
| Safety | Physical tether attached |
| Operator | Within arm's reach |

**Tests:**
1. Stand still (30 seconds, no commands)
2. Walk forward (1 m/s, 10m)
3. Walk backward (0.5 m/s, 5m)
4. Turn in place (90 degrees left and right)
5. Lateral strafe (0.3 m/s, 3m each direction)
6. Stop-and-go (walk 3m, stop 5s, repeat 5x)
7. Speed ramp (0 -> 1.5 m/s over 10 seconds)

**Pass criteria:** No falls, <10% torque violations, <5% battery drain per 10 minutes.

### 9.3 Stage 2: Tethered Rough Ground (Lab)

| Property | Value |
|----------|-------|
| Duration | ~30 minutes |
| Environment | Lab with terrain props |
| Safety | Physical tether attached |
| Props | Foam mats (3-5cm), wood boards (5-10cm), plastic sheet, rubber tiles |

**Tests:**
1. Walk across foam mats (variable surface compliance)
2. Step up 5cm ledge (wood board)
3. Step up 10cm ledge (stacked boards)
4. Walk on low-friction surface (plastic sheet)
5. Walk on high-friction surface (rubber tiles)
6. Manual push recovery (moderate lateral push while walking)

**Pass criteria:** Traverses all terrain props, recovers from manual pushes within 2 seconds.

### 9.4 Stage 3: Untethered Flat Ground (Outdoor)

| Property | Value |
|----------|-------|
| Duration | ~30 minutes |
| Environment | Outdoor flat surface (parking lot, sidewalk) |
| Safety | No tether, operator within 5m, E-stop ready |

**Tests:**
1. Sustained walking at 1-2 m/s (50m out and back)
2. Speed test at 2+ m/s (straight line, 30m)
3. Figure-8 pattern (10m radius)
4. Continuous walking for 5 minutes

**Pass criteria:** Stable 5-minute continuous run, <0.5 second command response time.

### 9.5 Stage 4: Untethered Rough Terrain (Outdoor)

| Property | Value |
|----------|-------|
| Duration | ~45 minutes |
| Environment | Outdoor varied terrain |
| Safety | Operator within 5m, E-stop ready |

**Tests:**
1. Pavement-to-grass transition
2. Grass walking (30m)
3. Gravel surface (20m)
4. Curb step-up (standard 6-inch / 15cm)
5. Curb step-down (standard 6-inch / 15cm)
6. Continuous mixed-terrain walking for 10 minutes

**Pass criteria:** No stumbling, successful curb steps, 10-minute endurance.

### 9.6 Stage 5: Full Course (Matching 4-Env Eval)

| Property | Value |
|----------|-------|
| Duration | 1+ hour |
| Environment | Physical course matching simulation arenas |
| Safety | Operator present, E-stop ready |

**Physical terrain zones to construct:**
- Friction variations (waxed floor, wet surface, rubber mat)
- Grass/vegetation sections
- Rock/boulder field (gravel -> cobblestones -> larger rocks)
- Staircase (variable step height)

**Pass criteria:**
- Completes each terrain zone
- Real-world distances within 20% of simulation results
- 0% flip rate
- 95% torque compliance (within motor limits)

## 10. Telemetry and Logging

### 10.1 On-Robot Logging

| Property | Value |
|----------|-------|
| Format | JSONL (one step per line) |
| Frequency | 20 Hz (every control step) |
| Storage | On-board SSD |

**Fields per step:**

```json
{
  "step": 1234,
  "time_s": 24.68,
  "velocity_cmd": [1.0, 0.0, 0.0],
  "action": [0.1, -0.1, ...],
  "joint_torques": [12.3, -8.1, ...],
  "battery_pct": 85.2,
  "body_pitch": 0.03,
  "body_roll": -0.01,
  "base_velocity": [0.98, 0.02, -0.01]
}
```

### 10.2 Real-Time Dashboard (Optional)

UDP streaming for live monitoring:
- Compact binary format
- 20 Hz update rate
- Displays: velocity, torque, orientation, battery, safety status
- Alerts on safety threshold violations

## 11. Known Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PD gain mismatch | Medium | High (unstable gait) | Verify Kp/Kd before every deployment |
| Height scan pipeline error | Medium | High (policy input garbage) | Validate 187-dim output before first step |
| Joint ordering mismatch | Low | Critical (random joint commands) | Unit test mapping against known positions |
| Actuator latency > 60ms | Medium | Medium (degraded gait) | Trained with 40ms delay, 50% safety margin |
| Battery under-voltage | Low | Medium (sudden shutdown) | 20% warning, 10% sit-down |
| Terrain beyond training | Medium | High (fall) | Conservative speed limits, operator E-stop |
| Communication dropout | Low | Critical (runaway) | 100ms timeout E-stop |
