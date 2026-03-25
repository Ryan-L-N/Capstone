# SIM_TO_REAL Implementation Plan

## Why This Exists

Our Spot RL training (Mason hybrid, terrain 3.83, 0% flip) excels on flat/friction
terrain but struggles on boulders (30.4 m, 4/5) and stairs (15.7 m, 2/5). A
third-party evaluation before our CMU PhD meeting identified 10 sim-to-real risks.

This pipeline addresses both problems simultaneously: train 6 terrain specialists
that each dominate one terrain type, with all sim-to-real mitigations baked in,
then distill into a single deployment-ready generalist.

---

## Expert Rationale

### Why 6 specialists instead of 1 generalist?

Our existing generalist (ROBUST_TERRAINS_CFG with 12 terrain types) spreads training
across too many challenges. The curriculum pushes the robot through easy→hard difficulty,
but with only ~10% of terrain being stairs at any time, the policy never gets enough
stair-climbing practice to master it.

**Solution:** Give each expert 80% of its specialty terrain. The stairs expert spends
80% of its training time on stairs. The boulder expert spends 80% on boulders. Each
becomes a domain master.

### Why train from scratch?

Starting from a pretrained checkpoint biases the expert toward the previous terrain
distribution. Training from scratch lets each expert develop its own optimal gait
strategy for its terrain type without inheriting the generalist's compromises.

### Why 20% flat baseline in each expert?

Prevents catastrophic forgetting of basic locomotion. An expert that can climb stairs
but can't walk on flat ground is useless. The 20% flat ensures every expert maintains
a stable trot gait.

---

## Terrain Configurations

### Expert 1: FRICTION (80% friction + 20% flat)
- Friction plane: mu ranges from 0.05 (ice) to 1.5 (high-grip rubber)
- 10 difficulty rows progressively reduce friction coefficient
- Teaches: balance on slippery surfaces, traction management

### Expert 2: STAIRS UP (40% pyramid_stairs + 40% hf_stairs + 20% flat)
- Step heights: 3 cm → 25 cm (10 difficulty levels)
- Step width: 0.3 m
- Two stair types for diversity (geometric pyramid + heightfield)
- Teaches: foot clearance, weight transfer uphill, step-up motion

### Expert 3: STAIRS DOWN (80% pyramid_stairs_down + 20% flat)
- Same step heights as Expert 2 but descending
- Teaches: controlled descent, braking, foot placement on drop-offs

### Expert 4: BOULDERS (40% boxes + 20% discrete_obstacles + 20% repeated_boxes + 20% flat)
- Obstacle heights: 5 cm → 30 cm
- Random scatter (discrete_obstacles: 40 per patch)
- Regular grid (boxes) + irregular placement
- Teaches: stepping over/around obstacles, body clearance, recovery

### Expert 5: SLOPES (35% up + 35% down + 10% wave + 20% flat)
- Slope angles: 0° → 29° (0.0 → 0.5 rad)
- Wave terrain: 5-20 cm amplitude undulations
- Teaches: traction on inclines, orientation control, slope transitions

### Expert 6: MIXED ROUGH (40% random_rough + 40% stepping_stones + 20% flat)
- Random rough: 2-15 cm noise amplitude
- Stepping stones: 25-50 cm width, 10-40 cm gaps
- Teaches: precise foot placement, balance on uneven ground

---

## Reward Tuning Strategy

Based on Trial 12b obstacle tuning lesson: tune rewards along the same kinematic
chain together. The step-up motion requires:
- **foot_clearance** (lift foot high enough to clear obstacle)
- **joint_pos** (allow extreme joint angles for high steps)
- **base_orientation** (allow body tilt during climbing)

These three are loosened together for stair/boulder experts.

### Base Rewards (16 terms — active in ALL experts)

```
# Task rewards
air_time:                5.0     base_angular_velocity:   5.0
base_linear_velocity:    5.0     foot_clearance:          0.5  (overridden)
gait:                   10.0

# Penalties
action_smoothness:      -1.0     air_time_variance:      -1.0
base_motion:            -2.0     base_orientation:       -3.0  (overridden)
foot_slip:              -0.5     joint_acc:              -1e-4
joint_pos:              -0.7     joint_torques:          -1e-3
joint_vel:              -1e-2

# Proven additions
terrain_relative_height: -2.0    dof_pos_limits:         -3.0

# S2R additions
motor_power:            -0.005   torque_limit:           -0.3

# Frozen
body_height_tracking:    0.0     stumble:                 0.0
```

### Per-Expert Overrides

| Expert | foot_clearance | base_orientation | joint_pos | foot_slip | gait |
|--------|---------------|------------------|-----------|-----------|------|
| Friction | 0.5 | -3.0 | -0.7 | **-1.5** | 10.0 |
| Stairs Up | **2.0** | **-2.0** | **-0.3** | -0.5 | 10.0 |
| Stairs Down | **2.0** | **-2.0** | **-0.3** | -0.5 | 10.0 |
| Boulders | **2.5** | **-2.0** | **-0.3** | -0.5 | 10.0 |
| Slopes | 0.5 | **-2.5** | -0.7 | **-1.5** | 10.0 |
| Mixed Rough | **1.5** | -3.0 | **-0.5** | -0.5 | **12.0** |

---

## Sim-to-Real Hardening

Every expert trains with ALL of these from step 0:

### Environment config level (base_s2r_env_cfg.py)
- `enable_corruption = True`
- Observation noise: base_vel ±0.2, ang_vel ±0.2, height_scan ±0.2,
  joint_pos ±0.08, joint_vel ±0.8
- External forces: ±3.0 N, torques ±1.0 Nm
- Mass randomization: ±5.0 kg (vs Mason's ±2.5)
- Friction: static 0.15-1.3, dynamic 0.1-1.0 (vs Mason's 0.3-1.0, 0.3-0.8)
- Push velocity: ±0.5 m/s every 7-12 s (vs Mason's 10-15 s)
- Motor power penalty: -0.005
- Torque limit penalty: -0.3 (hip 45 Nm, knee 100 Nm)
- Joint torques weight: -1e-3 (vs Mason's -5e-4)

### Wrapper level (applied in train_expert.py)
- ActionDelayWrapper: 2 steps = 40 ms
- ObservationDelayWrapper: 1 step = 20 ms
- SensorNoiseWrapper: Gaussian, 5% dropout, OU-process IMU drift

### Deferred to distillation
- 20 Hz control rate (experts train at 50 Hz for learning quality)
- Temporal observation history (if needed based on eval results)

---

## Distillation Pipeline

### Phase 1: Expert Evaluation
After training all 6 experts, evaluate each on the 4-env gauntlet. Select best
checkpoint per expert based on terrain metric + specialist terrain score.

### Phase 2: Distillation
- Load 6 frozen experts
- Create MultiExpertRouter (235 → 64 → 6 softmax)
- Train student [512,256,128] on DISTILLATION_TERRAINS_CFG (balanced all-terrain)
- 20 Hz control rate (decimation = 25)
- All S2R wrappers active
- Alpha annealing: 0.8 → 0.2 (expert demo → self-play)
- 8000 iterations

### Phase 3: Student Evaluation
- 4-env gauntlet (target: 5/5 all terrains)
- 5-ring composite gauntlet (target: 600/600)
- S2R stress tests (latency, 20 Hz, dropout)
- Torque audit (95% within limits)

---

## Deployment Pipeline

### Stage 1: ONNX Export
Convert student policy to ONNX. Verify output matches PyTorch (max diff < 1e-5).

### Stage 2: Spot SDK Integration
- Observation builder: Spot state → 235-dim tensor
- Action executor: 12-dim action → JointCommand (scale 0.2, Kp=60, Kd=1.5)
- 20 Hz control loop

### Stage 3: Safety Layer
- Joint limit hard clamp
- Torque monitoring (E-stop at 120%)
- Orientation watchdog (E-stop at 60 deg)
- Velocity watchdog (E-stop at 4.0 m/s)
- Communication timeout (hold → sit after 100 ms)

### Stage 4: Calibration
- Joint zero verification
- PD gain sweep
- Friction estimation
- Latency measurement

### Stage 5: Staged Real-World Testing
1. Tethered flat (lab) — stand, walk, turn
2. Tethered rough (lab) — foam, wood, plastic
3. Untethered flat (outdoor) — 5 min continuous
4. Untethered rough (outdoor) — grass, gravel, curbs
5. Full course — physical 4-quadrant gauntlet

---

## Training Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Network | [512, 256, 128] | Mason baseline |
| Activation | ELU | Mason baseline |
| PPO epochs | 5 | Mason baseline |
| Mini-batches | 4 | Mason baseline |
| LR schedule | Adaptive KL | Mason baseline |
| Cosine LR range | 1e-3 → 1e-5 | Train.py |
| Warmup | 50 iters | Train.py |
| init_noise_std | 1.0 | Mason baseline |
| max_noise_std | 0.5 | Training rule (Bug #28d) |
| min_noise_std | 0.3 | Training rule (Bug #24) |
| Episode length | 20 s | Mason baseline |
| Decimation (expert) | 10 (50 Hz) | Mason baseline |
| Decimation (distill) | 25 (20 Hz) | Real Spot SDK rate |
| Physics dt | 0.002 s | Isaac Sim |
| num_envs | 4096 | H100 safe thermal |
| Expert iterations | 10000 | ~1.6B steps |
| Distillation iterations | 8000 | ~785M steps |
| Save interval | 100 | User preference |

---

## Estimated Timeline

| Phase | Duration | GPU Hours |
|-------|----------|-----------|
| Code + documentation | ~5 h | 0 |
| Expert training (6 x 10K iters) | ~3 days | ~102 h |
| Expert evaluation | ~0.5 day | ~3 h |
| Distillation training (8K iters) | ~0.5 day | ~14 h |
| Student evaluation | ~0.5 day | ~3 h |
| Deployment layer code | ~1 day | 0 |
| Staged real-world testing | ~3 days | 0 |
| **Total** | **~9 days** | **~122 h** |

---

## Success Criteria

| Metric | Target | Current | Gap |
|--------|--------|---------|-----|
| Friction 4-env | 49.5 m (5/5) | 49.5 m | 0 |
| Grass 4-env | 49.5 m (5/5) | 49.5 m | 0 |
| Boulder 4-env | 49.5 m (5/5) | 30.4 m | 19.1 m |
| Stairs 4-env | 49.5 m (5/5) | 15.7 m | 33.8 m |
| 5-ring gauntlet | 600/600 | ~200 | ~400 |
| Flip rate | 0% | 0% | 0 |
| Torque compliance | 95% | N/A | N/A |
| 20 Hz stability | Pass all | N/A | N/A |
| Real-world (Stage 5) | Within 20% of sim | N/A | N/A |

---

*AI2C Tech Capstone — MS for Autonomy, March 2026*
