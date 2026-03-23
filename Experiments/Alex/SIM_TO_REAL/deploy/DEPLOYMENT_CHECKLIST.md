# Sim-to-Real Deployment Checklist

5-stage progressive testing protocol for deploying the S2R distilled student
policy on real Boston Dynamics Spot hardware.

**SAFETY FIRST:** Always have a human operator with hardware E-stop within 3m.
Never run the policy unattended. Battery must be >30% before starting any test.

---

## Pre-Deployment Verification

- [ ] Student policy passes 4-env gauntlet in sim (target: 5/5 all terrains)
- [ ] Student policy passes S2R stress test (40ms delay + 20Hz + 5% dropout)
- [ ] ONNX export verified (max diff < 1e-5 vs PyTorch)
- [ ] Spot SDK connection tested (lease, power, stand, sit)
- [ ] Joint name mapping verified (training order matches SDK order)
- [ ] PD gains configured (Kp=60, Kd=1.5 — matching training)
- [ ] Safety layer tested with mock data (E-stop triggers verified)
- [ ] Height scan pipeline tested (depth camera -> 187-dim -> policy)
- [ ] Telemetry logging verified (JSONL output, UDP stream)
- [ ] E-stop button tested (hardware kill switch functional)
- [ ] Calibration complete (joint zeros, latency measured)

---

## Stage 1: Tethered Flat Ground (Lab)

**Setup:** Spot on flat concrete, safety tether to overhead rail, human with E-stop.

**Tests:**
- [ ] Stand still (zero velocity command) — 60 seconds
- [ ] Walk forward at 0.5 m/s — 10 seconds
- [ ] Walk backward at 0.3 m/s — 10 seconds
- [ ] Turn in place (±1.0 rad/s yaw) — 10 seconds each direction
- [ ] Strafe left/right at 0.3 m/s — 10 seconds each
- [ ] Stop-and-go (alternate 0/1 m/s every 5 seconds) — 60 seconds
- [ ] Speed ramp (0 -> 2.0 m/s over 10 seconds, then stop)

**Pass criteria:**
- [ ] No falls
- [ ] No E-stop triggers
- [ ] Smooth diagonal trot gait
- [ ] <10% torque limit violations
- [ ] Joint positions stay within URDF limits
- [ ] Battery drain rate acceptable (<5% per 10 min)

**Data collected:** Telemetry JSONL, video recording, torque logs.

---

## Stage 2: Tethered Rough Ground (Lab)

**Setup:** Same as Stage 1, plus terrain obstacles on floor.

**Terrain props:**
- Foam exercise mats (3-5 cm uneven surface)
- Wood boards (5-10 cm height, ~30 cm wide)
- Smooth plastic sheet (low friction simulation)
- Rubber floor tiles (high friction)

**Tests:**
- [ ] Walk over foam mats at 0.5 m/s
- [ ] Step up onto 5 cm wood board
- [ ] Step up onto 10 cm wood board
- [ ] Walk on plastic sheet (low friction) at 0.3 m/s
- [ ] Walk on rubber tiles at 1.0 m/s
- [ ] Manual push from side (~3 N) during walk — observe recovery
- [ ] Manual push from behind (~3 N) during walk — observe recovery

**Pass criteria:**
- [ ] Traverses all terrain props
- [ ] Recovers from both pushes within 2 seconds
- [ ] No falls
- [ ] Height scan correctly detects obstacles (verify in telemetry)

---

## Stage 3: Untethered Flat Ground (Outdoor)

**Setup:** Open field, clear 10m radius, human with E-stop within 3m.

**Tests:**
- [ ] Walk forward 1.0 m/s for 2 minutes
- [ ] Walk forward 2.0 m/s for 1 minute
- [ ] Figure-8 pattern (1.0 m/s forward + ±1.0 rad/s yaw)
- [ ] Stop and stand for 30 seconds
- [ ] Walk backward 0.5 m/s for 10 seconds
- [ ] Full-speed test: 3.0 m/s forward for 10 seconds (if safe)
- [ ] 5-minute continuous run (varied velocity commands)

**Pass criteria:**
- [ ] Stable 5-minute continuous run
- [ ] Responds to all velocity commands within 0.5 seconds
- [ ] Maintains gait at all tested speeds
- [ ] No stumbles or near-falls
- [ ] Battery drain rate within specifications

---

## Stage 4: Untethered Rough Terrain (Outdoor)

**Setup:** Mixed outdoor terrain, human with E-stop within 5m.

**Tests:**
- [ ] Concrete -> grass transition at 1.0 m/s
- [ ] Grass -> gravel transition at 0.5 m/s
- [ ] Gravel -> concrete transition at 1.0 m/s
- [ ] Walk on grass (5 cm height) for 2 minutes
- [ ] Walk on gravel path for 2 minutes
- [ ] Curb step-up (~10 cm) at 0.3 m/s
- [ ] Curb step-down (~10 cm) at 0.3 m/s
- [ ] 10-minute continuous run over mixed terrain

**Pass criteria:**
- [ ] Handles all terrain transitions without stumbling
- [ ] Successfully steps up and down curb
- [ ] 10-minute run without falls
- [ ] Battery life acceptable for intended deployment duration

---

## Stage 5: Full Course (Matching 4-Env Eval)

**Setup:** Build physical versions of the 4 evaluation terrains.

**Physical terrain zones:**
1. **Friction zone:** Smooth plastic (low friction) -> rubber mats (high friction)
2. **Grass zone:** Natural grass of increasing height (2cm -> 15cm)
3. **Boulder zone:** Scattered objects (5cm -> 25cm height)
4. **Stairs zone:** Physical staircase (5cm -> 20cm step height)

**Tests:**
- [ ] Traverse friction zone end-to-end
- [ ] Traverse grass zone end-to-end
- [ ] Navigate boulder zone (at least 3 obstacles)
- [ ] Climb staircase (at least 3 steps)
- [ ] Descend staircase
- [ ] Full 4-quadrant course (if feasible)

**Pass criteria:**
- [ ] Completes each zone without falling
- [ ] Real-world scores within 20% of simulation scores
- [ ] 0% flip rate
- [ ] 95% of torques within motor limits
- [ ] Height scan correctly maps real terrain (verify in telemetry)

---

## Post-Deployment Analysis

- [ ] Compare sim vs real telemetry (JSONL format allows direct comparison)
- [ ] Identify sim-to-real gap (speed error, stability, fall rate)
- [ ] Document worst-case failure modes
- [ ] If needed: fine-tune policy on real terrain data (10-50 episodes)
- [ ] Write deployment report for CMU review

---

*AI2C Tech Capstone — MS for Autonomy, March 2026*
