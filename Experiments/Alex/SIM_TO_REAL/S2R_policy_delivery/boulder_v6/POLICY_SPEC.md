# Boulder V6 Expert Policy — Specification

**Checkpoint:** `boulder_v6_expert_4500.pt` (6.6 MB)
**ONNX:** `boulder_v6_expert_4500.onnx` (1.1 MB, actor only)
**Training:** 4,500 iterations, ~451M steps, ~11.3 hours on H100
**Peak terrain score:** 6.13 | **Flip rate:** 6.4%
**Base checkpoint:** `distilled_6899.pt` (actor_only_resume + 300 iter critic warmup)

---

## 4-Environment Eval Results (1 episode each)

| Environment | Progress | Zones | Result |
|-------------|----------|-------|--------|
| Friction | 49.5m | 5/5 | COMPLETE |
| Grass | 21.2m | 3/5 | FLIP |
| Boulder | 26.6m | 3/5 | FLIP |
| Stairs | 22.9m | 3/5 | FLIP |

**Notable:** 22.9m on stairs without any stair-specific training. Best single-policy obstacle performance.

---

## Network Architecture

**Type:** MLP (fully connected) with ELU activations
**Framework:** PyTorch / RSL-RL PPO ActorCritic

### Actor (policy — used for deployment)

```
Input (235) -> Linear(235, 512) -> ELU -> Linear(512, 256) -> ELU -> Linear(256, 128) -> ELU -> Linear(128, 12) -> Output (12)
```

| Layer | Type | In | Out | Parameters |
|-------|------|-----|------|------------|
| actor.0 | Linear | 235 | 512 | 120,832 |
| actor.1 | ELU | 512 | 512 | 0 |
| actor.2 | Linear | 512 | 256 | 131,328 |
| actor.3 | ELU | 256 | 256 | 0 |
| actor.4 | Linear | 256 | 128 | 32,896 |
| actor.5 | ELU | 128 | 128 | 0 |
| actor.6 | Linear | 128 | 12 | 1,548 |
| **Total** | | | | **286,604** |

Plus `std` parameter: 12 floats (log-space exploration noise — not used at inference).

### Critic (value function — training only, NOT in ONNX)

Same architecture as actor but final layer outputs 1 (value estimate).

---

## Observation Vector (235 dimensions)

All observations are concatenated in the order listed below. This order is **critical** — the network was trained expecting exactly this layout.

### Height Scan (indices 0-186)

| Index | Name | Dims | Description | Noise |
|-------|------|------|-------------|-------|
| 0-186 | `height_scan` | 187 | Terrain height raycaster (17x11 grid) | +/-0.1 |

### Proprioceptive (indices 187-234)

| Index | Name | Dims | Description | Noise |
|-------|------|------|-------------|-------|
| 187-189 | `base_lin_vel` | 3 | Body-frame linear velocity [vx, vy, vz] (m/s) | +/-0.1 |
| 190-192 | `base_ang_vel` | 3 | Body-frame angular velocity [wx, wy, wz] (rad/s) | +/-0.1 |
| 193-195 | `projected_gravity` | 3 | Gravity vector in body frame [gx, gy, gz] | +/-0.05 |
| 196-198 | `velocity_commands` | 3 | Target velocity [vx_cmd, vy_cmd, wz_cmd] | none |
| 199-210 | `joint_pos` | 12 | Joint positions relative to default (rad) | +/-0.05 |
| 211-222 | `joint_vel` | 12 | Joint velocities relative to default (rad/s) | +/-0.5 |
| 223-234 | `actions` | 12 | Previous action output (normalized) | none |

**Height scanner details:**
- Grid: 17 columns x 11 rows = 187 rays
- Resolution: 0.1 m between rays
- Coverage: 1.6 m (forward/back) x 1.0 m (left/right)
- Mounted on body link with 20.0 m vertical offset (rays point down)
- Alignment: yaw-aligned (rotates with body heading)
- Values: relative height difference from scanner origin, clipped to [-1.0, 1.0]
- Fill value for flat ground: **0.0** (NOT 1.0)
- Update rate: 50 Hz (matches control frequency)

---

## Joint Ordering (12 DOF)

**CRITICAL: Joints are TYPE-GROUPED, not leg-grouped.**

| Index | Joint Name | Type | Leg |
|-------|-----------|------|-----|
| 0 | `fl_hx` | Abduction (HX) | Front-Left |
| 1 | `fr_hx` | Abduction (HX) | Front-Right |
| 2 | `hl_hx` | Abduction (HX) | Hind-Left |
| 3 | `hr_hx` | Abduction (HX) | Hind-Right |
| 4 | `fl_hy` | Hip (HY) | Front-Left |
| 5 | `fr_hy` | Hip (HY) | Front-Right |
| 6 | `hl_hy` | Hip (HY) | Hind-Left |
| 7 | `hr_hy` | Hip (HY) | Hind-Right |
| 8 | `fl_kn` | Knee (KN) | Front-Left |
| 9 | `fr_kn` | Knee (KN) | Front-Right |
| 10 | `hl_kn` | Knee (KN) | Hind-Left |
| 11 | `hr_kn` | Knee (KN) | Hind-Right |

**Grouping pattern:** All 4 HX joints, then all 4 HY joints, then all 4 KN joints.
**Leg order within each group:** FL, FR, HL, HR.

---

## Action Space (12 dimensions)

| Property | Value |
|----------|-------|
| Type | Joint position targets (PD control) |
| Scale | 0.2 (action x 0.2 = position offset in rad) |
| Offset | Default standing pose (`use_default_offset=True`) |
| PD gains | Kp = 60, Kd = 1.5 |
| Control frequency | 50 Hz |
| Physics frequency | 500 Hz (decimation = 10) |

**Interpretation:** `target_joint_pos = default_pos + (action * 0.2)`

The PD controller then computes torques at 500 Hz:
```
torque = Kp * (target_pos - current_pos) + Kd * (0 - current_vel)
```

---

## Default Standing Pose (radians)

```
fl_hx=0.1, fr_hx=-0.1, hl_hx=0.1, hr_hx=-0.1,
fl_hy=0.9, fr_hy=0.9,  hl_hy=1.1, hr_hy=1.1,
fl_kn=-1.5, fr_kn=-1.5, hl_kn=-1.5, hr_kn=-1.5
```

---

## Inference Pseudocode

```python
import torch
import torch.nn as nn

# Build actor
actor = nn.Sequential(
    nn.Linear(235, 512), nn.ELU(),
    nn.Linear(512, 256), nn.ELU(),
    nn.Linear(256, 128), nn.ELU(),
    nn.Linear(128, 12),
)

# Load weights (actor only)
ckpt = torch.load("boulder_v6_expert_4500.pt", map_location="cpu")
sd = ckpt["model_state_dict"]
actor_sd = {k.replace("actor.", ""): v for k, v in sd.items() if k.startswith("actor.")}
actor.load_state_dict(actor_sd)
actor.eval()

# At each 50 Hz control step:
obs = build_observation_vector()  # shape: (235,) — see order above
with torch.no_grad():
    action = actor(obs.unsqueeze(0)).squeeze(0)  # shape: (12,)
target_joint_pos = default_standing_pos + action * 0.2
```

Or using the ONNX file:

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("boulder_v6_expert_4500.onnx")

# At each 50 Hz control step:
obs = build_observation_vector()  # shape: (1, 235) float32
action = session.run(["action"], {"observation": obs})[0]  # shape: (1, 12)
target_joint_pos = default_standing_pos + action * 0.2
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Expert type | boulder_v6 |
| Terrain | Dense boulders (40% discrete 80/patch + 25% boxes + 20% rough + 10% repeated + 5% flat) |
| Base checkpoint | distilled_6899.pt (actor_only_resume) |
| Critic warmup | 300 iterations |
| Num envs | 4,096 |
| LR range | 1e-6 to 1e-4 |
| Noise range | 0.3 to 0.6 |
| Standing envs | 20% |
| Solver iterations | 8 pos / 2 vel |
| S2R hardening | Action delay 40ms, obs delay 20ms, height scan dropout 5%, IMU drift, external pushes |

### Key Reward Weights

| Reward | Weight | Notes |
|--------|--------|-------|
| base_linear_velocity | 10.0 | Adaptive (terrain-aware) |
| gait | 10.0 | Adaptive diagonal trot |
| foot_clearance | 5.0 | Adaptive (4cm flat, 25cm rough) |
| dont_wait | -5.0 | Anti-stalling penalty |
| air_time | 4.0 | Hop-enabling for boulders |
| height_gain | 1.5 | Upward body movement |
| base_roll | -7.0 | Anti-roll exploit blocker |
| terrain_relative_height | -1.0 | Adaptive body height |
| foot_slip | -0.8 | Adaptive slip penalty |
| base_motion | -0.8 | Lateral/vertical body motion penalty |
| flying_gait | -0.5 | Mild — allows hopping |
| action_smoothness | -0.4 | Adaptive (loose on rough) |

---

## Checkpoint File Structure

The `.pt` file contains:

| Key | Description |
|-----|-------------|
| `model_state_dict` | All network weights (actor + critic + std) |
| `optimizer_state_dict` | Adam optimizer state (training only) |
| `iter` | Training iteration (4500) |
| `infos` | Curriculum/training metadata |

---

## Files in This Delivery

| File | Size | Description |
|------|------|-------------|
| `boulder_v6_expert_4500.pt` | 6.6 MB | Full checkpoint (actor + critic + optimizer) |
| `boulder_v6_expert_4500.onnx` | 1.1 MB | Actor-only ONNX (opset 11, dynamic batch) |
| `POLICY_SPEC.md` | -- | This document |
