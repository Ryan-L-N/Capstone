# Mason Hybrid No-Coach Policy — Specification

**Checkpoint:** `hybrid_nocoach_19999.pt` (6.6 MB)
**ONNX:** `hybrid_nocoach_19999.onnx` (actor only)
**Training:** 20,000 iterations, ~2.0B steps, 42.6 hours on H100
**Peak terrain score:** 3.74 | **Flip rate:** 0%

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

### Height Scan (indices 0–186)

| Index | Name | Dims | Description | Noise |
|-------|------|------|-------------|-------|
| 0–186 | `height_scan` | 187 | Terrain height raycaster (17x11 grid) | ±0.1 |

### Proprioceptive (indices 187–234)

| Index | Name | Dims | Description | Noise |
|-------|------|------|-------------|-------|
| 187–189 | `base_lin_vel` | 3 | Body-frame linear velocity [vx, vy, vz] (m/s) | ±0.1 |
| 190–192 | `base_ang_vel` | 3 | Body-frame angular velocity [wx, wy, wz] (rad/s) | ±0.1 |
| 193–195 | `projected_gravity` | 3 | Gravity vector in body frame [gx, gy, gz] | ±0.05 |
| 196–198 | `velocity_commands` | 3 | Target velocity [vx_cmd, vy_cmd, wz_cmd] | none |
| 199–210 | `joint_pos` | 12 | Joint positions relative to default (rad) | ±0.05 |
| 211–222 | `joint_vel` | 12 | Joint velocities relative to default (rad/s) | ±0.5 |
| 223–234 | `actions` | 12 | Previous action output (normalized) | none |

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

The 12-dimensional joint vector follows this exact order for `joint_pos`, `joint_vel`, `actions`, and action outputs:

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
| Scale | 0.2 (action × 0.2 = position offset in rad) |
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
ckpt = torch.load("hybrid_nocoach_19999.pt", map_location="cpu")
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

session = ort.InferenceSession("hybrid_nocoach_19999.onnx")

# At each 50 Hz control step:
obs = build_observation_vector()  # shape: (1, 235) float32
action = session.run(["actions"], {"observations": obs})[0]  # shape: (1, 12)
target_joint_pos = default_standing_pos + action * 0.2
```

---

## Checkpoint File Structure

The `.pt` file contains:

| Key | Description |
|-----|-------------|
| `model_state_dict` | All network weights (actor + critic + std) |
| `optimizer_state_dict` | Adam optimizer state (training only) |
| `iter` | Training iteration (19999) |
| `infos` | Curriculum/training metadata |

### Tensor listing (model_state_dict)

```
std:              [12]           — exploration noise (ignore at inference)
actor.0.weight:   [512, 235]    — input layer
actor.0.bias:     [512]
actor.2.weight:   [256, 512]    — hidden layer 1
actor.2.bias:     [256]
actor.4.weight:   [128, 256]    — hidden layer 2
actor.4.bias:     [128]
actor.6.weight:   [12, 128]     — output layer
actor.6.bias:     [12]
critic.0.weight:  [512, 235]    — (not needed for deployment)
critic.0.bias:    [512]
critic.2.weight:  [256, 512]
critic.2.bias:    [256]
critic.4.weight:  [128, 256]
critic.4.bias:    [128]
critic.6.weight:  [1, 128]
critic.6.bias:    [1]
```

---

## Files in This Delivery

| File | Size | Description |
|------|------|-------------|
| `hybrid_nocoach_19999.pt` | 6.6 MB | Full checkpoint (actor + critic + optimizer) |
| `hybrid_nocoach_19999.onnx` | ~1.1 MB | Actor-only ONNX (opset 17, dynamic batch) |
| `POLICY_SPEC.md` | — | This document |
