# Phase-FW-Plus 22100 — Test Delivery for Ryan

**Checkpoint:** `parkour_phasefwplus_22100.pt` (8.4 MB)
**ONNX:** `parkour_phasefwplus_22100.onnx` (1.09 MB, actor only — verified max-diff 6.68e-06)
**Training:** Phase-FW-Plus, 22,100 iters total (resumed from `parkour_phase9_18500.pt`,
              ran 3,600 new iters on Phase-FW-Plus terrain before late-stage collapse;
              this is the LAST GOOD ckpt before iter-3700 cascade).

---

## Why this ckpt for testing

This is the **strongest 4-env ckpt of the project** plus partial-progress on FW USD stairs:

| Metric | Phase-FW-Plus 22100 | Best of prior phases |
|---|---|---|
| Friction | **COMPLETE 49.5m z5 / 99.5s** ⚡ | Phase-10: 99.8s |
| Grass | COMPLETE 49.5m z5 / 114.7s | Phase-9: 114.8s |
| Boulder (cap=0.67) | FLIP 31.6m z4 | Phase-9/10/10b ~31.5m |
| **Stairs** | **TIMEOUT 41.1m z5 ALIVE / 240s** | Phase-9: alive z5 41.4m (terminated) |

**Project firsts:**
- Sub-100s friction zone-5 traversal (NEW PROJECT SPEED RECORD)
- Stair zone-5 alive survival across full 4-min wall-clock cap

**Known limitation (the reason a follow-up training is happening):**
FW SM_Staircase USDs (architectural open-riser stairs) — Spot **stays alive**
on them but doesn't climb. Phase-FW-Plus's open-riser sub-terrain training
got Spot from "falls into gap" → "stuck at bottom but stable." The next
training phase tackles this via from-scratch with FW-realistic geometry
baked in from iter 0.

---

## ⚠ Differences from boulder_v6 — READ if deploying

Same as Phase-9's POLICY_DETAILS:

| Parameter | boulder_v6 | **Phase-FW-Plus 22100** |
|---|---|---|
| `action_scale` | 0.2 | **0.3** |
| `lin_vel_x` cmd range (training) | (-2, 3) m/s | (-1.0, 1.8) m/s |
| `lin_vel_y` cmd range | (-1.5, 1.5) | (-0.8, 0.8) |
| `ang_vel_z` cmd range | (-2, 2) | (-2.0, 2.0) |
| Critic obs dim (training only) | 235 | 485 (asymmetric) |
| Backward walking | not trained | YES (vx floor -1.0) |

Everything else (obs layout, joint ordering, default pose, PD gains, control
rate) is identical to boulder_v6 — drop-in compatible.

---

## Network Architecture

Same MLP as boulder_v6 / Phase-9:

```
Input (235) -> Linear(235, 512) -> ELU
            -> Linear(512, 256) -> ELU
            -> Linear(256, 128) -> ELU
            -> Linear(128, 12) -> Output (12)
```

286,604 actor parameters. Critic is 485 → 512 → 256 → 128 → 1 (asymmetric,
training-only — NOT in ONNX).

---

## Observation + Joint Spec (identical to boulder_v6 / Phase-9)

**Observation order (235 dims):**
- 0–186: height_scan (17×11 grid raycaster, fill 0.0 for flat ground, range [-1, 1])
- 187–234: proprioceptive (base_lin_vel, base_ang_vel, projected_gravity,
           velocity_commands, joint_pos, joint_vel, last_actions)

**Joint order (12 DOF) — TYPE-GROUPED:**
```
[fl_hx, fr_hx, hl_hx, hr_hx,    # all hx
 fl_hy, fr_hy, hl_hy, hr_hy,    # all hy
 fl_kn, fr_kn, hl_kn, hr_kn]    # all kn
```

**Default standing pose (radians):**
```python
[0.1, -0.1, 0.1, -0.1,
 0.9,  0.9, 1.1,  1.1,
-1.5, -1.5, -1.5, -1.5]
```

**Spawn height:** z = 0.55 m

**PD gains:** Kp=60, Kd=1.5
**Control rate:** 50 Hz (decimation=10 at 500 Hz physics)

---

## Inference Pseudocode

```python
import torch, torch.nn as nn

actor = nn.Sequential(
    nn.Linear(235, 512), nn.ELU(),
    nn.Linear(512, 256), nn.ELU(),
    nn.Linear(256, 128), nn.ELU(),
    nn.Linear(128, 12),
)
ckpt = torch.load("parkour_phasefwplus_22100.pt", map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"]
actor_sd = {k.replace("actor.", ""): v for k, v in sd.items() if k.startswith("actor.")}
actor.load_state_dict(actor_sd); actor.eval()

# At each 50 Hz control step:
obs = build_observation_vector()  # shape: (235,)
with torch.no_grad():
    action = actor(obs.unsqueeze(0)).squeeze(0)  # shape: (12,)
target_joint_pos = default_standing_pos + action * 0.3   # NOTE 0.3
```

---

## Recommended boulder operating point

`--zone_slowdown_cap 0.67` past x>20m (for the 4-env eval; equivalent
deployment knob = cap forward velocity at 0.67 m/s when on rough boulder
zones). Phase-9/Phase-FW-Plus boulder gait wants slower than 1.0 m/s on
high-density rocks.

---

## What we'd love to know from your testing

1. **Does friction speed transfer to your eval framework?** (99.5s sim time
   on the 4-env arena is a project record — does it look as fast in your
   setup?)
2. **Stair zone-5 alive depth** — does your stair eval also show survival
   past 41m at the gait/riser limit?
3. **Cole quarter-density** — untested by us on this ckpt. Phase-9 was good
   here (Phase-5 was 25/25 record). Curious if Phase-FW-Plus regresses Cole.
4. **Any FW USD progress** — Phase-9 sometimes drops feet into gaps and
   falls. Phase-FW-Plus stays alive but stalls. If yours shows different
   behavior on FW USDs, please flag.

---

## Files

| File | Size | Description |
|------|------|-------------|
| `parkour_phasefwplus_22100.pt` | 8.4 MB | Full ckpt (actor + critic + std) |
| `parkour_phasefwplus_22100.onnx` | 1.09 MB | Actor-only ONNX (opset 11, dynamic batch) |
| `POLICY_DETAILS.md` | — | This document |

---

## Caveats

This is a **TEST delivery** — not the canonical ship ckpt. The successor
training run (from-scratch with open-riser baked in from iter 0) is in
progress. If Phase-FW-Plus 22100 evaluates poorly in your setup, please
report — we'll send the from-scratch ckpt when it lands.

For comparison: `Experiments/Cole/Final_Capstone_Policy_handoff/` has Phase-5 / 8 /
9 ckpts with the same docs format if you want to A/B against earlier
phases.

Ping Alex / Gabriel with what you find.
