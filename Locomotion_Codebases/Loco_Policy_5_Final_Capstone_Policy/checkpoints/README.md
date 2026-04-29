# Loco Policy 5 — Final Capstone Policy Checkpoints

This directory is intentionally empty in the main-branch deliverable —
binary `.pt` and `.onnx` files are gitignored to keep the source-only
deliverable clean.

## Ship checkpoint

**`parkour_phasefwplus_22100.pt`** — the canonical Loco Policy 5
production policy. Lives at:

| Location | Purpose |
|---|---|
| `Experiments/Ryan/Final_Capstone_Policy_22100/parkour_phasefwplus_22100.pt` | The .pt (8.4 MB) |
| `Experiments/Ryan/Final_Capstone_Policy_22100/parkour_phasefwplus_22100.onnx` | Actor-only ONNX (1.09 MB, opset 11, max-diff < 6.7e-6 vs PyTorch) for hardware deployment |
| `Experiments/Ryan/Final_Capstone_Policy_22100/POLICY_DETAILS.md` | Full spec sheet (obs layout, joint order, PD gains, inference pseudocode) |
| `Experiments/Cole/Final_Capstone_Policy_handoff/` | Mirror for Cole (NAV_ALEX integration) — has the .pt, code wrappers, and `parkour_phase3_7000.pt` for Cole quarter-density baseline |
| H100 `~/Locomotion_Codebases/Loco_Policy_5_Final_Capstone_Policy/logs/rsl_rl/spot_final_capstone_policy/model_22100.pt` | Source training output (gitignored) |
| H100 `~/PARKOUR_NAV/logs/rsl_rl/spot_parkour_nav/model_22100.pt` | Pre-rename source path (preserved on H100 for backwards compat) |

## Loading 22100

```python
import torch
import torch.nn as nn

# Construct the actor MLP
actor = nn.Sequential(
    nn.Linear(235, 512), nn.ELU(),
    nn.Linear(512, 256), nn.ELU(),
    nn.Linear(256, 128), nn.ELU(),
    nn.Linear(128, 12),
)

# Load weights
ckpt = torch.load(
    "Experiments/Ryan/Final_Capstone_Policy_22100/parkour_phasefwplus_22100.pt",
    map_location="cpu",
    weights_only=False,
)
sd = ckpt["model_state_dict"]
actor_sd = {
    k.replace("actor.", ""): v
    for k, v in sd.items()
    if k.startswith("actor.")
}
actor.load_state_dict(actor_sd)
actor.eval()

# At each 50 Hz control step:
obs = build_observation_vector()                 # shape (235,)
with torch.no_grad():
    action = actor(obs.unsqueeze(0)).squeeze(0)  # shape (12,)
target_joint_pos = default_standing_pos + action * 0.3  # NOTE: action_scale=0.3
```

**Default standing pose (radians, type-grouped):**
```
[0.1, -0.1,  0.1, -0.1,    # all 4 hx (hip-x)
 0.9,  0.9,  1.1,  1.1,    # all 4 hy (hip-y)
-1.5, -1.5, -1.5, -1.5]    # all 4 kn (knee)
```

PD gains: Kp=60, Kd=1.5. Control rate: 50 Hz (decimation=10 at 500 Hz physics).
Spawn height: z = 0.55 m.

## Other checkpoints worth keeping

These are NOT shipped but are useful for ablation / fallback:

| Checkpoint | Why kept |
|---|---|
| `parkour_phase5_11000.pt` (in `Experiments/Cole/Final_Capstone_Policy_handoff/`) | Cole-quarter (25/25) and Cole-max-density (21/25) record holder. Use for Cole-heavy deployments. |
| `parkour_phase8_16497.pt` | Zero-fall 4-env baseline (alternate to 22100, less stair-aggressive). |
| `parkour_phase9_18500.pt` | Stair zone-5 alive specialist (alternate to 22100; flipped where 22100 survived). |
| `parkour_phase3_7000.pt` (in Cole handoff) | Pre-FW-Plus reference for nav-only deployments. |

## Why we shipped 22100 instead of training a fresh ckpt

See parent README + `../docs/SHIP_DECISION.md`. TL;DR: 5 attempts on
Apr 29 to either fine-tune 22100 or train fresh on improved terrain
all hit a stuck-at-level-0 reward-hack equilibrium that wasn't present
in the original Phase-3 parkour_scratch baseline (Apr 23-24). Until
that regression is diagnosed, 22100 holds project-record numbers and
ships as-is.
