# Loco Policy 5 — Final Capstone Policy

**Public name:** Final Capstone Policy
**Codename history:** PARKOUR_NAV → "Project Hail Mary" → Final Capstone Policy
**Ship checkpoint:** `parkour_phasefwplus_22100.pt` (hosted in
`Experiments/Ryan/Final_Capstone_Policy_22100/`)
**Status:** SHIPPED — 22100 is the canonical Loco Policy 5 deliverable.

---

## What this policy is

The team's most-evolved locomotion policy. Originally framed as a unified
parkour + nav teacher-student build ("PARKOUR_NAV"), it became the
project's "Hail Mary" — the longest-running training pedigree (Phase 3
parkour_scratch → Phase 5 → 6 → 7 → 8 → 9 → 10 → 10b → FW-Plus → final
22100), accumulating ~22,100 iters across multiple fine-tune phases.

22100 is a teacher-only ckpt with asymmetric privileged critic (485-dim
critic obs, 235-dim actor obs). It was trained on a 50% stair-relevant
terrain mix with the value-loss watchdog (Bug #25), clamped L2 penalty
wrappers (Bug #29), and `nan_to_num` pre+post hooks for numerical
stability.

## Why 22100 is THE ship

| Capability | parkour_phasefwplus_22100 | Best of prior phases / non-Loco-5 baselines |
|---|---|---|
| Friction zone-5 | **COMPLETE 49.5m / 99.5s** project speed record | Phase-10b: 99.8s |
| Grass zone-5 | COMPLETE 49.5m / 114.7s | Phase-9: 114.8s (parity) |
| Stairs | **TIMEOUT 41.1m zone-5 ALIVE / 240s** — first 4-min wall-clock survival | Phase-9: 41.4m alive but TERMINATED early |
| Boulder | TIMEOUT 30.4m zone 4 alive (cap=0.67) | Phase-9/10/10b: parity ~31m zone 4 |
| 4-env falls | 0 across all 4 environments | — |
| Cole quarter | untested locally (env import broke during eval; documented gap) | Phase-5 11000: 25/25 quarter, Phase-9: 25/25 quarter |
| Cole max-density | untested | Phase-5 10000: 21/25 (project record) |

22100 holds the 4-env crown by every metric, is project-record on
friction speed, and is the first ckpt to survive a 4-minute wall-clock
stair eval. Cole is an open question — Phase-5 11000 is the documented
Cole record holder and remains a fallback for deployments where Cole
is the primary deployment scene.

## Why we shipped 22100 (not a fresh re-train)

5 from-scratch / resume attempts on Apr 29, 2026 all hit walls.
Detailed post-mortem in `docs/SHIP_DECISION.md`. Convergent finding:
today's training is unable to escape the level-0 reward-hack equilibrium
under the current cmd_vel range + curriculum threshold.

Phase-3 parkour_scratch (Apr 23-24) achieved terrain_levels 3.5→2.2 in
iter 6 — Apr 29's 5 attempts couldn't pass terrain_levels 0.005 in
1000+ iters. The contrast is stark and the regression hasn't been
diagnosed.

The pragmatic call: **22100 is project-record across 4-env, ship it
and accept the open Cole question.**

---

## Layout

```
Loco_Policy_5_Final_Capstone_Policy/
├── README.md                                    — this file
├── docs/
│   ├── design.md                               — original 3-week design doc (scaffold era)
│   ├── FINAL_CAPSTONE_POLICY_EXPLAINED.md      — plain-English Hail Mary walkthrough
│   └── SHIP_DECISION.md                        — Apr 29 final ship rationale
├── configs/
│   ├── __init__.py                             — gym.register Isaac-Velocity-FinalCapstonePolicy-Spot-v0
│   ├── final_capstone_policy_env_cfg.py        — FinalCapstonePolicyEnvCfg + asymmetric obs
│   ├── final_capstone_policy_agent_cfg.py      — FinalCapstonePolicyPPORunnerCfg
│   └── final_capstone_policy_terrain_cfg.py    — FINAL_CAPSTONE_POLICY_TERRAINS_CFG
├── terrains/
│   ├── open_riser_straight.py                  — solid-riser straight flight (FW-realistic)
│   ├── open_riser_switchback.py                — solid-riser two-flight switchback
│   └── open_riser_stairs.py                    — pyramid-topology open-riser variant (legacy)
├── modules/
│   └── privileged_obs.py                       — friction, mass, foot-contact for asymm critic
├── scripts/
│   ├── train.py                                — entry point (cosine LR + value-loss watchdog + clamped DR)
│   └── eval.py                                 — wraps 4_env_test + Cole skill_nav_lite with --action_scale 0.3
└── checkpoints/
    └── README.md                               — where 22100 actually lives + how to load it
```

## How to use 22100 (eval)

```bash
# 4-env battery (3 seeds × 3 eps headless)
python Locomotion_Codebases/Loco_Policy_5_Final_Capstone_Policy/scripts/eval.py \
    --target 4_env --headless --num_episodes 3 --seeds 42,123,7 \
    --checkpoint Experiments/Ryan/Final_Capstone_Policy_22100/parkour_phasefwplus_22100.pt

# Cole quarter-density
python Locomotion_Codebases/Loco_Policy_5_Final_Capstone_Policy/scripts/eval.py \
    --target cole --headless --num_episodes 3 --seeds 42,123,7 --cole_density quarter \
    --checkpoint Experiments/Ryan/Final_Capstone_Policy_22100/parkour_phasefwplus_22100.pt
```

The eval launcher bakes in `--action_scale 0.3`, wires the right
`--mason` flag for the 4-env eval, and passes the correct Cole APF
parameters per the `skill_nav_lite_integration` recipe.

For production hardware deployment, use the ONNX export:
`Experiments/Ryan/Final_Capstone_Policy_22100/parkour_phasefwplus_22100.onnx`
(verified max-diff < 1e-5 vs PyTorch).

## How to deploy 22100 (Spot SDK)

See `Experiments/Ryan/Final_Capstone_Policy_22100/POLICY_DETAILS.md`
for the full spec sheet:
- 235-dim observation (48 proprio + 187 height-scan)
- 12-DOF action, type-grouped (4 hx, 4 hy, 4 kn)
- action_scale = 0.3
- PD gains Kp=60, Kd=1.5
- Control rate 50 Hz (decimation=10 at 500 Hz physics)
- Pseudocode for inference is in POLICY_DETAILS.md

## Imports + path setup

`scripts/train.py` and `scripts/eval.py` add four paths to `sys.path` at
module load:

1. `Loco_Policy_5_Final_Capstone_Policy/` — for `from configs...`,
   `from terrains...`, `from modules...`
2. `Loco_Policy_3_Student_Teacher_Training/` — for
   `from wrappers.progressive_s2r import ProgressiveS2RWrapper`
3. `Loco_Policy_2_ARL_Hybrid/configs/` — for
   `from arl_hybrid_env_cfg import SpotARLHybridEnvCfg` (the inherited
   parent config)
4. `Loco_Shared/` — for `from quadruped_locomotion.utils...`

The eval launcher additionally walks one level up to the repo root and
joins `Experiments/Alex/NAV_ALEX/scripts/cole_arena_skillnav_lite.py`
for the Cole-arena Skill-Nav-Lite eval (NAV_ALEX stays under
Experiments/Alex/ — it's a navigation codebase, not part of the
locomotion-codebases wrapper).

## Inheritance chain

```
Loco_Policy_2 ARL Hybrid: SpotARLHybridEnvCfg
    └── Loco_Policy_3 S2R: SpotS2RBaseEnvCfg
            └── (24 expert subclasses)
                    └── Loco_Policy_5: FinalCapstonePolicyEnvCfg  ← THIS POLICY
```

Changes to ARL Hybrid (Loco_2) or S2R Base (Loco_3) propagate here. The
Loco_Policy_5 env_cfg adds:
- Asymmetric `critic` ObservationGroup with privileged terms
  (terrain_height_grid, friction, added_mass, foot_contact_forces)
- Parkour-DR ranges (friction 0.3-2.0, mass 0-3kg, motor ±20%)
- 3D command randomization (vx, vy, ωz) with command-curriculum hooks
- action_scale = 0.3 (matches 22100 spec sheet)

## Known limitations

1. **FW USD stairs (open-riser geometry)** — 22100 stays alive on the
   4 SM_Staircase USDs but doesn't successfully climb them. Root cause:
   the USDs ship with tread-only collision (no riser faces). Fix is
   geometry-side, in progress at
   `Experiments/Colby/FW_Stairs_Riser_Project/`. Once Colby's modified
   USDs land, 22100 should climb them without retraining (the procedural
   stair training already covers solid-riser geometry).

2. **Cole density extrapolation** — 22100 is untested on Cole locally
   (env import broke during today's eval; documented as known gap).
   Phase-5 11000 is the documented Cole-quarter / Cole-max record
   holder (25/25 quarter, 21/25 max). For Cole-heavy deployments,
   prefer phase5_11000 + the unnerfed nav recipe in the Cole handoff.

3. **No student distillation** — Phase-2 student distill never ran.
   22100 is the asymmetric-critic teacher with privileged obs. For
   deployment, the actor-only ONNX is what gets shipped; the privileged
   critic is only used during training and is not part of the
   deliverable.

## Don't re-train this without reading docs/SHIP_DECISION.md

5 attempts on Apr 29, 2026 hit a stuck-at-level-0 reward-hack equilibrium
that was not present in the original Phase-3 parkour_scratch run. The
diagnostic hasn't been done — could be (a) a regression in the terrain
mix between Apr 23 and Apr 29, (b) the new defense-stack edits changing
gradient flow, or (c) something else. Until that's diagnosed, any new
fine-tune from 22100 should:

- Use `lr_max=3e-5` with the compound watchdog (Bug #25 + #29 stack)
- Resume actor-only with `--critic_warmup_iters 200`
- Use the cleaned terrain mix (no experimental open-riser sub-terrains)
- Cap iter count at 1000 first to avoid burning H100 hours on a stuck run
- Pre-commit kill criteria: terrain_levels < 0.3 by iter 500 → kill

If those don't get past the level-0 wall, accept that 22100 is the
ship and move to the next deliverable.
