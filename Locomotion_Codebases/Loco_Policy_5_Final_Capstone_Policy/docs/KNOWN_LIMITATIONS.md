# Known Limitations — `parkour_phasefwplus_22100.pt`

**As-shipped state, May 1 2026.** This document catalogues what the
canonical Final Capstone Policy can and cannot do, based on the 100-ep
canonical eval and the rendered teleop tests.

## Where 22100 is strong

| Capability | Evidence |
|---|---|
| Friction zone-5 ice traversal | 96/100 COMPLETE on canonical eval, project speed record (best 99.5s sim time) |
| Grass zone-5 traversal | 75/100 COMPLETE with **0 falls** across all 100 episodes |
| Procedural pyramid stair climb | trained on `MeshPyramidStairsTerrainCfg` step heights 5–42cm at three width variants (25/30/40cm) |
| Wide DR robustness | trained at friction (0.3, 2.0) static / (0.2, 1.8) dynamic, mass 0–3kg, push 8s @ ±0.6 m/s |
| Backward locomotion | trained with `lin_vel_x = (-1.0, 1.5)` — backward at 1.0 m/s confirmed |

## Limitation 1 — Boulder zone-3 wedge

**Symptom:** 0/100 boulder COMPLETE in canonical eval. **0 falls** —
robot stays upright but cannot push past the dense boulder field
starting at zone 3 (20m).

| Outcome | Count | Max reach |
|---|---|---|
| Wedged at 20–25m (z3 entry) | 38 |
| Wedged at 25–30m (mid z3) | 3 |
| Broke through to z4 (~30m) | 9 | 31.4m |
| Reached z5 / completed | 0 | — |

**Diagnosis:** the canonical `--zone_slowdown_cap 0.67` for boulder is
the empirically-tuned cap that prevented z3 falls during training, but
it also caps the policy's authority to push through dense obstacles.
At rendered scale Phase-9 reached zone 4 alive at 41.4m on the
*procedural* boulder field; the canonical eval's HfDiscreteObstacles
distribution at full density is harder.

**Workaround for production:**
- The 22100 policy at `--zone_slowdown_cap 1.0` (no slowdown) might
  push through more reliably at the cost of higher fall rate
- Cole's nav stack with global A* + APF deviator (per Apr 19 quarter-
  density 19.7/25) routes around the worst boulder concentrations
  rather than plowing through

**Where to dig:** post-ship TODO 0b in `SHIP_DECISION.md` —
boulder-zone-aware reward shaping + tiered velocity capping
(`--zone3_cap` / `--zone4_cap` split). Requires fine-tune which is
currently blocked by the level-0 trap (see Limitation 4).

## Limitation 2 — Stairs zone 2–3 fall rate

**Symptom:** 0/100 stairs COMPLETE on canonical eval. **51 FELL** in
zone 2 (~18m) and **49 TIMEOUT** in zone 3 (~25m). Roughly half-and-half.

| Outcome | Count | Median time | Max reach |
|---|---|---|---|
| FELL in zone 2 | 51 | ~70s | 18m |
| TIMEOUT in zone 3 | 49 | 240s | 25m |
| Reached z4+ | 0 | — | — |

**Diagnosis:** the eval-arena stairs are 17–23cm risers at zones 2–5
(per `Locomotion_Codebases/4_env_test/src/configs/zone_params.py`).
22100 was trained on procedural pyramid stairs with risers 5–42cm —
inside that range. But the eval-arena stairs are *step-up walls* with
explicit risers, while procedural pyramid is a single Z-up pyramid.
Different geometric topology means the trained policy can climb the
training-distribution stairs (per `Episode_Reward/terrain_relative_height`
~0 on training) but flips on the slightly different eval-arena
geometry.

**Workaround for production:** `--zone_slowdown_cap 1.0` is the
canonical cap. Lower caps (e.g., 0.5) trade reach for stability.

**Where to dig:** post-ship TODO 0c in `SHIP_DECISION.md` — stair-zone-
aware reward shaping with a "land both forefeet on tread" bonus and
foot-drop-into-riser-gap penalty. Requires fine-tune (see Limitation 4).

## Limitation 3 — FW USD staircase non-engagement

**Symptom:** rendered teleop on `SM_Staircase_02` with Colby's
risered USDs: 22100 walks AROUND the staircase (`+14.6m -Y` drift,
`fell=False`, final z=0.51 = spawn height). Phase-9 (18500), 20850
(pre-ship), and 9600 (peak terrain) all the same. 23499 (collapsed
checkpoint) doesn't respond to commands at all.

**Diagnosis:** the FW staircase is **50° slope, 5.3m total rise** —
*outside* 22100's training distribution. The training pyramid stairs
were 30–35° at most. When the policy's height-scan rays detect a
geometry too steep for what it learned, it picks "go around" over
"climb".

The training reward stack does not penalize lateral bypass:
`terrain_out_of_bounds.distance_buffer = 3.0m` allows up to 3m of
sideways motion before terminating. That's plenty of room to skirt
past a 4.6m-deep stair footprint.

**Workaround for production: GEOMETRIC SOFTENING.** Scale the X-run on
each `SM_Staircase_*.usd` by ~1.7× — drops slope from 50° to ~35°,
which is *inside* 22100's trained distribution. Single Xform op per
USD. Colby's `add_risers.py` (committed Apr 30) already solved the
"falls through tread gaps" half of the problem; geometric softening
solves the "won't engage" half.

**Where to dig:**
- `Experiments/Colby/FW_Stairs_Riser_Project/RISER_INTEGRATION_PLAN.md`
  has the full integration plan
- Single-line USD Xform edit: `xformOp:scale = (1.7, 1.0, 1.0)` on each
  staircase root prim (multiplied with the existing scale 0.01)
- Re-run `bake_stair_collision.py` after the scale change to refresh
  CollisionAPI

## Limitation 4 — Cannot fine-tune from 22100 (level-0 trap)

**Symptom:** 8 consecutive retrains from 22100 (5 Apr 29 + rev1, rev2,
v3) all collapse at the "stuck-at-level-0 reward hack" failure mode.
Symptoms identical: terrain_levels demotes to 0.0, body_flip ~80%,
mean_reward 7–20 (drift-forward equilibrium without curriculum
promotion).

**Diagnosis:** the regression is deeper than any single config edit.
See `SHIP_DECISION.md` "Final verdict" section for the full ablation —
the trap is robust to resume-vs-scratch, original-vs-rebalanced reward
weights, tighter-vs-original `terrain_out_of_bounds`, tighter-vs-
original `_STAIR_RISER_RANGE`, and curriculum proportion changes.

**Implication:** any production improvement to 22100 (boulder push,
stair fall reduction, FW stair climb) that requires retraining is
**currently blocked**. The project ships 22100 + Colby's geometric
softening as the final answer.

**Where to dig:** `FUTURE_WORK.md` regression diagnosis section —
identify what changed between Apr 27 (Phase-FW-Plus succeeded) and
Apr 29 (everything broke).

## Limitation 5 — Action_scale 0.3 inferences only

22100 was trained at `action_scale=0.3`. Eval and teleop scripts must
pass `--action_scale 0.3`. Other action scales degrade performance:

- 0.2 (Mason baseline default): policy under-actuates, walks but slowly
- 0.5: policy over-actuates, oscillates and falls

The default in `SpotRoughTerrainPolicy.__init__()` is set to mason's
0.2 unless `mason_baseline=True` is passed AND `action_scale_override`
is provided. **Always pass `--action_scale 0.3` explicitly.**

## Limitation 6 — Asymmetric-critic teacher; no student distillation

The shipped checkpoint is the **teacher** with privileged critic
observations (true friction, mass, foot contacts). Deployment uses the
actor-only ONNX (`parkour_phasefwplus_22100.onnx`) which strips the
critic. No student distillation was performed.

**Implication:** The student/proprio-only deployment path is not
available in the current ship. Real hardware deployment would benefit
from a distilled student that doesn't need privileged observations
(see `SHIP_DECISION.md` post-ship TODO #4).

**Where to dig:** the Phase-2 distillation pipeline at
`Loco_Policy_3_Student_Teacher_Training/scripts/train_distill_s2r.py` —
expects a teacher checkpoint. 22100 is compatible.
