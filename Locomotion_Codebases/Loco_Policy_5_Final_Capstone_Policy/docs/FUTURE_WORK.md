# Future Work — Final Capstone Policy follow-up agenda

**As of May 1 2026**, 22100 is shipped and the active project work
closes. This document captures the research agenda for whoever picks
this up next — the loose ends that 22100 leaves on the table and the
concrete experiments that would close them.

Companion docs:
- `SHIP_DECISION.md` — what shipped and why
- `KNOWN_LIMITATIONS.md` — what 22100 can't do (boulder z3, stair fall,
  FW USD non-engagement, fine-tune-from-22100 collapse)
- `RISER_INTEGRATION_PLAN.md` — Colby's stair geometry plan + Apr 30
  results addendum

## Priority 1 — Geometric softening on FW staircase USDs

**Why:** the cleanest path to closing the FW USD stair limitation
(KNOWN_LIMITATIONS Limitation 3). 22100 climbs procedural pyramid
stairs at ~30–35° slope. The FW USDs are 50°. Soften the USDs and
22100 should walk up them out of the box.

**Effort:** half a day on Colby's side. No GPU work. No risk to the
shipped policy.

**Steps:**
1. For each `SM_Staircase_*.usd` in `Experiments/Colby/FW_Stairs_Riser_Project/usd_source/`:
   - Add `xformOp:scale = (1.7, 1.0, 1.0)` on the staircase root prim
     (multiplied with the existing scale 0.01)
   - Re-run `bake_stair_collision.py` to refresh CollisionAPI on the
     scaled mesh
2. Verify with `verify_fw_stair_layout_v2.py --rendered`:
   - bbox should now be ~7.7m × 1.99m × 5.32m (was 4.57 × 1.99 × 5.32)
   - slope should compute to ~35° (rise=5.32m, run=7.77m → 34.4°)
3. Re-run rendered teleop `run_fw_stairs.py --teleop --stairs 02
   --usd_root <softened-path>` — drive Spot up. If it climbs to
   z>4m within 30s of W input, the geometric softening worked.
4. If climbing works, propagate the scaled USDs to
   `Collected_Final_World/SubUSDs/` (the deployment scene path).

**Acceptance criterion:** 2/4 SM_Staircase USDs are climbable by 22100
on rendered teleop with W-only input.

## Priority 2 — Diagnose the level-0 trap regression

**Why:** if the project ever revisits a retrain (boulder push, stair
fall reduction, FW USD distribution training), the level-0 trap MUST
be solved first. 8 consecutive failed retrains is conclusive evidence
that something in the pipeline broke between Apr 27 (Phase-FW-Plus
succeeded) and Apr 29 (everything broke). Until that's identified,
no fine-tune from 22100 is viable.

**Effort:** 1–2 days of forensic git diff + smoke tests.

**Suspects (untested):**
- **Isaac Lab version drift.** Check `git log -p` on the Isaac Lab
  install path between the dates. Was there an Apr 27–29 update?
  `terrain_levels_vel` source code may have shifted threshold logic.
- **`terrain_levels_vel` curriculum threshold tuning.** The promote
  criterion is `mean_distance > 0.5 × cmd_vel × episode_length`. If
  the threshold or the distance metric changed, level-0 escape
  becomes harder. Trace the curriculum mdp function across Apr 27–29.
- **Cmd_vel resampling.** The command range was tightened from
  `(-2.0, 3.0)` (Cheng baseline) to `(-1.0, 1.5)` somewhere in the
  Phase-9 → Phase-FW-Plus transition. If the resampling time changed
  or the curriculum interaction broke, that could explain why the
  policy never demonstrates "fast enough" walking to satisfy the
  curriculum.
- **Privileged-obs pipeline.** `S2RObservationsCfg` may have changed
  what the critic sees. If critic baseline shifted, value function
  could be miscalibrated for level-0 → level-1 promotion.
- **Value-loss watchdog interaction.** The watchdog (Bug #25) halves
  LR on `vf_loss > 100`. If it fires spuriously during normal
  curriculum exploration in early training, it freezes learning.
  Check `phase_v3_final_collapsed.log` for watchdog activations
  during the 450–550 iter crash.

**Concrete experiment that would close this:**
1. `git log --since=2026-04-27 --until=2026-04-30 --
   Locomotion_Codebases/ Experiments/` — full diff of all training-
   relevant edits in the regression window.
2. For each suspect edit, revert it on a fresh from-scratch run and
   see if the level-0 trap clears. Stop at the first revert that
   produces healthy curriculum advancement (terrain_levels reaches
   3.0+ within 1000 iters).
3. Document the regression in `RAWDOG.md` (the bug compendium).

## Priority 3 — Boulder zone-3 push (post-fine-tune)

**Why:** KNOWN_LIMITATIONS Limitation 1 — 0/100 boulder COMPLETE on
canonical eval. Once Priority 2 is solved, this is the high-leverage
fine-tune target.

**Plan (gated on Priority 2):**
- Fine-tune from 22100 with boulder-zone-aware reward shaping:
  - Bonus for reaching ≥31m on the boulder arena
  - Mild penalty for stalling in zones 1–2 (which currently see no
    falls but the reward shouldn't reward standing still)
- Implement tiered velocity capping in `run_capstone_eval.py`:
  - `--zone3_cap` / `--zone4_cap` split (currently one global cap
    applied at x≥20m)
  - Full command in zones 1–2, capped at 0.67–0.8 m/s in zones 3–4
- Or: add a boulder-specific terrain variant that has zone-2
  densities at the start (skip zone 1) to force speed-up training

**Acceptance criterion:** ≥80% of 100-ep boulder eval reaches 31m
(1m past zone 3 entry).

## Priority 4 — Stair fall rate reduction (post-fine-tune)

**Why:** KNOWN_LIMITATIONS Limitation 2 — 51 FELL out of 100 in zones
2–3. Halves of the eval-arena stair distribution are unstable.

**Plan (gated on Priority 2):**
- Fine-tune from 22100 with stair-zone-aware reward shaping:
  - Bonus for landing both forefeet on a tread surface
  - Penalty for foot-drop into riser-gap (anti-stair-fall)
  - Preserve the existing `terrain_relative_height` anti-belly-crawl
    penalty
- Use `--zone_slowdown_cap 1.5` for runtime: full 2.0 m/s in zones
  1–2, capped to 1.5 m/s past x≥20m for zones 3–5

**Acceptance criterion:** ≥40% of 100-ep stair eval COMPLETE,
≤20% FELL.

## Priority 5 — Friction zone speed push (post-fine-tune)

**Why:** SHIP_DECISION TODO 0a — 22100 hits 5/5 friction zone 5
COMPLETE but the early zones are slow (best 117s = 0.42 m/s avg). The
project record `Final_Capstone_Policy_22100/POLICY_DETAILS.md` says
zone 5 alone in 99.5s.

**Plan (gated on Priority 2):**
- Widen `lin_vel_x` max 1.5 → 3.0 m/s in
  `final_capstone_policy_env_cfg.py:__post_init__()`
- Fine-tune from 22100 (actor-only resume + critic warmup 200) with
  Bug #25/#29 defense stack engaged
- Verify `--zone_slowdown_cap` (capped at 0.8–1.0 m/s on x≥30m for
  zones 4–5) prevents slip-induced flips on wet ice / oil

**Acceptance criterion:** Sustained 2.0 m/s through zones 1–2,
sustained 2.5 m/s for the full 49.5m as a stretch goal.

## Priority 6 — Phase-2 student distillation

**Why:** KNOWN_LIMITATIONS Limitation 6 — the shipped checkpoint is
asymmetric-critic teacher. Real hardware deployment without privileged
observations (true friction, mass, foot contacts) would benefit from a
distilled proprio-only student.

**Plan:**
- BC + DAGGER pipeline at
  `Loco_Policy_3_Student_Teacher_Training/scripts/train_distill_s2r.py`
- Teacher = 22100, student = 235-dim non-asymmetric
- Target: student matches teacher within 5% on canonical 4-env eval
- ONNX export of the student actor for deployment

**Effort:** 2–3 days H100 + 1 day eval validation.

## Priority 7 — FW USDs as actual training terrain

**Why:** if geometric softening (Priority 1) is rejected for any
reason (deployment scene must keep the original 50° slope), the only
remaining path is training the policy ON the FW USDs directly.

**Plan:**
- Implement `MeshUSDStaticTerrainCfg` that bakes a USD into trimesh
  via `pxr.Usd` at startup and serves it as a sub-terrain. Per the
  May 1 Explore agent finding, Isaac Lab has no native USD-loading
  for terrains — this is custom code.
- Effort estimate: 4–8h for the custom sub-terrain class + integration
- Add to `final_capstone_policy_terrain_cfg.py` at low proportion
  (5–10%) — diversity in training, not dominant
- Curriculum mapping: `SM_StaircaseHalf_02` (small) → level 0,
  `SM_StaircaseHalf_01` (wide) → level 3, `SM_Staircase_02`
  (straight 5.3m) → level 6, `SM_Staircase_01` (full switchback)
  → level 9
- Gated on Priority 2 (level-0 trap fix)

**Acceptance criterion:** 2/4 SM_Staircase USDs climbable by the
new policy on rendered teleop, no regression on 4-env baseline.

## Priority 8 — Real-hardware sim-to-real transfer

**Why:** the project's premise is RL on real Spot. Sim-to-real wasn't
attempted because the policy is asymmetric-critic + privileged obs.

**Plan (gated on Priority 6):**
- Distilled student from Priority 6 deploys to real Spot via
  `Experiments/Alex/ARL_DELIVERY/04_Teleop_System/spot_teleop.py` (the
  Xbox controller teleop).
- Per `Locomotion_Codebases/HOW_TO_RUN_4ENV_EVAL.md`, the deployment
  pipeline is mostly written; just needs the proprio-only student and
  the ONNX export.

**Risk:** real Spot has different inertia, joint friction, and contact
dynamics than the Isaac Sim spot. The wide DR (friction 0.3–2.0,
mass 0–3kg) was meant to cover this, but real-world testing is the
only way to validate.

## Priority 9 — Cross-policy ensemble for boulder + stair

**Why:** different checkpoints have different strengths.
`parkour_phase4_9400.pt` reaches z4 boulder where 22100 wedges at z3.
22100 has friction + grass robustness 9400 lacks. An ensemble (or
mixture-of-experts gate) could pick the best policy per environment.

**Plan:**
- Ship-time policy selector: detect environment from height_scan
  signature, dispatch to the best policy for that env
- Lightweight: a small classifier head reads the obs and outputs a
  weight over (22100, 9400, ...) experts
- Trained end-to-end on the canonical 4-env eval

**Effort:** 3–5 days research + experimentation. Speculative.

## Notes for whoever picks this up

- **Don't try to fine-tune from 22100 until Priority 2 is solved.**
  You'll lose 12–24h of H100 time per attempt to the level-0 trap.
- **The SHIP_DECISION.md "open follow-up work" section** is the older
  roadmap; this doc supersedes it for items 1–5 and adds 6–9.
- **Eval data** at `Experiments/Ryan/22100 Final Eval 100/` has the
  100-ep canonical eval JSONLs + SUMMARY.md for any analysis.
- **Branch lineage** for the 8 failed retrains:
  - Apr 29 attempts: live in `parkour_hailmary*` checkpoint names
  - rev1 + rev2 + v3: live on branches `phase-fw-plus-2` and
    `phase-v3-from-scratch` for archival
- **The H100 has the full training history.** All checkpoints from
  iter 0 to iter 23499 are at
  `~/PARKOUR_NAV/logs/rsl_rl/spot_parkour_nav/model_*.pt`. The
  collapsed Phase-v3 logs are at `~/phase_v3_final_collapsed.log` and
  `~/phase_fw_plus_2_collapsed.log` (rev1) on H100.
