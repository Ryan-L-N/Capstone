# Ship Decision — `parkour_phasefwplus_22100.pt`

**Date:** Apr 29, 2026
**Decision:** Ship `parkour_phasefwplus_22100.pt` as the canonical Loco
Policy 5 single-canonical deliverable. No further re-training before
ship.

## Context

The Apr 29 session opened with an in-flight training run
(`parkour_phase_final_b`) on H100 that was 67% through 8000 iters with
`vf_loss = 7.6e20` (mean reward swinging -23 to -617). That run was
killed and 5 follow-on attempts were made to either:

(a) train a fresh policy from scratch on the cleaned terrain, OR
(b) fine-tune from `parkour_phasefwplus_22100.pt` with a corrected
    defense stack and conservative LR.

All 5 attempts hit walls. The convergent finding pointed at a regression
in the training pipeline that wasn't present in the project's original
successful runs. Pragmatic call: 22100 holds project-record numbers,
ship it, and treat the regression as a follow-up investigation.

## The 5 attempts

### Attempt 1 — `parkour_hailmary`

| Property | Value |
|---|---|
| Approach | from-scratch, experimental open-riser-no-walls sub-terrains |
| `lr_max` | 3e-4 |
| `action_scale` | 0.3 |
| Iters reached | 371 |
| Outcome | stuck-at-level-0 reward hack |
| Symptoms | reward +68 plateau, terrain_levels 0.005, body_flip 6%, error_vel_xy 2.4, time_out 94% |
| Root cause | the new solid-riser-no-walls open-riser sub-terrains produced bimodal returns (walks vs side-drift); curriculum demoted everything to level 0 |

### Attempt 2 — `parkour_hailmary2`

| Property | Value |
|---|---|
| Approach | resume from 22100 (actor-only) + critic_warmup_iters=100 |
| `lr_max` | 3e-4 (same as Phase-FW-Plus) |
| Iters reached | 105 (collapsed at iter 101, 6 iters past warmup) |
| Outcome | exponential vf_loss explosion at unfreeze: 1.6 → 5.2e5 → 1.3e14 → 4e22 → 6e30 |
| Root cause | actor-only resume with a fresh asymmetric critic always produces a brittle unfreeze transient; lr=3e-4 was too high for the first PPO update post-warmup |

### Attempt 3 — `parkour_hailmary3`

| Property | Value |
|---|---|
| Approach | resume + warmup, with the watchdog upgraded to compound-multiplicative LR halving + lr_max=3e-5 (10× smaller) + critic_warmup_iters=200 |
| Iters reached | 122 |
| Outcome | chronic critic oscillation during warmup (15+ vf_loss spikes between iter 86-122, ranging 100-3000), watchdog penalty hit floor at 0.001 |
| Root cause | open-riser sub-terrains produced too much V_target variance for the critic to fit at any LR |

### Attempt 4 — `parkour_hailmary4`

| Property | Value |
|---|---|
| Approach | same as #3 but with the experimental open-riser sub-terrains REMOVED (9% reallocated to flat 6% + flat_clutter 2% + pyramid_stairs_medium 1%) |
| Iters reached | 115 |
| Outcome | warmup phase clean (0 spikes!), then exponential explosion 9 iters after unfreeze: 24K → 1.1e7 → 2.9e9 → 6.7e11 → ... → 5.7e18 |
| Root cause | even with cleaned terrain, the actor-only-resume + asymmetric-critic combo produces a fundamentally unstable unfreeze transient. The compound watchdog couldn't halve LR fast enough to damp the runaway. |

### Attempt 5 — `parkour_hailmary5`

| Property | Value |
|---|---|
| Approach | full from-scratch (no resume) on the cleaned terrain (same mix as #4) with Phase-3 scratch params (lr_max=3e-4, max_noise_std=0.5) |
| Iters reached | 1059 |
| Outcome | stuck-at-level-0 reward hack identical to #1 (different terrain, same equilibrium) |
| Symptoms at iter 1000 | reward 95.96, terrain_levels 0.000, body_flip 2.4%, error_vel_xy 2.38, time_out 97.6% |
| Per the original kill criteria: iter 500 terrain_levels < 0.3 → kill (was 0.000) |
| Root cause | the policy found a "drift forward, don't flip" equilibrium that earns the lin_vel reward on flat ground without satisfying the curriculum's distance threshold |

## Convergent finding

Across 5 attempts spanning resume vs from-scratch, narrow vs wide LR,
experimental vs cleaned terrain — every fresh training pipeline today
hit one of two failure modes:

1. **Stuck-at-level-0 reward hack** — the curriculum can't promote any
   environment past level 0 because the policy learns a flat-ground
   drift equilibrium that doesn't satisfy the distance threshold. The
   reward looks healthy (90-100s range), the body_flip rate is excellent
   (~2%), but the policy never tackles harder terrain.

2. **Unfreeze runaway from actor-only-resume** — when resuming from
   22100's well-tuned actor into a fresh asymmetric critic, the first
   PPO update post-warmup destabilizes the value function exponentially.
   Compound LR halving + lower LR delays the explosion but doesn't
   prevent it.

The original Phase-3 parkour_scratch run (Apr 23-24) DID NOT hit failure
mode 1 — it advanced to terrain_levels 3.5→2.2 within 6 iters and
eventually trained to project records. The contrast is stark and the
diagnostic hasn't been done. Suspected causes (untested):

- a regression in `final_capstone_policy_terrain_cfg.py` between Apr 23
  and Apr 29 that nudged the curriculum threshold
- one of the Apr 29 reward-stack edits (action_smoothness weight,
  joint_torques weight, command range tweaks) producing a different
  reward landscape than Phase-3 trained on
- the now-active value-loss watchdog interacting with PPO update
  dynamics in an unintended way
- a change in Isaac Lab / Isaac Sim version since Apr 23

## Why ship 22100 anyway

| Capability | parkour_phasefwplus_22100 result |
|---|---|
| Friction zone-5 | **COMPLETE 49.5m / 99.5s** — project speed record |
| Grass zone-5 | COMPLETE 49.5m / 114.7s |
| Stairs zone-5 | **41.1m ALIVE / 240s** — first 4-min wall-clock survival in project history |
| Boulder zone-4 | TIMEOUT 30.4m alive |
| 4-env falls | 0 across all 4 environments |

These numbers are project-best by every metric we measure. Re-training
under the regression conditions of Apr 29 risks producing a worse
policy. Ship 22100, document the regression, move on.

## What's NOT shipping

The Phase-2 student distillation never ran. Loco Policy 5 ships as the
asymmetric-critic teacher checkpoint. For deployment, the actor-only
ONNX is what runs on hardware — the privileged critic is training-only
and not part of the deliverable.

## Open follow-up work (post-ship)

0. **TODO (a): Push friction speed — 2.0 m/s through zone 2 + reach
   ≥31m (1m past zone 3).** Friction zones are 10m each (zone 1: 0-10m
   = 60-grit sandpaper, zone 2: 10-20m = dry rubber, zone 3: 20-30m =
   wet concrete, zone 4: 30-40m = wet ice, zone 5: 40-50m = oil on
   steel). Local rendered eval (Apr 29) showed 22100 hitting 5/5
   COMPLETE on friction zone 5 with times 117–174s (best 117.2s =
   ~0.42 m/s average across the full 49.5m). The completion is great
   but the early zones are slow.
   Target: **sustained 2.0 m/s through zones 1-2 (0-20m)** and
   **reach ≥31m at the boundary into zone 4** (1m past zone 3) —
   stretch goal is sustained 2.5 m/s for the full 49.5m.
   To get there:
   - Widen the training command range: `lin_vel_x` max 1.5 → 3.0 m/s
     in `final_capstone_policy_env_cfg.py` `__post_init__`.
   - Fine-tune from 22100 (actor-only resume + critic warmup 200) at
     the wider range, with the Bug #25/#29 defense stack engaged.
     `lr_max=3e-5`, 1000-iter cap, kill if vf_loss > 100 sustained.
   - Verify the policy can sustain 2.0 m/s on zones 1-2 without
     flipping at zones 4-5 (low friction). May need to retain the
     `--zone_slowdown_cap` flag for the icy zones (capped at 0.8-1.0
     m/s on x≥30m to prevent slip-induced flips on wet ice / oil).
   - Re-eval 4-env battery + Cole quarter to confirm the speed bump
     doesn't regress stair-zone-5 alive depth or Cole performance.

0c. **TODO (c): Push stairs to 2.0 m/s through zone 2 + reach ≥31m
   (1m past zone 3).** Stair zones are 10m each with riser heights
   ramping by zone (zone 1: 3cm baby steps, zone 2: 7-10cm, zone 3:
   ~17cm — historically the V14-V19 project wall, zone 4: ~20cm,
   zone 5: 23cm max). Project record on stair-heavy terrain
   (Phase-FW-Plus 22100): zone-5 ALIVE TIMEOUT 41.1m / 240s. On the
   standard 4-env stairs eval (lower-density stair config), reaching
   ≥31m is an open question — needs the local rendered smoke result
   that's still running.
   Target: **sustained 2.0 m/s through zones 1-2** (3-10cm baby risers
   should be fast), **slow to 1.5 m/s past zone 2** (zones 3+ where
   risers ramp 17cm → 23cm), and **reach ≥31m at the boundary into
   zone 4.**
   To get there:
   - Same recipe as friction/boulder: widen `lin_vel_x` cmd range
     1.5 → 3.0 m/s, fine-tune from 22100 with Bug #25/#29 stack at
     `lr_max=3e-5`.
   - Use the `--zone_slowdown_cap 1.5` flag for the runtime config:
     full 2.0 m/s in zones 1-2, capped to 1.5 m/s past x≥20m for
     zones 3-5 (handles the 17-23cm risers without tipping). Memory's
     Phase-9 boulder lock was 0.67 m/s — stairs is faster than boulder
     because the geometry is more predictable (steady riser cadence
     vs. random rocks), so 1.5 should be sustainable.
   - Add stair-zone-aware reward shaping during the fine-tune: bonus
     for landing both forefeet on a tread surface (anti-foot-drop
     into riser-gap), preserve the existing terrain_relative_height
     anti-belly-crawl penalty.

0b. **TODO (b): Push boulder to 2.0 m/s through zone 2 + 1m past zone 3.**
   Boulder zones are 10m each (zone 1: 0-10m, zone 2: 10-20m, zone 3:
   20-30m, zone 4: 30-40m). Project record (Phase-9 22100 with
   `--zone_slowdown_cap 0.67`): FELL 31.5m zone 4 / 111s. Target:
   **2.0 m/s sustained through zones 1-2** (faster traversal of the
   easy boulder densities) and **reach ≥31m (1m past zone 3)** in zone
   4 reliably (not just intermittently).
   To get there:
   - Implement tiered velocity capping in `run_capstone_eval.py`: full
     command in zones 1-2, cap at 0.67-0.8 m/s in zones 3-4. Currently
     `--zone_slowdown_cap` is one global cap applied at x≥20m; needs
     splitting into `--zone3_cap` / `--zone4_cap` or similar.
   - Fine-tune from 22100 with boulder-zone-aware reward shaping: bonus
     for reaching ≥31m on boulder, mild penalty for falling in zones
     1-2 (which currently don't see falls but the reward shouldn't
     reward standing still in easy zones either).
   - Or: add a boulder-specific terrain variant that has zone-2
     densities at the start (skip zone 1) to force speed-up training.
   - Re-eval to confirm zones 1-2 hit 2.0 m/s avg + zone-3-to-31m
     traversal rate ≥80% across 100-ep battery.

1. **Diagnostic: why is Apr 29 stuck at level 0?** Compare commits
   on `final_capstone_policy_terrain_cfg.py` and
   `final_capstone_policy_env_cfg.py` between
   `parkour_phase3_7000` (Apr 23-24, the last clean from-scratch
   training) and HEAD. The git diff should surface the regression.

2. **FW USD stair eval** with Colby's modified (riser-baked) USDs.
   Current 22100 stays alive on the 4 SM_Staircase USDs but doesn't
   climb them — root cause is geometry-side (tread-only collision),
   not policy. Once Colby's modified USDs land, run
   `Experiments/Colby/FW_Stairs_Riser_Project/scripts/run_fw_stair_eval.py`
   with 22100 to verify the geometry fix. Expected: 0/4 → 2-4/4 PASS.

3. **Cole quarter + max retest** for 22100. Today's local Cole eval
   crashed at startup due to a `nav_locomotion.modules.grid_astar_planner`
   import error (local PYTHONPATH issue, not a 22100 issue). Re-attempt
   on H100 where the path is set up correctly to fill the documented gap.

4. **Phase-2 student distillation** if the team wants a proprio-only
   deployment ckpt. Resume from 22100 with the BC + DAGGER pipeline at
   `Loco_Policy_3_Student_Teacher_Training/scripts/train_distill_s2r.py`,
   targeting a 235-dim non-asymmetric student.

## Apr 30 / May 1 update — eval data + FW stair findings + Phase-FW-Plus-2 retrain

### Canonical 4-env 100-ep eval (the ship matrix)

Headless run on H100, 100 episodes per arena, completed Apr 30 17:44 UTC
(~23h wall). JSONL dump for Ryan: `Experiments/Ryan/22100 Final Eval 100/`.

| Env | n | COMPLETE | FELL | TIMEOUT | Median time | Max reach |
|---|---|---|---|---|---|---|
| friction | 100 | **96 (96%)** | 3 | 1 | 365s | 49.5m (96 z5) |
| grass | 100 | **75 (75%)** | 0 | 25 | 315s | 49.5m (93 z5) |
| boulder | 100 | 0 (0%) | **0** | 100 | — | 31.4m (z3 wedge) |
| stairs | 100 | 0 (0%) | **51** | 49 | — | 25.0m (z2-z3) |

Friction + grass are ship-quality. Boulder stays alive (0 falls) but
wedges at the dense zone-3 boulder field. Stairs splits roughly 50/50
between fall-in-zone-2 and timeout-in-zone-3.

### FW stair USD findings (Colby's risers + 22100)

Colby shipped `add_risers.py` (development branch, commit `fbdf7d2`) that
bakes solid riser triangles into the SM_Staircase_* USDs by detecting
horizontal tread faces by normal and inserting vertical riser quads at
the front edge of each upper tread. Real Path-A riser geometry, not a
buried-ramp cheat.

Tested 4 policies on rendered teleop against Colby's risered
`SM_Staircase_02` USD:

| Policy | Behavior | fell |
|---|---|---|
| `parkour_phase9_18500` (stair specialist) | bypasses on +Y/-Y side | False |
| `parkour_phasefwplus_22100` (ship) | bypasses, ends at (-11.24, +14.6, 0.51) | False |
| `parkour_phasefwplus_20850` (pre-ship) | similar bypass | False |
| `parkour_phasefwplus_23499` (over-train) | **mode collapse**, no command response | — |

**Diagnosis: behavioral, not capability.** All three intact policies
treat the FW staircase as an obstacle to navigate around rather than
terrain to climb. The 9600 ckpt (pulled from H100 for diagnostic) has
peak terrain_level (5.99) and reaches z4 on procedural boulders but
**FLIPS at spawn** on friction (DR was narrower at iter 9600). Confirms
22100 was the right ship — it traded raw climbing for DR robustness.

The training reward did not penalize bypass: `terrain_out_of_bounds`
fired only at 3.0m drift, so 0-3m of sideways-around-the-stair was
inside the reward envelope.

Updates to TODO #2 (originally "expected 2-4/4 PASS with Colby's risers"):
**0/4 PASS observed**. Colby's risers solve falls (`fell=False` in all
tests) but the policy still doesn't climb. The fix is **policy-side**
(retrain to engage), not geometry-side. Phase-FW-Plus-2 below.

### Phase-FW-Plus-2 retrain (in progress, May 1)

Launched H100 screen `phase_fw_plus_2` on May 1 00:36 UTC. +2000 iters
from `parkour_phasefwplus_22100.pt` → target iter 24100.

**Two changes** (committed on `phase-fw-plus-2` branch, `9e8e161`):

1. `final_capstone_policy_terrain_cfg.py` — narrow-tread stair
   variants bumped (`pyramid_stairs_narrow` 4%→12%, `hf_stairs_narrow`
   4%→8%). Net stair proportion 39%→42%, narrow-tread (FW match) 8%→20%.
   `_STAIR_RISER_RANGE` tightened (0.05, 0.42)→(0.10, 0.25) to
   concentrate curriculum on FW-realistic 0.10-0.20m band.

2. `base_s2r_env_cfg.py` — `terrain_out_of_bounds.distance_buffer`
   3.0m → 1.5m. Forces engage-or-fail: enough margin for normal step
   correction (~2× tread width), insufficient for full bypass.

No reward weight changes (V19 destabilization warning preserved).

**H100 sync:** SCP'd `parkour_nav_terrain_cfg.py` directly + sed-patched
`base_s2r_env_cfg.py` (avoided full-file SCP because the post-reorg file
imports `arl_hybrid_env_cfg` which doesn't exist on H100's legacy
layout). Backups at `*.pre_fw_plus_2.bak`.

**Initial metrics (iter 22114, ~14 iters past resume):**
- vf_loss = 0.75 (watchdog floor 100, well clear)
- noise_std = 0.30 (lower bound, healthy)
- terrain_levels = 3.46 (near 22100's 3.67 baseline)
- Mean reward = 87.6 (vs 22100's 168.45 — expected drop from tighter
  termination; should recover as policy adapts)
- body_flip_over = 12.8% (elevated; will track as kill-switch metric)

**Pre-committed kill switches:**
- friction COMPLETE rate <80% on smoke-eval gate → abort
- vf_loss spikes >5× in any 100 iters → abort (mode collapse signature)
- mean_reward drops >50% from baseline 168 → abort

### rev1 — COLLAPSED (May 1 01:35 UTC, killed at iter 22452)

The two changes combined were **stronger together than modeled**.
Reward kill-switch fired in <400 iters:

| Metric | iter 22114 | iter 22452 | Baseline | Direction |
|---|---|---|---|---|
| Mean reward | 87.6 | 16-20 | 168 | -90% (KILL) |
| terrain_levels | 3.46 | 0.0012 | 3.67 | floored |
| body_flip_over | 12.8% | 81.6% | — | 6× regression |
| vf_loss | 0.75 | 0.80 | — | technically fine |
| noise_std | 0.30 | 0.30 | — | floor (not collapsing) |

Failure mode = "Stuck-at-level-0 reward hack" (exact match to the Apr 29
attempts above). Combined effect:
- `_STAIR_RISER_RANGE` (0.05, 0.42)→(0.10, 0.25) made level 0 stairs 10cm
  minimum risers (was 5cm). Curriculum had **no easy mode to demote to**.
- `distance_buffer` 3.0→1.5 terminated stair drift quickly.
- → Too many flips → curriculum demoted → still hard at level 0 →
  equilibrium at "flip in place" at terrain_levels=0.001.

vf_loss stayed below watchdog threshold (0.80) ONLY because the policy
was sitting in a near-zero-reward equilibrium — not actively trying
hard things. The watchdog wouldn't catch this; only the reward-drop kill
switch did.

H100 collapsed log preserved: `~/phase_fw_plus_2_collapsed.log`. The
22200/22300/22400 checkpoints captured the descent into the level-0
trap; useful as case-study artifacts but not deployable.

### rev2 — narrow-tread proportion bump only (May 1 01:34 UTC, in progress)

Reverted both the riser range tightening AND the distance_buffer change.
Kept ONLY the additive narrow-tread proportion bump — pure exposure
increase, no harder geometry, no tighter termination:

| Change | rev1 | rev2 |
|---|---|---|
| `pyramid_stairs_narrow` 4% → 12% | KEPT | **KEPT** |
| `hf_stairs_narrow` 4% → 8% | KEPT | **KEPT** |
| `_STAIR_RISER_RANGE` | (0.10, 0.25) | **(0.05, 0.42)** reverted |
| `distance_buffer` | 1.5 | **3.0** reverted |

Hypothesis: pure additive exposure to FW-shaped stairs (without making
level 0 harder or terminating drift faster) won't trigger the level-0
demotion trap. Tests whether more reps on narrow-tread stairs alone is
enough to shift the policy's "go around" preference toward "go up".

Branch state on `origin/phase-fw-plus-2`:
- `9e8e161` - original Phase-FW-Plus-2 changes (collapsed)
- `42a7910` - documentation update
- `48de017` - rev2 revert (currently training)

If rev2 also collapses, fall back to geometric softening on Colby's USDs.

If candidate beats 22100 on FW stair engagement (Spot z >1.5m within
30s of W input on rendered teleop) without regressing 4-env baseline
>5%, promote as `parkour_phasefwplus2_NNNN.pt`. Otherwise revert to 22100.

---

## What lives where (post-ship pointer map)

| Artifact | Path |
|---|---|
| Ship .pt | `Experiments/Ryan/Final_Capstone_Policy_22100/parkour_phasefwplus_22100.pt` |
| Ship ONNX | `Experiments/Ryan/Final_Capstone_Policy_22100/parkour_phasefwplus_22100.onnx` |
| Spec sheet | `Experiments/Ryan/Final_Capstone_Policy_22100/POLICY_DETAILS.md` |
| Cole handoff | `Experiments/Cole/Final_Capstone_Policy_handoff/` |
| Source training code | `Locomotion_Codebases/Loco_Policy_5_Final_Capstone_Policy/` |
| FW stair eval framework | `Experiments/Colby/FW_Stairs_Riser_Project/` |
| Top-level deliverable mirror | `Final Policies/Locomotion Policies/` (Ryan's existing main-branch ship dir) |
| Project narrative | `Locomotion_Codebases/Loco_Policy_5_Final_Capstone_Policy/docs/FINAL_CAPSTONE_POLICY_EXPLAINED.md` |
| RAWDOG bug compendium | `Locomotion_Codebases/docs/HOW_TO_TRAIN_YOUR_RAWDOG.md` |
| **100-ep canonical eval data** (Apr 30) | `Experiments/Ryan/22100 Final Eval 100/` (4 JSONLs + SUMMARY.md) |
| **Colby's FW riser-baked USDs** (Apr 30) | `Experiments/Colby/FW_Stairs_Riser_Project/usd_source/SM_Staircase_*.usd` (commit `fbdf7d2` on `development`) |
| **Phase-FW-Plus-2 retrain branch** | `origin/phase-fw-plus-2` (commit `9e8e161`) — narrow-tread bump + tighter termination |
| H100 in-flight training screen | `phase_fw_plus_2` on `t2user@172.24.254.24` (May 1 00:36 UTC, ETA iter 24100 ~12h) |
