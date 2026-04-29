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
