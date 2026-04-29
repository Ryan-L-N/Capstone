# Lessons Learned — Training Environment 1

Record issues, debugging sessions, and resolved problems here as they occur during simulation runs.

**Format:** Date → Phase → Issue → Root Cause → Fix → Notes for future environments

---

## Template (copy this block for each new entry)

**Date:**
**Phase:** *(Baseline / Phase 1 / Phase 2 / Phase 3 / Phase 4)*
**Script:** *(baseline_runner.py / training_env_1.py / other)*
**Issue:**
**Symptoms:**
**Root Cause:**
**Fix:**
**Notes for future environments:**

---

## Open Issues

*(None currently)*

---

## Resolved Issues

**Date:** 2026-02-18
**Phase:** Baseline
**Script:** baseline_runner.py
**Issue:** Friction not changing between surfaces
**Symptoms:** `[WARNING] PhysicsMaterial prim not found — friction unchanged.` printed for every surface change. All 7 surfaces ran at the initial asphalt_dry friction (0.75/0.65).
**Root Cause:** The original `set_ground_friction()` searched for `"PhysicsMaterial"` as a substring in prim path strings, and also tried a hardcoded set of candidate paths. None matched — the actual prim path or attribute schema name in this IsaacSim version is different from what was assumed.
**Fix:** Rewrote `set_ground_friction()` in both `baseline_runner.py` and `training_env_1.py` to use three attribute-based strategies instead of path strings: (1) `physics:staticFriction` attribute scan, (2) `UsdPhysics.MaterialAPI.HasAPI()` scan, (3) `physxMaterial:staticFriction` attribute scan. Created `find_friction_prim.py` to dump the full stage prim tree and identify the correct path/attribute name if the warning persists.
**Status (2026-02-18):** Baseline completed and generated `logs/friction_baseline_summary.csv` with per-surface data. Moving to Resolved. **However:** slip ratios and body angles are nearly identical across the full μ range (0.07–0.75), which is consistent with either (a) the fix working and the stock policy being friction-agnostic, or (b) friction still not changing. Run `find_friction_prim.py --headless` before Phase 1 training to confirm one of the three strategies succeeds and prints the matched prim path. If no prim is found, hard-code the path.
**Notes for future environments:** Always verify friction prim discovery on a fresh IsaacSim version by running `find_friction_prim.py` before starting a full run. If none of the three strategies work, hard-code the path discovered by the diagnostic into both scripts.

---

**Date:** 2026-02-18
**Phase:** Baseline
**Script:** baseline_runner.py
**Issue:** All baseline episodes terminate via out-of-bounds, not timeout
**Symptoms:** 100% of episodes end with `out_of_bounds=1`; no falls recorded on any surface.
**Root Cause:** The fixed command `[vx=1.5, vy=0, yaw=0]` produces a small but consistent lateral drift in the stock policy. Over ~60 s the robot drifts past the ±15 m lateral boundary.
**Fix:** Not a bug — expected behaviour of a stock open-loop policy with no lateral correction. Noted for training design: the trained Locomotion Command Policy should learn to correct lateral drift (issue non-zero vy commands) rather than purely maximise forward speed.
**Notes for future environments:** Consider widening the arena or increasing LATERAL_LIMIT during baseline-only runs if longer episodes are needed. For training, keep the limit tight to penalise lateral drift as a learned objective.

---

*Last updated: 2026-02-18*
