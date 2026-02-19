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

**Date:** 2026-02-18
**Phase:** Baseline
**Script:** baseline_runner.py
**Issue:** Friction not changing between surfaces
**Symptoms:** `[WARNING] PhysicsMaterial prim not found — friction unchanged.` printed for every surface change. All 7 surfaces ran at the initial asphalt_dry friction (0.75/0.65).
**Root Cause:** The original `set_ground_friction()` searched for `"PhysicsMaterial"` as a substring in prim path strings, and also tried a hardcoded set of candidate paths. None matched — the actual prim path or attribute schema name in this IsaacSim version is different from what was assumed.
**Fix:** Rewrote `set_ground_friction()` in both `baseline_runner.py` and `training_env_1.py` to use three attribute-based strategies instead of path strings: (1) `physics:staticFriction` attribute scan, (2) `UsdPhysics.MaterialAPI.HasAPI()` scan, (3) `physxMaterial:staticFriction` attribute scan. Created `find_friction_prim.py` to dump the full stage prim tree and identify the correct path/attribute name if the warning persists.
**Notes for future environments:** Always verify friction prim discovery on a fresh IsaacSim version by running `find_friction_prim.py` before starting a full run. If none of the three strategies work, hard-code the path discovered by the diagnostic into both scripts.

---

## Resolved Issues

*(Move entries here once confirmed fixed)*

---

*Last updated: —*
