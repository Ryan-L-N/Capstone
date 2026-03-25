# Run 1 Restart Log - Learning Rate Experiment

**Date:** March 11, 2026  
**Time:** ~6:00 AM  
**Reason:** Permanent plateau detected in Stage 5

---

## Problem Identification

Run 1 exhibited a **permanent plateau** in Stage 5 (20m→40m waypoint spacing):

- **Duration:** 65+ hours in Stage 5, 35+ hours at plateau
- **Success Rate:** 69-70% (unchanging)
- **Variance:** ±1% (extremely stable - policy locked)
- **Distance from Goal:** 21% below 90% advancement threshold
- **Probability of Natural Recovery:** <5%

**Verdict:** Policy trapped in local optimum. Continuing wastes compute resources.

---

## Learning Rate Analysis

### Current Training Results (All Starting from Stage 2):

| Run | Learning Rate | Stage 5 Status | Success Rate | Duration | Outcome |
|-----|--------------|----------------|--------------|----------|---------|
| Run 2 | **5e-5** (0.00005) | ✅ **COMPLETED** | 81% (exit) | 24.5 hours | **SUCCESS** |
| Run 1 | **1e-4** (0.0001) | ❌ Plateaued | 69% (stuck) | 65+ hours | **FAIL** |
| Run 3 | **1.5e-4** (0.00015) | ❌ Plateaued | 71% (stuck) | 71+ hours | **FAIL** |

### Learning Rate Landscape:

```
5e-5  ────────────────────── ✅ SUCCESS (proven)
         [UNEXPLORED GAP]
7.5e-5 ───────────────────── 🔬 NEW EXPERIMENT
         [UNEXPLORED GAP]
1e-4  ────────────────────── ❌ FAIL (permanent plateau)
         [EXPLORED]
1.5e-4 ───────────────────── ❌ FAIL (extended plateau)
```

---

## Decision: Restart with LR=7.5e-5

### Why 7.5e-5?

**Position:** Exactly halfway between success (5e-5) and failure (1e-4)

**Scientific Value:**
- ✅ Explores the critical gap between success/failure threshold
- ✅ Identifies the exact learning rate boundary for Stage 5 generalization
- ✅ Potentially faster than Run 2 (24.5h → ~18-22h estimated)
- ✅ Provides valuable data for future curriculum stages

**Expected Outcomes:**
- **Best Case:** Complete Stage 5 in ~18-22 hours (faster than Run 2)
- **Moderate Case:** Plateau at 75-85% SR (better than original Run 1)
- **Worst Case:** Plateau at similar 69-70% (confirms 1e-4 threshold)

**Success Probability:** ~70% (moderate confidence)

---

## Implementation Details

### Configuration Change:
- **File:** `nav_config_run1.yaml`
- **Old Value:** `learning_rate: 1.0e-4`
- **New Value:** `learning_rate: 7.5e-5`
- **Comment Updated:** "RUN 1: Sweet spot experiment - halfway between success/failure"

### Restart Procedure:
1. Stop current Run 1 training process ✅
2. Keep existing checkpoint directory: `checkpoints/run_1_fixed_v5`
3. Archive current training log (already at 6326 lines)
4. Restart from **Stage 4 completion checkpoint**: `run_1_fixed_v5/stage_4_complete.pt`
5. Continue to iteration 100000 (same as other runs)

### Starting Conditions:
- **Checkpoint:** `run_1_fixed_v5/stage_4_complete.pt` ⭐ **SKIP STAGES 2-4**
- **Starting Stage:** Stage 5 (20m→40m waypoint spacing) - **THE CRITICAL STAGE**
- **Starting Iteration:** ~565 (where original Run 1 entered Stage 5)
- **Starting Performance:** ~92% SR at Stage 4 completion (original Run 1)
- **Environments:** 1024 parallel robots
- **Time Saved:** ~23 hours by skipping Stages 2-4
- **Other Hyperparameters:** Identical to original Run 1 (entropy, gamma, clip, etc.)

---

## Monitoring Plan

### Success Indicators (Stage 5):
- ✅ **Initial entry SR >90%** (like Runs 2 & 3, unlike original Run 1's 0-50%)
- ✅ **Controlled descent pattern:** 100% → 95% → 85% → 81%+ (Run 2 trajectory)
- ✅ **Completion time <25 hours** in Stage 5
- ✅ **Advancement to Stage 6** with 90%+ SR

### Potential Failure Signs:
- ❌ Poor entry to Stage 5 (<70% initial SR)
- ❌ Rapid descent: 100% → <75% in first 10 hours
- ❌ Plateau formation at 75-80% lasting >24 hours
- ❌ Variance reducing to ±1% (policy locking)

### Checkpoints:
- **Hour 0:** Starting Stage 5 fresh (from stage_4_complete.pt)
- **Hour 12:** Should be progressing through Stage 5 (75-85% SR)
- **Hour 24:** Should be completing Stage 5 or entering Stage 6
- **Hour 36:** If still in Stage 5, re-evaluate strategy

---

## Expected Timeline

**Starting from Stage 4 completion checkpoint:**
- **Stages 2-4:** ⏩ **SKIPPED** (saves ~23 hours)
- **Stage 5:** ~18-22 hours (20m→40m waypoints) ← **KEY STAGE**
- **Total to Stage 6:** ~18-22 hours from restart

**Comparison:**
- Run 2 (5e-5): 24.5 hours in Stage 5, completed successfully
- Run 1 New (7.5e-5): **~18-22 hours estimated** (if successful)
- Run 1 Old (1e-4): 65+ hours in Stage 5, never completed (permanent plateau)

---

## Hypothesis

**Primary Hypothesis:**  
Stage 5's 40m waypoint spacing requires a learning rate ≤7.5e-5 for successful generalization. Higher learning rates cause aggressive policy updates that overshoot the optimal trajectory through the challenging 20m→40m transition.

**Alternative Hypothesis:**  
The success threshold is exactly at 5e-5, and any increase (including 7.5e-5) will result in plateau formation. This experiment will confirm or refute this.

**Null Hypothesis:**  
Learning rate has no effect on Stage 5 completion, and Run 1's failure was due to other factors (random seed, initialization, environment dynamics).

---

## Success Criteria

### Experiment Success:
- Run 1 completes Stage 5 with ≥90% SR
- Total Stage 5 duration <30 hours
- Advances to Stage 6 without catastrophic forgetting

### Experiment Failure:
- Run 1 plateaus at <85% SR for >24 hours
- Exhibits ±1% variance indicating policy lock
- Cannot advance within 50 hours

### Data Value:
Regardless of outcome, this experiment provides critical data on the learning rate sensitivity of Stage 5's generalization challenge. If 7.5e-5 fails, we know the threshold is between 5e-5 and 7.5e-5. If it succeeds, we can test higher rates (8e-5, 9e-5) for optimal training speed.

---

## Post-Experiment Analysis (To Be Completed)

### Questions to Answer:
1. Did 7.5e-5 complete Stage 5? (Yes/No)
2. How did Stage 5 entry SR compare? (vs 0-50% original, vs 100% Runs 2&3)
3. What was the Stage 5 duration? (vs 24.5h Run 2, vs 65+h original)
4. What was the exit SR from Stage 5? (target: 81-90%)
5. Did it advance to Stage 6 successfully? (vs Run 2's collapse)

### Learning Rate Boundary Refinement:
- If **SUCCESS**: Test 8e-5 or 8.5e-5 in future runs (find upper limit)
- If **FAIL**: Test 6e-5 or 6.5e-5 in future runs (find lower limit)
- Goal: Identify optimal LR that maximizes speed while ensuring completion

---

## Notes

- This is a **controlled experiment** with only one variable changed (LR)
- All other conditions identical to original Run 1
- Starting from same checkpoint ensures fair comparison
- Run 2 and Run 3 continue uninterrupted for parallel monitoring
- Run 2 (Stage 6, recovering from collapse) provides cautionary data on obstacle introduction
- Run 3 (Stage 5, 71% plateau) provides comparison for alternative plateau behavior

**Restart initiated:** March 11, 2026, ~6:00 AM
