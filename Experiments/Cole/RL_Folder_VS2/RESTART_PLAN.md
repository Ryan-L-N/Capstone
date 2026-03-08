# Training Restart Plan - Learning Rate Comparison
**Date:** March 7, 2026, 06:30 AM  
**Status:** READY TO LAUNCH  
**Restart Reason:** Recovering from catastrophic policy collapse with fixed PPO implementation

---

## Executive Summary

Restarting all three training runs from **Run 3's best checkpoint (23% success)** with:
- ✅ Fixed PPO implementation (KL early stopping, multi-epoch, minibatch training)
- ✅ Three different learning rates to find optimal balance
- ✅ Same starting point for fair comparison
- ✅ All fixes from TRAINING_COLLAPSE_INCIDENT_REPORT.md deployed

**Goal:** Achieve 80% success rate on Stage 2 (25 waypoints at 5m spacing) to advance curriculum.

---

## Learning Rate Experiment Design

### Test Configuration

All runs start from **identical conditions**:
- **Checkpoint:** `checkpoints/run_3_fresh/best_model.pt` (23% success, saved Mar 6 @ 6:35 PM)
- **Stage:** Stage 2 (25 waypoints at 5m spacing)
- **PPO Safeguards:** All enabled
  - Target KL: 0.015
  - PPO Epochs: 10
  - Minibatch Size: 256
  - Max Grad Norm: 0.5

### Learning Rate Variations

| Run | Learning Rate | Strategy | Hypothesis |
|-----|---------------|----------|------------|
| **Run 1** | `1.0e-4` | **Moderate** | Balanced learning speed and stability |
| **Run 2** | `5.0e-5` | **Conservative** | Maximum stability, slowest but safest |
| **Run 3** | `1.5e-4` | **Aggressive** | Faster learning, testing upper stability bound |

### Expected Outcomes

**Run 2 (5e-5 - Conservative):**
```
Expected KL Divergence:  0.005 - 0.010 (very low)
Early Stop Frequency:    Rare (< 5% of iterations)
Learning Speed:          Slow but steady
Stability:              ★★★★★ Highest
Risk of Collapse:        Minimal
Expected Time to 80%:    Longest (maybe 500-800 iterations)
```

**Run 1 (1e-4 - Moderate):**
```
Expected KL Divergence:  0.008 - 0.014 (healthy range)
Early Stop Frequency:    Occasional (10-20% of iterations)
Learning Speed:          Balanced
Stability:              ★★★★☆ Good
Risk of Collapse:        Low with safeguards
Expected Time to 80%:    Medium (300-500 iterations)
```

**Run 3 (1.5e-4 - Aggressive):**
```
Expected KL Divergence:  0.012 - 0.015 (near threshold)
Early Stop Frequency:    Frequent (30-50% of iterations)
Learning Speed:          Fastest
Stability:              ★★★☆☆ Moderate
Risk of Collapse:        Low-Medium (safeguards should prevent)
Expected Time to 80%:    Shortest (200-400 iterations) IF stable
```

---

## Configuration Files

### nav_config_run1.yaml (Moderate - 1e-4)
```yaml
ppo:
  learning_rate: 1.0e-4  # RUN 1: Moderate learning rate for balanced learning
  clip_param: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  gamma: 0.99
  gae_lambda: 0.95
  max_grad_norm: 0.5
  target_kl: 0.015      # Early stopping threshold
  ppo_epochs: 10        # Multi-epoch training
  minibatch_size: 256   # Minibatch updates
```

### nav_config_run2.yaml (Conservative - 5e-5)
```yaml
ppo:
  learning_rate: 5.0e-5  # RUN 2: Conservative learning rate for maximum stability
  clip_param: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  gamma: 0.99
  gae_lambda: 0.95
  max_grad_norm: 0.5
  target_kl: 0.015      # Early stopping threshold
  ppo_epochs: 10        # Multi-epoch training
  minibatch_size: 256   # Minibatch updates
```

### nav_config_run3.yaml (Aggressive - 1.5e-4)
```yaml
ppo:
  learning_rate: 1.5e-4  # RUN 3: Slightly aggressive for faster learning (testing upper bound)
  clip_param: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  gamma: 0.99
  gae_lambda: 0.95
  max_grad_norm: 0.5
  target_kl: 0.015      # Early stopping threshold
  ppo_epochs: 10        # Multi-epoch training
  minibatch_size: 256   # Minibatch updates
```

---

## Launch Procedure

### Step 1: Stop Old Processes

**Old processes to terminate:**
- 5 kit processes (PIDs: 11736, 18356, 28876, 34620, 35996)
- All in collapsed/degenerate state
- Combined ~8.6 GB RAM

**Command:**
```powershell
Stop-Process -Name "kit" -Force
```

**Verification:**
```powershell
Get-Process -Name "kit" -ErrorAction SilentlyContinue
# Should return nothing
```

### Step 2: Start Training Runs

**Working Directory:** `Experiments\Cole\RL_Folder_VS2`

**Run 1 - Moderate (1e-4):**
```powershell
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Users\user\Desktop\Capstone_vs_1.2\Immersive-Modeling-and-Simulation-for-Autonomy\Experiments\Cole\RL_Folder_VS2'; C:\isaac-sim\python.bat train_navigation.py --headless --iterations 100000 --checkpoint 'checkpoints/run_3_fresh/best_model.pt' --checkpoint-dir 'checkpoints/run_1_fixed' --config 'nav_config_run1.yaml'"
```

**Run 2 - Conservative (5e-5):**
```powershell
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Users\user\Desktop\Capstone_vs_1.2\Immersive-Modeling-and-Simulation-for-Autonomy\Experiments\Cole\RL_Folder_VS2'; C:\isaac-sim\python.bat train_navigation.py --headless --iterations 100000 --checkpoint 'checkpoints/run_3_fresh/best_model.pt' --checkpoint-dir 'checkpoints/run_2_fixed' --config 'nav_config_run2.yaml'"
```

**Run 3 - Aggressive (1.5e-4):**
```powershell
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Users\user\Desktop\Capstone_vs_1.2\Immersive-Modeling-and-Simulation-for-Autonomy\Experiments\Cole\RL_Folder_VS2'; C:\isaac-sim\python.bat train_navigation.py --headless --iterations 100000 --checkpoint 'checkpoints/run_3_fresh/best_model.pt' --checkpoint-dir 'checkpoints/run_3_fixed' --config 'nav_config_run3.yaml'"
```

### Step 3: Initial Monitoring (First 10 Iterations)

**Wait 1 hour** after launch, then check:

```powershell
# Check all runs are active
Get-Process -Name "kit" | Measure-Object

# Check first 10 iterations of each run
Get-Content "Experiments\Cole\RL_Folder_VS2\checkpoints\run_1_fixed\training_log.txt" -Tail 50
Get-Content "Experiments\Cole\RL_Folder_VS2\checkpoints\run_2_fixed\training_log.txt" -Tail 50
Get-Content "Experiments\Cole\RL_Folder_VS2\checkpoints\run_3_fixed\training_log.txt" -Tail 50
```

**Verify healthy metrics:**
- ✅ KL divergence within expected ranges
- ✅ Success rate maintaining ~23% or improving
- ✅ Episodes completing (waypoints > 0)
- ✅ No immediate collapse

---

## Monitoring Plan

### Critical Metrics to Track

**Every 50 Iterations:**

| Metric | Run 1 Target | Run 2 Target | Run 3 Target | Red Flag |
|--------|--------------|--------------|--------------|----------|
| Success Rate | Improving | Improving | Improving | Dropping |
| Approx KL | 0.008-0.014 | 0.005-0.010 | 0.012-0.015 | > 0.020 |
| Early Stop % | 10-20% | < 5% | 30-50% | > 80% |
| Mean Waypoints | Increasing | Increasing | Increasing | Stuck at 0 |
| PPO Epochs | 5-10 | 8-10 | 3-7 | Always 1 |

### Success Criteria (At 100 Iterations)

**Minimum Acceptable:**
- Success rate > 25% (improved from starting 23%)
- No policy collapse
- KL divergence stable

**Good Progress:**
- Success rate > 35%
- Steady upward trend
- Appropriate KL for learning rate

**Excellent Progress:**
- Success rate > 50%
- Clear learning trajectory toward 80%
- Healthy training metrics

### Comparison Analysis (At 200 Iterations)

**Which learning rate is winning?**

Compare:
1. **Final success rate** - Primary metric
2. **Learning curve slope** - How fast improving?
3. **Stability** - Any collapse events?
4. **Iteration efficiency** - Success per iteration
5. **KL behavior** - Staying within bounds?

**Decision Point:**
- If one run clearly superior → Continue that learning rate for remaining runs
- If all stable → Continue all three to completion
- If one unstable → Stop and analyze

---

## Checkpoint Strategy

### Automatic Checkpoints
- Every 50 iterations (unchanged)
- Separate directories per run:
  - `checkpoints/run_1_fixed/`
  - `checkpoints/run_2_fixed/`
  - `checkpoints/run_3_fixed/`

### Best Model Tracking
- Automatically saved when success rate improves
- `best_model.pt` in each run's directory
- Includes: policy weights, optimizer state, iteration, stage

### Rollback Plan
If any run shows signs of collapse:
1. Stop that run immediately
2. Load most recent checkpoint (< 50 iterations ago)
3. Reduce learning rate by 50%
4. Resume with adjusted config

---

## Expected Timeline

### Phase 1: Startup & Validation (First 2 Hours)
```
Minutes 0-30:    Isaac Sim initialization (3 instances)
Minutes 30-60:   First 10 iterations (verify healthy start)
Minutes 60-120:  Next 10 iterations (confirm stability)
```

### Phase 2: Early Training (Hours 2-24)
```
Iterations 20-100:  Initial learning phase (~1 day)
Expected Progress:  23% → 35-45% success rate
Critical Period:    Watching for any collapse signs
```

### Phase 3: Mid Training (Hours 24-48)
```
Iterations 100-200: Steady improvement phase
Expected Progress:  35-45% → 55-70% success rate
Comparison Point:   Identify winning learning rate
```

### Phase 4: Final Push (Hours 48-72)
```
Iterations 200-300: Approaching threshold
Expected Progress:  55-70% → 80% (Stage 2 complete!)
Success Outcome:    Advance to Stage 3
```

**Total Expected Time:** 3-7 days depending on learning rate efficiency

---

## Known Risks & Mitigations

### Risk 1: Collapse Recurrence
**Probability:** Low (safeguards in place)  
**Impact:** High (lost training time)  
**Mitigation:** 
- KL early stopping at 0.015
- Multi-epoch + minibatch training
- Reduced learning rates
- Frequent checkpoints

### Risk 2: Learning Too Slow
**Probability:** Medium (conservative LR might be too slow)  
**Impact:** Medium (just takes longer)  
**Mitigation:**
- Testing 3 learning rates
- Can increase if all too slow
- Can stop slow runs early

### Risk 3: All Runs Same Performance
**Probability:** Medium (LR might not matter much)  
**Impact:** Low (still making progress)  
**Mitigation:**
- Still testing PPO fixes
- Pick fastest run
- Learn for next training

### Risk 4: Random Init Luck Again
**Probability:** Low (starting from same checkpoint)  
**Impact:** Medium (unfair comparison)  
**Mitigation:**
- All start from same weights
- Only LR is different
- Multiple runs reduce variance

---

## Success Definition

### Stage 2 Completion Criteria
```
Success Rate ≥ 80%  (80 out of last 100 episodes complete all 25 waypoints)
```

**When achieved:**
1. Training automatically advances to Stage 3
2. `best_model.pt` saved
3. Logs mark curriculum advancement
4. Continue training on Stage 3 configuration

### Overall Experiment Success
```
✅ At least one run reaches 80% success rate
✅ No policy collapse events
✅ KL divergence stays within bounds
✅ Identify optimal learning rate for this task
✅ Validate PPO fixes prevent future collapse
```

---

## Post-Experiment Analysis

After one run reaches 80% success (or all reach 300 iterations):

### Comparison Metrics
1. **Time to 80%** - Which LR got there fastest?
2. **Final success rate** - Which achieved highest?
3. **Stability** - Which had smoothest learning curve?
4. **Efficiency** - Success rate gain per iteration
5. **KL behavior** - Which stayed healthiest?

### Recommended Learning Rate
Based on results, document recommendation for:
- **Future Stage 2 training** - Optimal for navigation learning
- **Stage 3+ training** - May need adjustment for harder stages
- **General guideline** - For similar hierarchical RL tasks

### Lessons Learned
Document:
- What learning rate worked best and why
- How effective were PPO safeguards
- Any unexpected behaviors
- Recommendations for future training runs

---

## Rollback Information

### If Restart Fails

**Best checkpoint still available:**
- `checkpoints/run_3_fresh/best_model.pt`
- 23% success rate
- Saved: March 6, 2026 @ 6:35 PM
- Before collapse occurred

**Alternative checkpoints:**
- `checkpoints/run_1/best_model.pt` (13% success)
- `checkpoints/run_3_fresh/checkpoint_350.pt` (before peak)
- `checkpoints/run_3_fresh/checkpoint_300.pt` (early good state)

### If All Runs Collapse Again

**Action Plan:**
1. Stop all training immediately
2. Review KL divergence logs
3. Further reduce learning rates (try 1e-5, 3e-5, 5e-5)
4. Consider increasing `target_kl` to 0.020
5. Verify PPO fix implementation
6. Check for environment/system issues

---

## Contact & Documentation

**Related Documents:**
- [TRAINING_COLLAPSE_INCIDENT_REPORT.md](TRAINING_COLLAPSE_INCIDENT_REPORT.md) - Original problem analysis
- `nav_config_run1.yaml` - Run 1 configuration
- `nav_config_run2.yaml` - Run 2 configuration  
- `nav_config_run3.yaml` - Run 3 configuration
- `ppo_trainer.py` - Fixed PPO implementation
- `train_navigation.py` - Training script

**Training Logs:**
- `checkpoints/run_1_fixed/training_log.txt`
- `checkpoints/run_2_fixed/training_log.txt`
- `checkpoints/run_3_fixed/training_log.txt`

**Launched By:** AI Agent  
**Launch Date:** March 7, 2026  
**Launch Time:** ~06:40 AM  

---

**Status:** 🚀 READY TO LAUNCH - All configurations prepared, awaiting execution
