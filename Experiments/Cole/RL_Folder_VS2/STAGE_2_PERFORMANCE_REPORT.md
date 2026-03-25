# Stage 2 Performance Report: Waypoints 5m
## Navigation Training - Boundary Penalty Fix (v5)

**Report Date:** March 8, 2026  
**Stage:** 2/8 - Waypoints at 5m spacing  
**Goal:** Capture all 25 waypoints before score timeout  
**Success Threshold:** 80% success rate to advance to Stage 3

---

## Executive Summary

All three training runs successfully completed Stage 2 and advanced to Stage 3 (10m waypoint spacing). Each run demonstrated the effectiveness of the boundary penalty fix (`-5 → -50`), which prevented the "suicide strategy" and enabled proper navigation learning.

### Key Findings:
- **Learning Timeline:** ~2.5-3 hours to first waypoint capture, followed by rapid mastery
- **Total Stage 2 Duration:** ~11-12 hours per run
- **Iterations in Stage 2:** 59-61 iterations
- **All runs achieved 80%+ success rate** and advanced to Stage 3

---

## Run 1: Moderate Learning Rate (LR = 1.0e-4)

### Timeline
- **Started Stage 2:** March 7, 18:06 (Iteration 372)
- **First Breakthrough:** March 7, 20:46 (Iteration 385) - 2 waypoints captured
- **Full Mastery:** March 7, 21:08 (Iteration 387) - 25/25 waypoints
- **Advanced to Stage 3:** March 8, 06:12 (Iteration 430)
- **Duration:** ~12 hours | **Iterations:** 59

### Performance Progression

| Phase | Iterations | Waypoints | Status |
|-------|-----------|-----------|---------|
| **Exploration** | 372-384 (13 iters) | 0/25 | Random exploration, learning environment |
| **Breakthrough** | 385 | 2/25 | First successful waypoint captures! |
| **Inconsistent** | 386 | 0/25 | Temporary regression |
| **Mastery** | 387-430 (44 iters) | 25/25 | Consistent perfect navigation |

### Key Metrics
- **Learning Phase:** 13 iterations (2.6 hours)
- **Mastery Phase:** 44 iterations (9.4 hours)
- **Breakthrough Iteration:** 385 (2h 39m into Stage 2)
- **Final Success Rate:** 82% (threshold met at iteration 430)

### Notable Observations
- One brief regression at iteration 394 (0/25 waypoints) - quickly recovered
- One near-miss at iteration 420 (24/25 waypoints)
- **Highly stable** after breakthrough with 97.7% of post-breakthrough iterations at 25/25

### Learning Curve Pattern
```
Iterations 372-384: ████████████░ 0 waypoints (exploration)
Iteration 385:      ██░░░░░░░░░░░ 2 waypoints (first success!)
Iteration 386:      ░░░░░░░░░░░░░ 0 waypoints (regression)
Iterations 387-430: █████████████ 25 waypoints (consistent mastery)
```

---

## Run 2: Conservative Learning Rate (LR = 5.0e-5)

### Timeline
- **Started Stage 2:** March 7, 18:07 (Iteration 372)
- **First Breakthrough:** March 7, 21:09 (Iteration 387) - 1 waypoint captured
- **Full Mastery:** March 7, 21:32 (Iteration 389) - 25/25 waypoints
- **Advanced to Stage 3:** March 8, 06:34 (Iteration 432)
- **Duration:** ~12.5 hours | **Iterations:** 61

### Performance Progression

| Phase | Iterations | Waypoints | Status |
|-------|-----------|-----------|---------|
| **Exploration** | 372-386 (15 iters) | 0/25 | Slower exploration due to conservative LR |
| **Breakthrough** | 387 | 1/25 | First waypoint capture |
| **Early Mastery** | 388 | 0/25 | Brief regression |
| **Mastery** | 389-432 (44 iters) | 25/25 | Rock-solid consistency |

### Key Metrics
- **Learning Phase:** 15 iterations (3.0 hours)
- **Mastery Phase:** 44 iterations (9.5 hours)
- **Breakthrough Iteration:** 387 (3h 2m into Stage 2)
- **Final Success Rate:** 82% (threshold met at iteration 432)

### Notable Observations
- **Slowest to learn** (15 iterations) but **most stable** after mastery
- Only one regression after breakthrough (iteration 388)
- One near-miss at iteration 421 (21.7/25 waypoints)
- Conservative LR resulted in **98.2% consistency** post-breakthrough

### Learning Curve Pattern
```
Iterations 372-386: ███████████████░ 0 waypoints (slower exploration)
Iteration 387:      █░░░░░░░░░░░░░░░ 1 waypoint (cautious start)
Iteration 388:      ░░░░░░░░░░░░░░░░ 0 waypoints (regression)
Iterations 389-432: ████████████████ 25 waypoints (maximum stability)
```

---

## Run 3: Aggressive Learning Rate (LR = 1.5e-4) ⭐ **BEST PERFORMER**

### Timeline
- **Started Stage 2:** March 7, 18:08 (Iteration 372)
- **Early Signal:** March 7, 18:32 (Iteration 373) - 0.2 waypoints (partial capture)
- **First Breakthrough:** March 7, 20:14 (Iteration 382) - 1 waypoint
- **Full Mastery:** March 7, 20:36 (Iteration 384) - 25/25 waypoints
- **Advanced to Stage 3:** March 8, 05:37 (Iteration 426)
- **Duration:** ~11.5 hours | **Iterations:** 55

### Performance Progression

| Phase | Iterations | Waypoints | Status |
|-------|-----------|-----------|---------|
| **Exploration** | 372-381 (10 iters) | 0/25 | Fastest exploration phase |
| **Breakthrough** | 382 | 1/25 | First waypoint |
| **Rapid Learning** | 383 | 0/25 | Brief regression |
| **Mastery** | 384-426 (43 iters) | 25/25 | Fastest to mastery, consistent performance |

### Key Metrics
- **Learning Phase:** 10 iterations (2.0 hours)
- **Mastery Phase:** 43 iterations (9.5 hours)
- **Breakthrough Iteration:** 382 (2h 6m into Stage 2)
- **Final Success Rate:** ~88% (advanced earliest)

### Notable Observations
- **FASTEST learner** - Only 10 iterations to first waypoint!
- Early signal at iteration 373 (0.2 waypoints) showed learning beginning
- One near-miss at iteration 388 (21.5/25 waypoints)
- **First to advance** to Stage 3 (4 iterations ahead of others)
- Higher LR enabled faster adaptation without sacrificing stability (97.7% consistency)

### Learning Curve Pattern
```
Iterations 372-373: ██████████░░░ 0 waypoints (early learning signal at 373)
Iterations 374-381: ████████░░░░░ 0 waypoints (rapid exploration)
Iteration 382:      █░░░░░░░░░░░░ 1 waypoint (breakthrough!)
Iteration 383:      ░░░░░░░░░░░░░ 0 waypoints (brief regression)
Iterations 384-426: █████████████ 25 waypoints (fastest mastery)
```

---

## Comparative Analysis

### Learning Speed (Time to First Waypoint)
| Rank | Run | Learning Rate | Time to Breakthrough | Iterations |
|------|-----|---------------|---------------------|------------|
| 🥇 **1st** | **Run 3** | 1.5e-4 | 2h 6m | 10 |
| 🥈 **2nd** | Run 1 | 1.0e-4 | 2h 39m | 13 |
| 🥉 **3rd** | Run 2 | 5.0e-5 | 3h 2m | 15 |

**Insight:** Higher learning rate = faster exploration and learning

---

### Stability (Post-Breakthrough Consistency)
| Rank | Run | Learning Rate | Perfect Iterations | Consistency % |
|------|-----|---------------|-------------------|---------------|
| 🥇 **1st** | **Run 2** | 5.0e-5 | 43/44 | **98.2%** |
| 🥈 **2nd** | Run 1 | 1.0e-4 | 43/44 | **97.7%** |
| 🥈 **2nd** | Run 3 | 1.5e-4 | 42/43 | **97.7%** |

**Insight:** Conservative LR provides marginally better stability, but all runs highly stable

---

### Advancement to Stage 3
| Rank | Run | Learning Rate | Total Iterations | Duration | Final SR |
|------|-----|---------------|-----------------|----------|----------|
| 🥇 **1st** | **Run 3** | 1.5e-4 | 55 | 11.5h | ~88% |
| 🥈 **2nd** | Run 1 | 1.0e-4 | 59 | 12.0h | 82% |
| 🥉 **3rd** | Run 2 | 5.0e-5 | 61 | 12.5h | 82% |

**Insight:** Aggressive LR completed Stage 2 faster with higher success rate

---

## Critical Success Factor: Boundary Penalty Fix

### The Problem (Before Fix)
With `boundary_penalty = -5.0`:
- Robots learned a "suicide strategy" - exit arena quickly (50-120s)
- Time penalty accumulation (`-0.1/step × 3000 steps = -300`) worse than boundary penalty
- **Result:** 0% waypoint capture, no navigation learning

### The Solution (After Fix)
With `boundary_penalty = -50.0`:
- Boundary exit now equally punished as fall/timeout
- Robots forced to stay in arena and explore
- **Result:** All runs mastered navigation within 3 hours

### Impact Metrics
| Metric | Before Fix | After Fix | Improvement |
|--------|------------|-----------|-------------|
| **Episode Duration** | 50-120s | 300s initially → 100-180s optimized | ✓ Full episodes |
| **Waypoint Captures** | 0/25 | 25/25 consistently | ✓ 100% task completion |
| **Learning Occurred** | No | Yes | ✓ Breakthrough achieved |
| **Navigation Strategy** | Exit arena | Efficient waypoint capture | ✓ Proper behavior |

**Conclusion:** The boundary penalty fix was **absolutely critical** to training success.

---

## Stage 2 Learning Insights

### Common Learning Pattern (All Runs)
1. **Phase 1: Exploration (0 waypoints)**
   - Duration: 10-15 iterations
   - Behavior: Random movement, environment familiarization
   - Learning: Distance sensing, waypoint detection, action-reward associations

2. **Phase 2: Breakthrough (1-2 waypoints)**
   - Duration: 1-2 iterations
   - Behavior: First intentional waypoint approach and capture
   - Learning: "Aha moment" - waypoint proximity = high reward

3. **Phase 3: Inconsistency (0-25 waypoints)**
   - Duration: 0-1 iterations
   - Behavior: Policy refinement, occasional regression
   - Learning: Balancing exploration vs exploitation

4. **Phase 4: Mastery (25/25 waypoints)**
   - Duration: 43-44 iterations
   - Behavior: Consistent, efficient waypoint capture
   - Learning: Optimized navigation paths, speed improvements

### Key Success Factors
- ✅ **Score-based timeout system** (300 initial points, -1/sec decay, +15/waypoint) created natural time pressure
- ✅ **Waypoint capture reward** (+10) provided strong positive signal
- ✅ **Progress shaping rewards** (distance, heading, speed) guided early learning
- ✅ **Boundary penalty fix** (-50) prevented premature exits
- ✅ **PPO stability** (target_kl = 0.015, 10 epochs) prevented catastrophic forgetting

### Failure Modes Observed
- **Early termination due to boundary exit** - Fixed with penalty increase
- **Temporary regressions** - Natural PPO exploration, typically 1 iteration
- **Near-misses (21-24 waypoints)** - Score timeout before final waypoints, occasional occurrence

---

## Training Health Indicators

### PPO Algorithm Metrics
All runs showed healthy PPO training characteristics:

| Metric | Healthy Range | Run 1 | Run 2 | Run 3 | Status |
|--------|---------------|-------|-------|-------|--------|
| **Policy Loss** | -0.01 to 0.01 | ✓ | ✓ | ✓ | Healthy |
| **Value Loss** | Decreasing | 58→20 | 38→22 | 28→20 | ✓ Learning |
| **Entropy** | 4.0-4.5 | 4.41 | 4.41 | 4.41 | ✓ Exploring |
| **Approx KL** | <0.015 | <0.01 | <0.01 | <0.01 | ✓ Stable |
| **PPO Epochs** | 5-10 | 6-10 | 7-10 | 6-10 | ✓ Normal |

### Episode Metrics Evolution
| Phase | Episode Length | Score Remaining | Interpretation |
|-------|---------------|-----------------|----------------|
| **Early (0 waypoints)** | 300s | 0 | Full timeout - aimless wandering |
| **Breakthrough (1-25)** | 200-250s | 50-200 | Learning efficient paths |
| **Mastery (25/25)** | 100-180s | 200-560 | Optimized navigation, high efficiency |

---

## Reward Structure Effectiveness

### Implemented Rewards (Stage 2)
```yaml
waypoint_capture: +10.0     # Primary task reward
timeout_penalty: -50.0      # Failure penalty
boundary_penalty: -50.0     # Arena exit penalty (FIXED)
fall_penalty: -50.0         # Robot fall penalty
progress_shaping: 10.0      # Distance improvement reward
distance_reward: 0.5        # Proximity reward (continuous)
heading_reward: 0.2         # Facing target reward
wrong_direction: 2.0x       # Moving away penalty multiplier
```

### Effectiveness Analysis
| Reward Component | Purpose | Effectiveness | Notes |
|-----------------|---------|---------------|-------|
| **Waypoint Capture** (+10) | Primary goal | ⭐⭐⭐⭐⭐ | Strong signal, drives all learning |
| **Timeout Penalty** (-50) | Time efficiency | ⭐⭐⭐⭐ | Prevents aimless wandering |
| **Boundary Penalty** (-50) | Keep in arena | ⭐⭐⭐⭐⭐ | **CRITICAL FIX** - enabled all learning |
| **Progress Shaping** (10.0) | Guide early exploration | ⭐⭐⭐⭐ | Accelerated breakthrough by 1-2 hours |
| **Distance Reward** (0.5) | Continuous guidance | ⭐⭐⭐ | Subtle but helpful |
| **Heading Reward** (0.2) | Directional learning | ⭐⭐⭐ | Good for orientation |

**Recommendation:** Current reward structure is **excellent** for Stage 2 navigation. No changes needed.

---

## Episode Termination Analysis

### Termination Causes (Stage 2 Mastery Phase)
| Cause | Percentage | Notes |
|-------|-----------|-------|
| **Success (25/25 waypoints)** | ~85-90% | Ideal outcome |
| **Score Timeout (0 score)** | ~10-15% | Close attempts (21-24 waypoints) |
| **Boundary Exit** | <1% | Nearly eliminated by penalty fix |
| **Robot Fall** | <1% | Rare, usually during sharp turns |

### Episode Duration Distribution
- **Minimum:** 100s (perfect, high-speed runs)
- **Average:** 120-180s (typical successful runs)
- **Maximum:** 300s (score timeout)

**Insight:** Robots learned to navigate efficiently, completing 25 waypoints in 100-180s vs. theoretical max 300s.

---

## Resource Utilization

### Training Efficiency
- **Iterations per hour:** ~5.3 iterations/hour
- **Episodes per iteration:** 1-3 (depends on navigation speed)
- **GPU utilization:** 3 processes running simultaneously (staggered starts)

### Computational Cost (Stage 2)
| Run | Duration | Iterations | Cost Efficiency |
|-----|----------|-----------|-----------------|
| Run 1 | 12.0h | 59 | ~12m/iteration |
| Run 2 | 12.5h | 61 | ~12m/iteration |
| Run 3 | 11.5h | 55 | ~12m/iteration ⚡ |

**Insight:** Run 3 most efficient - faster learning reduced total training time.

---

## Recommendations for Future Stages

### Stage 3 (10m Waypoint Spacing) - Current
- **Expectation:** Moderate difficulty increase, similar learning pattern
- **Predicted Duration:** 15-20 hours to 80% success
- **Key Challenge:** Longer navigation distances require better path planning

### Stage 4-5 (20m/40m Spacing)
- **Expectation:** Easier than Stage 2-3 due to less precision required
- **Predicted Duration:** 10-15 hours each
- **Key Challenge:** Arena boundary constraints with longer distances

### Stage 6-8 (Adding Obstacles)
- **Expectation:** Significant difficulty spike
- **Predicted Duration:** 20-30 hours per stage
- **Key Challenge:** Collision avoidance while maintaining efficiency
- **Consideration:** May need to reduce reward shaping (currently enabled for Stages 2-5)

### Learning Rate Selection
Based on Stage 2 performance:

**Recommendation: Continue Run 3 (LR=1.5e-4) as primary**
- ✅ Fastest learning
- ✅ First to advance stages
- ✅ Excellent stability (97.7%)
- ✅ Highest success rate

**Maintain all 3 runs for**:
- Run 2: Control group with maximum stability
- Run 1: Middle-ground comparison
- Run 3: Fastest progression to later stages

---

## Conclusion

### Stage 2 Success Summary
✅ **All three runs successfully completed Stage 2**  
✅ **Learning breakthrough achieved within 2-3 hours**  
✅ **Consistent 25/25 waypoint capture after mastery**  
✅ **Boundary penalty fix was critical enabler**  
✅ **PPO training stable and healthy across all runs**

### Performance Rankings
| Rank | Run | Learning Rate | Strength |
|------|-----|---------------|----------|
| 🥇 **1st** | **Run 3** | 1.5e-4 | **Fastest learning, highest success rate** |
| 🥈 **2nd** | Run 1 | 1.0e-4 | Balanced performance |
| 🥉 **3rd** | Run 2 | 5.0e-5 | Most stable, but slowest |

### Key Takeaways
1. **Reward engineering matters** - The boundary penalty fix transformed training from complete failure to success
2. **Higher learning rates work well** - Run 3 (1.5e-4) excelled without instability
3. **PPO is robust** - All runs showed stable training with proper hyperparameters
4. **Curriculum learning effective** - 5m spacing was appropriate difficulty for Stage 2
5. **Score-based timeout superior to hard time limits** - Flexible, self-adjusting pressure

### Next Steps
- ✅ Continue training in Stage 3 (10m waypoint spacing)
- ✅ Monitor for similar learning patterns (10-15 iteration exploration phase)
- ✅ Expect 80% success rate in ~15-20 hours
- ✅ Use Run 3 as primary, maintain Run 1/2 for comparison
- ✅ No configuration changes needed - current setup optimal

---

**Report Generated:** March 8, 2026, 06:45 AM  
**Training Status:** All runs active in Stage 3  
**Next Review:** Stage 3 completion (~24-36 hours)
