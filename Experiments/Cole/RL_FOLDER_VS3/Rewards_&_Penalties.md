# Navigation Curriculum: Rewards & Penalties by Stage

## Overview
7-stage curriculum progression from stability training → object pushing → waypoint navigation. Each stage has specific reward/penalty structure optimized for its learning objective.

---

## Stage 1: Stability Foundation
**Objective:** Learn basic locomotion stability without waypoints  
**Duration:** 180 seconds max  
**Obstacle Density:** 5% light, 5% heavy, 5% small

### Rewards
| Component | Weight | Trigger/Condition |
|-----------|--------|------------------|
| Waypoint Capture | +25.0 | N/A (no waypoints) |
| Push Exploration | +0.1 (0.2x scaled) | Contact detected |
| Push Sustained | +0.16 (0.2x scaled) | Moving while pushing |
| Speed Reward | +0.3 | Path clear & far from target |

### Penalties
| Component | Weight | Trigger/Condition |
|-----------|--------|------------------|
| Stagnation | -0.1/step | Movement < 0.05m per timestep |
| Stuck Pushing | -0.06 (0.2x scaled) | High contact but not moving |
| Wasted Effort | -0.06 (0.2x scaled) | High joint effort without progress |
| Fall | -100.0 | Robot z < 0.25m |
| Boundary | -5.0 | Robot exits 25m radius arena |
| Time Decay | -1.0/sec | Constant score decay |
| Timeout | -100.0 | Score reaches 0 |

### Other Mechanics
- **Success Criterion:** Remain stable for full 180s without falling
- **Advance Condition:** 80% success rate over 100 episodes

---

## Stage 2: Enhanced Stability
**Objective:** Maintain stability with increased obstacle density  
**Duration:** 180 seconds max  
**Obstacle Density:** 10% light, 10% heavy, 10% small (doubled from Stage 1)

### Rewards
| Component | Weight | Trigger/Condition |
|-----------|--------|------------------|
| Waypoint Capture | +25.0 | N/A (no waypoints) |
| Push Exploration | +0.1 (0.2x scaled) | Contact detected |
| Push Sustained | +0.16 (0.2x scaled) | Moving while pushing |
| Speed Reward | +0.3 | Path clear & far from target |

### Penalties
| Component | Weight | Trigger/Condition |
|-----------|--------|------------------|
| Stagnation | -0.1/step | Movement < 0.05m per timestep |
| Stuck Pushing | -0.06 (0.2x scaled) | High contact but not moving |
| Wasted Effort | -0.06 (0.2x scaled) | High joint effort without progress |
| Fall | -100.0 | Robot z < 0.25m |
| Boundary | -5.0 | Robot exits 25m radius arena |
| Time Decay | -1.0/sec | Constant score decay |
| Timeout | -100.0 | Score reaches 0 |

### Other Mechanics
- **Success Criterion:** Remain stable for full 180s with increased obstacles
- **Advance Condition:** 80% success rate over 100 episodes
- **Progress Shaping:** Disabled (focus on raw stability)

---

## Stage 3: Object Pushing Training
**Objective:** Learn to detect and push lightweight objects  
**Duration:** 300 seconds max (5 minutes)  
**Obstacle Density:** 10% light, 10% heavy, 10% small  
**Push Scaling Factor:** 1.0x (full reward emphasis)

### Rewards
| Component | Weight | Trigger/Condition |
|-----------|--------|------------------|
| Push Exploration | +0.5 (1.0x scaled) | Contact force > 0.4 AND effort < 0.7 |
| Push Sustained | +0.8 (1.0x scaled) | Moving (speed > 0.2) while pushing |
| Push Success | +15.0 per object | Successfully push object ≥ 1m |
| Speed Reward | +0.3 | Path clear & far from target |

### Penalties
| Component | Weight | Trigger/Condition |
|-----------|--------|------------------|
| Stuck Pushing | -0.3 (1.0x scaled) | Contact but not moving (speed ≤ 0.2) |
| Wasted Effort | -0.3 (1.0x scaled) | High effort (>0.8) with low progress |
| Fall | -100.0 | Robot z < 0.25m |
| Boundary | -5.0 | Robot exits arena |
| Time Decay | -1.0/sec | Constant score decay |
| Timeout | -100.0 | Score reaches 0 |

### Other Mechanics
- **Success Criterion:** Find and push **5 unique lightweight objects** at least **1m each**
- **Object Tracking:** Each unique contact (>0.5sec separation) = new object ID
- **Contact Detection:** Contact force > 0.3 with speed > 0.1m/s = active pushing
- **No Stagnation Penalty** (unlike Stages 1-2)
- **No Progress Shaping** (focus on force-based rewards)
- **Advance Condition:** Complete 5 pushes in single episode → auto-advance

---

## Stage 4: Short-Range Navigation
**Objective:** Capture waypoints with minimal waypoint spacing  
**Duration:** Unlimited (no time limit)  
**Waypoint Distance:** 5m intervals (25 waypoints total = 125m path)  
**Obstacle Density:** 10% light, 10% heavy, 10% small  
**Push Scaling Factor:** 0.4x (reduced emphasis)

### Rewards
| Component | Weight | Trigger/Condition |
|-----------|--------|------------------|
| Waypoint Capture | +25.0 | Distance to waypoint < 0.5m |
| Progress Shaping | +10.0 × Δd | Getting closer to waypoint |
| Heading Reward | +0.2 | Facing toward waypoint (alignment > 0) |
| Proximity Deceleration | +0.15 | Speed near ideal (slowing near waypoint) |
| Speed Reward | +0.3 | Path clear AND >3m from waypoint |
| Push Exploration | +0.2 (0.4x scaled) | Contact detected |
| Push Sustained | +0.32 (0.4x scaled) | Moving while pushing |

### Penalties
| Component | Weight | Trigger/Condition |
|-----------|--------|------------------|
| Stuck Pushing | -0.12 (0.4x scaled) | Contact but not moving |
| Wasted Effort | -0.12 (0.4x scaled) | High effort without progress |
| Fall | -100.0 | Robot z < 0.25m |
| Boundary | -5.0 | Robot exits arena |
| Time Decay | -1.0/sec | Constant score decay |
| Timeout | -100.0 | Score reaches 0 |

### Other Mechanics
- **Progress Shaping:** Enabled (reward for moving closer)
- **No Stagnation Penalty** (navigation focus)
- **Speed in Context:** Only when >3m away from waypoint (not during final approach)
- **Proximity Deceleration:** Encourages smooth capture (ideal speed = 0.2 × distance_to_waypoint)
- **Success Criterion:** Capture all 25 waypoints
- **Advance Condition:** 80% success rate over 100 episodes

---

## Stage 5: Medium-Range Navigation
**Objective:** Navigate longer distances with mixed obstacles  
**Duration:** Unlimited  
**Waypoint Distance:** 10m intervals (25 waypoints = 250m path)  
**Obstacle Density:** 10% light, 10% heavy, 10% small  
**Push Scaling Factor:** 0.4x

### Rewards
| Component | Weight | Trigger/Condition |
|-----------|--------|------------------|
| Waypoint Capture | +25.0 | Distance to waypoint < 0.5m |
| Progress Shaping | +10.0 × Δd | Getting closer to waypoint |
| Heading Reward | +0.2 | Facing toward waypoint |
| Proximity Deceleration | +0.15 | Speed near ideal |
| Speed Reward | +0.3 | Path clear AND >3m from waypoint |
| Push Exploration | +0.2 (0.4x scaled) | Contact detected |
| Push Sustained | +0.32 (0.4x scaled) | Moving while pushing |

### Penalties
| Component | Weight | Trigger/Condition |
|-----------|--------|------------------|
| Stuck Pushing | -0.12 (0.4x scaled) | Contact but not moving |
| Wasted Effort | -0.12 (0.4x scaled) | High effort without progress |
| Fall | -100.0 | Robot z < 0.25m |
| Boundary | -5.0 | Robot exits arena |
| Time Decay | -1.0/sec | Constant score decay |
| Timeout | -100.0 | Score reaches 0 |

### Other Mechanics
- **Progress Shaping:** Enabled
- **Increased Challenge:** Waypoint spacing 2x farther than Stage 4
- **Same Reward Structure:** As Stage 4, but longer paths test endurance
- **Success Criterion:** Capture all 25 waypoints at 10m intervals
- **Advance Condition:** 80% success rate over 100 episodes

---

## Stage 6: Long-Range with Dense Obstacles
**Objective:** Navigate very long distances with increasing density  
**Duration:** Unlimited  
**Waypoint Distance:** 20m intervals (25 waypoints = 500m path)  
**Obstacle Density:** 10% light, 10% heavy, 10% small  
**Push Scaling Factor:** 0.4x

### Rewards
| Component | Weight | Trigger/Condition |
|-----------|--------|------------------|
| Waypoint Capture | +25.0 | Distance to waypoint < 0.5m |
| Progress Shaping | +10.0 × Δd | Getting closer to waypoint |
| Heading Reward | +0.2 | Facing toward waypoint |
| Proximity Deceleration | +0.15 | Speed near ideal |
| Speed Reward | +0.3 | Path clear AND >3m from waypoint |
| Push Exploration | +0.2 (0.4x scaled) | Contact detected |
| Push Sustained | +0.32 (0.4x scaled) | Moving while pushing |

### Penalties
| Component | Weight | Trigger/Condition |
|-----------|--------|------------------|
| Stuck Pushing | -0.12 (0.4x scaled) | Contact but not moving |
| Wasted Effort | -0.12 (0.4x scaled) | High effort without progress |
| Fall | -100.0 | Robot z < 0.25m |
| Boundary | -5.0 | Robot exits arena |
| Time Decay | -1.0/sec | Constant score decay |
| Timeout | -100.0 | Score reaches 0 |

### Other Mechanics
- **Progress Shaping:** Enabled
- **Longest Fixed Distance:** 20m waypoint spacing tests long-range planning
- **Same Reward Structure:** As Stages 4-5
- **Success Criterion:** Capture all 25 waypoints at 20m intervals
- **Advance Condition:** 80% success rate over 100 episodes

---

## Stage 7: Expert Navigation
**Objective:** Variable distance navigation - final mastery stage  
**Duration:** Unlimited  
**Waypoint Distance:** 20m first, then 40m subsequent (25 waypoints = 980m total path)  
**Obstacle Density:** 10% light, 10% heavy, 10% small  
**Push Scaling Factor:** 0.4x

### Rewards
| Component | Weight | Trigger/Condition |
|-----------|--------|------------------|
| Waypoint Capture | +25.0 | Distance to waypoint < 0.5m |
| Progress Shaping | +10.0 × Δd | Getting closer to waypoint |
| Heading Reward | +0.2 | Facing toward waypoint |
| Proximity Deceleration | +0.15 | Speed near ideal |
| Speed Reward | +0.3 | Path clear AND >3m from waypoint |
| Push Exploration | +0.2 (0.4x scaled) | Contact detected |
| Push Sustained | +0.32 (0.4x scaled) | Moving while pushing |

### Penalties
| Component | Weight | Trigger/Condition |
|-----------|--------|------------------|
| Stuck Pushing | -0.12 (0.4x scaled) | Contact but not moving |
| Wasted Effort | -0.12 (0.4x scaled) | High effort without progress |
| Fall | -100.0 | Robot z < 0.25m |
| Boundary | -5.0 | Robot exits arena |
| Time Decay | -1.0/sec | Constant score decay |
| Timeout | -100.0 | Score reaches 0 |

### Other Mechanics
- **Progress Shaping:** Enabled
- **Variable Distance:** Longer gaps (40m) after first waypoint combines stages 4-6
- **Maximum Path Length:** 980m total = ultra-long range endurance test
- **Success Criterion:** Capture all 25 waypoints (TRAINING COMPLETE)
- **Final Stage:** No advance condition, training concludes at this stage

---

## Universal Penalties (All Stages)
| Event | Penalty | Description |
|-------|---------|-------------|
| Fall | -100.0 | Robot height drops below 0.25m |
| Out of Bounds | -5.0 | Robot exits 25m radius arena |
| Timeout | -100.0 | Episode score reaches 0 points |
| Time Decay | -1.0/sec | Continuous score decay throughout episode |

---

## Push Reward Scaling Across Stages

| Stage | Name | Push Scale | Rationale |
|-------|------|-----------|-----------|
| 1 | Stability Foundation | 0.2x | Low priority (focus stability) |
| 2 | Enhanced Stability | 0.2x | Low priority (focus stability) |
| 3 | Object Pushing | 1.0x | **Full reward** (dedicated pushing) |
| 4 | Short-Range Nav | 0.4x | Reduced (navigation primary) |
| 5 | Medium-Range Nav | 0.4x | Reduced (navigation primary) |
| 6 | Long-Range Nav | 0.4x | Reduced (navigation primary) |
| 7 | Expert Navigation | 0.4x | Reduced (navigation primary) |

**Note:** Push rewards = 0.2x or 0.4x × base rewards shown in stage tables above.

---

## Stage Advancement Criteria

All stages use **80% success rate over 100 episodes** window:
- **Stage 1→2:** 80 successes in last 100 episodes
- **Stage 2→3:** 80 successes in last 100 episodes
- **Stage 3→4:** Complete 5 object pushes in single episode (auto-advance)
- **Stage 4→5:** 80% success over 100 episodes
- **Stage 5→6:** 80% success over 100 episodes
- **Stage 6→7:** 80% success over 100 episodes
- **Stage 7:** Final stage - training concludes

---

## Progress Shaping Details

**Stages with Progress Shaping:** 4, 5, 6, 7  
**Stages without Progress Shaping:** 1, 2, 3

### How It Works
- Reward = alpha × (distance_last_step - distance_current_step)
- Alpha = 10.0
- Encourages steady approach to waypoint
- Negative reward if moving away = soft penalty without hard -X penalty

### Example
- Previous distance to waypoint: 5.0m
- Current distance: 4.8m
- Progress = 0.2m gained
- Reward = 10.0 × 0.2 = **+2.0 bonus**

---

## Summary Table: Rewards by Stage Type

| Stage Group | Stages | Primary Objective | Key Reward | Push Scale |
|-------------|--------|------------------|-----------|-----------|
| **Stability** | 1, 2 | Basic locomotion | None (score-based) | 0.2x |
| **Pushing** | 3 | Object interaction | Push success +15.0 | 1.0x |
| **Navigation** | 4-7 | Waypoint capture | Capture +25.0, Progress +10.0Δd | 0.4x |

---

## Key Design Decisions

1. **Push Rewards:** Scale by stage priority (low in early stages, high in Stage 3, reduced in navigation)
2. **Progress Shaping:** Only in navigation stages (4-7) where longer-term planning needed
3. **No Wrong Direction Penalty:** Removed to simplify learning
4. **Distance Reward:** Disabled (progress shaping sufficient)
5. **Stagnation Penalty:** Stages 1-2 only (ensure movement learning)
6. **Doubled Fall Penalty:** -100 (vs -50 originally) = high cost for falling
7. **Heading + Proximity:** Enable smooth waypoint capture approach control
8. **Speed Reward Context:** Only when >3m away (not interfering with capture precision)

