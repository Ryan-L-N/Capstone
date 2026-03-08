# Stage 2 Configuration Updates

**Date:** March 7, 2026  
**Author:** Cole (MS for Autonomy Project)  
**Status:** Active Configuration

## Summary

Updated Stage 2 ("Waypoints 5m") to include a point-based timeout system and enhanced reward structure. These changes ensure episodes complete in reasonable time while providing clear learning signals to the policy.

---

## Previous Configuration Issues

### Problems
1. **No time limit** - Episodes could run indefinitely (`max_time: null`)
2. **No waypoint reward** - Capturing waypoints gave 0 RL reward (`waypoint_capture: 0.0`)
3. **No timeout penalty** - Running out of time had no negative consequence
4. **Empty rollout buffers** - Long episodes that didn't complete resulted in no training data
5. **All losses = 0** - Training loop couldn't compute gradients on empty buffers

### Observed Behavior
- Episodes running 345+ seconds to capture only 3 waypoints
- Training iterations with 0 episodes completed
- Policy loss, value loss, entropy all showing 0.0000
- No learning progress after 400+ iterations

---

## New Configuration

### Score Point System

**Game Mechanics:**
```yaml
scoring:
  initial_points: 300.0        # Starting score
  time_decay_per_sec: 1.0      # -1 point per second
  waypoint_bonus: 15.0         # +15 points per waypoint
```

**Episode Duration Calculation:**
- Base time: **300 seconds** (from initial points)
- Each waypoint: **+15 seconds** of extension
- Net cost per waypoint: Varies with robot speed

**Example Scenarios:**

| Scenario | Time per Waypoint | Points per WP | Max Waypoints |
|----------|-------------------|---------------|---------------|
| Fast (15s/wp) | 15s | -15 + 15 = 0 | ∞ (sustainable) |
| Medium (20s/wp) | 20s | -20 + 15 = -5 | 60 total |
| Slow (30s/wp) | 30s | -30 + 15 = -15 | 20 total |

**Current Performance (3 waypoints in 345s = 115s/waypoint):**
- Points per waypoint: -115 + 15 = **-100 points**
- Can complete: ~3 waypoints before timeout
- **Robot must learn to move faster!**

---

### RL Reward Structure

#### Immediate Rewards

```yaml
reward:
  waypoint_capture: 10.0       # NEW: +10 reward per waypoint
  timeout_penalty: -50.0       # NEW: -50 reward when score reaches 0
  fall_penalty: -50.0          # Existing: -50 for falling
  boundary_penalty: -5.0       # Existing: -5 for leaving arena
```

**Waypoint Capture:**
- **Score system:** +15 points (extends episode)
- **RL reward:** +10 reward (learning signal)
- Both happen simultaneously when robot enters 0.5m capture radius

**Timeout Event:**
- Triggers when `score <= 0`
- Episode terminates immediately
- **-50 reward penalty** applied
- Strong negative feedback to encourage speed

#### Continuous Shaping Rewards

Active throughout Stage 2 to guide learning:

```yaml
reward:
  progress_shaping: 10.0           # Alpha for distance-based shaping
  distance_reward: 0.5             # Reward for being close to waypoint
  heading_reward: 0.2              # Reward for facing toward waypoint
  speed_reward: 0.3                # Reward for moving fast (when path clear)
  wrong_direction_penalty: 2.0     # 2x penalty multiplier when moving away
```

**Progress Shaping:**
- Formula: `reward = α × (distance_previous - distance_current)`
- Positive when getting closer, negative when moving away
- **2x penalty** applied when moving in wrong direction
- Provides continuous guidance between waypoints

**Distance Reward:**
- Formula: `reward = 0.5 × max(0, 1 - distance/20)`
- Maximum at 1m from waypoint (0.5 reward)
- Zero at 20m from waypoint
- Continuous pull toward target

**Heading Reward:**
- Rewards aligning velocity vector with waypoint direction
- Only applied when robot is moving (speed > 0.1 m/s)
- Formula: `reward = 0.2 × max(0, cos(angle))`

**Speed Reward:**
- Only active when path is clear (no obstacles within 2m)
- Formula: `reward = 0.3 × min(speed/5.0, 1.0)`
- Maximum reward at 5 m/s forward speed
- Encourages fast movement in open areas

---

## Environment Configuration

### Arena
- **Size:** 50m diameter (25m radius)
- **Robot start:** [0, 0, 0.7] (center)
- **Boundaries:** 2m margin from edge for waypoint placement
- **Obstacles:** None (Stage 2 is empty arena)

### Waypoints
- **Total count:** 25 waypoints
- **Spacing:** 5m between consecutive waypoints
- **Capture radius:** 0.5m
- **First waypoint:** 5m from start position

### Robot Actions
```yaml
action:
  vx_range: [-0.5, 5.0]      # Forward/backward velocity (m/s) - INCREASED
  vy_range: [-0.5, 0.5]      # Strafe velocity (m/s)
  omega_range: [-1.5, 1.5]   # Turn rate (rad/s)
```

**Note:** Forward velocity range increased from 2.0 m/s to **5.0 m/s** to enable faster navigation.

---

## Success Criteria

### Episode Success
An episode is considered successful if:
1. **All 25 waypoints captured** (reached within 0.5m)
2. **No fall** (robot height stays > 0.25m)
3. **Stays in bounds** (within 25m radius)
4. **Completes before timeout** (score > 0)

### Stage Advancement
To advance from Stage 2 to Stage 3:
```yaml
curriculum:
  success_window: 100        # Last 100 episodes
  success_threshold: 0.80    # 80% success rate required
```

**Requirements:**
- 80 out of last 100 episodes must be successful
- Demonstrates consistent navigation ability
- Robot learned to balance speed and stability

---

## Implementation Details

### Code Changes

**1. Configuration Files (nav_config*.yaml):**
```yaml
# Added waypoint capture reward
reward:
  waypoint_capture: 10.0

# Added timeout penalty
reward:
  timeout_penalty: -50.0
```

**2. Navigation Environment (navigation_env.py):**

**Added in `__init__`:**
```python
self.reward_timeout = self.config['reward'].get('timeout_penalty', -50.0)
```

**Modified in `step()`:**
```python
# Waypoint capture now gives reward
if dist_to_wp < self.capture_radius:
    waypoint_captured = True
    self.waypoints_captured += 1
    self.current_waypoint_idx += 1
    self.score += self.waypoint_bonus_points  # +15 score points
    reward += self.reward_waypoint             # +10 RL reward
    
# Timeout penalty when score depleted
if timeout and self.score <= 0:
    reward += self.reward_timeout  # -50 penalty
    done = True
```

---

## Expected Training Behavior

### Phase 1: Initial Learning (0-100 iterations)
- **Goal:** Learn basic waypoint following
- **Expected:** 0-10% success rate
- **Timeouts:** Frequent (robot moving slowly)
- **Learning:** Understanding reward shaping signals

### Phase 2: Speed Development (100-500 iterations)
- **Goal:** Increase movement speed to avoid timeouts
- **Expected:** 10-40% success rate
- **Behavior:** Robot speeds up but may overshoot waypoints
- **Learning:** Balancing speed vs. accuracy

### Phase 3: Optimization (500-1000 iterations)
- **Goal:** Efficient waypoint-to-waypoint navigation
- **Expected:** 40-80% success rate
- **Behavior:** Smooth paths, appropriate speeds
- **Learning:** Fine-tuning velocity commands

### Phase 4: Mastery (1000+ iterations)
- **Goal:** Consistent success
- **Expected:** 80%+ success rate
- **Ready:** Advance to Stage 3 (10m waypoint spacing)

---

## Monitoring Metrics

### Key Metrics to Track

**Episode-Level:**
- `waypoints_captured`: Should increase over training (target: 25/25)
- `episode_time`: Should decrease as robot speeds up
- `score`: Final score (higher = faster completion)
- `success_rate`: Rolling 100-episode average (target: 80%)

**Step-Level:**
- `reward`: Should be mostly positive in successful episodes
- `timeout_penalty`: Should decrease in frequency
- `waypoint_capture_bonus`: Should increase in frequency

**Training Algorithm:**
- `policy_loss`: Should converge (not stay at 0)
- `value_loss`: Should decrease over time
- `approx_kl`: Should stay < 0.015 (early stopping threshold)
- `epochs_completed`: Should reach 5-10 per iteration

### Warning Signs

**Training Not Working:**
- All losses = 0.0000 (empty rollout buffers)
- No episodes completing after 100+ iterations
- KL divergence > 0.015 (policy unstable)
- Success rate stuck at 0%

**Robot Too Slow:**
- Timeouts every episode
- Only capturing 1-5 waypoints per episode
- Episode times > 200 seconds
- Need to increase speed reward or reduce waypoint count

**Robot Too Fast:**
- Frequent falls (> 20% of episodes)
- Overshooting waypoints
- Erratic movements
- Need to increase fall penalty or reduce max velocity

---

## Files Modified

1. `nav_config.yaml` - Base configuration
2. `nav_config_run1.yaml` - Run 1 (LR=1e-4)
3. `nav_config_run2.yaml` - Run 2 (LR=5e-5)
4. `nav_config_run3.yaml` - Run 3 (LR=1.5e-4)
5. `navigation_env.py` - Environment step logic
6. `ppo_trainer.py` - Fixed early stopping bug (separate issue)

---

## Next Steps

1. **Launch Training:** Start 3 runs with different learning rates
2. **Monitor First Hour:** Check if episodes completing successfully
3. **Verify Metrics:** Confirm all losses are non-zero
4. **Track Progress:** Watch success rate climb toward 80%
5. **Stage Advancement:** Automatically advance when threshold reached

---

## Notes

- **Max velocity increased to 5.0 m/s** to enable faster navigation
- **Score system creates natural time pressure** without arbitrary limits
- **Timeout penalty** provides strong negative feedback for inefficiency
- **Waypoint rewards** provide frequent positive feedback
- **All 4 config files updated** to ensure consistency across runs
