# RL Policy Development Roadmap
## Status: Ready to Begin Implementation

### Current Status ✅

**What We Have:**
1. ✅ Complete baseline environment (`RL_Development.py`)
   - Circular waypoint arena (25 waypoints, 50m diameter)
   - Dynamic obstacles (light/medium/heavy weight classes)
   - Small static obstacles  
   - Spot robot with SpotFlatTerrainPolicy
   - Episode management, scoring, CSV logging

2. ✅ RL framework files created:
   - `spot_rl_env.py` - Full RL environment wrapper (needs integration tweaks)
   - `training_config.py` - PPO hyperparameters and training profiles
   - `train_spot.py` - Training script with RSL-RL integration
   - `eval_policy.py` - Policy evaluation and visualization
   - `minimal_rl_demo.py` - Simple integration example

3. ✅ Documentation:
   - `README_RL.md` - Complete technical documentation
   - `QUICK_REFERENCE.md` - Command reference
   - `IMPLEMENTATION_SUMMARY.md` - Architecture overview

**What Needs Work:**
- Integration between `spot_rl_env.py` and `RL_Development.py`
- Action control interface (currently Spot uses pre-trained policy internally)
- Observation space implementation (contact sensors, collision forces)
- Testing and debugging

---

## Development Path Forward

### Phase 1: Basic Integration (1-2 hours)
**Goal:** Get a minimal RL training loop working

#### Step 1.1: Test Minimal Demo
```bash
cd "C:\Users\user\Desktop\Capstone_vs_1.2\Immersive-Modeling-and-Simulation-for-Autonomy\Experiments\Cole\RL Folder"
C:\isaac-sim\python.bat minimal_rl_demo.py --headless
```

**Expected Output:**
- Environment creates successfully
- Observations are generated (11-dim vector)
- Rewards are calculated
- Episode runs for 1000 steps or until termination
- No crashes

#### Step 1.2: Fix Integration Issues
**Tasks:**
- [ ] Verify `minimal_rl_demo.py` runs without errors
- [ ] Check that observations contain sensible values
- [ ] Verify reward function produces reasonable values
- [ ] Test with and without headless mode

**Common Issues & Fixes:**
| Issue | Solution |
|-------|----------|
| ImportError for RL_Development | Check file path, ensure module is importable |
| Spot not moving | Baseline uses internal policy, need to expose action control |
| Observations all zeros | Check Spot state queries are working |
| Simulation crashes | Reduce number of obstacles or use headless mode |

---

### Phase 2: Action Control Integration (2-3 hours)
**Goal:** Enable RL policy to directly control Spot's actions

#### Step 2.1: Modify CircularWaypointEnv
Current issue: Spot uses internal `SpotFlatTerrainPolicy` which we can't override from RL.

**Option A: Direct Velocity Control**
```python
# In CircularWaypointEnv.step():
def step_with_action(self, action: np.ndarray):
    """
    Step environment with external RL action.
    
    action: [vx, vy, omega] - desired velocities
    """
    # Convert RL action to Spot command
    command = self._convert_rl_action(action)
    
    # Apply to Spot
    self.spot.advance(self.world.get_physics_dt(), command)
    
    # Continue with rest of step logic...
```

**Option B: Wrap Spot's Policy**
```python
# Create custom policy that accepts RL actions
class RLSpotPolicy:
    def forward(self, obs):
        # Return action from RL policy
        return self.rl_policy(obs)
```

**Recommended:** Option A (simpler, more direct control)

#### Step 2.2: Test Action Control
```python
# Test script: test_action_control.py
env = CircularWaypointEnv(...)

for step in range(100):
    action = np.array([1.0, 0.0, 0.0])  # Forward only
    env.step_with_action(action)
    
    # Verify Spot moves forward
    pos, _ = env.spot.get_world_pose()
    print(f"Position: {pos}")
```

---

### Phase 3: Full Observation Space (2-3 hours)
**Goal:** Implement complete 92-dimensional observation space

#### Step 3.1: Joint State Observations
```python
def get_joint_states(env):
    """Get joint positions and velocities (24-dim)."""
    # Query Spot's articulation
    joint_positions = env.spot.get_joint_positions()  # 12-dim
    joint_velocities = env.spot.get_joint_velocities()  # 12-dim
    return np.concatenate([joint_positions, joint_velocities])
```

#### Step 3.2: Obstacle Sensing
```python
def get_nearest_obstacles(env, n=5):
    """Get 5 nearest obstacles with properties (35-dim)."""
    spot_pos, _ = env.spot.get_world_pose()
    
    obstacles = []
    for obs in env.obstacle_manager.obstacles:
        dist = distance_2d(spot_pos[:2], obs["pos"])
        obstacles.append((dist, obs))
    
    # Sort by distance, take nearest n
    obstacles.sort(key=lambda x: x[0])
    
    obs_features = []
    for i in range(n):
        if i < len(obstacles):
            dist, obs = obstacles[i]
            obs_features.extend([
                (obs["pos"][0] - spot_pos[0]) / 50.0,  # rel_x
                (obs["pos"][1] - spot_pos[1]) / 50.0,  # rel_y
                dist / 50.0,  # distance
                obs["mass"] / 32.7,  # mass (normalized)
                obs.get("friction", 0.5),  # friction
                encode_shape(obs["shape"]),  # shape ID
                1.0 if obs["mass"] > 32.7 else 0.0,  # is_static
            ])
        else:
            obs_features.extend([0.0] * 7)  # Padding
    
    return np.array(obs_features)
```

#### Step 3.3: Contact Sensors
```python
def get_foot_contacts(env):
    """Get binary contact state for each foot (4-dim)."""
    # Use Isaac Sim contact sensors
    # For now, approximate from physics contacts
    contacts = [0.0, 0.0, 0.0, 0.0]  # FL, FR, RL, RR
    
    # Query contact reports from PhysX
    # (This requires setting up contact sensors on Spot's feet)
    
    return np.array(contacts)
```

**Testing:**
- Print observation vector at each step
- Verify dimensions: 92 total
- Check that values are normalized (mostly in [-1, 1] range)
- Ensure no NaN or inf values

---

### Phase 4: Training Loop Integration (3-4 hours)
**Goal:** Get PPO training working with RSL-RL

#### Step 4.1: Install RSL-RL
```bash
pip install rsl-rl
```

#### Step 4.2: Create Vectorized Environment
```python
# In train_spot.py
from rsl_rl.env import VecEnv

class SpotVecEnv(VecEnv):
    """Vectorized environment wrapper for parallel training."""
    
    def __init__(self, num_envs, config):
        self.num_envs = num_envs
        self.envs = []
        
        # Create multiple environments in grid
        for i in range(num_envs):
            env = CircularWaypointEnv(...)
            self.envs.append(env)
    
    def step(self, actions):
        # Step all environments in parallel
        obs, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            o, r, d, i = env.step_with_action(action)
            obs.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(i)
        
        return np.array(obs), np.array(rewards), np.array(dones), infos
```

#### Step 4.3: Run Debug Training
```bash
# Start with minimal number of environments
python train_spot.py --config debug --num_envs 4
```

**Success Criteria:**
- Training loop starts without crashes
- Policy updates every N steps
- Rewards logged to console
- Checkpoints saved

---

### Phase 5: Hyperparameter Tuning (Ongoing)
**Goal:** Optimize training for best performance

#### Metrics to Track:
- **Waypoints per episode:** Should increase over time
- **Success rate:** % episodes reaching > 15 waypoints
- **Fall rate:** Should decrease to < 5%
- **Episode length:** Longer = exploring more
- **Collision rate:** Should decrease as policy learns obstacle avoidance

#### Tuning Strategy:
1. **Start conservative:** `config=stable`, low LR (1e-4)
2. **Monitor for 1000 iterations:** Check if learning is happening
3. **Adjust rewards:** If not reaching waypoints, increase `waypoint_reached`
4. **Increase learning rate:** If learning too slow, try 3e-4 or 5e-4
5. **Add curriculum:** Start with fewer obstacles, gradually increase

**Example Adjustments:**
```python
# If Spot keeps falling:
config.rewards.stability_reward = 0.5  # Increase from 0.2
config.rewards.fall_penalty = 150.0  # Increase from 100.0

# If not reaching waypoints:
config.rewards.waypoint_reached = 150.0  # Increase from 100.0
config.rewards.distance_reduction = 3.0  # Increase from 2.0

# If too many collisions:
config.rewards.collision_penalty = 1.0  # Increase from 0.5
config.rewards.smart_bypass = 5.0  # Increase from 3.0
```

---

## Quick Start Commands

### 1. Test Minimal Demo (First Step!)
```bash
cd "C:\Users\user\Desktop\Capstone_vs_1.2\Immersive-Modeling-and-Simulation-for-Autonomy\Experiments\Cole\RL Folder"
C:\isaac-sim\python.bat minimal_rl_demo.py --headless
```

### 2. After Integration Fixes
```bash
# Test with 1 environment
python test_rl_setup.py --headless

# Debug training (4 envs, 100 iterations)
python train_spot.py --config debug

# Monitor training
tensorboard --logdir logs/tensorboard
```

### 3. Full Training (When Ready)
```bash
# Standard training (4096 envs, 5000 iterations, ~6 hours)
python train_spot.py --config default

# Evaluate trained policy
python eval_policy.py --checkpoint logs/checkpoints/model_005000.pt --episodes 10 --render
```

---

## Troubleshooting Guide

### Common Issues

**1. ImportError: No module named 'RL_Development'**
```bash
# Fix: Ensure you're in the correct directory
cd "C:\Users\user\Desktop\Capstone_vs_1.2\Immersive-Modeling-and-Simulation-for-Autonomy\Experiments\Cole\RL Folder"
```

**2. Spot doesn't respond to RL actions**
- **Cause:** Default CircularWaypointEnv uses internal policy
- **Fix:** Add `step_with_action()` method (see Phase 2)

**3. Observations contain NaN or inf**
- **Cause:** Division by zero or invalid calculations
- **Fix:** Add safety checks, normalize properly
```python
# Safe normalization
dist_norm = np.clip(dist / 50.0, 0.0, 1.0)
```

**4. Training crashes with CUDA out of memory**
- **Fix:** Reduce number of environments
```bash
python train_spot.py --num_envs 1024  # Instead of 4096
```

**5. Reward stays flat (not learning)**
- **Cause:** Reward scale too small or policy can't explore
- **Fix:** 
  - Increase reward magnitudes (×10)
  - Increase learning rate
  - Check if actions actually affect environment

**6. Policy learns to exploit (e.g., spinning in place)**
- **Cause:** Reward shaping encourages unwanted behavior
- **Fix:** Add penalty for specific exploit
```python
# Penalize spinning
if abs(omega) > 0.8 and abs(vx) < 0.1:
    reward -= 5.0  # Spinning without forward motion
```

---

## Success Metrics

### Immediate (Phase 1-2): Integration Works
- ✅ Minimal demo runs without crashes
- ✅ Observations are generated correctly
- ✅ Rewards calculated properly
- ✅ RL actions control Spot's movement

### Short-term (Phase 3-4): Training Starts
- ✅ Debug training completes 100 iterations
- ✅ Policy updates successfully
- ✅ Waypoints per episode > 2 (better than random)
- ✅ No major crashes or bugs

### Medium-term (Phase 5): Learning Emerges
- ✅ Waypoints per episode increasing over time
- ✅ Fall rate decreasing
- ✅ Some episodes reaching 5+ waypoints
- ✅ Visible improvement in behavior

### Long-term: Surpass Baseline
- ✅ Average waypoints > 5 (baseline: 1.68)
- ✅ Success rate > 20% (baseline: 0%)
- ✅ Fall rate < 20% (baseline: 35%)
- ✅ Demonstrates obstacle nudging/bypassing

### Ultimate Goal: Expert Performance
- ✅ Average waypoints > 15
- ✅ Success rate > 50%
- ✅ Fall rate < 5%
- ✅ Intelligent obstacle interaction (nudge vs bypass)
- ✅ Energy-efficient, natural locomotion

---

## Next Steps (Priority Order)

### **TODAY:**
1. [ ] Run `minimal_rl_demo.py` to verify basic integration
2. [ ] Fix any immediate errors that arise
3. [ ] Document any necessary changes to `RL_Development.py`

### **THIS WEEK:**
4. [ ] Implement action control interface (Phase 2)
5. [ ] Test action control with simple forward/turn commands
6. [ ] Implement full observation space (Phase 3)
7. [ ] Verify all observations have sensible values

### **NEXT WEEK:**
8. [ ] Set up PPO training loop (Phase 4)
9. [ ] Run debug training with 4 environments
10. [ ] Monitor first 100 iterations for bugs
11. [ ] Fix integration issues as they arise

### **ONGOING:**
12. [ ] Full training with default config
13. [ ] Hyperparameter tuning based on results
14. [ ] Curriculum learning implementation
15. [ ] Final evaluation and comparison with baseline

---

## Resources

**Documentation:**
- [README_RL.md](README_RL.md) - Full technical documentation
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Command cheat sheet
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Architecture overview

**Key Files:**
- `RL_Development.py` - Baseline environment (1571 lines)
- `minimal_rl_demo.py` - Simple integration example (150 lines)
- `spot_rl_env.py` - Full RL wrapper (needs integration tweaks)
- `training_config.py` - Hyperparameters and profiles
- `train_spot.py` - Training script (needs testing)
- `eval_policy.py` - Evaluation utilities

**External Resources:**
- RSL-RL: https://github.com/leggedrobotics/rsl_rl
- Isaac Sim Docs: https://docs.omniverse.nvidia.com/isaacsim/
- PPO Paper: Schulman et al. (2017)

---

**Author:** Cole  
**Date:** February 27, 2026  
**Status:** Phase 1 Ready - Integration Testing

Let's begin with Step 1.1: Run the minimal demo!
