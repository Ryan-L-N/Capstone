# Lessons Learned – Grass Terrain Experiment

## Overview

This document tracks lessons learned, pitfalls avoided, and best practices discovered during the grass terrain navigation experiment. It should be updated after each phase completion.

> ⚠️ **SCOPE**: Single 60ft × 30ft (18.3m × 9.1m) room with procedural grass.

---

## Document Structure

1. **Environment Setup** – Getting Isaac Sim running correctly
2. **Do Not Repeat (DNR)** – Mistakes to avoid
3. **Best Practices** – Successful approaches
4. **Grass-Specific Insights** – Terrain-specific learnings
5. **Platform Comparisons** – Spot vs. V60 observations
6. **Phase-by-Phase Notes** – Chronological learnings

---

## Environment Setup

### ES-001: Correct Conda Environment for Isaac Sim 5.1.0

**Issue**: Running scripts with wrong Python environment causes `ModuleNotFoundError: No module named 'numpy'` or missing Isaac Sim modules.

**Root Cause**: Multiple conda environments exist:
- `isaaclab` — Python 3.13 (WRONG - too new)
- `isaaclab311` — Python 3.11 (CORRECT for Isaac Sim 5.1.0)

**Solution**: Always activate the correct environment before running scripts:
```powershell
conda activate isaaclab311
cd "c:\Users\Gabriel Santiago\OneDrive\Desktop\Nvidia Omniverse"
& "$env:CONDA_PREFIX\python.exe" <script.py>
```

**Key Points**:
- Isaac Sim 5.1.0 requires Python 3.11
- Use `$env:CONDA_PREFIX\python.exe` to ensure correct Python is used
- The `nvidia_venv` virtual environment is broken (references deleted Python 3.10 path)

**Status**: ✅ Validated (January 26, 2026)

---

### ES-002: SpotFlatTerrainPolicy API Change in Isaac Sim 5.1.0

**Issue**: `AttributeError: 'SpotFlatTerrainPolicy' object has no attribute '_default_joint_pos'`

**Root Cause**: The `_default_joint_pos` attribute was removed or renamed in Isaac Sim 5.1.0. Old scripts from 4.x versions use this pattern:
```python
# OLD (Isaac Sim 4.x) - NO LONGER WORKS
spot.robot.set_joints_default_state(spot._default_joint_pos)
```

**Solution**: Remove the `set_joints_default_state` call - SpotFlatTerrainPolicy handles initialization internally:
```python
# NEW (Isaac Sim 5.1.0)
spot = SpotFlatTerrainPolicy(
    prim_path="/World/Spot",
    name="Spot",
    position=np.array([1.0, 1.0, 0.8]),
)

# Just reset and initialize - no manual joint state setting needed
world.reset()
spot.initialize()
```

**Status**: ✅ Validated (January 26, 2026)

---

### ES-003: Quick Reference - Running Isaac Sim Scripts

**Complete startup sequence**:
```powershell
# 1. Activate correct environment
conda activate isaaclab311

# 2. Navigate to workspace
cd "c:\Users\Gabriel Santiago\OneDrive\Desktop\Nvidia Omniverse"

# 3. Run script with explicit Python path
& "$env:CONDA_PREFIX\python.exe" "path\to\script.py"
```

**Verification**:
```powershell
python --version  # Should show Python 3.11.x
conda list numpy  # Should show numpy 1.26.0
```

**Status**: ✅ Validated (January 26, 2026)

---

### ES-004: SpotFlatTerrainPolicy advance() → forward() in Isaac Sim 5.1.0

**Issue**: `AttributeError: 'SpotFlatTerrainPolicy' object has no attribute 'advance'`

**Root Cause**: The `advance()` method was renamed to `forward()` in Isaac Sim 5.1.0. The signature is the same:

```python
# OLD (Isaac Sim 4.x) - NO LONGER WORKS
spot.advance(step_size, np.array(command))

# NEW (Isaac Sim 5.1.0) - same signature, just renamed
spot.forward(step_size, np.array(command))
```

**Method Signature**:
```python
forward(dt, command)
    """Compute the desired torques and apply them to the articulation
    
    Arguments:
        dt (float) -- Timestep update in the world
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)
    """
```

**Available Methods in 5.1.0**:
- `forward(dt, command)` - Send velocity command to robot
- `initialize()` - Initialize the policy
- `load_policy(policy_file_path, policy_env_path)` - Load policy weights
- `policy()` - Get policy object
- `post_reset()` - Post-reset callback
- `reset()` - Reset the robot
- `robot` - Access the robot articulation

**Status**: ✅ Validated (January 26, 2026)

---

### ES-005: USD Path Sanitization for Debug Markers

**Issue**: `Invalid Path` error when creating visual debug markers with `.` or `-` in names.

**Root Cause**: USD paths cannot contain certain characters like `.` or `-`. Creating a prim path like `/World/Markers/target_point_1.2` fails.

**Solution**: Sanitize marker names before creating USD prims:
```python
def create_marker(name, position, color):
    # Sanitize name for USD path compatibility
    safe_name = name.replace(".", "_").replace("-", "_")
    prim_path = f"/World/Markers/{safe_name}"
    # ... create visual marker
```

**Status**: ✅ Validated (January 26, 2026)

---

### ES-006: USD Prim Duplicate Prevention

**Issue**: `pxr.Tf.ErrorException: The xformOp 'xformOp:translate' already exists in xformOpOrder`

**Root Cause**: Creating USD prims (like markers) multiple times without cleaning up causes duplicate attribute errors.

**Solution**: Check for and remove existing prims before creating:
```python
def create_target_marker(stage, position, name="target"):
    marker_path = f"/World/{name}_marker"
    
    # Delete existing marker if it exists
    existing_prim = stage.GetPrimAtPath(marker_path)
    if existing_prim.IsValid():
        stage.RemovePrim(marker_path)
    
    marker = UsdGeom.Cylinder.Define(stage, marker_path)
    # ... set attributes
```

**Status**: ✅ Validated (January 26, 2026)

---

### ES-007: Quadruped Navigation Controller Oscillation

**Issue**: Spot robot oscillates in place instead of navigating to target, position cycles between X=0.3-2.5m while target is at X=17.3m.

**Root Cause**: Proportional-only heading control with high gain (2.0) causes overshooting. Robot turns, overshoots heading, turns back, overshoots again - never moving forward consistently.

**Symptoms**:
- Distance to goal doesn't decrease over time
- Robot wobbles/oscillates
- Commands alternate between positive and negative turn rates

**Solution**: Implement "turn-then-move" strategy:
```python
# Turn-then-move strategy
abs_yaw_error = abs(yaw_error)

if abs_yaw_error > 0.3:  # ~17 degrees - need to turn first
    # Turn in place with moderate gain
    wz = np.clip(yaw_error * 1.5, -TURN_RATE, TURN_RATE)
    vx = 0.0  # Don't move forward while turning
else:
    # Well aligned - move forward with small corrections
    wz = np.clip(yaw_error * 0.8, -TURN_RATE * 0.5, TURN_RATE * 0.5)
    vx = FORWARD_SPEED
```

**Key Insights**:
- Separate "turn" and "move" phases reduces oscillation
- Lower gain (0.8-1.5) prevents overshooting
- Threshold of ~17° (0.3 rad) works well for quadrupeds
- Don't move forward while significantly misaligned

**Status**: 🔄 Testing (January 26, 2026)

---

### ES-008: Physical Grass Interference for RL Training

**Issue**: Visual-only grass (cylinder primitives without physics) creates no meaningful training data for RL. Navigation times identical to grass-free baseline (~17.6s regardless of grass height).

**Root Cause**: Grass cylinders were created as visual elements only - no collision bodies, no friction modification, no physical interaction with the robot.

**Impact**: Without physical interference, the RL agent cannot learn:
- How to push through grass resistance
- Gait modifications for tall grass
- Energy-efficient traversal strategies

**Solution**: Implement THREE physical grass properties:

```python
# 1. COLLISION DETECTION - Grass clusters push against robot
from pxr import UsdPhysics, PhysxSchema
UsdPhysics.CollisionAPI.Apply(grass_cluster_prim)
body_api = UsdPhysics.RigidBodyAPI.Apply(grass_cluster_prim)
body_api.CreateKinematicEnabledAttr(True)  # Static collision body

# 2. FRICTION MODIFIERS - Slow movement in grass zones
# Create invisible friction surface below grass
friction_surface = UsdGeom.Cube.Define(stage, f"{zone_path}/friction_surface")
grass_friction = BASE_GROUND_FRICTION + friction_mod  # Height-dependent
material = UsdShade.Material.Define(stage, f"{zone_path}/grass_material")
physics_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
physics_api.CreateStaticFrictionAttr(grass_friction)
UsdShade.MaterialBindingAPI(friction_surface).Bind(material)

# 3. CONTACT TRACKING - Detect zone entry for RL metrics
grass_contact_count = 0
time_in_grass = 0.0

def is_in_grass_zone(pos):
    return (grass_x_min <= pos[0] <= grass_x_max and
            grass_y_min <= pos[1] <= grass_y_max)

# In physics callback:
in_grass = is_in_grass_zone(pos)
if in_grass:
    time_in_grass += step_size
    # APPLY SPEED PENALTY
    command = [command[0] * speed_penalty, command[1], command[2] * speed_penalty]
```

**Height-Based Physical Properties**:
| Level | Height | Speed Penalty | Collision Radius | Friction Mod |
|-------|--------|---------------|------------------|--------------|
| H1 | 0.1m | 95% | 0.015m | +0.04 |
| H2 | 0.3m | 85% | 0.020m | +0.12 |
| H3 | 0.5m | 70% | 0.025m | +0.20 |
| H4 | 0.7m | 50% | 0.030m | +0.28 |

**Actual Results with Physical Grass (H1 Testing)**:
> ⚠️ **CRITICAL FINDING**: Physical collision causes 100% failure rate even at lightest H1 level!
- H1 Seed 42: 0/5 success (100% falls) - Robot destabilizes within grass zone
- H1 Seed 123: First run failed at 20.1s (5.4m from target)
- Robot consistently falls when legs contact 1166 collision cylinders
- Speed penalty working (95% = 0.95 m/s effective)
- Grass contact detection working correctly

**Implication for RL Training**:
- This creates VALUABLE negative reward data for training
- RL agent will need to learn stable gait through physical obstacles
- May need to tune collision radius down or implement progressive difficulty
- Current collision is too aggressive for baseline policy

**RL Training Metrics Collected**:
- `grass_contact_count`: Zone entries (gait optimization signal)
- `time_in_grass`: Total grass traversal time
- `effective_speed`: Actual speed through grass (path/time_in_grass)
- `stalled`: Robot stuck without falling (ES-009)

**Status**: ✅ Implemented (January 26, 2026)

---

### ES-009: Stall Detection for RL Training

**Issue**: Robot may get stuck in grass without technically falling. The fall detection (z < 0.3m) doesn't catch scenarios where the robot is physically blocked but still upright.

**Root Cause**: Physical collision bodies create resistance that may stop forward progress without destabilizing the robot enough to trigger a fall.

**Symptom**: Robot remains upright but makes no forward progress - effectively "stuck" in grass.

**Solution**: Implement stall detection based on forward progress:
```python
# Stall detection parameters
STALL_TIMEOUT = 10.0      # seconds - no progress = stalled
STALL_THRESHOLD = 0.1     # meters - minimum forward progress

# Track forward progress
last_progress_x = start_x
last_progress_time = 0.0

# In main loop:
current_x = pos[0]
if current_x > last_progress_x + STALL_THRESHOLD:
    # Made forward progress, reset timer
    last_progress_x = current_x
    last_progress_time = sim_time
elif sim_time - last_progress_time > STALL_TIMEOUT:
    # No forward progress for 10 seconds = stalled
    stalled = True
    nav.mark_failed(f"Stalled - no forward progress for {STALL_TIMEOUT}s")
    break
```

**Key Insights**:
- Stall = 10 seconds with <0.1m forward progress
- Equivalent to fall for RL training (negative reward)
- Captures "stuck" scenarios that fall detection misses
- Valuable data for RL: robot needs to learn unstuck strategies

**Metrics Added**:
- `stalled`: Boolean flag in run data
- `stall_count`: Summary statistic per test batch

**Status**: ✅ Implemented (January 26, 2026)

---

### ES-010: Proper PhysX Grass Physics Implementation

**Issue**: Dense collision bodies (1166+ cylinders) cause 100% robot fall rate. Physics simulation becomes unstable with many contact points.

**Root Cause**: Using collision cylinders for EVERY grass blade creates:
1. Excessive contact solve iterations
2. Contact "explosions" when robot brushes through
3. Jittering at stalk bases
4. Performance degradation

**Solution**: Friction-based grass with SPARSE proxy stalks

#### A) GrassMaterial - Primary Grass Effect
Create a PhysX material that simulates grass resistance through friction:

```python
GRASS_MATERIAL_CONFIG = {
    "static_friction": 0.20,      # Low grass friction
    "dynamic_friction": 0.15,
    "restitution": 0.05,          # Nearly no bounce
    "friction_combine_mode": "average",
    "restitution_combine_mode": "min",
}

# Robot foot material (reference)
ROBOT_FOOT_MATERIAL = {
    "static_friction": 0.55,
    "dynamic_friction": 0.45,
}
```

**Friction Combine Calculation (PhysX "average" mode)**:
- Effective μ_static = (0.20 + 0.55) / 2 = **0.375**
- Effective μ_dynamic = (0.15 + 0.45) / 2 = **0.30**
- This gives realistic grass slip behavior (0.30-0.40 range)

#### B) Proxy Stalks - Sparse Visual/Sensor Feedback

Target density for 1800 ft² room:
| Option | Density | Stalk Count | Recommendation |
|--------|---------|-------------|----------------|
| Sparse | 0.5/ft² | 900 | ✅ RECOMMENDED |
| Medium | 1.0/ft² | 1800 | Acceptable |
| Dense | 2.0/ft² | 3600 | ❌ Use friction instead |

**Proxy Stalk Physics Settings**:
```python
PROXY_STALK_CONFIG = {
    # Geometry
    "height_min": 0.30,      # meters
    "height_max": 0.50,      # meters
    "radius_min": 0.005,     # 5mm
    "radius_max": 0.015,     # 15mm
    "base_sink": 0.02,       # Sink 2cm into ground (prevents jitter)
    
    # Physics
    "use_kinematic": True,   # Kinematic = stable, no physics solve
    "self_collision": False, # CRITICAL: Stalks don't hit each other
}
```

#### C) ENABLE_PROXY_STALKS Toggle
```python
ENABLE_PROXY_STALKS = True   # Stalks have collision
ENABLE_PROXY_STALKS = False  # Friction-only mode (better performance)
```

**Implementation**: See `grass_physics_config.py`

**Validation Checklist**:
1. ☐ Friction test: Robot should slip ~10-20% more than on normal ground
2. ☐ Contact forces: No spikes >100N during normal walking
3. ☐ Performance: >30 FPS with 900 stalks, >60 FPS friction-only
4. ☐ Stability: No jitter, no contact explosions over 60s test

**Signs of Instability & First Fixes**:
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Stalk jitter | Base not sunk | Increase `base_sink` to 0.03m |
| Contact explosion | Self-collision | Ensure collision group filtering |
| Robot falls immediately | Too many colliders | Reduce density or use friction-only |
| FPS < 20 | Too many stalks | Set `ENABLE_PROXY_STALKS = False` |

**Status**: ✅ Implemented (January 26, 2026)

---

### ES-011: Obstacle Placement in Grass Terrain

**Issue**: Random obstacle placement can create impossible scenarios or cluster obstacles unfairly.

**Solution**: Implement constrained random placement with minimum separation:

```python
# Placement constraints
MIN_OBSTACLE_SPAWN_DIST = 3.0    # meters from spawn point
MIN_OBSTACLE_TARGET_DIST = 3.0  # meters from target
MIN_OBSTACLE_SEPARATION = 1.5   # meters between obstacles

def is_valid_placement(pos, existing_obstacles, spawn, target):
    # Check spawn clearance
    if np.linalg.norm(pos - spawn) < MIN_OBSTACLE_SPAWN_DIST:
        return False
    # Check target clearance
    if np.linalg.norm(pos - target) < MIN_OBSTACLE_TARGET_DIST:
        return False
    # Check obstacle separation
    for obs in existing_obstacles:
        if np.linalg.norm(pos - obs) < MIN_OBSTACLE_SEPARATION:
            return False
    return True
```

**Key Points**:
- Use seeded RNG (`np.random.seed()`) for reproducible layouts
- Retry placement up to 100 times before giving up
- Log obstacle positions for debugging failed runs

**Status**: ✅ Implemented (January 26, 2026)

---

### ES-012: Hidden vs Visible Obstacle Tracking

**Issue**: Need to differentiate between obstacles robot "should" detect (visible above grass) vs obstacles hidden in grass.

**Root Cause**: For RL training and policy evaluation, the distinction matters - hidden obstacles test proprioceptive recovery, visible obstacles test perception.

**Solution**: Track visibility flag per obstacle and separate collision counts:

```python
# Obstacle configuration with visibility
OBSTACLE_TYPES = {
    "TallCube": {"height": 0.8, "visible": True},   # 0.8m > H3 (0.5m)
    "Cylinder": {"height": 0.6, "visible": True},   # 0.6m > H3 (0.5m)
    "LowBlock": {"height": 0.3, "visible": False},  # 0.3m < H3 (0.5m) - HIDDEN
}

# Track collisions separately
visible_collisions = 0
hidden_collisions = 0

def log_collision(obstacle):
    if obstacle.visible:
        visible_collisions += 1
    else:
        hidden_collisions += 1
```

**Metrics**:
- `mean_visible_collisions`: Perception failure metric
- `mean_hidden_collisions`: Proprioceptive/recovery failure metric

**Phase 4 Results**: 1.72 visible vs 1.62 hidden collisions - roughly equal, indicating robot has no obstacle avoidance capability at all.

**Status**: ✅ Implemented (January 26, 2026)

---

### ES-013: Collision Detection via Proximity (No Physics Contact)

**Issue**: Using PhysX contact callbacks for collision detection causes physics instability with many obstacles.

**Solution**: Use proximity-based collision detection instead of physics contacts:

```python
COLLISION_RADIUS = 0.4  # meters - robot body radius approximation

def check_collisions(robot_pos, obstacles, already_collided):
    new_collisions = []
    for i, obs in enumerate(obstacles):
        if i in already_collided:
            continue
        dist = np.sqrt((robot_pos[0] - obs.x)**2 + (robot_pos[1] - obs.y)**2)
        if dist < COLLISION_RADIUS + obs.radius:
            new_collisions.append(i)
            already_collided.add(i)
    return new_collisions
```

**Benefits**:
- No physics simulation overhead
- Deterministic collision detection
- Can log collision without physics "explosion"
- Works with kinematic obstacles (no rigid body solve)

**Status**: ✅ Implemented (January 26, 2026)

---

### ES-014: Extended Timeout for Obstacle Navigation

**Issue**: Standard 180-second timeout insufficient for obstacle-heavy scenarios where robot may need to navigate around obstacles.

**Solution**: Extend timeout for obstacle phases:

```python
# Phase-specific timeouts
TIMEOUT_BASELINE = 120      # Phase 1: No obstacles
TIMEOUT_GRASS = 180         # Phase 2-3: Grass only
TIMEOUT_OBSTACLES = 240     # Phase 4: Grass + obstacles
```

**Rationale**: Robot may need additional time to:
- Recover from collisions
- Navigate around obstacles (if policy supports)
- Handle stall scenarios

**Status**: ✅ Implemented (January 26, 2026)

---

### ES-015: Model Filename Mismatch in Evaluation Scripts

**Issue**: Evaluation script defaulted to `best_model.pt` but training saves as `best_policy.pt`

**Root Cause**: Different naming conventions between generic RL training (PyTorch convention: `best_model.pt`) and policy-specific training (IsaacLab convention: `best_policy.pt`)

**Solution**: 
```python
# Wrong
parser.add_argument("--model", default="../logs/spot_eureka/best_model.pt")

# Correct
parser.add_argument("--model", default="../logs/spot_eureka/best_policy.pt")
```

**Prevention**: Always verify actual saved filenames before writing evaluation code

**Status**: ✅ Fixed (January 27, 2026)

---

### ES-016: Relative Path Resolution When CWD Changes

**Issue**: Model paths failed when current working directory changed during Isaac Sim startup

**Symptoms**: 
```
FileNotFoundError: [Errno 2] No such file or directory: '../logs/spot_eureka/best_policy.pt'
```

**Root Cause**: Isaac Sim changes CWD during initialization, breaking relative paths

**Solution**: Resolve paths relative to script location, not CWD:
```python
# Get script directory for reliable path resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
default_model = os.path.join(script_dir, "logs", "spot_eureka", "best_policy.pt")

# Convert relative paths to absolute
if not os.path.isabs(args.model):
    args.model = os.path.join(script_dir, args.model)
```

**Prevention**: ALWAYS use absolute paths or script-relative paths in Isaac Sim

**Status**: ✅ Fixed (January 27, 2026)

---

### ES-017: UsdGeom.Xformable API for Visual Markers

**Issue**: `GetAttribute("xformOp:translate")` returns empty attribute, causing visual marker placement to fail

**Symptoms**:
```
USD Warning: Coding Error: in Set at line X of pxr/usd/usd/attribute.cpp
```

**Root Cause**: USD requires explicit creation of transform operations before setting values

**Solution**: Use proper UsdGeom.Xformable API:
```python
# Wrong - GetAttribute may return empty
prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(x, y, z))

# Correct - AddTranslateOp creates if needed
xformable = UsdGeom.Xformable(prim)
xformable.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
xformable.AddScaleOp().Set(Gf.Vec3f(size, size, size))
```

**Prevention**: Always use UsdGeom.Xformable wrapper for USD transform operations

**Status**: ✅ Fixed (January 27, 2026)

---

### ES-018: Checkpoint Format Mismatch - Direct State Dict vs Wrapped Dictionary

**Issue**: Model loading failed with `KeyError: 'policy_state_dict'`

**Root Cause**: Training script saves raw state_dict directly, not wrapped in dictionary:
```python
# Training saves this way:
torch.save(policy.state_dict(), "best_policy.pt")

# But evaluation expected:
checkpoint = torch.load("best_policy.pt")
policy.load_state_dict(checkpoint['policy_state_dict'])  # KeyError!
```

**Solution**: Handle both formats:
```python
checkpoint = torch.load(model_path, weights_only=True)

# Check if it's a wrapped checkpoint or direct state_dict
if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
    policy.load_state_dict(checkpoint['policy_state_dict'])
elif isinstance(checkpoint, dict) and all(k.startswith(('policy', 'value', 'features')) for k in checkpoint.keys()):
    # Direct state_dict format
    policy.load_state_dict(checkpoint)
else:
    raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")
```

**Prevention**: Standardize checkpoint format across all scripts, or always check format before loading

**Status**: ✅ Fixed (January 27, 2026)

---

### ES-019: CRITICAL - Eureka Reward Function Stability vs Locomotion Imbalance

**Issue**: Trained policy learns to BALANCE in place instead of WALKING toward goal

**Symptoms**:
- Robot stands perfectly stable but moves only 0.02-0.24m over 1000 timesteps
- Episode rewards converge but policy doesn't locomote
- 0% success rate (never reaches goal)

**Root Cause**: Eureka-generated reward function priorities imbalanced:

```python
# From best_reward_final.py - Problematic weights:
stability_weight = 0.5        # Too high - encourages standing still
height_weight = 0.3           # Penalizes deviation from standing height
progress_weight = 0.3         # Too low relative to stability
velocity_weight = 0.2         # Too low to overcome stability preference
fall_penalty = -10.0          # Large penalty makes robot risk-averse
```

**Analysis**: The policy found a local optimum - standing perfectly still gives:
- ✅ High stability reward (no movement = perfect stability)
- ✅ Perfect height reward (standing height maintained)
- ✅ Zero fall penalty (no fall risk)
- ❌ Zero progress reward (but this is outweighed by above)

**Mathematical Breakdown** (per timestep):
- Standing still: `0.5*stability + 0.3*height + 0*progress = ~0.8`
- Walking: `0.2*stability + 0.1*height + 0.3*progress - movement_penalties = ~0.4`

The policy correctly optimized for the reward as written, but the reward didn't represent our actual goal.

**Solutions** (not yet implemented):
1. **Increase locomotion weights**: `progress_weight = 1.0`, `velocity_weight = 0.5`
2. **Add stillness penalty**: Punish zero velocity
3. **Curriculum approach**: Start with locomotion-only reward, add stability later
4. **Remove/reduce stability term**: Let natural physics enforce stability

**Prevention**: Before training, verify reward function incentivizes primary objective (locomotion) more than secondary objectives (stability)

**Status**: 🔴 NOT FIXED - Requires reward function redesign

---

### ES-020: Policy Evaluation - Distinguishing Training Success from Task Success

**Issue**: Training metrics showed "success" (converged rewards, max episode length) but policy failed the actual task

**Symptoms**:
- Best reward: -4,646.55 (improved from initial)
- Episode length: 1000/1000 (maxed out - good sign?)
- But: Robot doesn't reach goal (task failure)

**Root Cause**: Reward convergence ≠ Task completion. The policy optimized the reward function perfectly - it just wasn't the right reward function.

**Key Insight**: Episode length hitting maximum can mean two very different things:
1. ✅ Robot is doing well and surviving (good)
2. ❌ Robot is doing nothing but surviving (bad - our case)

**Solution**: Always include task-specific success metrics during training:
```python
# Track actual task metrics, not just reward
metrics_to_track = {
    'reward': current_reward,
    'episode_length': step_count,
    # ADD THESE:
    'distance_to_goal': np.linalg.norm(robot_pos - goal_pos),
    'distance_traveled': total_distance_moved,
    'average_velocity': total_distance / time,
    'goal_reached': distance_to_goal < success_threshold,
}
```

**Prevention**: 
1. Define success criteria BEFORE training (not just reward)
2. Log task-specific metrics alongside reward
3. Evaluate qualitatively (visually) during training, not just at end

**Status**: ✅ Lesson Learned - Apply to future training

---

### ES-021: Curriculum Training Failure - "Downward Dog" Posture Collapse

**Issue**: 6M timestep curriculum training produced policy that either falls immediately or enters "downward dog" (face-down crouch) position

**Training Configuration**:
```python
# 3-Phase Curriculum (6M timesteps total):
Phase 1 (0-2.4M):   progress=5.0, velocity=3.0, stability=0.0, stillness=-2.0
Phase 2 (2.4M-4.2M): stability blends from 0.0 → 1.0
Phase 3 (4.2M-6M):   progress=5.0, velocity=3.0, stability=1.0, efficiency=0.5
```

**Observed Behavior**:
1. **Phase 1**: Robot learned fast locomotion (~1.0 m/s) but highly unstable - fell every 20-30 steps
2. **Phase 2**: As stability blended in, rewards started degrading
3. **Phase 3**: Complete collapse - rewards dropped to -180,000 range, robot entered "downward dog"

**Terminal Output Evidence**:
```
Steps: 4,964,344 | Reward: -8741.7 | Dist: 0.59m | Speed: 1.069m/s | Goals: 346/72596 | Phase 3 (blend=1.00)
💥 Robot fallen at step 25
```

**Visual Evaluation Results** (5 episodes):
| Episode | Steps | Result | Notes |
|---------|-------|--------|-------|
| 1 | 1000 | Timeout | Walked 10m but drifted sideways, never reached goal |
| 2 | 43 | Fall | Immediate instability |
| 3 | 28 | Fall | Immediate instability |
| 4 | 28 | Fall | Immediate instability |
| 5 | 30 | Fall | Immediate instability |

**Success Rate**: 0/5 (0%)

**Root Cause Analysis**:

1. **No Base Posture Requirement**: Phase 1 had zero stability, allowing robot to learn wild, uncontrolled gaits
2. **Incompatible Skill Transfer**: The fast-but-unstable gait from Phase 1 couldn't adapt when stability was enforced
3. **Local Minimum in Phase 3**: When penalized for instability, robot found local optimum: "downward dog" minimizes movement penalties
4. **Best Model Corruption**: `best_policy.pt` was saved during Phase 3 chaos with terrible behavior

**"Downward Dog" Position Explained**:
- Robot crouches face-down with front legs extended forward
- This position minimizes: height deviation, body angle, joint velocities
- But is completely non-functional for locomotion
- Stability reward designed for STANDING was exploited for CROUCHING

**Solution for Next Training**:

Add **Phase 0: Standing Curriculum** before locomotion:
```python
# New 4-Phase Curriculum:
Phase 0 (0-15%):    Stand on all fours with WIDE tolerances
                    - height_target=0.5m, tolerance=±0.15m
                    - body_angle_tolerance=±20°
                    - leg_angle_tolerance=±30°
                    - NO movement required, just stable stance

Phase 1 (15-50%):   Locomotion with minimum stability (20%)
                    - progress=5.0, velocity=3.0, stability=0.2
                    - Stillness penalty active

Phase 2 (50-80%):   Blend stability up to 60%
                    - Gradually tighten tolerances

Phase 3 (80-100%):  Full reward with tight tolerances
```

**Key Changes**:
1. **Standing before walking**: Robot must learn stable quadruped stance first
2. **Wide tolerances initially**: ±15m height, ±20° angles acceptable at start
3. **Never zero stability**: Always maintain minimum 20% stability weight
4. **Explicit standing reward**: Reward all four feet on ground, body upright

**Status**: 🔴 FAILED (February 5, 2026) - Requires redesigned curriculum

---

### ES-022: Isaac Sim Robot default_pos Returns List, Not NumPy Array

**Issue**: `AttributeError: 'list' object has no attribute 'shape'`

**Context**: When creating `train_spot_curriculum_v2.py` with standing curriculum, accessing `self.spot.default_pos` and calling `.copy()` or `.shape` fails.

**Root Cause**: The `SpotFlatTerrainPolicy.default_pos` property returns a Python list, not a NumPy array:
```python
# WRONG - causes AttributeError
self.default_joint_pos = self.spot.default_pos.copy()  # list has no .copy()
print(f"Shape: {self.default_joint_pos.shape}")        # list has no .shape

# ERROR:
# AttributeError: 'list' object has no attribute 'shape'
```

**Solution**: Explicitly convert to NumPy array:
```python
# CORRECT
self.default_joint_pos = np.array(self.spot.default_pos)
print(f"Default joint positions: {len(self.default_joint_pos)} joints")  # Use len() instead of .shape
```

**Key Lesson**: Never assume Isaac Sim robot properties return NumPy arrays. Always explicitly convert:
- `np.array(robot.default_pos)` for joint positions
- `np.array(robot.get_joint_positions())` for current positions
- Check type before using NumPy methods

**Status**: ✅ FIXED (February 5, 2026)

---

### ES-023: Observation Dimension Mismatch in Custom RL Training

**Issue**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x49 and 56x256)`

**Context**: Custom PPO policy network initialized with `obs_dim=56` but actual observation vector had 49 elements.

**Root Cause**: Observation dimension was calculated incorrectly in environment class:
```python
# WRONG - miscounted observation components
self.obs_dim = 48 + 3 + 1 + 4  # = 56

# But actual observation in _get_observation() was:
obs = np.concatenate([
    joint_pos,                          # 12
    joint_vel,                          # 12
    self.prev_action,                   # 12
    [pos[2]],                           # 1
    [roll, pitch, yaw],                 # 3
    goal_dir,                           # 2
    [goal_dist / 20.0],                 # 1
    [np.sin(rel_yaw), np.cos(rel_yaw)], # 2
    [roll, pitch],                      # 2
    [pos[2] - self.target_height],      # 1
    [float(self.phase)],                # 1
])  # Total: 12+12+12+1+3+2+1+2+2+1+1 = 49, NOT 56!
```

**Solution**: Count observation components explicitly with comments:
```python
# Observation: joint_pos(12) + joint_vel(12) + prev_action(12) + height(1) + 
#             orientation(3) + goal_dir(2) + goal_dist(1) + rel_yaw(2) + 
#             body_angles(2) + height_error(1) + phase(1) = 49
self.obs_dim = 49
```

**Prevention Pattern**:
```python
# Best practice: Define observation once and compute dimension
self.obs_components = {
    'joint_pos': 12,
    'joint_vel': 12,
    'prev_action': 12,
    'height': 1,
    'orientation': 3,
    'goal_dir': 2,
    'goal_dist': 1,
    'rel_yaw': 2,
    'body_angles': 2,
    'height_error': 1,
    'phase': 1,
}
self.obs_dim = sum(self.obs_components.values())  # Automatically correct!
```

**Key Lesson**: When adding new observation components:
1. Update `_get_observation()` method
2. Update `obs_dim` calculation
3. Update comments documenting the observation space
4. Consider using a dictionary to auto-compute dimension

**Status**: ✅ FIXED (February 5, 2026)

---

### ES-024: Standing Curriculum Impossible - Raw Joint Control Cannot Learn to Stand

**Issue**: V2 curriculum training with "Phase 0: STANDING" was stuck indefinitely - robot fell at step 2 every single episode, no learning occurred.

**Context**: After V1 curriculum failed with "downward dog" posture (ES-021), attempted V2 with explicit standing phase (Phase 0) where robot had to maintain standing posture before being allowed to walk. Training ran for 1.7 hours (276K+ steps, 117K+ episodes) with:
- 0 goals reached
- Reward stuck at 8.7
- Every episode: "Robot fallen at step 2"

**Root Cause**: Training raw joint control from scratch cannot learn to stand:
1. **Random policy = immediate collapse**: At training start, policy outputs random 12-dimensional joint positions
2. **Any random joint configuration = fall**: The Spot robot has 12 joints that must be coordinated precisely to maintain balance
3. **2-step episodes = no gradient signal**: Only 2 steps of data collected per episode provides insufficient information for PPO to learn
4. **Standing is a CONTROL problem, not a REWARD problem**: No amount of reward engineering helps if physics doesn't allow exploration

**The Math Problem**:
```
Episode length: 2 steps
Data per episode: 2 observation-action pairs
Required coordination: 12 joints simultaneously
Probability of random config being stable: ~0%
Gradient signal: Near zero (reward variance too low)
```

**Why SpotFlatTerrainPolicy Works**:
- Boston Dynamics spent YEARS developing Spot's locomotion controller
- Pre-trained neural network handles 12-joint coordination
- Uses proprioceptive feedback + model-based control
- Already knows "standing" as a base capability

**Solutions Ranked**:
1. **Best: Use pre-trained locomotion** - Train high-level velocity commands [vx, vy, wz] that SpotFlatTerrainPolicy converts to stable walking (V3 approach)
2. **Alternative: Imitation learning** - Clone expert demonstrations of standing/walking
3. **Possible but slow: Massively parallel training** - 4096+ parallel envs with domain randomization (IsaacLab approach)
4. **Not recommended: Longer training** - 10M+ steps might eventually find standing, but inefficient

**V3 Architecture Change**:
```
V1/V2 (FAILED): 
  Observation → Policy → 12 joint positions → Physics → Falls immediately

V3 (PROPOSED):
  Observation → Policy → 3 velocity commands (vx, vy, wz) → SpotFlatTerrainPolicy → Stable walking
```

**Key Insight**: Don't train what's already solved. Use SpotFlatTerrainPolicy as a "motor primitive" and train navigation on top of it.

**Prevention Pattern**:
- For quadruped robots: Always use pre-trained locomotion controller for low-level control
- For RL: Only train the parts that need training (navigation, obstacle avoidance, etc.)
- Validate standing/walking works BEFORE adding RL training loop

**Status**: ✅ DOCUMENTED (February 5, 2026) - Created V3 approach using pre-trained locomotion

---

### ES-025: Height Scan Fill Value Mismatch — Standalone Policy Deployment

**Issue**: Rough terrain policy (30k iterations, H100) fell over within 1.5 seconds when switching to ROUGH gait in the obstacle course, despite working perfectly in Isaac Lab's `play.py`.

**Context**: After the 48h training run completed successfully (model_29999.pt), the policy was integrated into `spot_obstacle_course.py` with dual gait switching (G key / RB button toggles FLAT ↔ ROUGH). The policy wrapper (`spot_rough_terrain_policy.py`) filled the 187-dimension height scan with a constant value.

**Root Cause**: The height scan fill value was **1.0** but should have been **0.0**.

The original analysis (see ROUGH_POLICY_DEBUG_HANDOFF.md, Section 5 "Height Scan = 1.0 is Correct") traced through Isaac Lab's `observations.py` and `ray_caster.py` and concluded that the 20m Z-offset in the RayCaster would make height_scan clip to 1.0. **This analysis was wrong.**

**Definitive proof**: Running the actual Isaac Lab training environment and printing raw observations showed:
```
height_scan range: [-0.000002, 0.148083]
height_scan mean:  0.003959
```
The height scan is approximately **0.0** on flat ground, NOT 1.0.

**Diagnostic**: A parameter sweep of the trained actor confirmed extreme sensitivity:
| height_scan value | Action norm | Behavior |
|-------------------|-------------|----------|
| 0.0 | 3.08 | Normal walking |
| 0.2 | 2.37 | Optimal (minimum) |
| 0.5 | 4.12 | Aggressive |
| 1.0 | **7.42** | **Catastrophic — instant fall** |

With `hs=1.0`, the policy produced action norms 2.4x larger than with `hs=0.0`, generating joint commands that immediately destabilized the robot.

**Fix**: In `spot_rough_terrain_policy.py`:
```python
# WRONG (caused falls):
SCAN_CLIP_HI = 1.0
obs[48:235] = SCAN_CLIP_HI  # Filled with 1.0

# CORRECT (robot walks):
SCAN_FILL_VAL = 0.0
obs[48:235] = SCAN_FILL_VAL  # Filled with 0.0 (flat ground assumption)
```

**Why the original analysis was wrong**: The height_scan formula is:
```python
height_scan = sensor.data.pos_w[:, 2] - ray_hits_w[..., 2] - offset
```
Where `offset` defaults to 0.5. On flat ground: `(body_z + 20.0) - ground_z - 0.5`. But the *body_z* is the actual robot height (~0.5m), and the RayCaster applies transforms that make ground hits resolve to ~20.5m. The result after offset subtraction is approximately 0.0, not 1.0. The source code tracing missed how `combine_frame_transforms` resolves the actual ray hit positions.

**Key Lesson**: **Never trust source code tracing alone for understanding observation values. Always print the actual runtime values from the training environment.** A 2-minute diagnostic script would have caught this immediately. Instead, days were spent debugging other hypotheses (GPU vs CPU PhysX, solver iterations, tensor types).

**Impact**: This was the **primary blocker** for standalone rough terrain policy deployment. With this fix:
- Action norms: 7.20 → 3.08 (57% reduction)
- Robot behavior: instant fall → stable walking
- Obstacle course: now fully functional with dual gait switching

**Status**: ✅ FIXED (February 16, 2026)

---

### ES-026: GPU PhysX Required for Trained Policy Deployment

**Issue**: Policies trained in Isaac Lab (GPU PhysX) do not transfer cleanly to standalone Isaac Sim (CPU PhysX by default).

**Root Cause**: GPU and CPU PhysX implementations produce subtly different dynamics for articulated bodies. The policy's learned actions are tuned to GPU PhysX's specific constraint solver behavior.

**Solution**: Use GPU PhysX in standalone deployment:
```python
from isaacsim.core.api import World

world = World(
    physics_dt=1.0/500.0,
    rendering_dt=10.0/500.0,
    backend="torch",
    device="cuda:0",
)
```
With `NumpyRobotWrapper` to convert between CUDA tensors and numpy for the robot API:
```python
class NumpyRobotWrapper:
    """Wraps ArticulationView for GPU PhysX — converts torch tensors to numpy."""
    def get_joint_positions(self):
        t = self._av.get_joint_positions()
        return t.cpu().numpy()[0] if isinstance(t, torch.Tensor) else np.array(t).flatten()
```

**Key Points**:
- `set_joint_efforts(numpy_array)` **silently does nothing** on GPU PhysX — must use CUDA tensors
- `set_joint_positions()` similarly requires CUDA tensors
- `backend="torch"` + `device="cuda:0"` enables GPU PhysX pipeline
- NumpyRobotWrapper class handles all conversions transparently

**Status**: ✅ IMPLEMENTED (February 16, 2026)

---

## Do Not Repeat (DNR)

### DNR-G001: Uniform Grass Distribution
**Issue**: Initial grass generation used perfectly uniform distribution  
**Problem**: Unrealistic; real grass has natural variation  
**Solution**: Add 20-30% local density variation  
**Status**: 🔄 To be validated

### DNR-G002: Ignoring Grass Physics Timestep
**Issue**: Grass physics computed at render rate (50 Hz)  
**Problem**: Unstable grass behavior, jittering  
**Solution**: Grass physics at simulation rate (500 Hz)  
**Status**: 🔄 To be validated

### DNR-G003: Static Grass Model
**Issue**: Grass blades don't bend or recover  
**Problem**: No physical interaction feedback  
**Solution**: Implement spring-damper model per blade  
**Status**: 🔄 To be validated

### DNR-G004: Ignoring Sensor Occlusion
**Issue**: LiDAR rays pass through grass  
**Problem**: Unrealistic perception  
**Solution**: Enable grass collision for sensor rays  
**Status**: 🔄 To be validated

### DNR-G005: Belly Contact Not Tracked
**Issue**: Robot body passing through tall grass undetected  
**Problem**: Missing important metric  
**Solution**: Add belly contact force sensor  
**Status**: 🔄 To be validated

### DNR-G006: Eureka Reward Stability-Locomotion Imbalance ⚠️ CRITICAL
**Issue**: Eureka-generated reward function prioritized stability over locomotion  
**Problem**: Policy learned to stand still instead of walk (local optimum)  
**Evidence**: 0.09m average progress in 3000 total steps, 0% goal success  
**Root Cause**: `stability_weight=0.5` > `progress_weight=0.3` + `velocity_weight=0.2`  
**Solution**: 
1. Primary objective (locomotion) weight MUST exceed secondary (stability)
2. Add explicit stillness penalty: `stillness_penalty = -0.5 * (velocity < 0.01)`
3. Consider removing stability reward entirely - physics enforces it naturally  
**Status**: 🔴 VIOLATED (January 27, 2026) - Caused complete training failure

### DNR-G008: Curriculum Without Base Posture Requirements ⚠️ CRITICAL
**Issue**: Curriculum that removes stability rewards entirely in early phases produces unstable gaits  
**Problem**: Robot learns to move fast but falls constantly; when stability reintroduced, policy collapses  
**Evidence**: 
- Phase 1 (pure locomotion): Robot learned ~1.0 m/s speed but fell every 20-30 steps
- Phase 3 (full stability): Rewards collapsed to -180,000, robot entered "downward dog" posture
- Final evaluation: 0/5 success, episodes 2-5 fell within 30 steps
**Root Cause**: Three-phase curriculum design flaw:
```python
# Problematic curriculum (6M timesteps):
Phase 1 (0-40%):  progress=5.0, velocity=3.0, stability=0.0   # No stability!
Phase 2 (40-70%): stability gradually increases 0.0 → 1.0
Phase 3 (70-100%): progress=5.0, velocity=3.0, stability=1.0  # Sudden full stability
```
When stability rewards turned on in Phase 3, the fast-but-unstable gait was heavily penalized, causing:
1. Massive negative rewards (-180K range)
2. Policy learned to minimize penalties by going into "downward dog" (face-down crouch)
3. "Best" model saved during Phase 3 chaos
**Solution**: 
1. ALWAYS include minimum stability from start (at least 20%)
2. Never fully remove posture requirements
3. Add explicit "standing on all fours" curriculum phase before locomotion
4. Use wide tolerances initially, tighten gradually
**Status**: 🔴 VIOLATED (February 5, 2026) - 6M timestep training wasted

### DNR-G009: Trusting Source Code Tracing Over Runtime Values
**Issue**: Height scan fill value set to 1.0 based on source code analysis of RayCaster
**Problem**: Policy produced catastrophic action norms (7.42), causing instant robot collapse
**Evidence**: Actual runtime height_scan values were ~0.0, not 1.0. Source code tracing missed how `combine_frame_transforms` resolves ray hit positions.
**Solution**: ALWAYS print actual observation values from the training environment before deploying. A 2-minute diagnostic script catches what hours of source code reading misses.
**Status**: 🔴 VIOLATED (February 16, 2026) — Days wasted debugging wrong hypotheses

### DNR-G007: Assuming Training Convergence = Task Success
**Issue**: Reward convergence used as sole indicator of training success  
**Problem**: Policy can optimize wrong objective perfectly (standing still = high reward)  
**Evidence**: Best reward improved to -4,646.55 but robot never reached goal  
**Solution**: Always track task-specific metrics (distance_to_goal, velocity, success_rate) alongside reward  
**Status**: 🔴 VIOLATED (January 27, 2026) - Delayed discovery of failure

---

## Best Practices

### BP-G001: Grass Zone Boundary Buffer
**Practice**: Clear 0.5m buffer between grass zone and room walls  
**Rationale**: Prevents robot from getting trapped at boundary  
**Implementation**: Define grass zone as interior region only

### BP-G002: Height-Dependent Friction Model
**Practice**: Use μ_effective = μ_base + (H × 0.4)  
**Rationale**: Taller grass creates more resistance  
**Implementation**: Update friction per grass height level

### BP-G003: Grass Clearance Reward
**Practice**: Reward robot for maintaining leg lift above grass height  
**Rationale**: Encourages proper gait modification  
**Implementation**: R = +0.2 if leg_clearance > H_grass

### BP-G004: Gait Smoothness Metric
**Practice**: Track phase consistency across legs  
**Rationale**: Grass terrain disrupts normal gait  
**Implementation**: GS = 1 / (1 + σ_phase_offset)

### BP-G005: Progressive Height Exposure
**Practice**: Train sequentially from H1 → H2 → H3 → H4  
**Rationale**: Sudden tall grass causes policy collapse  
**Implementation**: Curriculum learning approach

---

## Grass-Specific Insights

### GSI-001: Height Impact on Navigation (ES-010 Approach)

| Height | Base Friction | Speed Penalty | Nav Time | Grass Time | TCR |
|--------|---------------|---------------|----------|------------|-----|
| H1 (0.1m) | 0.95 | 95% | 17.3s | 11.8s | 100% |
| H2 (0.3m) | 0.92 | 92% | 18.3s | 12.4s | 100% |
| H3 (0.5m) | 0.90 | 90% | 19.2s | 13.5s | 100% |
| H4 (0.7m) | 0.85 | 85% | 21.1s | 15.0s | 100% |

**Insight**: Friction-based grass creates measurable slowdown (10-20%) without causing falls. Linear relationship between height and traversal time.

### GSI-002: Density Impact on Navigation

| Density | Blades/m² | Friction Mod | Speed Penalty | Nav Time | TCR |
|---------|-----------|--------------|---------------|----------|-----|
| G1 | 100 | +0.02 | 95% | 17.7s | 100% |
| G2 | 200 | +0.03 | 90% | 19.0s | 100% |
| G3 | 300 | +0.05 | 85% | 21.4s | 100% |

**Insight**: Density affects energy/time more than path efficiency. 21% increase in traversal time from G1 to G3.

### GSI-003: Height × Density × Obstacles Interaction

**Phase 4 Finding**: Combined H3 + G2 + O1 creates 60% failure rate (TCR = 40%)

| Configuration | TCR | Primary Failure Mode |
|---------------|-----|----------------------|
| H3 + G2 + O0 (control) | 100% | None |
| H3 + G2 + O1 (18 obstacles) | 40% | Falls (70%), Stalls (30%) |

**Critical Finding**: SpotFlatTerrainPolicy has ZERO obstacle avoidance capability. Both visible and hidden obstacles cause equal collision rates (1.72 vs 1.62).

### GSI-004: Hidden Obstacle Danger

**Finding**: Obstacles hidden in grass (height < grass height) are extremely dangerous.

| Obstacle Type | Height | Visibility | Collision Impact |
|---------------|--------|------------|------------------|
| TallCube | 0.8m | Visible | Stall (robot stops) |
| Cylinder | 0.6m | Visible | Stall/Fall |
| LowBlock | 0.3m | Hidden | **Trip → Fall** |

**Key Insight**: LowBlock obstacles hidden in H3 (0.5m) grass cause trips and falls because the robot's legs catch on them unexpectedly. This is the most dangerous obstacle type.

### GSI-005: Grass Zone Friction Validation

**Finding**: ES-010 friction-based grass physics works correctly:
- Robot enters grass zone at ~3-4 seconds
- Speed penalty applied immediately
- Exit grass zone at ~17-19 seconds (depending on conditions)
- No physics instability or falls due to grass alone

---

## Platform Comparisons

### PC-001: Spot vs. V60 in Tall Grass

| Aspect | Spot | Vision 60 |
|--------|------|-----------|
| Leg clearance (H3) | Marginal | Good |
| Leg clearance (H4) | Insufficient | Marginal |
| Belly contact risk | High | Low |
| Camera occlusion | Higher | Lower |
| Energy efficiency | Lower | Higher |

**Summary**: V60's longer legs provide significant advantage in tall grass

### PC-002: Gait Adaptation Comparison

| Behavior | Spot | Vision 60 |
|----------|------|-----------|
| Step height increase | Limited by leg length | More range available |
| Gait modification | Required at H2+ | Required at H3+ |
| Stability in grass | Good | Very good |

### PC-003: Sensor Configuration Impact

| Sensor | Spot Position | V60 Position | Grass Impact |
|--------|--------------|--------------|--------------|
| Front camera | 0.5m height | 0.65m height | V60 better visibility |
| LiDAR | Body-mounted | Body-mounted | Similar occlusion |

---

## Phase-by-Phase Notes

### Phase 1: Baseline (No Grass)
- **Date**: January 26, 2026
- **Key Finding**: SpotFlatTerrainPolicy achieves consistent ~17.5s navigation across flat terrain
- **TCR**: 100% (10/10 runs)
- **Mean Navigation Time**: 17.5s
- **Issues**: Initial API compatibility issues with Isaac Sim 5.1.0 (advance→forward, _default_joint_pos removed)
- **Resolution**: Updated API calls per ES-002 and ES-004

### Phase 2: Grass Height Variation (ES-010 Friction-Based)
- **Date**: January 26, 2026
- **Key Finding**: Friction-based grass physics achieves 100% TCR across ALL height levels (H1-H4)
- **TCR**: 100% for all levels
- **Mean Navigation Times**:
  | Level | Height | Nav Time | Grass Time |
  |-------|--------|----------|------------|
  | H1 | 0.1m | 17.3s | 11.8s |
  | H2 | 0.3m | 18.3s | 12.4s |
  | H3 | 0.5m | 19.2s | 13.5s |
  | H4 | 0.7m | 21.1s | 15.0s |
- **H_opt Selected**: H3 (0.5m) - Moderate challenge with measurable slowdown, optimal for training
- **Issues**: Initial collision-cylinder approach caused 100% fall rate
- **Resolution**: ES-010 friction-based approach with speed penalties (no collision bodies)

### Phase 3: Grass Density Variation
- **Date**: January 26, 2026  
- **Key Finding**: Friction-based density affects traversal time but maintains 100% TCR
- **TCR**: 100% for all density levels (G1, G2, G3)
- **Mean Navigation Times** (at H_opt = H3):
  | Level | Blades/m² | Nav Time | Grass Time | Speed Penalty |
  |-------|-----------|----------|------------|---------------|
  | G1 | 100 | 17.7s | 11.9s | 95% |
  | G2 | 200 | 19.0s | 13.2s | 90% |
  | G3 | 300 | 21.4s | 15.4s | 85% |
- **G_opt Selected**: G2 (200 blades/m²) - Moderate density, 90% speed penalty
- **Issues**: None - friction-based approach scaled well
- **Resolution**: N/A

### Phase 4: Combined Grass + Obstacles
- **Date**: January 26-27, 2026
- **Key Finding**: Hidden obstacles in grass cause significant navigation failures
- **Test Configuration**:
  - H_opt: H3 (0.5m height, 0.90 base friction)
  - G_opt: G2 (200 blades/m², 0.03 friction modifier, 90% speed penalty)
  - Obstacles: O0 (control, 0 obstacles), O1 (18 obstacles: 6 TallCube, 6 Cylinder, 6 LowBlock)
- **Results**:
  | Level | Total Runs | TCR | Mean Nav Time | Mean Collisions | Falls | Stalls |
  |-------|------------|-----|---------------|-----------------|-------|--------|
  | O0 | 10 | 100% | 19.0s | 0.00 | 0 | 0 |
  | O1 | 50 | 40% | 23.1s | 3.34 (1.72v/1.62h) | 21 | 9 |
- **Critical Insights**:
  1. Control baseline (O0) validates Phase 3 consistency
  2. Hidden obstacles (50% below grass) equally dangerous as visible (1.62 vs 1.72 collisions)
  3. Falls are primary failure mode (21/30 failures)
  4. SpotFlatTerrainPolicy lacks obstacle avoidance capability
- **Issues**: Robot cannot detect hidden obstacles, leading to trips and falls
- **Resolution**: Future work - implement obstacle detection or RL training for robust traversal

### Phase 5: Advanced RL Training (Eureka Attempt)
- **Date**: February 4, 2026
- **Key Finding**: Eureka-generated reward functions can produce policies that optimize for stability over locomotion
- **Issues**: Multiple critical issues encountered (see ES-015 through ES-020)
- **Resolution**: Need to manually tune reward weights to emphasize forward progress over stability

#### Phase 5 Summary
- **Training Duration**: 500K timesteps (~51 minutes)
- **Best Reward**: -4,646.55
- **Episode Length**: Maxed at 1000 steps
- **Visual Evaluation Result**: 0/3 success rate, robot moved only 0.09m average over 3 episodes
- **Root Cause**: Policy learned to balance in place rather than walk forward

#### Phase 5 Detailed Results
| Episode | Steps | Forward Progress | Final Distance to Goal |
|---------|-------|------------------|----------------------|
| 1 | 1000 | 0.24m | 16.06m |
| 2 | 1000 | 0.02m | 16.28m |
| 3 | 1000 | 0.02m | 16.28m |
| **Avg** | **1000** | **0.09m** | **16.21m** |

**Start Distance**: 16.30m → **Net Progress**: 0.09m (0.55%)

### Phase 6: Cross-Platform Transfer
- **Date**: TBD
- **Key Finding**: TBD
- **Issues**: TBD
- **Resolution**: TBD

### Phase 7: Analysis
- **Date**: TBD
- **Key Finding**: TBD

---

## Environment Setup Lessons

### ESL-001: Grass Generation Performance
**Issue**: Dense grass (G3) causes significant performance drop  
**Mitigation**: Use LOD (Level of Detail) for distant grass  
**Target**: Maintain 50+ FPS at G3

### ESL-002: Grass Collision Optimization
**Issue**: Per-blade collision is expensive  
**Solution**: Use collision groups/volumes, not individual blades  
**Implementation**: Grass zone as single collision volume with height-dependent friction

### ESL-003: Grass Rendering vs. Physics
**Issue**: Visual grass count ≠ physics grass count  
**Solution**: Decouple render instances from physics instances  
**Approach**: 10:1 ratio (10 visual blades per 1 physics blade)

---

## RL Training Lessons (Grass-Specific)

### RTL-001: Grass Clearance Reward Tuning
**Finding**: Clearance reward weight of 0.2 works well  
**Issue if too high**: Robot over-lifts, wastes energy  
**Issue if too low**: Robot drags through grass

### RTL-002: Observation Space for Grass
**Recommended additions**:
- Local grass height estimate
- Recent grass interaction forces
- Grass density ahead (from camera)

### RTL-003: Curriculum for Grass Height
**Successful approach**:
```
Episode 0-1000: H1 only
Episode 1000-3000: H1-H2 random
Episode 3000-6000: H1-H3 random
Episode 6000+: H1-H4 random
```

---

## Comparison to Flat Room Experiment

### Key Differences

| Aspect | Flat Room | Grass Terrain |
|--------|-----------|---------------|
| Primary challenge | Obstacle avoidance | Terrain traversal |
| Sensor focus | Distance sensing | Occlusion handling |
| Gait modification | Not required | Essential |
| Energy efficiency | Lower priority | Higher priority |
| Platform difference | Minimal | Significant |

### Transferable Learnings

From flat room experiment:
1. ✅ Single-room scope approach
2. ✅ Phased validation methodology
3. ✅ Cross-platform transfer protocol
4. ✅ Variable matrix structure

Grass-specific additions:
1. ➕ Height-based friction model
2. ➕ Sensor occlusion tracking
3. ➕ Gait smoothness metric
4. ➕ Combined phase (grass + obstacles)

---

## Open Questions

### OQ-001: Optimal Grass Height for Training ✅ ANSWERED
**Question**: What is the ideal training height for generalization?  
**Hypothesis**: H2-H3 (moderate) produces best transfer  
**Answer**: H3 (0.5m) selected as H_opt. Provides measurable challenge (19.2s vs 17.5s baseline) while maintaining 100% TCR. Tall enough to create hidden obstacle scenarios.
**Status**: ✅ Answered in Phase 2

### OQ-002: Grass Density Effect Threshold ✅ ANSWERED
**Question**: At what density does navigation fundamentally change?  
**Hypothesis**: G2 to G3 transition is critical  
**Answer**: Navigation remains viable at all densities (100% TCR). G2 selected as G_opt for moderate challenge. G3 adds 13% more time but doesn't fundamentally change success rate.
**Status**: ✅ Answered in Phase 3

### OQ-003: Obstacle Impact in Grass ✅ ANSWERED (NEW)
**Question**: How do obstacles in grass affect navigation success?  
**Hypothesis**: Hidden obstacles would be more dangerous than visible  
**Answer**: Both equally dangerous (1.72 vs 1.62 collisions). SpotFlatTerrainPolicy has no obstacle avoidance. TCR drops from 100% → 40% with 18 obstacles.
**Status**: ✅ Answered in Phase 4

### OQ-004: Cross-Platform Policy Transfer in Grass
**Question**: How much does V60's height advantage affect transfer?  
**Hypothesis**: V60 policies may not transfer well to Spot  
**Status**: To be tested in Phase 6

---

## Update Log

| Date | Phase | Contributor | Changes |
|------|-------|-------------|---------|
| Jan 26, 2026 | Setup | Gabriel Santiago | Initial document, ES-001 through ES-010 |
| Jan 26, 2026 | Phase 1 | Gabriel Santiago | Baseline tests completed, API issues resolved |
| Jan 26, 2026 | Phase 2 | Gabriel Santiago | Friction-based grass (ES-010), 100% TCR all heights, H_opt=H3 |
| Jan 26, 2026 | Phase 3 | Gabriel Santiago | Density tests completed, 100% TCR all levels, G_opt=G2 |
| Jan 27, 2026 | Phase 4 | Gabriel Santiago | Obstacle tests completed, ES-011 through ES-014, O1 TCR=40% |
| Jan 27, 2026 | Phase 5 | Gabriel Santiago | **EUREKA ATTEMPT FAILED** - ES-015 through ES-020, DNR-G006, DNR-G007 added |
| Feb 5, 2026 | Phase 5 | Gabriel Santiago | **CURRICULUM V1 FAILED** - ES-021, DNR-G008 (downward dog collapse) |
| Feb 5, 2026 | Phase 5 | Gabriel Santiago | **V2 BUGS FIXED** - ES-022 (list vs array), ES-023 (obs_dim mismatch) |
| Feb 5, 2026 | Phase 5 | Gabriel Santiago | **V2 STANDING CURRICULUM FAILED** - ES-024 (raw joint control cannot learn to stand), created V3 with pre-trained locomotion |
| Feb 13, 2026 | H100 Training | Gabriel Santiago | 48h H100 training launched (30k iterations, 8192 envs). Debug run lessons in `48h_training/LESSONS_LEARNED.md` |
| Feb 16, 2026 | H100 Training | Gabriel Santiago | **TRAINING COMPLETE** — 30k/30k iterations. Final reward +143.74, terrain level 4.42, gait 5.28 |
| Feb 16, 2026 | Deployment | Gabriel Santiago | **HEIGHT SCAN FIX** — ES-025. Changed fill value 1.0→0.0. Robot now walks in standalone obstacle course |
| Feb 16, 2026 | Deployment | Gabriel Santiago | ES-026 (GPU PhysX required), DNR-G009 (trust runtime values over source tracing) |

### Phase 5 Eureka Training Summary (January 27, 2026)

| Metric | Value |
|--------|-------|
| Timesteps | 500,000 |
| Duration | ~51 minutes |
| Best Reward | -4,646.55 |
| Success Rate | **0%** (robot balances, doesn't walk) |
| Bugs Fixed | 6 (ES-015 through ES-020) |
| Critical Issue | Reward imbalance (stability > locomotion) |

**Action Required**: Redesign reward function before retrying Phase 5.

---

## Cross-References

- `../experimental_design_flat_room/lessons_learned.md`: Flat room learnings
- `/phases/*.md`: Phase protocols
- `/experiments/experiment_matrix.md`: Run tracking
