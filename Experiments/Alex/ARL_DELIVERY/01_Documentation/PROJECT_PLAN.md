# MS for Autonomy - Project Plan

## Problem Statement

**Core Problem**: The Spot robot struggles to navigate cluttered indoor environments—think of a child's messy room filled with toys, boxes, and scattered objects of various sizes. Current locomotion policies work well in open spaces but fail when the robot must weave through, push aside, or carefully step around dense object fields.

**Goal**: Train Spot to autonomously navigate from point A to point B in highly cluttered indoor spaces by learning to:
1. **Navigate around** immovable obstacles (furniture, walls)
2. **Push through** movable clutter (toys, boxes, debris)
3. **Adapt locomotion** based on obstacle density and type
4. **Scale to multi-room** environments with stairs and beams

### Target Environment
- **Indoor rooms** (4m × 4m to 6m × 6m)
- **Flat floor** (carpet/hardwood simulation)
- **Dense clutter**: 10 → 20 → 40+ objects per room
- **Object types**: Varying sizes, movable vs immovable
- **Multi-room**: Connected rooms, doorways, stairs, beam obstacles

---

## Lessons from Previous Work

From the Spot-Quadruped-Training project:

| Lesson | Implication |
|--------|-------------|
| **Physics callback at 500Hz** | Robot control must use `world.add_physics_callback()`, NOT render loop |
| **SpotFlatTerrainPolicy works** | Good baseline for flat ground—build on top of it |
| **State machine navigation** | STABILIZE → NAVIGATE → AVOID → PUSH pattern effective |
| **vy = 0 limitation** | Default policy only moves forward/backward + rotate; no strafing |

### What We Can Reuse
- Physics callback pattern for stable control
- State machine architecture for behavior switching
- SpotFlatTerrainPolicy as the locomotion backbone

### What We Need to Add
- **Obstacle detection & classification** (movable vs immovable)
- **Push behavior** for movable objects
- **Dense clutter navigation** strategy
- **Multi-room awareness** and goal planning
- **Advanced gaits** (crawl, jump) for beam obstacles

---

## Progressive Difficulty Curriculum

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1: Single Room - 10 Movable Objects                          │
│  └── Learn basic clutter navigation + pushing                       │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 2: Single Room - 20 Objects                                  │
│  └── Scale up density, refine avoidance                             │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 3: Single Room - 40 Objects (Mixed Movable/Immovable)        │
│  └── Learn to distinguish + appropriate response                    │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 4: Two Rooms - Increasing Object Count                       │
│  └── Multi-room navigation, doorway traversal                       │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 5: Two Rooms + Stairs                                        │
│  └── Vertical navigation between floors                             │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 6: Beam Obstacles (Crawl + Jump)                             │
│  └── Advanced gaits for horizontal obstacles                        │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 7: Full Integration & Evaluation                             │
│  └── End-to-end testing, benchmarks, stress tests                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Single Room - 10 Movable Objects

### Objectives
- Create room environment with flat floor
- Implement dynamic random object spawning
- Train navigation through sparse clutter
- Learn basic push behavior

### Environment Specification
```
Room: 5m × 5m flat floor, 4 walls
Objects: 10 randomly placed movable items
Object sizes: Small (0.1-0.2m), Medium (0.2-0.4m), Large (0.4-0.6m)
Object types: Cubes, cylinders, spheres (toy-like)
Mass range: 0.5kg - 5kg (pushable by Spot)
Goal: Random target position in room
```

### Object Spawning Rules
```python
object_config = {
    "count": 10,
    "size_distribution": {
        "small": 0.5,    # 50% small objects
        "medium": 0.35,  # 35% medium
        "large": 0.15,   # 15% large
    },
    "placement": "random_non_overlapping",
    "min_distance_from_robot": 1.0,  # meters
    "min_distance_between_objects": 0.3,
}
```

### Behavior Goals
1. **Navigate to goal** without getting stuck
2. **Push small objects** out of the way when blocked
3. **Go around large objects** when pushing is inefficient
4. **Maintain stability** while pushing

### Reward Function
```python
reward = (
    # Goal progress
    + 2.0 * progress_to_goal        # Reward getting closer
    + 50.0 * goal_reached           # Big bonus for success
    
    # Clutter handling
    + 0.5 * object_cleared          # Pushed object out of path
    - 0.1 * unnecessary_contact     # Bumping when not needed
    
    # Efficiency
    - 0.01 * time_penalty           # Encourage speed
    - 0.05 * excessive_rotation     # Discourage spinning
    
    # Safety
    - 10.0 * fall_penalty           # Don't fall over
    - 5.0 * stuck_penalty           # No progress for N steps
)
```

### Deliverables
- [ ] `room_environment.py` - Single room with walls
- [ ] `object_spawner.py` - Random object placement
- [ ] `clutter_nav_env.py` - RL environment for Phase 1
- [ ] `push_behavior.py` - Object pushing logic
- [ ] Trained policy for 10-object navigation
- [ ] Success rate metrics (target: >80%)

---

## Phase 2: Single Room - 20 Objects

### Objectives
- Scale clutter density 2×
- Refine path planning through tighter spaces
- Learn efficient push vs avoid decisions

### Environment Changes
```python
object_config = {
    "count": 20,  # Doubled
    "size_distribution": {
        "small": 0.5,
        "medium": 0.35,
        "large": 0.15,
    },
    # Tighter packing allowed
    "min_distance_between_objects": 0.2,
}
```

### New Challenges
- **Tighter gaps**: Must choose whether to squeeze through or push
- **Object clusters**: Groups of objects blocking paths
- **Path planning**: Simple direct path often blocked

### Training Approach
1. **Initialize from Phase 1 policy** (transfer learning)
2. **Curriculum**: Start at 12 objects, increment to 20
3. **Focus reward on efficiency**: Path length, time

### Additional Reward Terms
```python
# Phase 2 additions
+ 0.3 * tight_gap_traversed       # Navigated narrow passage
- 0.2 * path_length_penalty       # Encourage shorter paths
+ 0.1 * smooth_trajectory         # Reward fluid motion
```

### Deliverables
- [ ] Updated environment config for 20 objects
- [ ] Curriculum schedule (12 → 15 → 18 → 20)
- [ ] Trained policy handling 20 objects
- [ ] Comparison metrics vs Phase 1
- [ ] Video demos of dense navigation

---

## Phase 3: Single Room - 40 Objects (Mixed)

### Objectives
- Introduce **immovable objects** (furniture-like)
- Learn to **classify** movable vs immovable
- Adapt strategy based on object type

### Object Categories
| Category | Count | Properties | Robot Strategy |
|----------|-------|------------|----------------|
| **Small Movable** | 20 | 0.1-0.3m, <2kg | Push freely |
| **Medium Movable** | 10 | 0.3-0.5m, 2-5kg | Push with effort |
| **Small Immovable** | 5 | 0.2-0.4m, fixed | Navigate around |
| **Large Immovable** | 5 | 0.5-1.0m, fixed | Navigate around |

### Classification Approach
```python
# Robot learns to classify through interaction
def classify_obstacle(contact_force, displacement):
    """
    After initial contact:
    - If object moves easily → MOVABLE
    - If object resists → IMMOVABLE
    - Cache classification for future encounters
    """
    if displacement > threshold:
        return "MOVABLE"
    else:
        return "IMMOVABLE"
```

### Sensor Requirements
- **Contact force sensing** on body/legs
- **Object tracking** (did it move after contact?)
- **Proximity sensing** for obstacle detection

### Reward Function Updates
```python
# Phase 3 specific rewards
+ 1.0 * correct_classification    # Identified movable vs immovable
- 2.0 * push_immovable            # Wasted effort on fixed object
+ 0.5 * efficient_avoidance       # Went around immovable smoothly
- 1.0 * repeated_failed_push      # Keep pushing something that won't move
```

### Deliverables
- [ ] `mixed_object_spawner.py` - Movable + immovable objects
- [ ] `obstacle_classifier.py` - Contact-based classification
- [ ] Trained policy handling 40 mixed objects
- [ ] Classification accuracy metrics (target: >90%)
- [ ] Decision-making visualization

---

## Phase 4: Two Rooms - Increasing Objects

### Objectives
- Expand to **multi-room** environment
- Navigate through **doorways**
- Handle **increasing clutter** across rooms

### Environment Layout
```
┌─────────────────┬─────────────────┐
│                 │                 │
│     ROOM A      D     ROOM B      │
│   (Start)       O    (Goal)       │
│                 O                 │
│    Objects      R    Objects      │
│                 │                 │
└─────────────────┴─────────────────┘

Room A: 5m × 5m
Room B: 5m × 5m  
Doorway: 1m wide opening
```

### Progressive Object Scaling
| Stage | Room A Objects | Room B Objects | Total |
|-------|---------------|----------------|-------|
| 4.1 | 10 | 10 | 20 |
| 4.2 | 15 | 15 | 30 |
| 4.3 | 20 | 20 | 40 |
| 4.4 | 25 | 25 | 50 |

### New Challenges
- **Doorway navigation**: Tight passage, often blocked by objects
- **Room-to-room planning**: High-level goal awareness
- **Clutter near doorway**: Objects accumulate at chokepoints

### Navigation Strategy
```python
class TwoRoomNavigator:
    def plan(self, current_pos, goal_pos):
        # Determine which room robot is in
        current_room = self.get_room(current_pos)
        goal_room = self.get_room(goal_pos)
        
        if current_room == goal_room:
            # Direct navigation within room
            return self.local_plan(current_pos, goal_pos)
        else:
            # Must go through doorway
            waypoints = [
                self.doorway_approach_point(current_room),
                self.doorway_center,
                self.doorway_exit_point(goal_room),
                goal_pos
            ]
            return waypoints
```

### Deliverables
- [ ] `two_room_env.py` - Multi-room environment
- [ ] `doorway_navigation.py` - Doorway traversal logic
- [ ] `room_planner.py` - High-level room-to-room planning
- [ ] Trained policies for 20 → 50 object scenarios
- [ ] Doorway success rate metrics

---

## Phase 5: Two Rooms + Stairs

### Objectives
- Add **staircase** between rooms (different floors)
- Learn stair **ascent and descent**
- Combine horizontal and vertical navigation

### Environment Layout
```
         FLOOR 2
┌─────────────────────────┐
│                         │
│        ROOM B           │
│       (Goal)            │
│                         │
└────────────┬────────────┘
             │ STAIRS
             │ (5 steps)
┌────────────┴────────────┐
│                         │
│        ROOM A           │
│       (Start)           │
│                         │
└─────────────────────────┘
         FLOOR 1
```

### Stair Specifications
```python
stairs_config = {
    "step_count": 5,
    "step_height": 0.15,      # 15cm per step
    "step_depth": 0.30,       # 30cm tread
    "step_width": 1.0,        # 1m wide
    "total_rise": 0.75,       # 75cm total
    "handrails": False,       # No rails in sim
}
```

### Stair Navigation Phases
1. **Approach**: Align with stair base
2. **Ascent**: Careful stepping up each stair
3. **Landing**: Stabilize at top
4. **Descent**: Controlled stepping down (harder!)

### Gait Modifications for Stairs
```python
# Stair-specific locomotion adjustments
stair_gait_params = {
    "step_height_increase": 0.08,  # Lift legs higher
    "velocity_reduction": 0.5,      # Slower movement
    "body_pitch_forward": 5.0,      # Lean into climb
    "stability_priority": True,     # Prioritize balance
}
```

### Reward Function for Stairs
```python
# Stair-specific rewards
+ 5.0 * step_completed            # Successfully climbed one step
+ 20.0 * stair_traversed          # Completed all stairs
- 15.0 * fall_on_stairs           # Higher penalty for stair falls
- 1.0 * step_missed               # Foot slipped off step
+ 0.5 * stable_ascent             # Maintained balance throughout
```

### Deliverables
- [ ] `stair_environment.py` - Room + stairs layout
- [ ] `stair_locomotion.py` - Modified gait for stairs
- [ ] `stair_detector.py` - Identify stair approach
- [ ] Trained policy for stair + clutter navigation
- [ ] Ascent/descent success rates (target: >90%)

---

## Phase 6: Beam Obstacles (Crawl + Jump)

### Objectives
- Introduce **horizontal beam obstacles**
- Learn **crawling** under low beams
- Learn **jumping** over low obstacles
- Seamless gait transitions

### Beam Types
| Type | Height | Width | Robot Action |
|------|--------|-------|--------------|
| **Low beam** | 0.3m | 0.1m | Jump over |
| **Mid beam** | 0.5m | 0.1m | Crawl under OR jump |
| **High beam** | 0.4m above ground | spans room | Crawl under |
| **Gap** | N/A | 0.5m | Jump across |

### Environment Configuration
```python
beam_config = {
    "low_beams": 2,      # Jump over these
    "crawl_beams": 2,    # Crawl under these (ceiling height 0.4m)
    "mixed_beams": 2,    # Robot chooses strategy
    "placement": "random_horizontal",
    "beam_length": "room_width",  # Spans the room
}
```

### Crawl Behavior
```python
class CrawlGait:
    """
    Low-clearance locomotion:
    - Lower body height by 50%
    - Splay legs outward
    - Slow, deliberate movements
    - Belly nearly touching ground
    """
    def __init__(self):
        self.body_height = 0.15  # Very low
        self.velocity_limit = 0.2  # Slow
        self.leg_splay = 1.3  # Wider stance
```

### Jump Behavior
```python
class JumpGait:
    """
    Obstacle clearing:
    1. Approach and stop
    2. Crouch (load springs)
    3. Launch (coordinated leg extension)
    4. Flight phase
    5. Land and stabilize
    """
    def execute_jump(self, obstacle_height, obstacle_distance):
        crouch_depth = self.calculate_crouch(obstacle_height)
        launch_force = self.calculate_force(obstacle_height, obstacle_distance)
        return self.jump_trajectory(crouch_depth, launch_force)
```

### Gait Selection Logic
```python
def select_gait(clearance_above, obstacle_ahead):
    """
    Decide locomotion mode based on environment
    """
    if clearance_above < ROBOT_HEIGHT:
        return "CRAWL"
    elif obstacle_ahead and obstacle_ahead.height < JUMP_THRESHOLD:
        return "JUMP"
    elif obstacle_ahead:
        return "NAVIGATE_AROUND"
    else:
        return "WALK"
```

### Training Strategy
1. **Train crawl separately** in confined tunnel environment
2. **Train jump separately** with simple hurdles
3. **Combine policies** with gait switching
4. **Fine-tune transitions** for smoothness

### Deliverables
- [ ] `beam_environment.py` - Rooms with beam obstacles
- [ ] `crawl_policy.py` - Low-clearance locomotion
- [ ] `jump_policy.py` - Obstacle jumping
- [ ] `gait_switcher.py` - Automatic gait selection
- [ ] Crawl success rate (target: >85%)
- [ ] Jump success rate (target: >80%)

---

## Phase 7: Full Integration & Evaluation

### Objectives
- Combine all capabilities into unified system
- Comprehensive benchmarking
- Stress testing and failure analysis

### Complete Test Environment
```
┌────────────────────────────────────────┐
│            FLOOR 2                     │
│  ┌─────────────┬─────────────┐         │
│  │   ROOM C    │   ROOM D    │         │
│  │  (20 obj)   │  (20 obj)   │         │
│  │   + beams   │   + beams   │         │
│  └──────┬──────┴──────┬──────┘         │
│         │    STAIRS   │                │
│  ┌──────┴──────┬──────┴──────┐         │
│  │   ROOM A    │   ROOM B    │         │
│  │  (20 obj)   │  (20 obj)   │         │
│  │  (Start)    │             │         │
│  └─────────────┴─────────────┘         │
│            FLOOR 1                     │
└────────────────────────────────────────┘
Total: 4 rooms, 80+ objects, stairs, beams
```

### Benchmark Scenarios

| Test | Description | Success Criteria |
|------|-------------|------------------|
| **Sparse Room** | 10 objects, single room | >95% success, <30s |
| **Dense Room** | 40 objects, single room | >85% success, <60s |
| **Two Room** | 50 objects, doorway | >80% success, <90s |
| **Stairs** | 2 floors, 40 objects | >75% success, <120s |
| **Beams** | Crawl + jump required | >70% success, <150s |
| **Full Course** | All challenges | >60% success, <180s |
| **Stress Test** | 100+ objects, all obstacles | Measure failure modes |

### Metrics
```python
evaluation_metrics = {
    # Primary
    "success_rate": float,          # Reached goal
    "completion_time": float,       # Seconds to goal
    "collision_count": int,         # Body collisions
    
    # Navigation quality
    "path_efficiency": float,       # Actual / optimal path
    "objects_pushed": int,          # Movable objects displaced
    "failed_pushes": int,           # Pushed immovable objects
    
    # Gait performance
    "crawl_attempts": int,
    "crawl_successes": int,
    "jump_attempts": int,
    "jump_successes": int,
    "stair_traversals": int,
    
    # Stability
    "falls": int,
    "recovery_attempts": int,
    "stuck_events": int,
}
```

### Baseline Comparisons
1. **SpotFlatTerrainPolicy** (no training) - Expected: fails in clutter
2. **Phase 1 policy** - Expected: works sparse, fails dense
3. **Phase 3 policy** - Expected: works single room
4. **Full pipeline** - Expected: handles all scenarios

### Deliverables
- [ ] `full_environment.py` - 4-room test environment
- [ ] `benchmark_runner.py` - Automated evaluation
- [ ] `metrics_dashboard.py` - Real-time metrics display
- [ ] Comparison tables and charts
- [ ] Failure case analysis document
- [ ] Demo videos for each scenario
- [ ] Final technical report

---

## Timeline

| Phase | Focus | Duration | Dependencies |
|-------|-------|----------|--------------|
| **Phase 1** | 10 objects, push behavior | 2 weeks | - |
| **Phase 2** | 20 objects, density scaling | 1.5 weeks | Phase 1 |
| **Phase 3** | 40 mixed objects, classification | 2 weeks | Phase 2 |
| **Phase 4** | Two rooms, doorways | 2 weeks | Phase 3 |
| **Phase 5** | Stairs | 2.5 weeks | Phase 4 |
| **Phase 6** | Crawl + Jump | 3 weeks | Phase 3 |
| **Phase 7** | Integration + Evaluation | 2 weeks | All |

**Total: ~15 weeks**

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| **Simulation** | NVIDIA Isaac Sim 4.2+ |
| **Robot** | Spot via `omni.isaac.quadruped` |
| **Locomotion** | SpotFlatTerrainPolicy (baseline) |
| **RL Framework** | Isaac Lab / RSL-RL |
| **Algorithm** | PPO with curriculum |
| **Physics** | 500Hz simulation, physics callbacks |
| **Metrics** | TensorBoard, custom logging |

---

## Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Push behavior destabilizes robot | High | Careful reward tuning, stability bonuses |
| Classification accuracy too low | Medium | More contact sensing, longer observation |
| Stairs cause frequent falls | High | Slow curriculum, extensive stair-only training |
| Jump timing difficult | High | Privileged learning, teacher-student |
| Dense clutter causes deadlocks | Medium | Recovery behaviors, backtrack capability |
| Sim-to-real gap | High | Domain randomization, noise injection |

---

## Files to Create

### Phase 1
```
MS_for_autonomy/
├── Phase-1/
│   ├── room_environment.py
│   ├── object_spawner.py
│   ├── clutter_nav_env.py
│   └── push_behavior.py
```

### Later Phases
```
MS_for_autonomy/
├── Phase-2/
├── Phase-3/
│   ├── mixed_object_spawner.py
│   └── obstacle_classifier.py
├── Phase-4/
│   ├── two_room_env.py
│   └── doorway_navigation.py
├── Phase-5/
│   ├── stair_environment.py
│   └── stair_locomotion.py
├── Phase-6/
│   ├── beam_environment.py
│   ├── crawl_policy.py
│   └── jump_policy.py
├── Phase-7/
│   ├── full_environment.py
│   └── benchmark_runner.py
└── common/
    ├── spot_physics_wrapper.py
    ├── reward_functions.py
    └── metrics_logger.py
```

---

## Next Steps

1. **Now**: Review this plan, adjust scope if needed
2. **Phase 1 Start**: Create room environment + object spawner
3. **First Milestone**: Spot navigates 10 random objects to goal

---

*Last Updated: January 21, 2026*
