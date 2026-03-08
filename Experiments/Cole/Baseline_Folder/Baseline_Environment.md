# Baseline Environment (Testing Environment 3)

**Circular Waypoint Navigation Arena for Boston Dynamics Spot RL Training**

Author: Cole (MS for Autonomy Project)  
Date: February 2026  
File: `Baseline_Environment.py`

---

## Overview

The Baseline Environment is a comprehensive Isaac Sim training environment designed for Boston Dynamics Spot robot locomotion research. It features a circular arena with sequential waypoint navigation, realistic obstacle physics, and detailed performance tracking through CSV logging.

### Key Features

- **Circular Arena**: 50m diameter flat terrain with randomized obstacles
- **Sequential Waypoint Navigation**: 25 waypoints (A-Y) with optimized spacing
- **Weight-Based Obstacle Physics**: Three-tier system (light/medium/heavy) with realistic interaction
- **Small Static Hazards**: Golf ball to softball sized unmovable obstacles
- **Complete Sensor Suite**: 10 cameras + IMU + joint encoders + contact sensors
- **Comprehensive CSV Tracking**: Episode data with aggregate statistics
- **Natural Locomotion**: Forward-priority movement with heading alignment
- **Real-Time Status**: 1-second interval updates showing score, target, and distance

---

## Environment Specifications

### Arena Geometry

```
Shape:     Circle
Diameter:  50 meters
Radius:    25 meters
Center:    (0, 0, 0)
Terrain:   Flat ground plane
Area:      ≈ 1,963.5 m²
```

### Waypoint Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Total Count | 25 | Labeled A through Y |
| First Waypoint (A) | 20m | Exact distance from origin (0,0) |
| Subsequent (B-Y) | ≥40m | Minimum spacing between consecutive waypoints |
| Reach Threshold | 1.5m | Distance threshold for waypoint collection |
| Spawning Mode | Sequential | One visible marker at a time |
| Boundary Margin | 2.0m | Waypoints kept inside arena edge |

**Waypoint Generation Algorithm:**
- Waypoint A placed exactly 20m from start position
- Waypoints B-Y each placed ≥40m from previous waypoint
- All waypoints stay within arena boundaries (margin: 2m)
- Random angular distribution around arena center

### Obstacle System

#### Main Obstacles (20% Arena Coverage ≈ 392.7 m²)

**Weight-Based Physics Categories:**

| Category | Weight Range | Physics Behavior | Friction | Examples |
|----------|-------------|------------------|----------|----------|
| **Light** | < 0.45 kg (< 1 lb) | Dynamic, easily pushable/rollable | 0.4-0.5 | Small cubes, light cylinders |
| **Medium** | 0.45-32.7 kg (1-72 lb) | Dynamic, pushable with effort | 0.6-0.7 | Medium boxes, barrels |
| **Heavy** | > 32.7 kg (> 72 lb) | Static, immovable barrier | 0.9 | Large blocks, heavy obstacles |

**Obstacle Colors:**
- Light: Orange (1.0, 0.55, 0.0)
- Medium: Bronze/Tan (0.8, 0.6, 0.2)
- Heavy: Steel Blue (0.27, 0.51, 0.71)

**Shape Variety (7 types):**
1. Rectangle
2. Square
3. Trapezoid
4. Sphere
5. Diamond (Pyramid)
6. Oval (Ellipse)
7. Cylinder

**Footprint Range:**
- Minimum: 0.0174 m² (27 in²)
- Maximum: 0.55 m² (Spot's length × width)
- Height Range: 0.3-1.5 m

**Placement Rules:**
- 2m × 2m buffer zone around spawn point (0,0)
- 3m clearance from arena boundary
- 5m clearance from waypoints
- Random rotation (0-360°)

#### Small Static Obstacles (10% of Remaining Area ≈ 138.8 m²)

| Parameter | Value |
|-----------|-------|
| Size Range | 0.043-0.102 m (1.7-4 inches) |
| Coverage | 10% of remaining 80% = 8% total |
| Physics | Static (immovable, infinite mass) |
| Friction | 0.9 |
| Color | Dark Gray (0.4, 0.4, 0.4) |
| Clearance | 0.3m minimum spacing |

**Purpose:** Fixed hazards requiring precise navigation and avoidance, simulating real-world small debris.

---

## Robot Configuration

### Boston Dynamics Spot

| Parameter | Value | Description |
|-----------|-------|-------------|
| Mass | 32.7 kg | Actual Spot weight |
| Dimensions | 1.1m × 0.5m × 0.6m | Length × Width × Height |
| Start Position | (0, 0, 0.7) | Center of arena, 0.7m above ground |
| Max Speed | 2.235 m/s | 5 mph |
| Min Speed | 0.3 m/s | Crawl speed |
| Fall Threshold | 0.25m | Z-height below which robot has fallen |

### Sensor Suite (10 Cameras + Additional)

**Vision System:**
1. **Front Stereo Pair** (640×480, 30Hz)
   - Left: Position (0.48, 0.06, 0.20)
   - Right: Position (0.48, -0.06, 0.20)

2. **Left Stereo Pair** (640×480, 30Hz)
   - Front: Position (0.05, 0.22, 0.20)
   - Rear: Position (-0.30, 0.22, 0.20)

3. **Right Stereo Pair** (640×480, 30Hz)
   - Front: Position (0.05, -0.22, 0.20)
   - Rear: Position (-0.30, -0.22, 0.20)

4. **Rear Stereo Pair** (640×480, 30Hz)
   - Left: Position (-0.35, 0.06, 0.20)
   - Right: Position (-0.35, -0.06, 0.20)

5. **Overhead Stereo Pair** (640×480, 30Hz)
   - Front: Position (0.05, 0.04, 0.35)
   - Rear: Position (0.05, -0.04, 0.35)

**Additional Sensors:**
- **IMU**: 9-axis (accelerometer, gyroscope, magnetometer)
- **Joint Encoders**: 12 joints (3 per leg × 4 legs)
- **Contact Sensors**: 4 foot pressure sensors
- **Body Orientation**: Quaternion from USD pose

---

## Navigation Behavior

### Optimized Locomotion Strategy

**Forward-Priority Movement:**
- Natural forward walking (no backward movement or strafing)
- Rotation in place to face waypoint before advancing
- Small incremental heading corrections while moving
- Unnatural motions reserved only for collision avoidance

**Heading Alignment:**
- Threshold: 10 degrees (0.175 radians)
- Must align heading before moving forward
- Proportional gain: 0.5 for corrections
- Max turn rate: 1.0 rad/s

**Speed Control:**
- Base speed: Maximum (2.235 m/s)
- Obstacle proximity slowdown: <3m distance
  - Factor: max(0.3, distance/3.0)
- Waypoint approach slowdown: <2m distance
  - Factor: max(0.4, distance/2.0)
- Speed clamped: [0.3, 2.235] m/s

**Obstacle Interaction:**
- **Light obstacles**: Push forward naturally, physics handles collision
- **Medium obstacles**: Attempt to push; if blocked, navigate around
- **Heavy obstacles**: Static bodies block movement, require heading adjustment

---

## Scoring System

### Episode Rewards

| Event | Points | Calculation |
|-------|--------|-------------|
| Starting Score | +300.0 | Initial episode budget |
| Time Decay | -1.0/sec | Continuous penalty for elapsed time |
| Waypoint Reached | +15.0 | Bonus per waypoint collected |
| Episode Termination | 0.0 | Score resets to zero on fall |

**Score Formula:**
```
Current Score = 300 + (Waypoints × 15) - (Elapsed Time × 1)
```

### Termination Conditions

| Condition | Trigger | Failure Reason |
|-----------|---------|----------------|
| **Fall** | Z-position < 0.25m | "Fell Over" |
| **Score Depletion** | Score ≤ 0 | "Ran Out of Points" |
| **Completion** | All 25 waypoints reached | "Completed All Waypoints" |

---

## CSV Tracking System

### File Configuration

**Filename:** `Baseline_CSV.csv`  
**Location:** Same directory as `Baseline_Environment.py`

### CSV Structure

```csv
# AGGREGATE STATISTICS (Updated after each episode)
Metric,Value
Average_Waypoints_Reached,X.XX
Max_Waypoints_Reached,X
Failure_Rate_Fell_Over,XX.XX%
Failure_Rate_Ran_Out_of_Points,XX.XX%
Completion_Rate,XX.XX%

Episode,Waypoints_Reached,Failure_Reason,Final_Score
1,3,Fell Over,0.00
2,5,Ran Out of Points,-12.34
3,8,Ran Out of Points,-3.21
```

### Aggregate Statistics (Updated After Each Episode)

| Metric | Description | Format |
|--------|-------------|--------|
| Average_Waypoints_Reached | Mean waypoints across all episodes | Float (2 decimals) |
| Max_Waypoints_Reached | Highest waypoint count achieved | Integer |
| Failure_Rate_Fell_Over | Percentage of episodes ending in falls | Percentage (2 decimals) |
| Failure_Rate_Ran_Out_of_Points | Percentage of score depletion failures | Percentage (2 decimals) |
| Completion_Rate | Percentage of full completions (25/25) | Percentage (2 decimals) |

### Episode Data Columns

| Column | Type | Description |
|--------|------|-------------|
| Episode | Integer | Episode number (1, 2, 3, ...) |
| Waypoints_Reached | Integer | Number of waypoints collected (0-25) |
| Failure_Reason | String | Termination category (see above) |
| Final_Score | Float | Score at episode end (2 decimals) |

### Data Persistence

- **Never Overwrites**: Historical data preserved across runs
- **Loads on Startup**: Aggregate stats calculated from existing data
- **Appends New Episodes**: Each episode added sequentially
- **Complete Rewrite**: Entire CSV rewritten after each episode to update aggregate stats at top

---

## Terminal Status Output

**Update Frequency:** Every 1 second (real-time)

**Status Format:**
```
[STATUS] Score: 299.0 | Target: A | Distance: 19.95m
[STATUS] Score: 298.0 | Target: A | Distance: 19.96m
[STATUS] Score: 297.0 | Target: A | Distance: 19.95m
```

**Displayed Information:**
- Current score (real-time calculation)
- Target waypoint label (A-Y)
- Distance to target in meters (2 decimal precision)

---

## Usage

### Basic Commands

**Single Episode (GUI):**
```bash
C:\isaac-sim\python.bat Baseline_Environment.py --episodes 1
```

**Multiple Episodes (Headless):**
```bash
C:\isaac-sim\python.bat Baseline_Environment.py --episodes 10 --headless
```

**With Custom Seed:**
```bash
C:\isaac-sim\python.bat Baseline_Environment.py --episodes 5 --seed 42
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--headless` | Flag | False | Run without GUI (faster training) |
| `--episodes` | Integer | 1 | Number of episodes to run |
| `--seed` | Integer | None | Random seed (None = random) |

### Example Training Sessions

**Quick Test (1 Episode, Visual):**
```bash
cd Testing_Environments
C:\isaac-sim\python.bat Baseline_Environment.py --episodes 1
```

**Baseline Collection (10 Episodes, Headless):**
```bash
C:\isaac-sim\python.bat Baseline_Environment.py --episodes 10 --headless
```

**Reproducible Run (Fixed Seed):**
```bash
C:\isaac-sim\python.bat Baseline_Environment.py --episodes 20 --seed 12345 --headless
```

---

## Physics Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Physics Timestep | 1/500 sec (0.002s) | Simulation update rate |
| Rendering Timestep | 10/500 sec (0.02s) | Visual update rate |
| Gravity | -9.81 m/s² | Standard Earth gravity |
| Ground Friction | 0.7 | Arena floor friction coefficient |

---

## Key Classes and Components

### `CircularWaypointEnv`

Main environment class managing episode lifecycle, scoring, and CSV logging.

**Key Methods:**
- `reset(episode)`: Reset environment for new episode
- `step(step_size)`: Execute one physics step, check termination conditions
- `_log_to_csv(reason)`: Write episode data and update aggregate statistics
- `_calculate_aggregate_stats()`: Compute statistics across all episodes
- `_write_csv_with_stats()`: Rewrite entire CSV with updated aggregate stats
- `close()`: Cleanup resources

**State Variables:**
- `score`: Current episode score
- `waypoints_reached`: Count of collected waypoints (0-25)
- `episode_num`: Current episode number
- `episode_logged`: Flag preventing duplicate CSV entries
- `last_status_print_time`: Timer for 1-second status updates

### `ObstacleManager`

Handles obstacle generation, placement, and collision detection.

**Key Methods:**
- `spawn_one(idx, margin)`: Create single obstacle with weight-based physics
- `populate(target_coverage_pct)`: Fill arena to target coverage
- `spawn_small_static(target_coverage_pct)`: Place small unmovable hazards
- `nearest_obstacle_distance(x, y)`: Find closest obstacle for avoidance
- `remove_prims()`: Clean up obstacles for episode reset

**Shape Creators (7 Mesh Functions):**
- `create_rectangle_mesh()` - Rectangular box
- `create_square_mesh()` - Square pillar
- `create_trapezoid_mesh()` - Tapered prism
- `create_sphere_mesh()` - UV sphere
- `create_diamond_mesh()` - Pyramid
- `create_oval_mesh()` - Elliptical cylinder
- `create_cylinder_mesh()` - Circular cylinder

### Helper Functions

**Geometry:**
- `inside_arena(x, y, radius, margin)`: Check if point is inside arena
- `random_inside_arena(margin, rng)`: Sample uniform random position
- `distance_2d(a, b)`: Euclidean distance between 2D points

**Navigation:**
- `quaternion_to_yaw(quat)`: Convert quaternion to yaw angle
- `normalize_angle(angle)`: Wrap angle to [-π, π]
- `compute_speed_command(spot_x, spot_y, target_x, target_y, obstacle_mgr)`: Calculate speed based on obstacles and waypoint proximity

**Waypoints:**
- `generate_waypoints(rng)`: Create 25 waypoints with optimized spacing
- `spawn_waypoint_marker(stage, label, pos)`: Create flag-on-pole visual marker
- `remove_waypoint_markers(stage, marker_paths)`: Delete marker prims

**Physics:**
- `apply_rigid_body_physics(stage, prim_path, mass_kg, friction)`: Apply USD physics attributes
- `build_world(world, stage)`: Setup ground plane and lighting

**Sensors:**
- `setup_spot_sensors(spot_prim_path)`: Install complete 10-camera suite + IMU

---

## Performance Metrics

### Expected Baseline Performance (Untrained Policy)

| Metric | Expected Range | Notes |
|--------|----------------|-------|
| Avg Waypoints | 0-5 | Random walk with forward bias |
| Max Waypoints | 0-8 | Best case with minimal obstacles |
| Fall Rate | 30-50% | Common due to obstacle collisions |
| Ran Out Rate | 50-70% | Time decay faster than waypoint collection |
| Completion Rate | 0% | Unlikely without trained policy |

### Training Goals (Post-RL Training)

| Metric | Target | Description |
|--------|--------|-------------|
| Avg Waypoints | >15 | Consistent navigation |
| Max Waypoints | 25 | Full completion capability |
| Fall Rate | <10% | Stable locomotion |
| Completion Rate | >50% | Majority episodes complete all waypoints |

---

## File Structure

```
Testing_Environments/
├── Baseline_Environment.py      # Main environment code (1571 lines)
├── Baseline_Environment.md      # This documentation file
├── Baseline_CSV.csv             # Episode data and aggregate stats
└── Cole_md.md                   # Original specification document
```

---

## Dependencies

**Isaac Sim:**
- `isaacsim.SimulationApp` - Simulation application
- `omni.isaac.core.World` - World management
- `omni.isaac.quadruped.robots.SpotFlatTerrainPolicy` - Spot robot
- `omni.isaac.sensor.Camera` - Camera sensors
- `pxr` (USD) - Scene graph and physics

**Python Standard Library:**
- `csv` - CSV file I/O
- `math` - Mathematical functions
- `os` - File path operations
- `string` - ASCII alphabet for waypoint labels
- `argparse` - Command-line argument parsing

**Third-Party:**
- `numpy` - Array operations and random number generation

---

## Troubleshooting

### Common Issues

**Issue: GUI crashes with Unicode error**
- **Cause:** Windows console can't display special characters
- **Status:** Fixed (all Unicode replaced with ASCII)

**Issue: CSV not created**
- **Cause:** Episode terminated before logging
- **Status:** Fixed with failsafe logging in main loop

**Issue: Duplicate CSV entries**
- **Cause:** Termination conditions checked multiple times per frame
- **Status:** Fixed with `episode_logged` flag

**Issue: Obstacles too heavy to push**
- **Cause:** Weight category misclassification
- **Solution:** Check obstacle `weight_class` in spawn code (light < 0.45kg)

**Issue: Spot falling immediately**
- **Cause:** Physics initialization timing
- **Solution:** First physics step skipped via `physics_ready` flag

### Debug Commands

**Check CSV structure:**
```powershell
Get-Content Baseline_CSV.csv | Select-Object -First 15
```

**Monitor episode progress:**
```powershell
Get-Content Baseline_CSV.csv | Select-String "Episode"
```

**Count total episodes:**
```powershell
(Get-Content Baseline_CSV.csv | Select-String "^\d+,").Count
```

---

## Future Enhancements

### Planned Features

1. **Terrain Variation**: Add slopes, steps, and uneven ground
2. **Dynamic Obstacles**: Moving obstacles requiring reactive navigation
3. **Multi-Robot Training**: Multiple Spot robots in same arena
4. **Reward Shaping**: Additional rewards for efficient paths
5. **Camera Data Logging**: Save visual observations for analysis
6. **Obstacle Density Profiles**: Variable difficulty levels

### Potential Modifications

- **Waypoint Distribution**: Clustered vs. spread patterns
- **Obstacle Types**: Deformable objects, liquids, gaps
- **Scoring System**: Efficiency bonuses, collision penalties
- **Arena Shape**: Rectangular, L-shaped, multi-room layouts
- **Time Limits**: Fixed duration episodes
- **Checkpoints**: Partial progress saving

---

## References

- **Isaac Sim Documentation**: https://docs.omniverse.nvidia.com/isaacsim/
- **Boston Dynamics Spot Specs**: https://bostondynamics.com/products/spot/
- **USD Physics API**: https://openusd.org/release/api/usd_physics_page_front.html

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Feb 2026 | Initial baseline environment |
| 1.1 | Feb 2026 | Added small static obstacles |
| 1.2 | Feb 2026 | CSV tracking system with aggregate stats |
| 1.3 | Feb 2026 | Real-time status output (1-second intervals) |
| 1.4 | Feb 2026 | Max waypoints metric in aggregate stats |

---

## Contact

**Author:** Cole  
**Project:** MS for Autonomy - Boston Dynamics Spot Locomotion Training  
**Date:** February 2026

For questions or issues, refer to `Cole_md.md` specification document.
