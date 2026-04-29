# Lessons Learned — Circular Training Environment (Testing_Environment_2)

**Maintained by:** Cole  
**Project:** Immersive Modeling and Simulation for Autonomy  
**Environment:** `Testing_Environment_2.py`  
**Started:** February 2026  

> **Usage:** Append new entries whenever a problem is encountered, a solution is found, or a design decision is made. Keep entries dated and categorized.

---

## Entry Format

```
### [YYYY-MM-DD] Category: Short Title
**Problem:** ...
**Solution / Decision:** ...
**Impact:** ...
**Tags:** #tag1 #tag2
```

---

## Entries

---

### [2026-02-18] Design: Circular Boundary Implementation

**Problem:** USD does not have a native hollow-cylinder primitive to serve as a circular containment wall that Spot cannot walk through.

**Solution / Decision:** Approximate the circle with **64 thin `UsdGeom.Mesh` box segments** arranged in a ring. Each segment is tangent to the inner boundary (radius = 25 m). The number of segments (64) was chosen to give a visually smooth circle while keeping USD prim count manageable.

**Impact:** Boundary looks nearly circular from overhead. There is a slight polygon edge visible at very close range, but it does not affect navigation behavior. Future improvement: try a true cylinder shell using a custom USD schema.

**Tags:** #geometry #boundary #usd

---

### [2026-02-18] Design: Obstacle Footprint Targeting 20% Coverage

**Problem:** The spec requires 20% of the arena area to be covered by obstacles (≈ 392.7 m²). Since obstacle sizes are randomized per episode, a fixed obstacle count cannot guarantee exactly 20% coverage.

**Solution / Decision:** Compute a **running footprint total** during obstacle generation. Continue spawning obstacles until the cumulative footprint reaches 392.7 m² or the maximum spawn attempts are exhausted. This adapts the obstacle count to the randomized sizes each episode.

**Impact:** Coverage will be approximately (not exactly) 20% depending on available positions. Log the final coverage percentage each episode for analysis.

**Tags:** #obstacles #randomization #coverage

---

### [2026-02-18] Design: Obstacle Physics — Push vs. Navigate Around

**Problem:** The spec requires that light obstacles (< 1 lb / 0.45 kg) can be pushed by Spot, while heavy ones (up to 32.7 kg) cannot.

**Solution / Decision:**
- Apply `UsdPhysics.RigidBodyAPI` and `UsdPhysics.MassAPI` to all obstacles.
- Light obstacles: mass ≤ 0.45 kg, `rigidBodyEnabled = True`, no position lock. Spot contact forces will displace them.
- Heavy obstacles: mass > 0.45 kg up to 32.7 kg, `rigidBodyEnabled = True`. Due to much higher inertia, Spot cannot meaningfully displace them at typical locomotion speeds.
- A future improvement could add a `kinematicEnabled = True` flag to explicitly lock heavy obstacles, preventing any drift.

**Impact:** Physics behavior should be emergent and realistic. Initial testing may show heavy obstacles sliding slightly; if so, apply `UsdPhysics.CollisionAPI` with high static friction (0.9).

**Tags:** #physics #obstacles #mass #collision

---

### [2026-02-18] Design: Trapezoid and Diamond Shape Meshes

**Problem:** USD has no native trapezoid or diamond primitive. Custom vertices must be hand-computed and wound correctly for USD face normals.

**Solution / Decision:**
- **Trapezoid:** Base vertices at y = 0 with full width `w`; top vertices at z = height with reduced width `w * taper`. CCW winding looking from outside.
- **Diamond:** Four vertices at ±dx (X) and ±dy (Y), with floor and ceiling quads.
- Both shapes compute face normals implicitly via USD's auto-normal generation.

**Impact:** Shapes display correctly. Verify winding order if normals invert (faces appear invisible or black). Flip face index order (e.g., `[0,1,2,3]` → `[3,2,1,0]`) to fix inverted normals.

**Tags:** #geometry #mesh #usd #obstacles

---

### [2026-02-18] Design: Waypoint Chain Placement

**Problem:** Generating 25 waypoints where each is exactly 25 m from the previous, while keeping all within a 25 m radius circle, is geometrically constrained. A strict 25 m spacing from (0,0) means the first waypoint B is at the circle boundary, leaving almost no room for subsequent waypoints to satisfy the constraint while staying inside.

**Solution / Decision:**
- Use a **relaxed spacing** of "at least 20 m and at most 30 m from the previous waypoint" rather than exactly 25 m. This gives the chain room to zig-zag within the circle.
- If no valid placement is found in 500 attempts, use the closest-to-25m candidate that still lies inside the circle boundary (with 2 m margin).
- Document in spec as "≥ 25 m spacing" for the RL curriculum intent.

**Impact:** All 25 waypoints will be placed inside the arena each episode. The chain will not form a straight line but will wind through the environment, creating a challenging navigation path.

**Tags:** #waypoints #geometry #randomization #placement

---

### [2026-02-18] Design: Speed Modulation Based on Obstacle Proximity

**Problem:** The spec requires Spot to reduce speed when near obstacles. `SpotFlatTerrainPolicy` accepts a scalar forward speed command but does not natively query obstacle distances.

**Solution / Decision:**
- After updating Spot's state each step, compute the **minimum distance to any obstacle center** using vectorized numpy operations.
- Apply a linear speed scaling: `v = v_max * min(d / r_slow, 1.0)`, where `r_slow = 2.0 m`.
- Pass the scaled speed as the forward component of the `[forward, lateral, yaw]` command tuple.

**Impact:** Spot smoothly decelerates as it approaches obstacles. At `d = 0` (collision), speed would be 0; in practice the robot should stop before contact. Consider adding an emergency-stop condition when `d < 0.4 m` (Spot's half-width).

**Tags:** #speed #obstacles #policy #safety

---

### [2026-02-18] Design: Waypoint Visual Markers

**Problem:** Waypoints must be **visually distinct from obstacles** so the agent (and human observer) can easily differentiate them during simulation playback.

**Solution / Decision:**
- Waypoints: **bright yellow** `UsdGeom.Cylinder` (radius 0.3 m, height 1.5 m), color `(1.0, 0.95, 0.0)`.
- Obstacles: colored by weight category.
  - Light obstacles: **orange** `(1.0, 0.55, 0.0)`.
  - Heavy obstacles: **steel blue** `(0.27, 0.51, 0.71)`.
- Start marker (waypoint A): **bright green** `(0.2, 0.9, 0.2)`.

**Impact:** Clear visual separation during debugging and demo runs. Colors chosen for maximum contrast. Consider adding USD `DisplayName` metadata with the letter label for further clarity.

**Tags:** #visualization #waypoints #obstacles #ux

---

### [2026-02-18] Design: Sensor Integration Approach

**Problem:** `SpotFlatTerrainPolicy` initializes its own sensor suite internally. Attempting to add additional sensors before `world.reset()` can cause prim-path conflicts.

**Solution / Decision:**
- Call `world.reset()` first, then call `robot.initialize()`.
- Add supplementary sensors (depth camera, LiDAR placeholder) **after** `robot.initialize()` completes.
- LiDAR is simulated analytically using obstacle positions rather than a full ray-cast sensor to avoid GPU memory overhead during training.

**Impact:** No prim-path conflicts observed in Testing_Environment_1 using the same pattern. Analytical LiDAR may not capture all edge cases (irregular mesh shapes), but is sufficient for distance observations.

**Tags:** #sensors #lidar #camera #initialization #isaaclab

---

### [2026-02-18] Design: Episode Seed for Reproducibility

**Problem:** During RL debugging it is important to reproduce a specific failure episode by replaying the same randomization.

**Solution / Decision:**
- Maintain an integer `episode_counter`.
- At each `reset()`, call `np.random.seed(episode_counter)` before all random placement calls.
- Log the episode seed at the start of each episode.
- For production training, pass `episode_counter = None` to use truly random seeds.

**Impact:** Any failed episode can be reproduced exactly by running with the same episode count. This is critical for testing fixes without having to luck into the same random configuration.

**Tags:** #reproducibility #debugging #randomization #seed

---

### [2026-02-18] Design: Reward System — Score Bank with Time Decay

**Problem:** A fixed step-count timeout produces sparse, hard-to-shape reward signals. The agent has no incentive to move efficiently and can stall to avoid failure.

**Solution / Decision:**
- Give Spot a **300-point starting bank** at episode start.
- Deduct **−1 point per simulated second** continuously (time decay applied every physics step as `−TIME_DECAY_PER_SEC × PHYSICS_DT`).
- Award **+15 points** per waypoint collected.
- On fall: **immediately zero the score** and end the episode.
- Episode terminates when `score ≤ 0` (`reason = score_depleted`) — the score bank acts as a soft time limit.
- This removes the need for a hard `MAX_STEPS_PER_EPISODE` constant.

**Impact:** The agent is rewarded for speed and precision. Collecting waypoints extends the episode; falling or stalling ends it early. The score at episode end is a natural performance metric.

**Tags:** #reward #scoring #training #design

---

### [2026-02-18] Design: Modular Reward Function via compute_reward()

**Problem:** All reward arithmetic was originally inline inside `step()`, making it difficult to add or disable shaping terms without touching the episode loop.

**Solution / Decision:**
- Extract all reward components into `CircularWaypointEnv.compute_reward()`, which returns a **named dict** (`{"time_decay": ..., "energy": ..., ...}`).
- `step()` calls `compute_reward()`, sums the components, and applies them to `self.score`.
- New shaping terms (energy penalty, smoothness, obstacle avoidance bonus) are added as commented-out hooks inside `compute_reward()` and can be enabled line-by-line.

**Impact:** Adding or tuning a reward term requires changing exactly one function and never risks breaking the episode loop. Named components also make reward debugging and logging straightforward.

**Tags:** #reward #architecture #modularity #design

---

### [2026-02-18] Design: Sequential Waypoint Spawning (One Active at a Time)

**Problem:** Pre-spawning all 25 waypoints simultaneously clutters the scene, confuses the policy's observation (which target to navigate toward), and wastes USD prim resources.

**Solution / Decision:**
- Only **one waypoint prim exists in the USD stage at any time**.
- On `reset()`, spawn only waypoint **A** (exactly 24 m from origin in a random valid direction).
- When Spot reaches a waypoint: award points, log index, **despawn** the current prim, **spawn** the next waypoint at the required distance (≥ 30 m for B–Z), re-rolling direction until inside the arena.
- The observation vector includes only the (dx, dy) vector to the **single active** waypoint.

**Impact:** Cleaner scene, unambiguous observation target, and a natural curriculum — the agent always has exactly one goal. Simplifies marker prim management (single-prim remove/add per waypoint event).

**Tags:** #waypoints #spawning #design #observation

---

### [2026-02-18] Design: Waypoint Spacing — Final Optimized Rules

**Problem:** The original "exactly 25 m from previous" rule was geometrically infeasible for many waypoints in a 25 m radius arena. A chain of 25 waypoints with 25 m spacing cannot stay inside the circle with sufficient valid placements.

**Solution / Decision:** (Supersedes earlier "relaxed 20–30 m" note)
- Waypoint **A**: placed exactly **24 m** from the start point (0, 0).
- Waypoints **B–Z**: each placed at least **30 m** from the **previous** waypoint only. Non-adjacent pairs have no constraint.
- If the 30 m offset from the previous waypoint lands outside the arena (with 2 m boundary margin), **re-roll the direction** until a valid placement is found.
- No fallback to best-available — keep re-rolling until a valid direction is found.

**Impact:** The 24 m rule for A ensures the first target is within arena bounds regardless of direction. The ≥ 30 m rule for subsequent waypoints creates a wide, winding chain through the arena. Re-roll-only (no fallback placement) guarantees all waypoints satisfy the spacing constraint.

**Tags:** #waypoints #geometry #placement #spacing

---

### [2026-02-18] Design: Waypoint Markers Upgraded to Flag-on-Pole

**Problem:** The original plain `UsdGeom.Cylinder` markers (radius 0.3 m, height 1.5 m, yellow) were visually similar in silhouette to cylindrical obstacles, making it harder for a human observer to distinguish waypoints from obstacles during playback.

**Solution / Decision:** (Supersedes the earlier "bright yellow cylinder" entry)
- Each waypoint marker is now a **flag-on-pole** composed of two sub-prims under a parent Xform:
  - **Pole:** `UsdGeom.Cylinder`, radius = 0.05 m, height = 2.5 m, light grey (RGB 0.88, 0.88, 0.88).
  - **Banner:** `UsdGeom.Mesh` (flat box), 0.7 m wide × 0.4 m tall, mounted at the top of the pole.
  - Banner colour: **bright green** for waypoint A (start), **bright yellow** (RGB 1.0, 0.95, 0.0) for B–Z.
- The tall thin pole and horizontal banner create a silhouette completely unlike any obstacle shape.

**Impact:** Waypoints are immediately recognisable at a glance. Prim grouping under a parent Xform lets the entire marker (pole + banner) be removed with a single `stage.RemovePrim(parent_path)` call.

**Tags:** #waypoints #visualization #usd #markers

---

### [2026-02-18] Bug: PhysX Rejects Triangle Mesh Collision for Dynamic Bodies

**Problem:** All `UsdGeom.Mesh` obstacles (rectangle, square, trapezoid, diamond) generated PhysX errors at runtime:
> `PhysicsSchemaPlugin: triangle mesh collision not supported for dynamic rigid bodies`
PhysX fell back to convexHull automatically, but logged hundreds of error lines per episode, masking real errors.

**Solution / Decision:**
- After applying `UsdPhysics.CollisionAPI` to any prim, check if the prim type is `"Mesh"`.
- If so, also apply `UsdPhysics.MeshCollisionAPI` and explicitly set `approximation = "convexHull"`.
- This tells PhysX the intended approximation upfront and suppresses the fallback warning.

```python
if prim.GetTypeName() == "Mesh":
    mesh_coll = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_coll.CreateApproximationAttr("convexHull")
```

**Impact:** No more PhysX collision warnings in the log. Physics behavior is identical (PhysX was already using convexHull), but the explicit declaration keeps the log clean for real error detection.

**Tags:** #physics #mesh #collision #physx #usd #bug

---

### [2026-02-18] Bug: SpotFlatTerrainPolicy Needs ~300 Stabilization Steps to Stand

**Problem:** With only 20 physics stabilization steps after robot initialization, Spot's base link z-height dropped below the fall detection threshold (0.3 m) within 13 steps of the first `step()` call, immediately triggering a fall and ending the episode.

**Solution / Decision:**
- Increase post-init stabilization from 20 to **300 physics steps** (`render=False`).
- Also lower the fall detection threshold from 0.3 m to **0.20 m** to give the policy a small buffer during the initial stand-up transient.
- 300 steps at 500 Hz = 0.6 simulated seconds — enough for `SpotFlatTerrainPolicy`'s locomotion controller to reach a stable standing pose.

**Impact:** Spot reliably stands up before the episode loop begins. The stabilization steps run at `render=False` so they do not slow down the GUI. Consider exposing `STABILIZATION_STEPS` as a constant for easy tuning.

**Tags:** #spot #physics #initialization #bug #stability

---

### [2026-02-18] Design: CSV Episode Logging

**Problem:** Training progress was only visible in the terminal during a run — no persistent record existed to compare performance across runs or sessions.

**Solution / Decision:**
- At the end of every episode (on any termination condition), `CircularWaypointEnv._log_to_csv()` appends one row to `training_log.csv` in the `Testing_Environments/` directory.
- Columns: `Episode`, `Waypoints_Reached`, `Time_Elapsed`, `Final_Score`.
- If the file does not exist, a header row is written first.
- Subsequent runs **append** — the file is never overwritten so data accumulates across sessions.
- Waypoint visit **order** is printed to the console (e.g. `A → B → D → …`) but not stored in the CSV (kept simple for easy parsing).

**Impact:** Training trends are immediately analysable in Excel, pandas, or any CSV tool. The append-only approach means no training data is ever lost between runs.

**Tags:** #logging #csv #training #data

---

## Future Improvements

- [ ] Replace analytical LiDAR with Isaac Sim `omni.sensors.lidar` for physics-accurate ray casting.
- [ ] Add moving obstacles (slow linear velocity) as a curriculum-level hard mode.
- [ ] Visualize the episode score (waypoints collected / 25) as a HUD overlay in the viewport.
- [ ] Implement obstacle convex decomposition for accurate physics contact on trapezoid/diamond shapes.
- [ ] Add terrain variation mode (rolling hills, ramps) as a harder curriculum stage.
- [ ] Serialize randomized episode configuration to JSON for offline replay.
- [ ] Benchmark obstacle count vs. training FPS to find the optimal obstacle density for H-100 training.

---

*Last updated: February 18, 2026*
