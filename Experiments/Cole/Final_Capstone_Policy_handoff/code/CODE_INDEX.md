# Code Index — Cole max-density nav stack

These are **reference copies** of the exact files used for the Cole TRUE-max-density
run (7/25 WPs, 0 falls, 514m, 6m 45s) on `parkour_phase3_6800.pt`. Same pipeline
works on `parkour_phase3_7000.pt` (the ckpt in this handoff).

**Canonical paths** (where these live on the `development` branch — run from there):
- `Experiments/Alex/NAV_ALEX/scripts/cole_arena_skillnav_lite.py`
- `Experiments/Alex/NAV_ALEX/source/nav_locomotion/nav_locomotion/modules/{skill_nav_lite,online_obstacle_tracker,grid_astar_planner,depth_raycast_detector}.py`
- `Experiments/Alex/4_env_test/src/spot_rough_terrain_policy.py`

**Run from the canonical paths, not from this folder.** The scripts import via the
installed `nav_locomotion` package; standalone copies will not resolve imports.

---

## What each file does

### `cole_arena_skillnav_lite.py` (46 KB — entrypoint)
Top-level eval driver. Argparse, Isaac Sim boot, arena build, robot spawn, navigator
wiring, main step loop. Flags of interest:

- `--cole_arena` — use Cole's richer arena builder (moveable + non-moveable + small-static, 7 shapes)
- `--moveable_pct` / `--nonmoveable_pct` / `--small_static_pct` — arena densities
- `--apf_radius` / `--apf_gain` / `--apf_tangent` — reactive avoider tuning
- `--max_lin_speed` / `--waypoint_reach` — command speed + WP hit radius
- `--rough_heightscan` — enable 187-dim height-scan obs for the locomotion policy
- `--depth_sensor` — forward-facing PhysX raycast occupancy grid (8m / 90° / 64×16 rays / 0.4m grid). Implies `--online_map`.
- `--online_map` — start with zero obstacle knowledge, reveal within `--sense_radius`, replan A* every `--replan_period_sec`
- `--global_planner` — grid A* with ground-truth obstacles (cheat mode)
- `--loco_checkpoint` / `--loco_action_scale` / `--loco_decimation` — which locomotion policy to run and its params

### `nav_modules/skill_nav_lite.py` (9.5 KB — the APF navigator)
The reactive layer. Holds the waypoint list, current target, and the APF force computation.
Main entry: `compute_commands(robot_xy, yaw) -> [vx, vy, wz]`.

- Attractor: unit vector from robot to current sub-WP
- Repeller: for each obstacle within `obstacle_influence_radius`, add a force
  scaled by `strength = gain * (1/clearance − 1/R) / clearance²`
- Tangent bias: 80% of the force is perpendicular-to-radial (steers past the
  obstacle instead of just bouncing off)
- Stuck-escape logic: if robot hasn't moved >0.3m in 30 ticks, fire a 15-tick
  yaw-spin to escape local minima

### `nav_modules/grid_astar_planner.py` (7.7 KB — the global planner)
Grid A* on an occupancy grid. Inflates obstacles by `robot_radius + safety_margin`,
plans shortest path to goal, densifies the waypoint list so the APF's attractor
is always close (≤ `planner_subwp_step` meters).

### `nav_modules/online_obstacle_tracker.py` (2.2 KB — map builder for --online_map)
Maintains a known-obstacle list. Reveals obstacles within `sense_radius` each tick.
Triggers A* replan when a new obstacle enters the known set.

### `nav_modules/depth_raycast_detector.py` (6.2 KB — sensor for --depth_sensor)
Forward-facing ray fan (configurable FOV + ray count) cast each tick via PhysX.
Converts hits into occupancy grid cells after `min_hits` accumulation. Feeds the
online_obstacle_tracker instead of ground-truth positions.

### `spot_rough_terrain_policy.py` (32 KB — the locomotion wrapper)
Loads the `.pt` checkpoint, handles:
- Joint default pose (type-grouped `[hx×4, hy×4, kn×4]` — critical)
- PhysX raycast height scan (187 rays, clip [-0.2, 0.3])
- Gain application (Kp=60, Kd=1.5, solver 4/0)
- Forward pass: obs → action → joint position targets at 50Hz

The `_FALLBACK_DEFAULT_POS` constant in this file is what resets the robot to a
clean pose between episodes. If this is wrong, the robot spawns deformed.

---

## The Cole max-density + depth sensor + A* command

```bash
python Experiments/Alex/NAV_ALEX/scripts/cole_arena_skillnav_lite.py \
    --loco_checkpoint Experiments/Cole/Final_Capstone_Policy_handoff/parkour_phase3_7000.pt \
    --loco_action_scale 0.3 --loco_decimation 1 \
    --cole_arena --rough_heightscan --episodes 1 --seed 42 \
    --moveable_pct 1.0 --nonmoveable_pct 1.0 --small_static_pct 1.0 \
    --apf_radius 1.5 --apf_gain 0.9 --apf_tangent 0.8 \
    --max_lin_speed 2.4 --waypoint_reach 0.9 \
    --depth_sensor --rendered
```

Runs the whole stack: Isaac Sim → Cole arena at full clutter → Spot spawn →
policy load → navigator with APF + online map + 2s A* replans + depth sensor →
main loop until COMPLETE / FELL / FLIP / TIMEOUT / SCORE_DEPLETED.

---

## Pipeline at a glance

```
                   ┌────────────────────────────────┐
                   │  cole_arena_skillnav_lite.py   │  (main loop, Isaac Sim)
                   └───────────────┬────────────────┘
                                   │ each tick (50 Hz):
                                   ▼
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ depth_raycast_       │──▶│ online_obstacle_     │──▶│ grid_astar_planner   │
│   detector.py        │   │   tracker.py         │   │                      │
│ (PhysX raycast →     │   │ (grow known map,     │   │ (replan sub-WPs      │
│  occupancy cells)    │   │  trigger replan)     │   │  every 2 s)          │
└──────────────────────┘   └──────────────────────┘   └──────────┬───────────┘
                                                                  │
                                                                  ▼
                                                        ┌──────────────────────┐
                                                        │ skill_nav_lite.py    │
                                                        │ (APF + goal attract, │
                                                        │  stuck-escape)       │
                                                        └──────────┬───────────┘
                                                                   │
                                                                   ▼
                                                       ┌──────────────────────┐
                                                       │ spot_rough_terrain_  │
                                                       │   policy.py          │
                                                       │ (RL locomotion:      │
                                                       │  [vx,vy,wz] → joints)│
                                                       └──────────────────────┘
```

Everything except the box at the bottom is **classical / deterministic**. The
learning happened only at the locomotion layer. Your navigation intelligence
is algorithmic; the policy's job is to not fall while executing your commands.
