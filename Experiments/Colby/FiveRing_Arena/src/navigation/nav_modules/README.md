# nav_modules — A* + APF for FiveRing_Arena

Drop-in copy of the Skill-Nav Lite recipe used on Cole's rich arena (per Apr 19
benchmark: 19.7/25 mean over 3 quarter-density seeds, 0/3 falls on FM V3 + 7m
sensor + cadence-gated A* replan).

Source of truth: `Experiments/Cole/PARKOUR_NAV_handoff/code/nav_modules/` —
files here are byte-identical. Re-sync if upstream changes.

## Files

| File | Purpose |
|---|---|
| `skill_nav_lite.py` | APF + waypoint P-controller. Class `SkillNavLiteNavigator` consumes a 2D base-frame waypoint, emits `[vx, vy, wz]` for `SpotRoughTerrainPolicy`. |
| `grid_astar_planner.py` | Global A* plan through the known obstacle field. `plan_path()` returns a smoothed, line-of-sight-pruned waypoint chain. APF runs as a local deviator on top. |
| `depth_raycast_detector.py` | Raycast frontend → obstacle tuples `(cx, cy, size)`. |
| `online_obstacle_tracker.py` | Running obstacle map for cadence-gated replans. |

## Wiring sketch (FiveRing_Arena)

```python
from src.navigation.nav_modules.skill_nav_lite import SkillNavLiteNavigator
from src.navigation.nav_modules.grid_astar_planner import plan_path

# 1. Build obstacle list from the 5 ring centers + radii
obstacles = [(rx, ry, ring_diameter_m) for (rx, ry), ring_diameter_m in arena.rings]

# 2. Plan once at episode start (or when goal moves)
path = plan_path(
    start_xy=robot_xy, goal_xy=final_target_xy,
    obstacles=obstacles, bounds=(-25.0, 25.0),
    grid_res=0.5, inflate=0.4,
)

# 3. Per-tick: feed the next waypoint to the navigator, get [vx, vy, wz]
nav = SkillNavLiteNavigator(R=1.5, gain=0.9, tangent=0.8, speed=2.4, reach=0.9)
cmd_xyw = nav.step(robot_pose=(x, y, yaw), waypoint=path[next_idx], obstacles=obstacles)

# 4. Hand cmd_xyw to SpotRoughTerrainPolicy as the velocity command
```

## Tuning recipe (validated)

`R=1.5, gain=0.9, tangent=0.8, speed=2.4, reach=0.9` — keep verbatim unless
re-running the 11-iter sweep. Dropping any one of these regresses on Cole rich.

## Replacing `ring_follower.py`

`ring_follower.py` is a pure straight-line follower. To upgrade with obstacle
avoidance, replace its `advance()` call with steps 2–4 above. Keep the existing
`SpotRoughTerrainPolicy` interface — `cmd_xyw` plugs in unchanged.
