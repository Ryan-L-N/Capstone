# Skill-Nav Lite on Cole's Arena — How to Run

Waypoint-driven navigation for Spot in Cole's 25m obstacle arena. No new RL training required — layers a Skill-Nav-style P-controller + APF obstacle avoidance on top of the trained Flat Master V3 locomotion policy.

**Validated result (Apr 18 2026):** **25/25 waypoints, 1085m, 646s, zero falls, zero OOB** through 40 random obstacles (Flat Master V3 + APF at max_spd=2.2 m/s, R=1.8, gain=1.1). Earlier baseline at 1.5 m/s: 15/25.

---

## 1. Prerequisites

- Windows + miniconda with `isaaclab311` env (Isaac Sim 5.1.0, Python 3.11).
- Flat Master V3 checkpoint at `Capstone/Experiments/Alex/SIM_TO_REAL/checkpoints/flat_v3_3700.pt` (235 → [512,256,128] → 12 actor).
- NAV_ALEX source tree intact:
  - `source/nav_locomotion/nav_locomotion/modules/skill_nav_lite.py`
  - `scripts/cole_arena_skillnav_lite.py`
- `4_env_test/src/spot_rough_terrain_policy.py` (imported as the loco wrapper).

## 2. Run commands

### Full obstacle arena (default — 40 obstacles)
```bash
conda activate isaaclab311
cd "Capstone/Experiments/Alex/NAV_ALEX"
python scripts/cole_arena_skillnav_lite.py --episodes 1 --rendered
```

### Empty arena (smoke test — isolates nav/loco from collisions)
```bash
python scripts/cole_arena_skillnav_lite.py --episodes 1 --rendered --no_obstacles
```

### Headless (faster, for batch runs)
```bash
python scripts/cole_arena_skillnav_lite.py --episodes 3 --headless
```

Results CSV lands in `NAV_ALEX/results/cole_arena_skillnav_lite.csv`.

## 3. Key flags

| Flag | Default | What it does |
|---|---|---|
| `--episodes N` | 3 | Episodes per run |
| `--headless` / `--rendered` | headless | Viewport on/off |
| `--no_obstacles` | off | Empty arena (bypass 40-cube generation) |
| `--num_obstacles K` | 40 | Obstacle count (ignored if `--no_obstacles`) |
| `--loco_checkpoint PATH` | `flat_v3_3700.pt` | RSL-RL checkpoint for loco policy |
| `--stock_flat` | off | Use stock Isaac `SpotFlatTerrainPolicy` instead (debug — will fall) |
| `--kp_lin`, `--kp_ang`, `--max_lin_speed` | 1.0 / 2.0 / **2.2** | P-controller gains (tuned) |
| `--apf_radius`, `--apf_gain`, `--apf_tangent` | **1.8 / 1.1 / 0.6** | APF obstacle avoidance (tuned for 2.2 m/s) |

## 4. What to expect

On first run, you should see:
```
[OK] Flat Master V3 engaged (flat_v3_3700.pt, mason obs, flat heightscan)
[NAV] APF obstacles: 40  R=1.2m  gain=0.8  tangent=0.6
  WP-A at (x.x, y.y)
  Stabilizing...
    [STAB 0] z=0.592
    [STAB 200] z=0.540
  Stabilized. z=0.584m
    [nav   1 t=  0.1s] pos=... -> A d=...m cmd=[...]
  [WP A] reached (1/25)
  [WP B] reached (2/25)
  ...
  Result: SCORE_DEPLETED | WP: 15/25 | Dist: 674m | Time: 525s
```

## 5. Critical gotchas (don't change these)

These were painful to find. Changing any breaks the stack:

1. **`robot_policy._decimation = 10`** in the script. Mason was trained at 50Hz control (500Hz physics / 10). Setting `_decimation = 1` forces 500Hz policy eval and the robot crashes to the ground during stabilization.
2. **250-step stabilization.** Shorter and the robot hasn't settled when the first nav command arrives — it falls mid-turn. Seen at 50 steps.
3. **`ground_height_fn=lambda x: 0.0`** in `SpotRoughTerrainPolicy(...)`. The PhysX raycast otherwise detects the cubes as terrain and corrupts Flat Master's height-scan observation (trained on flat ground only).
4. **Always drive `vx = 0.3` minimum** during yaw-first turns (in `skill_nav_lite.py`). A pure turn-in-place command (vx=vy=0, wz=±2) takes the walking policy out of its gait distribution and it face-plants.
5. **APF params `R=1.2m, gain=0.8, tangent_bias=0.6`.** Earlier `R=2.5, gain=3.0` saturated the arena and the robot orbited forever in local minima. Tangent bias lets the robot slide around obstacles instead of bouncing head-on.
6. **`mason_baseline=True`** on the `SpotRoughTerrainPolicy` — the Flat Master V3 checkpoint uses Mason obs order (height_scan first, then proprio).

## 6. Speed ↔ collision tradeoff

Speed scaling works as long as APF margin scales with it. Measured on same seed, same 40 obstacles, Flat Master V3:

| `max_lin_speed` | APF `R` / `gain` | WPs | Dist | Falls | Status |
|---|---|---|---|---|---|
| 1.5 m/s | 1.2 / 0.8 | 15/25 | 674m | 0 | SCORE_DEPLETED |
| 2.2 m/s | 1.8 / 1.1 | **25/25** | **1085m** | **0** | **COMPLETE** |

Rule of thumb when raising speed: grow `R` roughly linearly with `max_lin_speed` so the robot sees obstacles the same number of decision-ticks in advance.

## 7. File tree

```
NAV_ALEX/
├── scripts/
│   └── cole_arena_skillnav_lite.py          # Entry point
├── source/nav_locomotion/nav_locomotion/modules/
│   └── skill_nav_lite.py                    # SkillNavLiteNavigator (APF P-controller)
└── HOW_TO_SKILL_NAV_LITE.md                 # This file

Capstone/Experiments/Alex/
├── SIM_TO_REAL/checkpoints/
│   └── flat_v3_3700.pt                      # Trained loco policy (required)
└── 4_env_test/src/
    └── spot_rough_terrain_policy.py         # Loco wrapper (Mason obs, analytical height scan)
```

## 8. Debugging

If the robot falls during the first seconds:
- Check stab z progression in the log. Should climb from ~0.37 up to ~0.52-0.58.
- If it keeps dropping: `_decimation = 1` (wrong — should be 10) or Flat Master checkpoint missing/corrupt.

If robot gets stuck orbiting (SCORE_DEPLETED, 0 WPs, no falls):
- APF is too strong. Drop `obstacle_repulse_gain` toward 0.5, or `obstacle_influence_radius` to 1.0.

If robot collides with an obstacle:
- APF too weak, or you're using `--stock_flat` (stock policy doesn't walk well). Raise gain, or back to Flat Master V3.
