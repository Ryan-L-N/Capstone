# 5-Ring Navigation Gauntlet

Concentric ring evaluation course for quadruped locomotion policies. A 50m-radius circular arena with 5 rings of increasing difficulty — tests both locomotion robustness and navigation in a single progressive run.

## Arena Layout

```
         _______________________________________________
        /   Ring 5 (40-50m) — Extreme Mixed (ice+boulders) \
       /    Ring 4 (30-40m) — Boulder Field                 \
      /     Ring 3 (20-30m) — Vegetation + Drag              \
     /      Ring 2 (10-20m) — Low Friction                    \
    |       Ring 1 (0-10m)  — Flat (navigation warmup)         |
    |            [SPAWN at center (0, 0)]                       |
     \                                                        /
       \____________________________________________________/

10 waypoints per ring (36° spacing), visited clockwise
50 ring waypoints + 4 transition waypoints = 54 total
```

## Ring Specifications

| Ring | Radius | Terrain | Friction | Obstacles | Weight |
|------|--------|---------|----------|-----------|--------|
| 1 | 0-10m | Flat | μ=0.80 | None | 10 |
| 2 | 10-20m | Low Friction | μ_s=0.35, μ_d=0.25 | None | 20 |
| 3 | 20-30m | Vegetation | μ=0.70 | Grass stalks 5/m² | 30 |
| 4 | 30-40m | Boulder Field | μ=0.75 | Boulders 15-50cm | 40 |
| 5 | 40-50m | Extreme Mixed | μ_s=0.25, μ_d=0.15 | Boulders 40-80cm | 50 |

**Composite score:** max 150 points

## Usage

```bash
# Headless (H100)
python src/run_ring_eval.py --headless --policy rough \
    --checkpoint checkpoints/model.pt --num_episodes 100

# Rendered (local)
python src/run_ring_eval.py --rendered --policy rough \
    --checkpoint checkpoints/model.pt --num_episodes 5

# Shell launcher
./scripts/run_eval.sh --num_episodes 100
```

## Output

JSONL files in `results/` with per-ring scoring:
```json
{
  "episode_id": "ring_rough_ep0001",
  "total_waypoints_reached": 37,
  "rings_completed": 3,
  "composite_score": 74.0,
  "max_score": 150,
  "ring_scores": {"ring_1": {...}, "ring_2": {...}, ...}
}
```

## Project Structure

```
5_ring_test/
├── src/
│   ├── configs/
│   │   ├── eval_cfg.py        # Physics constants
│   │   └── ring_params.py     # Ring specs, waypoints, scoring
│   ├── envs/
│   │   ├── base_arena.py      # quat_to_yaw, disable_ground
│   │   ├── ring_arena.py      # Circular ring ground + obstacles
│   │   ├── boulder_meshes.py  # Polyhedron mesh generators
│   │   └── vegetation.py      # Stalk creation + drag logic
│   ├── navigation/
│   │   └── ring_follower.py   # Circular waypoint follower
│   ├── metrics/
│   │   └── ring_collector.py  # Per-ring scoring + JSONL
│   ├── spot_rough_terrain_policy.py  # Policy wrapper
│   └── run_ring_eval.py       # Main entry point
├── checkpoints/               # Policy .pt files
├── results/                   # JSONL output
└── scripts/
    └── run_eval.sh            # Shell launcher
```
