# 4-Quadrant Navigation Gauntlet

Circular evaluation arena with 4 terrain quadrants of increasing difficulty. Tests locomotion robustness across friction, vegetation, boulders, and stairs in a single progressive run.

## Arena Layout

```
              Grass (90-180°)
            stalks + drag
                  |
                  |
  Boulders ------[SPAWN]------- Friction
  (180-270°)      |              (0-90°)
  D8-D20 rocks    |            decreasing mu
                  |
             Stairs (270-360°)
            pyramid waypoints
```

**50m radius, 4 quadrants, 5 difficulty levels per quadrant (0-10m to 40-50m)**

## Quadrant Specifications

| Quadrant | Lvl 1 (0-10m) | Lvl 3 (20-30m) | Lvl 5 (40-50m) |
|----------|--------------|----------------|----------------|
| **Friction** | mu=0.90 | mu=0.35 | mu=0.05 |
| **Grass** | 2/m², drag=2 | 10/m², drag=10 | 20/m², drag=20 |
| **Boulders** | 3-5cm gravel | 25-35cm rocks | 80-120cm boulders |
| **Stairs** | 3cm×5 steps | 13cm×7 steps | 23cm×10 steps |

## Scoring

- 2 waypoints per level, 10 per quadrant, 40 total + 3 transitions = **43 waypoints**
- Level weights: [10, 20, 30, 40, 50] per quadrant
- Max per quadrant: 150, **max total: 600**

## Usage

```bash
# Rendered (local)
python src/run_ring_eval.py --rendered --policy rough \
    --checkpoint checkpoints/model.pt --num_episodes 1

# Headless (H100)
python src/run_ring_eval.py --headless --policy rough \
    --checkpoint checkpoints/model.pt --num_episodes 100
```

## Output

JSONL with per-quadrant + per-level scoring:
```json
{
  "total_waypoints_reached": 28,
  "composite_score": 320.0,
  "max_score": 600,
  "quadrant_scores": {
    "friction": {"waypoints": 10, "score": 150.0, "levels": {...}},
    "grass": {"waypoints": 8, "score": 110.0, "levels": {...}},
    ...
  }
}
```
