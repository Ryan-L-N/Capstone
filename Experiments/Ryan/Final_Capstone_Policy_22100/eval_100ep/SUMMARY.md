# 22100 Final Capstone Policy — Canonical 4-Env 100-Episode Eval

## What this is

Production evaluation of the shipped Final Capstone Policy
(`parkour_phasefwplus_22100.pt`) against the canonical 4-arena test bench
(friction / grass / boulder / stairs), 100 episodes per arena, run on the
H100 server in headless mode.

Run started Apr 29 18:27 UTC, completed Apr 30 17:44 UTC (~23 hours wall).
Source: `~/4_env_test/results/canonical_100iter_prod/` on the H100.

## Run config

```
Checkpoint:   parkour_phasefwplus_22100.pt
Mode:         headless
Episodes:     100 per env
Max ep time:  240s wall (~600s sim time at H100 headless speed)
Action scale: 0.3
--mason flag: enabled (Mason obs ordering)
Per-env target_vx + zone_slowdown_cap:
  friction: target_vx=3.0  cap=1.0
  grass:    target_vx=3.0  cap=3.0
  boulder:  target_vx=2.0  cap=0.67
  stairs:   target_vx=2.0  cap=1.0
```

## Headline results

| Env | n | COMPLETE | FELL | TIMEOUT | Median COMPLETE time | Max progress | z5 reach |
|---|---|---|---|---|---|---|---|
| **friction** | 100 | **96 (96%)** | 3 | 1 | 365s sim | 49.5m | 96 |
| **grass** | 100 | **75 (75%)** | 0 | 25 | 315s sim | 49.5m | 93 |
| **boulder** | 100 | 0 (0%) | 0 | 100 | — | 31.4m (z3–z4 wedge) | 0 |
| **stairs** | 100 | 0 (0%) | 51 | 49 | — | 25.0m (z2–z3) | 0 |

## Interpretation

**Strong:**
- Friction 96% complete (zone 5 ice μ=0.10–0.30) — robust DR transfer
- Grass 75% complete with 0 falls — clean policy
- 96/100 friction and 93/100 grass episodes reached zone 5

**Open:**
- Boulder: 0 falls but 100% timeout. Robot remains upright but cannot push
  past the dense zone-3 boulder field (38/100 stuck at 20–25m, 9/100 broke
  through to zone 4 ~30m)
- Stairs: 51 falls + 49 timeouts. Roughly half the episodes fall in zone 2
  and the other half wedge in zone 3

## File schema (JSONL, one episode per line)

```json
{
  "episode_id": "friction_rough_ep0042",
  "policy": "rough",
  "environment": "friction",
  "completion": true,
  "progress": 49.501,
  "zone_reached": 5,
  "time_to_complete": 363.58,
  "stability_score": 0.197757,
  "mean_roll": 0.027677,
  "mean_pitch": 0.06346,
  "height_variance": 0.000277,
  "mean_ang_vel": 0.2077,
  "fall_detected": false,
  "fall_location": null,
  "fall_zone": null,
  "mean_velocity": 1.3422,
  "total_energy": 0.0,
  "episode_length": 368.72
}
```

`progress` is meters traveled (max 50m, the goal). `zone_reached` is 1–5
(each zone = 10m). `time_to_complete` and `episode_length` are sim time
seconds. `fall_detected` is True for any FELL or FLIP outcome.

## How to load + analyze

```python
import json
import pandas as pd

dfs = {}
for env in ["friction", "grass", "boulder", "stairs"]:
    with open(f"{env}_rough_episodes.jsonl") as f:
        dfs[env] = pd.DataFrame([json.loads(l) for l in f])

# Example: completion rate per env
for env, df in dfs.items():
    pct = df["completion"].mean() * 100
    print(f"{env}: {pct:.0f}% complete")
```
