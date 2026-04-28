# Final World Stairs Test

Focused eval that imports the 4 `SM_Staircase` USDs from the Final World scene
and runs each through an automated **up → turn 180° → down** sequence. Built
to validate stair-walking on architectural geometry separately from the
larger Final World teleop scene.

## Prerequisites

The `SM_Staircase_*.usd` files in `Collected_Final_World/SubUSDs/` must have
`UsdPhysics.CollisionAPI` baked in. The original Final World assets shipped
without collision (visual mesh only); run this once to fix:

```bash
python C:/Users/Gabriel\ Santiago/OneDrive/Desktop/Collected_Final_World/bake_stair_collision.py
```

Backups are saved as `*.usd.bak`. Verify with `check_stair_collision.py`.

## Usage

```bash
# All 4 stairs, headless (fast)
python Experiments/Alex/final_world_stairs_test/scripts/run_fw_stairs.py \
    --checkpoint Experiments/Alex/PARKOUR_NAV/checkpoints/parkour_phase7_15000.pt \
    --stairs all --headless

# One stair, rendered (watch it)
python Experiments/Alex/final_world_stairs_test/scripts/run_fw_stairs.py \
    --checkpoint Experiments/Alex/PARKOUR_NAV/checkpoints/parkour_phase7_15000.pt \
    --stairs 01 --rendered
```

## Sequence

For each staircase USD:
1. **Stabilize** — 400 steps zero-cmd while robot settles on flat ground
2. **Ascend** — command forward at `--up_speed` (default 0.8 m/s) until robot
   reaches the top end of the stair's long axis OR falls. Max 20s sim.
3. **Turn 180°** — command yaw rate at `--turn_rate` (default 1.5 rad/s)
   until 180° rotation reached. Max 4s sim.
4. **Descend** — command forward at `--down_speed` (default 0.6 m/s) until
   robot's z returns to within 0.2m of starting z. Max 20s sim.

Each phase is independently scored. **PASS** = ascend + turn + descend with no fall.

## Output

`results/fw_stairs_results.csv`:
```
stair,ascended,turned,descended,fell,z_high,success
01,True,True,True,False,1.85,True
02,...
```

## Why this experiment exists

- `Final_World.usd` is large + slow to load (lots of props, materials,
  lighting). Iterating policy on stairs alone is much faster here.
- Surfaces a clean A/B between procedural training stairs (4_env_test) and
  imported architectural stairs (Final World) — same policy, different
  geometry, isolates the architectural-stair gap.
- Future training (e.g. Phase-8+) can use the same arena to validate that
  fine-tunes don't regress architectural stair handling.

## Known gotchas

- Stair USDs **must have collision baked in** or robot walks through them
  (see Prerequisites). If a stair PASSes ascent visually but the policy never
  changed gait, suspect missing collision.
- The script translates each stair so its base sits at world origin. If the
  USD has unusual interior structure (railings, landings) the `top` detection
  via long-axis x/y may fire prematurely.
- `triangleMesh` collision approximation preserves stair-step ridges
  (vs convex hull which rounds them into a ramp).
