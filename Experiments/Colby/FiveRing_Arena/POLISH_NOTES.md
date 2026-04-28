# 5-Ring (4-Quadrant) Arena — Visual Polish for Colby

## What this folder is

The 5-ring arena (built originally as Alex's 4-quadrant test) is a single
combined-terrain gauntlet that exercises Spot across friction, grass,
boulder, and stairs in one continuous arena — concentric rings of
increasing difficulty per quadrant.

You're the visual / scene-design lead now. **The functional pieces are
done** (terrain physics, waypoints, collision, eval scoring) and the
arena is the standard benchmark for any ckpt we ship. What's left is
making it **look** like a serious robotics demo arena rather than a
debug scene.

## Quick orientation

```
FiveRing_Arena/
├── README.md                         ← Alex's original README
├── POLISH_NOTES.md                   ← THIS FILE
├── scripts/
│   └── run_eval.sh                   ← shell wrapper that calls run_ring_eval.py
└── src/
    ├── run_ring_eval.py              ← autonomous benchmark (waypoint follower)
    ├── run_ring_teleop.py            ← manual keyboard control (just added)
    ├── spot_rough_terrain_policy.py  ← policy wrapper (don't touch unless needed)
    ├── configs/
    │   ├── eval_cfg.py               ← physics dt, spawn pos, etc
    │   └── ring_params.py            ← arena layout: ring widths, level definitions
    ├── envs/
    │   ├── ring_arena.py             ← TOP-LEVEL builder: create_quadrant_arena()
    │   ├── base_arena.py             ← shared utilities
    │   ├── boulder_meshes.py         ← boulder generation
    │   ├── stair_pyramids.py         ← stair pyramid placement
    │   └── vegetation.py             ← grass drag-zone visuals
    ├── navigation/
    │   └── ring_follower.py          ← waypoint waypoint sequence (used by eval; also seeds stair positions)
    └── metrics/
        └── ring_collector.py         ← per-episode JSONL writer
```

## How to run it

**Eval (autonomous benchmark)** — single Spot follows the waypoint chain:
```bash
conda activate isaaclab311
cd Experiments/Colby/FiveRing_Arena
python src/run_ring_eval.py --policy rough \
    --checkpoint <path-to-policy.pt> \
    --num_episodes 1 --rendered \
    --mason --action_scale 0.3
```

**Teleop (manual)** — drive Spot with WASD/G/SHIFT/R:
```bash
python src/run_ring_teleop.py \
    --checkpoint <path-to-policy.pt> \
    --mason --action_scale 0.3
```

For a starter ckpt: `Experiments/Cole/PARKOUR_NAV_handoff/parkour_phase9_18500.pt`
or `Experiments/Ryan/PARKOUR_NAV_phasefwplus_22100/parkour_phasefwplus_22100.pt`.

## Where the visual design lives

Most of the "look" comes from these files (touch these, leave the
mechanics alone):

### `envs/ring_arena.py` — the master builder
- Sets ring geometry (4 quadrants × 5 levels = 20 difficulty patches +
  4 transition zones between quadrants).
- Calls into the per-element builders below.
- Currently uses **flat-color displayColor** on every prim (gray ground,
  green grass, brown boulders) — replace with **MaterialBinding** to a
  proper PBR material lib (UsdShade) for real surface texture.

### `envs/vegetation.py` — grass quadrant
- Procedurally places grass-blade meshes (cones / cubes scaled
  vertically) at random positions per level.
- Currently single dark-green displayColor. **Big visual win** would be:
  - Variable blade heights + tilt angles (already partial)
  - Color variation (yellower at the edges, lush green in center)
  - A lighter "ground beneath grass" material (currently same gray as
    rest of arena)
  - Optional: add some bushes / taller plants at level boundaries for
    visual interest

### `envs/boulder_meshes.py` — boulder quadrant
- Generates boulders as procedural box + sphere primitives at varied
  scales and rotations.
- Currently single brown displayColor. **Quick polish:**
  - Vary rock colors (dark gray, sandy tan, mossy green for biome variety)
  - Add some rounded boulder variants (currently just box-leaning shapes)
  - Optional: scatter small pebbles between large rocks at low density

### `envs/stair_pyramids.py` — stair quadrant
- Generates step pyramids at the waypoint positions provided by
  `ring_follower.QuadrantFollower.stairs_waypoint_positions()`.
- Currently single light-gray displayColor. **Polish:**
  - Differentiate tread vs riser surfaces (slight color variation)
  - Concrete-ish texture on treads, slightly weathered look
  - Optional: handrails (just thin cylinders along the top edge of each
    flight — no collision needed, purely visual)

### Friction quadrant (handled inline in `ring_arena.py`)
- Currently flat plane with displayColor varying by friction zone (light
  for high-mu, dark for low-mu / "ice").
- **Polish:** consider making low-friction zones look like ice (blue tint
  + faint reflection material) and high-friction look like dry concrete.

### Lighting (`run_ring_teleop.py:80-100` and `run_ring_eval.py:185-200`)
- Currently a generic warm DistantLight + cool DomeLight.
- **Big visual win:** time-of-day variation. The ARL 04 Teleop System
  has a sun-angle slider you could mirror.

## What NOT to change

- `configs/ring_params.py` — defines the level structure (NUM_QUADRANTS,
  LEVEL_WEIGHTS, ring widths). The eval scoring depends on these.
- `navigation/ring_follower.py` — waypoint definitions. If you move
  visual elements, **don't move where Spot expects to navigate to**.
- `metrics/ring_collector.py` — the JSONL output format is consumed by
  Alex's plotting / reporter.
- `spot_rough_terrain_policy.py` — policy wrapper, has the type-grouped
  DOF order bug fix from Apr 18.

## Suggested polish order (highest visual ROI first)

1. **PBR materials on ground / boulders / stairs** — single biggest
   "looks like a real demo" upgrade. Use Isaac Sim's built-in material
   library (`/Isaac/Materials/`) — drag-and-drop in Isaac Sim's GUI then
   read the resulting material path and bind it via UsdShade in code.
2. **Ice material on low-friction zones** — semi-reflective, blue tint.
3. **Better lighting** — sun angle 30° (afternoon), warm tone, plus
   subtle ambient sky fill. The current 3000-intensity sun is too flat.
4. **Vegetation variety** — grass + small bushes + taller blades, varied
   green tones. Currently all same.
5. **Cosmetic skybox** — Isaac Sim's `/Skybox/` selection (sunset,
   overcast, daytime — pick one and lock it in).
6. **Boulder variety** — color + shape range. Currently looks like
   chocolate cake squares.
7. **Stair handrails** — thin cylinders, no collision. Pure visual.

## Verification

After visual changes, run the eval to make sure you didn't break geometry:
```bash
python src/run_ring_eval.py --policy rough \
    --checkpoint <ckpt> --num_episodes 1 --rendered
```

Spot should still navigate the same waypoints and score similarly to the
baseline. If your changes drop the score by >5%, something physical broke
(usually a collision flag missing on a new geometric element).

## Files of interest from prior teams

- `Experiments/Alex/4_env_test/results/` — has reference results from the
  per-env arena (separate single-env tests, before 5-ring was built).
- `Experiments/Cole/PARKOUR_NAV_handoff/PROJECT_HAIL_MARY_EXPLAINED.md`
  — full project narrative + ckpt history.
- `Experiments/Ryan/PARKOUR_NAV_phasefwplus_22100/POLICY_DETAILS.md`
  — current best ckpt's full spec sheet.

Ping Alex / Gabriel if anything is unclear. Have fun making it pretty.

— Gabriel
