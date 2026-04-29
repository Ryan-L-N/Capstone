# FW Stairs Riser Integration — Plan for Colby

## The problem

The 4 SM_Staircase USDs in `Collected_Final_World/SubUSDs/` model
**architectural open-frame stairs** — only the horizontal **tread** surfaces
have collision geometry. The **vertical riser faces are absent**. When
Spot tries to climb them, three failure modes happen (all video-confirmed
across Phase-9, Phase-10, Phase-10b, Phase-FW-Plus):

1. **Foot drops into the gap between treads** — toe slides through where
   a riser would normally be
2. **Body wedges between treads** — chassis drops through the gap and
   gets pinched by thin tread strips
3. **Lateral drift off the open-side stair** — no walls; bot slides
   sideways into surrounding void

We trained 4 fine-tune phases of policy adaptations (Phase-7→Phase-FW-Plus)
to handle this geometry. Each made marginal progress; none crossed the
"actually climbs" threshold. The current Phase-Final from-scratch run is
the last training-side attempt.

**Your job (Colby's task):** instead of pushing the policy to handle
open-riser geometry, **modify the USDs to ADD solid riser faces between
the existing treads.** Spot's policy is already strong on solid-riser
stairs (Phase-9 hit zone-5 alive at 41.4m on procedural pyramid stairs).
If we can give the FW stairs solid riser collision, the existing policies
should walk them.

## What's in this folder

```
FW_Stairs_Riser_Project/
├── RISER_INTEGRATION_PLAN.md      ← THIS FILE (your roadmap)
├── README.md                       ← copied from final_world_stairs_test
├── scripts/
│   ├── inspect_fw_stair_geometry.py   ← bbox + mesh-prim Z analysis
│   ├── extract_fw_stair_waypoints.py  ← vertex-based waypoint extractor
│   ├── verify_fw_stair_layout_v2.py   ← visual layout sanity check (with BBoxCache fix)
│   ├── run_fw_stair_eval.py           ← automated state-machine eval driver
│   ├── verify_fw_stair_layout.py      ← v1 (kept for reference; v2 is the canonical)
│   └── run_fw_stairs.py               ← original abandoned script (reference only)
├── data/
│   ├── fw_stair_eval_config.json      ← per-USD spawn poses + phase sequences (ascend AND descend)
│   ├── fw_stair_inspection.json       ← per-USD bbox + native units
│   └── fw_stair_waypoints.json        ← vertex-extracted bottom/top/landing waypoints
└── usd_source/
    ├── SM_Staircase_01.usd            ← full switchback (5.3m tall)
    ├── SM_Staircase_02.usd            ← straight (5.3m tall)
    ├── SM_StaircaseHalf_01.usd        ← half switchback (3.1m tall)
    ├── SM_StaircaseHalf_02.usd        ← straight half (3.1m tall)
    ├── SM_Staircase_*.usd.bak         ← pre-CollisionAPI backups
    ├── bake_stair_collision.py        ← script that baked the current triangleMesh collision
    └── inspect_fw_stairs.py           ← original bbox inspector
```

## What's already done

1. **CollisionAPI baked** on all 4 USDs (`triangleMesh` approximation
   — preserves the existing tread surfaces, but does NOT add risers).
   Source: `usd_source/bake_stair_collision.py`. Backups in `*.usd.bak`.
2. **Vertex extraction** — we know where every tread surface lives in
   each USD (`data/fw_stair_waypoints.json`).
3. **Geometry verifier** — `verify_fw_stair_layout_v2.py` confirms the
   USD lifts cleanly into a scene at scale=0.01 with the bbox at z=0.
4. **Eval framework** — `run_fw_stair_eval.py` runs Spot through each USD
   automatically with `--direction ascend|descend|both`. Pre-baked spawn
   poses + phase state machines per USD.

## The geometry problem you're solving

Each USD is a **single triangle mesh** (mesh count = 1 per USD). The
existing CollisionAPI is `MeshCollisionAPI(approximation="none")` =
exact triangle mesh. The mesh CONTAINS triangles for tread surfaces but
NOT for riser faces.

**Goal:** add new triangles (or new mesh prims) that fill in the riser
faces — vertical rectangles between consecutive tread edges.

For `SM_Staircase_02` (straight 5.3m, the simplest case):
- Treads exist at Z heights ~0.18m, 0.36m, 0.54m, ... 5.3m
  (~30 treads × ~0.18m riser)
- Run direction: `-X` (bottom at +X=4.4, top at +X=0)
- A riser face would be a rectangle at the **front edge of each tread**,
  spanning from `(tread_top_z - riser_height)` to `tread_top_z` in Z,
  full Y width of the stair, at the X coord of that tread's leading edge.

For switchbacks (`SM_Staircase_01`, half_01), each flight has its own
riser direction. You'll need to handle two flight orientations.

## Three implementation paths (ranked by effort/risk)

### Path A: Procedural riser meshes via PXR Python (recommended)

**Effort:** 1-2 days. **Risk:** medium (USD authoring complexity).

For each USD:
1. Open the existing mesh, extract vertex positions sorted by Z.
2. Cluster vertices into "tread surfaces" by Z bin (use
   `data/fw_stair_waypoints.json` cluster centers as starting points).
3. For each pair of consecutive Z clusters, identify the front edge of
   the upper tread (the boundary in XY between tread surface and empty
   space).
4. Generate 2 triangles per riser face: a vertical rectangle from
   `(prev_tread_z, x_front, y_min)` to `(curr_tread_z, x_front, y_max)`.
5. Append the new triangles to the mesh's `points`, `faceVertexCounts`,
   and `faceVertexIndices` attributes.
6. Save USD; verify with `verify_fw_stair_layout_v2.py`.

**Key PXR APIs:**
- `UsdGeom.Mesh.GetPointsAttr().Get()` / `.Set()` — vertex positions
- `UsdGeom.Mesh.GetFaceVertexIndicesAttr()` — triangle indices
- `UsdGeom.Mesh.GetFaceVertexCountsAttr()` — per-face vertex count (should all be 3 for triangles)

After you've added the riser triangles, **re-run** `bake_stair_collision.py`
on the modified USDs to refresh the CollisionAPI/MeshCollisionAPI
attributes (`triangleMesh` approximation will pick up the new triangles
automatically).

### Path B: Add separate riser-mesh prims (cleaner but more PXR work)

**Effort:** 2-3 days. **Risk:** low (cleaner separation of concerns).

Instead of mutating the original mesh, create a NEW mesh prim per riser:

```
/SM_Staircase_02/Risers/Riser_00  ← UsdGeom.Mesh with 2 triangles
/SM_Staircase_02/Risers/Riser_01
...
```

This keeps the original geometry intact and lets you toggle risers on/off
by enabling/disabling the `Risers` Xform. Apply `CollisionAPI` +
`MeshCollisionAPI(triangleMesh)` to each new prim individually.

### Path C: External DCC tool (Blender / Maya)

**Effort:** 1 day if you have the tools. **Risk:** low.

Open each USD in Blender, manually draw the riser faces in edit mode,
re-export. Don't bake collision until after — let the eval framework's
`bake_stair_collision.py` handle that.

---

## Verification workflow

After modifying USDs, run this check sequence:

1. **Visual sanity** (each USD):
   ```bash
   conda activate isaaclab311
   cd Experiments/Colby/FW_Stairs_Riser_Project
   python scripts/verify_fw_stair_layout_v2.py --stair SM_Staircase_02.usd --rendered
   ```
   Confirm: USD geometry visible, no buried geometry, marker spheres land
   on the right surfaces (cyan = spawn at bottom, yellow = ascend target).

2. **Collision check** (each USD):
   ```bash
   python usd_source/bake_stair_collision.py
   ```
   Re-bake CollisionAPI on the modified USDs. The script overwrites
   in-place; the original `*.usd.bak` files remain pristine.

3. **Eval with Phase-9 ckpt** (the strongest known good policy):
   ```bash
   python scripts/run_fw_stair_eval.py \
       --checkpoint <path-to-parkour_phase9_18500.pt> \
       --action_scale 0.3 --mason \
       --stairs all --direction ascend --rendered
   ```
   The Phase-9 ckpt is in `Experiments/Cole/Final_Capstone_Policy_handoff/parkour_phase9_18500.pt`
   (already shipped). Run with `--direction ascend` first; if that passes,
   try `--direction both`.

4. **Pass criterion:** at least 2/4 USDs PASS ascent on Phase-9 with the
   modified geometry. If solid risers fix the climb problem, the success
   rate should jump from current 0/4 to 2-4/4.

If verification passes, push the modified USDs back to
`Collected_Final_World/SubUSDs/` (or somewhere similar) and notify
Alex/Gabriel — this unblocks the entire FW deployment story.

---

## Useful background context

- **The policies we have** (see `Experiments/Cole/Final_Capstone_Policy_handoff/`):
  - `parkour_phase5_11000.pt` — Cole record holder (21/25 max-density)
  - `parkour_phase8_16497.pt` — zero-fall 4-env baseline
  - `parkour_phase9_18500.pt` — stair zone-5 alive specialist (test
    against this one first; if it can't walk modified USDs, none can)

- **Action scale + observation spec** for these policies: see
  `Experiments/Ryan/Final_Capstone_Policy_22100/POLICY_DETAILS.md` —
  full spec sheet. TL;DR: action_scale=0.3, type-grouped DOF order,
  235-dim obs, 50 Hz control.

- **Why we did this side quest now:** we already ran 4 attempts to fix
  the FW stair issue policy-side. Each one cost a 12-18h H-100 cycle
  and gave marginal-or-zero improvement. Modifying the geometry is the
  logical next move because the existing policies are already strong on
  solid-riser geometry — we just need the deployment scene to match the
  training distribution.

- **Eval framework caveats:**
  - `BBoxCache` MUST be recreated after `op_t.Set()` for transforms to
    show up in computed bounds. The `verify_fw_stair_layout_v2.py` does
    this correctly. Don't reuse a single cache across multiple
    transform changes.
  - The USDs use `metersPerUnit = 0.01` (cm units) and `up_axis = Z`.
    Apply `scale=0.01` on the holder Xform. The `setup_stair_in_stage()`
    function in `run_fw_stair_eval.py` shows the correct pattern.

---

## Questions / asks

1. If you find that the riser geometry is non-trivial to author
   procedurally (especially for the switchback `SM_Staircase_01`), ask
   Alex / Gabriel — we have the vertex coordinates for every tread and
   can probably help script the riser placement.
2. If verification with Phase-9 still doesn't pass after adding risers,
   that's important data — it means the issue isn't *just* the missing
   riser geometry. Could be tread height, surface friction, mesh seams
   in the original USD, etc.
3. Once you have a modified USD that Spot can climb, please commit it
   alongside the eval log so we have the before/after evidence.

Ping me with anything that doesn't make sense.

— Gabriel
