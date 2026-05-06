"""Bake invisible collision-only riser ramps into the FW staircase USDs.

The cheat: instead of authoring per-tread riser triangles, drop a single
tilted quad under each stair flight. The quad is invisible to rendering
(treads still look like stairs) but solid for collision — so any of Spot's
feet that would have fallen through the gap between treads now hit the
ramp instead.

Geometry is data-driven from data/fw_stair_waypoints.json (bottom_surface,
landing_at_mid_z, top_surface). For "straight" topology one flight; for
"switchback_full" two flights joined at the landing.

Run with isaaclab311 conda env from this directory:
    python bake_riser_cheat.py

Each USD is overwritten in place. The original *.usd.bak files made by
bake_stair_collision.py are still pristine — re-run that first if you
need to revert. The script is idempotent: if a /Risers/RiserCheat group
already exists it gets removed and rebuilt.

After this, run scripts/verify_fw_stair_layout_v2.py to visually confirm
the treads still look right (the ramps are invisible), then
scripts/run_fw_stair_eval.py to test that Spot now climbs.
"""
import json
import math
import os
import shutil

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 64, "height": 64})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(THIS_DIR)
WAYPOINTS_JSON = os.path.join(PROJECT_DIR, "data", "fw_stair_waypoints.json")
INSPECTION_JSON = os.path.join(PROJECT_DIR, "data", "fw_stair_inspection.json")

# Source USDs live in this dir (these are the ones we modify in place).
USD_DIR = THIS_DIR

TARGETS = [
    "SM_Staircase_01.usd",
    "SM_Staircase_02.usd",
    "SM_StaircaseHalf_01.usd",
    "SM_StaircaseHalf_02.usd",
]

# Switchback flight perpendicular width (m). Real flight widths vary by USD;
# we pad slightly so the ramp covers the flight footprint reliably.
SWITCHBACK_FLIGHT_WIDTH_M = 2.5

# Lateral safety pad added to straight-stair widths (m) — catches sideways
# drift off open-side stairs.
STRAIGHT_WIDTH_PAD_M = 0.3

# Tiny vertical offset (m) applied to ramp endpoints — drop the ramp by this
# much so it sits *just below* the tread leading edges, ensuring Spot's foot
# always lands on the tread first when a tread is available, but the ramp
# catches the fall when there's nothing else.
RAMP_DROP_M = 0.02


def _flight_quad_world_m(p0_xyz_m, p1_xyz_m, width_m):
    """Build 4 corners (in meters) of a tilted quad spanning a stair flight.

    Quad lies along the slope from p0 (bottom) to p1 (top), centered laterally
    on the flight axis, with width `width_m` perpendicular in the XY plane.
    Corner order (CCW from above): bot-left, bot-right, top-right, top-left.
    """
    bx, by, bz = p0_xyz_m
    tx, ty, tz = p1_xyz_m

    # XY direction along the slope (unit vector)
    dx, dy = tx - bx, ty - by
    n = math.hypot(dx, dy)
    if n < 1e-6:
        # Degenerate horizontal projection (e.g. true vertical) — use +X
        dx, dy = 1.0, 0.0
        n = 1.0
    ux, uy = dx / n, dy / n
    # Perpendicular (90° CCW in XY)
    px, py = -uy, ux
    h = width_m * 0.5

    # Drop ONLY the top endpoint below the back-top tread so treads win the
    # contact race up there. Keep the bottom endpoint at the front-bottom
    # tread surface — dropping the bottom buries the ramp under the world
    # ground plane (since the front-bottom is essentially AT z=0 after the
    # eval's lift_z), causing Spot to walk on flat ground instead of the
    # ramp for the first ~0.9m of the staircase.
    bz_r = bz  # no drop at the bottom
    tz_r = tz - RAMP_DROP_M

    v_bot_left = (bx - px * h, by - py * h, bz_r)
    v_bot_right = (bx + px * h, by + py * h, bz_r)
    v_top_right = (tx + px * h, ty + py * h, tz_r)
    v_top_left = (tx - px * h, ty - py * h, tz_r)
    return [v_bot_left, v_bot_right, v_top_right, v_top_left]


def _to_cm(corners_m):
    """USDs are authored in cm (metersPerUnit=0.01), waypoints are in m."""
    return [Gf.Vec3f(c[0] * 100.0, c[1] * 100.0, c[2] * 100.0) for c in corners_m]


def _add_riser_mesh(stage, parent_path, name, corners_m):
    mesh_path = f"{parent_path}/{name}"
    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(mesh_path))
    points = _to_cm(corners_m)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3, 3])
    # Two triangles spanning the quad.
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    # Make it invisible to rendering — collision only.
    UsdGeom.Imageable(mesh).MakeInvisible()
    # Apply collision APIs (mirror bake_stair_collision.py's choices).
    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    mc = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mc.CreateApproximationAttr().Set("none")  # exact triangle mesh
    return mesh


def _flights_for_usd(usd_name, wp, insp):
    """Return list of (p0_m, p1_m, width_m) tuples — one per flight."""
    topology = wp.get("topology", "straight")
    bot = wp["bottom_surface"]
    top = wp["top_surface"]
    p_bot = (bot["xy"][0], bot["xy"][1], bot["z"])
    p_top = (top["xy"][0], top["xy"][1], top["z"])

    bbox = wp["bbox_xy"]
    bbox_w = bbox["width"]
    bbox_d = bbox["depth"]

    if topology == "switchback_full":
        landing = wp["landing_at_mid_z"]
        p_mid = (landing["xy"][0], landing["xy"][1], landing["z"])
        # Sanity-check the landing: if its xy projection is wildly off-center
        # between bottom and top (one flight's run would be < 25% of the
        # bottom→top run), the vertex extractor probably misidentified it.
        # Fall back to a single straight ramp.
        run_total = math.hypot(p_top[0] - p_bot[0], p_top[1] - p_bot[1])
        run_a = math.hypot(p_mid[0] - p_bot[0], p_mid[1] - p_bot[1])
        run_b = math.hypot(p_top[0] - p_mid[0], p_top[1] - p_mid[1])
        if run_total > 1e-6 and (run_a / run_total < 0.25 or run_b / run_total < 0.25):
            print(
                f"    (landing looks unreliable: run_a={run_a:.2f} run_b={run_b:.2f} "
                f"total={run_total:.2f} — falling back to single straight ramp)",
                flush=True,
            )
        else:
            # Two flights: bottom→landing, landing→top.
            # Use SWITCHBACK_FLIGHT_WIDTH_M but cap to half the smaller bbox dim
            # so we don't blow past the staircase footprint.
            w = min(SWITCHBACK_FLIGHT_WIDTH_M, max(bbox_w, bbox_d) * 0.45)
            return [
                (p_bot, p_mid, w),
                (p_mid, p_top, w),
            ]
    # Straight: one flight, full bbox-perp width with small pad.
    # Determine perpendicular width from bbox: if x-extent dominates the
    # XY motion, perp is along Y, else along X.
    dx, dy = p_top[0] - p_bot[0], p_top[1] - p_bot[1]
    if abs(dx) >= abs(dy):
        perp_dim = bbox["y"][1] - bbox["y"][0]
    else:
        perp_dim = bbox["x"][1] - bbox["x"][0]
    width_m = perp_dim + STRAIGHT_WIDTH_PAD_M
    return [(p_bot, p_top, width_m)]


def bake_one(usd_name, wp, insp):
    usd_path = os.path.join(USD_DIR, usd_name)
    print(f"\n=== {usd_name} ===", flush=True)
    if not os.path.exists(usd_path):
        print(f"  ERROR: missing {usd_path}", flush=True)
        return False

    # Take a one-time pre-cheat backup so we can revert (separate from the
    # *.usd.bak left by bake_stair_collision.py).
    pre_cheat = usd_path + ".precheat.bak"
    if not os.path.exists(pre_cheat):
        shutil.copy2(usd_path, pre_cheat)
        print(f"  Backup -> {os.path.basename(pre_cheat)}", flush=True)

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        print(f"  ERROR: open failed", flush=True)
        return False

    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        # Fall back to the first top-level Xform/Mesh.
        for p in stage.GetPseudoRoot().GetChildren():
            default_prim = p
            break
    if not default_prim:
        print(f"  ERROR: no default prim", flush=True)
        return False

    parent_path = str(default_prim.GetPath()) + "/RiserCheat"

    # Idempotency: nuke any existing /RiserCheat from a prior run.
    if stage.GetPrimAtPath(parent_path):
        stage.RemovePrim(parent_path)
        print(f"  Removed prior RiserCheat group", flush=True)

    # Author the holder Xform.
    UsdGeom.Xform.Define(stage, Sdf.Path(parent_path))

    flights = _flights_for_usd(usd_name, wp, insp)
    for i, (p0_m, p1_m, w_m) in enumerate(flights):
        corners = _flight_quad_world_m(p0_m, p1_m, w_m)
        _add_riser_mesh(stage, parent_path, f"Flight_{i:02d}", corners)
        rise = p1_m[2] - p0_m[2]
        run = math.hypot(p1_m[0] - p0_m[0], p1_m[1] - p0_m[1])
        slope_deg = math.degrees(math.atan2(rise, run)) if run > 1e-6 else 90.0
        print(
            f"  Flight {i}: bot=({p0_m[0]:+.2f},{p0_m[1]:+.2f},{p0_m[2]:+.2f}) "
            f"top=({p1_m[0]:+.2f},{p1_m[1]:+.2f},{p1_m[2]:+.2f}) "
            f"width={w_m:.2f}m slope={slope_deg:.1f}deg",
            flush=True,
        )

    stage.GetRootLayer().Save()
    print(f"  Saved {usd_name} (added {len(flights)} riser flight(s))", flush=True)
    return True


def main():
    if not os.path.exists(WAYPOINTS_JSON):
        raise SystemExit(f"Missing {WAYPOINTS_JSON}")
    with open(WAYPOINTS_JSON) as f:
        wpts = json.load(f)
    insp = {}
    if os.path.exists(INSPECTION_JSON):
        with open(INSPECTION_JSON) as f:
            insp = json.load(f)

    ok = 0
    for t in TARGETS:
        if t not in wpts:
            print(f"\n=== {t} ===\n  WARN: no waypoints entry, skipping", flush=True)
            continue
        if bake_one(t, wpts[t], insp.get(t, {})):
            ok += 1

    print(f"\n{ok}/{len(TARGETS)} USDs updated", flush=True)


if __name__ == "__main__":
    main()
    app.close()
