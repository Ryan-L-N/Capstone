"""Extract waypoints from each SM_Staircase USD by reading raw mesh vertices.

Strategy:
  - Open USD, get the single Mesh prim's `points` attribute (vertex array).
  - Scale by metersPerUnit so all coords are in meters.
  - Bin vertices by Z into 0.05m strata, find the lowest (`bottom`) and
    highest (`top`) strata that have substantial vertex count (>10).
  - For each stratum, compute its XY centroid — that's the spatial location
    of the bottom-most / top-most surface.
  - For switchback detection: project all vertices onto the dominant XY axis
    (X if width > depth, else Y), bin by that, look for a "kink" in the
    Z-vs-axis curve — that's the landing.

Output: data/fw_stair_waypoints.json with per-USD waypoints in
        AFTER-SCALE world meters, ASSUMING the USD is referenced at origin
        with NO additional transforms (caller must apply identical scale=0.01).
"""
import json
import os

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 64, "height": 64})

import numpy as np  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

ROOT = r"C:\Users\Gabriel Santiago\OneDrive\Desktop\Collected_Final_World\SubUSDs"
TARGETS = [
    "SM_Staircase_01.usd",
    "SM_Staircase_02.usd",
    "SM_StaircaseHalf_01.usd",
    "SM_StaircaseHalf_02.usd",
]

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "fw_stair_waypoints.json",
)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


def extract(usd_path):
    name = os.path.basename(usd_path)
    print(f"\n=== {name} ===", flush=True)
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        print("  ERROR: open failed", flush=True)
        return None

    mpu = UsdGeom.GetStageMetersPerUnit(stage)

    # Find the single Mesh prim and grab its points
    pts_native = None
    mesh_path = None
    for p in stage.Traverse():
        if p.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(p)
            attr = mesh.GetPointsAttr()
            arr = attr.Get()
            if arr is not None and len(arr) > 0:
                pts_native = np.array([(v[0], v[1], v[2]) for v in arr], dtype=np.float64)
                mesh_path = str(p.GetPath())
                break

    if pts_native is None:
        print("  ERROR: no mesh points found", flush=True)
        return None

    pts_m = pts_native * mpu  # to meters
    print(f"  mesh path: {mesh_path}", flush=True)
    print(f"  vertices: {len(pts_m)}", flush=True)

    z_min = float(pts_m[:, 2].min())
    z_max = float(pts_m[:, 2].max())
    z_range = z_max - z_min
    print(f"  z range:  {z_min:+.3f} -> {z_max:+.3f}  ({z_range:.3f}m total rise)", flush=True)

    # Bin Z into 0.05m strata
    z_bins = np.arange(z_min, z_max + 0.05, 0.05)
    z_idx = np.digitize(pts_m[:, 2], z_bins) - 1
    z_idx = np.clip(z_idx, 0, len(z_bins) - 2)
    counts = np.bincount(z_idx, minlength=len(z_bins) - 1)

    # Bottom-most stratum with >= 20 verts
    bot_bin = None
    for i in range(len(counts)):
        if counts[i] >= 20:
            bot_bin = i
            break
    # Top-most stratum
    top_bin = None
    for i in range(len(counts) - 1, -1, -1):
        if counts[i] >= 20:
            top_bin = i
            break

    if bot_bin is None or top_bin is None or bot_bin >= top_bin:
        print(f"  ERROR: bin extraction failed (bot={bot_bin}, top={top_bin})", flush=True)
        return None

    bot_pts = pts_m[(z_idx >= bot_bin) & (z_idx <= bot_bin + 1)]
    top_pts = pts_m[(z_idx >= top_bin - 1) & (z_idx <= top_bin)]

    bot_xy = (float(bot_pts[:, 0].mean()), float(bot_pts[:, 1].mean()))
    bot_z  = float(bot_pts[:, 2].mean())
    top_xy = (float(top_pts[:, 0].mean()), float(top_pts[:, 1].mean()))
    top_z  = float(top_pts[:, 2].mean())
    print(f"  bottom surface  xy=({bot_xy[0]:+.3f}, {bot_xy[1]:+.3f})  z={bot_z:+.3f}  (n={len(bot_pts)})", flush=True)
    print(f"  top    surface  xy=({top_xy[0]:+.3f}, {top_xy[1]:+.3f})  z={top_z:+.3f}  (n={len(top_pts)})", flush=True)

    # Bbox in xy
    x_min, x_max = float(pts_m[:, 0].min()), float(pts_m[:, 0].max())
    y_min, y_max = float(pts_m[:, 1].min()), float(pts_m[:, 1].max())
    width  = x_max - x_min
    depth  = y_max - y_min
    print(f"  xy bbox  x=[{x_min:+.3f}, {x_max:+.3f}]  y=[{y_min:+.3f}, {y_max:+.3f}]  ({width:.2f} x {depth:.2f})", flush=True)

    # Switchback heuristic:
    # - "full switchback" = footprint ~square (width ~= depth) AND Z is high
    # - "half switchback" = wider in one of width/depth (>1.5x ratio)
    # - "straight" = narrow in one dimension (<2.5m short axis)
    short_axis_len = min(width, depth)
    long_axis_len = max(width, depth)
    aspect = long_axis_len / max(short_axis_len, 0.01)
    short_is_x = width < depth  # short axis along X = stair runs in Y

    if short_axis_len < 2.5 and aspect > 1.8:
        topology = "straight"
    elif aspect < 1.3:
        topology = "switchback_full"
    else:
        topology = "switchback_half"
    print(f"  topology guess: {topology}  (aspect={aspect:.2f}, short_axis_len={short_axis_len:.2f})", flush=True)

    # Approach direction: along the long axis, from bot_xy toward top_xy.
    # For straight stairs: bot/top XY differ by long_axis_len.
    # For switchbacks: bot/top XY are close, but split along the dominant axis.
    bot_to_top_dx = top_xy[0] - bot_xy[0]
    bot_to_top_dy = top_xy[1] - bot_xy[1]
    bot_to_top_dist = (bot_to_top_dx**2 + bot_to_top_dy**2) ** 0.5
    print(f"  bottom -> top XY distance = {bot_to_top_dist:.3f} m", flush=True)

    # Find landings: midZ stratum centroid (z near (bot_z + top_z) / 2)
    mid_z = (bot_z + top_z) / 2.0
    mid_pts = pts_m[np.abs(pts_m[:, 2] - mid_z) < 0.10]
    landing_xy = None
    if len(mid_pts) > 50:
        landing_xy = (float(mid_pts[:, 0].mean()), float(mid_pts[:, 1].mean()))
        print(f"  landing(@mid-z) xy=({landing_xy[0]:+.3f}, {landing_xy[1]:+.3f})  z={mid_z:+.3f}  (n={len(mid_pts)})", flush=True)
    else:
        print(f"  no mid-z landing detected (n={len(mid_pts)})", flush=True)

    # For ALL flights, sample at quarter heights for additional waypoints
    quarter_pts = pts_m[(pts_m[:, 2] > bot_z + (top_z-bot_z) * 0.20) & (pts_m[:, 2] < bot_z + (top_z-bot_z) * 0.30)]
    quarter_xy = None
    if len(quarter_pts) > 20:
        quarter_xy = (float(quarter_pts[:, 0].mean()), float(quarter_pts[:, 1].mean()))

    three_quarter_pts = pts_m[(pts_m[:, 2] > bot_z + (top_z-bot_z) * 0.70) & (pts_m[:, 2] < bot_z + (top_z-bot_z) * 0.80)]
    three_quarter_xy = None
    if len(three_quarter_pts) > 20:
        three_quarter_xy = (float(three_quarter_pts[:, 0].mean()), float(three_quarter_pts[:, 1].mean()))

    return {
        "file": name,
        "meters_per_unit": mpu,
        "topology": topology,
        "aspect_ratio": round(aspect, 3),
        "z_min": round(z_min, 4),
        "z_max": round(z_max, 4),
        "rise_m": round(z_range, 3),
        "bbox_xy": {
            "x": [round(x_min, 4), round(x_max, 4)],
            "y": [round(y_min, 4), round(y_max, 4)],
            "width": round(width, 3),
            "depth": round(depth, 3),
        },
        "bottom_surface": {"xy": [round(bot_xy[0], 4), round(bot_xy[1], 4)], "z": round(bot_z, 4)},
        "top_surface":    {"xy": [round(top_xy[0], 4), round(top_xy[1], 4)], "z": round(top_z, 4)},
        "landing_at_mid_z": (
            {"xy": [round(landing_xy[0], 4), round(landing_xy[1], 4)], "z": round(mid_z, 4)}
            if landing_xy else None
        ),
        "quarter_xy": list(round(v, 4) for v in quarter_xy) if quarter_xy else None,
        "three_quarter_xy": list(round(v, 4) for v in three_quarter_xy) if three_quarter_xy else None,
        "bot_to_top_xy_dist_m": round(bot_to_top_dist, 3),
    }


results = {}
for f in TARGETS:
    p = os.path.join(ROOT, f)
    if not os.path.exists(p):
        print(f"NOT FOUND: {p}", flush=True)
        continue
    info = extract(p)
    if info:
        results[f] = info

with open(OUT_PATH, "w") as fp:
    json.dump(results, fp, indent=2)
print(f"\nWrote {OUT_PATH}", flush=True)

app.close()
