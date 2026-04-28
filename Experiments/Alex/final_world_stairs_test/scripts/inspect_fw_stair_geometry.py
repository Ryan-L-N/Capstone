"""Inspect each SM_Staircase USD and dump everything we need to decide
waypoints + spawn pose. Writes a JSON we can hand-edit before the eval.

For each USD, reports:
  - metersPerUnit (cm vs m)
  - up_axis (Y vs Z)
  - bbox (in meters, after scale-correction)
  - mesh count + per-mesh world-z center (sorted)
  - tread Z-clusters: distinct height steps along the long axis
  - inferred flight 1 / flight 2 directions (xy projection)

Output: data/fw_stair_inspection.json — one entry per USD
"""
import json
import os
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "width": 64, "height": 64})

from pxr import Usd, UsdGeom, Gf, UsdPhysics  # noqa: E402

ROOT = r"C:\Users\Gabriel Santiago\OneDrive\Desktop\Collected_Final_World\SubUSDs"
TARGETS = [
    "SM_Staircase_01.usd",
    "SM_Staircase_02.usd",
    "SM_StaircaseHalf_01.usd",
    "SM_StaircaseHalf_02.usd",
]

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "fw_stair_inspection.json",
)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


def cluster_zs(zs, gap=0.04):
    """Cluster sorted z values; return list of (mean_z, count)."""
    if not zs:
        return []
    zs = sorted(zs)
    clusters = [[zs[0]]]
    for z in zs[1:]:
        if z - clusters[-1][-1] < gap:
            clusters[-1].append(z)
        else:
            clusters.append([z])
    return [(sum(c) / len(c), len(c)) for c in clusters]


def measure(usd_path):
    print(f"\n=== {os.path.basename(usd_path)} ===", flush=True)
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        print("  ERROR: open failed", flush=True)
        return None

    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    up_axis = UsdGeom.GetStageUpAxis(stage)
    print(f"  metersPerUnit = {mpu}", flush=True)
    print(f"  up_axis       = {up_axis}", flush=True)

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    # Walk meshes once — collect per-mesh world bbox in NATIVE units
    meshes = []
    for p in stage.Traverse():
        if p.IsA(UsdGeom.Mesh):
            try:
                bbox = cache.ComputeWorldBound(p).ComputeAlignedRange()
                if bbox.IsEmpty():
                    continue
                meshes.append({
                    "path": str(p.GetPath()),
                    "min": [bbox.GetMin()[i] for i in range(3)],
                    "max": [bbox.GetMax()[i] for i in range(3)],
                })
            except Exception as e:
                print(f"  bbox error on {p.GetPath()}: {e}", flush=True)

    if not meshes:
        print("  no meshes found", flush=True)
        return None

    # Aggregate native-unit bbox across all meshes
    all_min = [min(m["min"][i] for m in meshes) for i in range(3)]
    all_max = [max(m["max"][i] for m in meshes) for i in range(3)]

    # Convert to meters (this is the actual eval-scene scale)
    bbox_min_m = [v * mpu for v in all_min]
    bbox_max_m = [v * mpu for v in all_max]
    size_m = [bbox_max_m[i] - bbox_min_m[i] for i in range(3)]

    print(f"  native bbox    min={tuple(round(v,2) for v in all_min)} max={tuple(round(v,2) for v in all_max)}", flush=True)
    print(f"  meters bbox    min={tuple(round(v,3) for v in bbox_min_m)} max={tuple(round(v,3) for v in bbox_max_m)}", flush=True)
    print(f"  size (m)       W x D x H = {size_m[0]:.2f} x {size_m[1]:.2f} x {size_m[2]:.2f}", flush=True)
    print(f"  mesh count     {len(meshes)}", flush=True)

    # If up_axis is Y, we need to remap so Z is the height
    # (Isaac Lab + our eval expect Z-up.)
    # For analysis we use the up-axis component as "height".
    height_axis = 1 if up_axis == "Y" else 2
    width_axis = 0
    depth_axis = 2 if up_axis == "Y" else 1

    # Per-mesh height-axis center, in meters
    mesh_heights_m = sorted([
        ((m["min"][height_axis] + m["max"][height_axis]) / 2.0) * mpu
        for m in meshes
    ])
    clusters = cluster_zs(mesh_heights_m, gap=0.04)
    print(f"  height clusters ({len(clusters)} levels):", flush=True)
    for h, cnt in clusters:
        print(f"    z={h:+.3f}m  ({cnt} mesh{'es' if cnt!=1 else ''})", flush=True)

    # Inferred riser height (assume monotonic stair flights)
    if len(clusters) >= 2:
        diffs = [clusters[i+1][0] - clusters[i][0] for i in range(len(clusters)-1)]
        # Filter out very large gaps (between flights or to top platform)
        rising_diffs = [d for d in diffs if 0.05 < d < 0.40]
        if rising_diffs:
            riser_est = sum(rising_diffs) / len(rising_diffs)
            print(f"  inferred riser ≈ {riser_est*100:.1f} cm  ({len(rising_diffs)} stair gaps)", flush=True)

    # Flight direction analysis: at the lowest cluster (bottom flight),
    # what's the XY centroid? At the highest cluster (top), what's the XY centroid?
    # The XY direction from bottom->top tells us the dominant flight axis.
    # For switchbacks, mid-cluster centroid will be offset from the line.
    sorted_meshes = sorted(meshes, key=lambda m: (m["min"][height_axis] + m["max"][height_axis]) / 2.0)
    n = len(sorted_meshes)

    def centroid_xy(mlist):
        if not mlist:
            return [0, 0]
        x = sum((m["min"][0] + m["max"][0]) / 2.0 for m in mlist) / len(mlist) * mpu
        y_axis = depth_axis  # this is the "horizontal depth" axis
        y = sum((m["min"][y_axis] + m["max"][y_axis]) / 2.0 for m in mlist) / len(mlist) * mpu
        return [x, y]

    bot_third = sorted_meshes[: max(1, n // 3)]
    mid_third = sorted_meshes[n // 3 : 2 * n // 3]
    top_third = sorted_meshes[2 * n // 3 :]

    bot_xy = centroid_xy(bot_third)
    mid_xy = centroid_xy(mid_third)
    top_xy = centroid_xy(top_third)
    print(f"  centroid bot    xy(m) = ({bot_xy[0]:+.2f}, {bot_xy[1]:+.2f})  z={mesh_heights_m[0]:+.3f}", flush=True)
    print(f"  centroid mid    xy(m) = ({mid_xy[0]:+.2f}, {mid_xy[1]:+.2f})", flush=True)
    print(f"  centroid top    xy(m) = ({top_xy[0]:+.2f}, {top_xy[1]:+.2f})  z={mesh_heights_m[-1]:+.3f}", flush=True)

    # Switchback detection: if mid-centroid is offset from bot->top line, it's a switchback
    bot_to_top_dx = top_xy[0] - bot_xy[0]
    bot_to_top_dy = top_xy[1] - bot_xy[1]
    mid_to_line_dx = mid_xy[0] - (bot_xy[0] + bot_to_top_dx * 0.5)
    mid_to_line_dy = mid_xy[1] - (bot_xy[1] + bot_to_top_dy * 0.5)
    mid_offset = (mid_to_line_dx**2 + mid_to_line_dy**2) ** 0.5
    line_len = (bot_to_top_dx**2 + bot_to_top_dy**2) ** 0.5
    print(f"  bot->top XY span    = {line_len:.2f} m", flush=True)
    print(f"  mid offset from line = {mid_offset:.2f} m  (>0.5 = switchback)", flush=True)
    is_switchback = mid_offset > 0.5

    return {
        "file": os.path.basename(usd_path),
        "meters_per_unit": mpu,
        "up_axis": up_axis,
        "bbox_min_m": bbox_min_m,
        "bbox_max_m": bbox_max_m,
        "size_m": size_m,
        "mesh_count": len(meshes),
        "height_levels_m": [round(c[0], 4) for c in clusters],
        "centroid_bottom_xy_m": bot_xy,
        "centroid_mid_xy_m": mid_xy,
        "centroid_top_xy_m": top_xy,
        "bottom_z_m": round(mesh_heights_m[0], 4),
        "top_z_m": round(mesh_heights_m[-1], 4),
        "bot_to_top_xy_span_m": round(line_len, 3),
        "mid_offset_from_line_m": round(mid_offset, 3),
        "is_switchback": is_switchback,
    }


results = {}
for f in TARGETS:
    p = os.path.join(ROOT, f)
    if not os.path.exists(p):
        print(f"NOT FOUND: {p}", flush=True)
        continue
    info = measure(p)
    if info is not None:
        results[f] = info

with open(OUT_PATH, "w") as fp:
    json.dump(results, fp, indent=2)
print(f"\nWrote {OUT_PATH}", flush=True)

app.close()
