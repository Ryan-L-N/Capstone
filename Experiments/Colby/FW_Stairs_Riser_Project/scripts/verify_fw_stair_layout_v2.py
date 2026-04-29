"""Verifier v2 — measure-then-correct placement.

Instead of guessing lift_z from offline vertex extraction (which picked up
railing-post bottoms and got fooled), this script:
  1. Reference the USD bare (scale=0.01, translate=0).
  2. Step the world once so composition resolves.
  3. Query the COMPOSED world bbox of the referenced prim.
  4. Set the holder translate so the bbox MIN_Z lands at world z=0.
  5. Re-step, re-query bbox to confirm.
  6. Drop spawn + waypoint markers using the corrected world coords.

Usage:
    python scripts/verify_fw_stair_layout_v2.py --stair SM_Staircase_01.usd
"""
import argparse
import json
import math
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--stair", type=str, default="SM_Staircase_01.usd")
parser.add_argument("--headless", action="store_true", default=False)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402
app = SimulationApp({"headless": args.headless, "width": 1920, "height": 1080})

from omni.isaac.core import World  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(os.path.dirname(THIS_DIR), "data", "fw_stair_eval_config.json")
USD_ROOT = r"C:\Users\Gabriel Santiago\OneDrive\Desktop\Collected_Final_World\SubUSDs"

with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

if args.stair not in CONFIG:
    print(f"ERROR: '{args.stair}' not in config", flush=True)
    app.close(); sys.exit(1)

cfg = CONFIG[args.stair]
print(f"\n=== Verifying {args.stair} ({cfg['topology']}) ===", flush=True)

world = World(stage_units_in_meters=1.0)
stage = world.stage

# Lighting
UsdLux.DistantLight.Define(stage, "/World/SunLight").GetIntensityAttr().Set(2500.0)
UsdLux.DomeLight.Define(stage, "/World/DomeLight").GetIntensityAttr().Set(800.0)

# Ground plane
ground_path = "/World/Ground/Plane"
ground_x = UsdGeom.Xform.Define(stage, "/World/Ground")
ground_mesh = UsdGeom.Cube.Define(stage, ground_path)
ground_mesh.GetSizeAttr().Set(1.0)
gxform = UsdGeom.Xformable(ground_x.GetPrim())
gxform.ClearXformOpOrder()
gxform.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
gxform.AddScaleOp().Set(Gf.Vec3d(40.0, 40.0, 0.1))
UsdPhysics.CollisionAPI.Apply(ground_mesh.GetPrim())

# Reference the USD into a holder, scale ONLY (no translate yet)
holder_path = "/World/Stair_holder"
inner_path  = f"{holder_path}/Stair"
holder = UsdGeom.Xform.Define(stage, holder_path)
inner  = UsdGeom.Xform.Define(stage, inner_path)
inner.GetPrim().GetReferences().AddReference(os.path.join(USD_ROOT, args.stair))

hxform = UsdGeom.Xformable(holder.GetPrim())
hxform.ClearXformOpOrder()
op_t = hxform.AddTranslateOp()
op_s = hxform.AddScaleOp()
op_t.Set(Gf.Vec3d(0.0, 0.0, 0.0))
op_s.Set(Gf.Vec3d(cfg["scale"], cfg["scale"], cfg["scale"]))

# Step once so reference composes
world.reset()
world.step(render=False)

# Measure world bbox of the holder subtree
bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_], useExtentsHint=False)
bbox = bbox_cache.ComputeWorldBound(holder.GetPrim()).ComputeAlignedRange()
print(f"  PRE-LIFT bbox  min=({bbox.GetMin()[0]:.3f}, {bbox.GetMin()[1]:.3f}, {bbox.GetMin()[2]:.3f})", flush=True)
print(f"                 max=({bbox.GetMax()[0]:.3f}, {bbox.GetMax()[1]:.3f}, {bbox.GetMax()[2]:.3f})", flush=True)

# Compute lift to put bbox MIN_Z at world z=0 + small clearance
clearance = 0.01
observed_min_z = bbox.GetMin()[2]
correct_lift = -observed_min_z + clearance
print(f"  observed min_z = {observed_min_z:.3f}m  -> setting lift_z = {correct_lift:.3f}m", flush=True)

op_t.Set(Gf.Vec3d(0.0, 0.0, correct_lift))
world.step(render=False)
# CRITICAL: BBoxCache caches results — recreate after transform change.
bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_], useExtentsHint=False)
bbox_cache.Clear()
bbox = bbox_cache.ComputeWorldBound(holder.GetPrim()).ComputeAlignedRange()
post_min_z = bbox.GetMin()[2]
post_max_z = bbox.GetMax()[2]
print(f"  POST-LIFT bbox min_z = {post_min_z:.3f}  max_z = {post_max_z:.3f}", flush=True)
if abs(post_min_z) > 0.05:
    print(f"  WARNING: bbox MIN_Z did not land at 0 (got {post_min_z}). Reference composition shadowing the translate?", flush=True)

# Now compute corrected world coords for spawn + waypoints by adding correct_lift to z
# (the original config's coords assumed lift_z = cfg['lift_z'], so the delta is)
delta_z = correct_lift - cfg["lift_z"]
print(f"  delta_z (config vs observed) = {delta_z:+.3f}m", flush=True)


def add_sphere(path, world_xy, world_z, radius, rgb):
    s = UsdGeom.Sphere.Define(stage, path)
    s.GetRadiusAttr().Set(radius)
    sx = UsdGeom.Xformable(s.GetPrim())
    sx.ClearXformOpOrder()
    sx.AddTranslateOp().Set(Gf.Vec3d(world_xy[0], world_xy[1], world_z))
    s.GetDisplayColorAttr().Set([Gf.Vec3f(*rgb)])
    return s


def add_arrow(path, base_xy, base_z, yaw_rad, length=0.8):
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.GetHeightAttr().Set(length)
    cyl.GetRadiusAttr().Set(0.04)
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.ClearXformOpOrder()
    op_t = xf.AddTranslateOp()
    op_r = xf.AddRotateXYZOp()
    op_r.Set(Gf.Vec3f(0.0, 90.0, math.degrees(yaw_rad)))
    cx = base_xy[0] + math.cos(yaw_rad) * length / 2.0
    cy = base_xy[1] + math.sin(yaw_rad) * length / 2.0
    op_t.Set(Gf.Vec3d(cx, cy, base_z + 0.1))
    cyl.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.4, 1.0)])
    return cyl


# Spawn marker (cyan) — XY from config, Z = z_world (config Z is body height above ground = 0.55)
spawn = cfg["spawn"]
add_sphere("/World/Markers/Spawn", spawn["world_xy"], spawn["world_z"], 0.18, (0.0, 1.0, 1.0))
add_arrow("/World/Markers/Spawn_arrow", spawn["world_xy"], spawn["world_z"], math.radians(spawn["yaw_deg"]))
print(f"  spawn @ ({spawn['world_xy'][0]:.2f}, {spawn['world_xy'][1]:.2f}, {spawn['world_z']:.2f})  yaw={spawn['yaw_deg']} deg", flush=True)

# Waypoint markers — apply delta_z correction to z
ascend_n = 0
for i, ph in enumerate(cfg["phases"]):
    if ph["name"].startswith("ascend"):
        ascend_n += 1
        target_xy = ph.get("target_xy")
        target_z = ph.get("target_z_min_m", 0.0) + delta_z
        if target_xy is not None:
            color = (1.0, 0.85, 0.0) if ascend_n == 1 else (0.0, 0.9, 0.4)
            add_sphere(f"/World/Markers/Ascend_{ascend_n}", target_xy, target_z, 0.22, color)
            print(f"  {ph['name']}: target=({target_xy[0]:.2f}, {target_xy[1]:.2f}, {target_z:.2f})", flush=True)
    elif ph["name"].startswith("turn"):
        prev = None
        for p in cfg["phases"][:i]:
            if "target_xy" in p:
                prev = (p["target_xy"], p.get("target_z_min_m", 0.0) + delta_z)
        if prev is not None:
            add_sphere(f"/World/Markers/Turn_{i}", prev[0], prev[1] + 0.3, 0.28, (1.0, 0.0, 1.0))
            print(f"  {ph['name']}: pivot at {prev[0]} z={prev[1]+0.3:.2f}", flush=True)

world.step(render=False)
print("\nLayout loaded. Inspect visually.", flush=True)
print("If geometry is correctly placed but markers are off, edit fw_stair_eval_config.json", flush=True)
print("If geometry STILL appears buried, the reference composition may be re-shadowing.", flush=True)

if not args.headless:
    while app.is_running():
        world.step(render=True)

app.close()
