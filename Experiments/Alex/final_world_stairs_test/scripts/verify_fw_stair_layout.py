"""Visual sanity check for the FW stair eval setup.

Loads ONE SM_Staircase USD using the parent/child holder Xform pattern that
worked-around-the-USD-reference-composition-bug from the earlier side quest.
Places colored sphere markers at the spawn pose and each phase waypoint
defined in fw_stair_eval_config.json. NO policy is loaded — this is purely
to confirm the USD shows up in the right place, the waypoints land where
they should (on the bottom step / landing / top platform), and Spot's
spawn pose faces the correct direction.

Usage:
    python scripts/verify_fw_stair_layout.py --stair SM_Staircase_01.usd

Color key (sphere markers):
    cyan    spawn position       (Spot starts here)
    yellow  ascend waypoint      (target_xy of an ascend phase)
    magenta turn pivot           (where 180-degree turn happens)
    green   final target         (top of the stairs)

A blue arrow is drawn at spawn pointing in the spawn yaw direction (where
Spot will be facing on episode start).
"""
import argparse
import json
import math
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--stair", type=str, default="SM_Staircase_01.usd",
                    help="USD filename inside SubUSDs to verify.")
parser.add_argument("--headless", action="store_true", default=False)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402
app = SimulationApp({"headless": args.headless, "width": 1920, "height": 1080})

import numpy as np  # noqa: E402
from omni.isaac.core import World  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

# Load eval config
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(os.path.dirname(THIS_DIR), "data", "fw_stair_eval_config.json")
USD_ROOT = r"C:\Users\Gabriel Santiago\OneDrive\Desktop\Collected_Final_World\SubUSDs"

with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

if args.stair not in CONFIG:
    print(f"ERROR: '{args.stair}' not in config. Available: {[k for k in CONFIG if not k.startswith('_')]}", flush=True)
    app.close(); sys.exit(1)

cfg = CONFIG[args.stair]
print(f"\n=== Verifying: {args.stair} ({cfg['topology']}) ===", flush=True)
print(f"  scale={cfg['scale']}  lift_z={cfg['lift_z']}", flush=True)

# Build world
world = World(stage_units_in_meters=1.0)
stage = world.stage

# Lighting
distant = UsdLux.DistantLight.Define(stage, "/World/SunLight")
distant.GetIntensityAttr().Set(2500.0)
ambient = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
ambient.GetIntensityAttr().Set(800.0)

# Ground plane (large flat patch — Spot needs floor before stair)
ground = UsdGeom.Xform.Define(stage, "/World/Ground")
ground_mesh = UsdGeom.Cube.Define(stage, "/World/Ground/Plane")
ground_mesh.GetSizeAttr().Set(1.0)
ground_xform = UsdGeom.Xformable(ground.GetPrim())
ground_xform.ClearXformOpOrder()
op_t = ground_xform.AddTranslateOp()
op_s = ground_xform.AddScaleOp()
op_t.Set(Gf.Vec3d(0, 0, -0.05))   # ground top at z=0
op_s.Set(Gf.Vec3d(40.0, 40.0, 0.1))
UsdPhysics.CollisionAPI.Apply(ground_mesh.GetPrim())

# Stair USD via parent/child Xform pattern (the FW-side-quest fix that never
# got tested). Holder Xform is what we transform; the inner reference stays
# at native USD coords. Lift_z translates the holder so the USD's bottom-most
# surface lands at world z=0.
holder_path = "/World/Stair_holder"
inner_path  = f"{holder_path}/Stair"
holder_x = UsdGeom.Xform.Define(stage, holder_path)
inner_x  = UsdGeom.Xform.Define(stage, inner_path)
inner_x.GetPrim().GetReferences().AddReference(os.path.join(USD_ROOT, args.stair))

holder_xform = UsdGeom.Xformable(holder_x.GetPrim())
holder_xform.ClearXformOpOrder()
op_s = holder_xform.AddScaleOp()
op_t = holder_xform.AddTranslateOp()
op_s.Set(Gf.Vec3d(cfg["scale"], cfg["scale"], cfg["scale"]))
# After scale, the bottom-most z is approximately -lift_z (since the USD has
# a negative z_min in native coords that scales linearly). Translate UP by
# lift_z so the bottom sits at world z=0.
op_t.Set(Gf.Vec3d(0.0, 0.0, cfg["lift_z"]))


def add_sphere_marker(path, world_xy, world_z, radius, rgb):
    """Drop a small visual-only sphere at (xy, z) in world coords."""
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.GetRadiusAttr().Set(radius)
    sphere_xform = UsdGeom.Xformable(sphere.GetPrim())
    sphere_xform.ClearXformOpOrder()
    sphere_xform.AddTranslateOp().Set(Gf.Vec3d(world_xy[0], world_xy[1], world_z))
    color_attr = sphere.GetDisplayColorAttr()
    color_attr.Set([Gf.Vec3f(*rgb)])
    return sphere


def add_arrow(path, base_xy, base_z, yaw_rad, length=0.8):
    """Draw a thin cylinder pointing in `yaw_rad` direction (CCW from +X)."""
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.GetHeightAttr().Set(length)
    cyl.GetRadiusAttr().Set(0.04)
    # default cylinder is along Z; rotate so it points along +X then by yaw,
    # plus translate to base position
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.ClearXformOpOrder()
    op_t = xf.AddTranslateOp()
    op_r = xf.AddRotateXYZOp()
    # Lay cylinder along +X by rotating Y by 90 deg, then yaw around Z
    yaw_deg = math.degrees(yaw_rad)
    op_r.Set(Gf.Vec3f(0.0, 90.0, yaw_deg))
    # Translate so the cylinder center is `length/2` ahead of base
    cx = base_xy[0] + math.cos(yaw_rad) * length / 2.0
    cy = base_xy[1] + math.sin(yaw_rad) * length / 2.0
    op_t.Set(Gf.Vec3d(cx, cy, base_z + 0.1))
    cyl.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.4, 1.0)])  # blue
    return cyl


# Spawn marker (cyan)
spawn = cfg["spawn"]
add_sphere_marker(
    "/World/Markers/Spawn",
    spawn["world_xy"],
    spawn["world_z"],
    radius=0.18,
    rgb=(0.0, 1.0, 1.0),
)
yaw_rad = math.radians(spawn["yaw_deg"])
add_arrow("/World/Markers/Spawn_facing", spawn["world_xy"], spawn["world_z"], yaw_rad)
print(f"  spawn: world_xy={spawn['world_xy']}  z={spawn['world_z']}  yaw={spawn['yaw_deg']} deg", flush=True)

# Phase markers
ascend_count = 0
for i, phase in enumerate(cfg["phases"]):
    pname = phase["name"]
    if pname == "stabilize":
        continue
    if pname.startswith("ascend"):
        ascend_count += 1
        target_xy = phase.get("target_xy")
        target_z = phase.get("target_z_min_m", 0.0)
        if target_xy is not None:
            color = (1.0, 0.85, 0.0) if ascend_count == 1 else (0.0, 0.9, 0.4)  # yellow then green
            add_sphere_marker(
                f"/World/Markers/Ascend_{ascend_count}",
                target_xy,
                target_z,
                radius=0.22,
                rgb=color,
            )
            print(f"  phase '{pname}': target_xy={target_xy}  target_z>={target_z}m  vx={phase.get('vx')}  max={phase.get('max_time_s')}s", flush=True)
    elif pname.startswith("turn"):
        # Use last ascend target as turn pivot
        prev_target = None
        for p in cfg["phases"][:i]:
            if "target_xy" in p:
                prev_target = p["target_xy"]
                prev_z = p.get("target_z_min_m", 0.0)
        if prev_target is not None:
            add_sphere_marker(
                f"/World/Markers/Turn_{i}",
                prev_target,
                prev_z + 0.3,
                radius=0.28,
                rgb=(1.0, 0.0, 1.0),  # magenta
            )
            print(f"  phase '{pname}': pivot at {prev_target} z={prev_z}  yaw_target={phase.get('yaw_target_deg')} deg", flush=True)

# Reset and step
world.reset()
print("\nLayout loaded. Color key:", flush=True)
print("  CYAN sphere   = spawn position", flush=True)
print("  BLUE arrow    = spawn facing (yaw direction)", flush=True)
print("  YELLOW sphere = first ascend target", flush=True)
print("  MAGENTA       = turn pivot (switchback only)", flush=True)
print("  GREEN sphere  = final ascend target / top", flush=True)
print("\nVerify visually:", flush=True)
print("  - The USD geometry shows up (not buried, not floating)", flush=True)
print("  - The CYAN spawn sphere sits ~1.5m in front of the bottom step", flush=True)
print("  - The BLUE arrow points TOWARD the stair", flush=True)
print("  - YELLOW lands on the landing platform (or top, for straight stairs)", flush=True)
print("  - GREEN lands on the top platform", flush=True)
print("\nClose the window when done. Use --headless to script-test only.", flush=True)

if not args.headless:
    while app.is_running():
        world.step(render=True)

app.close()
