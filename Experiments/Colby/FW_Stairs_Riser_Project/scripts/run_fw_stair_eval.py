"""FW USD stair eval — drives Spot through the JSON-configured phase
sequence on each SM_Staircase USD, reports pass/fail per stair.

Uses:
  - measure-then-correct lift (auto-place USD bottom at z=0)
  - per-USD spawn pose + waypoints from data/fw_stair_eval_config.json
  - state machine: stabilize -> ascend -> [turn] -> [ascend] -> done

Usage:
    python scripts/run_fw_stair_eval.py \\
        --checkpoint <path-to-policy.pt> \\
        --action_scale 0.3 \\
        --stairs all \\
        [--rendered]
"""
import argparse
import csv
import json
import math
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Final Capstone Policy ckpt (e.g., parkour_phase9_18500.pt)")
parser.add_argument("--action_scale", type=float, default=0.3)
parser.add_argument("--stairs", type=str, default="all",
                    help="all, or comma-separated USD names without .usd")
parser.add_argument("--rendered", action="store_true", default=False)
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--mason", action="store_true", default=True)
parser.add_argument("--direction", type=str, default="ascend",
                    choices=["ascend", "descend", "both"],
                    help="ascend = walk up; descend = spawn at top, walk down; both = run ascend then descend.")
args = parser.parse_args()

headless = args.headless and not args.rendered

from isaacsim import SimulationApp  # noqa: E402
app = SimulationApp({"headless": headless, "width": 1920, "height": 1080})

import numpy as np  # noqa: E402
from omni.isaac.core import World  # noqa: E402
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_ROOT = os.path.dirname(THIS_DIR)
ALEX_ROOT = os.path.dirname(TEST_ROOT)
CONFIG_PATH = os.path.join(TEST_ROOT, "data", "fw_stair_eval_config.json")
USD_ROOT = r"C:\Users\Gabriel Santiago\OneDrive\Desktop\Collected_Final_World\SubUSDs"
RESULTS_DIR = os.path.join(TEST_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Add 4_env_test src to path for rough policy
sys.path.insert(0, os.path.join(ALEX_ROOT, "4_env_test", "src"))
from spot_rough_terrain_policy import SpotRoughTerrainPolicy  # noqa: E402

with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

# Resolve stair list
if args.stairs == "all":
    stair_names = [k for k in CONFIG.keys() if not k.startswith("_")]
else:
    stair_names = [s if s.endswith(".usd") else s + ".usd" for s in args.stairs.split(",")]

print(f"\n=== FW Stair Eval ===", flush=True)
print(f"  ckpt:        {os.path.basename(args.checkpoint)}", flush=True)
print(f"  action_scale: {args.action_scale}", flush=True)
print(f"  stairs:      {stair_names}", flush=True)

PHYSICS_DT = 1.0 / 50.0  # 50 Hz control


def yaw_to_quat(yaw_rad):
    """Quaternion (w, x, y, z) for rotation about Z by yaw_rad."""
    half = yaw_rad / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def setup_stair_in_stage(stage, world, usd_filename, cfg):
    """Reference USD with measure-then-correct lift. Returns (lift_z, holder_path)."""
    # Clean up any prior holder (collect paths first, remove after — can't
    # modify the stage during traversal or the iterator expires).
    to_remove = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path == "/World/Stair_holder" or path.startswith("/World/Stair_holder/"):
            to_remove.append(prim.GetPath())
    for p in to_remove:
        stage.RemovePrim(p)

    holder_path = "/World/Stair_holder"
    inner_path = f"{holder_path}/Stair"
    holder = UsdGeom.Xform.Define(stage, holder_path)
    inner = UsdGeom.Xform.Define(stage, inner_path)
    inner.GetPrim().GetReferences().AddReference(os.path.join(USD_ROOT, usd_filename))

    hxform = UsdGeom.Xformable(holder.GetPrim())
    hxform.ClearXformOpOrder()
    op_t = hxform.AddTranslateOp()
    op_s = hxform.AddScaleOp()
    op_t.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    op_s.Set(Gf.Vec3d(cfg["scale"], cfg["scale"], cfg["scale"]))

    world.step(render=False)
    bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_], useExtentsHint=False)
    bbox = bbox_cache.ComputeWorldBound(holder.GetPrim()).ComputeAlignedRange()
    observed_min_z = bbox.GetMin()[2]
    correct_lift = -observed_min_z + 0.01
    op_t.Set(Gf.Vec3d(0.0, 0.0, correct_lift))
    world.step(render=False)
    bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_], useExtentsHint=False)
    bbox = bbox_cache.ComputeWorldBound(holder.GetPrim()).ComputeAlignedRange()
    return correct_lift, holder_path, bbox.GetMin()[2], bbox.GetMax()[2]


def run_eval_for_stair(world, stage, usd_filename, cfg, robot_policy, flat_policy, direction="ascend"):
    """Run the configured phase sequence on one USD, return result dict."""
    print(f"\n--- {usd_filename} ({cfg['topology']}) [{direction}] ---", flush=True)

    # Place USD
    lift_z, holder_path, bbox_min_z, bbox_max_z = setup_stair_in_stage(stage, world, usd_filename, cfg)
    print(f"  USD placed: bbox z = [{bbox_min_z:.2f}, {bbox_max_z:.2f}] (lift={lift_z:.2f})", flush=True)

    # Compute world-z offset between config (assumed lift_z) and actual lift
    delta_z = lift_z - cfg["lift_z"]

    # Spawn pose + phase list — pick by direction
    if direction == "descend":
        if "spawn_descend" not in cfg or "phases_descend" not in cfg:
            print(f"  SKIP: no descend config for {usd_filename}", flush=True)
            return None
        spawn = cfg["spawn_descend"]
        phases = cfg["phases_descend"]
    else:
        spawn = cfg["spawn"]
        phases = cfg["phases"]
    spawn_pos = np.array([spawn["world_xy"][0], spawn["world_xy"][1], spawn["world_z"]],
                         dtype=np.float64)
    yaw_rad = math.radians(spawn["yaw_deg"])
    spawn_quat = yaw_to_quat(yaw_rad)
    print(f"  spawn @ {spawn_pos} yaw={spawn['yaw_deg']} deg", flush=True)

    # Reset robot pose
    robot_policy.robot.set_world_pose(position=spawn_pos, orientation=spawn_quat)
    robot_policy.robot.set_linear_velocity(np.zeros(3))
    robot_policy.robot.set_angular_velocity(np.zeros(3))
    world.step(render=not headless)

    # Per-stair phase loop
    phase_results = []
    fell = False
    for ph in phases:
        pname = ph["name"]
        if fell:
            phase_results.append({"name": pname, "status": "skipped"})
            continue

        # Stabilize
        if pname == "stabilize":
            steps = ph.get("duration_steps", 400)
            cmd = np.array([ph.get("vx", 0.0), ph.get("vy", 0.0), ph.get("wz", 0.0)],
                           dtype=np.float64)
            for _ in range(steps):
                robot_policy.forward(PHYSICS_DT, cmd)
                world.step(render=not headless)
            phase_results.append({"name": pname, "status": "ok"})
            continue

        # Ascend / Descend / generic forward
        if pname.startswith("ascend") or pname.startswith("descend"):
            cmd = np.array([ph.get("vx", 0.8), 0.0, 0.0], dtype=np.float64)
            target_xy = ph.get("target_xy")
            arrive_radius = ph.get("arrive_radius_m", 0.6)
            target_z_min = ph.get("target_z_min_m", 0.0) + delta_z
            target_z_max = ph.get("target_z_max_m", 1e6) + delta_z
            fall_z = ph.get("fall_z_m", -0.3) + delta_z
            max_steps = int(ph.get("max_time_s", 30) * 50)

            arrived = False
            timed_out = True
            for step in range(max_steps):
                robot_policy.forward(PHYSICS_DT, cmd)
                world.step(render=not headless)
                pos, _ = robot_policy.robot.get_world_pose()
                pos_np = np.array(pos)
                if pos_np[2] < fall_z:
                    fell = True
                    timed_out = False
                    print(f"    {pname}: FELL at step {step}, z={pos_np[2]:.2f}", flush=True)
                    break
                # Arrival: z within band AND xy near target
                if target_xy is not None and target_z_min <= pos_np[2] <= target_z_max:
                    dx = pos_np[0] - target_xy[0]
                    dy = pos_np[1] - target_xy[1]
                    if (dx * dx + dy * dy) ** 0.5 < arrive_radius:
                        arrived = True
                        timed_out = False
                        print(f"    {pname}: ARRIVED at step {step}, pos={pos_np}", flush=True)
                        break

            if fell:
                status = "fell"
            elif arrived:
                status = "arrived"
            elif timed_out:
                status = "timeout"
                pos, _ = robot_policy.robot.get_world_pose()
                print(f"    {pname}: TIMEOUT at pos={np.array(pos)}", flush=True)
            phase_results.append({"name": pname, "status": status})
            continue

        # Turn
        if pname.startswith("turn"):
            wz = ph.get("wz", 1.5)
            yaw_target = math.radians(ph.get("yaw_target_deg", 0))
            yaw_tol = math.radians(ph.get("yaw_tol_deg", 15))
            max_steps = int(ph.get("max_time_s", 5) * 50)
            cmd = np.array([ph.get("vx", 0.1), 0.0, wz], dtype=np.float64)
            turned = False
            for step in range(max_steps):
                robot_policy.forward(PHYSICS_DT, cmd)
                world.step(render=not headless)
                _, q = robot_policy.robot.get_world_pose()
                # quat w x y z; yaw = atan2(2(wz+xy), 1-2(yy+zz))
                qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
                yaw_now = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                yaw_err = math.atan2(math.sin(yaw_target - yaw_now), math.cos(yaw_target - yaw_now))
                if abs(yaw_err) < yaw_tol:
                    turned = True
                    print(f"    {pname}: turned to yaw={math.degrees(yaw_now):.1f} deg at step {step}", flush=True)
                    break
            phase_results.append({
                "name": pname,
                "status": "turned" if turned else "timeout",
            })
            continue

    # Final pose
    pos, _ = robot_policy.robot.get_world_pose()
    pos_np = np.array(pos)
    success = (not fell) and all(
        p["status"] in ("ok", "arrived", "turned") for p in phase_results
    )
    print(f"  RESULT: {'PASS' if success else 'FAIL'}  fell={fell}  final_z={pos_np[2]:.2f}", flush=True)
    return {
        "stair": usd_filename,
        "topology": cfg["topology"],
        "phases": phase_results,
        "final_pos": [float(v) for v in pos_np],
        "fell": fell,
        "pass": success,
    }


# Build world ONCE
world = World(stage_units_in_meters=1.0)
stage = world.stage

# Lighting
UsdLux.DistantLight.Define(stage, "/World/SunLight").GetIntensityAttr().Set(2500.0)
UsdLux.DomeLight.Define(stage, "/World/DomeLight").GetIntensityAttr().Set(800.0)

# Ground
ground_x = UsdGeom.Xform.Define(stage, "/World/Ground")
ground_mesh = UsdGeom.Cube.Define(stage, "/World/Ground/Plane")
ground_mesh.GetSizeAttr().Set(1.0)
gxform = UsdGeom.Xformable(ground_x.GetPrim())
gxform.ClearXformOpOrder()
gxform.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
gxform.AddScaleOp().Set(Gf.Vec3d(40.0, 40.0, 0.1))
UsdPhysics.CollisionAPI.Apply(ground_mesh.GetPrim())

# Spawn Spot ONCE — we'll reset its pose between stairs
SPAWN_INIT = np.array([0.0, 0.0, 0.55], dtype=np.float64)
flat_policy = SpotFlatTerrainPolicy(
    prim_path="/World/Robot",
    name="Robot",
    position=SPAWN_INIT,
)
world.reset()
world.step(render=not headless)
flat_policy.initialize()
flat_policy.post_reset()

robot_policy = SpotRoughTerrainPolicy(
    flat_policy=flat_policy,
    checkpoint_path=os.path.abspath(args.checkpoint),
    arl_baseline=args.mason,
    action_scale=args.action_scale,
)
robot_policy.initialize()
robot_policy.apply_gains()
robot_policy._decimation = 1

# Run each stair
results = []
directions = ["ascend", "descend"] if args.direction == "both" else [args.direction]
for direction in directions:
    for stair in stair_names:
        if stair not in CONFIG:
            print(f"SKIP: {stair} not in config", flush=True)
            continue
        res = run_eval_for_stair(world, stage, stair, CONFIG[stair], robot_policy, flat_policy, direction=direction)
        if res is not None:
            res["direction"] = direction
            results.append(res)

# Write CSV
ckpt_tag = os.path.basename(args.checkpoint).replace(".pt", "")
csv_path = os.path.join(RESULTS_DIR, f"fw_stair_eval_{ckpt_tag}_{args.direction}.csv")
with open(csv_path, "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["direction", "stair", "topology", "phases", "fell", "pass", "final_x", "final_y", "final_z"])
    for r in results:
        w.writerow([
            r.get("direction", "ascend"),
            r["stair"], r["topology"],
            ";".join(f"{p['name']}={p['status']}" for p in r["phases"]),
            r["fell"], r["pass"],
            f"{r['final_pos'][0]:.3f}", f"{r['final_pos'][1]:.3f}", f"{r['final_pos'][2]:.3f}",
        ])
print(f"\nWrote {csv_path}", flush=True)
print("=== Summary ===", flush=True)
for r in results:
    print(f"  [{r.get('direction', 'ascend'):<7}] {r['stair']:<28}  pass={r['pass']}  fell={r['fell']}", flush=True)

os._exit(0)
