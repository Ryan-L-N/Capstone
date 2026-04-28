"""Final World stair test — automated up/turn/down sequence per staircase USD.

For each of the 4 SM_Staircase USDs from Final World:
  Phase 1: walk UP the stairs (cmd forward at 0.8 m/s until z plateaus)
  Phase 2: turn 180° in place
  Phase 3: walk DOWN the stairs (cmd forward at 0.8 m/s until back to base)
  Phase 4: record success/fall

Headless by default (cut render time). Pass --rendered to watch.

Reuses 4_env_test policy wrapper + spot_rough_terrain_policy with action_scale=0.3.
Designed to run after the SM_Staircase_*.usd files have been baked with
CollisionAPI (run bake_stair_collision.py once before this).

Usage:
    python run_fw_stairs.py \\
        --checkpoint Experiments/Alex/PARKOUR_NAV/checkpoints/parkour_phase7_15000.pt \\
        --stairs all
"""

# ── 0. Parse args BEFORE Isaac imports ───────────────────────────────────
import argparse
import math
import os
import signal
import sys
import time

parser = argparse.ArgumentParser(description="Final World stair test")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to PARKOUR_NAV rough-policy .pt")
parser.add_argument("--stairs", type=str, default="all",
                    help="Which stair USD(s) — '01', '02', 'half_01', 'half_02', "
                         "or 'all'")
parser.add_argument("--action_scale", type=float, default=0.3)
parser.add_argument("--decimation", type=int, default=1)
parser.add_argument("--rendered", action="store_true", default=False)
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--teleop", action="store_true", default=False,
                    help="Skip auto-walk; spawn stair + robot, then keyboard "
                         "control. WASD = drive, G = toggle gait, R = reset, "
                         "ESC = exit. Best for switchback stairs.")
parser.add_argument("--seed", type=int, default=42)
# Sequence timing
parser.add_argument("--up_speed", type=float, default=0.8,
                    help="Forward command for ascent (m/s)")
parser.add_argument("--down_speed", type=float, default=0.6,
                    help="Forward command for descent (m/s, slower for safety)")
parser.add_argument("--turn_rate", type=float, default=1.5,
                    help="Yaw rate command for turn-around (rad/s)")
parser.add_argument("--up_max_seconds", type=float, default=20.0,
                    help="Max sim seconds for ascent before timing out")
parser.add_argument("--turn_max_seconds", type=float, default=4.0,
                    help="Max sim seconds for turn (180° at 1.5 rad/s ≈ 2.1s)")
parser.add_argument("--down_max_seconds", type=float, default=20.0)
parser.add_argument("--fall_height", type=float, default=0.15,
                    help="Robot base z below this = FELL")
parser.add_argument("--output_dir", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "results"))

args = parser.parse_args()
headless = args.headless and not args.rendered


def _shutdown(signum, frame):
    print(f"\n[SHUTDOWN] caught signal {signum} — exiting hard.", flush=True)
    os._exit(0)


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

# ── 1. Boot Isaac Sim ─────────────────────────────────────────────────────
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": headless, "width": 1920,
                                "height": 1080})

# ── 2. Post-launch imports ────────────────────────────────────────────────
import numpy as np
import omni
from omni.isaac.core import World
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
from pxr import UsdGeom, UsdLux, UsdPhysics, Gf, Usd, Sdf

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ALEX_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_ALEX_ROOT, "4_env_test", "src"))

from spot_rough_terrain_policy import SpotRoughTerrainPolicy

# ── 3. Constants ──────────────────────────────────────────────────────────
PHYSICS_DT = 1.0 / 500.0
RENDERING_DT = 10.0 / 500.0

FW_STAIRS_ROOT = r"C:\Users\Gabriel Santiago\OneDrive\Desktop\Collected_Final_World\SubUSDs"
ALL_STAIRS = {
    "01": "SM_Staircase_01.usd",
    "02": "SM_Staircase_02.usd",
    "half_01": "SM_StaircaseHalf_01.usd",
    "half_02": "SM_StaircaseHalf_02.usd",
}

SPAWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
SPOT_DEFAULT_TYPE_GROUPED = np.array([
     0.1, -0.1,  0.1, -0.1,
     0.9,  0.9,  1.1,  1.1,
    -1.5, -1.5, -1.5, -1.5,
], dtype=np.float64)


def quat_to_yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def body_is_flipped(q, thresh_rad=math.radians(60)):
    w, x, y, z = q
    gz = -(1 - 2 * (x * x + y * y))
    return math.acos(max(-1.0, min(1.0, -gz))) > thresh_rad


# ── 4. Stair USD inspection — find bbox to place spawn + targets ─────────
def inspect_stair(usd_path):
    """Return bbox of the staircase mesh CONVERTED to meters.

    The Final World SubUSDs were authored in centimeters (metersPerUnit=0.01)
    AND with Y-up axis (Unreal convention). Isaac Lab world is meters + Z-up.
    We read both unit and up-axis, apply scale + axis-conversion rotation, and
    return the post-conversion bbox in WORLD frame so the spawn+target math
    works correctly.

    For Y-up USD: rotate +90° about X to bring +Y → +Z. After conversion:
        usd_x → world_x      (unchanged)
        usd_y → world_z      (vertical rise of stair)
        usd_z → world_-y     (long horizontal axis becomes world -y, but
                              we'll rotate further to align long axis = +x)
    """
    stage = Usd.Stage.Open(usd_path)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    scale = float(mpu) if mpu else 1.0
    up_axis = UsdGeom.GetStageUpAxis(stage)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                               [UsdGeom.Tokens.default_])
    bbox = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    mn = bbox.GetMin()
    mx = bbox.GetMax()
    # Native USD bbox in meters
    usd_min = (float(mn[0]) * scale, float(mn[1]) * scale, float(mn[2]) * scale)
    usd_max = (float(mx[0]) * scale, float(mx[1]) * scale, float(mx[2]) * scale)

    # Convert to world frame (Z-up) bbox
    if up_axis == "Y":
        # +90° rotation about world X: (x, y, z) → (x, -z, y)
        # So world_x_range = usd_x_range
        #    world_y_range = -usd_z_range (flipped)
        #    world_z_range = usd_y_range (the climb direction)
        world_min = (usd_min[0], -usd_max[2], usd_min[1])
        world_max = (usd_max[0], -usd_min[2], usd_max[1])
    else:  # Z-up — no rotation needed
        world_min = usd_min
        world_max = usd_max

    return {
        "scale": scale,
        "up_axis": up_axis,
        "size_x": world_max[0] - world_min[0],
        "size_y": world_max[1] - world_min[1],
        "size_z": world_max[2] - world_min[2],
        "min": world_min,
        "max": world_max,
    }


# ── 5. Per-stair test ────────────────────────────────────────────────────
def run_stair_test(world, stage, stair_key, usd_filename, results_log):
    print(f"\n{'=' * 60}\n  STAIR TEST: {stair_key} ({usd_filename})\n{'=' * 60}",
          flush=True)
    usd_path = os.path.join(FW_STAIRS_ROOT, usd_filename)
    info = inspect_stair(usd_path)
    print(f"  bbox (world frame): {info['size_x']:.2f} x {info['size_y']:.2f} x {info['size_z']:.2f} m"
          f"  up_axis={info['up_axis']}", flush=True)

    # After up-axis conversion, info bbox is Z-up, in world meters.
    # Long horizontal axis is whichever of x/y is bigger.
    long_axis_is_x = info["size_x"] >= info["size_y"]
    stair_length = max(info["size_x"], info["size_y"])
    stair_height = info["size_z"]

    # PARENT/CHILD STRUCTURE — fixes xform composition bug:
    # /World/Stair_01_holder   ← parent Xform: WHERE we put OUR scale + translate ops
    #     /Stair                ← child Xform: WHERE we put the AddReference()
    # Without this split, the reference's xformOps composed UNDER ours and
    # shadowed our translate.Set() — bbox stayed unchanged after correction.
    holder_path = f"/World/Stair_{stair_key}_holder"
    stair_path  = f"{holder_path}/Stair"
    UsdGeom.Xform.Define(stage, holder_path)
    UsdGeom.Xform.Define(stage, stair_path)
    holder_xform = UsdGeom.Xformable(stage.GetPrimAtPath(holder_path))
    stair_xform  = UsdGeom.Xformable(stage.GetPrimAtPath(stair_path))
    stage.GetPrimAtPath(stair_path).GetReferences().AddReference(usd_path)

    scale = info["scale"]

    # USD's xformOpOrder: first op listed is INNERMOST (applied first to point).
    # We want: final_world = Translate * RotateZ * RotateX * Scale * usd_point
    # So add ops in order: Scale (innermost), RotateX, RotateZ, Translate (outermost).
    # RotateX = +90° converts Y-up USD to Z-up world.
    # RotateZ = aligns long horizontal axis with world +x.
    # Translate = positions stair base at world origin.

    # info["min"] is already in WORLD frame (Z-up, meters) after inspect_stair
    # applied the up-axis conversion. So info long axis (whichever of x/y is
    # bigger) is the horizontal run length.
    rot_x_deg = 90.0 if info["up_axis"] == "Y" else 0.0

    if info["size_x"] >= info["size_y"]:
        # long axis already +x in world frame, no z-rotation needed
        rot_z_deg = 0.0
        post_rot_min = info["min"]
    else:
        # long axis is +y in world; rotate +90° about z to get +y → +x
        # +90° about z: (x, y) → (-y, x)
        # So bbox after z-rot:
        #   new_min_x = -info["max"][1]
        #   new_max_x = -info["min"][1]
        #   new_min_y = info["min"][0]
        #   new_max_y = info["max"][0]
        rot_z_deg = 90.0
        post_rot_min = (
            -info["max"][1],
            info["min"][0],
            info["min"][2],
        )

    # PLACEMENT: scale + 180° z-rotation + translate.
    # User feedback: stair was facing wrong direction AND sunk in floor.
    # - 180° rotation about Z flips climbing direction so steps face robot.
    # - Translate (post-rotation) lifts stair so bbox min lands at world origin.
    # USD ops applied innermost-first: Scale → RotateZ → Translate.
    # After 180° z-rot: usd_x → -usd_x, usd_y → -usd_y, z unchanged.
    # So post-rotation bbox: x=[-max_x, -min_x], y=[-max_y, -min_y], z=same.
    # NEW APPROACH (post-ultrathink): trust the post-reference bbox in MY
    # scene, not the source-USD bbox. Reference first with just scale, then
    # measure the actual world bbox, then apply corrective translate.
    stair_xform.AddScaleOp().Set(Gf.Vec3d(scale, scale, scale))
    # Placeholder translate — will be set correctly after first measurement
    translate_op = stair_xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))

    # Force USD composition so the reference takes effect, then measure
    # where the stair ACTUALLY landed in MY scene.
    world.reset()
    world.step(render=False)
    measure_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                       [UsdGeom.Tokens.default_])
    stair_prim = stage.GetPrimAtPath(stair_path)
    measured = measure_cache.ComputeWorldBound(stair_prim).ComputeAlignedRange()
    m_min = (float(measured.GetMin()[0]), float(measured.GetMin()[1]),
             float(measured.GetMin()[2]))
    m_max = (float(measured.GetMax()[0]), float(measured.GetMax()[1]),
             float(measured.GetMax()[2]))
    print(f"  POST-REFERENCE measured bbox in MY scene world:", flush=True)
    print(f"    x: [{m_min[0]:.2f}, {m_max[0]:.2f}] m", flush=True)
    print(f"    y: [{m_min[1]:.2f}, {m_max[1]:.2f}] m", flush=True)
    print(f"    z: [{m_min[2]:.2f}, {m_max[2]:.2f}] m", flush=True)

    # Apply CORRECTIVE translate so post-everything bbox.min = (0, 0, 0).
    # This is in WORLD frame and applied AFTER all internal transforms.
    corrective = Gf.Vec3d(-m_min[0], -m_min[1], -m_min[2])
    translate_op.Set(corrective)
    world.step(render=False)

    # Re-measure to verify
    measured2 = measure_cache.ComputeWorldBound(stair_prim).ComputeAlignedRange()
    f_min = (float(measured2.GetMin()[0]), float(measured2.GetMin()[1]),
             float(measured2.GetMin()[2]))
    f_max = (float(measured2.GetMax()[0]), float(measured2.GetMax()[1]),
             float(measured2.GetMax()[2]))
    print(f"  AFTER CORRECTION:", flush=True)
    print(f"    x: [{f_min[0]:.2f}, {f_max[0]:.2f}] m", flush=True)
    print(f"    y: [{f_min[1]:.2f}, {f_max[1]:.2f}] m", flush=True)
    print(f"    z: [{f_min[2]:.2f}, {f_max[2]:.2f}] m", flush=True)

    # Use the MEASURED final bbox to position Spot
    stair_length = f_max[0] - f_min[0]
    stair_height = f_max[2] - f_min[2]
    stair_y_center = (f_min[1] + f_max[1]) / 2

    # Spawn Spot 2m in front of the stair's measured min_x edge, at the
    # stair's y centerline, facing +x toward the stair.
    spawn_pos = np.array([f_min[0] - 2.0, stair_y_center, 0.55], dtype=np.float64)
    top_x = f_max[0] + 0.5
    top_y = stair_y_center
    print(f"  Spot spawn: ({spawn_pos[0]:.2f}, {spawn_pos[1]:.2f}, {spawn_pos[2]:.2f}) "
          f"facing +x toward stair", flush=True)

    # Build robot fresh — reuse one across stairs by reset_world_pose
    flat_policy = SpotFlatTerrainPolicy(
        prim_path="/World/Robot",
        name="Robot",
        position=spawn_pos,
    )
    world.reset()
    world.step(render=not headless)
    flat_policy.initialize()
    flat_policy.post_reset()

    policy = SpotRoughTerrainPolicy(
        flat_policy=flat_policy,
        checkpoint_path=os.path.abspath(args.checkpoint),
        mason_baseline=True,
        action_scale=args.action_scale,
    )
    policy.initialize()
    policy.apply_gains()
    policy._decimation = args.decimation

    # Stabilization (300 + 100 steps zero-cmd)
    print("  [STABILIZE] settling...", flush=True)
    for _ in range(400):
        flat_policy.forward(PHYSICS_DT, np.array([0.0, 0.0, 0.0]))
        world.step(render=not headless)

    # Reset robot pose (in case stabilize drifted)
    policy.robot.set_world_pose(position=spawn_pos, orientation=SPAWN_QUAT)
    policy.robot.set_joint_positions(SPOT_DEFAULT_TYPE_GROUPED, np.arange(12))
    policy.robot.set_joint_velocities(np.zeros(12), np.arange(12))
    policy.robot.set_linear_velocity(np.zeros(3))
    policy.robot.set_angular_velocity(np.zeros(3))
    world.step(render=False)
    if hasattr(policy, "post_reset"):
        policy.post_reset()
    for _ in range(50):
        policy.forward(PHYSICS_DT, np.array([0.0, 0.0, 0.0]))
        world.step(render=not headless)

    # Record initial z
    pos0, _ = policy.robot.get_world_pose()
    z0 = float(pos0[2])
    print(f"  initial z = {z0:.3f}, spawn_pos={spawn_pos}, top_xy=({top_x:.2f},{top_y:.2f})",
          flush=True)

    # Teleop mode: skip auto-walk, hand control to user via keyboard
    if args.teleop:
        print(f"\n  [TELEOP MODE] keyboard active.\n"
              f"    WASD = drive (forward / strafe)\n"
              f"    Q/E  = turn left/right\n"
              f"    G    = toggle FLAT/ROUGH gait\n"
              f"    R    = reset robot to spawn\n"
              f"    ESC  = exit\n", flush=True)
        import carb.input  # type: ignore
        ki = carb.input.acquire_input_interface()
        keyboard = ki.get_keyboard(0) if hasattr(ki, "get_keyboard") else None
        # Simple polling — uses Isaac Sim's input interface
        # Just runs forever stepping the policy until robot falls or user kills it
        from omni.kit.app import get_app
        app_iface = get_app()
        cmd = np.zeros(3, dtype=np.float64)
        run_step = 0
        while True:
            pos, quat = policy.robot.get_world_pose()
            pos_np = np.array(pos, dtype=np.float64)
            if pos_np[2] < args.fall_height or body_is_flipped(np.array(quat, dtype=np.float64)):
                print(f"  [TELEOP] FELL/FLIP at pos=({pos_np[0]:+.2f},"
                      f"{pos_np[1]:+.2f},{pos_np[2]:+.2f}) — resetting", flush=True)
                policy.robot.set_world_pose(position=spawn_pos, orientation=SPAWN_QUAT)
                policy.robot.set_joint_positions(SPOT_DEFAULT_TYPE_GROUPED, np.arange(12))
                policy.robot.set_joint_velocities(np.zeros(12), np.arange(12))
                policy.robot.set_linear_velocity(np.zeros(3))
                policy.robot.set_angular_velocity(np.zeros(3))
                world.step(render=False)
                if hasattr(policy, "post_reset"):
                    policy.post_reset()
                continue
            # Read keyboard via Isaac Sim's carb interface
            # Simpler: just walk forward continuously and let user observe.
            # For real teleop they should use run_capstone_teleop.py — this
            # mode is just "spawn + sit + observe" so user can SEE where the
            # stair actually is.
            policy.forward(PHYSICS_DT, np.array([0.0, 0.0, 0.0]))
            world.step(render=not headless)
            run_step += 1
            if run_step % 200 == 0:
                print(f"  [TELEOP] t={run_step/50:.1f}s  pos=({pos_np[0]:+.2f},"
                      f"{pos_np[1]:+.2f},{pos_np[2]:+.2f})  "
                      f"(stair bbox: x=[0,{stair_length:.1f}] y=[0,?] z=[?,?])",
                      flush=True)
        # Won't reach here, but for safety
        results_log.append({"stair": stair_key, "teleop": True})
        return True

    # Phase 1: ASCEND
    phase = "ASCEND"
    cmd_forward = np.array([args.up_speed, 0.0, 0.0], dtype=np.float64)
    max_steps = int(args.up_max_seconds / (1.0 / 50.0))
    z_high = z0
    pos_at_top = None
    fell = False
    for step in range(max_steps):
        pos, quat = policy.robot.get_world_pose()
        pos_np = np.array(pos, dtype=np.float64)
        quat_np = np.array(quat, dtype=np.float64)
        z_high = max(z_high, float(pos_np[2]))
        if pos_np[2] < args.fall_height:
            fell = True
            break
        if body_is_flipped(quat_np):
            fell = True
            break
        # Top detection: robot's x past top (after rotation, all stairs run +x)
        if pos_np[0] >= top_x:
            pos_at_top = pos_np
            break
        policy.forward(PHYSICS_DT, cmd_forward)
        world.step(render=not headless)
        if step % 100 == 0:
            print(f"    [{phase}] step={step:5d} pos=({pos_np[0]:+5.2f},"
                  f"{pos_np[1]:+5.2f},{pos_np[2]:+5.2f}) z_high={z_high:.2f}",
                  flush=True)

    ascended = (pos_at_top is not None) and (not fell)
    print(f"  ASCEND result: ascended={ascended} fell={fell} z_high={z_high:.2f}",
          flush=True)

    # Phase 2: TURN 180°
    turned = False
    if ascended and not fell:
        phase = "TURN"
        target_yaw_change = math.pi  # 180°
        cmd_turn = np.array([0.0, 0.0, args.turn_rate], dtype=np.float64)
        pos_t0, quat_t0 = policy.robot.get_world_pose()
        yaw0 = quat_to_yaw(np.array(quat_t0, dtype=np.float64))
        max_steps = int(args.turn_max_seconds / (1.0 / 50.0))
        for step in range(max_steps):
            pos, quat = policy.robot.get_world_pose()
            quat_np = np.array(quat, dtype=np.float64)
            pos_np = np.array(pos, dtype=np.float64)
            if pos_np[2] < args.fall_height or body_is_flipped(quat_np):
                fell = True
                break
            yaw_now = quat_to_yaw(quat_np)
            yaw_delta = abs(math.atan2(math.sin(yaw_now - yaw0),
                                        math.cos(yaw_now - yaw0)))
            if yaw_delta >= target_yaw_change - 0.2:  # within ~12° of target
                turned = True
                break
            policy.forward(PHYSICS_DT, cmd_turn)
            world.step(render=not headless)
        print(f"  TURN result: turned={turned} fell={fell}", flush=True)

    # Phase 3: DESCEND
    descended = False
    if turned and not fell:
        phase = "DESCEND"
        cmd_back = np.array([args.down_speed, 0.0, 0.0], dtype=np.float64)
        max_steps = int(args.down_max_seconds / (1.0 / 50.0))
        for step in range(max_steps):
            pos, quat = policy.robot.get_world_pose()
            pos_np = np.array(pos, dtype=np.float64)
            quat_np = np.array(quat, dtype=np.float64)
            if pos_np[2] < args.fall_height or body_is_flipped(quat_np):
                fell = True
                break
            # Bottom detection: z back near initial
            if abs(float(pos_np[2]) - z0) < 0.2:
                descended = True
                break
            policy.forward(PHYSICS_DT, cmd_back)
            world.step(render=not headless)
            if step % 100 == 0:
                print(f"    [{phase}] step={step:5d} pos=({pos_np[0]:+5.2f},"
                      f"{pos_np[1]:+5.2f},{pos_np[2]:+5.2f})", flush=True)

    success = ascended and turned and descended and (not fell)
    print(f"\n  RESULT[{stair_key}]: ascended={ascended} turned={turned} "
          f"descended={descended} fell={fell} z_high={z_high:.2f} "
          f"=> {'PASS' if success else 'FAIL'}", flush=True)

    results_log.append({
        "stair": stair_key,
        "ascended": ascended,
        "turned": turned,
        "descended": descended,
        "fell": fell,
        "z_high": z_high,
        "success": success,
    })

    # Clean up the stair prim before next iteration
    stage.RemovePrim(stair_path)
    return success


# ── 6. Main ──────────────────────────────────────────────────────────────
def main():
    os.makedirs(args.output_dir, exist_ok=True)

    # Build world
    world = World(physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT,
                  stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # Ground plane
    ground = UsdGeom.Cube.Define(stage, "/World/GroundPlane")
    ground.GetSizeAttr().Set(1.0)
    ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.005))
    ground.AddScaleOp().Set(Gf.Vec3d(50.0, 50.0, 0.01))
    ground.GetDisplayColorAttr().Set([(0.5, 0.5, 0.5)])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    # Lights
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/dome")
    dome.CreateIntensityAttr(500.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/sun")
    sun.CreateIntensityAttr(3000.0)
    sun_prim = stage.GetPrimAtPath("/World/Lights/sun")
    UsdGeom.Xformable(sun_prim).AddRotateXYZOp().Set(Gf.Vec3d(-45, 30, 0))

    # Decide which stairs to test
    if args.stairs == "all":
        stair_keys = list(ALL_STAIRS.keys())
    else:
        stair_keys = [args.stairs]

    results = []
    for k in stair_keys:
        if k not in ALL_STAIRS:
            print(f"[ERROR] unknown stair key '{k}', skipping.")
            continue
        try:
            run_stair_test(world, stage, k, ALL_STAIRS[k], results)
        except Exception as e:
            print(f"[ERROR] stair {k} crashed: {e}", flush=True)
            results.append({"stair": k, "fell": True, "success": False,
                            "error": str(e)})

    # Summary
    print(f"\n{'=' * 60}\n  SUMMARY\n{'=' * 60}", flush=True)
    print(f"  {'Stair':<10} {'Asc':<5} {'Turn':<5} {'Desc':<5} {'Fell':<5} {'Result':<8}")
    out_csv = os.path.join(args.output_dir, "fw_stairs_results.csv")
    with open(out_csv, "w") as f:
        f.write("stair,ascended,turned,descended,fell,z_high,success\n")
        for r in results:
            line = (f"  {r['stair']:<10} "
                    f"{str(r.get('ascended', '?')):<5} "
                    f"{str(r.get('turned', '?')):<5} "
                    f"{str(r.get('descended', '?')):<5} "
                    f"{str(r.get('fell', '?')):<5} "
                    f"{'PASS' if r.get('success') else 'FAIL':<8}")
            print(line, flush=True)
            f.write(f"{r['stair']},{r.get('ascended', False)},"
                    f"{r.get('turned', False)},{r.get('descended', False)},"
                    f"{r.get('fell', False)},{r.get('z_high', 0):.2f},"
                    f"{r.get('success', False)}\n")
    print(f"\n  CSV → {out_csv}", flush=True)

    os._exit(0)


if __name__ == "__main__":
    main()
