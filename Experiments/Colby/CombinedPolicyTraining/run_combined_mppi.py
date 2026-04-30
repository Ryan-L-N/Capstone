"""
run_combined_mppi.py
====================
MPPI navigation + frozen locomotion evaluated in the 4-quadrant FiveRing arena.

Navigation:  Cole's MPPINavigator  (Experiments/Cole/MPPI_FOLDER/mppi_navigator.py)
Locomotion:  parkour_phasefwplus_22100.pt  (frozen, action_scale=0.3, mason obs)
Arena:       FiveRing 4-quadrant gauntlet  (Experiments/Colby/FiveRing_Arena/)

Control loop:
    QuadrantFollower  →  current waypoint (x, y)
    MPPINavigator     →  [vx, vy, omega]  (20 Hz planning, applied at physics rate)
    SpotRoughTerrainPolicy  →  12 joint targets  (50 Hz, frozen weights)

Usage:
    conda activate isaaclab311
    cd Experiments/Colby/CombinedPolicyTraining

    # Rendered single episode
    python run_combined_mppi.py --rendered

    # Headless batch eval
    python run_combined_mppi.py --headless --num_episodes 20

    # Override checkpoint
    python run_combined_mppi.py --checkpoint <path/to/policy.pt> --rendered
"""

# ── 0. Args — MUST be parsed before any Isaac / omni imports ────────────────
import argparse
import os
import sys
import time
import signal
from pathlib import Path

THIS_DIR  = Path(__file__).resolve().parent          # CombinedPolicyTraining/
REPO_ROOT = THIS_DIR.parents[2]                      # repo root
ARENA_SRC = THIS_DIR.parent / "FiveRing_Arena" / "src"
MPPI_DIR  = REPO_ROOT / "Experiments" / "Cole" / "MPPI_FOLDER"

DEFAULT_CKPT = str(
    REPO_ROOT / "Experiments" / "Ryan" / "PARKOUR_NAV_phasefwplus_22100"
    / "parkour_phasefwplus_22100.pt"
)

parser = argparse.ArgumentParser(description="MPPI nav + frozen loco in FiveRing arena")
parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT,
                    help="Frozen locomotion checkpoint (.pt). Defaults to phasefwplus_22100.")
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--rendered", action="store_true",
                    help="Force rendered mode (overrides --headless)")
parser.add_argument("--output_dir", type=str, default="results/mppi/",
                    help="Directory for JSONL result files")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--zone_slowdown_cap", type=float, default=0.67,
                    help="Max vx (m/s) in boulder quadrant. 0.67 recommended for phasefwplus.")
parser.add_argument("--stair_speed_cap", type=float, default=0.6,
                    help="Max vx (m/s) in staircase quadrant.")

args, _ = parser.parse_known_args()

headless = args.headless and not args.rendered

# ── 1. Pre-load torch before Isaac Sim (CUDA DLL ordering) ──────────────────
import torch  # noqa: E402

# ── 2. Isaac Sim boot ────────────────────────────────────────────────────────
from isaaclab.app import AppLauncher

_orig_argv = sys.argv[:]
sys.argv = [sys.argv[0]] + (["--headless"] if headless else [])
app_launcher = AppLauncher(headless=headless)
simulation_app = app_launcher.app
sys.argv = _orig_argv

# ── 3. Enable required extensions ───────────────────────────────────────────
import omni.kit.app
_ext_mgr = omni.kit.app.get_app().get_extension_manager()
for _ext in [
    "isaacsim.core",
    "isaacsim.core.prims",
    "isaacsim.robot.quadruped",
    "omni.isaac.core",
    "omni.isaac.quadruped",
]:
    try:
        _ext_mgr.set_extension_enabled_immediate(_ext, True)
    except Exception:
        pass

# ── 4. Remaining imports (safe after AppLauncher) ────────────────────────────
import numpy as np
import omni
from omni.isaac.core import World
from pxr import UsdGeom, Gf, UsdPhysics, UsdLux

# Arena modules (FiveRing_Arena/src on path)
sys.path.insert(0, str(ARENA_SRC))

from configs.eval_cfg import (
    PHYSICS_DT, RENDERING_DT, CONTROL_DT,
    MAX_CONTROL_STEPS, SPAWN_POSITION, STUCK_TIMEOUT,
)
from configs.ring_params import (
    MAX_SCORE, QUADRANT_DEFS, GRASS_LEVELS, STAIRS_LEVELS,
    NUM_LEVELS, LEVEL_WIDTH,
)
from envs.base_arena import quat_to_yaw
from envs.ring_arena import create_quadrant_arena
from envs.vegetation import get_velocity_scale
from navigation.ring_follower import QuadrantFollower
from metrics.ring_collector import QuadrantCollector

# Cole's MPPI navigator (no Isaac Sim dependency — pure numpy)
sys.path.insert(0, str(MPPI_DIR))
from mppi_navigator import MPPINavigator

# ── Constants ────────────────────────────────────────────────────────────────

STABILIZE_STEPS = 100
SPAWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])   # facing +X (toward friction quadrant)

# parkour_phasefwplus_22100 training config
ACTION_SCALE = 0.3
MASON_OBS    = True   # height_scan first in obs vector

# MPPI bounds must match the loco policy's command range
# phasefwplus: vx [-1.0, 1.8], vy ±0.8, wz ±2.0
MPPI_VX_MIN    = -1.0
MPPI_VX_MAX    =  1.8
MPPI_VY_MIN    = -0.8
MPPI_VY_MAX    =  0.8
MPPI_OMEGA_MIN = -2.0
MPPI_OMEGA_MAX =  2.0

# ── Terrain helpers ──────────────────────────────────────────────────────────

def _get_grass_drag(x: float, y: float) -> float:
    r = np.sqrt(x * x + y * y)
    angle = np.arctan2(y, x) % (2 * np.pi)
    grass_def = QUADRANT_DEFS[1]  # Q2 = grass
    if grass_def["angle_start"] <= angle < grass_def["angle_end"]:
        lvl_idx = min(int(r / LEVEL_WIDTH), NUM_LEVELS - 1)
        return GRASS_LEVELS[lvl_idx]["drag_coeff"]
    return 0.0


def _in_boulder_quadrant(x: float, y: float) -> bool:
    angle = np.arctan2(y, x) % (2 * np.pi)
    bd = QUADRANT_DEFS[2]  # Q3 = boulders
    return bd["angle_start"] <= angle < bd["angle_end"]


def _in_stairs_quadrant(x: float, y: float) -> bool:
    angle = np.arctan2(y, x) % (2 * np.pi)
    sd = QUADRANT_DEFS[3]  # Q4 = stairs
    return sd["angle_start"] <= angle < sd["angle_end"]


def _in_friction_quadrant(x: float, y: float) -> bool:
    angle = np.arctan2(y, x) % (2 * np.pi)
    fd = QUADRANT_DEFS[0]  # Q1 = friction
    return fd["angle_start"] <= angle < fd["angle_end"]


# ── Signal handler ───────────────────────────────────────────────────────────

_collector_ref = None


def _shutdown(signum, frame):
    print(f"\n[SHUTDOWN] {signal.Signals(signum).name} — saving and exiting...",
          flush=True)
    if _collector_ref is not None:
        try:
            _collector_ref.save()
        except Exception:
            pass
    os._exit(0)


signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global _collector_ref
    np.random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  MPPI Nav + Frozen Loco — 4-Quadrant Gauntlet")
    print(f"  Checkpoint :  {args.checkpoint}")
    print(f"  Episodes   :  {args.num_episodes}")
    print(f"  Headless   :  {headless}")
    print(f"  Output     :  {args.output_dir}")
    print(f"{'='*60}\n")

    # ── World ──────────────────────────────────────────────────────────────
    world = World(
        physics_dt=PHYSICS_DT,
        rendering_dt=RENDERING_DT,
        stage_units_in_meters=1.0,
    )
    stage = omni.usd.get_context().get_stage()

    # Ground plane (failsafe beneath arena)
    ground = UsdGeom.Cube.Define(stage, "/World/GroundPlane")
    ground.GetSizeAttr().Set(1.0)
    ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.005))
    ground.AddScaleOp().Set(Gf.Vec3d(200.0, 200.0, 0.01))
    ground.GetDisplayColorAttr().Set([(0.4, 0.4, 0.4)])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    # Lighting
    UsdGeom.Xform.Define(stage, "/World/Lights")
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/dome")
    dome.CreateIntensityAttr(900.0)
    dome.CreateColorAttr(Gf.Vec3f(0.72, 0.85, 1.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/sun")
    sun.CreateIntensityAttr(1800.0)
    sun.CreateAngleAttr(0.5)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.82, 0.60))
    UsdGeom.Xformable(stage.GetPrimAtPath("/World/Lights/sun")).AddRotateXYZOp().Set(
        Gf.Vec3d(-32, 55, 0)
    )

    # ── Arena ──────────────────────────────────────────────────────────────
    follower = QuadrantFollower()
    stairs_positions = follower.stairs_waypoint_positions()
    print("Building arena...", flush=True)
    arena_summary = create_quadrant_arena(stage, stairs_positions)

    # ── Locomotion policy ──────────────────────────────────────────────────
    print("Loading locomotion policy...", flush=True)
    from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
    flat_policy = SpotFlatTerrainPolicy(
        prim_path="/World/Robot",
        name="Robot",
        position=np.array(SPAWN_POSITION),
    )
    world.reset()
    world.step(render=not headless)
    flat_policy.initialize()
    flat_policy.post_reset()

    from spot_rough_terrain_policy import SpotRoughTerrainPolicy
    spot = SpotRoughTerrainPolicy(
        flat_policy=flat_policy,
        checkpoint_path=args.checkpoint,
        ground_height_fn=None,           # PhysX raycasting for all terrain
        mason_baseline=MASON_OBS,
        action_scale=ACTION_SCALE,
        heightscan_ignore_obstacles=False,  # above-first rays so stairs are visible
    )
    spot.initialize()
    spot.apply_gains()
    spot._decimation = 1             # run policy every physics step (matches eval convention)
    print("Locomotion policy ready.", flush=True)

    # ── MPPI navigator ─────────────────────────────────────────────────────
    navigator = MPPINavigator(
        vx_min=MPPI_VX_MIN,
        vx_max=MPPI_VX_MAX,
        vy_min=MPPI_VY_MIN,
        vy_max=MPPI_VY_MAX,
        omega_min=MPPI_OMEGA_MIN,
        omega_max=MPPI_OMEGA_MAX,
    )
    print("MPPI navigator ready.", flush=True)

    # ── Metrics ────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_collector = QuadrantCollector(args.output_dir, "mppi")
    _collector_ref = metrics_collector

    # ── Initial stabilization ──────────────────────────────────────────────
    print("Stabilizing...", flush=True)
    for _ in range(STABILIZE_STEPS):
        spot.forward(PHYSICS_DT, np.zeros(3))
        world.step(render=not headless)

    # ── Episode loop ───────────────────────────────────────────────────────
    total_start  = time.time()
    stuck_steps  = int(STUCK_TIMEOUT / CONTROL_DT)

    for ep_idx in range(args.num_episodes):
        ep_id    = f"mppi_loco_ep{ep_idx:04d}"
        ep_start = time.time()

        # Reset
        spot.robot.set_world_pose(
            position=np.array(SPAWN_POSITION),
            orientation=SPAWN_QUAT,
        )
        spot.post_reset()
        follower.reset()
        navigator.reset()
        metrics_collector.start_episode(ep_id)

        for _ in range(10):
            spot.forward(PHYSICS_DT, np.zeros(3))
            world.step(render=not headless)

        last_sig_pos  = np.array(SPAWN_POSITION[:2])
        last_sig_step = 0

        for step in range(MAX_CONTROL_STEPS):
            pos, quat = spot.robot.get_world_pose()
            lin_vel   = spot.robot.get_linear_velocity()
            ang_vel   = spot.robot.get_angular_velocity()

            pos_np   = np.array(pos,     dtype=np.float64)
            quat_np  = np.array(quat,    dtype=np.float64)
            yaw      = quat_to_yaw(quat_np)
            sim_time = step * CONTROL_DT

            metrics_collector.step(
                root_pos=pos_np,
                root_quat=quat_np,
                root_lin_vel=np.array(lin_vel, dtype=np.float64),
                root_ang_vel=np.array(ang_vel, dtype=np.float64),
                sim_time=sim_time,
                ground_z=0.0,
            )

            if metrics_collector.episode_done():
                break

            # Stuck detection
            dist_moved = np.linalg.norm(pos_np[:2] - last_sig_pos)
            if dist_moved > 1.0:
                last_sig_pos  = pos_np[:2].copy()
                last_sig_step = step
            elif step - last_sig_step > stuck_steps:
                print(f"    [STUCK] No movement for {STUCK_TIMEOUT}s — ending episode")
                break

            # Navigation: MPPI solves toward the current waypoint, then follower
            # checks proximity and advances to the next waypoint as a side effect.
            target = follower.current_target  # (x, y, z) before possible advancement

            if target is not None:
                cmd = navigator.solve(
                    pos=pos_np[:2],
                    yaw=yaw,
                    target=np.array([target[0], target[1]]),
                    obstacles=[],  # arena obstacles not passed — MPPI does point-to-point
                )
            else:
                cmd = np.zeros(3)

            # Waypoint advancement check (discard velocity output)
            follower.compute_commands(pos_np, yaw)

            # Grass drag
            drag = _get_grass_drag(float(pos_np[0]), float(pos_np[1]))
            if drag > 0.0:
                cmd[0] *= get_velocity_scale(drag)

            # Boulder slowdown (phasefwplus gait wants < 1.0 m/s on rough rocks)
            if _in_boulder_quadrant(float(pos_np[0]), float(pos_np[1])):
                cmd[0] = min(cmd[0], args.zone_slowdown_cap)

            # Stair slowdown — approach at walking pace so the policy can adapt
            if _in_stairs_quadrant(float(pos_np[0]), float(pos_np[1])):
                cmd[0] = min(cmd[0], args.stair_speed_cap)

            # Outer ring cap — level 3+ terrain is harder across all quadrants
            r = np.sqrt(pos_np[0] ** 2 + pos_np[1] ** 2)
            if r >= 2.0 * LEVEL_WIDTH:
                cmd[0] = min(cmd[0], 1.5)

            # Friction quadrant: low-mu surface requires slow, straight movement
            # Cap omega too — spinning in place on ice causes falls
            if _in_friction_quadrant(float(pos_np[0]), float(pos_np[1])) and r >= 2.0 * LEVEL_WIDTH:
                cmd[0] = min(cmd[0], 0.6)
                cmd[2] = np.clip(cmd[2], -0.5, 0.5)

            spot.forward(PHYSICS_DT, cmd)
            world.step(render=not headless)

            if follower.is_done:
                break

        result   = metrics_collector.end_episode(follower=follower)
        ep_time  = time.time() - ep_start

        total_wps = result.get("total_waypoints_reached", 0)
        score     = result.get("composite_score", 0)
        fell      = result.get("fall_detected", False)

        if follower.is_done:
            status = "COMPLETE"
        elif fell:
            status = f"FELL {result.get('fall_quadrant','?')} L{result.get('fall_level','?')}"
        else:
            status = "TIMEOUT"

        q_summary = "".join(
            f" {qd['name'][:4]}={result.get('quadrant_scores', {}).get(qd['name'], {}).get('waypoints', 0)}/10"
            for qd in QUADRANT_DEFS
        )
        print(f"  [{ep_idx+1:4d}/{args.num_episodes}] {ep_id}  "
              f"{status:16s}  wps={total_wps:2d}/40  "
              f"score={score:6.1f}/{MAX_SCORE}{q_summary}  time={ep_time:.1f}s")

        if (ep_idx + 1) % 50 == 0:
            metrics_collector.save()
            print(f"  >> Checkpoint saved at episode {ep_idx + 1}")

    metrics_collector.save()

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  MPPI + Loco evaluation complete!")
    print(f"  Episodes : {args.num_episodes}")
    print(f"  Time     : {total_time:.1f}s  ({total_time/60:.1f} min)")
    print(f"  Results  : {args.output_dir}")
    print(f"{'='*60}\n")

    os._exit(0)


if __name__ == "__main__":
    main()
