"""Main entry point for the 4-quadrant navigation gauntlet evaluation.

Runs N episodes where Spot starts at the center and navigates 43 waypoints
across 4 terrain quadrants (Friction, Grass, Boulders, Stairs) with
5 difficulty levels each.

Usage (headless on H100):
    ./isaaclab.sh -p src/run_ring_eval.py --headless \
        --policy rough --checkpoint checkpoints/model.pt \
        --num_episodes 100 --output_dir results/

Usage (rendered locally):
    ./isaaclab.sh -p src/run_ring_eval.py \
        --policy rough --checkpoint checkpoints/model.pt \
        --num_episodes 1 --rendered
"""

# ── 0. Parse args BEFORE any Isaac imports ──────────────────────────────
import argparse
import signal
import sys
import os
import time

_metrics_collector_ref = None


def _graceful_shutdown(signum, frame):
    """Handle SIGINT/SIGTERM: save data and force-exit."""
    sig_name = signal.Signals(signum).name
    print(f"\n[SHUTDOWN] Caught {sig_name} — saving data and exiting...",
          flush=True)
    if _metrics_collector_ref is not None:
        try:
            _metrics_collector_ref.save()
            print("[SHUTDOWN] Metrics saved.", flush=True)
        except Exception as e:
            print(f"[SHUTDOWN] Warning: could not save metrics: {e}",
                  flush=True)
    print("[SHUTDOWN] Force-exiting (skipping SimulationApp.close).",
          flush=True)
    os._exit(0)


signal.signal(signal.SIGINT, _graceful_shutdown)
signal.signal(signal.SIGTERM, _graceful_shutdown)

parser = argparse.ArgumentParser(description="4-quadrant navigation gauntlet evaluation")
parser.add_argument("--policy", type=str, default="rough",
                    choices=["flat", "rough"],
                    help="Policy type to evaluate (default: rough)")
parser.add_argument("--num_episodes", type=int, default=100,
                    help="Number of episodes to run (default: 100)")
parser.add_argument("--headless", action="store_true", default=False,
                    help="Run without GUI rendering")
parser.add_argument("--rendered", action="store_true", default=False,
                    help="Force rendered mode (overrides --headless)")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to rough policy checkpoint (.pt)")
parser.add_argument("--output_dir", type=str, default="results/",
                    help="Output directory for JSONL files")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
parser.add_argument("--mason", action="store_true", default=False,
                    help="Use Mason obs order (height_scan first)")

args, remaining = parser.parse_known_args()

headless = args.headless and not args.rendered

# ── 1. Create Isaac Sim via AppLauncher ──────────────────────────────────
from isaaclab.app import AppLauncher

_orig_argv = sys.argv[:]
sys.argv = [sys.argv[0]] + (["--headless"] if headless else [])
app_launcher = AppLauncher(headless=headless)
simulation_app = app_launcher.app
sys.argv = _orig_argv

# ── 1b. Enable required Isaac Sim extensions ────────────────────────────
import omni.kit.app
_ext_mgr = omni.kit.app.get_app().get_extension_manager()
_required_exts = [
    "isaacsim.core",
    "isaacsim.core.prims",
    "isaacsim.robot.quadruped",
    "omni.isaac.core",
    "omni.isaac.quadruped",
]
for ext in _required_exts:
    try:
        _ext_mgr.set_extension_enabled_immediate(ext, True)
    except Exception:
        pass

# ── 2. Now safe to import Isaac and project modules ─────────────────────
import numpy as np
import omni
from omni.isaac.core import World
from pxr import UsdGeom, Gf, UsdPhysics, UsdLux

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.eval_cfg import (
    PHYSICS_DT, RENDERING_DT, CONTROL_DT,
    MAX_CONTROL_STEPS, FALL_THRESHOLD, SPAWN_POSITION,
    ROBOT_CONFIGS, STUCK_TIMEOUT,
)
from configs.ring_params import (
    NUM_QUADRANTS, NUM_LEVELS, LEVEL_WEIGHTS, MAX_SCORE,
    QUADRANT_DEFS, GRASS_LEVELS, STAIRS_LEVELS, LEVEL_WIDTH,
    get_quadrant_and_level, pyramid_summit_height,
)
from envs.base_arena import quat_to_yaw
from envs.ring_arena import create_quadrant_arena
from envs.vegetation import get_velocity_scale
from navigation.ring_follower import QuadrantFollower
from metrics.ring_collector import QuadrantCollector


# ── Constants ───────────────────────────────────────────────────────────
STABILIZE_STEPS = 100
SPAWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # facing +X (toward friction quadrant)


def _get_grass_drag(x, y):
    """Return drag coefficient if position is in grass quadrant, else 0."""
    r = np.sqrt(x * x + y * y)
    angle = np.arctan2(y, x) % (2 * np.pi)
    grass_def = QUADRANT_DEFS[1]  # Q2 = grass
    if grass_def["angle_start"] <= angle < grass_def["angle_end"]:
        lvl_idx = min(int(r / LEVEL_WIDTH), NUM_LEVELS - 1)
        return GRASS_LEVELS[lvl_idx]["drag_coeff"]
    return 0.0


def _get_stairs_ground_z(x, y, pyramid_summits):
    """Return ground height at position if on a stair pyramid, else 0.

    This is a simplification — uses flat ground for fall detection everywhere
    except we could enhance this with PhysX raycasting. For now, the height
    scanner handles terrain via PhysX rays.
    """
    # Stairs quadrant fall detection uses flat ground (pyramids are small)
    # The policy's height scanner sees real geometry via PhysX raycasting
    return 0.0


def main():
    robot_cfg = ROBOT_CONFIGS["spot"]
    spawn_pos = SPAWN_POSITION

    quad_names = [q["label"] for q in QUADRANT_DEFS]
    print(f"\n{'='*60}")
    print(f"  4-Quadrant Navigation Gauntlet")
    print(f"  Policy:      {args.policy}")
    print(f"  Episodes:    {args.num_episodes}")
    print(f"  Headless:    {headless}")
    print(f"  Output:      {args.output_dir}")
    print(f"  Quadrants:   {', '.join(quad_names)}")
    print(f"  Waypoints:   40 quadrant + 4 transition = 44 total")
    print(f"  Max score:   {MAX_SCORE} (4 × {sum(LEVEL_WEIGHTS)})")
    print(f"{'='*60}\n")

    # ── 3. Create World ─────────────────────────────────────────────────
    world = World(
        physics_dt=PHYSICS_DT,
        rendering_dt=RENDERING_DT,
        stage_units_in_meters=1.0,
    )
    stage = omni.usd.get_context().get_stage()

    # ── 4. Ground plane failsafe ───────────────────────────────────────
    ground = UsdGeom.Cube.Define(stage, "/World/GroundPlane")
    ground.GetSizeAttr().Set(1.0)
    ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.005))
    ground.AddScaleOp().Set(Gf.Vec3d(200.0, 200.0, 0.01))
    ground.GetDisplayColorAttr().Set([(0.4, 0.4, 0.4)])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    # ── 4b. Scene lighting ────────────────────────────────────────────
    light_path = "/World/Lights"
    UsdGeom.Xform.Define(stage, light_path)

    dome = UsdLux.DomeLight.Define(stage, f"{light_path}/dome")
    dome.CreateIntensityAttr(500.0)
    dome.CreateColorAttr(Gf.Vec3f(0.85, 0.90, 1.0))

    sun = UsdLux.DistantLight.Define(stage, f"{light_path}/sun")
    sun.CreateIntensityAttr(3000.0)
    sun.CreateAngleAttr(1.0)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.85))
    sun_prim = stage.GetPrimAtPath(f"{light_path}/sun")
    UsdGeom.Xformable(sun_prim).AddRotateXYZOp().Set(Gf.Vec3d(-45, 30, 0))

    # ── 5. Generate waypoints first (stairs needs positions for pyramids)
    follower = QuadrantFollower()
    stairs_wp_positions = follower.stairs_waypoint_positions()

    # ── 6. Build quadrant arena ─────────────────────────────────────────
    print("Building 4-quadrant arena...")
    arena_summary = create_quadrant_arena(stage, stairs_wp_positions)
    pyramid_summits = arena_summary.get("pyramid_summits", {})

    # ── 7. Load policy and spawn robot ──────────────────────────────────
    print(f"Loading {args.policy} policy...", flush=True)

    from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy

    flat_policy = SpotFlatTerrainPolicy(
        prim_path="/World/Robot",
        name="Robot",
        position=np.array(spawn_pos),
    )

    world.reset()
    world.step(render=not headless)
    flat_policy.initialize()
    flat_policy.post_reset()

    if args.policy == "rough":
        from spot_rough_terrain_policy import SpotRoughTerrainPolicy
        robot_policy = SpotRoughTerrainPolicy(
            flat_policy=flat_policy,
            checkpoint_path=args.checkpoint,
            ground_height_fn=None,  # PhysX raycasting for all terrain
            mason_baseline=args.mason,
        )
        robot_policy.initialize()
        robot_policy.apply_gains()
        robot_policy._decimation = 1
    else:
        robot_policy = flat_policy

    spot = robot_policy
    print("Spot loaded and initialized.", flush=True)

    # ── 8. Create metrics collector ─────────────────────────────────────
    metrics_collector = QuadrantCollector(args.output_dir, args.policy)
    os.makedirs(args.output_dir, exist_ok=True)

    global _metrics_collector_ref
    _metrics_collector_ref = metrics_collector

    # ── 9. Stabilization period ─────────────────────────────────────────
    print("Stabilizing robot...", flush=True)
    for _ in range(STABILIZE_STEPS):
        spot.forward(PHYSICS_DT, np.array([0.0, 0.0, 0.0]))
        world.step(render=not headless)

    # ── 10. Episode loop ────────────────────────────────────────────────
    total_start = time.time()
    stuck_steps = int(STUCK_TIMEOUT / CONTROL_DT)

    for ep_idx in range(args.num_episodes):
        ep_id = f"quad_{args.policy}_ep{ep_idx:04d}"
        ep_start = time.time()

        # Reset robot to spawn
        spot.robot.set_world_pose(
            position=np.array(spawn_pos),
            orientation=SPAWN_QUAT,
        )

        if hasattr(spot, 'post_reset'):
            spot.post_reset()

        follower.reset()
        metrics_collector.start_episode(ep_id)

        # Brief stabilization
        for _ in range(10):
            spot.forward(PHYSICS_DT, np.array([0.0, 0.0, 0.0]))
            world.step(render=not headless)

        # Stuck detection state
        last_significant_pos = np.array(spawn_pos[:2])
        last_significant_step = 0

        for step in range(MAX_CONTROL_STEPS):
            pos, quat = spot.robot.get_world_pose()
            lin_vel = spot.robot.get_linear_velocity()
            ang_vel = spot.robot.get_angular_velocity()

            pos_np = np.array(pos, dtype=np.float64)
            quat_np = np.array(quat, dtype=np.float64)
            lin_vel_np = np.array(lin_vel, dtype=np.float64)
            ang_vel_np = np.array(ang_vel, dtype=np.float64)

            yaw = quat_to_yaw(quat_np)
            sim_time = step * CONTROL_DT

            # Ground height for stairs
            ground_z = _get_stairs_ground_z(
                float(pos_np[0]), float(pos_np[1]), pyramid_summits)

            metrics_collector.step(
                root_pos=pos_np,
                root_quat=quat_np,
                root_lin_vel=lin_vel_np,
                root_ang_vel=ang_vel_np,
                sim_time=sim_time,
                ground_z=ground_z,
            )

            if metrics_collector.episode_done():
                break

            # Stuck detection
            dist_from_last = np.linalg.norm(pos_np[:2] - last_significant_pos)
            if dist_from_last > 1.0:
                last_significant_pos = pos_np[:2].copy()
                last_significant_step = step
            elif step - last_significant_step > stuck_steps:
                print(f"    [STUCK] No movement for {STUCK_TIMEOUT}s — ending episode")
                break

            # Navigation commands
            cmd = follower.compute_commands(pos_np, yaw)

            # Grass drag
            drag = _get_grass_drag(float(pos_np[0]), float(pos_np[1]))
            if drag > 0:
                vscale = get_velocity_scale(drag)
                cmd[0] *= vscale

            spot.forward(PHYSICS_DT, cmd)
            world.step(render=not headless)

            if follower.is_done:
                break

        # End episode
        result = metrics_collector.end_episode(follower=follower)
        ep_time = time.time() - ep_start

        total_wps = result.get("total_waypoints_reached", 0)
        score = result.get("composite_score", 0)
        fell = result.get("fall_detected", False)
        fall_quad = result.get("fall_quadrant", None)
        fall_lvl = result.get("fall_level", None)

        if follower.is_done:
            status = "COMPLETE"
        elif fell:
            status = f"FELL {fall_quad} L{fall_lvl}"
        else:
            status = "TIMEOUT"

        # Per-quadrant summary
        q_summary = ""
        for qd in QUADRANT_DEFS:
            qn = qd["name"]
            qs = result.get("quadrant_scores", {}).get(qn, {})
            qwps = qs.get("waypoints", 0)
            q_summary += f" {qn[:4]}={qwps}/10"

        print(f"  [{ep_idx+1:4d}/{args.num_episodes}] {ep_id}  "
              f"{status:16s}  wps={total_wps:2d}/40  "
              f"score={score:6.1f}/{MAX_SCORE}{q_summary}  time={ep_time:.1f}s")

        if (ep_idx + 1) % 50 == 0:
            metrics_collector.save()
            print(f"  >> Saved checkpoint at episode {ep_idx+1}")

    # ── 11. Final save ──────────────────────────────────────────────────
    metrics_collector.save()

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  4-Quadrant Gauntlet complete!")
    print(f"  Total episodes: {args.num_episodes}")
    print(f"  Total time:     {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Avg per episode: {total_time/args.num_episodes:.1f}s")
    print(f"  Results saved to: {args.output_dir}")
    print(f"{'='*60}\n")

    print("Exiting (skipping SimulationApp.close to avoid GPU hang).",
          flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
