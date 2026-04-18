"""Cole arena eval with Skill-Nav Lite — waypoint P-controller, no CNN, no RL nav.

Mirrors cole_arena_eval.py but swaps ActorCriticCNN for SkillNavLiteNavigator.
By default runs on SpotFlatTerrainPolicy (Isaac stock flat-terrain gait);
pass --loco_checkpoint to also engage SpotRoughTerrainPolicy (Boulder V6).

Usage:
    python scripts/cole_arena_skillnav_lite.py --episodes 3 --rendered

    python scripts/cole_arena_skillnav_lite.py \
        --loco_checkpoint checkpoints/boulder_v6_expert_4500.pt \
        --episodes 5 --headless
"""

import argparse
import math
import os
import sys
import string
import csv
import numpy as np

parser = argparse.ArgumentParser(description="Skill-Nav Lite in Cole's waypoint arena")
DEFAULT_FLAT_MASTER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "SIM_TO_REAL", "checkpoints", "flat_v3_3700.pt",
)
parser.add_argument("--loco_checkpoint", type=str, default=DEFAULT_FLAT_MASTER,
                    help="RSL-RL loco checkpoint. Defaults to flat_v3_3700 (Flat Master V3).")
parser.add_argument("--stock_flat", action="store_true",
                    help="Use stock SpotFlatTerrainPolicy instead of Flat Master V3 (debug).")
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--rendered", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--kp_lin", type=float, default=1.0)
parser.add_argument("--kp_ang", type=float, default=2.0)
parser.add_argument("--max_lin_speed", type=float, default=2.2)
parser.add_argument("--apf_radius", type=float, default=1.8)
parser.add_argument("--apf_gain", type=float, default=1.1)
parser.add_argument("--apf_tangent", type=float, default=0.6)
parser.add_argument("--no_obstacles", action="store_true",
                    help="Empty arena (skip 40-cube generation) — isolates nav stack from collisions.")
parser.add_argument("--num_obstacles", type=int, default=40,
                    help="Number of obstacles to spawn (default 40). Ignored if --no_obstacles.")
args = parser.parse_args()

headless = args.headless and not args.rendered

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": headless, "width": 1920, "height": 1080})

import omni
from omni.isaac.core import World
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
from pxr import UsdGeom, UsdLux, UsdPhysics, Gf

nav_alex_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
alex_root = os.path.dirname(nav_alex_root)
sys.path.insert(0, os.path.join(nav_alex_root, "source", "nav_locomotion"))
sys.path.insert(0, os.path.join(alex_root, "4_env_test", "src"))

from nav_locomotion.modules.skill_nav_lite import SkillNavLiteNavigator

USE_ROUGH = not args.stock_flat
if USE_ROUGH:
    from spot_rough_terrain_policy import SpotRoughTerrainPolicy

ARENA_RADIUS = 25.0
SPOT_START = np.array([0.0, 0.0, 0.6])
FALL_HEIGHT = 0.25

SPOT_DEFAULT_TYPE_GROUPED = np.array([
    0.1, -0.1, 0.1, -0.1,
    0.9,  0.9, 1.1,  1.1,
   -1.5, -1.5, -1.5, -1.5,
], dtype=np.float32)

WAYPOINT_COUNT = 25
WAYPOINT_LABELS = list(string.ascii_uppercase[:WAYPOINT_COUNT])
WAYPOINT_DIST_A = 20.0
WAYPOINT_SPACING_BZ = 40.0
WAYPOINT_REACH_DIST = 0.5
WAYPOINT_BOUNDARY_MARGIN = 2.0

EPISODE_START_SCORE = 300.0
TIME_DECAY_PER_SEC = 1.0
WAYPOINT_REWARD = 15.0

PHYSICS_DT = 1.0 / 500.0
RENDER_DT = 10.0 / 500.0
NAV_DT = 1.0 / 10.0
PHYSICS_PER_LOCO = 10
LOCO_STEPS_PER_NAV = 5


def inside_arena(x, y, margin=0.0):
    return x ** 2 + y ** 2 < (ARENA_RADIUS - margin) ** 2


def random_inside_arena(margin=0.0, rng=None):
    rng = rng or np.random.default_rng()
    r_limit = ARENA_RADIUS - margin
    while True:
        x = rng.uniform(-r_limit, r_limit)
        y = rng.uniform(-r_limit, r_limit)
        if x ** 2 + y ** 2 < r_limit ** 2:
            return np.array([x, y])


def distance_2d(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def quat_to_yaw(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def build_arena(stage, rng, num_obstacles=40):
    ground_path = "/World/ground"
    UsdGeom.Cylinder.Define(stage, ground_path)
    gp = stage.GetPrimAtPath(ground_path)
    gp.GetAttribute("radius").Set(float(ARENA_RADIUS + 5.0))
    gp.GetAttribute("height").Set(0.1)
    xf = UsdGeom.Xformable(gp)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
    UsdPhysics.CollisionAPI.Apply(gp)
    mat = UsdPhysics.MaterialAPI.Apply(gp)
    mat.CreateStaticFrictionAttr(0.8)
    mat.CreateDynamicFrictionAttr(0.6)

    light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    light.CreateIntensityAttr(1500)

    obstacles = []
    for i in range(num_obstacles):
        pos = random_inside_arena(margin=3.0, rng=rng)
        if distance_2d(pos, [0, 0]) < 3.0:
            continue
        size = rng.uniform(0.15, 0.5)
        height = rng.uniform(0.15, 0.6)
        mass = rng.uniform(0.5, 80.0)
        path = f"/World/obstacles/obs_{i}"
        UsdGeom.Cube.Define(stage, path)
        prim = stage.GetPrimAtPath(path)
        prim.GetAttribute("size").Set(float(size))
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(height / 2)))
        xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, float(height / size)))
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(float(mass))
        if mass > 65.0:
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
        obstacles.append((float(pos[0]), float(pos[1]), float(size)))
    print(f"[ARENA] r={ARENA_RADIUS}m with {len(obstacles)} obstacles")
    return obstacles


def generate_waypoints(rng):
    waypoints = []
    angle = rng.uniform(0, 2 * math.pi)
    a_pos = np.array([WAYPOINT_DIST_A * math.cos(angle), WAYPOINT_DIST_A * math.sin(angle)])
    waypoints.append({"label": "A", "pos": a_pos})
    for i in range(1, WAYPOINT_COUNT):
        for _ in range(1000):
            pos = random_inside_arena(margin=WAYPOINT_BOUNDARY_MARGIN, rng=rng)
            if distance_2d(pos, waypoints[-1]["pos"]) >= WAYPOINT_SPACING_BZ:
                waypoints.append({"label": WAYPOINT_LABELS[i], "pos": pos})
                break
        else:
            waypoints.append({
                "label": WAYPOINT_LABELS[i],
                "pos": random_inside_arena(margin=WAYPOINT_BOUNDARY_MARGIN, rng=rng),
            })
    return waypoints


def main():
    print(f"\n{'=' * 72}")
    print("COLE ARENA EVAL — SKILL-NAV LITE")
    print(f"{'=' * 72}")
    print(f"Loco policy:     {'Flat Master V3 (' + os.path.basename(args.loco_checkpoint) + ')' if USE_ROUGH else 'SpotFlatTerrainPolicy (stock)'}")
    print(f"Episodes:        {args.episodes}")
    print(f"kp_lin={args.kp_lin}  kp_ang={args.kp_ang}  max_lin_speed={args.max_lin_speed}")

    world = World(physics_dt=PHYSICS_DT, rendering_dt=RENDER_DT, stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    rng = np.random.default_rng(args.seed)
    n_obs = 0 if args.no_obstacles else args.num_obstacles
    obstacles_info = build_arena(stage, rng, num_obstacles=n_obs)

    flat_policy = SpotFlatTerrainPolicy(prim_path="/World/Spot", name="Spot", position=SPOT_START)
    world.reset()
    world.step(render=not headless)
    flat_policy.initialize()
    flat_policy.post_reset()

    if USE_ROUGH:
        robot_policy = SpotRoughTerrainPolicy(
            flat_policy=flat_policy,
            checkpoint_path=args.loco_checkpoint,
            mason_baseline=True,
            ground_height_fn=lambda x: 0.0,
        )
        robot_policy.initialize()
        robot_policy.apply_gains()
        robot_policy._decimation = 10
        step_loco = lambda cmd: robot_policy.forward(PHYSICS_DT, list(cmd))
        print(f"[OK] Flat Master V3 engaged ({os.path.basename(args.loco_checkpoint)}, mason obs, flat heightscan)")
    else:
        step_loco = lambda cmd: flat_policy.forward(PHYSICS_DT, list(cmd))
        print("[OK] Stock SpotFlatTerrainPolicy engaged (debug mode)")

    csv_path = os.path.join(nav_alex_root, "results", "cole_arena_skillnav_lite.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Episode", "Waypoints", "Distance", "Score", "Time", "Status"])

    navigator = SkillNavLiteNavigator(
        waypoints=[],
        reach_threshold=WAYPOINT_REACH_DIST,
        kp_lin=args.kp_lin,
        kp_ang=args.kp_ang,
        max_lin_speed=args.max_lin_speed,
        obstacles=obstacles_info,
        obstacle_influence_radius=args.apf_radius,
        obstacle_repulse_gain=args.apf_gain,
        obstacle_tangent_bias=args.apf_tangent,
        robot_radius=0.45,
    )
    print(f"[NAV] APF obstacles: {len(obstacles_info)}  R={args.apf_radius}m  gain={args.apf_gain}  tangent={args.apf_tangent}  max_spd={args.max_lin_speed}")

    all_results = []
    for ep in range(1, args.episodes + 1):
        waypoints = generate_waypoints(rng)
        navigator.reset(waypoints=waypoints)

        score = EPISODE_START_SCORE
        wp_reached = 0
        total_dist = 0.0
        last_pos = np.array([SPOT_START[0], SPOT_START[1]])
        sim_time = 0.0
        status = "TIMEOUT"
        current_vel_cmd = np.zeros(3, dtype=np.float32)

        flat_policy.robot.set_world_pose(
            position=np.array(SPOT_START),
            orientation=np.array([1, 0, 0, 0], dtype=np.float32),
        )
        flat_policy.robot.set_joint_positions(SPOT_DEFAULT_TYPE_GROUPED)

        print(f"\n--- Episode {ep}/{args.episodes} ---")
        print(f"  WP-A at ({waypoints[0]['pos'][0]:.1f}, {waypoints[0]['pos'][1]:.1f})")

        print("  Stabilizing...", flush=True)
        for i in range(250):
            try:
                step_loco([0.0, 0.0, 0.0])
                world.step(render=not headless)
            except Exception as e:
                print(f"    [STAB {i}] EXC: {type(e).__name__}: {e}", flush=True)
                raise
            if i % 50 == 0:
                try:
                    p, _ = flat_policy.robot.get_world_pose()
                    print(f"    [STAB {i}] z={float(p[2]):.3f}", flush=True)
                except Exception as e:
                    print(f"    [STAB {i}] pose fail: {e}", flush=True)
        try:
            spos, _ = flat_policy.robot.get_world_pose()
            print(f"  Stabilized. z={float(spos[2]):.3f}m", flush=True)
        except Exception:
            print("  Stabilized.", flush=True)

        step_count = 0
        max_steps = 500000
        nav_step_counter = 0
        nav_tick = 0

        while step_count < max_steps and simulation_app.is_running():
            try:
                step_loco(current_vel_cmd)
                world.step(render=not headless)
            except Exception as e:
                import traceback
                print(f"  [LOOP EXC @ step {step_count}] {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                status = "EXC"
                break
            step_count += 1
            sim_time = step_count * PHYSICS_DT
            nav_step_counter += 1

            if nav_step_counter >= (PHYSICS_PER_LOCO * LOCO_STEPS_PER_NAV):
                nav_step_counter = 0
                try:
                    spot_pos, spot_quat = flat_policy.robot.get_world_pose()
                except Exception:
                    continue

                spot_x, spot_y, spot_z = float(spot_pos[0]), float(spot_pos[1]), float(spot_pos[2])
                spot_yaw = quat_to_yaw(spot_quat)

                if spot_z < FALL_HEIGHT:
                    status = "FELL"
                    print(f"  [FELL] z={spot_z:.2f} at ({spot_x:.1f}, {spot_y:.1f})")
                    break
                if not inside_arena(spot_x, spot_y, margin=-1.0):
                    status = "OOB"
                    print(f"  [OOB] at ({spot_x:.1f}, {spot_y:.1f})")
                    break

                score -= TIME_DECAY_PER_SEC * NAV_DT
                if score <= 0:
                    status = "SCORE_DEPLETED"
                    break

                cur_pos = np.array([spot_x, spot_y])
                total_dist += distance_2d(cur_pos, last_pos)
                last_pos = cur_pos

                if navigator.check_reached(cur_pos):
                    wp_reached += 1
                    score += WAYPOINT_REWARD
                    label = WAYPOINT_LABELS[wp_reached - 1]
                    print(f"  [WP {label}] reached ({wp_reached}/{WAYPOINT_COUNT})")
                    if navigator.done:
                        status = "COMPLETE"
                        break

                current_vel_cmd = navigator.get_velocity_command(cur_pos, spot_yaw)
                nav_tick += 1

                if nav_tick <= 20 or nav_tick % 20 == 0:
                    wp = navigator.current_waypoint
                    if wp is not None:
                        d_wp = distance_2d(cur_pos, wp)
                        print(f"    [nav {nav_tick:3d} t={sim_time:5.1f}s] pos=({spot_x:6.2f},{spot_y:6.2f}) "
                              f"yaw={math.degrees(spot_yaw):+6.1f} -> {navigator.current_label} "
                              f"d={d_wp:5.2f}m cmd=[{current_vel_cmd[0]:+.2f},{current_vel_cmd[1]:+.2f},"
                              f"{current_vel_cmd[2]:+.2f}]", flush=True)

            if step_count % 5000 == 0:
                try:
                    wp = navigator.current_waypoint
                    wp_str = f"wp={navigator.current_label}({wp[0]:.1f},{wp[1]:.1f})" if wp is not None else "wp=done"
                    print(f"  [t={sim_time:.1f}s] pos=({spot_x:.1f},{spot_y:.1f}) "
                          f"{wp_str} reached={wp_reached} cmd=[{current_vel_cmd[0]:.2f},"
                          f"{current_vel_cmd[1]:.2f},{current_vel_cmd[2]:.2f}]")
                except Exception:
                    pass

        print(f"  Loop exit: step={step_count} sim_running={simulation_app.is_running()} status={status}", flush=True)
        print(f"  Result: {status} | WP: {wp_reached}/{WAYPOINT_COUNT} | "
              f"Dist: {total_dist:.1f}m | Score: {score:.1f} | Time: {sim_time:.1f}s", flush=True)
        csv_writer.writerow([ep, wp_reached, f"{total_dist:.1f}", f"{score:.1f}",
                             f"{sim_time:.1f}", status])
        all_results.append({
            "episode": ep, "waypoints": wp_reached, "distance": total_dist,
            "score": score, "time": sim_time, "status": status,
        })

    csv_file.close()
    print(f"\n{'=' * 72}")
    print("SKILL-NAV LITE SUMMARY")
    print(f"{'=' * 72}")
    if all_results:
        wp_counts = [r["waypoints"] for r in all_results]
        print(f"Episodes:  {len(all_results)}")
        print(f"Waypoints: {np.mean(wp_counts):.1f} avg, {max(wp_counts)} max")
        print(f"Statuses:  {', '.join(r['status'] for r in all_results)}")
    print(f"CSV saved: {csv_path}")
    print(f"{'=' * 72}")

    os._exit(0)


if __name__ == "__main__":
    main()
