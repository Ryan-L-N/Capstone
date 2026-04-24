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
parser.add_argument("--max_lin_speed", type=float, default=2.2,
                    help="Max linear speed (m/s). Flat-40 default 2.2. Cole rich arena best: 2.4.")
parser.add_argument("--apf_radius", type=float, default=1.8,
                    help="APF influence radius (m). Flat-40 default 1.8. Cole rich arena best: 1.5.")
parser.add_argument("--apf_gain", type=float, default=1.1,
                    help="APF repulse gain. Flat-40 default 1.1. Cole rich arena best: 0.9.")
parser.add_argument("--apf_tangent", type=float, default=0.6,
                    help="Tangent-bias 0..1 (higher = slide around obstacles). Flat-40: 0.6. Cole rich: 0.8.")
parser.add_argument("--waypoint_reach", type=float, default=0.5,
                    help="Waypoint reach threshold (m). Larger = skip tight-zone creep near WPs.")
parser.add_argument("--no_obstacles", action="store_true",
                    help="Empty arena (skip obstacle generation) — isolates nav stack from collisions.")
parser.add_argument("--num_obstacles", type=int, default=40,
                    help="Simple-mode: number of cubes (default 40). Only used without --cole_arena.")
parser.add_argument("--cole_arena", action="store_true",
                    help="Use Cole's rich obstacle set (moveable + non-moveable + small static, 7 shapes).")
parser.add_argument("--moveable_pct", type=float, default=3.0,
                    help="Cole arena: target area %% for moveable (orange) obstacles.")
parser.add_argument("--nonmoveable_pct", type=float, default=3.0,
                    help="Cole arena: target area %% for non-moveable (blue) obstacles.")
parser.add_argument("--small_static_pct", type=float, default=0.3,
                    help="Cole arena: target area %% for small static (gray) hazards.")
parser.add_argument("--start_buffer", type=float, default=4.0,
                    help="Clearance radius around spawn (m) with no obstacles.")
parser.add_argument("--rough_heightscan", action="store_true",
                    help="Use real PhysX raycast for heightscan (required for Boulder V6 and other "
                         "rough-terrain policies; BREAKS Flat Master V3 — it treats obstacles as OOD terrain).")
parser.add_argument("--loco_decimation", type=int, default=10,
                    help="Policy decimation. world.step() advances 10 physics substeps per call. "
                         "FM V3 default 10 (production-proven for this script's flat/APF loop). "
                         "Boulder V6 REQUIRES --loco_decimation 1 (matches proven cole_arena_eval.py "
                         "integration; dec=10 runs V6 at 5Hz and it collapses).")
parser.add_argument("--loco_action_scale", type=float, default=None,
                    help="Override policy action_scale. None = SpotRoughTerrainPolicy default "
                         "(0.2 for Mason). PARKOUR_NAV teacher/student train at 0.3 — pass 0.3 "
                         "when loading a PARKOUR_NAV checkpoint.")
parser.add_argument("--fall_height", type=float, default=0.25,
                    help="Fall threshold (m). FM V3 stands ~0.54. Boulder V6 crouches ~0.24 at zero cmd.")
parser.add_argument("--stab_vx", type=float, default=0.0,
                    help="Forward vx command during stabilization. Non-zero keeps Boulder V6 gait-active "
                         "(V6 was trained to walk, not stand — zero cmd causes it to crouch-collapse).")
parser.add_argument("--stab_steps", type=int, default=250,
                    help="Stabilization steps. Default 250 (0.5s). Lower = less time for drift.")
parser.add_argument("--yaw_first_vx", type=float, default=0.3,
                    help="Forward vx held during yaw-first turning (|heading_err|>45deg). "
                         "FM V3 works at 0.3. Boulder V6 needs ~0.8-1.2 to stay gait-active when "
                         "turning in place (0.3 leaves V6 in standing crouch).")
parser.add_argument("--stuck_window", type=int, default=30,
                    help="Nav ticks to watch for stuck detection (10 Hz → 30 = 3s).")
parser.add_argument("--stuck_dist", type=float, default=0.3,
                    help="If max displacement over stuck_window < this (m), trigger escape.")
parser.add_argument("--escape_duration", type=int, default=15,
                    help="Escape-mode ticks (10 Hz → 15 = 1.5s).")
parser.add_argument("--escape_vx", type=float, default=1.5,
                    help="Forward vx during escape (push through local APF minimum).")
parser.add_argument("--escape_wz", type=float, default=1.5,
                    help="Yaw rate during escape (sign flipped from normal yaw to break symmetry).")
parser.add_argument("--skip_wp_after_escapes", type=int, default=3,
                    help="Skip unreachable waypoint after this many failed escapes.")
parser.add_argument("--global_planner", action="store_true",
                    help="Use grid A* to expand each waypoint into a dense sub-waypoint path "
                         "routed around known obstacles. APF still runs as a local deviator.")
parser.add_argument("--planner_res", type=float, default=0.5,
                    help="A* grid cell size in m.")
parser.add_argument("--planner_inflate", type=float, default=0.75,
                    help="Obstacle inflation in m (robot_radius + safety margin).")
parser.add_argument("--planner_subwp_step", type=float, default=2.0,
                    help="Max spacing between A* sub-waypoints in m.")
parser.add_argument("--online_map", action="store_true",
                    help="Start with zero obstacle knowledge; reveal obstacles within "
                         "sense_radius as the robot explores. Simulates limited-range "
                         "sensing on unknown arenas. Re-plans A* when map grows.")
parser.add_argument("--sense_radius", type=float, default=3.5,
                    help="Sensing range in m for online map mode.")
parser.add_argument("--replan_period_sec", type=float, default=2.0,
                    help="Minimum seconds between A* replans in online map mode.")
parser.add_argument("--depth_sensor", action="store_true",
                    help="Use real forward-facing PhysX raycast depth sensor for obstacle "
                         "detection (honest sensing; no ground-truth cheat). Implies --online_map.")
parser.add_argument("--sensor_fov_h", type=float, default=90.0,
                    help="Depth sensor horizontal FOV (deg).")
parser.add_argument("--sensor_fov_v", type=float, default=30.0,
                    help="Depth sensor vertical FOV (deg).")
parser.add_argument("--sensor_rays_h", type=int, default=64,
                    help="Depth sensor horizontal ray count.")
parser.add_argument("--sensor_rays_v", type=int, default=16,
                    help="Depth sensor vertical ray count.")
parser.add_argument("--sensor_max_dist", type=float, default=8.0,
                    help="Depth sensor max range in m.")
parser.add_argument("--sensor_grid_res", type=float, default=0.4,
                    help="Depth sensor occupancy grid cell size in m.")
parser.add_argument("--sensor_min_hits", type=int, default=2,
                    help="Occupancy cell hits required before it becomes an obstacle.")
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
from nav_locomotion.modules.grid_astar_planner import expand_waypoints as astar_expand_waypoints
from nav_locomotion.modules.grid_astar_planner import plan_path as astar_plan_path
from nav_locomotion.modules.online_obstacle_tracker import OnlineObstacleTracker
from nav_locomotion.modules.depth_raycast_detector import DepthRaycastObstacleDetector

USE_ROUGH = not args.stock_flat
if USE_ROUGH:
    from spot_rough_terrain_policy import SpotRoughTerrainPolicy

ARENA_RADIUS = 25.0
SPOT_START = np.array([0.0, 0.0, 0.54])
FALL_HEIGHT = args.fall_height

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
# Rendered-mode: never call world.step(render=True) — it advances physics by rendering_dt
# (20ms) not physics_dt (2ms), freezing the control cmd across 10 physics steps and causing
# stab-collapse / drift. Instead step physics at 500Hz and call world.render() every
# RENDER_EVERY_N_STEPS for a 50Hz visual — keeps control cadence correct and avoids the
# Kit render-pipeline overload that fast-shutdowns the app at ~500Hz render rate.
RENDER_EVERY_N_STEPS = 10
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


def _apply_rigid_body_physics(stage, prim_path, mass_kg, friction=0.5, convex_hull=True):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    if prim.GetTypeName() == "Mesh" and convex_hull:
        mesh_coll = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_coll.CreateApproximationAttr("convexHull")
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(mass_kg))


def _mk_rect(stage, path, w, d, h, color):
    mesh = UsdGeom.Mesh.Define(stage, path)
    hw, hd = w / 2, d / 2
    pts = [
        Gf.Vec3f(-hw, -hd, 0), Gf.Vec3f(hw, -hd, 0), Gf.Vec3f(hw, hd, 0), Gf.Vec3f(-hw, hd, 0),
        Gf.Vec3f(-hw, -hd, h), Gf.Vec3f(hw, -hd, h), Gf.Vec3f(hw, hd, h), Gf.Vec3f(-hw, hd, h),
    ]
    mesh.GetPointsAttr().Set(pts)
    mesh.GetFaceVertexCountsAttr().Set([4] * 6)
    mesh.GetFaceVertexIndicesAttr().Set([
        0, 1, 2, 3, 4, 7, 6, 5, 0, 4, 5, 1,
        2, 6, 7, 3, 0, 3, 7, 4, 1, 5, 6, 2,
    ])
    mesh.GetDisplayColorAttr().Set([color])


def _mk_trapezoid(stage, path, w_bot, w_top, d, h, color):
    mesh = UsdGeom.Mesh.Define(stage, path)
    hb, ht, hd = w_bot / 2, w_top / 2, d / 2
    pts = [
        Gf.Vec3f(-hb, -hd, 0), Gf.Vec3f(hb, -hd, 0), Gf.Vec3f(hb, hd, 0), Gf.Vec3f(-hb, hd, 0),
        Gf.Vec3f(-ht, -hd, h), Gf.Vec3f(ht, -hd, h), Gf.Vec3f(ht, hd, h), Gf.Vec3f(-ht, hd, h),
    ]
    mesh.GetPointsAttr().Set(pts)
    mesh.GetFaceVertexCountsAttr().Set([4] * 6)
    mesh.GetFaceVertexIndicesAttr().Set([
        0, 1, 2, 3, 4, 7, 6, 5, 0, 4, 5, 1,
        2, 6, 7, 3, 0, 3, 7, 4, 1, 5, 6, 2,
    ])
    mesh.GetDisplayColorAttr().Set([color])


def _mk_sphere(stage, path, radius, color, segments=12):
    mesh = UsdGeom.Mesh.Define(stage, path)
    pts = [Gf.Vec3f(0, 0, radius)]
    for i in range(1, segments):
        phi = math.pi * i / segments
        rho = radius * math.sin(phi)
        z = radius * math.cos(phi)
        for j in range(segments):
            theta = 2 * math.pi * j / segments
            pts.append(Gf.Vec3f(rho * math.cos(theta), rho * math.sin(theta), z))
    pts.append(Gf.Vec3f(0, 0, -radius))
    face_counts, indices = [], []
    for j in range(segments):
        indices.extend([0, 1 + j, 1 + (j + 1) % segments]); face_counts.append(3)
    for i in range(1, segments - 1):
        for j in range(segments):
            a = 1 + (i - 1) * segments + j
            b = 1 + (i - 1) * segments + (j + 1) % segments
            c = 1 + i * segments + (j + 1) % segments
            d = 1 + i * segments + j
            indices.extend([a, b, c, d]); face_counts.append(4)
    top = len(pts) - 1
    for j in range(segments):
        a = 1 + (segments - 2) * segments + j
        b = 1 + (segments - 2) * segments + (j + 1) % segments
        indices.extend([top, b, a]); face_counts.append(3)
    mesh.GetPointsAttr().Set(pts)
    mesh.GetFaceVertexCountsAttr().Set(face_counts)
    mesh.GetFaceVertexIndicesAttr().Set(indices)
    mesh.GetDisplayColorAttr().Set([color])


def _mk_diamond(stage, path, base, h, color):
    mesh = UsdGeom.Mesh.Define(stage, path)
    hw = base / 2
    pts = [
        Gf.Vec3f(-hw, -hw, 0), Gf.Vec3f(hw, -hw, 0),
        Gf.Vec3f(hw, hw, 0), Gf.Vec3f(-hw, hw, 0),
        Gf.Vec3f(0, 0, h),
    ]
    mesh.GetPointsAttr().Set(pts)
    mesh.GetFaceVertexCountsAttr().Set([4, 3, 3, 3, 3])
    mesh.GetFaceVertexIndicesAttr().Set([
        0, 1, 2, 3, 0, 4, 1, 1, 4, 2, 2, 4, 3, 3, 4, 0,
    ])
    mesh.GetDisplayColorAttr().Set([color])


def _mk_oval(stage, path, r_major, r_minor, h, color, segments=16):
    mesh = UsdGeom.Mesh.Define(stage, path)
    pts = []
    for z in [0, h]:
        for i in range(segments):
            theta = 2 * math.pi * i / segments
            pts.append(Gf.Vec3f(r_major * math.cos(theta), r_minor * math.sin(theta), z))
    face_counts, indices = [], []
    for i in range(segments):
        i_next = (i + 1) % segments
        indices.extend([i, i_next, i_next + segments, i + segments])
        face_counts.append(4)
    mesh.GetPointsAttr().Set(pts)
    mesh.GetFaceVertexCountsAttr().Set(face_counts)
    mesh.GetFaceVertexIndicesAttr().Set(indices)
    mesh.GetDisplayColorAttr().Set([color])


def _mk_cylinder(stage, path, radius, h, color, segments=16):
    _mk_oval(stage, path, radius, radius, h, color, segments)


COLE_SHAPES = ["rectangle", "square", "trapezoid", "sphere", "diamond", "oval", "cylinder"]
COLOR_MOVEABLE = Gf.Vec3f(1.0, 0.55, 0.0)
COLOR_NONMOVE = Gf.Vec3f(0.27, 0.51, 0.71)
COLOR_SMALL = Gf.Vec3f(0.4, 0.4, 0.4)
SPOT_MASS_KG = 32.7
OBS_MOVEABLE_MAX = SPOT_MASS_KG * 0.25
OBS_MIN_MASS = 0.0909
OBS_MAX_MASS = 2 * SPOT_MASS_KG
OBS_MIN_FOOT = 0.0174
OBS_MAX_FOOT = 1.1 * 0.5


def _spawn_cole_obstacle(stage, rng, idx, placed, is_moveable, start_buffer=4.0,
                          margin=1.5, clearance=2.0):
    shape = rng.choice(COLE_SHAPES)
    for _ in range(100):
        pos = random_inside_arena(margin=margin, rng=rng)
        if distance_2d(pos, [0, 0]) < start_buffer:
            continue
        if any(distance_2d(pos, p) < clearance for p in placed):
            continue
        break
    else:
        return None

    if is_moveable:
        mass = float(rng.uniform(OBS_MIN_MASS, OBS_MOVEABLE_MAX))
        color = COLOR_MOVEABLE
        friction = 0.4 if shape in ["sphere", "cylinder", "oval"] else 0.5
    else:
        mass = float(rng.uniform(OBS_MOVEABLE_MAX + 0.1, OBS_MAX_MASS))
        color = COLOR_NONMOVE
        friction = 0.9

    path = f"/World/ColeObstacles/Obst_{idx:03d}"
    rot_deg = float(rng.uniform(0, 360))
    if shape == "rectangle":
        w = float(rng.uniform(0.15, 1.1)); d = float(rng.uniform(0.15, 0.7))
        foot = w * d
        if foot < OBS_MIN_FOOT:
            s = math.sqrt(OBS_MIN_FOOT / foot); w *= s; d *= s
        elif foot > OBS_MAX_FOOT:
            s = math.sqrt(OBS_MAX_FOOT / foot); w *= s; d *= s
        h = float(rng.uniform(0.3, 1.2))
        _mk_rect(stage, path, w, d, h, color); dims = (w, d, h)
    elif shape == "square":
        side = float(rng.uniform(math.sqrt(OBS_MIN_FOOT), math.sqrt(OBS_MAX_FOOT)))
        h = float(rng.uniform(0.3, 1.2))
        _mk_rect(stage, path, side, side, h, color); dims = (side, side, h)
    elif shape == "trapezoid":
        w_bot = float(rng.uniform(0.2, 1.0)); w_top = float(rng.uniform(0.15, w_bot))
        d = float(rng.uniform(0.2, 0.7))
        avg = (w_bot + w_top) / 2; foot = avg * d
        if foot < OBS_MIN_FOOT:
            s = math.sqrt(OBS_MIN_FOOT / foot); w_bot *= s; w_top *= s; d *= s
        elif foot > OBS_MAX_FOOT:
            s = math.sqrt(OBS_MAX_FOOT / foot); w_bot *= s; w_top *= s; d *= s
        h = float(rng.uniform(0.4, 1.2))
        _mk_trapezoid(stage, path, w_bot, w_top, d, h, color)
        dims = ((w_bot + w_top) / 2, d, h)
    elif shape == "sphere":
        r = float(rng.uniform(math.sqrt(OBS_MIN_FOOT / math.pi), math.sqrt(OBS_MAX_FOOT / math.pi)))
        _mk_sphere(stage, path, r, color); dims = (2 * r, 2 * r, 2 * r)
    elif shape == "diamond":
        base = float(rng.uniform(math.sqrt(OBS_MIN_FOOT), math.sqrt(OBS_MAX_FOOT)))
        h = float(rng.uniform(0.5, 1.5))
        _mk_diamond(stage, path, base, h, color); dims = (base, base, h)
    elif shape == "oval":
        r_maj = float(rng.uniform(0.15, 0.6)); r_min = float(rng.uniform(0.1, r_maj))
        foot = math.pi * r_maj * r_min
        if foot < OBS_MIN_FOOT:
            s = math.sqrt(OBS_MIN_FOOT / foot); r_maj *= s; r_min *= s
        elif foot > OBS_MAX_FOOT:
            s = math.sqrt(OBS_MAX_FOOT / foot); r_maj *= s; r_min *= s
        h = float(rng.uniform(0.3, 1.0))
        _mk_oval(stage, path, r_maj, r_min, h, color); dims = (2 * r_maj, 2 * r_min, h)
    else:  # cylinder
        r = float(rng.uniform(math.sqrt(OBS_MIN_FOOT / math.pi), math.sqrt(OBS_MAX_FOOT / math.pi)))
        h = float(rng.uniform(0.4, 1.5))
        _mk_cylinder(stage, path, r, h, color); dims = (2 * r, 2 * r, h)

    prim = stage.GetPrimAtPath(path)
    xf = UsdGeom.Xformable(prim)
    z = dims[0] / 2 if shape == "sphere" else 0.01
    xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(z)))
    xf.AddRotateXYZOp().Set(Gf.Vec3d(0, 0, rot_deg))
    _apply_rigid_body_physics(stage, path, mass, friction)
    rigid = UsdPhysics.RigidBodyAPI.Get(stage, path)
    if rigid and not is_moveable:
        rigid.CreateKinematicEnabledAttr(True)

    footprint_r = max(dims[0], dims[1]) / 2
    foot_area = {
        "rectangle": dims[0] * dims[1],
        "square": dims[0] * dims[1],
        "trapezoid": dims[0] * dims[1],
        "sphere": math.pi * (dims[0] / 2) ** 2,
        "cylinder": math.pi * (dims[0] / 2) ** 2,
        "oval": math.pi * (dims[0] / 2) * (dims[1] / 2),
        "diamond": (dims[0] ** 2) / 2,
    }[shape]
    placed.append(pos)
    return {"pos": (float(pos[0]), float(pos[1])), "radius": float(footprint_r),
            "area": foot_area, "moveable": bool(is_moveable), "shape": shape}


def _spawn_cole_small(stage, rng, idx, main_placed, small_placed, start_buffer=4.0,
                       margin=1.5, clearance=0.3):
    shape = rng.choice(COLE_SHAPES)
    size = float(rng.uniform(0.043, 0.102))
    for _ in range(50):
        pos = random_inside_arena(margin=margin, rng=rng)
        if distance_2d(pos, [0, 0]) < start_buffer:
            continue
        if any(distance_2d(pos, p) < clearance for p in main_placed):
            continue
        if any(distance_2d(pos, p) < clearance for p in small_placed):
            continue
        break
    else:
        return None

    path = f"/World/ColeSmall/Small_{idx:03d}"
    rot_deg = float(rng.uniform(0, 360))
    if shape == "sphere":
        _mk_sphere(stage, path, size, COLOR_SMALL); dims = (size * 2, size * 2, size * 2)
    elif shape == "cylinder":
        _mk_cylinder(stage, path, size / 2, size, COLOR_SMALL); dims = (size, size, size)
    elif shape == "square":
        _mk_rect(stage, path, size, size, size, COLOR_SMALL); dims = (size, size, size)
    elif shape == "rectangle":
        d = size * float(rng.uniform(0.6, 1.4))
        _mk_rect(stage, path, size, d, size, COLOR_SMALL); dims = (size, d, size)
    elif shape == "diamond":
        _mk_diamond(stage, path, size, size, COLOR_SMALL); dims = (size, size, size)
    elif shape == "trapezoid":
        _mk_trapezoid(stage, path, size, size * 0.7, size, size, COLOR_SMALL)
        dims = ((size + size * 0.7) / 2, size, size)
    else:  # oval
        r_min = size * float(rng.uniform(0.6, 1.4))
        _mk_oval(stage, path, size, r_min, size * 0.8, COLOR_SMALL)
        dims = (size * 2, r_min * 2, size * 0.8)

    prim = stage.GetPrimAtPath(path)
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(size / 2)))
    xf.AddRotateXYZOp().Set(Gf.Vec3d(0, 0, rot_deg))
    _apply_rigid_body_physics(stage, path, 1000.0, friction=0.9)
    rigid = UsdPhysics.RigidBodyAPI.Get(stage, path)
    if rigid:
        rigid.CreateKinematicEnabledAttr(True)
    foot_area = {
        "rectangle": dims[0] * dims[1], "square": dims[0] * dims[1],
        "trapezoid": dims[0] * dims[1],
        "sphere": math.pi * (dims[0] / 2) ** 2,
        "cylinder": math.pi * (dims[0] / 2) ** 2,
        "oval": math.pi * (dims[0] / 2) * (dims[1] / 2),
        "diamond": (dims[0] ** 2) / 2,
    }[shape]
    small_placed.append(pos)
    return {"pos": (float(pos[0]), float(pos[1])), "radius": float(max(dims[0], dims[1]) / 2),
            "area": foot_area, "moveable": False, "shape": shape, "small": True}


def build_cole_arena(stage, rng, moveable_pct=3.0, nonmoveable_pct=3.0, small_pct=0.3,
                      start_buffer=4.0):
    """Cole's richer arena: moveable + non-moveable + small static, 7 shapes."""
    ground_path = "/World/ground"
    UsdGeom.Cylinder.Define(stage, ground_path)
    gp = stage.GetPrimAtPath(ground_path)
    gp.GetAttribute("radius").Set(float(ARENA_RADIUS + 5.0))
    gp.GetAttribute("height").Set(0.1)
    xf = UsdGeom.Xformable(gp); xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
    UsdPhysics.CollisionAPI.Apply(gp)
    mat = UsdPhysics.MaterialAPI.Apply(gp)
    mat.CreateStaticFrictionAttr(0.8); mat.CreateDynamicFrictionAttr(0.6)

    light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    light.CreateIntensityAttr(1500)

    arena_area = math.pi * (ARENA_RADIUS - 1.5) ** 2
    mv_target = arena_area * moveable_pct / 100.0
    nm_target = arena_area * nonmoveable_pct / 100.0
    sm_target = arena_area * small_pct / 100.0

    obstacles = []
    placed_positions = []
    small_positions = []
    idx = 0

    mv_area = 0.0
    n_moveable = 0
    while mv_area < mv_target and idx < 300:
        obs = _spawn_cole_obstacle(stage, rng, idx, placed_positions,
                                    is_moveable=True, start_buffer=start_buffer)
        idx += 1
        if obs is not None:
            obstacles.append(obs)
            mv_area += obs["area"]
            n_moveable += 1

    nm_area = 0.0
    n_nonmove = 0
    while nm_area < nm_target and idx < 600:
        obs = _spawn_cole_obstacle(stage, rng, idx, placed_positions,
                                    is_moveable=False, start_buffer=start_buffer,
                                    clearance=1.5)
        idx += 1
        if obs is not None:
            obstacles.append(obs)
            nm_area += obs["area"]
            n_nonmove += 1

    sm_area = 0.0
    n_small = 0
    sidx = 0
    while sm_area < sm_target and sidx < 200:
        obs = _spawn_cole_small(stage, rng, sidx, placed_positions, small_positions,
                                 start_buffer=start_buffer)
        sidx += 1
        if obs is not None:
            obstacles.append(obs)
            sm_area += obs["area"]
            n_small += 1

    total_cov = (mv_area + nm_area + sm_area) / arena_area * 100.0
    print(f"[COLE ARENA] moveable={n_moveable} ({mv_area:.1f} m²) "
          f"non-moveable={n_nonmove} ({nm_area:.1f} m²) "
          f"small-static={n_small} ({sm_area:.1f} m²) "
          f"total={total_cov:.1f}% coverage")
    return obstacles


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
    if args.no_obstacles:
        obstacles_info = build_arena(stage, rng, num_obstacles=0)
        apf_list = []
    elif args.cole_arena:
        cole_obs = build_cole_arena(
            stage, rng,
            moveable_pct=args.moveable_pct,
            nonmoveable_pct=args.nonmoveable_pct,
            small_pct=args.small_static_pct,
            start_buffer=args.start_buffer,
        )
        obstacles_info = cole_obs
        apf_list = [(o["pos"][0], o["pos"][1], 2.0 * o["radius"]) for o in cole_obs]
    else:
        obstacles_info = build_arena(stage, rng, num_obstacles=args.num_obstacles)
        apf_list = list(obstacles_info)

    flat_policy = SpotFlatTerrainPolicy(prim_path="/World/Spot", name="Spot", position=SPOT_START)
    world.reset()
    world.step(render=False)
    if not headless:
        world.render()
    flat_policy.initialize()
    flat_policy.post_reset()

    if USE_ROUGH:
        rough_policy_kwargs = dict(
            flat_policy=flat_policy,
            checkpoint_path=args.loco_checkpoint,
            mason_baseline=True,
        )
        if args.loco_action_scale is not None:
            rough_policy_kwargs["action_scale"] = args.loco_action_scale
        if not args.rough_heightscan:
            rough_policy_kwargs["ground_height_fn"] = lambda x: 0.0
        robot_policy = SpotRoughTerrainPolicy(**rough_policy_kwargs)
        robot_policy.initialize()
        robot_policy.apply_gains()
        robot_policy._decimation = args.loco_decimation
        step_loco = lambda cmd: robot_policy.forward(PHYSICS_DT, list(cmd))
        hs_mode = "PhysX raycast" if args.rough_heightscan else "flat analytical"
        nn_hz = 500 / args.loco_decimation
        print(f"[OK] Rough policy engaged ({os.path.basename(args.loco_checkpoint)}, mason obs, "
              f"{hs_mode} heightscan, NN@{nn_hz:.0f}Hz)")
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
        reach_threshold=args.waypoint_reach,
        kp_lin=args.kp_lin,
        kp_ang=args.kp_ang,
        max_lin_speed=args.max_lin_speed,
        yaw_first_vx=args.yaw_first_vx,
        obstacles=apf_list,
        obstacle_influence_radius=args.apf_radius,
        obstacle_repulse_gain=args.apf_gain,
        obstacle_tangent_bias=args.apf_tangent,
        robot_radius=0.45,
        stuck_window=args.stuck_window,
        stuck_dist_threshold=args.stuck_dist,
        escape_duration=args.escape_duration,
        escape_vx=args.escape_vx,
        escape_wz=args.escape_wz,
        skip_wp_after_escapes=args.skip_wp_after_escapes,
    )
    print(f"[NAV] APF obstacles: {len(apf_list)}  R={args.apf_radius}m  gain={args.apf_gain}  tangent={args.apf_tangent}  max_spd={args.max_lin_speed}")
    print(f"[NAV] Stuck-escape: window={args.stuck_window} dist={args.stuck_dist}m dur={args.escape_duration} vx={args.escape_vx} wz={args.escape_wz}")

    all_results = []
    planner_bounds = (-ARENA_RADIUS, ARENA_RADIUS)
    for ep in range(1, args.episodes + 1):
        waypoints = generate_waypoints(rng)
        original_majors = list(waypoints)  # snapshot of the 25 ground-truth WPs for replan

        tracker = None
        depth_sensor = None
        use_online = args.online_map or args.depth_sensor
        if args.depth_sensor and apf_list:
            depth_sensor = DepthRaycastObstacleDetector(
                robot_prim_path="/World/Spot",
                fov_horiz_deg=args.sensor_fov_h,
                fov_vert_deg=args.sensor_fov_v,
                n_rays_h=args.sensor_rays_h,
                n_rays_v=args.sensor_rays_v,
                max_distance=args.sensor_max_dist,
                grid_res=args.sensor_grid_res,
                grid_bounds=planner_bounds,
                min_hits_per_cell=args.sensor_min_hits,
            )
            navigator.set_obstacles([])
            print(f"  [DEPTH SENSOR] fov={args.sensor_fov_h}x{args.sensor_fov_v}deg  "
                  f"rays={args.sensor_rays_h}x{args.sensor_rays_v}  max={args.sensor_max_dist}m  "
                  f"grid_res={args.sensor_grid_res}m  min_hits={args.sensor_min_hits}")
            print(f"  [DEPTH SENSOR] ground-truth obstacles hidden: {len(apf_list)} (detector will discover by raycast)")
        elif args.online_map and apf_list:
            tracker = OnlineObstacleTracker(apf_list, sense_radius=args.sense_radius)
            navigator.set_obstacles([])
            print(f"  [ONLINE MAP] sense_radius={args.sense_radius}m replan_period={args.replan_period_sec}s "
                  f"(ground-truth obstacles hidden: {tracker.n_total})")
        else:
            navigator.set_obstacles(apf_list)

        if args.global_planner and apf_list and not args.online_map and not args.depth_sensor:
            expanded, n_planned, n_fallback = astar_expand_waypoints(
                waypoints,
                robot_xy=(SPOT_START[0], SPOT_START[1]),
                obstacles=apf_list,
                bounds=planner_bounds,
                grid_res=args.planner_res,
                inflate=args.planner_inflate,
                subwp_step=args.planner_subwp_step,
            )
            print(f"  [PLANNER] A* expanded {len(waypoints)} WPs -> "
                  f"{len(expanded)} sub-WPs (planned={n_planned}, fallback={n_fallback})")
            waypoints = expanded
        navigator.reset(waypoints=waypoints)
        last_replan_time = 0.0
        pending_replan = False
        major_reached_labels = set()

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
        # Zero all velocities so the rough policy doesn't see a falling/drifting body at step 0.
        # Without this, lin_vel_b can carry -9 m/s from pre-reset simulator state, which is OOD
        # for Boulder V6 → saturated actions → immediate collapse.
        try:
            flat_policy.robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
            flat_policy.robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
            flat_policy.robot.set_joint_velocities(np.zeros(12, dtype=np.float32))
        except Exception as _e:
            print(f"  [WARN] Could not zero velocities: {_e}")
        if USE_ROUGH:
            robot_policy.post_reset()

        print(f"\n--- Episode {ep}/{args.episodes} ---")
        print(f"  WP-A at ({waypoints[0]['pos'][0]:.1f}, {waypoints[0]['pos'][1]:.1f})")

        print(f"  Stabilizing ({args.stab_steps} steps, stab_vx={args.stab_vx})...", flush=True)
        stab_cmd = [float(args.stab_vx), 0.0, 0.0]
        for i in range(args.stab_steps):
            try:
                step_loco(stab_cmd)
                world.step(render=False)
                if not headless and (i % RENDER_EVERY_N_STEPS == 0):
                    world.render()
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
                world.step(render=False)
                if not headless and (step_count % RENDER_EVERY_N_STEPS == 0):
                    world.render()
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
                    reached_label = navigator.waypoints[navigator.wp_idx - 1]["label"]
                    is_major = "." not in str(reached_label)
                    if is_major:
                        wp_reached += 1
                        score += WAYPOINT_REWARD
                        major_reached_labels.add(reached_label)
                        print(f"  [WP {reached_label}] reached ({wp_reached}/{WAYPOINT_COUNT})")
                    if navigator.done and tracker is None and depth_sensor is None:
                        status = "COMPLETE"
                        break
                    if navigator.done and (tracker is not None or depth_sensor is not None):
                        # All currently-planned sub-WPs done but majors may remain.
                        remaining = [w for w in original_majors if w["label"] not in major_reached_labels]
                        if not remaining:
                            status = "COMPLETE"
                            break

                if depth_sensor is not None:
                    grew = depth_sensor.sense(cur_pos, spot_yaw, spot_z)
                    if grew:
                        pending_replan = True
                    need_replan = pending_replan and (sim_time - last_replan_time) >= args.replan_period_sec
                    if need_replan:
                        known = depth_sensor.known_obstacles()
                        navigator.set_obstacles(known)
                        remaining = [w for w in original_majors if w["label"] not in major_reached_labels]
                        if remaining:
                            expanded, n_planned, n_fallback = astar_expand_waypoints(
                                remaining,
                                robot_xy=(float(cur_pos[0]), float(cur_pos[1])),
                                obstacles=known,
                                bounds=planner_bounds,
                                grid_res=args.planner_res,
                                inflate=args.planner_inflate,
                                subwp_step=args.planner_subwp_step,
                            )
                            navigator.replace_waypoints(expanded)
                            print(f"    [SENSOR] known={depth_sensor.n_known}/{depth_sensor.n_total_cells} cells "
                                  f"replan -> {len(expanded)} sub-WPs "
                                  f"(planned={n_planned}, fallback={n_fallback})")
                        last_replan_time = sim_time
                        pending_replan = False
                elif tracker is not None:
                    grew = tracker.sense(cur_pos)
                    if grew:
                        pending_replan = True
                    need_replan = pending_replan and (sim_time - last_replan_time) >= args.replan_period_sec
                    if need_replan:
                        known = tracker.known_obstacles()
                        navigator.set_obstacles(known)
                        remaining = [w for w in original_majors if w["label"] not in major_reached_labels]
                        if remaining:
                            expanded, n_planned, n_fallback = astar_expand_waypoints(
                                remaining,
                                robot_xy=(float(cur_pos[0]), float(cur_pos[1])),
                                obstacles=known,
                                bounds=planner_bounds,
                                grid_res=args.planner_res,
                                inflate=args.planner_inflate,
                                subwp_step=args.planner_subwp_step,
                            )
                            navigator.replace_waypoints(expanded)
                            print(f"    [MAP] known={tracker.n_known}/{tracker.n_total} "
                                  f"replan -> {len(expanded)} sub-WPs "
                                  f"(planned={n_planned}, fallback={n_fallback})")
                        last_replan_time = sim_time
                        pending_replan = False

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
