"""
Testing Environment — MPPI Navigation (No Neural Network Required)
===================================================================
Identical arena to RL_FOLDER_VS3/Testing_Environment.py but with the
NavigationPolicy + FSM stack replaced by a pure MPPI planner.

• No checkpoint needed — just run and compare waypoints vs. RL baseline.
• MPPI runs at 20 Hz (same decimation as RL policy).
• Obstacle data fed directly into MPPI rollouts — no raycasting needed.

Usage
-----
    python Testing_Environment_MPPI.py --episodes 3
    python Testing_Environment_MPPI.py --headless --episodes 10
    python Testing_Environment_MPPI.py --horizon 30 --num_samples 1024 --temperature 0.03

Author : Cole (MS for Autonomy Project)
Date   : 2026
"""

import csv
import math
import os
import string
import argparse
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing BEFORE SimulationApp
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Testing Environment — MPPI Navigation")
parser.add_argument("--headless",     action="store_true",  help="Run without GUI")
parser.add_argument("--episodes",     type=int,   default=1,    help="Number of episodes")
parser.add_argument("--seed",         type=int,   default=None, help="Global RNG seed (None = random)")
parser.add_argument("--horizon",      type=int,   default=25,   help="MPPI lookahead steps (default 25 → 1.25 s)")
parser.add_argument("--num_samples",  type=int,   default=512,  help="MPPI trajectory samples K (default 512)")
parser.add_argument("--temperature",  type=float, default=0.05, help="MPPI temperature lambda (default 0.05)")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Isaac Sim boot — MUST happen before any other Isaac imports
# ─────────────────────────────────────────────────────────────────────────────
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import omni
from omni.isaac.core import World
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
from omni.isaac.sensor import Camera
from pxr import UsdGeom, UsdLux, UsdPhysics, Gf, Sdf

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mppi_navigator import MPPINavigator  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Arena
ARENA_RADIUS         = 25.0          # meters (diameter = 50 m)
ARENA_CENTER_X       = 0.0
ARENA_CENTER_Y       = 0.0

# Spot
SPOT_START_X         = 0.0
SPOT_START_Y         = 0.0
SPOT_START_Z         = 0.7
SPOT_MASS_KG         = 32.7
SPOT_LENGTH          = 1.1
SPOT_WIDTH           = 0.5
SPOT_HEIGHT          = 0.6

FALL_HEIGHT_THRESHOLD = 0.25

# Speed
SPOT_MAX_SPEED_MPS   = 5 * 0.44704
SPOT_MIN_SPEED_MPS   = 0.3
OBSTACLE_SLOW_RADIUS = 2.0

# Navigation Control
HEADING_ALIGNMENT_THRESHOLD = math.radians(10)
HEADING_CORRECTION_GAIN = 0.5
MAX_TURN_RATE = 1.0
FORWARD_PRIORITY = True

# Obstacles
OBSTACLE_AREA_FRAC   = 0.10
OBSTACLE_MIN_FOOT    = 0.0174
OBSTACLE_MAX_FOOT    = SPOT_LENGTH * SPOT_WIDTH

OBSTACLE_MOVEABLE_MAX = SPOT_MASS_KG * 0.25
OBSTACLE_MIN_MASS     = 0.0909
OBSTACLE_MAX_MASS     = 2 * SPOT_MASS_KG
OBSTACLE_CLEARANCE_BOUNDARY = 3.0
OBSTACLE_CLEARANCE_WAYPOINT = 5.0
STARTING_ZONE_BUFFER = 2.0

# Small Static Obstacles
SMALL_OBSTACLE_MIN_SIZE = 0.043
SMALL_OBSTACLE_MAX_SIZE = 0.102
SMALL_OBSTACLE_COVERAGE = 0.10 / 3
SMALL_OBSTACLE_CLEARANCE = 0.3
COLOR_SMALL_OBSTACLE = Gf.Vec3f(0.4, 0.4, 0.4)

# Waypoints
WAYPOINT_COUNT       = 25
WAYPOINT_LABELS      = list(string.ascii_uppercase[:WAYPOINT_COUNT])
WAYPOINT_DIST_A      = 20.0
WAYPOINT_SPACING_BZ  = 40.0
WAYPOINT_REACH_DIST  = 1.0
WAYPOINT_BOUNDARY_MARGIN = 2.0

# Colors (RGB)
COLOR_MOVEABLE_OBSTACLE     = Gf.Vec3f(1.0,  0.55, 0.0)
COLOR_NON_MOVEABLE_OBSTACLE = Gf.Vec3f(0.27, 0.51, 0.71)
COLOR_WAYPOINT              = Gf.Vec3f(1.0,  0.95, 0.0)
COLOR_START_WAYPOINT        = Gf.Vec3f(0.2,  0.9,  0.2)
COLOR_FLAG_POLE             = Gf.Vec3f(0.88, 0.88, 0.88)

# Waypoint flag geometry
WP_POLE_RADIUS  = 0.05
WP_POLE_HEIGHT  = 2.5
WP_FLAG_WIDTH   = 0.7
WP_FLAG_DEPTH   = 0.06
WP_FLAG_HEIGHT  = 0.40

# Physics
PHYSICS_DT   = 1.0 / 500.0
RENDERING_DT = 10.0 / 500.0

# Reward / Scoring
EPISODE_START_SCORE = 300.0
TIME_DECAY_PER_SEC  = 1.0
WAYPOINT_REWARD     = 15.0

# Misc
ARENA_AREA           = math.pi * ARENA_RADIUS ** 2
TARGET_OBSTACLE_AREA = ARENA_AREA * OBSTACLE_AREA_FRAC

# CSV logging
CSV_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MPPI_CSV.csv")
CSV_HEADERS  = ["Episode", "Waypoints_Reached", "Failure_Reason", "Final_Score"]

# Failure reason mappings
FAILURE_FELL_OVER = "Fell Over"
FAILURE_RAN_OUT   = "Ran Out of Points"
FAILURE_COMPLETED = "Completed All Waypoints"
FAILURE_OTHER     = "Other"

print("=" * 72)
print("MPPI TESTING ENVIRONMENT — CIRCULAR WAYPOINT NAVIGATION ARENA")
print(f"  Arena radius : {ARENA_RADIUS} m  (diameter {ARENA_RADIUS * 2} m)")
print(f"  Arena area   : {ARENA_AREA:.1f} m²")
print(f"  Waypoints    : {WAYPOINT_COUNT}  (labels {WAYPOINT_LABELS[0]}–{WAYPOINT_LABELS[-1]})")
print(f"  MPPI horizon : {args.horizon} steps  ({args.horizon * 0.05:.2f} s lookahead)")
print(f"  MPPI samples : {args.num_samples}")
print(f"  MPPI lambda  : {args.temperature}")
print("=" * 72)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — geometry utilities
# ─────────────────────────────────────────────────────────────────────────────

def inside_arena(x: float, y: float, radius: float = ARENA_RADIUS,
                 margin: float = 0.0) -> bool:
    """Return True if (x, y) is strictly inside the arena circle minus margin."""
    return (x - ARENA_CENTER_X) ** 2 + (y - ARENA_CENTER_Y) ** 2 < (radius - margin) ** 2


def random_inside_arena(margin: float = 0.0, rng: np.random.Generator = None) -> np.ndarray:
    """Sample a uniform random position inside the arena circle."""
    if rng is None:
        rng = np.random.default_rng()
    r_limit = ARENA_RADIUS - margin
    while True:
        x = rng.uniform(-r_limit, r_limit)
        y = rng.uniform(-r_limit, r_limit)
        if x ** 2 + y ** 2 < r_limit ** 2:
            return np.array([x, y])


def distance_2d(a, b) -> float:
    """Euclidean distance between two 2D points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def quaternion_to_yaw(quat: np.ndarray) -> float:
    """
    Convert quaternion [w, x, y, z] to yaw angle (rotation around z-axis).
    Returns angle in radians [-π, π].
    """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-π, π]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def apply_rigid_body_physics(stage, prim_path: str, mass_kg: float,
                              friction: float = 0.5) -> None:
    """Add RigidBodyAPI, CollisionAPI, and MassAPI to a USD prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return

    rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_api.CreateRigidBodyEnabledAttr(True)
    UsdPhysics.CollisionAPI.Apply(prim)

    if prim.GetTypeName() == "Mesh":
        mesh_coll = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_coll.CreateApproximationAttr("convexHull")

    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass_kg)

    physics_mat = UsdPhysics.MaterialAPI.Apply(prim)
    physics_mat.CreateStaticFrictionAttr(friction)
    physics_mat.CreateDynamicFrictionAttr(friction * 0.8)
    physics_mat.CreateRestitutionAttr(0.05)


# ─────────────────────────────────────────────────────────────────────────────
# OBSTACLES — shape creators (7 types)
# ─────────────────────────────────────────────────────────────────────────────

def create_rectangle_mesh(stage, path: str, w: float, d: float, h: float, color) -> None:
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


def create_square_mesh(stage, path: str, side: float, height: float, color) -> None:
    create_rectangle_mesh(stage, path, side, side, height, color)


def create_trapezoid_mesh(stage, path: str, w_bot: float, w_top: float,
                          depth: float, height: float, color) -> None:
    mesh = UsdGeom.Mesh.Define(stage, path)
    hb = w_bot / 2
    ht = w_top / 2
    hd = depth / 2
    pts = [
        Gf.Vec3f(-hb, -hd, 0), Gf.Vec3f(hb, -hd, 0), Gf.Vec3f(hb, hd, 0), Gf.Vec3f(-hb, hd, 0),
        Gf.Vec3f(-ht, -hd, height), Gf.Vec3f(ht, -hd, height),
        Gf.Vec3f(ht, hd, height), Gf.Vec3f(-ht, hd, height),
    ]
    mesh.GetPointsAttr().Set(pts)
    mesh.GetFaceVertexCountsAttr().Set([4, 4, 4, 4, 4, 4, 4, 4])
    mesh.GetFaceVertexIndicesAttr().Set([
        0, 1, 2, 3, 4, 7, 6, 5, 0, 4, 5, 1,
        2, 6, 7, 3, 0, 3, 7, 4, 1, 5, 6, 2,
    ])
    mesh.GetDisplayColorAttr().Set([color])


def create_sphere_mesh(stage, path: str, radius: float, color, segments: int = 16) -> None:
    mesh = UsdGeom.Mesh.Define(stage, path)
    pts = [Gf.Vec3f(0, 0, radius)]
    for i in range(1, segments):
        phi = math.pi * i / segments
        rho = radius * math.sin(phi)
        z = radius * math.cos(phi)
        for j in range(segments):
            theta = 2 * math.pi * j / segments
            x = rho * math.cos(theta)
            y = rho * math.sin(theta)
            pts.append(Gf.Vec3f(x, y, z))
    pts.append(Gf.Vec3f(0, 0, -radius))

    face_counts = []
    indices = []
    for j in range(segments):
        indices.extend([0, 1 + j, 1 + (j + 1) % segments])
        face_counts.append(3)
    for i in range(1, segments - 1):
        for j in range(segments):
            a = 1 + (i - 1) * segments + j
            b = 1 + (i - 1) * segments + (j + 1) % segments
            c = 1 + i * segments + (j + 1) % segments
            d = 1 + i * segments + j
            indices.extend([a, b, c, d])
            face_counts.append(4)
    top_idx = len(pts) - 1
    for j in range(segments):
        a = 1 + (segments - 2) * segments + j
        b = 1 + (segments - 2) * segments + (j + 1) % segments
        indices.extend([top_idx, b, a])
        face_counts.append(3)

    mesh.GetPointsAttr().Set(pts)
    mesh.GetFaceVertexCountsAttr().Set(face_counts)
    mesh.GetFaceVertexIndicesAttr().Set(indices)
    mesh.GetDisplayColorAttr().Set([color])


def create_diamond_mesh(stage, path: str, base_width: float, height: float, color) -> None:
    mesh = UsdGeom.Mesh.Define(stage, path)
    hw = base_width / 2
    pts = [
        Gf.Vec3f(-hw, -hw, 0), Gf.Vec3f(hw, -hw, 0),
        Gf.Vec3f(hw, hw, 0), Gf.Vec3f(-hw, hw, 0),
        Gf.Vec3f(0, 0, height),
    ]
    mesh.GetPointsAttr().Set(pts)
    mesh.GetFaceVertexCountsAttr().Set([4, 3, 3, 3, 3])
    mesh.GetFaceVertexIndicesAttr().Set([
        0, 1, 2, 3,
        0, 4, 1,
        1, 4, 2,
        2, 4, 3,
        3, 4, 0,
    ])
    mesh.GetDisplayColorAttr().Set([color])


def create_oval_mesh(stage, path: str, r_major: float, r_minor: float,
                     height: float, color, segments: int = 16) -> None:
    mesh = UsdGeom.Mesh.Define(stage, path)
    pts = []
    for z in [0, height]:
        for i in range(segments):
            theta = 2 * math.pi * i / segments
            x = r_major * math.cos(theta)
            y = r_minor * math.sin(theta)
            pts.append(Gf.Vec3f(x, y, z))

    face_counts = []
    indices = []
    for i in range(segments):
        i_next = (i + 1) % segments
        a = i
        b = i_next
        c = i_next + segments
        d = i + segments
        indices.extend([a, b, c, d])
        face_counts.append(4)

    mesh.GetPointsAttr().Set(pts)
    mesh.GetFaceVertexCountsAttr().Set(face_counts)
    mesh.GetFaceVertexIndicesAttr().Set(indices)
    mesh.GetDisplayColorAttr().Set([color])


def create_cylinder_mesh(stage, path: str, radius: float, height: float,
                         color, segments: int = 16) -> None:
    create_oval_mesh(stage, path, radius, radius, height, color, segments)


# ─────────────────────────────────────────────────────────────────────────────
# OBSTACLE MANAGER — spawning and tracking
# ─────────────────────────────────────────────────────────────────────────────

class ObstacleManager:
    """Manages random obstacle generation and collision tracking."""

    SHAPES = ["rectangle", "square", "trapezoid", "sphere", "diamond", "oval", "cylinder"]

    def __init__(self, stage, rng: np.random.Generator):
        self.stage = stage
        self.rng = rng
        self.obstacles = []
        self.small_obstacles = []

    def calculate_footprint_area(self, shape: str, dims: tuple) -> float:
        if shape == "rectangle":
            return dims[0] * dims[1]
        elif shape == "square":
            return dims[0] * dims[1]
        elif shape == "trapezoid":
            return dims[0] * dims[1]
        elif shape == "sphere":
            r = dims[0] / 2
            return math.pi * r * r
        elif shape == "oval":
            r_major = dims[0] / 2
            r_minor = dims[1] / 2
            return math.pi * r_major * r_minor
        elif shape == "cylinder":
            r = dims[0] / 2
            return math.pi * r * r
        elif shape == "diamond":
            return dims[0] * dims[1] / 2
        else:
            return dims[0] * dims[1]

    def spawn_one(self, idx: int, margin: float = 1.5, min_spawn_clearance: float = 2.0,
                  force_weight_class: str = None) -> None:
        shape = self.rng.choice(self.SHAPES)

        max_attempts = 100
        for attempt in range(max_attempts):
            pos_2d = random_inside_arena(margin=margin, rng=self.rng)

            if distance_2d(pos_2d, [SPOT_START_X, SPOT_START_Y]) < STARTING_ZONE_BUFFER:
                continue

            too_close = False
            for obs in self.obstacles:
                if distance_2d(pos_2d, obs["pos"]) < min_spawn_clearance:
                    too_close = True
                    break

            if not too_close:
                break
        else:
            return

        mass = self.rng.uniform(OBSTACLE_MIN_MASS, OBSTACLE_MAX_MASS)

        if force_weight_class == "moveable":
            mass = self.rng.uniform(OBSTACLE_MIN_MASS, OBSTACLE_MOVEABLE_MAX)
            weight_class = "moveable"
            color = COLOR_MOVEABLE_OBSTACLE
        elif force_weight_class == "non_moveable":
            mass = self.rng.uniform(OBSTACLE_MOVEABLE_MAX + 0.1, OBSTACLE_MAX_MASS)
            weight_class = "non_moveable"
            color = COLOR_NON_MOVEABLE_OBSTACLE
        else:
            if mass <= OBSTACLE_MOVEABLE_MAX:
                weight_class = "moveable"
                color = COLOR_MOVEABLE_OBSTACLE
            else:
                weight_class = "non_moveable"
                color = COLOR_NON_MOVEABLE_OBSTACLE

        path = f"/World/Obstacles/Obst_{idx:03d}"
        rot_deg = self.rng.uniform(0, 360)

        if shape == "rectangle":
            w = self.rng.uniform(0.15, 1.1)
            d = self.rng.uniform(0.15, 0.7)
            footprint = w * d
            if footprint < OBSTACLE_MIN_FOOT:
                scale = math.sqrt(OBSTACLE_MIN_FOOT / footprint)
                w *= scale
                d *= scale
            elif footprint > OBSTACLE_MAX_FOOT:
                scale = math.sqrt(OBSTACLE_MAX_FOOT / footprint)
                w *= scale
                d *= scale
            h = self.rng.uniform(0.3, 1.2)
            create_rectangle_mesh(self.stage, path, w, d, h, color)
            dims = (w, d, h)

        elif shape == "square":
            min_side = math.sqrt(OBSTACLE_MIN_FOOT)
            max_side = math.sqrt(OBSTACLE_MAX_FOOT)
            side = self.rng.uniform(min_side, max_side)
            h = self.rng.uniform(0.3, 1.2)
            create_square_mesh(self.stage, path, side, h, color)
            dims = (side, side, h)

        elif shape == "trapezoid":
            w_bot = self.rng.uniform(0.2, 1.0)
            w_top = self.rng.uniform(0.15, w_bot)
            d = self.rng.uniform(0.2, 0.7)
            avg_w = (w_bot + w_top) / 2
            footprint = avg_w * d
            if footprint < OBSTACLE_MIN_FOOT:
                scale = math.sqrt(OBSTACLE_MIN_FOOT / footprint)
                w_bot *= scale
                w_top *= scale
                d *= scale
            elif footprint > OBSTACLE_MAX_FOOT:
                scale = math.sqrt(OBSTACLE_MAX_FOOT / footprint)
                w_bot *= scale
                w_top *= scale
                d *= scale
            h = self.rng.uniform(0.4, 1.2)
            create_trapezoid_mesh(self.stage, path, w_bot, w_top, d, h, color)
            dims = ((w_bot + w_top) / 2, d, h)

        elif shape == "sphere":
            min_r = math.sqrt(OBSTACLE_MIN_FOOT / math.pi)
            max_r = math.sqrt(OBSTACLE_MAX_FOOT / math.pi)
            r = self.rng.uniform(min_r, max_r)
            create_sphere_mesh(self.stage, path, r, color)
            dims = (2 * r, 2 * r, 2 * r)

        elif shape == "diamond":
            min_base = math.sqrt(OBSTACLE_MIN_FOOT)
            max_base = math.sqrt(OBSTACLE_MAX_FOOT)
            base = self.rng.uniform(min_base, max_base)
            h = self.rng.uniform(0.5, 1.5)
            create_diamond_mesh(self.stage, path, base, h, color)
            dims = (base, base, h)

        elif shape == "oval":
            r_major = self.rng.uniform(0.15, 0.6)
            r_minor = self.rng.uniform(0.1, r_major)
            footprint = math.pi * r_major * r_minor
            if footprint < OBSTACLE_MIN_FOOT:
                scale = math.sqrt(OBSTACLE_MIN_FOOT / footprint)
                r_major *= scale
                r_minor *= scale
            elif footprint > OBSTACLE_MAX_FOOT:
                scale = math.sqrt(OBSTACLE_MAX_FOOT / footprint)
                r_major *= scale
                r_minor *= scale
            h = self.rng.uniform(0.3, 1.0)
            create_oval_mesh(self.stage, path, r_major, r_minor, h, color)
            dims = (2 * r_major, 2 * r_minor, h)

        else:  # cylinder
            min_r = math.sqrt(OBSTACLE_MIN_FOOT / math.pi)
            max_r = math.sqrt(OBSTACLE_MAX_FOOT / math.pi)
            r = self.rng.uniform(min_r, max_r)
            h = self.rng.uniform(0.4, 1.5)
            create_cylinder_mesh(self.stage, path, r, h, color)
            dims = (2 * r, 2 * r, h)

        prim = self.stage.GetPrimAtPath(path)
        xform = UsdGeom.Xformable(prim)

        if shape == "sphere":
            z = dims[0] / 2
        else:
            z = 0.01

        xform.AddTranslateOp().Set(Gf.Vec3d(pos_2d[0], pos_2d[1], z))
        xform.AddRotateXYZOp().Set(Gf.Vec3d(0, 0, rot_deg))

        if weight_class == "moveable":
            friction = 0.4 if shape in ["sphere", "cylinder", "oval"] else 0.5
            apply_rigid_body_physics(self.stage, path, mass, friction)
            rigid = UsdPhysics.RigidBodyAPI.Get(self.stage, path)
            if rigid:
                rigid.CreateRigidBodyEnabledAttr(True)
        else:
            friction = 0.9
            apply_rigid_body_physics(self.stage, path, mass, friction)
            rigid = UsdPhysics.RigidBodyAPI.Get(self.stage, path)
            if rigid:
                rigid.CreateRigidBodyEnabledAttr(False)

        self.obstacles.append({
            "path": path,
            "pos": pos_2d,
            "dims": dims,
            "mass": mass,
            "weight_class": weight_class,
            "shape": shape
        })

    def populate(self, moveable_coverage_pct: float = 10.0, non_moveable_coverage_pct: float = 10.0,
                 min_spawn_clearance: float = 2.0) -> None:
        arena_area = math.pi * (ARENA_RADIUS - 1.5) ** 2

        moveable_target_area     = arena_area * moveable_coverage_pct / 100.0
        non_moveable_target_area = arena_area * non_moveable_coverage_pct / 100.0

        moveable_area     = 0.0
        non_moveable_area = 0.0
        idx = 0
        weight_counts = {"moveable": 0, "non_moveable": 0}

        print(f"[INFO] Spawning obstacles:")
        print(f"[INFO]   - Moveable target: {moveable_coverage_pct}% = {moveable_target_area:.1f} m²")
        print(f"[INFO]   - Non-moveable target: {non_moveable_coverage_pct}% = {non_moveable_target_area:.1f} m²")

        print(f"[INFO] Phase 1: Spawning ORANGE moveable obstacles...")
        phase1_idx = 0
        while moveable_area < moveable_target_area and phase1_idx < 1000:
            old_count = len(self.obstacles)
            self.spawn_one(idx, margin=1.5, min_spawn_clearance=2.0, force_weight_class="moveable")
            if len(self.obstacles) > old_count:
                obs = self.obstacles[-1]
                footprint_area = self.calculate_footprint_area(obs["shape"], obs["dims"])
                moveable_area += footprint_area
                weight_counts["moveable"] += 1
                print(f"[SPAWN] Moveable #{weight_counts['moveable']} ({obs['shape']}): area={footprint_area:.2f}m², total={moveable_area:.1f}m²")
            idx += 1
            phase1_idx += 1

        print(f"[INFO] Phase 1 complete: {weight_counts['moveable']} moveable, {moveable_area:.1f}m²")

        print(f"[INFO] Phase 2: Spawning BLUE non-moveable obstacles...")
        phase2_idx = 0
        while non_moveable_area < non_moveable_target_area and phase2_idx < 1000:
            old_count = len(self.obstacles)
            self.spawn_one(idx, margin=1.5, min_spawn_clearance=1.5, force_weight_class="non_moveable")
            if len(self.obstacles) > old_count:
                obs = self.obstacles[-1]
                footprint_area = self.calculate_footprint_area(obs["shape"], obs["dims"])
                non_moveable_area += footprint_area
                weight_counts["non_moveable"] += 1
                print(f"[SPAWN] Non-moveable #{weight_counts['non_moveable']} ({obs['shape']}): area={footprint_area:.2f}m², total={non_moveable_area:.1f}m²")
            idx += 1
            phase2_idx += 1

        total_area = moveable_area + non_moveable_area
        total_coverage_pct = (total_area / arena_area) * 100.0
        print(f"\n[OK] {len(self.obstacles)} obstacles spawned (total {total_area:.1f} m², {total_coverage_pct:.1f}%)")

    def nearest_obstacle_distance(self, x: float, y: float) -> float:
        if not self.obstacles:
            return 999.0
        return min(distance_2d([x, y], obs["pos"]) for obs in self.obstacles)

    def remove_prims(self) -> None:
        for obs in self.obstacles:
            prim = self.stage.GetPrimAtPath(obs["path"])
            if prim.IsValid():
                self.stage.RemovePrim(obs["path"])
        self.obstacles.clear()

        for obs in self.small_obstacles:
            prim = self.stage.GetPrimAtPath(obs["path"])
            if prim.IsValid():
                self.stage.RemovePrim(obs["path"])
        self.small_obstacles.clear()

    def spawn_small_static(self, target_coverage_pct: float = SMALL_OBSTACLE_COVERAGE * 100) -> None:
        arena_area = math.pi * (ARENA_RADIUS - 1.5) ** 2
        target_area = arena_area * target_coverage_pct / 100.0
        total_area = 0.0
        idx = 0
        shape_counts = {shape: 0 for shape in self.SHAPES}

        print(f"[INFO] Spawning small static obstacles (target {target_coverage_pct:.0f}% = {target_area:.1f} m²)")

        while total_area < target_area and idx < 500:
            shape = self.rng.choice(self.SHAPES)
            size = self.rng.uniform(SMALL_OBSTACLE_MIN_SIZE, SMALL_OBSTACLE_MAX_SIZE)

            max_attempts = 50
            pos_2d = None
            for attempt in range(max_attempts):
                test_pos = random_inside_arena(margin=1.5, rng=self.rng)

                if distance_2d(test_pos, [SPOT_START_X, SPOT_START_Y]) < STARTING_ZONE_BUFFER:
                    continue

                too_close = False
                for obs in self.obstacles:
                    if distance_2d(test_pos, obs["pos"]) < SMALL_OBSTACLE_CLEARANCE:
                        too_close = True
                        break

                if not too_close:
                    for small_obs in self.small_obstacles:
                        if distance_2d(test_pos, small_obs["pos"]) < SMALL_OBSTACLE_CLEARANCE:
                            too_close = True
                            break

                if not too_close:
                    pos_2d = test_pos
                    break

            if pos_2d is not None:
                path = f"/World/SmallObstacles/Small_{idx:03d}"
                rot_deg = self.rng.uniform(0, 360)

                if shape == "sphere":
                    create_sphere_mesh(self.stage, path, size, COLOR_SMALL_OBSTACLE)
                    dims = (size * 2, size * 2, size * 2)
                elif shape == "cylinder":
                    h = size
                    create_cylinder_mesh(self.stage, path, size / 2, h, COLOR_SMALL_OBSTACLE)
                    dims = (size, size, h)
                elif shape == "square":
                    create_square_mesh(self.stage, path, size, size, COLOR_SMALL_OBSTACLE)
                    dims = (size, size, size)
                elif shape == "rectangle":
                    w = size
                    d = size * self.rng.uniform(0.6, 1.4)
                    h = size
                    create_rectangle_mesh(self.stage, path, w, d, h, COLOR_SMALL_OBSTACLE)
                    dims = (w, d, h)
                elif shape == "diamond":
                    create_diamond_mesh(self.stage, path, size, size, COLOR_SMALL_OBSTACLE)
                    dims = (size, size, size)
                elif shape == "trapezoid":
                    w_bot = size
                    w_top = size * 0.7
                    d = size
                    h = size
                    create_trapezoid_mesh(self.stage, path, w_bot, w_top, d, h, COLOR_SMALL_OBSTACLE)
                    dims = ((w_bot + w_top) / 2, d, h)
                elif shape == "oval":
                    r_major = size
                    r_minor = size * self.rng.uniform(0.6, 1.4)
                    h = size * 0.8
                    create_oval_mesh(self.stage, path, r_major, r_minor, h, COLOR_SMALL_OBSTACLE)
                    dims = (r_major * 2, r_minor * 2, h)
                else:
                    dims = (size, size, size)

                footprint = self.calculate_footprint_area(shape, dims)

                prim = self.stage.GetPrimAtPath(path)
                if prim.IsValid():
                    xform = UsdGeom.Xformable(prim)
                    xform.ClearXformOpOrder()
                    xform.AddTranslateOp().Set(Gf.Vec3d(pos_2d[0], pos_2d[1], size / 2))
                    xform.AddRotateXYZOp().Set(Gf.Vec3d(0, 0, rot_deg))

                mass = 1000.0
                apply_rigid_body_physics(self.stage, path, mass, friction=0.9)
                rigid = UsdPhysics.RigidBodyAPI.Get(self.stage, path)
                if rigid:
                    rigid.CreateRigidBodyEnabledAttr(False)

                self.small_obstacles.append({
                    "path": path,
                    "pos": pos_2d,
                    "size": size,
                    "shape": shape
                })

                total_area += footprint
                shape_counts[shape] += 1
                idx += 1
            else:
                idx += 1

        print(f"[OK] {len(self.small_obstacles)} small static obstacles spawned (coverage {total_area:.1f} m²)")


# ─────────────────────────────────────────────────────────────────────────────
# WAYPOINTS
# ─────────────────────────────────────────────────────────────────────────────

def generate_waypoints(rng: np.random.Generator) -> list:
    waypoints = []
    prev_x, prev_y = SPOT_START_X, SPOT_START_Y

    for i, label in enumerate(WAYPOINT_LABELS):
        if i == 0:
            target_dist = WAYPOINT_DIST_A
            max_attempts = 100
            for _ in range(max_attempts):
                theta = rng.uniform(0, 2 * math.pi)
                wx = prev_x + target_dist * math.cos(theta)
                wy = prev_y + target_dist * math.sin(theta)
                if inside_arena(wx, wy, ARENA_RADIUS, margin=0.5):
                    waypoints.append({"label": label, "pos": np.array([wx, wy])})
                    prev_x, prev_y = wx, wy
                    break
            else:
                print(f"[WARN] Could not place waypoint {label}")
                waypoints.append({"label": label, "pos": np.array([prev_x, prev_y])})
        else:
            target_dist = WAYPOINT_SPACING_BZ
            max_attempts = 100
            for _ in range(max_attempts):
                theta = rng.uniform(0, 2 * math.pi)
                wx = prev_x + target_dist * math.cos(theta)
                wy = prev_y + target_dist * math.sin(theta)
                if inside_arena(wx, wy, ARENA_RADIUS, margin=0.5):
                    waypoints.append({"label": label, "pos": np.array([wx, wy])})
                    prev_x, prev_y = wx, wy
                    break
            else:
                print(f"[WARN] Could not place waypoint {label}")
                waypoints.append({"label": label, "pos": np.array([prev_x, prev_y])})

    print(f"[OK] Generated {len(waypoints)} waypoints (A={WAYPOINT_DIST_A}m, rest≥{WAYPOINT_SPACING_BZ}m)")
    return waypoints


def spawn_waypoint_marker(stage, label: str, pos: np.ndarray) -> list:
    pole_path = f"/World/Waypoints/Marker_{label}_Pole"
    flag_path = f"/World/Waypoints/Marker_{label}_Flag"

    pole_mesh = UsdGeom.Mesh.Define(stage, pole_path)
    segments = 12
    pts = []
    for z in [0, WP_POLE_HEIGHT]:
        for i in range(segments):
            theta = 2 * math.pi * i / segments
            x = WP_POLE_RADIUS * math.cos(theta)
            y = WP_POLE_RADIUS * math.sin(theta)
            pts.append(Gf.Vec3f(x, y, z))
    pole_mesh.GetPointsAttr().Set(pts)
    face_counts = []
    indices = []
    for i in range(segments):
        i_next = (i + 1) % segments
        indices.extend([i, i_next, i_next + segments, i + segments])
        face_counts.append(4)
    pole_mesh.GetFaceVertexCountsAttr().Set(face_counts)
    pole_mesh.GetFaceVertexIndicesAttr().Set(indices)
    pole_mesh.GetDisplayColorAttr().Set([COLOR_FLAG_POLE])

    pole_prim = stage.GetPrimAtPath(pole_path)
    pole_xform = UsdGeom.Xformable(pole_prim)
    pole_xform.AddTranslateOp().Set(Gf.Vec3d(pos[0], pos[1], 0))

    flag_mesh = UsdGeom.Mesh.Define(stage, flag_path)
    hw = WP_FLAG_WIDTH / 2
    hh = WP_FLAG_HEIGHT / 2
    flag_pts = [
        Gf.Vec3f(-hw, 0, WP_POLE_HEIGHT - hh),
        Gf.Vec3f(hw,  0, WP_POLE_HEIGHT - hh),
        Gf.Vec3f(hw,  0, WP_POLE_HEIGHT + hh),
        Gf.Vec3f(-hw, 0, WP_POLE_HEIGHT + hh),
    ]
    flag_mesh.GetPointsAttr().Set(flag_pts)
    flag_mesh.GetFaceVertexCountsAttr().Set([4])
    flag_mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])

    if label == WAYPOINT_LABELS[0]:
        flag_mesh.GetDisplayColorAttr().Set([COLOR_WAYPOINT])
    else:
        flag_mesh.GetDisplayColorAttr().Set([COLOR_START_WAYPOINT])

    flag_prim = stage.GetPrimAtPath(flag_path)
    flag_xform = UsdGeom.Xformable(flag_prim)
    flag_xform.AddTranslateOp().Set(Gf.Vec3d(pos[0], pos[1], 0))

    return [pole_path, flag_path]


def remove_waypoint_markers(stage, marker_paths: list) -> None:
    for path in marker_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            stage.RemovePrim(path)


# ─────────────────────────────────────────────────────────────────────────────
# SPEED CONTROL
# ─────────────────────────────────────────────────────────────────────────────

def compute_speed_command(spot_x: float, spot_y: float,
                          target_x: float, target_y: float,
                          obstacle_mgr: ObstacleManager) -> float:
    dist_to_target = distance_2d([spot_x, spot_y], [target_x, target_y])
    dist_to_nearest_obs = obstacle_mgr.nearest_obstacle_distance(spot_x, spot_y)

    speed = SPOT_MAX_SPEED_MPS

    if dist_to_nearest_obs < 3.0:
        slowdown_factor = max(0.3, dist_to_nearest_obs / 3.0)
        speed *= slowdown_factor

    if dist_to_target < OBSTACLE_SLOW_RADIUS:
        approach_factor = max(0.4, dist_to_target / OBSTACLE_SLOW_RADIUS)
        speed *= approach_factor

    return np.clip(speed, SPOT_MIN_SPEED_MPS, SPOT_MAX_SPEED_MPS)


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT CLASS
# ─────────────────────────────────────────────────────────────────────────────

class CircularWaypointEnv:
    """Circular arena waypoint navigation environment — MPPI planner."""

    def __init__(self, world, stage, rng: np.random.Generator):
        self.world = world
        self.stage = stage
        self.rng = rng

        self.spot = None
        self.obstacle_mgr = ObstacleManager(stage, rng)

        self.waypoints = []
        self.current_waypoint_idx = 0
        self.current_marker_paths = []

        self.score = EPISODE_START_SCORE
        self.waypoints_reached = 0
        self.episode_start_time = 0.0
        self.episode_num = 0

        self.csv_file = None
        self.csv_writer = None
        self.all_episodes = []
        self.episode_logged = False
        self.last_status_print_time = 0.0

        self.physics_ready = False

        # Control frequency: policy runs at 20 Hz (every 25 physics steps at 500 Hz)
        self._control_decimation = 25
        self._physics_step_count = 0
        self._cached_command = np.array([0.0, 0.0, 0.0])

        # MPPI navigator — sole navigation decision maker, no neural network
        self._mppi_nav = MPPINavigator(
            horizon=args.horizon,
            num_samples=args.num_samples,
            temperature=args.temperature,
        )

    def reset(self, episode: int) -> None:
        print(f"\n{'-' * 72}")
        print(f"[RESET] Episode {episode}")
        print(f"{'-' * 72}")
        print(f"\n[DEBUG] COVERAGE CONFIGURATION:")
        print(f"[DEBUG]   moveable_coverage_pct = 10.0/3 = {10.0/3:.4f}%")
        print(f"[DEBUG]   non_moveable_coverage_pct = 10.0/3 = {10.0/3:.4f}%")
        print(f"[DEBUG]   SMALL_OBSTACLE_COVERAGE * 100 = {(0.10/3)*100:.4f}%")
        print(f"[DEBUG]   TOTAL PLANNED COVERAGE = {(10.0/3 + 10.0/3 + (0.10/3)*100):.2f}%")

        self.episode_num = episode
        self.score = EPISODE_START_SCORE
        self.waypoints_reached = 0
        self.episode_start_time = 0.0
        self.current_waypoint_idx = 0
        self.physics_ready = False
        self.episode_logged = False
        self.last_status_print_time = 0.0
        self._physics_step_count = 0
        self._cached_command = np.array([0.0, 0.0, 0.0])
        self._mppi_nav.reset()

        # Remove old obstacles
        self.obstacle_mgr.remove_prims()

        # Remove old waypoint markers
        remove_waypoint_markers(self.stage, self.current_marker_paths)
        self.current_marker_paths.clear()

        # Generate new waypoints
        self.waypoints = generate_waypoints(self.rng)

        # Spawn only the first waypoint marker (sequential spawning)
        if self.waypoints:
            wp = self.waypoints[self.current_waypoint_idx]
            paths = spawn_waypoint_marker(self.stage, wp["label"], wp["pos"])
            self.current_marker_paths.extend(paths)
            print(f"[INFO] Current target: Waypoint {wp['label']} at ({wp['pos'][0]:.1f}, {wp['pos'][1]:.1f})")

        # Large + small static obstacles (10% fill each)
        self.obstacle_mgr.populate(moveable_coverage_pct=10.0, non_moveable_coverage_pct=10.0,
                                   min_spawn_clearance=2.0)
        self.obstacle_mgr.spawn_small_static(target_coverage_pct=10.0)
        print(f"[DEBUG] obstacle_mgr: {len(self.obstacle_mgr.obstacles)} large, {len(self.obstacle_mgr.small_obstacles)} small")

        # Reset Spot position
        if self.spot is not None:
            self.spot.robot.set_world_pose(
                position=np.array([SPOT_START_X, SPOT_START_Y, SPOT_START_Z]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0])
            )
            self.spot.robot.set_joints_default_state(self.spot.default_pos)
            print(f"[OK] Spot reset to start ({SPOT_START_X}, {SPOT_START_Y}, {SPOT_START_Z})")
        else:
            print("[WARN] Spot not initialized yet")

        # Load existing CSV data
        if CSV_LOG_PATH and self.csv_file is None:
            self._load_existing_csv_data()
            print(f"[OK] CSV log ready: {CSV_LOG_PATH}")

        print(f"[OK] Environment reset complete")

    def step(self, step_size: float) -> bool:
        """Execute one physics step. Returns False if episode should terminate."""
        if not self.physics_ready:
            self.episode_start_time = self.world.current_time
            self.physics_ready = True
            return True

        # Get Spot position
        spot_pos, _ = self.spot.robot.get_world_pose()
        spot_x, spot_y, spot_z = spot_pos[0], spot_pos[1], spot_pos[2]

        # Fall check
        if spot_z < FALL_HEIGHT_THRESHOLD:
            if not self.episode_logged:
                print(f"[FALL] Spot fell (z={spot_z:.3f} < {FALL_HEIGHT_THRESHOLD})")
                self._log_to_csv(FAILURE_FELL_OVER)
                self.episode_logged = True
            return False

        # Waypoint reached check
        if self.current_waypoint_idx < len(self.waypoints):
            wp = self.waypoints[self.current_waypoint_idx]
            dist_to_wp = distance_2d([spot_x, spot_y], wp["pos"])

            if dist_to_wp < WAYPOINT_REACH_DIST:
                self.waypoints_reached += 1
                self.score += WAYPOINT_REWARD
                print(f"[WAYPOINT] Reached {wp['label']} (+{WAYPOINT_REWARD} pts) | Score: {self.score:.1f} | Total: {self.waypoints_reached}/{WAYPOINT_COUNT}")

                remove_waypoint_markers(self.stage, self.current_marker_paths)
                self.current_marker_paths.clear()

                self.current_waypoint_idx += 1

                if self.current_waypoint_idx < len(self.waypoints):
                    next_wp = self.waypoints[self.current_waypoint_idx]
                    paths = spawn_waypoint_marker(self.stage, next_wp["label"], next_wp["pos"])
                    self.current_marker_paths.extend(paths)
                    print(f"[INFO] Next target: Waypoint {next_wp['label']} at ({next_wp['pos'][0]:.1f}, {next_wp['pos'][1]:.1f})")
                    # Reset MPPI warm-start for new goal
                    self._mppi_nav.reset()
                else:
                    if not self.episode_logged:
                        print(f"[COMPLETE] All waypoints reached!")
                        self._log_to_csv(FAILURE_COMPLETED)
                        self.episode_logged = True
                    return False

                self.spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
                return True

        # Time decay
        elapsed = self.world.current_time - self.episode_start_time
        self.score = EPISODE_START_SCORE + self.waypoints_reached * WAYPOINT_REWARD + elapsed * (-TIME_DECAY_PER_SEC)

        # Status print every second
        if elapsed - self.last_status_print_time >= 1.0:
            if self.current_waypoint_idx < len(self.waypoints):
                wp = self.waypoints[self.current_waypoint_idx]
                dist_to_wp = distance_2d([spot_x, spot_y], wp["pos"])
                print(f"[STATUS] Score: {self.score:.1f} | Target: {wp['label']} | Distance: {dist_to_wp:.2f}m")
            else:
                print(f"[STATUS] Score: {self.score:.1f} | No active waypoint")
            self.last_status_print_time = elapsed

        # Score depletion check
        if self.score <= 0:
            if not self.episode_logged:
                print(f"[TIMEOUT] Score depleted (elapsed={elapsed:.1f}s)")
                self._log_to_csv(FAILURE_RAN_OUT)
                self.episode_logged = True
            return False

        # ═══════════════════════════════════════════════════════════════════════
        # MPPI NAVIGATION — Pure model-predictive planning, no neural network.
        # Runs at 20 Hz (every _control_decimation physics steps).
        # No FSM, no checkpoint, no raycasting — MPPI handles it all.
        # ═══════════════════════════════════════════════════════════════════════
        if self.current_waypoint_idx < len(self.waypoints):
            self._physics_step_count += 1

            if self._physics_step_count % self._control_decimation == 1:
                wp = self.waypoints[self.current_waypoint_idx]

                _, spot_orientation = self.spot.robot.get_world_pose()
                current_heading = quaternion_to_yaw(spot_orientation)

                # Build obstacle bounding circles for MPPI rollouts
                _all_obs = list(self.obstacle_mgr.obstacles) + list(self.obstacle_mgr.small_obstacles)
                obs_circles = []
                for _o in _all_obs:
                    _ox, _oy = _o["pos"][0], _o["pos"][1]
                    if "dims" in _o:
                        _or = math.sqrt(_o["dims"][0] ** 2 + _o["dims"][1] ** 2) / 2.0
                    else:
                        _or = _o.get("size", 0.3) / 2.0
                    obs_circles.append((_ox, _oy, _or))

                # MPPI solve → [vx, vy, omega]
                # SpotFlatTerrainPolicy only accepts [vx, 0.0, omega] — vy must be zero
                _cmd = self._mppi_nav.solve(
                    pos=np.array([spot_x, spot_y], dtype=np.float32),
                    yaw=current_heading,
                    target=wp["pos"],
                    obstacles=obs_circles,
                )
                self._cached_command = np.array([_cmd[0], 0.0, _cmd[2]], dtype=np.float32)

        # Apply the cached command every physics step
        self.spot.forward(step_size, self._cached_command)
        return True

    def _load_existing_csv_data(self) -> None:
        if os.path.exists(CSV_LOG_PATH):
            with open(CSV_LOG_PATH, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('Episode') and row['Episode'].isdigit():
                        self.all_episodes.append({
                            'episode': int(row['Episode']),
                            'waypoints': int(row['Waypoints_Reached']),
                            'reason': row['Failure_Reason'],
                            'score': float(row['Final_Score'])
                        })
            print(f"[INFO] Loaded {len(self.all_episodes)} previous episodes from CSV")

    def _calculate_aggregate_stats(self) -> dict:
        if not self.all_episodes:
            return {
                'avg_waypoints': 0.0,
                'max_waypoints': 0,
                'fell_rate': 0.0,
                'ran_out_rate': 0.0,
                'completion_rate': 0.0
            }

        total = len(self.all_episodes)
        avg_waypoints = sum(ep['waypoints'] for ep in self.all_episodes) / total
        max_waypoints = max(ep['waypoints'] for ep in self.all_episodes)
        fell_count = sum(1 for ep in self.all_episodes if ep['reason'] == FAILURE_FELL_OVER)
        ran_out_count = sum(1 for ep in self.all_episodes if ep['reason'] == FAILURE_RAN_OUT)
        completed_count = sum(1 for ep in self.all_episodes if ep['reason'] == FAILURE_COMPLETED)

        return {
            'avg_waypoints': avg_waypoints,
            'max_waypoints': max_waypoints,
            'fell_rate': (fell_count / total) * 100,
            'ran_out_rate': (ran_out_count / total) * 100,
            'completion_rate': (completed_count / total) * 100
        }

    def _write_csv_with_stats(self) -> None:
        stats = self._calculate_aggregate_stats()

        with open(CSV_LOG_PATH, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(["# AGGREGATE STATISTICS (Updated after each episode)"])
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Average_Waypoints_Reached", f"{stats['avg_waypoints']:.2f}"])
            writer.writerow(["Max_Waypoints_Reached", f"{stats['max_waypoints']}"])
            writer.writerow(["Failure_Rate_Fell_Over", f"{stats['fell_rate']:.2f}%"])
            writer.writerow(["Failure_Rate_Ran_Out_of_Points", f"{stats['ran_out_rate']:.2f}%"])
            writer.writerow(["Completion_Rate", f"{stats['completion_rate']:.2f}%"])
            writer.writerow([])

            writer.writerow(CSV_HEADERS)

            for ep in self.all_episodes:
                writer.writerow([
                    ep['episode'],
                    ep['waypoints'],
                    ep['reason'],
                    f"{ep['score']:.2f}"
                ])

    def _log_to_csv(self, reason: str) -> None:
        episode_data = {
            'episode': self.episode_num,
            'waypoints': self.waypoints_reached,
            'reason': reason,
            'score': self.score
        }
        self.all_episodes.append(episode_data)
        self._write_csv_with_stats()

        print(f"[CSV] Episode {self.episode_num} logged: {self.waypoints_reached} waypoints, score {self.score:.1f}, reason: {reason}")

        stats = self._calculate_aggregate_stats()
        print(f"[STATS] Avg Waypoints: {stats['avg_waypoints']:.2f} | Completion: {stats['completion_rate']:.1f}% | Fell: {stats['fell_rate']:.1f}% | Ran Out: {stats['ran_out_rate']:.1f}%")

    def close(self) -> None:
        print(f"[OK] CSV log finalized: {CSV_LOG_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# WORLD BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_world(world, stage) -> None:
    print(f"\n{'-' * 72}")
    print("[WORLD] Building environment...")
    print(f"{'-' * 72}")

    world.scene.add_default_ground_plane(
        z_position=0.0,
        name="ground_plane",
        prim_path="/World/GroundPlane",
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.01
    )
    print("[OK] Ground plane added (z=0)")

    floor_disc_path = "/World/FloorDisc"
    disc_mesh = UsdGeom.Mesh.Define(stage, floor_disc_path)
    segments = 64
    pts = [Gf.Vec3f(0, 0, 0.01)]
    for i in range(segments):
        theta = 2 * math.pi * i / segments
        x = ARENA_RADIUS * math.cos(theta)
        y = ARENA_RADIUS * math.sin(theta)
        pts.append(Gf.Vec3f(x, y, 0.01))

    face_counts = []
    indices = []
    for i in range(segments):
        indices.extend([0, i + 1, (i + 1) % segments + 1])
        face_counts.append(3)

    disc_mesh.GetPointsAttr().Set(pts)
    disc_mesh.GetFaceVertexCountsAttr().Set(face_counts)
    disc_mesh.GetFaceVertexIndicesAttr().Set(indices)
    disc_mesh.GetDisplayColorAttr().Set([Gf.Vec3f(0.3, 0.35, 0.3)])
    print("[OK] Floor disc visual added")

    distant_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    distant_light.CreateIntensityAttr(800.0)
    distant_light.CreateAngleAttr(0.53)
    xform = UsdGeom.Xformable(distant_light)
    xform.AddRotateXYZOp().Set(Gf.Vec3d(315, 45, 0))
    print("[OK] Distant light added")

    print(f"{'-' * 72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SENSOR SUITE
# ─────────────────────────────────────────────────────────────────────────────

def setup_spot_sensors(spot_prim_path: str):
    print("\n" + "-" * 72)
    print("ADDING BOSTON DYNAMICS SPOT SENSOR SUITE")
    print("-" * 72)

    sensors_created = []
    sensor_defs = [
        ("FrontStereoLeft",    "front_stereo_left",    [0.35,  0.06, 0.25]),
        ("FrontStereoRight",   "front_stereo_right",   [0.35, -0.06, 0.25]),
        ("LeftStereoFront",    "left_stereo_front",    [0.10,  0.25, 0.20]),
        ("LeftStereoRear",     "left_stereo_rear",     [-0.10, 0.25, 0.20]),
        ("RightStereoFront",   "right_stereo_front",   [0.10, -0.25, 0.20]),
        ("RightStereoRear",    "right_stereo_rear",    [-0.10,-0.25, 0.20]),
        ("RearStereoLeft",     "rear_stereo_left",     [-0.35, 0.06, 0.20]),
        ("RearStereoRight",    "rear_stereo_right",    [-0.35,-0.06, 0.20]),
        ("OverheadStereoFront","overhead_stereo_front",[0.05,  0.04, 0.35]),
        ("OverheadStereoRear", "overhead_stereo_rear", [0.05, -0.04, 0.35]),
    ]

    for prim_name, cam_name, pos in sensor_defs:
        try:
            cam = Camera(
                prim_path=f"{spot_prim_path}/{prim_name}",
                name=cam_name,
                frequency=30,
                resolution=(640, 480),
            )
            cam.set_world_pose(position=np.array(pos))
            sensors_created.append(f"{prim_name} (640x480, 30Hz)")
        except Exception as e:
            print(f"  [WARN] {prim_name}: {e}")

    print(f"\n[OK] {len(sensors_created)} cameras created")
    print("[OK] Additional built-in sensors: IMU, joint encoders, foot contact sensors")
    print("-" * 72 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 72)
    print("  SPOT MPPI NAVIGATION — CIRCULAR WAYPOINT ARENA")
    print("=" * 72)

    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)
    stage = omni.usd.get_context().get_stage()

    build_world(world, stage)

    rng = np.random.default_rng(args.seed)
    print(f"[INFO] Random seed: {args.seed}")

    spot_prim_path = "/World/Spot"
    spot = SpotFlatTerrainPolicy(
        prim_path=spot_prim_path,
        name="Spot",
        position=np.array([SPOT_START_X, SPOT_START_Y, SPOT_START_Z])
    )
    print(f"[OK] SpotFlatTerrainPolicy created at {spot_prim_path}")

    world.reset()
    print("[OK] World reset")

    spot.initialize()
    spot.robot.set_joints_default_state(spot.default_pos)
    print("[OK] Spot (flat) initialized")

    setup_spot_sensors(spot_prim_path)

    env = CircularWaypointEnv(world, stage, rng)
    env.spot = spot

    def on_physics_step(step_size: float):
        if not env.step(step_size):
            pass

    world.add_physics_callback("spot_control", on_physics_step)
    print("[OK] Physics callback registered")

    print(f"\n{'=' * 72}")
    print(f"  STARTING — {args.episodes} EPISODES  |  MPPI H={args.horizon} K={args.num_samples} λ={args.temperature}")
    print(f"{'=' * 72}\n")

    for episode in range(1, args.episodes + 1):
        env.reset(episode)

        while simulation_app.is_running():
            world.step(render=not args.headless)

            if env.score <= 0 or env.waypoints_reached >= WAYPOINT_COUNT:
                break

        if not env.episode_logged:
            if env.waypoints_reached >= WAYPOINT_COUNT:
                env._log_to_csv(FAILURE_COMPLETED)
            elif env.score <= 0:
                env._log_to_csv(FAILURE_RAN_OUT)
            env.episode_logged = True

        if not simulation_app.is_running():
            print("\n[EXIT] Simulation closed by user")
            break

    env.close()
    simulation_app.close()
    print("\n" + "=" * 72)
    print("  COMPLETE  —  Results saved to MPPI_CSV.csv")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
