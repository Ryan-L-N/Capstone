"""
Testing Environment 2 — Circular Waypoint Navigation Arena
===========================================================
Circular training environment for Boston Dynamics Spot RL locomotion.

Environment Geometry
--------------------
  - Shape    : Circle, diameter 50 m (radius 25 m)
  - Center   : (0, 0)
  - Terrain  : Flat ground plane
  - Boundary : 64-segment polygon wall, h = 1.5 m

Obstacles
---------
  - Cover ~20 % of total arena area (≈ 392.7 m²)
  - Fully randomized each episode (position, shape, size, weight)
  - Shapes: rectangle, square, trapezoid, sphere, diamond, oval, cylinder
  - Light (< 1 lb / 0.45 kg) → Spot can push them
  - Heavy (up to 32.7 kg)    → Spot must navigate around them

Spot Robot
----------
  - SpotFlatTerrainPolicy (FlatTerrain locomotion)
  - All standard sensors: camera, depth camera, LiDAR (analytic), IMU, encoders, contact
  - Max speed: 5 mph (2.235 m/s); speed scales down near obstacles

Waypoints
---------
  - 25 waypoints labeled A – Y
  - Start: A = (0, 0)
  - Each waypoint ≥ 25 m from previous, all within arena
  - Randomized positions each episode
  - Spot visits them in alphabetical order

Episode Randomization
---------------------
  - Obstacle positions, shapes, sizes, and weights
  - Waypoint positions (chain placement from A)
  - Episode seed logged for reproducibility

Author : Cole (MS for Autonomy Project)
Date   : February 2026
Spec   : Cole_md.md  (same directory)
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
parser = argparse.ArgumentParser(description="Testing Environment 2 — Circular Waypoint Arena")
parser.add_argument("--headless",  action="store_true", help="Run without GUI")
parser.add_argument("--episodes",  type=int, default=1,  help="Number of training episodes")
parser.add_argument("--seed",      type=int, default=None, help="Global RNG seed (None = random)")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Isaac Sim boot — MUST happen before any other Isaac imports
# ─────────────────────────────────────────────────────────────────────────────
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import omni
from omni.isaac.core import World
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
from pxr import UsdGeom, UsdLux, UsdPhysics, Gf, Sdf

import sys
sys.path.insert(0, r"C:\Users\user\Desktop\Capstone_vs_1.2\Immersive-Modeling-and-Simulation-for-Autonomy\Experiments\Cole")
from Spots.Spot_1 import SpotRobot  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Arena
ARENA_RADIUS         = 25.0          # meters (diameter = 50 m)
ARENA_CENTER_X       = 0.0
ARENA_CENTER_Y       = 0.0
WALL_SEGMENTS        = 64            # polygon approximation of the circle
WALL_HEIGHT          = 1.5           # meters
WALL_THICKNESS       = 0.3           # meters

# Spot
SPOT_START_X         = 0.0
SPOT_START_Y         = 0.0
SPOT_START_Z         = 0.7
SPOT_MASS_KG         = 32.7          # Boston Dynamics Spot weight
# Spot bounding box (meters)
SPOT_LENGTH          = 1.1
SPOT_WIDTH           = 0.5
SPOT_HEIGHT          = 0.6

FALL_HEIGHT_THRESHOLD = 0.25         # meters — z below this means Spot has fallen (matches Cole_vs3.py)

# Speed
SPOT_MAX_SPEED_MPS   = 5 * 0.44704  # 5 mph → m/s  ≈ 2.235 m/s
SPOT_MIN_SPEED_MPS   = 0.3          # minimum crawl speed
OBSTACLE_SLOW_RADIUS = 2.0          # meters — start slowing at this distance

# Obstacles
OBSTACLE_AREA_FRAC   = 0.20         # 20 % of arena
OBSTACLE_MIN_FOOT    = 0.0058       # m²  (9 in²)
OBSTACLE_MAX_FOOT    = SPOT_LENGTH * SPOT_WIDTH  # 0.55 m²
OBSTACLE_LIGHT_THRESH = 0.45        # kg  (≈ 1 lb)
OBSTACLE_MIN_MASS    = 0.05         # kg
OBSTACLE_MAX_MASS    = SPOT_MASS_KG # 32.7 kg
OBSTACLE_CLEARANCE_BOUNDARY = 3.0  # keep obstacles this far inside boundary
OBSTACLE_CLEARANCE_WAYPOINT = 5.0  # keep obstacles this far from waypoints

# Waypoints
WAYPOINT_COUNT       = 25
WAYPOINT_LABELS      = list(string.ascii_uppercase[:WAYPOINT_COUNT])  # A … Y
WAYPOINT_DIST_A      = 24.0         # meters — A is placed exactly 24 m from (0,0)
WAYPOINT_SPACING_BZ  = 30.0         # meters — B–Y each at least 30 m from previous
WAYPOINT_REACH_DIST  = 0.5          # meters — threshold to "collect" a waypoint
WAYPOINT_BOUNDARY_MARGIN = 2.0      # meters — keep waypoints inside circle
WAYPOINT_RADIUS_VIS  = 0.3          # cylinder radius for visual marker
WAYPOINT_HEIGHT_VIS  = 1.5          # cylinder height for visual marker

# Colors (RGB)
COLOR_LIGHT_OBSTACLE = Gf.Vec3f(1.0,  0.55, 0.0)   # orange
COLOR_HEAVY_OBSTACLE = Gf.Vec3f(0.27, 0.51, 0.71)  # steel blue
COLOR_WAYPOINT       = Gf.Vec3f(1.0,  0.95, 0.0)   # bright yellow
COLOR_START_WAYPOINT = Gf.Vec3f(0.2,  0.9,  0.2)   # bright green
COLOR_FLAG_POLE      = Gf.Vec3f(0.88, 0.88, 0.88)  # light grey pole
COLOR_BOUNDARY       = Gf.Vec3f(0.6,  0.6,  0.6)   # grey

# Waypoint flag geometry
WP_POLE_RADIUS       = 0.05    # meters — pole radius
WP_POLE_HEIGHT       = 2.5     # meters — pole height
WP_FLAG_WIDTH        = 0.7     # meters — banner width
WP_FLAG_DEPTH        = 0.06    # meters — banner thickness
WP_FLAG_HEIGHT       = 0.40    # meters — banner height

# Physics
PHYSICS_DT           = 1.0 / 500.0
RENDERING_DT         = 10.0 / 500.0

# ── Reward / Scoring system ──────────────────────────────────────────────
# Core score parameters
EPISODE_START_SCORE  = 300.0    # points at episode start
TIME_DECAY_PER_SEC   = 1.0      # points lost per real sim-second
WAYPOINT_REWARD      = 15.0     # points awarded per waypoint collected

# Modular shaping hooks (add new terms inside compute_reward)
# PENALTY_ENERGY_COEFF = -0.001  # example: per Nm² of torque — enable when ready
# REWARD_SMOOTHNESS    =  0.01   # example: smoothness bonus — enable when ready

# Misc
ARENA_AREA           = math.pi * ARENA_RADIUS ** 2   # ≈ 1963.5 m²
TARGET_OBSTACLE_AREA = ARENA_AREA * OBSTACLE_AREA_FRAC  # ≈ 392.7 m²

# CSV logging (persists across training runs)
CSV_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_log.csv")
CSV_HEADERS  = ["Episode", "Waypoints_Reached", "Time_Elapsed", "Final_Score"]

print("=" * 72)
print("TESTING ENVIRONMENT 2 — CIRCULAR WAYPOINT NAVIGATION ARENA")
print(f"  Arena radius : {ARENA_RADIUS} m  (diameter {ARENA_RADIUS * 2} m)")
print(f"  Arena area   : {ARENA_AREA:.1f} m²")
print(f"  Target obstacle coverage : {TARGET_OBSTACLE_AREA:.1f} m² ({OBSTACLE_AREA_FRAC*100:.0f}%)")
print(f"  Waypoints    : {WAYPOINT_COUNT}  (labels {WAYPOINT_LABELS[0]}–{WAYPOINT_LABELS[-1]})")
print(f"  Spot max speed : {SPOT_MAX_SPEED_MPS:.3f} m/s  ({5:.0f} mph)")
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


def apply_rigid_body_physics(stage, prim_path: str, mass_kg: float,
                              friction: float = 0.5) -> None:
    """Add RigidBodyAPI, CollisionAPI, and MassAPI to a USD prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return

    # Rigid body
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_api.CreateRigidBodyEnabledAttr(True)

    # Collision
    UsdPhysics.CollisionAPI.Apply(prim)

    # PhysX requires convexHull approximation for dynamic mesh bodies
    if prim.GetTypeName() == "Mesh":
        mesh_coll = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_coll.CreateApproximationAttr("convexHull")

    # Mass
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass_kg)

    # Material friction
    physics_mat = UsdPhysics.MaterialAPI.Apply(prim)
    physics_mat.CreateStaticFrictionAttr(friction)
    physics_mat.CreateDynamicFrictionAttr(friction * 0.8)
    physics_mat.CreateRestitutionAttr(0.05)


# ─────────────────────────────────────────────────────────────────────────────
# BOUNDARY — circular wall (64-segment polygon)
# ─────────────────────────────────────────────────────────────────────────────

def create_circular_boundary(stage, n_segments: int = WALL_SEGMENTS,
                              radius: float = ARENA_RADIUS,
                              height: float = WALL_HEIGHT,
                              thickness: float = WALL_THICKNESS) -> None:
    """
    Approximate a circular wall using N thin Mesh box segments.
    Each segment is a small rectangular slab placed tangent to the inner face of the circle.
    """
    angle_step = 2 * math.pi / n_segments
    # Arc length per segment (inner face)
    seg_len = 2 * math.pi * radius / n_segments + 0.02  # small overlap to close gaps

    for i in range(n_segments):
        # Midpoint angle of this segment
        theta = i * angle_step
        # Center of this wall segment — at the boundary radius
        cx = (radius + thickness / 2) * math.cos(theta)
        cy = (radius + thickness / 2) * math.sin(theta)

        seg_path = f"/World/BoundaryWall/Seg_{i:03d}"
        mesh = UsdGeom.Mesh.Define(stage, seg_path)

        # Half-dimensions in local space (segment aligned to X axis, then rotated)
        hw = seg_len / 2
        hd = thickness / 2

        # Local box vertices (unrotated, centered at origin, segment along X)
        local_pts = [
            Gf.Vec3f(-hw, -hd, 0),
            Gf.Vec3f( hw, -hd, 0),
            Gf.Vec3f( hw,  hd, 0),
            Gf.Vec3f(-hw,  hd, 0),
            Gf.Vec3f(-hw, -hd, height),
            Gf.Vec3f( hw, -hd, height),
            Gf.Vec3f( hw,  hd, height),
            Gf.Vec3f(-hw,  hd, height),
        ]

        # Rotate each point by theta around Z, then translate to (cx, cy)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        world_pts = []
        for pt in local_pts:
            rx = pt[0] * cos_t - pt[1] * sin_t + cx
            ry = pt[0] * sin_t + pt[1] * cos_t + cy
            world_pts.append(Gf.Vec3f(rx, ry, pt[2]))

        mesh.GetPointsAttr().Set(world_pts)
        mesh.GetFaceVertexCountsAttr().Set([4, 4, 4, 4, 4, 4])
        mesh.GetFaceVertexIndicesAttr().Set([
            0, 1, 2, 3,   # bottom
            4, 7, 6, 5,   # top
            0, 4, 5, 1,   # south face
            2, 6, 7, 3,   # north face
            0, 3, 7, 4,   # west face
            1, 5, 6, 2,   # east face
        ])
        mesh.GetDisplayColorAttr().Set([COLOR_BOUNDARY])

        # Static collision
        prim = stage.GetPrimAtPath(seg_path)
        UsdPhysics.CollisionAPI.Apply(prim)

    print(f"[OK] Circular boundary wall created ({n_segments} segments, "
          f"r={radius} m, h={height} m)")


# ─────────────────────────────────────────────────────────────────────────────
# OBSTACLES — shape creators
# ─────────────────────────────────────────────────────────────────────────────

def _set_color(prim_path: str, color: Gf.Vec3f, stage) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        gprim = UsdGeom.Gprim(prim)
        gprim.GetDisplayColorAttr().Set([color])


def create_obstacle_rectangle(stage, prim_path: str, cx: float, cy: float,
                               length: float, width: float, height: float) -> float:
    """Box with rectangular footprint. Returns footprint area (m²)."""
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    hl, hw = length / 2, width / 2
    pts = [
        Gf.Vec3f(cx - hl, cy - hw, 0), Gf.Vec3f(cx + hl, cy - hw, 0),
        Gf.Vec3f(cx + hl, cy + hw, 0), Gf.Vec3f(cx - hl, cy + hw, 0),
        Gf.Vec3f(cx - hl, cy - hw, height), Gf.Vec3f(cx + hl, cy - hw, height),
        Gf.Vec3f(cx + hl, cy + hw, height), Gf.Vec3f(cx - hl, cy + hw, height),
    ]
    mesh.GetPointsAttr().Set(pts)
    mesh.GetFaceVertexCountsAttr().Set([4, 4, 4, 4, 4, 4])
    mesh.GetFaceVertexIndicesAttr().Set([
        0, 1, 2, 3,   # bottom
        4, 7, 6, 5,   # top
        0, 4, 5, 1,   # front
        2, 6, 7, 3,   # back
        0, 3, 7, 4,   # left
        1, 5, 6, 2,   # right
    ])
    return length * width


def create_obstacle_square(stage, prim_path: str, cx: float, cy: float,
                            side: float, height: float) -> float:
    """Square footprint box. Returns footprint area (m²)."""
    return create_obstacle_rectangle(stage, prim_path, cx, cy, side, side, height)


def create_obstacle_trapezoid(stage, prim_path: str, cx: float, cy: float,
                               base_w: float, top_w: float,
                               depth: float, height: float) -> float:
    """
    Trapezoid footprint — base_w (bottom edge width) tapers to top_w at the height.
    'depth' is the Y dimension (same at all levels).
    Returns approximate footprint area (m²).
    """
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    hbw = base_w / 2
    htw = top_w  / 2
    hd  = depth  / 2

    # Floor (trapezoid):  base vertices at z=0, top-of-taper vertices at z=0 on the floor face
    # We extrude the trapezoid shape upward as a prism (all heights share same footprint)
    # Vertices: 8 total (4 per z level)
    # At z=0: (-hbw,-hd), (+hbw,-hd), (+hbw,+hd), (-hbw,+hd)      — wide base
    # At z=h:  (-htw,-hd), (+htw,-hd), (+htw,+hd), (-htw,+hd)      — narrow top
    pts = [
        Gf.Vec3f(cx - hbw, cy - hd, 0),        # 0 bottom front-left
        Gf.Vec3f(cx + hbw, cy - hd, 0),        # 1 bottom front-right
        Gf.Vec3f(cx + hbw, cy + hd, 0),        # 2 bottom back-right
        Gf.Vec3f(cx - hbw, cy + hd, 0),        # 3 bottom back-left
        Gf.Vec3f(cx - htw, cy - hd, height),   # 4 top front-left
        Gf.Vec3f(cx + htw, cy - hd, height),   # 5 top front-right
        Gf.Vec3f(cx + htw, cy + hd, height),   # 6 top back-right
        Gf.Vec3f(cx - htw, cy + hd, height),   # 7 top back-left
    ]
    mesh.GetPointsAttr().Set(pts)
    mesh.GetFaceVertexCountsAttr().Set([4, 4, 4, 4, 4, 4])
    mesh.GetFaceVertexIndicesAttr().Set([
        0, 1, 2, 3,   # bottom
        4, 7, 6, 5,   # top
        0, 4, 5, 1,   # front slant face
        2, 6, 7, 3,   # back slant face
        0, 3, 7, 4,   # left slant face
        1, 5, 6, 2,   # right slant face
    ])
    return base_w * depth  # use base footprint as conservative area


def create_obstacle_sphere(stage, prim_path: str, cx: float, cy: float,
                            radius: float) -> float:
    """Sphere resting on ground (center z = radius). Returns footprint area (m²)."""
    sphere = UsdGeom.Sphere.Define(stage, prim_path)
    sphere.GetRadiusAttr().Set(radius)
    xform = UsdGeom.Xformable(sphere.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(cx, cy, radius))
    return math.pi * radius ** 2


def create_obstacle_diamond(stage, prim_path: str, cx: float, cy: float,
                             dx: float, dy: float, height: float) -> float:
    """
    Diamond (rhombus) footprint. dx = half-width along X, dy = half-width along Y.
    Returns footprint area (m²).
    """
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    # 4 vertices at floor, 4 at top — each level is the same diamond shape
    # Diamond points: top(0,+dy), right(+dx,0), bottom(0,-dy), left(-dx,0)
    pts = [
        Gf.Vec3f(cx,      cy + dy, 0),        # 0 bot north
        Gf.Vec3f(cx + dx, cy,      0),        # 1 bot east
        Gf.Vec3f(cx,      cy - dy, 0),        # 2 bot south
        Gf.Vec3f(cx - dx, cy,      0),        # 3 bot west
        Gf.Vec3f(cx,      cy + dy, height),   # 4 top north
        Gf.Vec3f(cx + dx, cy,      height),   # 5 top east
        Gf.Vec3f(cx,      cy - dy, height),   # 6 top south
        Gf.Vec3f(cx - dx, cy,      height),   # 7 top west
    ]
    mesh.GetPointsAttr().Set(pts)
    mesh.GetFaceVertexCountsAttr().Set([4, 4, 3, 3, 3, 3, 3, 3])
    mesh.GetFaceVertexIndicesAttr().Set([
        0, 1, 2, 3,   # bottom quad (north, east, south, west)
        4, 7, 6, 5,   # top quad
        0, 4, 5,      # north-east side
        5, 6, 1,      # east-south side (note: fix winding)
        6, 7, 2,      # south-west side
        7, 4, 3,      # west-north side
        0, 1, 5, 4,   # ... override with quads for stability
        1, 2, 6, 5,
    ])
    # Simpler: use 4-sided side panels
    mesh.GetFaceVertexCountsAttr().Set([4, 4, 4, 4, 4, 4])
    mesh.GetFaceVertexIndicesAttr().Set([
        0, 1, 2, 3,   # bottom
        4, 7, 6, 5,   # top
        0, 4, 5, 1,   # NE face
        1, 5, 6, 2,   # SE face
        2, 6, 7, 3,   # SW face
        3, 7, 4, 0,   # NW face
    ])
    return 2 * dx * dy  # rhombus area = d1 * d2 / 2 * 2 = d1*d2


def create_obstacle_oval(stage, prim_path: str, cx: float, cy: float,
                          rx: float, ry: float, height: float) -> float:
    """
    Oval — sphere primitive scaled to create an ellipsoid.
    rx, ry are semi-axes in X and Y; z-radius = height/2.
    Returns footprint area (m²).
    """
    sphere = UsdGeom.Sphere.Define(stage, prim_path)
    sphere.GetRadiusAttr().Set(1.0)  # unit sphere, scaled below
    xform = UsdGeom.Xformable(sphere.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(cx, cy, height / 2))
    xform.AddScaleOp().Set(Gf.Vec3f(rx, ry, height / 2))
    return math.pi * rx * ry


def create_obstacle_cylinder(stage, prim_path: str, cx: float, cy: float,
                              radius: float, height: float) -> float:
    """Upright cylinder resting on ground. Returns footprint area (m²)."""
    cyl = UsdGeom.Cylinder.Define(stage, prim_path)
    cyl.GetRadiusAttr().Set(radius)
    cyl.GetHeightAttr().Set(height)
    xform = UsdGeom.Xformable(cyl.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(cx, cy, height / 2))
    return math.pi * radius ** 2


# ─────────────────────────────────────────────────────────────────────────────
# OBSTACLE MANAGER
# ─────────────────────────────────────────────────────────────────────────────

SHAPE_TYPES = ["rectangle", "square", "trapezoid", "sphere", "diamond", "oval", "cylinder"]


class ObstacleManager:
    """
    Generates and manages all obstacles for a single episode.
    Tracks obstacle positions and masses for physics queries (proximity, push logic).
    """

    def __init__(self, stage, rng: np.random.Generator):
        self.stage = stage
        self.rng = rng
        self.obstacles = []   # list of dicts: {path, cx, cy, radius, mass, shape, footprint}
        self._index = 0

    def _is_valid_position(self, cx: float, cy: float, obs_radius: float,
                           waypoints_xy: list) -> bool:
        """Check new obstacle against boundary and waypoint clearances."""
        # Inside arena with boundary margin
        if not inside_arena(cx, cy, margin=OBSTACLE_CLEARANCE_BOUNDARY + obs_radius):
            return False
        # Clearance from all waypoints
        for wpt in waypoints_xy:
            if distance_2d((cx, cy), wpt) < (OBSTACLE_CLEARANCE_WAYPOINT + obs_radius):
                return False
        # Clearance from other obstacles (min 0.2 m between any two)
        for existing in self.obstacles:
            if distance_2d((cx, cy), (existing["cx"], existing["cy"])) < (
                    obs_radius + existing["radius"] + 0.2):
                return False
        return True

    def _random_mass(self) -> float:
        return float(self.rng.uniform(OBSTACLE_MIN_MASS, OBSTACLE_MAX_MASS))

    def _random_height(self) -> float:
        return float(self.rng.uniform(0.05, min(0.8, SPOT_HEIGHT)))

    def spawn_one(self, waypoints_xy: list) -> float:
        """
        Attempt to spawn a single random obstacle.
        Returns the footprint area placed (m²), or 0.0 if placement failed.
        """
        shape = str(self.rng.choice(SHAPE_TYPES))
        mass  = self._random_mass()
        color = COLOR_LIGHT_OBSTACLE if mass <= OBSTACLE_LIGHT_THRESH else COLOR_HEAVY_OBSTACLE
        h     = self._random_height()
        idx   = self._index

        # ── Sample size within spec bounds ──────────────────────────────────
        if shape in ("rectangle",):
            # length in [0.076 m, SPOT_LENGTH], width in [0.076 m, SPOT_WIDTH]
            length = float(self.rng.uniform(0.076, SPOT_LENGTH))
            width  = float(self.rng.uniform(0.076, SPOT_WIDTH))
            obs_r  = max(length, width) / 2

        elif shape == "square":
            side  = float(self.rng.uniform(0.076, min(SPOT_LENGTH, SPOT_WIDTH)))
            obs_r = side / 2

        elif shape == "trapezoid":
            base_w = float(self.rng.uniform(0.1, SPOT_LENGTH))
            top_w  = float(self.rng.uniform(0.05, base_w))
            depth  = float(self.rng.uniform(0.076, SPOT_WIDTH))
            obs_r  = max(base_w, depth) / 2

        elif shape == "sphere":
            # radius such that footprint (πr²) ∈ [0.0058, 0.55]
            r_min = math.sqrt(OBSTACLE_MIN_FOOT / math.pi)
            r_max = math.sqrt(OBSTACLE_MAX_FOOT / math.pi)
            radius = float(self.rng.uniform(r_min, r_max))
            obs_r  = radius

        elif shape == "diamond":
            dx = float(self.rng.uniform(0.038, SPOT_LENGTH / 2))
            dy = float(self.rng.uniform(0.038, SPOT_WIDTH  / 2))
            obs_r = max(dx, dy)

        elif shape == "oval":
            rx = float(self.rng.uniform(0.038, SPOT_LENGTH / 2))
            ry = float(self.rng.uniform(0.038, SPOT_WIDTH  / 2))
            obs_r = max(rx, ry)

        elif shape == "cylinder":
            r_min = math.sqrt(OBSTACLE_MIN_FOOT / math.pi)
            r_max = math.sqrt(OBSTACLE_MAX_FOOT / math.pi)
            radius = float(self.rng.uniform(r_min, r_max))
            obs_r  = radius

        else:
            return 0.0

        # ── Find a valid position ────────────────────────────────────────────
        prim_path = f"/World/Obstacles/Obstacle_{idx:04d}"
        placed_area = 0.0

        for _ in range(300):
            cx, cy = random_inside_arena(
                margin=OBSTACLE_CLEARANCE_BOUNDARY + obs_r, rng=self.rng
            )
            if self._is_valid_position(cx, cy, obs_r, waypoints_xy):
                # Create the shape
                if shape == "rectangle":
                    placed_area = create_obstacle_rectangle(
                        self.stage, prim_path, cx, cy, length, width, h)
                elif shape == "square":
                    placed_area = create_obstacle_square(
                        self.stage, prim_path, cx, cy, side, h)
                elif shape == "trapezoid":
                    placed_area = create_obstacle_trapezoid(
                        self.stage, prim_path, cx, cy, base_w, top_w, depth, h)
                elif shape == "sphere":
                    placed_area = create_obstacle_sphere(
                        self.stage, prim_path, cx, cy, radius)
                elif shape == "diamond":
                    placed_area = create_obstacle_diamond(
                        self.stage, prim_path, cx, cy, dx, dy, h)
                elif shape == "oval":
                    placed_area = create_obstacle_oval(
                        self.stage, prim_path, cx, cy, rx, ry, h)
                elif shape == "cylinder":
                    placed_area = create_obstacle_cylinder(
                        self.stage, prim_path, cx, cy, radius, h)

                # Style
                _set_color(prim_path, color, self.stage)

                # Physics
                apply_rigid_body_physics(self.stage, prim_path, mass,
                                         friction=0.9 if mass > OBSTACLE_LIGHT_THRESH else 0.4)

                # Record
                self.obstacles.append({
                    "path": prim_path, "cx": cx, "cy": cy, "radius": obs_r,
                    "mass": mass, "shape": shape, "footprint": placed_area,
                    "pushable": mass <= OBSTACLE_LIGHT_THRESH,
                })
                self._index += 1
                return placed_area

        return 0.0  # failed to place

    def populate(self, waypoints_xy: list) -> None:
        """
        Spawn obstacles until cumulative footprint reaches TARGET_OBSTACLE_AREA
        or maximum placement attempts are exhausted.
        """
        self.obstacles.clear()
        self._index = 0
        cumulative_area = 0.0
        max_total_attempts = 5000

        attempt = 0
        while cumulative_area < TARGET_OBSTACLE_AREA and attempt < max_total_attempts:
            area = self.spawn_one(waypoints_xy)
            cumulative_area += area
            attempt += 1

        light = sum(1 for o in self.obstacles if o["pushable"])
        heavy = len(self.obstacles) - light
        coverage_pct = 100.0 * cumulative_area / ARENA_AREA
        print(f"[OK] Obstacles placed: {len(self.obstacles)} total "
              f"({light} light / {heavy} heavy), "
              f"coverage: {coverage_pct:.1f}%")

    def nearest_obstacle_distance(self, spot_x: float, spot_y: float) -> float:
        """Return distance (m) to the nearest obstacle center."""
        if not self.obstacles:
            return float("inf")
        min_d = float("inf")
        for obs in self.obstacles:
            d = distance_2d((spot_x, spot_y), (obs["cx"], obs["cy"])) - obs["radius"]
            if d < min_d:
                min_d = d
        return max(0.0, min_d)

    def remove_prims(self) -> None:
        """Delete all obstacle prims from the stage (for episode reset)."""
        for obs in self.obstacles:
            prim = self.stage.GetPrimAtPath(obs["path"])
            if prim.IsValid():
                self.stage.RemovePrim(obs["path"])
        self.obstacles.clear()
        self._index = 0


# ─────────────────────────────────────────────────────────────────────────────
# WAYPOINT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_waypoints(rng: np.random.Generator) -> dict:
    """
    Pre-compute all 25 waypoint positions (Final Optimized spec).

    Placement rules:
      A   : exactly WAYPOINT_DIST_A (24 m) from the start point (0, 0).
      B–Y : each at least WAYPOINT_SPACING_BZ (30 m) from the previous.
      Non-adjacent pairs: no constraint.
      Re-roll direction until a valid in-arena placement is found.

    All positions are pre-computed for obstacle-clearance checks.
    Only one marker is ever live in the scene at a time (sequential spawn).
    Returns {label: np.array([x, y])}.
    """
    waypoints = {}
    start = np.array([0.0, 0.0])  # Spot start position — NOT a waypoint

    # ── Waypoint A — exactly 24 m from (0, 0) ────────────────────────────
    label_a = WAYPOINT_LABELS[0]
    placed_a = False
    for _ in range(10_000):
        angle = rng.uniform(0, 2 * math.pi)
        candidate = start + np.array([WAYPOINT_DIST_A * math.cos(angle),
                                       WAYPOINT_DIST_A * math.sin(angle)])
        if inside_arena(candidate[0], candidate[1], margin=WAYPOINT_BOUNDARY_MARGIN):
            waypoints[label_a] = candidate
            placed_a = True
            break
    if not placed_a:
        # 24 m from center is always inside 23 m effective radius — shouldn't happen
        waypoints[label_a] = np.array([WAYPOINT_DIST_A, 0.0])
        print(f"  [WARN] Waypoint A: default placement used along +X axis")

    # ── Waypoints B–Y — each ≥ 30 m from previous ────────────────────────
    prev = waypoints[label_a]
    for label in WAYPOINT_LABELS[1:]:
        placed = False
        for _ in range(10_000):
            angle = rng.uniform(0, 2 * math.pi)
            candidate = prev + np.array([WAYPOINT_SPACING_BZ * math.cos(angle),
                                          WAYPOINT_SPACING_BZ * math.sin(angle)])
            if inside_arena(candidate[0], candidate[1], margin=WAYPOINT_BOUNDARY_MARGIN):
                waypoints[label] = candidate
                placed = True
                break
        if not placed:
            # Dense grid scan — last resort to find any valid direction
            for i in range(36_000):
                angle = i * (2 * math.pi / 36_000)
                c = prev + np.array([WAYPOINT_SPACING_BZ * math.cos(angle),
                                      WAYPOINT_SPACING_BZ * math.sin(angle)])
                if inside_arena(c[0], c[1], margin=WAYPOINT_BOUNDARY_MARGIN):
                    waypoints[label] = c
                    placed = True
                    break
        if not placed:
            # Absolute fallback — random inside arena (spacing cannot be satisfied)
            waypoints[label] = random_inside_arena(margin=WAYPOINT_BOUNDARY_MARGIN, rng=rng)
            print(f"  [WARN] Waypoint {label}: 30 m spacing infeasible from "
                  f"({prev[0]:+.1f}, {prev[1]:+.1f}); placed at "
                  f"({waypoints[label][0]:+.1f}, {waypoints[label][1]:+.1f})")
        prev = waypoints[label]

    return waypoints


# ─────────────────────────────────────────────────────────────────────────────
# WAYPOINT VISUAL MARKERS
# ─────────────────────────────────────────────────────────────────────────────

def spawn_waypoint_marker(stage, label: str, pos) -> str:
    """
    Spawn a single flag-on-pole marker for one waypoint.

    Part of the sequential spawning model — only one marker exists in
    the scene at a time.  Returns the parent prim path so the caller can
    despawn it with a single stage.RemovePrim(path) call.

    Waypoint A uses bright green; all others use bright yellow.
    """
    cx = float(pos[0])
    cy = float(pos[1])
    color = COLOR_START_WAYPOINT if label == WAYPOINT_LABELS[0] else COLOR_WAYPOINT

    parent_path = f"/World/Waypoints/WP_{label}"

    # ── Pole ─────────────────────────────────────────────────────────────
    pole_path = f"{parent_path}/Pole"
    pole = UsdGeom.Cylinder.Define(stage, pole_path)
    pole.GetRadiusAttr().Set(WP_POLE_RADIUS)
    pole.GetHeightAttr().Set(WP_POLE_HEIGHT)
    xp = UsdGeom.Xformable(pole.GetPrim())
    xp.AddTranslateOp().Set(Gf.Vec3d(cx, cy, WP_POLE_HEIGHT / 2))
    pole.GetDisplayColorAttr().Set([COLOR_FLAG_POLE])

    # ── Flag banner ───────────────────────────────────────────────────────
    flag_path = f"{parent_path}/Flag"
    flag = UsdGeom.Mesh.Define(stage, flag_path)
    fz_bot = WP_POLE_HEIGHT - WP_FLAG_HEIGHT
    fz_top = WP_POLE_HEIGHT
    hfw    = WP_FLAG_WIDTH / 2
    hfd    = WP_FLAG_DEPTH / 2
    fcx    = cx + hfw  # banner left edge aligns with pole
    pts = [
        Gf.Vec3f(fcx - hfw, cy - hfd, fz_bot),
        Gf.Vec3f(fcx + hfw, cy - hfd, fz_bot),
        Gf.Vec3f(fcx + hfw, cy + hfd, fz_bot),
        Gf.Vec3f(fcx - hfw, cy + hfd, fz_bot),
        Gf.Vec3f(fcx - hfw, cy - hfd, fz_top),
        Gf.Vec3f(fcx + hfw, cy - hfd, fz_top),
        Gf.Vec3f(fcx + hfw, cy + hfd, fz_top),
        Gf.Vec3f(fcx - hfw, cy + hfd, fz_top),
    ]
    flag.GetPointsAttr().Set(pts)
    flag.GetFaceVertexCountsAttr().Set([4, 4, 4, 4, 4, 4])
    flag.GetFaceVertexIndicesAttr().Set([
        0, 1, 2, 3,   # bottom
        4, 7, 6, 5,   # top
        0, 4, 5, 1,   # front
        2, 6, 7, 3,   # back
        0, 3, 7, 4,   # left
        1, 5, 6, 2,   # right
    ])
    flag.GetDisplayColorAttr().Set([color])

    return parent_path


def remove_waypoint_markers(stage, marker_paths: dict) -> None:
    """Remove all waypoint flag prims (pole + banner) via parent prim removal."""
    for path in marker_paths.values():
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            stage.RemovePrim(path)


# ─────────────────────────────────────────────────────────────────────────────
# SPEED CONTROL — proximity-based
# ─────────────────────────────────────────────────────────────────────────────

def compute_speed_command(spot_x: float, spot_y: float,
                          target_x: float, target_y: float,
                          obstacle_mgr: ObstacleManager) -> tuple:
    """
    Compute [forward, lateral, yaw] RL action for Spot.

    Speed is scaled down linearly when the nearest obstacle is within
    OBSTACLE_SLOW_RADIUS meters. Yaw command steers toward the current waypoint.

    Returns
    -------
    command : (forward, lateral, yaw) tuple
    dist_to_target : float
    nearest_obs_dist : float
    """
    dx = target_x - spot_x
    dy = target_y - spot_y
    dist_to_target = math.sqrt(dx * dx + dy * dy)

    # Heading toward target
    target_heading = math.atan2(dy, dx)

    # Speed scaling based on obstacle proximity
    nearest_d = obstacle_mgr.nearest_obstacle_distance(spot_x, spot_y)
    if nearest_d >= OBSTACLE_SLOW_RADIUS:
        speed_scale = 1.0
    elif nearest_d <= 0.0:
        speed_scale = SPOT_MIN_SPEED_MPS / SPOT_MAX_SPEED_MPS
    else:
        speed_scale = nearest_d / OBSTACLE_SLOW_RADIUS

    forward = SPOT_MAX_SPEED_MPS * max(speed_scale, SPOT_MIN_SPEED_MPS / SPOT_MAX_SPEED_MPS)
    forward = min(forward, dist_to_target)  # don't overshoot

    # Simple P-controller yaw
    yaw = float(np.clip(target_heading * 2.0, -1.5, 1.5))

    return (forward, 0.0, yaw), dist_to_target, nearest_d


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENVIRONMENT CLASS
# ─────────────────────────────────────────────────────────────────────────────

class CircularWaypointEnv:
    """
    Full circular waypoint navigation environment for RL training.

    Usage
    -----
    env = CircularWaypointEnv(world, stage)
    env.reset(episode=0)
    while not done:
        obs, reward, done, info = env.step()
    """

    def __init__(self, world: World, stage):
        self.world = world
        self.stage = stage

        # Episode state
        self.episode              = 0
        self.rng                  = None
        self.waypoints            = {}
        self.marker_paths         = {}
        self.obstacle_mgr         = None
        self.spot                 = None
        self.current_wp_idx       = 0      # index into WAYPOINT_LABELS
        self.step_count           = 0
        self.done                 = False

        # Score / timing (new reward system)
        self.score                = EPISODE_START_SCORE  # running score
        self.sim_time             = 0.0                  # elapsed sim seconds
        self.episode_total_reward = 0.0                  # cumulative RL reward signal
        self.waypoints_reached    = []                   # ordered list of collected labels

    # ── Observation ──────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        Build the 49-dimensional observation vector.

        [0:3]   base linear velocity (vx, vy, vz)
        [3:6]   base angular velocity (wx, wy, wz)
        [6:9]   projected gravity vector in body frame (gx, gy, gz)
        [9:21]  joint positions (12)
        [21:33] joint velocities (12)
        [33:45] previous action (12)
        [45:47] (dx, dy) to active waypoint
        [47]    distance to active waypoint
        [48]    nearest obstacle distance
        """
        if self.spot is None:
            return np.zeros(49, dtype=np.float32)

        pos, heading, vel, ang_vel = self.spot.get_state()
        sensor_data = self.spot.get_sensor_data()

        # Gravity projection (simplified: assume flat ground, gravity points down)
        g_body = np.array([0.0, 0.0, -9.81])

        # Joint positions / velocities — stubbed from sensor data
        joint_pos = np.zeros(12)
        joint_vel = np.zeros(12)
        prev_act  = np.zeros(12)

        # Waypoint relative vector
        if self.current_wp_idx < len(WAYPOINT_LABELS):
            active_label = WAYPOINT_LABELS[self.current_wp_idx]
            wp_pos = self.waypoints[active_label]
            dx = wp_pos[0] - pos[0]
            dy = wp_pos[1] - pos[1]
            dist_wp = math.sqrt(dx * dx + dy * dy)
        else:
            dx, dy, dist_wp = 0.0, 0.0, 0.0

        nearest_obs = float(self.obstacle_mgr.nearest_obstacle_distance(pos[0], pos[1])
                            if self.obstacle_mgr else float("inf"))
        nearest_obs = min(nearest_obs, 20.0)  # clip for observation normalization

        obs = np.concatenate([
            vel,            # 3
            ang_vel,        # 3
            g_body,         # 3
            joint_pos,      # 12
            joint_vel,      # 12
            prev_act,       # 12
            [dx, dy],       # 2
            [dist_wp],      # 1
            [nearest_obs],  # 1
        ]).astype(np.float32)  # total: 49
        return obs

    # ── Reward computation ───────────────────────────────────────────────────

    def compute_reward(self) -> dict:
        """
        Return a dict of named reward components for this step.

        All components are in *score points* (same units as self.score).
        Add new shaping terms here; step() sums the totals.
        """
        step_dt = PHYSICS_DT  # seconds per physics tick
        return {
            # Core time-decay: lose TIME_DECAY_PER_SEC points every simulated second
            "time_decay": -(TIME_DECAY_PER_SEC * step_dt),
            # Placeholder hooks — uncomment and tune when needed:
            # "energy_penalty": 0.0,
            # "smoothness_bonus": 0.0,
        }

    # ── CSV logging ──────────────────────────────────────────────────────────

    def _log_to_csv(self, reason: str) -> None:
        """Append one row per episode to the training CSV log."""
        file_exists = os.path.isfile(CSV_LOG_PATH)
        try:
            with open(CSV_LOG_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(CSV_HEADERS)
                writer.writerow([
                    self.episode + 1,
                    len(self.waypoints_reached),
                    round(self.sim_time, 2),
                    round(self.score, 2),
                ])
        except IOError as exc:
            print(f"[WARN] CSV log write failed: {exc}")

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self, episode: int = 0) -> np.ndarray:
        """
        Reset the environment for a new episode.
        Re-randomizes obstacles and waypoints.
        """
        self.episode = episode
        seed = args.seed if args.seed is not None else episode
        self.rng = np.random.default_rng(seed)
        np.random.seed(seed)  # also seed legacy numpy RNG used in Spot_1.py

        print(f"\n{'─'*72}")
        print(f"EPISODE {episode + 1}  |  seed = {seed}")
        print(f"{'─'*72}")

        # ── Clear previous episode artefacts ──────────────────────────────
        if self.obstacle_mgr is not None:
            self.obstacle_mgr.remove_prims()
        if self.marker_paths:
            remove_waypoint_markers(self.stage, self.marker_paths)
        self.marker_paths = {}

        # ── Pre-compute all waypoint positions ────────────────────────────
        self.waypoints = generate_waypoints(self.rng)

        # Spawn only waypoint A — sequential model (one marker at a time)
        first_label = WAYPOINT_LABELS[0]
        first_path  = spawn_waypoint_marker(self.stage, first_label,
                                             self.waypoints[first_label])
        self.marker_paths = {first_label: first_path}

        # Debug print
        print(f"  Waypoints pre-computed ({len(self.waypoints)}):")
        for lbl, pos in self.waypoints.items():
            print(f"    {lbl}: ({pos[0]:+.2f}, {pos[1]:+.2f})")
        print(f"  Active marker: {first_label} at "
              f"({self.waypoints[first_label][0]:+.2f}, "
              f"{self.waypoints[first_label][1]:+.2f})")

        # ── Populate obstacles ─────────────────────────────────────────────
        self.obstacle_mgr = ObstacleManager(self.stage, self.rng)
        # Include Spot's start position so no obstacle spawns on top of it
        protected_xy = [[SPOT_START_X, SPOT_START_Y]] + [list(p) for p in self.waypoints.values()]
        self.obstacle_mgr.populate(protected_xy)

        # ── Reset Spot ─────────────────────────────────────────────────────
        if self.spot is None:
            self.spot = SpotRobot(
                self.world, self.stage,
                prim_path="/World/Spot",
                name="Spot",
                position=np.array([SPOT_START_X, SPOT_START_Y, SPOT_START_Z]),
            )
        else:
            # Teleport to start and arm the first-step init sequence in the callback
            self.spot.reset_for_episode(
                position=np.array([SPOT_START_X, SPOT_START_Y, SPOT_START_Z])
            )

        # Stabilisation: first physics step triggers initialize()+post_reset()+
        # set_joints_default_state(); subsequent steps run forward(dt,[0,0,0])
        # via the registered callback, keeping the robot standing upright.
        for _ in range(20):
            self.world.step(render=False)

        # ── Reset episode counters ─────────────────────────────────────────
        self.current_wp_idx       = 0                    # navigate to A first (24 m from start)
        self.step_count           = 0
        self.done                 = False
        self.score                = EPISODE_START_SCORE  # 300 pts
        self.sim_time             = 0.0
        self.episode_total_reward = 0.0
        self.waypoints_reached    = []

        active_label = WAYPOINT_LABELS[self.current_wp_idx]
        wp = self.waypoints[active_label]

        print(f"[OK] Episode {episode + 1} ready | score={self.score:.0f} pts | "
              f"first target: waypoint {active_label} "
              f"at ({wp[0]:+.2f}, {wp[1]:+.2f})")

        return self._get_obs()

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self) -> tuple:
        """
        Execute one simulation step.

        Returns
        -------
        obs    : np.ndarray  (49,)
        reward : float       (step-level contribution to the score bank)
        done   : bool
        info   : dict        (includes running score, WP list, reason)
        """
        if self.done:
            return self._get_obs(), 0.0, True, {}

        # ── All waypoints already collected ──────────────────────────────────
        if self.current_wp_idx >= len(WAYPOINT_LABELS):
            self.done = True
            reason = "all_waypoints_collected"
            self._log_to_csv(reason)
            print(f"\n{'='*72}")
            print(f"  ALL {WAYPOINT_COUNT} WAYPOINTS COLLECTED! Episode complete.")
            print(f"  Visit order: {' → '.join(self.waypoints_reached)}")
            print(f"  Final score: {self.score:.1f}")
            print(f"{'='*72}")
            return self._get_obs(), 0.0, True, {"reason": reason, "score": self.score,
                                                 "waypoints_reached": self.waypoints_reached}

        active_label = WAYPOINT_LABELS[self.current_wp_idx]
        wp = self.waypoints[active_label]

        # ── Pre-step state ────────────────────────────────────────────────────
        pos, heading, vel, ang_vel = self.spot.get_state()

        command, dist_to_wp, nearest_obs = compute_speed_command(
            pos[0], pos[1], wp[0], wp[1], self.obstacle_mgr
        )
        self.spot.set_command(command)

        # ── Physics tick ──────────────────────────────────────────────────────
        self.world.step(render=True)
        self.step_count += 1
        self.sim_time = self.step_count * PHYSICS_DT

        # ── Post-step state ───────────────────────────────────────────────────
        pos, heading, vel, ang_vel = self.spot.get_state()

        # Debug: log z for first 30 steps
        if self.step_count <= 30:
            print(f"    [DBG] step={self.step_count}  z={pos[2]:.4f}")

        # ── Score/reward bookkeeping ──────────────────────────────────────────
        reason = "running"
        step_reward = 0.0

        # Time-decay applies every step (-1 pt / simulated second)
        reward_components = self.compute_reward()
        time_penalty = reward_components["time_decay"]   # e.g. -0.002 per step
        self.score += time_penalty
        step_reward  += time_penalty

        # Fall detection
        if pos[2] < FALL_HEIGHT_THRESHOLD:
            print(f"  [FALL] Spot fell at step {self.step_count} "
                  f"(z={pos[2]:.3f} m). Score zeroed.")
            self.score = 0.0
            self.done  = True
            reason = "fall"

        # Waypoint collection
        elif dist_to_wp < WAYPOINT_REACH_DIST:
            self.score += WAYPOINT_REWARD
            step_reward += WAYPOINT_REWARD
            self.waypoints_reached.append(active_label)
            print(f"  [WP] Waypoint {active_label} reached! "
                  f"(step {self.step_count}, "
                  f"sim_t={self.sim_time:.1f}s, "
                  f"score={self.score:.1f}, "
                  f"obs_d={nearest_obs:.2f} m)")

            # Despawn the collected marker
            remove_waypoint_markers(self.stage, self.marker_paths)
            self.marker_paths = {}

            self.current_wp_idx += 1

            # Check if that was the last waypoint
            if self.current_wp_idx >= len(WAYPOINT_LABELS):
                self.done = True
                reason = "all_waypoints_collected"
                print(f"\n{'='*72}")
                print(f"  ALL {WAYPOINT_COUNT} WAYPOINTS COLLECTED!")
                print(f"  Visit order: {' → '.join(self.waypoints_reached)}")
                print(f"  Final score: {self.score:.1f}")
                print(f"{'='*72}")
            else:
                # Spawn the next waypoint marker
                next_label = WAYPOINT_LABELS[self.current_wp_idx]
                next_path  = spawn_waypoint_marker(self.stage, next_label,
                                                    self.waypoints[next_label])
                self.marker_paths = {next_label: next_path}
                print(f"  [SPAWN] Waypoint {next_label} spawned at "
                      f"({self.waypoints[next_label][0]:+.2f}, "
                      f"{self.waypoints[next_label][1]:+.2f})")

        # Score-depleted termination
        if self.score <= 0.0 and not self.done:
            self.score = 0.0
            self.done  = True
            reason = "score_depleted"
            print(f"  [OUT] Score depleted at step {self.step_count} "
                  f"(sim_t={self.sim_time:.1f}s).")

        self.episode_total_reward += step_reward

        if self.done:
            self._log_to_csv(reason)

        obs  = self._get_obs()
        info = {
            "step":             self.step_count,
            "sim_time":         self.sim_time,
            "score":            self.score,
            "waypoint_idx":     self.current_wp_idx,
            "dist_to_wp":       dist_to_wp,
            "nearest_obs":      nearest_obs,
            "waypoints_reached": list(self.waypoints_reached),
            "waypoint_order":   " → ".join(self.waypoints_reached),
            "reward_components": reward_components,
            "reason":           reason,
        }
        return obs, step_reward, self.done, info


# ─────────────────────────────────────────────────────────────────────────────
# WORLD SETUP (lighting, ground, boundary)
# ─────────────────────────────────────────────────────────────────────────────

def build_world(world: World, stage) -> None:
    """Create the static world elements: lighting, ground, and circular boundary."""

    # Lighting
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(1000.0)
    dome.CreateColorAttr(Gf.Vec3f(0.95, 0.95, 1.0))

    distant_light = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    distant_light.CreateIntensityAttr(2500.0)
    distant_light.CreateAngleAttr(0.53)
    xf = UsdGeom.Xformable(distant_light.GetPrim())
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 135.0))
    print("[OK] Lighting configured")

    # Ground plane
    world.scene.add_default_ground_plane(
        z_position=0,
        name="default_ground_plane",
        prim_path="/World/defaultGroundPlane",
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.01,
    )
    print("[OK] Ground plane added")

    # Add arena floor disc (visual only, green circle)
    disc = UsdGeom.Cylinder.Define(stage, "/World/ArenaFloor")
    disc.GetRadiusAttr().Set(ARENA_RADIUS)
    disc.GetHeightAttr().Set(0.01)
    disc.GetDisplayColorAttr().Set([Gf.Vec3f(0.28, 0.55, 0.28)])  # grass green
    xf2 = UsdGeom.Xformable(disc.GetPrim())
    xf2.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.005))

    # Circular boundary wall
    create_circular_boundary(stage)
    print("[OK] World construction complete")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 72)
    print("TESTING ENVIRONMENT 2 — CIRCULAR WAYPOINT NAVIGATION ARENA")
    print("=" * 72)

    # ── Create World ──────────────────────────────────────────────────────────
    world = World(
        physics_dt=PHYSICS_DT,
        rendering_dt=RENDERING_DT,
        stage_units_in_meters=1.0,
    )
    stage = omni.usd.get_context().get_stage()

    build_world(world, stage)

    # ── Initial world reset ───────────────────────────────────────────────────
    world.reset()
    print("[OK] World physics enabled")

    # ── Create RL Environment ─────────────────────────────────────────────────
    env = CircularWaypointEnv(world, stage)

    # ── Training loop ─────────────────────────────────────────────────────────
    for episode_idx in range(args.episodes):
        obs  = env.reset(episode=episode_idx)
        done = False

        print(f"\n[RUN] Episode {episode_idx + 1}/{args.episodes} started "
              f"(score bank: {EPISODE_START_SCORE:.0f} pts, "
              f"-{TIME_DECAY_PER_SEC:.1f} pt/s, "
              f"+{WAYPOINT_REWARD:.0f} pt/WP)")

        while simulation_app.is_running() and not done:
            obs, reward, done, info = env.step()

            # Heartbeat every ~5 seconds of sim time
            if env.step_count % 2500 == 0:
                wps_done = len(info["waypoints_reached"])
                print(f"  [T={info['sim_time']:6.1f}s] "
                      f"score: {info['score']:6.1f}  "
                      f"WPs: {wps_done}/{WAYPOINT_COUNT}  "
                      f"dist_to_next: {info['dist_to_wp']:.2f} m  "
                      f"nearest_obs: {info['nearest_obs']:.2f} m")

        wps_final  = len(info["waypoints_reached"])
        order_str  = info.get("waypoint_order", "—")
        reason_str = info.get("reason", "unknown")
        print(f"\n[DONE] Episode {episode_idx + 1} finished: "
              f"{wps_final}/{WAYPOINT_COUNT} WPs collected  |  "
              f"final score = {info['score']:.1f}  |  "
              f"sim_time = {info['sim_time']:.1f}s  |  "
              f"reason = {reason_str}")
        if wps_final > 0:
            print(f"       Visit order: {order_str}")

    print("\n" + "=" * 72)
    print("ALL EPISODES COMPLETE. Closing Isaac Sim.")
    print("=" * 72)
    simulation_app.close()


if __name__ == "__main__":
    main()
