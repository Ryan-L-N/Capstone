"""
Testing Environment — Circular Waypoint Navigation Arena with Policy Testing
==============================================================================
Circular training environment for Boston Dynamics Spot RL locomotion.
Built with direct SpotFlatTerrainPolicy control (no wrapper).
Supports loading and testing trained navigation policies.

Environment Geometry
--------------------
  - Shape    : Circle, diameter 50 m (radius 25 m)
  - Center   : (0, 0)
  - Terrain  : Flat ground plane (open arena, no walls)

Waypoints (Final Optimized)
----------------------------
  - 25 waypoints labeled A – Y
  - A placed 20 m from start (0,0)
  - B–Y each ≥ 40 m from previous
  - Sequential spawning (one marker at a time)

Navigation Behavior (Optimized Locomotion)
------------------------------------------
  - PRIMARY: Walk forward naturally (no backward, strafing, side-stepping)
  - ALIGNMENT: Rotate in place to face waypoint before moving forward
  - CORRECTIONS: Small incremental heading adjustments while walking
  - RESTRICTIONS: Unnatural motions only for collision avoidance

Obstacle Interaction (Weight-Based Physics)
--------------------------------------------
  - MOVEABLE (≤ 32.7 kg): Spot can push, roll, or tip these obstacles
                          Dynamic rigid bodies with low friction
                          Examples: small cubes, lightweight cylinders, spheres
  
  - NON-MOVEABLE (> 32.7 kg): Immovable obstacles (static rigid bodies)
                              Spot must navigate around using natural locomotion
                              High friction, cannot be pushed or moved

Small Static Obstacles
----------------------
  - SIZE: Golf ball to softball (1.7-4 inches / 0.043-0.102 m)
  - PHYSICS: Unmovable (static rigid bodies, infinite mass)
  - SHAPES: All main obstacle shapes (sphere, cube, cylinder, etc.)
  - COVERAGE: 10% of remaining area (after main obstacles take 20%)
              = 8% of total arena area
  - PLACEMENT: Random each episode, no overlap with obstacles/waypoints/spawn
  - PURPOSE: Fixed hazards requiring precise navigation and avoidance
  - COLOR: Dark gray (visually distinct from waypoints and main obstacles)

Author : Cole (MS for Autonomy Project)
Date   : February 2026
Spec   : Cole_md.md
"""

import csv
import math
import os
import string
import argparse
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing BEFORE SimulationApp
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Testing Environment — Circular Waypoint Arena with Policy Testing")
parser.add_argument("--headless",  action="store_true", help="Run without GUI")
parser.add_argument("--episodes",  type=int, default=1,  help="Number of training episodes")
parser.add_argument("--seed",      type=int, default=None, help="Global RNG seed (None = random)")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained navigation policy checkpoint")
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
sys.path.insert(0, r"C:\Users\user\Desktop\Capstone_vs_1.2\Immersive-Modeling-and-Simulation-for-Autonomy\Experiments\Cole")
sys.path.insert(0, r"C:\Users\user\Desktop\Capstone_vs_1.2\Immersive-Modeling-and-Simulation-for-Autonomy\Experiments\Cole\RL_FOLDER_VS3")
from navigation_policy import NavigationPolicy, scale_action  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# LOAD TRAINED NAVIGATION POLICY
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
navigation_policy = None
if args.checkpoint:
    print("\n" + "=" * 72)
    print("LOADING TRAINED NAVIGATION POLICY")
    print("=" * 72)
    print(f"Checkpoint: {args.checkpoint}")
    
    navigation_policy = NavigationPolicy(obs_dim=34, action_dim=3).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'policy_state_dict' in checkpoint:
        navigation_policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"[OK] Loaded from iteration {checkpoint.get('iteration', 'unknown')}")
        print(f"[OK] Training stage: {checkpoint.get('current_stage', 'unknown')}")
    else:
        navigation_policy.load_state_dict(checkpoint)
    
    navigation_policy.eval()
    print("[OK] Policy loaded successfully")
    print("=" * 72 + "\n")

# Store training stage from checkpoint for observation encoding
training_stage = checkpoint.get('current_stage', 0) if (args.checkpoint and 'checkpoint' in locals()) else 0

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
SPOT_MASS_KG         = 32.7          # Boston Dynamics Spot weight
SPOT_LENGTH          = 1.1           # meters (bounding box)
SPOT_WIDTH           = 0.5           # meters
SPOT_HEIGHT          = 0.6           # meters

FALL_HEIGHT_THRESHOLD = 0.25         # meters — z below this means Spot has fallen

# Speed
SPOT_MAX_SPEED_MPS   = 5 * 0.44704  # 5 mph → m/s  ≈ 2.235 m/s
SPOT_MIN_SPEED_MPS   = 0.3          # minimum crawl speed
OBSTACLE_SLOW_RADIUS = 2.0          # meters — start slowing at this distance

# Navigation Control (Optimized Locomotion)
HEADING_ALIGNMENT_THRESHOLD = math.radians(10)  # 10 degrees — must align before moving forward
HEADING_CORRECTION_GAIN = 0.5                   # Proportional gain for incremental corrections
MAX_TURN_RATE = 1.0                             # rad/s — max angular velocity for turning
FORWARD_PRIORITY = True                         # Always prefer forward movement over strafing

# Obstacles
OBSTACLE_AREA_FRAC   = 0.10         # 10% of arena — moveable obstacles only
OBSTACLE_MIN_FOOT    = 0.0174       # m²  (27 in²)
OBSTACLE_MAX_FOOT    = SPOT_LENGTH * SPOT_WIDTH  # 0.55 m²

# Obstacle Weight Categories (based on Spot's interaction capability)
OBSTACLE_MOVEABLE_MAX = SPOT_MASS_KG * 0.25  # kg  (8.175 kg)  — Moveable: pushable or tippable by Spot (25% of Spot mass)
OBSTACLE_MIN_MASS     = 0.0909        # kg  (0.2 lb)   — Minimum obstacle mass
OBSTACLE_MAX_MASS     = 2 * SPOT_MASS_KG  # 65.4 kg — Non-moveable: immovable, must navigate around
OBSTACLE_CLEARANCE_BOUNDARY = 3.0  # keep obstacles this far inside boundary
OBSTACLE_CLEARANCE_WAYPOINT = 5.0  # keep obstacles this far from waypoints
STARTING_ZONE_BUFFER = 2.0          # meters — 2m×2m buffer zone around (0,0) - no obstacles

# Small Static Obstacles (unmovable hazards)
SMALL_OBSTACLE_MIN_SIZE = 0.043     # meters (1.7 inches - golf ball)
SMALL_OBSTACLE_MAX_SIZE = 0.102     # meters (4.0 inches - softball)
SMALL_OBSTACLE_COVERAGE = 0.10 / 3  # ~3.33% of arena (10% total / 3 types)
SMALL_OBSTACLE_CLEARANCE = 0.3      # meters — minimum distance between small obstacles
COLOR_SMALL_OBSTACLE = Gf.Vec3f(0.4, 0.4, 0.4)  # dark gray — visually distinct

# Waypoints
WAYPOINT_COUNT       = 25
WAYPOINT_LABELS      = list(string.ascii_uppercase[:WAYPOINT_COUNT])  # A … Y
WAYPOINT_DIST_A      = 20.0         # meters — A is placed exactly 20 m from (0,0)
WAYPOINT_SPACING_BZ  = 40.0         # meters — B–Y each at least 40 m from previous (Stage 4)
WAYPOINT_REACH_DIST  = 1.0          # meters — threshold to "collect" a waypoint
WAYPOINT_BOUNDARY_MARGIN = 2.0      # meters — keep waypoints inside circle

# Colors (RGB)
COLOR_MOVEABLE_OBSTACLE = Gf.Vec3f(1.0,  0.55, 0.0)   # orange  - moveable/pushable obstacles
COLOR_NON_MOVEABLE_OBSTACLE = Gf.Vec3f(0.27, 0.51, 0.71)  # steel blue - non-moveable/immovable obstacles
COLOR_WAYPOINT       = Gf.Vec3f(1.0,  0.95, 0.0)   # bright yellow
COLOR_START_WAYPOINT = Gf.Vec3f(0.2,  0.9,  0.2)   # bright green
COLOR_FLAG_POLE      = Gf.Vec3f(0.88, 0.88, 0.88)  # light grey pole

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
EPISODE_START_SCORE  = 300.0    # points at episode start
TIME_DECAY_PER_SEC   = 1.0      # points lost per real sim-second
WAYPOINT_REWARD      = 15.0     # points awarded per waypoint collected

# Misc
ARENA_AREA           = math.pi * ARENA_RADIUS ** 2   # ≈ 1963.5 m²
TARGET_OBSTACLE_AREA = ARENA_AREA * OBSTACLE_AREA_FRAC  # ≈ 392.7 m²

# CSV logging (persists across training runs)
CSV_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Baseline_CSV.csv")
CSV_HEADERS  = ["Episode", "Waypoints_Reached", "Failure_Reason", "Final_Score"]

# Failure reason mappings
FAILURE_FELL_OVER = "Fell Over"
FAILURE_RAN_OUT = "Ran Out of Points"
FAILURE_COMPLETED = "Completed All Waypoints"
FAILURE_OTHER = "Other"

print("=" * 72)
print("TESTING ENVIRONMENT — CIRCULAR WAYPOINT NAVIGATION ARENA")
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


def quaternion_to_yaw(quat: np.ndarray) -> float:
    """
    Convert quaternion [w, x, y, z] to yaw angle (rotation around z-axis).
    Returns angle in radians [-π, π].
    """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    # Yaw (z-axis rotation)
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


def raycast_obstacle_dists(spot_x: float, spot_y: float, yaw: float,
                           obstacle_mgr: 'ObstacleManager', num_rays: int = 18,
                           ray_length: float = 5.0) -> np.ndarray:
    """
    Perform per-ray directional raycasting to detect obstacles (matching training env).
    
    Each ray checks ray-circle intersection with every obstacle and the arena
    boundary, returning the nearest hit distance. Obstacles are treated as
    circles using a bounding radius derived from their dimensions.
    
    Args:
        spot_x, spot_y: robot position
        yaw: robot heading (radians)
        obstacle_mgr: obstacle manager for collision checks
        num_rays: number of rays (default 18, matching VS3 training config)
        ray_length: max raycast distance (default 5.0 m, matches training config)
    
    Returns:
        numpy array of normalized obstacle distances [0, 1], length num_rays
    """
    # Pre-compute obstacle bounding circles: (x, y, radius)
    obs_circles = []
    if obstacle_mgr.obstacles:
        for obs in obstacle_mgr.obstacles:
            ox, oy = obs["pos"][0], obs["pos"][1]
            dims = obs["dims"]
            # Bounding radius: half the diagonal of the footprint (width, depth)
            bounding_r = np.sqrt(dims[0]**2 + dims[1]**2) / 2.0
            obs_circles.append((ox, oy, bounding_r))
    if hasattr(obstacle_mgr, 'small_obstacles') and obstacle_mgr.small_obstacles:
        for obs in obstacle_mgr.small_obstacles:
            ox, oy = obs["pos"][0], obs["pos"][1]
            bounding_r = obs["size"] / 2.0
            obs_circles.append((ox, oy, bounding_r))
    
    distances = []
    
    for i in range(num_rays):
        # Ray angle in robot frame (0 to 2π)
        ray_angle = 2 * np.pi * i / num_rays
        # Convert to world frame
        world_angle = yaw + ray_angle
        
        cos_a = np.cos(world_angle)
        sin_a = np.sin(world_angle)
        
        # Start with arena boundary distance
        dist = ray_to_boundary(spot_x, spot_y, world_angle, ray_length)
        
        # Check ray-circle intersection with each obstacle
        for obs_x, obs_y, obs_r in obs_circles:
            # Vector from ray origin to obstacle center
            dx = obs_x - spot_x
            dy = obs_y - spot_y
            
            # Project obstacle center onto ray direction
            t_center = dx * cos_a + dy * sin_a
            
            if t_center < 0:
                # Obstacle is behind the ray origin
                continue
            
            # Perpendicular distance from obstacle center to ray line
            perp_dist_sq = (dx * dx + dy * dy) - t_center * t_center
            
            if perp_dist_sq > obs_r * obs_r:
                # Ray misses this obstacle
                continue
            
            # Ray hits obstacle — compute intersection distance
            half_chord = np.sqrt(obs_r * obs_r - perp_dist_sq)
            t_hit = t_center - half_chord
            
            if t_hit > 0 and t_hit < dist:
                dist = t_hit
        
        # Normalize to [0, 1]
        distances.append(dist / ray_length)
    
    return np.array(distances, dtype=np.float32)


def ray_to_boundary(x: float, y: float, angle: float, ray_length: float = 5.0) -> float:
    """
    Calculate distance from point to arena boundary along a ray.
    
    Ray: (x, y) + t * (cos(angle), sin(angle))
    Circular boundary: x^2 + y^2 = R^2
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # Quadratic formula: solve for ray-circle intersection
    a = cos_a**2 + sin_a**2  # Always 1.0
    b = 2 * (x * cos_a + y * sin_a)
    c = x**2 + y**2 - ARENA_RADIUS**2
    
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return ray_length  # No intersection, ray doesn't hit boundary
    
    t = (-b + np.sqrt(discriminant)) / (2 * a)
    return max(0, min(ray_length, t))  # Clamp to ray_length


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
    """Create a rectangular box mesh."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    hw, hd, hh = w / 2, d / 2, h / 2
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
    """Create a square pillar mesh."""
    create_rectangle_mesh(stage, path, side, side, height, color)


def create_trapezoid_mesh(stage, path: str, w_bot: float, w_top: float,
                          depth: float, height: float, color) -> None:
    """Create a trapezoidal prism mesh."""
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
    """Create a UV-sphere mesh."""
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
    """Create a diamond/pyramid mesh."""
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
    """Create an elliptical cylinder mesh."""
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
    """Create a cylindrical mesh."""
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
        self.obstacles = []  # list of {path, pos, dims, mass}
        self.small_obstacles = []  # list of small static obstacles {path, pos, size, shape}

    def calculate_footprint_area(self, shape: str, dims: tuple) -> float:
        """
        Calculate actual ground footprint area based on shape type.
        Uses actual geometric areas, not bounding boxes.
        
        dims format depends on shape (from spawn_one):
          - rectangle, square, trapezoid: (width/side, depth, height)
          - sphere, cylinder: (diameter, diameter, height) -> area = π * r²
          - oval: (2*r_major, 2*r_minor, height) -> area = π * r_major * r_minor
          - diamond: (base, base, height) -> area ≈ base² / 2
        """
        if shape == "rectangle":
            return dims[0] * dims[1]  # width * depth
        elif shape == "square":
            return dims[0] * dims[1]  # side²
        elif shape == "trapezoid":
            # dims[0] already stores average_width
            return dims[0] * dims[1]  # avg_width * depth
        elif shape == "sphere":
            r = dims[0] / 2  # diameter to radius
            return math.pi * r * r  # π * r²
        elif shape == "oval":
            r_major = dims[0] / 2  # diameter to radius
            r_minor = dims[1] / 2  # diameter to radius
            return math.pi * r_major * r_minor  # π * r_major * r_minor
        elif shape == "cylinder":
            r = dims[0] / 2  # diameter to radius
            return math.pi * r * r  # π * r²
        elif shape == "diamond":
            # Diamond (45° rotated square): footprint = base² / 2
            return dims[0] * dims[1] / 2  # base² / 2
        else:
            # Unknown shape: fallback to bounding box
            return dims[0] * dims[1]

    def spawn_one(self, idx: int, margin: float = 1.5, min_spawn_clearance: float = 2.0, 
                   force_weight_class: str = None) -> None:
        """
        Spawn a single random obstacle inside the arena.
        Enforces buffer zone around starting position and clearance from other obstacles.
        
        Weight Categories:
        - Moveable (≤ 32.7 kg):  Pushable or tippable by Spot, dynamic rigid body, low friction
        - Non-moveable (> 32.7 kg): Immovable, static rigid body, Spot must avoid
        
        Args:
            force_weight_class: If specified ('moveable' or 'non_moveable'), force this category
        """
        shape = self.rng.choice(self.SHAPES)
        
        # Keep trying random positions until we find one that's valid
        max_attempts = 100
        for attempt in range(max_attempts):
            pos_2d = random_inside_arena(margin=margin, rng=self.rng)
            
            # Check starting zone buffer (prevent obstacles near Spot's start)
            if distance_2d(pos_2d, [SPOT_START_X, SPOT_START_Y]) < STARTING_ZONE_BUFFER:
                continue
            
            # Check clearance from other obstacles
            too_close = False
            for obs in self.obstacles:
                if distance_2d(pos_2d, obs["pos"]) < min_spawn_clearance:
                    too_close = True
                    break
            
            if not too_close:
                break  # Found valid position
        else:
            # Could not find valid position after max_attempts
            return  # Skip this obstacle
        
        # Sample mass from full range
        mass = self.rng.uniform(OBSTACLE_MIN_MASS, OBSTACLE_MAX_MASS)
        
        # Categorize obstacle by weight or force category
        if force_weight_class == "moveable":
            # Force moveable: sample from moveable range
            mass = self.rng.uniform(OBSTACLE_MIN_MASS, OBSTACLE_MOVEABLE_MAX)
            weight_class = "moveable"
            color = COLOR_MOVEABLE_OBSTACLE
        elif force_weight_class == "non_moveable":
            # Force non-moveable: sample from non-moveable range
            mass = self.rng.uniform(OBSTACLE_MOVEABLE_MAX + 0.1, OBSTACLE_MAX_MASS)
            weight_class = "non_moveable"
            color = COLOR_NON_MOVEABLE_OBSTACLE
        else:
            # Auto-categorize based on sampled mass
            if mass <= OBSTACLE_MOVEABLE_MAX:
                weight_class = "moveable"
                color = COLOR_MOVEABLE_OBSTACLE
            else:
                weight_class = "non_moveable"
                color = COLOR_NON_MOVEABLE_OBSTACLE

        path = f"/World/Obstacles/Obst_{idx:03d}"
        rot_deg = self.rng.uniform(0, 360)

        if shape == "rectangle":
            # Footprint: w × d ∈ [0.0174, 0.55] m²
            w = self.rng.uniform(0.15, 1.1)
            d = self.rng.uniform(0.15, 0.7)
            # Ensure footprint within bounds
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
            # Footprint: side² ∈ [0.0174, 0.55] m²
            min_side = math.sqrt(OBSTACLE_MIN_FOOT)  # 0.132 m
            max_side = math.sqrt(OBSTACLE_MAX_FOOT)  # 0.742 m
            side = self.rng.uniform(min_side, max_side)
            h = self.rng.uniform(0.3, 1.2)
            create_square_mesh(self.stage, path, side, h, color)
            dims = (side, side, h)

        elif shape == "trapezoid":
            # Footprint: avg_width × depth ∈ [0.0174, 0.55] m²
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
            # Footprint: π × r² ∈ [0.0174, 0.55] m²
            min_r = math.sqrt(OBSTACLE_MIN_FOOT / math.pi)  # 0.074 m
            max_r = math.sqrt(OBSTACLE_MAX_FOOT / math.pi)  # 0.419 m
            r = self.rng.uniform(min_r, max_r)
            create_sphere_mesh(self.stage, path, r, color)
            dims = (2 * r, 2 * r, 2 * r)

        elif shape == "diamond":
            # Footprint: base² ∈ [0.0174, 0.55] m²
            min_base = math.sqrt(OBSTACLE_MIN_FOOT)  # 0.132 m
            max_base = math.sqrt(OBSTACLE_MAX_FOOT)  # 0.742 m
            base = self.rng.uniform(min_base, max_base)
            h = self.rng.uniform(0.5, 1.5)
            create_diamond_mesh(self.stage, path, base, h, color)
            dims = (base, base, h)

        elif shape == "oval":
            # Footprint: π × r_major × r_minor ∈ [0.0174, 0.55] m²
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
            # Footprint: π × r² ∈ [0.0174, 0.55] m²
            min_r = math.sqrt(OBSTACLE_MIN_FOOT / math.pi)  # 0.074 m
            max_r = math.sqrt(OBSTACLE_MAX_FOOT / math.pi)  # 0.419 m
            r = self.rng.uniform(min_r, max_r)
            h = self.rng.uniform(0.4, 1.5)
            create_cylinder_mesh(self.stage, path, r, h, color)
            dims = (2 * r, 2 * r, h)

        prim = self.stage.GetPrimAtPath(path)
        xform = UsdGeom.Xformable(prim)
        
        # Calculate proper z-position to place obstacle on ground
        # Spheres are centered at origin, need to lift by radius
        # All other shapes have their base at z=0, can place directly
        if shape == "sphere":
            z = dims[0] / 2  # radius (dims is diameter, so divide by 2)
        else:
            z = 0.01  # Small offset to prevent z-fighting with ground plane
            
        xform.AddTranslateOp().Set(Gf.Vec3d(pos_2d[0], pos_2d[1], z))
        xform.AddRotateXYZOp().Set(Gf.Vec3d(0, 0, rot_deg))

        # Apply physics properties based on weight class and shape
        if weight_class == "moveable":
            # Moveable obstacles: low to moderate friction, pushable/tippable by Spot
            friction = 0.4 if shape in ["sphere", "cylinder", "oval"] else 0.5
            apply_rigid_body_physics(self.stage, path, mass, friction)
            rigid = UsdPhysics.RigidBodyAPI.Get(self.stage, path)
            if rigid:
                rigid.CreateRigidBodyEnabledAttr(True)  # Dynamic/pushable
        
        else:  # non_moveable
            # Non-moveable obstacles: high friction, immovable (static)
            friction = 0.9
            apply_rigid_body_physics(self.stage, path, mass, friction)
            rigid = UsdPhysics.RigidBodyAPI.Get(self.stage, path)
            if rigid:
                rigid.CreateRigidBodyEnabledAttr(False)  # Static/immovable

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
        """
        Populate arena with obstacles with separate targets for moveable and non-moveable.
        Enforces 2m×2m buffer zone around starting position (0,0).
        
        Obstacles are categorized by weight:
        - Moveable: ≤ 32.7 kg (pushable or tippable by Spot) - Orange
        - Non-moveable: > 32.7 kg (immovable obstacles Spot must avoid) - Steel Blue
        
        Args:
            moveable_coverage_pct: Target coverage % for moveable obstacles (default 10%)
            non_moveable_coverage_pct: Target coverage % for non-moveable obstacles (default 10%)
        """
        arena_area = math.pi * (ARENA_RADIUS - 1.5) ** 2
        
        moveable_target_area = arena_area * moveable_coverage_pct / 100.0
        non_moveable_target_area = arena_area * non_moveable_coverage_pct / 100.0
        
        moveable_area = 0.0
        non_moveable_area = 0.0
        idx = 0
        
        # Track weight distribution
        weight_counts = {"moveable": 0, "non_moveable": 0}

        print(f"[INFO] Spawning obstacles:")
        print(f"[INFO]   - Moveable (Orange) target: {moveable_coverage_pct}% coverage = {moveable_target_area:.1f} m²")
        print(f"[INFO]   - Non-moveable (Blue) target: {non_moveable_coverage_pct}% coverage = {non_moveable_target_area:.1f} m²")
        print(f"[INFO]   - Total target: {moveable_coverage_pct + non_moveable_coverage_pct}% coverage = {moveable_target_area + non_moveable_target_area:.1f} m²")
        print(f"[INFO] Starting zone buffer: {STARTING_ZONE_BUFFER}m radius around (0,0)")

        # Phase 1: Spawn moveable obstacles until target reached
        print(f"[INFO] Phase 1: Spawning ORANGE moveable obstacles...")
        phase1_idx = 0
        while moveable_area < moveable_target_area and phase1_idx < 1000:
            old_count = len(self.obstacles)
            self.spawn_one(idx, margin=1.5, min_spawn_clearance=2.0, 
                          force_weight_class="moveable")
            
            if len(self.obstacles) > old_count:
                obs = self.obstacles[-1]
                dims = obs["dims"]
                shape = obs["shape"]
                footprint_area = self.calculate_footprint_area(shape, dims)
                moveable_area += footprint_area
                weight_counts["moveable"] += 1
                print(f"[SPAWN] Moveable #{weight_counts['moveable']} ({shape}): {dims[0]:.2f}×{dims[1]:.2f}m, area={footprint_area:.2f}m², total={moveable_area:.1f}m²")
            
            idx += 1
            phase1_idx += 1
        
        print(f"[INFO] Phase 1 complete: {weight_counts['moveable']} moveable obstacles, {moveable_area:.1f}m² coverage")

        # Phase 2: Spawn non-moveable obstacles until target reached (less strict clearance to find positions)
        print(f"[INFO] Phase 2: Spawning BLUE non-moveable obstacles...")
        phase2_idx = 0
        while non_moveable_area < non_moveable_target_area and phase2_idx < 1000:
            old_count = len(self.obstacles)
            self.spawn_one(idx, margin=1.5, min_spawn_clearance=1.5,  # Slightly less strict clearance for phase 2
                          force_weight_class="non_moveable")
            
            if len(self.obstacles) > old_count:
                obs = self.obstacles[-1]
                dims = obs["dims"]
                shape = obs["shape"]
                footprint_area = self.calculate_footprint_area(shape, dims)
                non_moveable_area += footprint_area
                weight_counts["non_moveable"] += 1
                print(f"[SPAWN] Non-moveable #{weight_counts['non_moveable']} ({shape}): {dims[0]:.2f}×{dims[1]:.2f}m, area={footprint_area:.2f}m², total={non_moveable_area:.1f}m²")
            
            idx += 1
            phase2_idx += 1

        print(f"[INFO] Phase 2 complete: {weight_counts['non_moveable']} non-moveable obstacles, {non_moveable_area:.1f}m² coverage")

        total_area = moveable_area + non_moveable_area
        total_coverage_pct = (total_area / arena_area) * 100.0
        print(f"\n[OK] {len(self.obstacles)} obstacles spawned (total coverage {total_area:.1f} m², {total_coverage_pct:.1f}%)")
        print(f"[INFO] Weight distribution:")
        print(f"[INFO]   - ORANGE Moveable: {weight_counts['moveable']} obstacles, {moveable_area:.1f} m² ({(moveable_area/arena_area)*100:.1f}% of arena)")
        print(f"[INFO]   - BLUE Non-moveable: {weight_counts['non_moveable']} obstacles, {non_moveable_area:.1f} m² ({(non_moveable_area/arena_area)*100:.1f}% of arena)")

    def nearest_obstacle_distance(self, x: float, y: float) -> float:
        """Return the distance to the nearest obstacle center."""
        if not self.obstacles:
            return 999.0
        return min(distance_2d([x, y], obs["pos"]) for obs in self.obstacles)

    def remove_prims(self) -> None:
        """Remove all obstacle prims from the stage."""
        for obs in self.obstacles:
            prim = self.stage.GetPrimAtPath(obs["path"])
            if prim.IsValid():
                self.stage.RemovePrim(obs["path"])
        self.obstacles.clear()
        
        # Remove small static obstacles
        for obs in self.small_obstacles:
            prim = self.stage.GetPrimAtPath(obs["path"])
            if prim.IsValid():
                self.stage.RemovePrim(obs["path"])
        self.small_obstacles.clear()

    def spawn_small_static(self, target_coverage_pct: float = SMALL_OBSTACLE_COVERAGE * 100) -> None:
        """
        Spawn small static obstacles (golf ball to softball size).
        These are unmovable hazards that Spot must navigate around.
        
        Coverage: Equal to moveable and non-moveable obstacles (10% of total arena).
        Size range: 1.7-4 inches (0.043-0.102 meters)
        All shapes available, all static (RigidBodyEnabled = False)
        """
        # Calculate target area: target_coverage_pct% of total arena (same as moveable/non-moveable)
        arena_area = math.pi * (ARENA_RADIUS - 1.5) ** 2
        target_area = arena_area * target_coverage_pct / 100.0
        total_area = 0.0
        idx = 0
        
        shape_counts = {shape: 0 for shape in self.SHAPES}
        
        print(f"[INFO] Spawning small static obstacles (target {target_coverage_pct:.0f}% of arena = {target_area:.1f} m²)")

        while total_area < target_area and idx < 500:
            shape = self.rng.choice(self.SHAPES)
            size = self.rng.uniform(SMALL_OBSTACLE_MIN_SIZE, SMALL_OBSTACLE_MAX_SIZE)
            
            # Keep trying random positions until we find one that's valid
            max_attempts = 50
            pos_2d = None
            for attempt in range(max_attempts):
                test_pos = random_inside_arena(margin=1.5, rng=self.rng)
                
                # Check starting zone buffer FIRST
                if distance_2d(test_pos, [SPOT_START_X, SPOT_START_Y]) < STARTING_ZONE_BUFFER:
                    continue
                
                # Check clearance from main obstacles
                too_close = False
                for obs in self.obstacles:
                    if distance_2d(test_pos, obs["pos"]) < SMALL_OBSTACLE_CLEARANCE:
                        too_close = True
                        break
                
                # Check clearance from other small obstacles
                if not too_close:
                    for small_obs in self.small_obstacles:
                        if distance_2d(test_pos, small_obs["pos"]) < SMALL_OBSTACLE_CLEARANCE:
                            too_close = True
                            break
                
                if not too_close:
                    pos_2d = test_pos
                    break  # Found valid position
            
            if pos_2d is not None:
                path = f"/World/SmallObstacles/Small_{idx:03d}"
                rot_deg = self.rng.uniform(0, 360)
                
                # Create small obstacle geometry based on shape and calculate actual footprint
                if shape == "sphere":
                    create_sphere_mesh(self.stage, path, size, COLOR_SMALL_OBSTACLE)
                    dims = (size * 2, size * 2, size * 2)  # diameter stored
                elif shape == "cylinder":
                    h = size
                    create_cylinder_mesh(self.stage, path, size / 2, h, COLOR_SMALL_OBSTACLE)
                    dims = (size, size, h)  # diameter stored
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
                    w_top = size * 0.7  # tapered top
                    d = size
                    h = size
                    create_trapezoid_mesh(self.stage, path, w_bot, w_top, d, h, COLOR_SMALL_OBSTACLE)
                    dims = ((w_bot + w_top) / 2, d, h)  # store average width like main obstacles
                elif shape == "oval":
                    r_major = size
                    r_minor = size * self.rng.uniform(0.6, 1.4)
                    h = size * 0.8
                    create_oval_mesh(self.stage, path, r_major, r_minor, h, COLOR_SMALL_OBSTACLE)
                    dims = (r_major * 2, r_minor * 2, h)  # diameter stored
                else:
                    dims = (size, size, size)
                
                # Calculate actual geometric footprint using helper function
                footprint = self.calculate_footprint_area(shape, dims)
                
                # Position and rotate
                prim = self.stage.GetPrimAtPath(path)
                if prim.IsValid():
                    xform = UsdGeom.Xformable(prim)
                    xform.ClearXformOpOrder()
                    xform.AddTranslateOp().Set(Gf.Vec3d(pos_2d[0], pos_2d[1], size / 2))
                    xform.AddRotateXYZOp().Set(Gf.Vec3d(0, 0, rot_deg))
                
                # Apply static physics (unmovable)
                mass = 1000.0  # effectively infinite for static body
                apply_rigid_body_physics(self.stage, path, mass, friction=0.9)
                rigid = UsdPhysics.RigidBodyAPI.Get(self.stage, path)
                if rigid:
                    rigid.CreateRigidBodyEnabledAttr(False)  # Static/immovable
                
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
        shape_summary = ", ".join([f"{count} {shape}" for shape, count in shape_counts.items() if count > 0])
        print(f"[INFO] Small obstacle shapes: {shape_summary}")


# ─────────────────────────────────────────────────────────────────────────────
# WAYPOINTS — Final Optimized generation (A=24m, B-Z≥30m sequential)
# ─────────────────────────────────────────────────────────────────────────────

def generate_waypoints(rng: np.random.Generator) -> list:
    """
    Generate 25 waypoints (A-Y) using Final Optimized specification:
    - Waypoint A: exactly 20m from origin (0, 0)
    - Waypoints B-Z: each ≥40m from previous waypoint
    - Re-roll direction if candidate falls outside arena
    - Sequential spawning: only one marker visible at a time
    """
    waypoints = []
    prev_x, prev_y = SPOT_START_X, SPOT_START_Y

    for i, label in enumerate(WAYPOINT_LABELS):
        if i == 0:  # Waypoint A
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
                print(f"[WARN] Could not place waypoint {label} after {max_attempts} attempts")
                waypoints.append({"label": label, "pos": np.array([prev_x, prev_y])})
        else:  # Waypoints B-Z
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
                print(f"[WARN] Could not place waypoint {label} after {max_attempts} attempts")
                waypoints.append({"label": label, "pos": np.array([prev_x, prev_y])})

    print(f"[OK] Generated {len(waypoints)} waypoints (Final Optimized: A={WAYPOINT_DIST_A}m, rest≥{WAYPOINT_SPACING_BZ}m)")
    return waypoints


def spawn_waypoint_marker(stage, label: str, pos: np.ndarray) -> list:
    """
    Spawn a flag-on-pole waypoint marker.
    Returns list of prim paths for later removal.
    """
    pole_path = f"/World/Waypoints/Marker_{label}_Pole"
    flag_path = f"/World/Waypoints/Marker_{label}_Flag"

    # Pole (cylinder)
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

    # Banner (rectangular flag at top of pole)
    flag_mesh = UsdGeom.Mesh.Define(stage, flag_path)
    hw = WP_FLAG_WIDTH / 2
    hh = WP_FLAG_HEIGHT / 2
    flag_pts = [
        Gf.Vec3f(-hw, 0, WP_POLE_HEIGHT - hh),
        Gf.Vec3f(hw, 0, WP_POLE_HEIGHT - hh),
        Gf.Vec3f(hw, 0, WP_POLE_HEIGHT + hh),
        Gf.Vec3f(-hw, 0, WP_POLE_HEIGHT + hh),
    ]
    flag_mesh.GetPointsAttr().Set(flag_pts)
    flag_mesh.GetFaceVertexCountsAttr().Set([4])
    flag_mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])

    # Color: active waypoint = yellow, future waypoints = green
    if label == WAYPOINT_LABELS[0]:
        flag_mesh.GetDisplayColorAttr().Set([COLOR_WAYPOINT])
    else:
        flag_mesh.GetDisplayColorAttr().Set([COLOR_START_WAYPOINT])

    flag_prim = stage.GetPrimAtPath(flag_path)
    flag_xform = UsdGeom.Xformable(flag_prim)
    flag_xform.AddTranslateOp().Set(Gf.Vec3d(pos[0], pos[1], 0))

    return [pole_path, flag_path]


def remove_waypoint_markers(stage, marker_paths: list) -> None:
    """Remove waypoint marker prims from the stage."""
    for path in marker_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            stage.RemovePrim(path)


# ─────────────────────────────────────────────────────────────────────────────
# SPEED CONTROL — obstacle avoidance and waypoint approach
# ─────────────────────────────────────────────────────────────────────────────

def compute_speed_command(spot_x: float, spot_y: float,
                          target_x: float, target_y: float,
                          obstacle_mgr: ObstacleManager) -> float:
    """
    Compute speed command based on:
    1. Proximity to obstacles (slow down if close)
    2. Distance to target waypoint (slow down as approaching)
    Returns speed in m/s clamped to [SPOT_MIN_SPEED_MPS, SPOT_MAX_SPEED_MPS].
    """
    dist_to_target = distance_2d([spot_x, spot_y], [target_x, target_y])
    dist_to_nearest_obs = obstacle_mgr.nearest_obstacle_distance(spot_x, spot_y)

    # Base speed
    speed = SPOT_MAX_SPEED_MPS

    # Slow down near obstacles
    if dist_to_nearest_obs < 3.0:
        slowdown_factor = max(0.3, dist_to_nearest_obs / 3.0)
        speed *= slowdown_factor

    # Slow down near target
    if dist_to_target < OBSTACLE_SLOW_RADIUS:
        approach_factor = max(0.4, dist_to_target / OBSTACLE_SLOW_RADIUS)
        speed *= approach_factor

    return np.clip(speed, SPOT_MIN_SPEED_MPS, SPOT_MAX_SPEED_MPS)


# ─────────────────────────────────────────────────────────────────────────────
# MPPI CONTROLLER — post-policy obstacle-avoiding command refinement
# ─────────────────────────────────────────────────────────────────────────────

class MPPIController:
    """
    Model Predictive Path Integral (MPPI) controller for obstacle-aware navigation.

    At each 20 Hz policy step, MPPI takes the RL policy's suggested velocity
    command as a nominal action, samples K noisy control sequences around it,
    simulates each trajectory forward with a unicycle kinematic model, evaluates
    a composite cost (obstacle clearance + goal reaching + heading alignment +
    arena boundary), and returns the weighted-optimal first action.

    Only active at inference time in Testing_Environment.py — the RL training
    loop in navigation_env.py is completely unaffected.

    Kinematic model (body frame):
        x   += (vx * cos(yaw) - vy * sin(yaw)) * dt
        y   += (vx * sin(yaw) + vy * cos(yaw)) * dt
        yaw += omega * dt

    Cost function:
        J = w_goal * dist(pos_T, goal)
          + sum_t [ w_heading * heading_err(t)^2
                  + w_obs    * sum_obs max(0, r_safe - clearance(t))^2
                  + w_bound  * I(outside_arena(t)) ]
    """

    def __init__(
        self,
        horizon: int = 25,
        num_samples: int = 200,
        temperature: float = 0.05,
        dt: float = 0.05,
    ):
        """
        Args:
            horizon:      Lookahead steps (25 × 0.05 s = 1.25 s)
            num_samples:  Number of sampled trajectories K
            temperature:  MPPI temperature λ — lower → sharper weighting
            dt:           Control timestep matching 20 Hz policy frequency
        """
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.dt = dt

        # Per-channel Gaussian noise std: [vx, vy, omega]
        self._sigma = np.array([1.0, 0.25, 0.4], dtype=np.float32)

        # Action bounds matching training env scaling
        self._act_min = np.array([-0.5, -0.5, -1.5], dtype=np.float32)
        self._act_max = np.array([ 3.0,  0.5,  1.5], dtype=np.float32)

        # Warm-start nominal sequence (T × 3): zero at startup
        self._nominal = np.zeros((horizon, 3), dtype=np.float32)

        # Cost weights
        self.w_goal    = 10.0   # terminal distance to goal
        self.w_heading =  1.5   # running heading alignment
        self.w_obs     = 150.0  # obstacle repulsion
        self.w_bound   = 50.0   # arena boundary violation
        self.r_safe    =  2.0   # safety clearance radius (metres)

    def reset(self) -> None:
        """Clear warm-start state at episode boundaries."""
        self._nominal[:] = 0.0

    def solve(
        self,
        pos: np.ndarray,
        yaw: float,
        target: np.ndarray,
        obstacles: list,
        nominal_action: np.ndarray,
    ) -> np.ndarray:
        """
        Compute a collision-avoiding velocity command [vx, vy, omega].

        The nominal sequence is seeded with the RL policy's action so MPPI
        always respects the planner's intent while steering clear of obstacles.

        Args:
            pos:           Current position [x, y]
            yaw:           Current heading (radians)
            target:        Goal position [x, y]
            obstacles:     List of (ox, oy, radius) tuples
            nominal_action: RL policy suggestion [vx, vy, omega]

        Returns:
            np.ndarray shape (3,): optimal [vx, vy, omega]
        """
        # Seed the entire nominal horizon with the RL policy suggestion
        self._nominal[:] = np.clip(nominal_action, self._act_min, self._act_max)

        # Sample K × T × 3 Gaussian perturbations
        eps = np.random.randn(self.num_samples, self.horizon, 3).astype(np.float32)
        eps *= self._sigma

        # Perturbed control sequences: (K, T, 3)
        U = self._nominal[np.newaxis, :, :] + eps
        U = np.clip(U, self._act_min, self._act_max)

        # Evaluate trajectories → (K,) costs
        costs = self._rollout(pos, yaw, target, obstacles, U)

        # MPPI weight: exp(-J/λ) with numerical stabilisation
        costs -= costs.min()
        weights = np.exp(-costs / (self.temperature + 1e-8))
        weights /= weights.sum() + 1e-8

        # Weighted sum → optimal sequence: (T, 3)
        optimal = np.einsum("k,ktd->td", weights, U)
        optimal = np.clip(optimal, self._act_min, self._act_max)

        # Shift warm-start: drop t=0, repeat last step for continuity
        self._nominal = np.roll(optimal, -1, axis=0)
        self._nominal[-1] = optimal[-1]

        return optimal[0]

    def _rollout(
        self,
        pos0: np.ndarray,
        yaw0: float,
        target: np.ndarray,
        obstacles: list,
        U: np.ndarray,
    ) -> np.ndarray:
        """Vectorised forward simulation over K trajectories. Returns (K,) costs."""
        K = self.num_samples
        tx, ty = float(target[0]), float(target[1])

        # State arrays: (K,)
        x   = np.full(K, float(pos0[0]), dtype=np.float32)
        y   = np.full(K, float(pos0[1]), dtype=np.float32)
        yaw = np.full(K, float(yaw0),    dtype=np.float32)
        costs = np.zeros(K, dtype=np.float32)

        # Pre-build obstacle matrix: (N, 3)
        if obstacles:
            obs_arr = np.array(obstacles, dtype=np.float32)
            ox  = obs_arr[:, 0]  # (N,)
            oy  = obs_arr[:, 1]
            or_ = obs_arr[:, 2]
        else:
            obs_arr = None

        for t in range(self.horizon):
            vx    = U[:, t, 0]   # (K,)
            vy    = U[:, t, 1]
            omega = U[:, t, 2]

            # Unicycle kinematics (body → world frame)
            cos_y = np.cos(yaw)
            sin_y = np.sin(yaw)
            x   += (vx * cos_y - vy * sin_y) * self.dt
            y   += (vx * sin_y + vy * cos_y) * self.dt
            yaw += omega * self.dt

            # Heading cost: penalise misalignment with goal direction
            dx      = tx - x
            dy      = ty - y
            desired = np.arctan2(dy, dx)                            # (K,)
            h_err   = np.arctan2(np.sin(desired - yaw),
                                 np.cos(desired - yaw))             # (K,)
            costs  += (self.w_heading * h_err ** 2) / self.horizon

            # Obstacle cost: penalise trajectories inside safety radius
            if obs_arr is not None:
                # Broadcast: x (K,1) vs ox (1,N) → dist (K,N)
                dist_c = np.sqrt(
                    (x[:, np.newaxis] - ox[np.newaxis, :]) ** 2 +
                    (y[:, np.newaxis] - oy[np.newaxis, :]) ** 2
                )
                clearance   = dist_c - or_[np.newaxis, :]          # (K, N)
                penetration = np.maximum(0.0, self.r_safe - clearance)
                costs += self.w_obs * np.sum(penetration ** 2, axis=1)

            # Arena boundary cost
            arena_r = np.sqrt(x ** 2 + y ** 2)
            costs += self.w_bound * (arena_r > (ARENA_RADIUS - 1.5)).astype(np.float32)

        # Terminal goal cost
        costs += self.w_goal * np.sqrt((tx - x) ** 2 + (ty - y) ** 2)

        return costs


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT CLASS — episode management, reset, step, rewards, logging
# ─────────────────────────────────────────────────────────────────────────────

class CircularWaypointEnv:
    """Circular arena waypoint navigation environment for Spot RL training."""

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
        self.all_episodes = []  # Track all episode results for aggregate stats
        self.episode_logged = False  # Flag to prevent duplicate logging
        self.last_status_print_time = 0.0  # Track last status print for 1-second intervals

        self.physics_ready = False

        # Control frequency decimation — policy runs at 20 Hz (every 25 physics steps)
        self._control_decimation = 25  # 500 Hz / 25 = 20 Hz
        self._physics_step_count = 0
        self._cached_command = np.array([0.0, 0.0, 0.0])
        self._dodge_walk_origin = None  # (x, y) set when cone clears; cleared after 1 m post-dodge walk
        self._yaw_dir = 0.0               # ±1.5 while rotating to clear obstacle cone, 0 otherwise
        self._mppi = MPPIController()

    def reset(self, episode: int) -> None:
        """Reset environment for a new episode."""
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
        self.episode_logged = False  # Reset logging flag for new episode
        self.last_status_print_time = 0.0  # Reset status print timer
        self._physics_step_count = 0
        self._cached_command = np.array([0.0, 0.0, 0.0])
        self._dodge_walk_origin = None
        self._yaw_dir = 0.0
        self._mppi.reset()

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

        # Large obstacles only (1% fill) — small static disabled
        self.obstacle_mgr.populate(moveable_coverage_pct=1.0, non_moveable_coverage_pct=1.0,
                                    min_spawn_clearance=2.0)
        # self.obstacle_mgr.spawn_small_static(target_coverage_pct=1.0)
        print(f"[DEBUG] obstacle_mgr registered: {len(self.obstacle_mgr.obstacles)} large, {len(self.obstacle_mgr.small_obstacles)} small obstacles")

        # Reset Spot position
        if self.spot is not None:
            self.spot.robot.set_world_pose(
                position=np.array([SPOT_START_X, SPOT_START_Y, SPOT_START_Z]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
            )
            self.spot.robot.set_joints_default_state(self.spot.default_pos)
            print(f"[OK] Spot reset to start position ({SPOT_START_X}, {SPOT_START_Y}, {SPOT_START_Z})")
        else:
            print("[WARN] Spot not initialized yet")

        # Load existing CSV data if file exists
        if CSV_LOG_PATH and self.csv_file is None:
            self._load_existing_csv_data()
            print(f"[OK] CSV log ready: {CSV_LOG_PATH}")

        print(f"[OK] Environment reset complete")

    def step(self, step_size: float) -> bool:
        """
        Execute one physics step. Returns False if episode should terminate.
        """
        if not self.physics_ready:
            self.episode_start_time = self.world.current_time
            self.physics_ready = True
            return True  # Skip first call

        # Get Spot position
        spot_pos, _ = self.spot.robot.get_world_pose()
        spot_x, spot_y, spot_z = spot_pos[0], spot_pos[1], spot_pos[2]

        # Check fall condition
        if spot_z < FALL_HEIGHT_THRESHOLD:
            if not self.episode_logged:
                print(f"[FALL] Spot fell (z={spot_z:.3f} < {FALL_HEIGHT_THRESHOLD})")
                self._log_to_csv(FAILURE_FELL_OVER)
                self.episode_logged = True
            return False

        # Check waypoint reached
        if self.current_waypoint_idx < len(self.waypoints):
            wp = self.waypoints[self.current_waypoint_idx]
            dist_to_wp = distance_2d([spot_x, spot_y], wp["pos"])

            if dist_to_wp < WAYPOINT_REACH_DIST:
                self.waypoints_reached += 1
                self.score += WAYPOINT_REWARD
                print(f"[WAYPOINT] Reached {wp['label']} (+{WAYPOINT_REWARD} pts) | Score: {self.score:.1f} | Total: {self.waypoints_reached}/{WAYPOINT_COUNT}")

                # Remove current marker
                remove_waypoint_markers(self.stage, self.current_marker_paths)
                self.current_marker_paths.clear()

                # Move to next waypoint
                self.current_waypoint_idx += 1

                # Spawn next marker (sequential spawning)
                if self.current_waypoint_idx < len(self.waypoints):
                    next_wp = self.waypoints[self.current_waypoint_idx]
                    paths = spawn_waypoint_marker(self.stage, next_wp["label"], next_wp["pos"])
                    self.current_marker_paths.extend(paths)
                    print(f"[INFO] Next target: Waypoint {next_wp['label']} at ({next_wp['pos'][0]:.1f}, {next_wp['pos'][1]:.1f})")
                else:
                    if not self.episode_logged:
                        print(f"[COMPLETE] All waypoints reached!")
                        self._log_to_csv(FAILURE_COMPLETED)
                        self.episode_logged = True
                    return False

                # Stop the robot for this step (matches training env behavior)
                self.spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
                return True

        # Apply time decay
        elapsed = self.world.current_time - self.episode_start_time
        self.score = EPISODE_START_SCORE + self.waypoints_reached * WAYPOINT_REWARD + elapsed * (-TIME_DECAY_PER_SEC)

        # Print status every second
        if elapsed - self.last_status_print_time >= 1.0:
            if self.current_waypoint_idx < len(self.waypoints):
                wp = self.waypoints[self.current_waypoint_idx]
                dist_to_wp = distance_2d([spot_x, spot_y], wp["pos"])
                print(f"[STATUS] Score: {self.score:.1f} | Target: {wp['label']} | Distance: {dist_to_wp:.2f}m")
            else:
                print(f"[STATUS] Score: {self.score:.1f} | No active waypoint")
            self.last_status_print_time = elapsed

        # Check score depletion
        if self.score <= 0:
            if not self.episode_logged:
                print(f"[TIMEOUT] Score depleted (elapsed={elapsed:.1f}s)")
                self._log_to_csv(FAILURE_RAN_OUT)
                self.episode_logged = True
            return False

        # ═════════════════════════════════════════════════════════════════════
        # POLICY-BASED NAVIGATION (Using Trained Neural Network)
        # Policy runs at 20 Hz (every 25 physics steps) to match training.
        # Between policy calls, the cached command is re-applied.
        # ═════════════════════════════════════════════════════════════════════
        if navigation_policy is not None and self.current_waypoint_idx < len(self.waypoints):
            self._physics_step_count += 1

            # Only call the policy every _control_decimation steps (20 Hz)
            if self._physics_step_count % self._control_decimation == 1:
                wp = self.waypoints[self.current_waypoint_idx]
                target_x, target_y = wp["pos"][0], wp["pos"][1]
                
                # Get Spot state
                _, spot_orientation = self.spot.robot.get_world_pose()
                current_heading = quaternion_to_yaw(spot_orientation)
                
                # Build observation vector EXACTLY matching VS3 training format (obs_dim=34)
                obs = np.zeros(34, dtype=np.float32)
                obs_idx = 0
                
                # (3) Base velocity
                try:
                    lin_vel = self.spot.robot.get_linear_velocity()
                    ang_vel = self.spot.robot.get_angular_velocity()
                    obs[obs_idx] = lin_vel[0]
                    obs[obs_idx + 1] = lin_vel[1]
                    obs[obs_idx + 2] = ang_vel[2]
                except:
                    obs[obs_idx:obs_idx + 3] = 0.0
                obs_idx += 3
                
                # (2) Heading as sin/cos pair
                obs[obs_idx] = np.sin(current_heading)
                obs[obs_idx + 1] = np.cos(current_heading)
                obs_idx += 2
                
                # (3) Waypoint info in ROBOT FRAME
                dx_world = target_x - spot_x
                dy_world = target_y - spot_y
                distance = np.sqrt(dx_world**2 + dy_world**2)

                cos_h = np.cos(current_heading)
                sin_h = np.sin(current_heading)
                dx_robot = dx_world * cos_h + dy_world * sin_h
                dy_robot = -dx_world * sin_h + dy_world * cos_h
                
                obs[obs_idx] = dx_robot
                obs[obs_idx + 1] = dy_robot
                obs[obs_idx + 2] = distance
                obs_idx += 3
                
                # (18) Obstacle distances from 18-ray raycasting
                obstacle_distances = raycast_obstacle_dists(spot_x, spot_y, current_heading,
                                                            self.obstacle_mgr, num_rays=18, ray_length=5.0)
                obs[obs_idx:obs_idx + 18] = obstacle_distances
                obs_idx += 18
                
                # (6) Stage encoding - one-hot vector
                stage_one_hot = np.zeros(6, dtype=np.float32)
                stage_one_hot[min(training_stage, 5)] = 1.0
                obs[obs_idx:obs_idx + 6] = stage_one_hot
                obs_idx += 6
                
                # (2) Mode encoding - [is_turning, is_approaching]
                heading_to_wp = np.arctan2(dy_world, dx_world)
                heading_error = heading_to_wp - current_heading
                heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
                if abs(heading_error) > 0.3:
                    obs[obs_idx] = 1.0
                    obs[obs_idx + 1] = 0.0
                else:
                    obs[obs_idx] = 0.0
                    obs[obs_idx + 1] = 1.0
                obs_idx += 2
                
                assert obs_idx == 34, f"Observation size mismatch: {obs_idx} != 34"
                
                # Run policy inference
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
                    action_mean, value = navigation_policy(obs_tensor)
                    action = action_mean.cpu().numpy()[0]
                
                # Scale action using SAME ranges as training env
                raw_vx = action[0]
                raw_vy = action[1]
                raw_omega = action[2]
                
                vx = np.clip(raw_vx, -1.0, 1.0) * (5.0 if raw_vx > 0 else 0.5)
                vy = np.clip(raw_vy, -1.0, 1.0) * 0.5
                omega = np.clip(raw_omega, -1.0, 1.0) * 1.5
                
                # ── Obstacle proximity calculations ──────────────────────────────
                # Compute actual Euclidean clearance to every obstacle for the
                # 10 m slowdown and 0.5 m emergency strafe checks.
                _all_obs = list(self.obstacle_mgr.obstacles) + list(self.obstacle_mgr.small_obstacles)
                if _all_obs:
                    _clearances = []
                    for _o in _all_obs:
                        _ox, _oy = _o["pos"][0], _o["pos"][1]
                        if "dims" in _o:
                            _or = np.sqrt(_o["dims"][0]**2 + _o["dims"][1]**2) / 2.0
                        else:
                            _or = _o.get("size", 0.3) / 2.0
                        _clearances.append(
                            np.sqrt((spot_x - _ox)**2 + (spot_y - _oy)**2) - _or
                        )
                    _min_clearance = min(_clearances)
                    _nearest_idx   = int(np.argmin(_clearances))
                    _nearest_obs   = _all_obs[_nearest_idx]
                    _nearest_ox    = _nearest_obs["pos"][0]
                    _nearest_oy    = _nearest_obs["pos"][1]
                else:
                    _min_clearance = 999.0
                    _nearest_ox    = spot_x
                    _nearest_oy    = spot_y

                # ── Ray-based forward-cone detection (unchanged) ──────────────────
                # Rays 0, 1, 17 cover the ±20° forward cone (18 rays × 20° each).
                #   ray 0  = straight ahead
                #   ray 1  = 20° to the LEFT   (positive angle in robot frame)
                #   ray 17 = 20° to the RIGHT  (negative angle / wrap-around)
                # ray_length = 5.0 m  →  3 m threshold = 0.60 normalised
                #                     →  5 m slowdown   = 1.00 normalised (full range)
                _DODGE_THRESHOLD   = 3.0 / 5.0   # obstacle within 3 m → nudge
                _SLOW_THRESHOLD    = 5.0 / 5.0   # obstacle within 5 m → slow down
                _ray_ahead = obstacle_distances[0]
                _ray_left  = obstacle_distances[1]
                _ray_right = obstacle_distances[17]
                _obstacle_in_front = (
                    _ray_ahead < _DODGE_THRESHOLD or
                    _ray_left  < _DODGE_THRESHOLD or
                    _ray_right < _DODGE_THRESHOLD
                )
                _obstacle_approaching = (
                    _ray_ahead < _SLOW_THRESHOLD or
                    _ray_left  < _SLOW_THRESHOLD or
                    _ray_right < _SLOW_THRESHOLD
                )
                # When slowing down, cap forward speed proportionally to gap remaining
                _min_fwd_ray = min(_ray_ahead, _ray_left, _ray_right)
                _slow_vx = vx * (_min_fwd_ray / _SLOW_THRESHOLD)  # 0→1 as gap shrinks

                # ── Navigation FSM ───────────────────────────────────────────────
                # Priority (highest → lowest):
                #   Heading correct   — yaw toward the waypoint if error > 30°.
                #   Obstacle nudge    — obstacle within 3 m: yaw away + forward.
                #   Obstacle slow     — obstacle within 4 m: continue RL but reduce speed.
                #   Default           — hand full control to the RL policy.

                if abs(heading_error) > math.radians(30):
                    # ── Heading correction ────────────────────────────────────────
                    # Misaligned with waypoint: yaw toward it while moving forward
                    # until heading error falls within 30°.
                    self._yaw_dir = 0.0
                    self._cached_command = np.array([min(1.0, vx), 0.0, omega], dtype=np.float32)

                elif _obstacle_in_front:
                    # ── Obstacle nudge ────────────────────────────────────────────
                    # Obstacle within 3 m in the forward cone.
                    # Pick the yaw direction once (toward the side with more space)
                    # and hold it — with forward speed — until the obstacle leaves.
                    if self._yaw_dir == 0.0:
                        # Obstacle closer on left → yaw right (CW = negative omega)
                        self._yaw_dir = -1.5 if _ray_left < _ray_right else 1.5
                    self._cached_command = np.array([min(1.0, vx), 0.0, self._yaw_dir], dtype=np.float32)

                elif _obstacle_approaching:
                    # ── Obstacle slowdown ─────────────────────────────────────────
                    # Obstacle within 4 m: keep RL steering but reduce forward speed
                    # proportionally to the remaining gap.
                    self._yaw_dir = 0.0
                    self._cached_command = np.array([max(0.1, _slow_vx), vy, omega], dtype=np.float32)

                else:
                    # ── Default: RL command ───────────────────────────────────────
                    self._yaw_dir = 0.0
                    self._cached_command = np.array([vx, vy, omega], dtype=np.float32)

            # Apply the cached command every physics step (same command held for 25 steps)
            self.spot.forward(step_size, self._cached_command)

        return True

    def _load_existing_csv_data(self) -> None:
        """Load existing episode data from CSV if file exists."""
        if os.path.exists(CSV_LOG_PATH):
            with open(CSV_LOG_PATH, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Skip aggregate stats rows (they don't have Episode numbers)
                    if row.get('Episode') and row['Episode'].isdigit():
                        self.all_episodes.append({
                            'episode': int(row['Episode']),
                            'waypoints': int(row['Waypoints_Reached']),
                            'reason': row['Failure_Reason'],
                            'score': float(row['Final_Score'])
                        })
            print(f"[INFO] Loaded {len(self.all_episodes)} previous episodes from CSV")

    def _calculate_aggregate_stats(self) -> dict:
        """Calculate aggregate statistics from all episodes."""
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
        """Rewrite entire CSV with aggregate stats at top, then all episode data."""
        stats = self._calculate_aggregate_stats()
        
        with open(CSV_LOG_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write aggregate statistics header
            writer.writerow(["# AGGREGATE STATISTICS (Updated after each episode)"])
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Average_Waypoints_Reached", f"{stats['avg_waypoints']:.2f}"])
            writer.writerow(["Max_Waypoints_Reached", f"{stats['max_waypoints']}"])
            writer.writerow(["Failure_Rate_Fell_Over", f"{stats['fell_rate']:.2f}%"])
            writer.writerow(["Failure_Rate_Ran_Out_of_Points", f"{stats['ran_out_rate']:.2f}%"])
            writer.writerow(["Completion_Rate", f"{stats['completion_rate']:.2f}%"])
            writer.writerow([])  # Blank line separator
            
            # Write episode data header
            writer.writerow(CSV_HEADERS)
            
            # Write all episode data
            for ep in self.all_episodes:
                writer.writerow([
                    ep['episode'],
                    ep['waypoints'],
                    ep['reason'],
                    f"{ep['score']:.2f}"
                ])

    def _log_to_csv(self, reason: str) -> None:
        """Log episode summary and update aggregate statistics."""
        # Add current episode to tracking
        episode_data = {
            'episode': self.episode_num,
            'waypoints': self.waypoints_reached,
            'reason': reason,
            'score': self.score
        }
        self.all_episodes.append(episode_data)
        
        # Rewrite entire CSV with updated aggregate stats
        self._write_csv_with_stats()
        
        print(f"[CSV] Episode {self.episode_num} logged: {self.waypoints_reached} waypoints, score {self.score:.1f}, reason: {reason}")
        
        # Print updated aggregate stats
        stats = self._calculate_aggregate_stats()
        print(f"[STATS] Avg Waypoints: {stats['avg_waypoints']:.2f} | Completion: {stats['completion_rate']:.1f}% | Fell: {stats['fell_rate']:.1f}% | Ran Out: {stats['ran_out_rate']:.1f}%")

    def close(self) -> None:
        """Clean up resources."""
        # Final CSV write already done in _log_to_csv
        print(f"[OK] CSV log finalized: {CSV_LOG_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# WORLD BUILDER — ground plane and lighting
# ─────────────────────────────────────────────────────────────────────────────

def build_world(world, stage) -> None:
    """Initialize world scene with lighting and ground plane."""
    print(f"\n{'-' * 72}")
    print("[WORLD] Building environment...")
    print(f"{'-' * 72}")

    # Add default ground plane (CRITICAL for preventing fall-through)
    world.scene.add_default_ground_plane(
        z_position=0.0,
        name="ground_plane",
        prim_path="/World/GroundPlane",
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.01
    )
    print("[OK] Ground plane added (z=0)")

    # Add floor disc visual (optional decoration)
    floor_disc_path = "/World/FloorDisc"
    disc_mesh = UsdGeom.Mesh.Define(stage, floor_disc_path)
    segments = 64
    pts = [Gf.Vec3f(0, 0, 0.01)]  # Center slightly above ground
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
    disc_mesh.GetDisplayColorAttr().Set([Gf.Vec3f(0.3, 0.35, 0.3)])  # Dark gray-green
    print("[OK] Floor disc visual added")

    # Add lighting
    distant_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    distant_light.CreateIntensityAttr(800.0)
    distant_light.CreateAngleAttr(0.53)
    xform = UsdGeom.Xformable(distant_light)
    xform.AddRotateXYZOp().Set(Gf.Vec3d(315, 45, 0))
    print("[OK] Distant light added")

    print(f"{'-' * 72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SENSOR SUITE SETUP — Boston Dynamics Spot Standard Sensors
# ─────────────────────────────────────────────────────────────────────────────

def setup_spot_sensors(spot_prim_path: str):
    """
    Add Boston Dynamics standard sensor suite to Spot.
    
    Real Spot includes:
    - 5 stereo camera pairs (10 cameras): front, left, right, rear, overhead
    - Depth perception from stereo pairs
    - 360° panoramic RGB vision
    - IMU (accessible via robot.get_linear_velocity(), get_angular_velocity())
    - Joint encoders (12 joints: 3/leg × 4 legs)
    - Foot contact sensors (4 foot pads)
    
    Args:
        spot_prim_path: USD path to Spot robot (e.g., "/World/Spot")
    """
    print("\n" + "-" * 72)
    print("ADDING BOSTON DYNAMICS SPOT SENSOR SUITE")
    print("-" * 72)
    
    sensors_created = []
    
    # ─────────────────────────────────────────────────────────────────────────
    # FRONT STEREO PAIR (primary navigation cameras)
    # ─────────────────────────────────────────────────────────────────────────
    try:
        front_left = Camera(
            prim_path=f"{spot_prim_path}/FrontStereoLeft",
            name="front_stereo_left",
            frequency=30,
            resolution=(640, 480),
        )
        front_left.set_world_pose(position=np.array([0.35, 0.06, 0.25]))
        sensors_created.append("Front Stereo Left (640×480, 30Hz)")
    except Exception as e:
        print(f"  [WARN] Front stereo left: {e}")
    
    try:
        front_right = Camera(
            prim_path=f"{spot_prim_path}/FrontStereoRight",
            name="front_stereo_right",
            frequency=30,
            resolution=(640, 480),
        )
        front_right.set_world_pose(position=np.array([0.35, -0.06, 0.25]))
        sensors_created.append("Front Stereo Right (640×480, 30Hz)")
    except Exception as e:
        print(f"  [WARN] Front stereo right: {e}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # LEFT STEREO PAIR
    # ─────────────────────────────────────────────────────────────────────────
    try:
        left_front = Camera(
            prim_path=f"{spot_prim_path}/LeftStereoFront",
            name="left_stereo_front",
            frequency=30,
            resolution=(640, 480),
        )
        left_front.set_world_pose(position=np.array([0.10, 0.25, 0.20]))
        sensors_created.append("Left Stereo Front (640×480, 30Hz)")
    except Exception as e:
        print(f"  [WARN] Left stereo front: {e}")
    
    try:
        left_rear = Camera(
            prim_path=f"{spot_prim_path}/LeftStereoRear",
            name="left_stereo_rear",
            frequency=30,
            resolution=(640, 480),
        )
        left_rear.set_world_pose(position=np.array([-0.10, 0.25, 0.20]))
        sensors_created.append("Left Stereo Rear (640×480, 30Hz)")
    except Exception as e:
        print(f"  [WARN] Left stereo rear: {e}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # RIGHT STEREO PAIR
    # ─────────────────────────────────────────────────────────────────────────
    try:
        right_front = Camera(
            prim_path=f"{spot_prim_path}/RightStereoFront",
            name="right_stereo_front",
            frequency=30,
            resolution=(640, 480),
        )
        right_front.set_world_pose(position=np.array([0.10, -0.25, 0.20]))
        sensors_created.append("Right Stereo Front (640×480, 30Hz)")
    except Exception as e:
        print(f"  [WARN] Right stereo front: {e}")
    
    try:
        right_rear = Camera(
            prim_path=f"{spot_prim_path}/RightStereoRear",
            name="right_stereo_rear",
            frequency=30,
            resolution=(640, 480),
        )
        right_rear.set_world_pose(position=np.array([-0.10, -0.25, 0.20]))
        sensors_created.append("Right Stereo Rear (640×480, 30Hz)")
    except Exception as e:
        print(f"  [WARN] Right stereo rear: {e}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # REAR STEREO PAIR
    # ─────────────────────────────────────────────────────────────────────────
    try:
        rear_left = Camera(
            prim_path=f"{spot_prim_path}/RearStereoLeft",
            name="rear_stereo_left",
            frequency=30,
            resolution=(640, 480),
        )
        rear_left.set_world_pose(position=np.array([-0.35, 0.06, 0.20]))
        sensors_created.append("Rear Stereo Left (640×480, 30Hz)")
    except Exception as e:
        print(f"  [WARN] Rear stereo left: {e}")
    
    try:
        rear_right = Camera(
            prim_path=f"{spot_prim_path}/RearStereoRight",
            name="rear_stereo_right",
            frequency=30,
            resolution=(640, 480),
        )
        rear_right.set_world_pose(position=np.array([-0.35, -0.06, 0.20]))
        sensors_created.append("Rear Stereo Right (640×480, 30Hz)")
    except Exception as e:
        print(f"  [WARN] Rear stereo right: {e}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # OVERHEAD STEREO PAIR (upward-looking for navigation under obstacles)
    # ─────────────────────────────────────────────────────────────────────────
    try:
        overhead_front = Camera(
            prim_path=f"{spot_prim_path}/OverheadStereoFront",
            name="overhead_stereo_front",
            frequency=30,
            resolution=(640, 480),
        )
        overhead_front.set_world_pose(position=np.array([0.05, 0.04, 0.35]))
        sensors_created.append("Overhead Stereo Front (640×480, 30Hz)")
    except Exception as e:
        print(f"  [WARN] Overhead stereo front: {e}")
    
    try:
        overhead_rear = Camera(
            prim_path=f"{spot_prim_path}/OverheadStereoRear",
            name="overhead_stereo_rear",
            frequency=30,
            resolution=(640, 480),
        )
        overhead_rear.set_world_pose(position=np.array([0.05, -0.04, 0.35]))
        sensors_created.append("Overhead Stereo Rear (640×480, 30Hz)")
    except Exception as e:
        print(f"  [WARN] Overhead stereo rear: {e}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[OK] {len(sensors_created)} cameras created:")
    for sensor in sensors_created:
        print(f"  + {sensor}")
    
    print("\n[OK] Additional built-in sensors:")
    print("  + IMU (9-axis: accelerometer, gyroscope, magnetometer)")
    print("  + Joint encoders (12 joints: 3/leg x 4 legs)")
    print("  + Foot contact sensors (4 pressure sensors on foot pads)")
    print("  + Body orientation (quaternion from USD pose)")
    print("\n[OK] Full Boston Dynamics Spot sensor suite installed")
    print("-" * 72 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — training loop with episode management
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Main training loop."""
    print("\n" + "=" * 72)
    print("  SPOT RL TRAINING — CIRCULAR WAYPOINT NAVIGATION")
    print("=" * 72)

    # Create world
    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)
    stage = omni.usd.get_context().get_stage()

    # Build environment
    build_world(world, stage)

    # Initialize RNG
    rng = np.random.default_rng(args.seed)
    print(f"[INFO] Random seed: {args.seed}")

    # Create Spot robot (direct SpotFlatTerrainPolicy - NO WRAPPER)
    spot_prim_path = "/World/Spot"
    spot = SpotFlatTerrainPolicy(
        prim_path=spot_prim_path,
        name="Spot",
        position=np.array([SPOT_START_X, SPOT_START_Y, SPOT_START_Z])
    )
    print(f"[OK] SpotFlatTerrainPolicy created at {spot_prim_path}")

    # Reset world (required before initialization)
    world.reset()
    print("[OK] World reset")

    # Initialize Spot
    spot.initialize()
    spot.robot.set_joints_default_state(spot.default_pos)
    print("[OK] Spot initialized with default joint positions")
    
    # Add full sensor suite
    setup_spot_sensors(spot_prim_path)

    # Create environment
    env = CircularWaypointEnv(world, stage, rng)
    env.spot = spot

    # Physics callback
    def on_physics_step(step_size: float):
        if not env.step(step_size):
            # Episode terminated
            pass

    world.add_physics_callback("spot_control", on_physics_step)
    print("[OK] Physics callback registered")

    # Episode loop
    print(f"\n{'=' * 72}")
    print(f"  STARTING TRAINING — {args.episodes} EPISODES")
    print(f"{'=' * 72}\n")

    for episode in range(1, args.episodes + 1):
        env.reset(episode)

        # Run until episode terminates
        while simulation_app.is_running():
            world.step(render=not args.headless)

            # Check if episode is done (score depleted, fall, or all waypoints)
            if env.score <= 0 or env.waypoints_reached >= WAYPOINT_COUNT:
                break

        # Log episode if not already logged (catch cases where main loop detects termination)
        if not env.episode_logged:
            if env.waypoints_reached >= WAYPOINT_COUNT:
                env._log_to_csv(FAILURE_COMPLETED)
            elif env.score <= 0:
                env._log_to_csv(FAILURE_RAN_OUT)
            env.episode_logged = True

        # Check if user closed simulation
        if not simulation_app.is_running():
            print("\n[EXIT] Simulation closed by user")
            break

    # Cleanup
    env.close()
    simulation_app.close()
    print("\n" + "=" * 72)
    print("  TRAINING COMPLETE")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
