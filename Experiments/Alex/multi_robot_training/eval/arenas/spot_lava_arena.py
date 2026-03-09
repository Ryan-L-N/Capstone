"""
Spot Lava Rock Arena - Dynamic Rubble Field
=============================================

50-foot diameter circular arena modeled after an aa lava field.
Filled with ~350 random polyhedral rocks that shift, roll, and tumble
when stepped on. Tests Spot's ability on truly unstable, loose terrain.

ARENA LAYERS:
  1. Uneven base terrain (static heightfield mesh)
  2. Embedded boulders (30 static + 20 dynamic, partially buried)
  3. Loose surface rubble (300 dynamic rocks in 3 size classes)
  4. Circular containment wall (36 segments)

KEYBOARD:
  W / S         Forward / Backward
  A / D         Turn left / Turn right
  SPACE         Emergency stop
  G             Cycle gait: FLAT -> ROUGH -> PARKOUR
  N             Toggle auto-gait (terrain-aware switching)
  T             Toggle auto-walk (vx=0.6 m/s)
  M             Toggle FPV camera
  X             Toggle selfright mode
  H             Show position info
  R             Reset robot to start
  ESC           Exit

XBOX CONTROLLER:
  Left Stick     Forward/back + turn
  A              Toggle auto-walk
  B              Toggle selfright
  Y              Reset to start
  LB             Toggle FPV camera
  RB             Cycle gait (FLAT/ROUGH/PARKOUR)
  Start          Toggle auto-gait (terrain-aware)
  Back           Emergency stop

Isaac Sim 5.1.0 + Isaac Lab 2.3.0
Created for AI2C Tech Capstone - MS for Autonomy, February 2026
"""

import numpy as np
import argparse
import sys
import os
import torch
import time

# Parse args BEFORE SimulationApp (required)
parser = argparse.ArgumentParser(description="Spot Lava Rock Arena")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--seed", type=int, default=42, help="Random seed for arena generation")
parser.add_argument("--rock-count", type=int, default=300,
                    help="Number of loose rubble rocks (default: 300)")
args = parser.parse_args()

# SimulationApp MUST be created before any omni.isaac imports
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": args.headless,
    "width": 1920,
    "height": 1080,
    "window_width": 1920,
    "window_height": 1080,
    "anti_aliasing": 0,
    "renderer": "RayTracedLighting",
})

# Now safe to import Isaac modules
import omni
from omni.isaac.core import World
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
from pxr import UsdGeom, Gf, UsdPhysics, UsdShade, PhysxSchema, UsdLux, Vt, Sdf
import carb.input
from omni.kit.viewport.utility import get_active_viewport
from scipy.spatial import ConvexHull

# Import rough terrain policy
try:
    from spot_rough_terrain_policy import SpotRoughTerrainPolicy
    HAS_ROUGH_POLICY = True
except ImportError:
    HAS_ROUGH_POLICY = False
    print("[WARNING] SpotRoughTerrainPolicy not available - gait cycling disabled")

# Optional: pygame for Xbox controller
HAS_PYGAME = False
try:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    HAS_PYGAME = True
except ImportError:
    pass


# =============================================================================
# CONFIGURATION
# =============================================================================

# Arena geometry
ARENA_DIAMETER_FT = 50.0
ARENA_RADIUS_M = ARENA_DIAMETER_FT * 0.3048 / 2   # 7.62m
ARENA_CENTER = np.array([0.0, 0.0])                # XY center

# Robot start position (edge of arena, facing center)
START_POS = np.array([-ARENA_RADIUS_M + 1.5, 0.0, 0.8])

# Base terrain
BASE_RESOLUTION = 0.20          # meters per grid cell
BASE_NOISE_AMPLITUDE = 0.08     # meters max height variation
BASE_NOISE_OCTAVES = 3
BASE_MOUND_COUNT = 8            # number of gentle mounds
BASE_MOUND_HEIGHT = 0.15        # max mound height (meters)
BASE_MOUND_WIDTH = 2.0          # mound radius (meters)

# Embedded boulders
N_BOULDERS_STATIC = 30
N_BOULDERS_DYNAMIC = 20
BOULDER_SIZE_RANGE = (0.30, 0.80)   # diameter in meters
BOULDER_MASS_RANGE = (50.0, 200.0)
BOULDER_BURY_FRACTION = (0.30, 0.50)

# Loose rubble
N_SMALL_RUBBLE = int(args.rock_count * 0.60)    # 180
N_MEDIUM_RUBBLE = int(args.rock_count * 0.30)   # 90
N_LARGE_RUBBLE = args.rock_count - N_SMALL_RUBBLE - N_MEDIUM_RUBBLE  # 30

SMALL_SIZE_RANGE = (0.03, 0.08)     # meters
SMALL_MASS_RANGE = (0.1, 1.0)       # kg
SMALL_VERTS = (8, 16)

MEDIUM_SIZE_RANGE = (0.08, 0.20)
MEDIUM_MASS_RANGE = (1.0, 8.0)
MEDIUM_VERTS = (12, 30)

LARGE_SIZE_RANGE = (0.20, 0.40)
LARGE_MASS_RANGE = (8.0, 30.0)
LARGE_VERTS = (20, 40)

# Rock physics
ROCK_FRICTION_RANGE = (0.70, 0.95)
ROCK_RESTITUTION = 0.05
ROCK_LINEAR_DAMPING = 0.3
ROCK_ANGULAR_DAMPING = 0.3
ROCK_SLEEP_THRESHOLD = 0.01

# Containment wall
WALL_HEIGHT = 0.6
WALL_THICKNESS = 0.15
WALL_SEGMENTS = 36

# Settling
SETTLING_STEPS = 1500   # physics steps at 500Hz = 3 seconds

# Lava cracks (emissive ground fractures)
N_MAIN_CRACKS = 8
CRACK_WIDTH_RANGE = (0.06, 0.18)     # meters (wider for visibility)
CRACK_STEP = 0.3                      # meters per segment
CRACK_GLOW_INTENSITY = 5000.0
CRACK_GLOW_COLOR = (1.0, 0.15, 0.02)  # Deep orange-red
CRACK_LIGHT_COUNT = 16                 # Underglow point lights (more for visibility)
CRACK_LIGHT_INTENSITY = 5000.0
CRACK_LIGHT_RADIUS = 2.0

# Steam vents
N_STEAM_VENTS = 6
STEAM_VENT_HEIGHT = (1.0, 2.5)          # meters tall
STEAM_VENT_RADIUS = (0.08, 0.20)        # base radius
STEAM_LIGHT_INTENSITY = 2000.0
STEAM_LIGHT_COLOR = (0.9, 0.85, 0.8)    # Warm white steam
STEAM_RISE_SPEED = 0.3                   # meters/second upward
STEAM_CYCLE_PERIOD = 6.0                 # seconds for full rise cycle
STEAM_OPACITY_BASE = 0.25               # opacity at base (most visible)
STEAM_OPACITY_TOP = 0.03                # opacity at top (fades out)
STEAM_ANIM_RATE = 10                    # update every N physics steps (50Hz)

# Rock material pool
N_BASALT_MATERIALS = 8
N_OXIDIZED_MATERIALS = 4

# Parkour checkpoint
PARKOUR_CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..", "48h_training", "model_44998.pt"
)

# Gait modes
GAIT_MODES = [
    {"name": "FLAT",    "description": "Flat terrain policy",           "policy_type": "flat"},
    {"name": "ROUGH",   "description": "Rough terrain policy (30k)",    "policy_type": "rough"},
    {"name": "PARKOUR", "description": "Parkour policy (jumping, 45k)", "policy_type": "parkour"},
]

# Self-right
SELFRIGHT_ROLL_ACCEL = 12.0
SELFRIGHT_MAX_ROLL_VEL = 2.5
SELFRIGHT_GROUND_LIFT = 0.8
SELFRIGHT_DAMPING = 3.0
SELFRIGHT_UPRIGHT_DEG = 35.0
SELFRIGHT_UPRIGHT_TIME = 0.3

# Xbox controller
XBOX_AXIS_TURN = 0
XBOX_AXIS_FWD = 1
XBOX_DEADZONE = 0.12
XBOX_BTN_A = 0
XBOX_BTN_B = 1
XBOX_BTN_Y = 3
XBOX_BTN_LB = 4
XBOX_BTN_RB = 5
XBOX_BTN_BACK = 6
XBOX_BTN_START = 7

STABILIZE_TIME = 1.0
RECOVERY_STABILIZE = 1.5
GAIT_SWITCH_STABILIZE = 0.5

# Terrain-aware auto-gait switching
TERRAIN_SCAN_INTERVAL     = 0.5    # seconds between assessments (~2Hz)
TERRAIN_EMA_ALPHA         = 0.3    # EMA smoothing factor (0=slow, 1=instant)
TERRAIN_FLAT_TO_ROUGH     = 0.04   # difficulty threshold: upgrade to ROUGH
TERRAIN_ROUGH_TO_FLAT     = 0.02   # difficulty threshold: downgrade to FLAT (hysteresis)
TERRAIN_ROUGH_TO_PARKOUR  = 0.12   # difficulty threshold: upgrade to PARKOUR
TERRAIN_PARKOUR_TO_ROUGH  = 0.08   # difficulty threshold: downgrade from PARKOUR
TERRAIN_CONFIRM_READINGS  = 3      # consecutive readings before switching

AUTO_GAIT_HUD_NAMES = {"FLAT": "A-FLAT", "ROUGH": "A-ROUGH", "PARKOUR": "A-PARK"}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def quat_to_yaw(quat):
    """Convert quaternion [w, x, y, z] to yaw angle."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def get_roll_pitch(quat):
    """Extract roll and pitch from quaternion [w,x,y,z]."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def is_rolled_over(quat):
    roll, pitch = get_roll_pitch(quat)
    return abs(roll) > np.radians(60) or abs(pitch) > np.radians(60)


def quat_forward_axis(quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    fx = 1.0 - 2.0 * (y * y + z * z)
    fy = 2.0 * (x * y + w * z)
    fz = 2.0 * (x * z - w * y)
    return np.array([fx, fy, fz])


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


class NumpyRobotWrapper:
    """Wraps SingleArticulation to always return numpy arrays.
    GPU PhysX returns CUDA tensors -- this converts them to numpy."""

    def __init__(self, robot):
        object.__setattr__(self, '_robot', robot)

    def __getattr__(self, name):
        return getattr(self._robot, name)

    def get_world_pose(self):
        pos, orient = self._robot.get_world_pose()
        return _to_np(pos), _to_np(orient)

    def get_linear_velocity(self):
        return _to_np(self._robot.get_linear_velocity())

    def get_angular_velocity(self):
        return _to_np(self._robot.get_angular_velocity())

    def get_joint_positions(self):
        return _to_np(self._robot.get_joint_positions())

    def get_joint_velocities(self):
        return _to_np(self._robot.get_joint_velocities())

    def set_world_pose(self, position=None, orientation=None):
        self._robot.set_world_pose(position=position, orientation=orientation)

    def set_joint_efforts(self, efforts):
        if isinstance(efforts, np.ndarray):
            efforts = torch.from_numpy(efforts).float().to("cuda:0")
        self._robot.set_joint_efforts(efforts)

    def set_angular_velocity(self, vel):
        self._robot.set_angular_velocity(vel)

    def set_linear_velocity(self, vel):
        self._robot.set_linear_velocity(vel)

    def apply_action(self, action):
        self._robot.apply_action(action)

    def set_joints_default_state(self, *args, **kwargs):
        self._robot.set_joints_default_state(*args, **kwargs)

    def initialize(self, *args, **kwargs):
        self._robot.initialize(*args, **kwargs)


class VelocitySmoother:
    def __init__(self, accel_rate=2.0, decel_rate=4.0):
        self.accel_rate = accel_rate
        self.decel_rate = decel_rate
        self.current_vx = 0.0
        self.current_wz = 0.0

    def update(self, target_vx, target_wz, dt):
        self.current_vx = self._ramp(self.current_vx, target_vx, dt)
        self.current_wz = self._ramp(self.current_wz, target_wz, dt)
        return self.current_vx, self.current_wz

    def _ramp(self, current, target, dt):
        diff = target - current
        if abs(diff) < 0.001:
            return target
        rate = self.decel_rate if abs(target) < abs(current) else self.accel_rate
        max_change = rate * dt
        change = np.clip(diff, -max_change, max_change)
        return current + change

    def reset(self):
        self.current_vx = 0.0
        self.current_wz = 0.0


class TerrainDifficultyAssessor:
    """Analyzes height-scan data to compute terrain difficulty and recommend gait.

    Uses SpotRoughTerrainPolicy._cast_height_rays() which casts 187 PhysX rays
    in a 17x11 grid (1.6m x 1.0m) ahead of the robot. Works independently of
    which policy is active since all policies share the same robot body.

    Difficulty metric: weighted combination of height variance and peak-to-peak range.
    Uses EMA smoothing and hysteresis thresholds with consecutive-reading confirmation
    to prevent rapid gait oscillation.
    """

    GAIT_LEVEL = {"flat": 0, "rough": 1, "parkour": 2}

    def __init__(self, spot_rough_policy, has_parkour=False):
        self._spot_rough = spot_rough_policy
        self._has_parkour = has_parkour
        self._ema_difficulty = 0.0
        self._confirm_up_to_rough = 0
        self._confirm_down_to_flat = 0
        self._confirm_up_to_parkour = 0
        self._confirm_down_to_rough = 0
        self.last_raw_difficulty = 0.0
        self.last_ema_difficulty = 0.0

    def reset(self):
        """Reset all state. Call on robot reset."""
        self._ema_difficulty = 0.0
        self._confirm_up_to_rough = 0
        self._confirm_down_to_flat = 0
        self._confirm_up_to_parkour = 0
        self._confirm_down_to_rough = 0
        self.last_raw_difficulty = 0.0
        self.last_ema_difficulty = 0.0

    def assess(self, current_gait_type):
        """Run one terrain assessment cycle.

        Args:
            current_gait_type: str, one of "flat", "rough", "parkour"

        Returns:
            str or None: Recommended gait ("flat"/"rough"/"parkour"), or None.
        """
        try:
            height_scan = self._spot_rough._cast_height_rays()
        except Exception:
            return None

        if height_scan is None or len(height_scan) == 0:
            return None

        # Compute raw difficulty metric
        variance = np.var(height_scan)
        ptp = np.ptp(height_scan)  # max - min
        raw_difficulty = 0.6 * variance + 0.4 * (ptp / 2.0) ** 2
        self.last_raw_difficulty = raw_difficulty

        # EMA smoothing
        self._ema_difficulty = (
            TERRAIN_EMA_ALPHA * raw_difficulty
            + (1.0 - TERRAIN_EMA_ALPHA) * self._ema_difficulty
        )
        self.last_ema_difficulty = self._ema_difficulty
        difficulty = self._ema_difficulty

        # Check transitions with hysteresis + confirmation
        if current_gait_type == "flat":
            if difficulty > TERRAIN_FLAT_TO_ROUGH:
                self._confirm_up_to_rough += 1
            else:
                self._confirm_up_to_rough = 0
            if self._confirm_up_to_rough >= TERRAIN_CONFIRM_READINGS:
                self._confirm_up_to_rough = 0
                return "rough"

        elif current_gait_type == "rough":
            if difficulty < TERRAIN_ROUGH_TO_FLAT:
                self._confirm_down_to_flat += 1
                self._confirm_up_to_parkour = 0
            elif self._has_parkour and difficulty > TERRAIN_ROUGH_TO_PARKOUR:
                self._confirm_up_to_parkour += 1
                self._confirm_down_to_flat = 0
            else:
                self._confirm_down_to_flat = 0
                self._confirm_up_to_parkour = 0
            if self._confirm_down_to_flat >= TERRAIN_CONFIRM_READINGS:
                self._confirm_down_to_flat = 0
                return "flat"
            if self._confirm_up_to_parkour >= TERRAIN_CONFIRM_READINGS:
                self._confirm_up_to_parkour = 0
                return "parkour"

        elif current_gait_type == "parkour":
            if difficulty < TERRAIN_PARKOUR_TO_ROUGH:
                self._confirm_down_to_rough += 1
            else:
                self._confirm_down_to_rough = 0
            if self._confirm_down_to_rough >= TERRAIN_CONFIRM_READINGS:
                self._confirm_down_to_rough = 0
                return "rough"

        return None


# =============================================================================
# TERRAIN GENERATION: VALUE NOISE (numpy only)
# =============================================================================

def _value_noise_2d(x, y, seed=0):
    """Simple hash-based value noise for 2D coordinates."""
    # Integer grid cell coordinates
    ix = int(np.floor(x))
    iy = int(np.floor(y))
    # Fractional part with smoothstep interpolation
    fx = x - ix
    fy = y - iy
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    # Hash corners
    def _hash(xi, yi):
        n = xi * 374761393 + yi * 668265263 + seed * 1013904223
        n = (n ^ (n >> 13)) * 1274126177
        n = n ^ (n >> 16)
        return (n & 0x7fffffff) / 0x7fffffff  # 0 to 1
    c00 = _hash(ix, iy)
    c10 = _hash(ix + 1, iy)
    c01 = _hash(ix, iy + 1)
    c11 = _hash(ix + 1, iy + 1)
    # Bilinear interpolation
    v0 = c00 * (1 - fx) + c10 * fx
    v1 = c01 * (1 - fx) + c11 * fx
    return v0 * (1 - fy) + v1 * fy


def fractal_noise(x, y, octaves=3, persistence=0.5, lacunarity=2.0, seed=0):
    """Multi-octave value noise."""
    total = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_val = 0.0
    for i in range(octaves):
        total += _value_noise_2d(x * frequency, y * frequency, seed + i * 7) * amplitude
        max_val += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return total / max_val  # Normalized 0 to 1


# =============================================================================
# ARENA CONSTRUCTION
# =============================================================================

def _create_omnipbr_material(stage, path, color, roughness, friction,
                             emissive_color=None, emissive_intensity=0.0,
                             opacity=None, no_physics=False):
    """Create an OmniPBR material with both visual and physics properties."""
    mat = UsdShade.Material.Define(stage, path)

    # OmniPBR visual shader
    shader = UsdShade.Shader.Define(stage, f"{path}/shader")
    shader_out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    shader.CreateIdAttr("OmniPBR")
    shader.GetImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")

    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(
        float(roughness))
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)

    if opacity is not None:
        shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(float(opacity))

    if emissive_color is not None:
        shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(emissive_color)
        shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(
            float(emissive_intensity))

    mat.CreateSurfaceOutput("mdl").ConnectToSource(shader_out)
    mat.CreateVolumeOutput("mdl").ConnectToSource(shader_out)
    mat.CreateDisplacementOutput("mdl").ConnectToSource(shader_out)

    if not no_physics:
        # Physics properties
        phys_mat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        phys_mat.CreateStaticFrictionAttr(float(friction))
        phys_mat.CreateDynamicFrictionAttr(float(friction * 0.85))
        phys_mat.CreateRestitutionAttr(ROCK_RESTITUTION)
        physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
        physx_mat.CreateFrictionCombineModeAttr().Set("max")
        physx_mat.CreateRestitutionCombineModeAttr().Set("min")

    return mat


def create_rock_material_pool(stage, rng):
    """Create a pool of shared OmniPBR rock materials.

    Returns a list of (material, weight) tuples for weighted random selection.
    80% dark basalt, 20% oxidized reddish-brown.
    """
    print("  Creating rock material pool...")
    root = "/World/LavaArena/RockMaterials"
    UsdGeom.Xform.Define(stage, root)

    materials = []  # List of (material, weight, color_tuple)

    # Dark basalt variants (weight=2.0 each -> ~80% of picks)
    for i in range(N_BASALT_MATERIALS):
        base = rng.uniform(0.05, 0.15)
        r = max(0.0, base + rng.uniform(-0.02, 0.02))
        g = max(0.0, base + rng.uniform(-0.02, 0.01))
        b = max(0.0, base + rng.uniform(-0.01, 0.01))
        color = Gf.Vec3f(r, g, b)
        roughness = rng.uniform(0.75, 0.95)
        friction = rng.uniform(0.75, 0.90)

        mat = _create_omnipbr_material(
            stage, f"{root}/basalt_{i}", color, roughness, friction)
        materials.append((mat, 2.0, (r, g, b)))

    # Oxidized reddish-brown variants
    for i in range(N_OXIDIZED_MATERIALS):
        r = rng.uniform(0.18, 0.35)
        g = rng.uniform(0.08, 0.16)
        b = rng.uniform(0.04, 0.10)
        color = Gf.Vec3f(r, g, b)
        roughness = rng.uniform(0.80, 0.95)
        friction = rng.uniform(0.75, 0.90)

        mat = _create_omnipbr_material(
            stage, f"{root}/oxidized_{i}", color, roughness, friction)
        materials.append((mat, 1.0, (r, g, b)))

    print(f"    Materials: {N_BASALT_MATERIALS} basalt + "
          f"{N_OXIDIZED_MATERIALS} oxidized ({len(materials)} total)")
    return materials


def pick_rock_material(rng, material_pool):
    """Pick a random material from the weighted pool.
    Returns (material, color_tuple)."""
    mats, weights, colors = zip(*material_pool)
    weights = np.array(weights)
    weights /= weights.sum()
    idx = rng.choice(len(mats), p=weights)
    return mats[idx], colors[idx]


def generate_random_rock(rng, size_min, size_max, vert_min, vert_max):
    """Generate a random convex hull rock shape.

    Returns (vertices, faces) where vertices is Nx3 and faces is Fx3.
    Vertex count is kept under 64 for GPU PhysX compatibility.
    """
    size = rng.uniform(size_min, size_max)
    n_points = rng.randint(vert_min, min(vert_max + 1, 50))

    # Random points on deformed sphere
    theta = rng.uniform(0, 2 * np.pi, n_points)
    phi = rng.uniform(0, np.pi, n_points)
    r = (size / 2) * (0.4 + 0.6 * rng.uniform(0, 1, n_points))

    # Axis scaling for elongated/flat shapes
    sx = rng.uniform(0.6, 1.4)
    sy = rng.uniform(0.6, 1.4)
    sz = rng.uniform(0.4, 1.0)  # Typically flatter in Z

    x = r * np.sin(phi) * np.cos(theta) * sx
    y = r * np.sin(phi) * np.sin(theta) * sy
    z = r * np.cos(phi) * sz

    points = np.column_stack([x, y, z])

    try:
        hull = ConvexHull(points)
        hull_verts = points[hull.vertices]
        # Remap face indices: simplices reference original points array,
        # but we only keep hull.vertices -- need to re-index
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(hull.vertices)}
        hull_faces = np.array([
            [old_to_new[f[0]], old_to_new[f[1]], old_to_new[f[2]]]
            for f in hull.simplices
        ])
        # Center at origin
        center = hull_verts.mean(axis=0)
        hull_verts -= center
        return hull_verts, hull_faces
    except Exception:
        # Fallback: simple box-like shape
        half = size / 2
        verts = np.array([
            [-half * sx, -half * sy, -half * sz],
            [+half * sx, -half * sy, -half * sz],
            [+half * sx, +half * sy, -half * sz],
            [-half * sx, +half * sy, -half * sz],
            [-half * sx, -half * sy, +half * sz],
            [+half * sx, -half * sy, +half * sz],
            [+half * sx, +half * sy, +half * sz],
            [-half * sx, +half * sy, +half * sz],
        ])
        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 5, 1], [0, 4, 5],
            [2, 7, 3], [2, 6, 7],
            [0, 3, 7], [0, 7, 4],
            [1, 5, 6], [1, 6, 2],
        ])
        return verts, faces


def get_base_height(x, y, seed=42):
    """Get base terrain height at (x, y) world coordinates."""
    # Only within arena circle
    dist = np.sqrt(x**2 + y**2)
    if dist > ARENA_RADIUS_M:
        return 0.0

    # Fractal noise base
    freq = 1.0 / 2.5  # wavelength ~2.5m
    h = fractal_noise(x * freq, y * freq, octaves=BASE_NOISE_OCTAVES, seed=seed)
    h = (h - 0.5) * 2.0 * BASE_NOISE_AMPLITUDE  # Center around 0

    return h


def create_base_terrain(stage, rng):
    """Create uneven base terrain as a static collision mesh."""
    print("  Creating base terrain mesh...")

    radius = ARENA_RADIUS_M
    res = BASE_RESOLUTION
    # Grid dimensions
    n = int(2 * radius / res) + 1
    half = radius

    # Generate vertices on circular grid
    vertices = []
    vertex_map = {}  # (ix, iy) -> vertex index

    for iy in range(n):
        for ix in range(n):
            x = -half + ix * res
            y = -half + iy * res

            # Only include points within circle (with small margin)
            if x**2 + y**2 > (radius + res)**2:
                continue

            z = get_base_height(x, y, seed=args.seed)

            # Add mounds
            for mi in range(BASE_MOUND_COUNT):
                mx_seed = rng.uniform(-radius * 0.7, radius * 0.7)
                my_seed = rng.uniform(-radius * 0.7, radius * 0.7)
                md = np.sqrt((x - mx_seed)**2 + (y - my_seed)**2)
                if md < BASE_MOUND_WIDTH * 2:
                    z += BASE_MOUND_HEIGHT * np.exp(-md**2 / (BASE_MOUND_WIDTH**2))

            vertex_map[(ix, iy)] = len(vertices)
            vertices.append(Gf.Vec3f(float(x), float(y), float(z)))

    # Generate triangle faces
    face_counts = []
    face_indices = []

    for iy in range(n - 1):
        for ix in range(n - 1):
            v00 = vertex_map.get((ix, iy))
            v10 = vertex_map.get((ix + 1, iy))
            v01 = vertex_map.get((ix, iy + 1))
            v11 = vertex_map.get((ix + 1, iy + 1))

            if v00 is not None and v10 is not None and v01 is not None and v11 is not None:
                # Two triangles per quad
                face_counts.append(3)
                face_indices.extend([v00, v10, v11])
                face_counts.append(3)
                face_indices.extend([v00, v11, v01])

    # Create USD mesh
    mesh_path = "/World/LavaArena/BaseTerrain"
    UsdGeom.Xform.Define(stage, "/World/LavaArena")
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(vertices))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_counts))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_indices))
    mesh.GetSubdivisionSchemeAttr().Set("none")

    # DisplayColor — dark basalt ground
    mesh.GetDisplayColorAttr().Set([Gf.Vec3f(0.07, 0.06, 0.05)])

    # Static collision (no RigidBody)
    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    collision_api.GetApproximationAttr().Set("none")  # Use exact triangle mesh for static

    # OmniPBR material for base terrain (dark basalt with physics)
    mat = _create_omnipbr_material(
        stage, "/World/LavaArena/BaseMaterial",
        color=Gf.Vec3f(0.07, 0.06, 0.05),
        roughness=0.92,
        friction=0.85)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

    print(f"    Vertices: {len(vertices)}, Faces: {len(face_counts)}")


def create_lava_rock(stage, path, vertices, faces, position, orientation,
                     mass, is_dynamic, material, color):
    """Create a single lava rock with collision, rigid body, and shared material."""
    mesh = UsdGeom.Mesh.Define(stage, path)

    # Set vertices and faces
    points = Vt.Vec3fArray([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2]))
                            for v in vertices])
    mesh.GetPointsAttr().Set(points)

    face_counts_arr = Vt.IntArray([3] * len(faces))
    face_indices_arr = Vt.IntArray(faces.flatten().tolist())
    mesh.GetFaceVertexCountsAttr().Set(face_counts_arr)
    mesh.GetFaceVertexIndicesAttr().Set(face_indices_arr)
    mesh.GetSubdivisionSchemeAttr().Set("none")

    # DisplayColor (reliable fallback for RTX Real-Time)
    mesh.GetDisplayColorAttr().Set([Gf.Vec3f(color[0], color[1], color[2])])

    # Transform
    prim = mesh.GetPrim()
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(float(position[0]),
                                       float(position[1]),
                                       float(position[2])))
    xf.AddOrientOp().Set(orientation)

    # Collision (convex hull approximation for GPU PhysX)
    UsdPhysics.CollisionAPI.Apply(prim)
    collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    collision_api.GetApproximationAttr().Set("convexHull")

    if is_dynamic:
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(float(mass))

        # PhysX rigid body settings
        physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physx_rb.CreateLinearDampingAttr().Set(ROCK_LINEAR_DAMPING)
        physx_rb.CreateAngularDampingAttr().Set(ROCK_ANGULAR_DAMPING)
        physx_rb.CreateSleepThresholdAttr().Set(ROCK_SLEEP_THRESHOLD)

    # Bind shared OmniPBR material (handles both visual and physics)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def random_quat(rng):
    """Generate a random orientation quaternion as Gf.Quatf (ES-011)."""
    u1, u2, u3 = rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1)
    s1 = np.sqrt(1 - u1)
    s2 = np.sqrt(u1)
    w = s2 * np.cos(2 * np.pi * u3)
    x = s1 * np.sin(2 * np.pi * u2)
    y = s1 * np.cos(2 * np.pi * u2)
    z = s2 * np.sin(2 * np.pi * u3)
    return Gf.Quatf(float(w), float(x), float(y), float(z))


def random_point_in_circle(rng, radius, min_radius=0.0):
    """Generate random (x, y) uniformly distributed in a circle/annulus."""
    while True:
        r = np.sqrt(rng.uniform(min_radius**2, radius**2))
        theta = rng.uniform(0, 2 * np.pi)
        return r * np.cos(theta), r * np.sin(theta)


def create_embedded_boulders(stage, rng, material_pool):
    """Create Layer 2: partially buried boulders."""
    print("  Creating embedded boulders...")
    root = "/World/LavaArena/Boulders"
    UsdGeom.Xform.Define(stage, root)

    n_total = N_BOULDERS_STATIC + N_BOULDERS_DYNAMIC

    for i in range(n_total):
        is_dynamic = (i >= N_BOULDERS_STATIC)

        # Random position in arena (avoid dead center where robot starts)
        x, y = random_point_in_circle(rng, ARENA_RADIUS_M - 0.5, min_radius=2.0)

        # Generate rock shape
        verts, faces = generate_random_rock(
            rng, BOULDER_SIZE_RANGE[0], BOULDER_SIZE_RANGE[1], 20, 40)

        # Compute rock height extent
        z_extent = verts[:, 2].max() - verts[:, 2].min()
        bury_frac = rng.uniform(BOULDER_BURY_FRACTION[0], BOULDER_BURY_FRACTION[1])
        z_base = get_base_height(x, y, seed=args.seed)
        z_pos = z_base - z_extent * bury_frac + z_extent / 2

        mass = rng.uniform(BOULDER_MASS_RANGE[0], BOULDER_MASS_RANGE[1]) if is_dynamic else 0.0
        orient = random_quat(rng)
        mat, col = pick_rock_material(rng, material_pool)

        path = f"{root}/boulder_{i}"
        create_lava_rock(stage, path, verts, faces,
                         position=(x, y, z_pos),
                         orientation=orient,
                         mass=mass,
                         is_dynamic=is_dynamic,
                         material=mat,
                         color=col)

    print(f"    Boulders: {N_BOULDERS_STATIC} static + {N_BOULDERS_DYNAMIC} dynamic")


def create_loose_rubble(stage, rng, material_pool):
    """Create Layer 3: loose surface rubble (3 size classes)."""
    print("  Creating loose rubble...")
    root = "/World/LavaArena/Rubble"
    UsdGeom.Xform.Define(stage, root)

    configs = [
        ("small",  N_SMALL_RUBBLE,  SMALL_SIZE_RANGE,  SMALL_MASS_RANGE,  SMALL_VERTS),
        ("medium", N_MEDIUM_RUBBLE, MEDIUM_SIZE_RANGE,  MEDIUM_MASS_RANGE, MEDIUM_VERTS),
        ("large",  N_LARGE_RUBBLE,  LARGE_SIZE_RANGE,   LARGE_MASS_RANGE,  LARGE_VERTS),
    ]

    idx = 0
    for class_name, count, size_range, mass_range, vert_range in configs:
        for j in range(count):
            # Random position in arena
            x, y = random_point_in_circle(rng, ARENA_RADIUS_M - 0.3)

            # Generate rock
            verts, faces = generate_random_rock(
                rng, size_range[0], size_range[1], vert_range[0], vert_range[1])

            # Place slightly above base terrain so rocks settle down
            z_extent = verts[:, 2].max() - verts[:, 2].min()
            z_base = get_base_height(x, y, seed=args.seed)
            z_pos = z_base + z_extent / 2 + 0.02  # 2cm above surface

            mass = rng.uniform(mass_range[0], mass_range[1])
            orient = random_quat(rng)
            mat, col = pick_rock_material(rng, material_pool)

            path = f"{root}/{class_name}_{j}"
            create_lava_rock(stage, path, verts, faces,
                             position=(x, y, z_pos),
                             orientation=orient,
                             mass=mass,
                             is_dynamic=True,
                             material=mat,
                             color=col)
            idx += 1

    print(f"    Small: {N_SMALL_RUBBLE}, Medium: {N_MEDIUM_RUBBLE}, "
          f"Large: {N_LARGE_RUBBLE} (total: {idx})")


def create_lava_cracks(stage, rng):
    """Create glowing lava cracks on the arena floor.

    Procedural branching crack paths radiating from center with
    emissive OmniPBR material for volcanic glow effect.
    """
    print("  Creating lava cracks...")
    root = "/World/LavaArena/Cracks"
    UsdGeom.Xform.Define(stage, root)

    # Generate crack paths
    crack_paths = []
    for i in range(N_MAIN_CRACKS):
        angle = (2 * np.pi * i / N_MAIN_CRACKS) + rng.uniform(-0.3, 0.3)
        x, y = 0.3 * np.cos(angle), 0.3 * np.sin(angle)
        path_pts = [(x, y)]

        length = rng.uniform(3.0, ARENA_RADIUS_M - 1.0)
        n_steps = int(length / CRACK_STEP)

        for s in range(n_steps):
            angle += rng.uniform(-0.35, 0.35)
            x += CRACK_STEP * np.cos(angle)
            y += CRACK_STEP * np.sin(angle)
            if x**2 + y**2 > (ARENA_RADIUS_M - 0.5)**2:
                break
            path_pts.append((x, y))

            # Branch occasionally
            if s > 3 and rng.uniform() < 0.15 and len(crack_paths) < 30:
                branch_angle = angle + rng.choice([-1, 1]) * rng.uniform(0.4, 1.0)
                branch = [(x, y)]
                bx, by = x, y
                for _ in range(rng.randint(3, 8)):
                    branch_angle += rng.uniform(-0.2, 0.2)
                    bx += CRACK_STEP * np.cos(branch_angle)
                    by += CRACK_STEP * np.sin(branch_angle)
                    if bx**2 + by**2 > (ARENA_RADIUS_M - 0.5)**2:
                        break
                    branch.append((bx, by))
                if len(branch) > 1:
                    crack_paths.append(branch)

        if len(path_pts) > 1:
            crack_paths.append(path_pts)

    # Build mesh from all crack paths
    all_verts = []
    all_face_counts = []
    all_face_indices = []
    vert_offset = 0
    light_positions = []  # Track positions for underglow lights

    for ci, crack in enumerate(crack_paths):
        width = rng.uniform(CRACK_WIDTH_RANGE[0], CRACK_WIDTH_RANGE[1])

        for j in range(len(crack) - 1):
            x0, y0 = crack[j]
            x1, y1 = crack[j + 1]

            # Perpendicular direction for width
            dx = x1 - x0
            dy = y1 - y0
            seg_len = np.sqrt(dx**2 + dy**2)
            if seg_len < 0.001:
                continue
            nx, ny = -dy / seg_len, dx / seg_len

            w = width * rng.uniform(0.7, 1.3)

            z0 = get_base_height(x0, y0, seed=args.seed) + 0.015
            z1 = get_base_height(x1, y1, seed=args.seed) + 0.015

            # Four vertices forming a quad
            all_verts.extend([
                Gf.Vec3f(float(x0 - nx * w), float(y0 - ny * w), float(z0)),
                Gf.Vec3f(float(x0 + nx * w), float(y0 + ny * w), float(z0)),
                Gf.Vec3f(float(x1 + nx * w), float(y1 + ny * w), float(z1)),
                Gf.Vec3f(float(x1 - nx * w), float(y1 - ny * w), float(z1)),
            ])

            # Two triangles per quad
            all_face_counts.extend([3, 3])
            all_face_indices.extend([
                vert_offset, vert_offset + 1, vert_offset + 2,
                vert_offset, vert_offset + 2, vert_offset + 3,
            ])
            vert_offset += 4

        # Collect midpoint of longer cracks for underglow lights
        if len(crack) > 5:
            mid = len(crack) // 2
            mx, my = crack[mid]
            mz = get_base_height(mx, my, seed=args.seed)
            light_positions.append((mx, my, mz))

    if not all_verts:
        print("    No crack geometry generated")
        return

    # Create crack mesh
    mesh = UsdGeom.Mesh.Define(stage, f"{root}/crack_mesh")
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(all_verts))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(all_face_counts))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(all_face_indices))
    mesh.GetSubdivisionSchemeAttr().Set("none")
    mesh.GetDoubleSidedAttr().Set(True)

    # Bright orange-red DisplayColor for crack visibility
    mesh.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.3, 0.05)])

    # Emissive lava glow material
    glow_color = Gf.Vec3f(CRACK_GLOW_COLOR[0], CRACK_GLOW_COLOR[1], CRACK_GLOW_COLOR[2])
    crack_mat = _create_omnipbr_material(
        stage, f"{root}/crack_material",
        color=Gf.Vec3f(1.0, 0.3, 0.05),    # Bright orange-red base
        roughness=0.3,
        friction=0.6,
        emissive_color=glow_color,
        emissive_intensity=CRACK_GLOW_INTENSITY)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(crack_mat)

    # Underglow point lights beneath cracks — MORE lights for visibility
    n_lights = min(CRACK_LIGHT_COUNT, len(light_positions))
    if n_lights < len(light_positions):
        chosen = rng.choice(len(light_positions), size=n_lights, replace=False)
    else:
        chosen = list(range(len(light_positions)))
    for li, idx in enumerate(chosen):
        lx, ly, lz = light_positions[idx]
        light = UsdLux.SphereLight.Define(stage, f"{root}/underglow_{li}")
        light.CreateIntensityAttr(float(CRACK_LIGHT_INTENSITY))
        light.CreateRadiusAttr(0.15)
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.2, 0.02))

        light_prim = stage.GetPrimAtPath(f"{root}/underglow_{li}")
        xf = UsdGeom.Xformable(light_prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(float(lx), float(ly), float(lz + 0.1)))

        light.CreateTreatAsPointAttr(True)

    n_segments = len(all_face_counts) // 2
    print(f"    Cracks: {len(crack_paths)} paths, {n_segments} segments, "
          f"{n_lights} underglow lights")


def create_steam_vents(stage, rng):
    """Create animated translucent steam vents using OmniPBR opacity.

    Each vent is a column of overlapping translucent spheres that float upward,
    grow, fade, and recycle back to the base — creating a continuous steam effect.
    Returns a list of puff data dicts for the physics callback to animate.
    """
    print("  Creating steam vents...")
    root = "/World/LavaArena/SteamVents"
    UsdGeom.Xform.Define(stage, root)

    all_puffs = []  # Animation data for each puff

    for i in range(N_STEAM_VENTS):
        # Random position in the arena (avoid center where robot starts)
        x, y = random_point_in_circle(rng, ARENA_RADIUS_M - 1.5, min_radius=2.5)
        z_base = get_base_height(x, y, seed=args.seed)

        height = rng.uniform(STEAM_VENT_HEIGHT[0], STEAM_VENT_HEIGHT[1])
        vent_root = f"{root}/vent_{i}"
        UsdGeom.Xform.Define(stage, vent_root)

        # Stack of overlapping spheres forming a cloud column
        n_puffs = rng.randint(5, 9)
        for p in range(n_puffs):
            # Stagger initial phase so puffs aren't all synchronized
            phase = p / max(n_puffs - 1, 1)  # 0 at base, 1 at top
            initial_t = phase  # normalized 0..1 progress through rise cycle

            # Initial position
            puff_z = z_base + phase * height
            base_radius = 0.06 + rng.uniform(0.0, 0.04)
            # Small random lateral offset per puff (gives organic look)
            offset_x = rng.uniform(-0.12, 0.12)
            offset_y = rng.uniform(-0.12, 0.12)

            puff_path = f"{vent_root}/puff_{p}"
            sphere = UsdGeom.Sphere.Define(stage, puff_path)
            sphere.GetRadiusAttr().Set(float(base_radius + phase * 0.30))

            # Light warm-grey color (no OmniPBR — doesn't render on RTX Real-Time)
            grey = rng.uniform(0.75, 0.92)
            sphere.GetDisplayColorAttr().Set([Gf.Vec3f(grey, grey * 0.96, grey * 0.92)])

            puff_xf = UsdGeom.Xformable(sphere.GetPrim())
            puff_xf.AddTranslateOp().Set(Gf.Vec3d(
                float(x + offset_x), float(y + offset_y), float(puff_z)))

            # Store animation data for this puff
            all_puffs.append({
                "path": puff_path,
                "base_x": float(x),
                "base_y": float(y),
                "z_base": float(z_base),
                "height": float(height),
                "offset_x": float(offset_x),
                "offset_y": float(offset_y),
                "base_radius": float(base_radius),
                "phase": float(initial_t),  # current normalized progress 0..1
                "speed": float(rng.uniform(0.8, 1.2)),  # per-puff speed variation
            })

        # Warm light at the base of each vent (ground glow)
        light = UsdLux.SphereLight.Define(stage, f"{vent_root}/glow")
        light.CreateIntensityAttr(float(STEAM_LIGHT_INTENSITY))
        light.CreateRadiusAttr(0.15)
        light.CreateColorAttr(Gf.Vec3f(
            STEAM_LIGHT_COLOR[0], STEAM_LIGHT_COLOR[1], STEAM_LIGHT_COLOR[2]))
        light.CreateTreatAsPointAttr(True)
        light_prim = stage.GetPrimAtPath(f"{vent_root}/glow")
        UsdGeom.Xformable(light_prim).AddTranslateOp().Set(
            Gf.Vec3d(float(x), float(y), float(z_base + 0.3)))

    print(f"    Steam vents: {N_STEAM_VENTS} cloud columns ({len(all_puffs)} animated puffs)")
    return all_puffs


def create_containment_wall(stage):
    """Create Layer 4: circular containment wall."""
    print("  Creating containment wall...")
    root = "/World/LavaArena/Wall"
    UsdGeom.Xform.Define(stage, root)

    # OmniPBR wall material (dark basalt)
    wall_mat = _create_omnipbr_material(
        stage, "/World/LavaArena/WallMaterial",
        color=Gf.Vec3f(0.12, 0.10, 0.09),
        roughness=0.90,
        friction=0.80)

    radius = ARENA_RADIUS_M + WALL_THICKNESS / 2
    angle_step = 2 * np.pi / WALL_SEGMENTS
    seg_length = 2 * radius * np.sin(angle_step / 2) + 0.05  # Slight overlap

    for i in range(WALL_SEGMENTS):
        angle = i * angle_step
        cx = radius * np.cos(angle)
        cy = radius * np.sin(angle)

        path = f"{root}/seg_{i}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(1.0)

        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(float(cx), float(cy),
                                           float(WALL_HEIGHT / 2)))
        # Rotate to face center
        yaw_deg = float(np.degrees(angle + np.pi / 2))
        xf.AddRotateZOp().Set(yaw_deg)
        xf.AddScaleOp().Set(Gf.Vec3d(float(seg_length), float(WALL_THICKNESS),
                                       float(WALL_HEIGHT)))

        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.10, 0.08, 0.07)])
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(wall_mat)

    print(f"    Wall: {WALL_SEGMENTS} segments, radius={ARENA_RADIUS_M:.1f}m, "
          f"height={WALL_HEIGHT}m")


def create_arena_lighting(stage):
    """Create dark volcanic lighting — moody overcast with lava glow."""
    light_path = "/World/Lights"
    UsdGeom.Xform.Define(stage, light_path)

    # Primary dome light — dim overcast grey-purple sky
    dome = UsdLux.DomeLight.Define(stage, f"{light_path}/dome")
    dome.CreateIntensityAttr(120.0)
    dome.CreateColorAttr(Gf.Vec3f(0.6, 0.55, 0.65))  # Grey-purple overcast

    # Secondary dome — very dim red-orange volcanic ambient
    haze_dome = UsdLux.DomeLight.Define(stage, f"{light_path}/haze_dome")
    haze_dome.CreateIntensityAttr(30.0)
    haze_dome.CreateColorAttr(Gf.Vec3f(1.0, 0.35, 0.15))  # Subtle volcanic glow

    # Distant light (sun) — dim, low angle, filtered through haze
    sun = UsdLux.DistantLight.Define(stage, f"{light_path}/sun")
    sun.CreateIntensityAttr(1500.0)
    sun.CreateAngleAttr(2.0)
    sun_prim = stage.GetPrimAtPath(f"{light_path}/sun")
    UsdGeom.Xformable(sun_prim).AddRotateXYZOp().Set(Gf.Vec3f(-15, 25, 0))
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.75, 0.55))  # Orange sun through thick smoke

    # Low red-orange rim light from "below" — lava glow bounce
    rim = UsdLux.DistantLight.Define(stage, f"{light_path}/lava_rim")
    rim.CreateIntensityAttr(400.0)
    rim.CreateAngleAttr(8.0)
    rim.CreateColorAttr(Gf.Vec3f(1.0, 0.25, 0.05))  # Deep orange-red underglow
    rim_prim = stage.GetPrimAtPath(f"{light_path}/lava_rim")
    UsdGeom.Xformable(rim_prim).AddRotateXYZOp().Set(Gf.Vec3f(80, 0, 0))

    print("  Lighting: dark volcanic (overcast dome + dim sun + lava rim)")


def create_volcanic_atmosphere():
    """Apply RTX post-processing for volcanic haze/fog."""
    try:
        import carb.settings
        settings = carb.settings.get_settings()

        # RTX fog — thick dark volcanic smoke
        settings.set("/rtx/fog/enabled", True)
        settings.set("/rtx/fog/fogColor", [0.12, 0.08, 0.05])
        settings.set("/rtx/fog/fogColorIntensity", 0.6)
        settings.set("/rtx/fog/fogStartDist", 3.0)
        settings.set("/rtx/fog/fogEndDist", 20.0)

        # Tone mapping — darker, moodier
        settings.set("/rtx/post/tonemap/filmIso", 100.0)
        settings.set("/rtx/post/tonemap/whitepoint", 6.0)

        print("  Atmosphere: RTX fog + tone mapping (volcanic haze)")
    except Exception as e:
        print(f"  Atmosphere: Could not apply RTX fog ({e}), using lighting only")


def build_lava_arena(stage, rng):
    """Build the complete lava rock arena with visual effects."""
    print()
    print("=" * 50)
    print("  BUILDING LAVA ROCK ARENA")
    print("=" * 50)

    create_arena_lighting(stage)
    create_volcanic_atmosphere()
    rock_materials = create_rock_material_pool(stage, rng)
    create_base_terrain(stage, rng)
    create_lava_cracks(stage, rng)
    create_embedded_boulders(stage, rng, rock_materials)
    create_loose_rubble(stage, rng, rock_materials)
    steam_puffs = create_steam_vents(stage, rng)
    create_containment_wall(stage)

    total_rocks = (N_BOULDERS_STATIC + N_BOULDERS_DYNAMIC +
                   N_SMALL_RUBBLE + N_MEDIUM_RUBBLE + N_LARGE_RUBBLE)
    dynamic_rocks = (N_BOULDERS_DYNAMIC +
                     N_SMALL_RUBBLE + N_MEDIUM_RUBBLE + N_LARGE_RUBBLE)
    print()
    print(f"  Arena complete: {total_rocks} rocks ({dynamic_rocks} dynamic)")
    print(f"  + lava cracks, {N_STEAM_VENTS} steam vents, volcanic atmosphere")
    print(f"  Diameter: {ARENA_DIAMETER_FT:.0f}ft ({ARENA_RADIUS_M * 2:.1f}m)")
    print("=" * 50)
    print()
    return steam_puffs


# =============================================================================
# KEYBOARD INPUT
# =============================================================================

key_state = {
    "forward": False,
    "backward": False,
    "left": False,
    "right": False,
    "stop": False,
    "reset": False,
    "exit": False,
}

fpv_active = [False]
selfright_active = [False]
selfright_upright_timer = [0.0]
auto_walk_active = [False]
smoother = VelocitySmoother(accel_rate=1.5, decel_rate=3.0)
joy_analog = {"fwd": 0.0, "turn": 0.0}
auto_gait_active = [False]
last_terrain_scan_time = [0.0]


def on_keyboard_event(event, *args_ev, **kwargs):
    is_pressed = (event.type == carb.input.KeyboardEventType.KEY_PRESS or
                  event.type == carb.input.KeyboardEventType.KEY_REPEAT)
    key = event.input

    if key == carb.input.KeyboardInput.W:
        key_state["forward"] = is_pressed
    elif key == carb.input.KeyboardInput.S:
        key_state["backward"] = is_pressed
    elif key == carb.input.KeyboardInput.A:
        key_state["left"] = is_pressed
    elif key == carb.input.KeyboardInput.D:
        key_state["right"] = is_pressed
    elif key == carb.input.KeyboardInput.SPACE:
        key_state["stop"] = is_pressed
    elif key == carb.input.KeyboardInput.ESCAPE:
        if is_pressed:
            key_state["exit"] = True

    elif event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if key == carb.input.KeyboardInput.G:
            if spot_rough is not None:
                gait_idx[0] = (gait_idx[0] + 1) % len(GAIT_MODES)
                gait = GAIT_MODES[gait_idx[0]]
                if gait["policy_type"] == "parkour" and spot_parkour is None:
                    gait_idx[0] = (gait_idx[0] + 1) % len(GAIT_MODES)
                    gait = GAIT_MODES[gait_idx[0]]
                if gait["policy_type"] in ("rough", "parkour"):
                    _switch_to_rough_or_parkour()
                else:
                    _switch_to_flat()
                print(f"\n  >> Gait: {gait['name']} - {gait['description']}")
            else:
                print(f"\n  >> Gait cycling disabled")
        elif key == carb.input.KeyboardInput.N:
            if spot_rough is not None:
                auto_gait_active[0] = not auto_gait_active[0]
                if auto_gait_active[0]:
                    print(f"\n  >> AUTO-GAIT: ON (terrain-aware switching)")
                else:
                    print(f"\n  >> AUTO-GAIT: OFF (manual gait control)")
            else:
                print(f"\n  >> AUTO-GAIT: unavailable (no rough policy)")
        elif key == carb.input.KeyboardInput.T:
            auto_walk_active[0] = not auto_walk_active[0]
            if auto_walk_active[0]:
                print(f"\n  >> AUTO-WALK: ON (vx=0.6)")
            else:
                print(f"\n  >> AUTO-WALK: OFF")
                smoother.reset()
        elif key == carb.input.KeyboardInput.M:
            fpv_active[0] = not fpv_active[0]
            if fpv_active[0]:
                viewport_api.camera_path = fpv_cam_path
                print(f"\n  >> Camera: FPV")
            else:
                viewport_api.camera_path = default_camera_path
                print(f"\n  >> Camera: ORBIT")
        elif key == carb.input.KeyboardInput.X:
            selfright_active[0] = not selfright_active[0]
            selfright_upright_timer[0] = 0.0
            if selfright_active[0]:
                print(f"\n  >> SELFRIGHT MODE: A/D=roll (X to cancel)")
            else:
                print(f"\n  >> SELFRIGHT cancelled")
        elif key == carb.input.KeyboardInput.R:
            key_state["reset"] = True
        elif key == carb.input.KeyboardInput.H:
            try:
                pos, _ = spot.robot.get_world_pose()
                dist = np.sqrt(pos[0]**2 + pos[1]**2)
                print(f"\n  >> Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.3f}) "
                      f"Dist from center: {dist:.2f}m / {ARENA_RADIUS_M:.2f}m")
            except Exception:
                pass

    return True


# =============================================================================
# XBOX CONTROLLER SETUP
# =============================================================================
joystick = None
joy_prev_buttons = []

if HAS_PYGAME:
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joy_prev_buttons = [False] * joystick.get_numbuttons()
        print(f"  Controller: {joystick.get_name()}")


# =============================================================================
# PRINT CONTROLS
# =============================================================================
print()
print("=" * 60)
print("  SPOT LAVA ROCK ARENA")
print("=" * 60)
if joystick:
    print()
    print("  XBOX CONTROLLER:")
    print("    Left Stick     Forward/back + turn")
    print("    A              Toggle auto-walk")
    print("    B              Toggle selfright")
    print("    Y              Reset to start")
    print("    LB             Toggle FPV camera")
    print("    RB             Cycle gait (FLAT/ROUGH/PARKOUR)")
    print("    Start          Toggle auto-gait (terrain-aware)")
    print("    Back           Emergency stop")
print()
print("  KEYBOARD:")
print("    W/S       Forward / Backward")
print("    A/D       Turn left / Turn right")
print("    SPACE     Emergency stop")
print("    G         Cycle gait: FLAT -> ROUGH -> PARKOUR")
print("    N         Toggle auto-gait (terrain-aware switching)")
print("    T         Toggle auto-walk (vx=0.6)")
print("    M         Toggle FPV camera")
print("    X         Selfright mode")
print("    H         Show position info")
print("    R         Reset to start")
print("    ESC       Exit")
print()


# =============================================================================
# MAIN SETUP
# =============================================================================

# GPU PhysX required for rough terrain policy
world = World(
    physics_dt=1.0 / 500.0,
    rendering_dt=10.0 / 500.0,
    stage_units_in_meters=1.0,
    device="cuda:0",
)
stage = omni.usd.get_context().get_stage()

# Ground plane (raw USD collision cube — flat plane under the arena)
ground = UsdGeom.Cube.Define(stage, "/World/GroundPlane")
ground.GetSizeAttr().Set(1.0)
ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.25))
ground.AddScaleOp().Set(Gf.Vec3d(50.0, 50.0, 0.5))
ground.GetDisplayColorAttr().Set([Gf.Vec3f(0.04, 0.03, 0.03)])
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
# OmniPBR dark basalt ground
_ground_mat = _create_omnipbr_material(
    stage, "/World/GroundMaterial",
    color=Gf.Vec3f(0.04, 0.03, 0.03),
    roughness=0.95,
    friction=0.80)
UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(_ground_mat)

# Build the lava arena
arena_rng = np.random.RandomState(args.seed)
steam_puff_data = build_lava_arena(stage, arena_rng)

# Spawn Spot
print("\n============================================================")
print("  INITIALIZING SPOT ROBOT")
print("============================================================")

spot_flat = SpotFlatTerrainPolicy(
    prim_path="/World/Spot",
    name="Spot",
    position=START_POS,
)
print(f"[GAIT] Flat terrain policy loaded")

spot_rough = None
spot_parkour = None
if HAS_ROUGH_POLICY:
    try:
        spot_rough = SpotRoughTerrainPolicy(flat_policy=spot_flat)
        print(f"[GAIT] Rough terrain policy loaded")
    except Exception as e:
        print(f"[GAIT] Failed to load rough policy: {e}")
        spot_rough = None

    try:
        ckpt = os.path.abspath(PARKOUR_CHECKPOINT)
        if os.path.exists(ckpt):
            spot_parkour = SpotRoughTerrainPolicy(
                flat_policy=spot_flat, checkpoint_path=ckpt)
            print(f"[GAIT] Parkour policy loaded from {ckpt}")
        else:
            print(f"[GAIT] Parkour checkpoint not found: {ckpt}")
    except Exception as e:
        print(f"[GAIT] Failed to load parkour policy: {e}")
        spot_parkour = None

spot = spot_flat
gait_idx = [0]

# --- GPU PhysX patches ---
try:
    import isaacsim.core.utils.torch as _torch_backend
    import isaacsim.core.utils.torch.tensor as _torch_tensor

    def _patched_move_data(data, device="cpu"):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        return data.to(device=device)

    def _patched_to_list(data):
        if isinstance(data, list):
            return data
        if isinstance(data, np.ndarray):
            return data.tolist()
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy().tolist()
        return [data] if not hasattr(data, '__iter__') else list(data)

    def _patched_clone_tensor(data, device):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32, device=device)
        else:
            data = data.to(device=device)
        return torch.clone(data)

    _torch_tensor.move_data = _patched_move_data
    _torch_tensor.to_list = _patched_to_list
    _torch_tensor.clone_tensor = _patched_clone_tensor
    for _name, _fn in [('move_data', _patched_move_data),
                        ('to_list', _patched_to_list),
                        ('clone_tensor', _patched_clone_tensor)]:
        if hasattr(_torch_backend, _name):
            setattr(_torch_backend, _name, _fn)
    print("[GPU] Patched torch tensor backend")
except Exception as e:
    print(f"[GPU] Could not patch torch tensor backend: {e}")

try:
    import isaacsim.core.utils.torch.transformations as _torch_xforms
    _orig_tf_matrices = _torch_xforms.tf_matrices_from_poses

    def _patched_tf_matrices(translations, orientations, device="cpu"):
        if isinstance(translations, np.ndarray):
            translations = torch.from_numpy(translations).float().to(device)
        if isinstance(orientations, np.ndarray):
            orientations = torch.from_numpy(orientations).float().to(device)
        return _orig_tf_matrices(translations, orientations, device=device)

    _torch_xforms.tf_matrices_from_poses = _patched_tf_matrices
    print("[GPU] Patched tf_matrices_from_poses")
except Exception as e:
    print(f"[GPU] Could not patch tf_matrices_from_poses: {e}")

try:
    _removed = []
    for _reg in [world.scene._scene_registry._geometry_objects]:
        for _name in list(_reg.keys()):
            _obj = _reg[_name]
            if type(_obj).__name__ == 'GroundPlane':
                world.scene.remove_object(_name)
                _removed.append(_name)
    if _removed:
        print(f"[GPU] Removed scene-registered ground planes: {_removed}")
except Exception as e:
    print(f"[GPU] Ground plane cleanup: {e}")

# Initialize
world.reset()

spot_flat.initialize()
spot_flat.robot.set_joints_default_state(spot_flat.default_pos)

# Wrap robot for GPU PhysX numpy compatibility
_real_robot = spot_flat.robot
spot_flat.robot = NumpyRobotWrapper(_real_robot)
print(f"[GPU] Robot wrapped for numpy compatibility")

if spot_rough is not None:
    spot_rough.robot = spot_flat.robot
if spot_parkour is not None:
    spot_parkour.robot = spot_flat.robot

# Save flat actuator properties
flat_saved_props = {}
_GPU_DEV = "cuda:0"

def _ensure_cuda(x):
    if isinstance(x, torch.Tensor):
        return x.clone().to(_GPU_DEV)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.copy()).float().to(_GPU_DEV)
    return torch.tensor(x, dtype=torch.float32, device=_GPU_DEV)

try:
    av = spot_flat.robot._articulation_view
    kps, kds = av.get_gains()
    flat_saved_props['kps'] = _ensure_cuda(kps)
    flat_saved_props['kds'] = _ensure_cuda(kds)
    flat_saved_props['efforts'] = _ensure_cuda(av.get_max_efforts())
    try:
        flat_saved_props['frictions'] = _ensure_cuda(av.get_friction_coefficients())
    except Exception:
        pass
    try:
        flat_saved_props['armatures'] = _ensure_cuda(av.get_armatures())
    except Exception:
        pass
    try:
        flat_saved_props['max_vels'] = _ensure_cuda(av.get_max_joint_velocities())
    except Exception:
        pass
    try:
        pos_iters = av.get_solver_position_iteration_counts()
        vel_iters = av.get_solver_velocity_iteration_counts()
        flat_saved_props['pos_iters'] = pos_iters.clone() if isinstance(pos_iters, torch.Tensor) else pos_iters
        flat_saved_props['vel_iters'] = vel_iters.clone() if isinstance(vel_iters, torch.Tensor) else vel_iters
    except Exception:
        pass
    print(f"[GAIT] Flat props saved")
except Exception as e:
    print(f"[GAIT] Could not save flat properties: {e}")


# --- Gait-switch helper functions (shared by G key, Xbox RB, auto-gait) ---

def _switch_to_rough_or_parkour():
    """Apply rough/parkour actuator gains."""
    if spot_rough is not None:
        spot_rough.apply_gains()


def _switch_to_flat():
    """Restore flat policy actuator properties from saved state."""
    try:
        av = spot_flat.robot._articulation_view
        if 'kps' in flat_saved_props:
            av.set_gains(kps=flat_saved_props['kps'],
                         kds=flat_saved_props['kds'])
        if 'efforts' in flat_saved_props:
            av.set_max_efforts(values=flat_saved_props['efforts'])
        if 'frictions' in flat_saved_props:
            av.set_friction_coefficients(flat_saved_props['frictions'])
        if 'armatures' in flat_saved_props:
            av.set_armatures(flat_saved_props['armatures'])
        if 'max_vels' in flat_saved_props:
            av.set_max_joint_velocities(flat_saved_props['max_vels'])
        if 'pos_iters' in flat_saved_props:
            av.set_solver_position_iteration_counts(flat_saved_props['pos_iters'])
        if 'vel_iters' in flat_saved_props:
            av.set_solver_velocity_iteration_counts(flat_saved_props['vel_iters'])
    except Exception as e:
        print(f"[GAIT] Restore error: {e}")


if spot_rough is not None:
    spot_rough.initialize()
    print(f"[GAIT] Rough policy initialized")
if spot_parkour is not None:
    spot_parkour.initialize()
    print(f"[GAIT] Parkour policy initialized")

# Terrain-aware auto-gait assessor
terrain_assessor = None
if spot_rough is not None:
    terrain_assessor = TerrainDifficultyAssessor(
        spot_rough_policy=spot_rough,
        has_parkour=(spot_parkour is not None),
    )
    print(f"[AUTO-GAIT] TerrainDifficultyAssessor initialized "
          f"(scan={TERRAIN_SCAN_INTERVAL}s, "
          f"flat->rough={TERRAIN_FLAT_TO_ROUGH}, "
          f"rough->flat={TERRAIN_ROUGH_TO_FLAT}, "
          f"confirm={TERRAIN_CONFIRM_READINGS})")
else:
    print(f"[AUTO-GAIT] Disabled (no rough policy available)")

print("Spot initialized")

# Move Spot out of the way during settling (rocks can knock it over)
spot_flat.robot.set_world_pose(
    position=np.array([0.0, 0.0, 15.0]),  # High above arena
    orientation=np.array([1.0, 0.0, 0.0, 0.0]))
spot_flat.robot.set_linear_velocity(np.zeros(3))
spot_flat.robot.set_angular_velocity(np.zeros(3))

# Settling period: let rocks fall and settle before robot spawns
print(f"\n  Settling rocks for {SETTLING_STEPS / 500.0:.1f}s...")
settle_start = time.time()
for _ in range(SETTLING_STEPS):
    world.step(render=False)
settle_elapsed = time.time() - settle_start
print(f"  Settling complete ({settle_elapsed:.1f}s wall time)")

# Now place Spot at the start position — rocks are settled
spot_flat.robot.set_world_pose(
    position=START_POS,
    orientation=np.array([1.0, 0.0, 0.0, 0.0]))
spot_flat.robot.set_linear_velocity(np.zeros(3))
spot_flat.robot.set_angular_velocity(np.zeros(3))
# Step a few times to let the robot land on the settled surface
for _ in range(10):
    world.step(render=False)
print(f"  Spot placed at start: ({START_POS[0]:.1f}, {START_POS[1]:.1f}, {START_POS[2]:.1f})")

# FPV camera
fpv_cam_path = "/World/Spot/body/fpv_camera"
fpv_cam = UsdGeom.Camera.Define(stage, fpv_cam_path)
fpv_cam.CreateFocalLengthAttr(18.0)
fpv_cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
cam_xform = UsdGeom.Xformable(fpv_cam.GetPrim())
cam_xform.AddTranslateOp().Set(Gf.Vec3d(0.4, 0.0, 0.15))
cam_xform.AddOrientOp().Set(Gf.Quatf(0.5, 0.5, -0.5, -0.5))

viewport_api = get_active_viewport()
default_camera_path = viewport_api.camera_path

# Keyboard
input_interface = carb.input.acquire_input_interface()
keyboard = omni.appwindow.get_default_app_window().get_keyboard()
keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

# Physics state
physics_ready = [False]
sim_time = [0.0]
last_hud_time = [0.0]
last_command = [np.array([0.0, 0.0, 0.0])]
recovery_timer = [0.0]
gait_switch_timer = [0.0]

# Steam animation state — pre-cache Xformable + translate ops for each puff
steam_anim_step = [0]
_steam_xforms = []  # list of (xformable, translate_op, puff_data_dict)
for _pd in steam_puff_data:
    _prim = stage.GetPrimAtPath(_pd["path"])
    _xf = UsdGeom.Xformable(_prim)
    _ops = _xf.GetOrderedXformOps()
    _translate_op = _ops[0] if _ops else None  # first op is the translate
    _steam_xforms.append((_xf, _translate_op, _pd))


def animate_steam(dt):
    """Advance all steam puffs upward, growing and fading. Recycle at top."""
    steam_anim_step[0] += 1
    if steam_anim_step[0] % STEAM_ANIM_RATE != 0:
        return  # only update at ~50Hz, not full 500Hz

    for _xf, _translate_op, pd in _steam_xforms:
        if _translate_op is None:
            continue

        # Advance phase (0→1 then wrap)
        pd["phase"] += (dt * STEAM_ANIM_RATE * pd["speed"]) / STEAM_CYCLE_PERIOD
        if pd["phase"] >= 1.0:
            pd["phase"] -= 1.0  # wrap around

        t = pd["phase"]

        # Position: rises from z_base to z_base + height
        z = pd["z_base"] + t * pd["height"]
        # Horizontal drift increases with height (wind wobble)
        drift_scale = 1.0 + t * 2.5
        dx = pd["offset_x"] * drift_scale
        dy = pd["offset_y"] * drift_scale

        _translate_op.Set(Gf.Vec3d(
            pd["base_x"] + dx,
            pd["base_y"] + dy,
            z))

        # Scale radius: small at base → big at top, simulates dissipation
        puff_prim = stage.GetPrimAtPath(pd["path"])
        radius = pd["base_radius"] + t * 0.35
        UsdGeom.Sphere(puff_prim).GetRadiusAttr().Set(float(radius))


# =============================================================================
# PHYSICS CALLBACK (500Hz)
# =============================================================================

def on_physics_step(step_size):
    global spot

    # ES-010B: Skip first callback
    if not physics_ready[0]:
        physics_ready[0] = True
        return

    # Switch active policy based on gait mode
    gait = GAIT_MODES[gait_idx[0]]
    if gait["policy_type"] == "flat" or (gait["policy_type"] == "rough" and spot_rough is None):
        if spot is not spot_flat:
            gait_switch_timer[0] = GAIT_SWITCH_STABILIZE
        spot = spot_flat
    elif gait["policy_type"] == "rough":
        if spot is not spot_rough:
            spot_rough.post_reset()
            gait_switch_timer[0] = GAIT_SWITCH_STABILIZE
        spot = spot_rough
    elif gait["policy_type"] == "parkour" and spot_parkour is not None:
        if spot is not spot_parkour:
            spot_parkour.post_reset()
            gait_switch_timer[0] = GAIT_SWITCH_STABILIZE
        spot = spot_parkour

    sim_time[0] += step_size

    # Animate steam puffs (runs at reduced rate internally)
    animate_steam(step_size)

    if sim_time[0] < STABILIZE_TIME:
        spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
        return

    # Reset
    if key_state["reset"]:
        key_state["reset"] = False
        spot.robot.set_world_pose(
            position=START_POS,
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        smoother.reset()
        recovery_timer[0] = RECOVERY_STABILIZE
        auto_walk_active[0] = False
        if terrain_assessor is not None:
            terrain_assessor.reset()
        print("\n  >> Robot reset to start!")
        spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
        return

    # Selfright
    if selfright_active[0]:
        pos, quat = spot.robot.get_world_pose()
        roll, pitch = get_roll_pitch(quat)
        abs_roll = abs(roll)

        if abs_roll < np.radians(SELFRIGHT_UPRIGHT_DEG) and abs(pitch) < np.radians(SELFRIGHT_UPRIGHT_DEG):
            selfright_upright_timer[0] += step_size
            if selfright_upright_timer[0] >= SELFRIGHT_UPRIGHT_TIME:
                selfright_active[0] = False
                selfright_upright_timer[0] = 0.0
                smoother.reset()
                recovery_timer[0] = RECOVERY_STABILIZE
                print(f"\n  >> SELFRIGHT COMPLETE!")
                spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
                return
        else:
            selfright_upright_timer[0] = 0.0

        roll_dir = 0.0
        if abs(joy_analog["turn"]) > 0.1:
            roll_dir = joy_analog["turn"]
        elif key_state["left"]:
            roll_dir = 1.0
        elif key_state["right"]:
            roll_dir = -1.0

        forward = quat_forward_axis(quat)
        forward_norm = forward / (np.linalg.norm(forward) + 1e-8)
        current_w = spot.robot.get_angular_velocity()

        if roll_dir != 0.0:
            if abs_roll > np.radians(120):
                phase_gain = 1.0
            elif abs_roll > np.radians(60):
                phase_gain = 0.6
            elif abs_roll > np.radians(30):
                phase_gain = 0.25
            else:
                phase_gain = 0.1

            accel = SELFRIGHT_ROLL_ACCEL * phase_gain
            roll_impulse = forward_norm * roll_dir * accel * step_size
            new_w = current_w + roll_impulse
            w_mag = np.linalg.norm(new_w)
            if w_mag > SELFRIGHT_MAX_ROLL_VEL:
                new_w = new_w * (SELFRIGHT_MAX_ROLL_VEL / w_mag)
            spot.robot.set_angular_velocity(new_w)

            if abs_roll > np.radians(45):
                lift_factor = min(1.0, (abs_roll - np.radians(45)) / np.radians(135))
                vel = spot.robot.get_linear_velocity()
                if vel[2] < SELFRIGHT_GROUND_LIFT * lift_factor:
                    vel[2] = SELFRIGHT_GROUND_LIFT * lift_factor
                    spot.robot.set_linear_velocity(vel)
        else:
            damped_w = current_w * (1.0 - SELFRIGHT_DAMPING * step_size)
            spot.robot.set_angular_velocity(damped_w)
        return

    # Recovery stabilization
    if recovery_timer[0] > 0:
        recovery_timer[0] -= step_size
        spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
        return

    # Gait switch stabilization
    if gait_switch_timer[0] > 0:
        gait_switch_timer[0] -= step_size
        spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
        return

    # --- Terrain-aware auto-gait assessment (reduced rate, ~2Hz) ---
    if (auto_gait_active[0]
            and terrain_assessor is not None
            and sim_time[0] - last_terrain_scan_time[0] >= TERRAIN_SCAN_INTERVAL):
        last_terrain_scan_time[0] = sim_time[0]

        current_gait = GAIT_MODES[gait_idx[0]]
        recommended = terrain_assessor.assess(current_gait["policy_type"])

        if recommended is not None and recommended != current_gait["policy_type"]:
            # Find the gait index matching the recommendation
            new_idx = None
            for gi, gm in enumerate(GAIT_MODES):
                if gm["policy_type"] == recommended:
                    new_idx = gi
                    break

            if new_idx is not None:
                # Validate: skip if required policy not loaded
                if recommended == "parkour" and spot_parkour is None:
                    new_idx = None
                elif recommended == "rough" and spot_rough is None:
                    new_idx = None

            if new_idx is not None and new_idx != gait_idx[0]:
                old_name = GAIT_MODES[gait_idx[0]]["name"]
                gait_idx[0] = new_idx
                new_gait = GAIT_MODES[new_idx]
                if new_gait["policy_type"] in ("rough", "parkour"):
                    _switch_to_rough_or_parkour()
                else:
                    _switch_to_flat()
                print(f"\n  >> AUTO-GAIT: {old_name} -> {new_gait['name']} "
                      f"(difficulty={terrain_assessor.last_ema_difficulty:.4f})")

    # Get robot state
    pos, quat = spot.robot.get_world_pose()
    yaw = quat_to_yaw(quat)

    # Compute velocity command
    if auto_walk_active[0]:
        # Auto-walk toward arena center
        target_vx = 0.6
        # Steer toward center
        dx = -pos[0]
        dy = -pos[1]
        desired_yaw = np.arctan2(dy, dx)
        yaw_error = desired_yaw - yaw
        while yaw_error > np.pi:
            yaw_error -= 2 * np.pi
        while yaw_error < -np.pi:
            yaw_error += 2 * np.pi
        target_wz = np.clip(yaw_error * 0.5, -0.5, 0.5)
        vx, wz = smoother.update(target_vx, target_wz, step_size)
        command = np.array([vx, 0.0, wz])
    elif key_state["stop"]:
        smoother.reset()
        command = np.array([0.0, 0.0, 0.0])
    else:
        target_vx = 0.0
        target_wz = 0.0

        if abs(joy_analog["fwd"]) > 0.01 or abs(joy_analog["turn"]) > 0.01:
            target_vx = joy_analog["fwd"] * 1.0
            target_wz = joy_analog["turn"] * 0.8
        else:
            if key_state["forward"]:
                target_vx = 1.0
            elif key_state["backward"]:
                target_vx = -0.4
            if key_state["left"]:
                target_wz = 0.8
            elif key_state["right"]:
                target_wz = -0.8

        vx, wz = smoother.update(target_vx, target_wz, step_size)
        command = np.array([vx, 0.0, wz])

    last_command[0] = command
    spot.forward(step_size, command)

    # HUD (every 0.5s)
    if sim_time[0] - last_hud_time[0] >= 0.5:
        last_hud_time[0] = sim_time[0]
        gait_name = GAIT_MODES[gait_idx[0]]['name']
        if auto_gait_active[0]:
            gait_str = AUTO_GAIT_HUD_NAMES.get(gait_name, f"A-{gait_name}")
        else:
            gait_str = gait_name
        cam_str = "FPV" if fpv_active[0] else "ORB"
        walk_str = "AUTO" if auto_walk_active[0] else "WASD"
        dist = np.sqrt(pos[0]**2 + pos[1]**2)
        roll, pitch = get_roll_pitch(quat)

        if selfright_active[0]:
            status = f"SELFRIGHT(R:{np.degrees(roll):+.0f})"
        elif is_rolled_over(quat):
            status = "FALLEN! (X=selfright)"
        else:
            status = "OK"

        terrain_str = ""
        if auto_gait_active[0] and terrain_assessor is not None:
            terrain_str = f" D:{terrain_assessor.last_ema_difficulty:.3f}"

        print(f"\r  [{sim_time[0]:6.1f}s] {gait_str:>7s} {cam_str} {walk_str} | "
              f"Pos:({pos[0]:5.1f},{pos[1]:5.1f}) Z:{pos[2]:.3f} "
              f"Dist:{dist:.1f}m | "
              f"P:{np.degrees(pitch):+5.1f} R:{np.degrees(roll):+5.1f} | "
              f"vx={command[0]:+.2f} wz={command[2]:+.2f} | "
              f"{status}{terrain_str}", end="     ")


world.add_physics_callback("lava_arena", on_physics_step)


# =============================================================================
# MAIN LOOP
# =============================================================================

print(f"\n  Stabilizing for {STABILIZE_TIME}s, then controls active...")
print("  Click on simulation window to capture keyboard input!")
print("=" * 60)
print()


def apply_deadzone(value, dz=XBOX_DEADZONE):
    if abs(value) < dz:
        return 0.0
    sign = 1.0 if value >= 0 else -1.0
    return sign * (abs(value) - dz) / (1.0 - dz)


try:
    while simulation_app.is_running() and not key_state["exit"]:
        world.step(render=True)

        # Xbox controller
        if joystick is not None:
            pygame.event.pump()

            raw_fwd = -joystick.get_axis(XBOX_AXIS_FWD)
            raw_turn = -joystick.get_axis(XBOX_AXIS_TURN)
            joy_analog["fwd"] = apply_deadzone(raw_fwd)
            joy_analog["turn"] = apply_deadzone(raw_turn)

            n_btns = joystick.get_numbuttons()
            for i in range(min(n_btns, len(joy_prev_buttons))):
                curr = joystick.get_button(i)
                if curr and not joy_prev_buttons[i]:
                    if i == XBOX_BTN_A:
                        auto_walk_active[0] = not auto_walk_active[0]
                        if auto_walk_active[0]:
                            print(f"\n  >> AUTO-WALK: ON")
                        else:
                            print(f"\n  >> AUTO-WALK: OFF")
                            smoother.reset()
                    elif i == XBOX_BTN_B:
                        selfright_active[0] = not selfright_active[0]
                        selfright_upright_timer[0] = 0.0
                    elif i == XBOX_BTN_Y:
                        key_state["reset"] = True
                    elif i == XBOX_BTN_LB:
                        fpv_active[0] = not fpv_active[0]
                        if fpv_active[0]:
                            viewport_api.camera_path = fpv_cam_path
                        else:
                            viewport_api.camera_path = default_camera_path
                    elif i == XBOX_BTN_RB:
                        if spot_rough is not None:
                            gait_idx[0] = (gait_idx[0] + 1) % len(GAIT_MODES)
                            gait = GAIT_MODES[gait_idx[0]]
                            if gait["policy_type"] == "parkour" and spot_parkour is None:
                                gait_idx[0] = (gait_idx[0] + 1) % len(GAIT_MODES)
                                gait = GAIT_MODES[gait_idx[0]]
                            if gait["policy_type"] in ("rough", "parkour"):
                                _switch_to_rough_or_parkour()
                            else:
                                _switch_to_flat()
                            print(f"\n  >> Gait: {gait['name']}")
                    elif i == XBOX_BTN_START:
                        if spot_rough is not None:
                            auto_gait_active[0] = not auto_gait_active[0]
                            if auto_gait_active[0]:
                                print(f"\n  >> AUTO-GAIT: ON")
                            else:
                                print(f"\n  >> AUTO-GAIT: OFF")
                    elif i == XBOX_BTN_BACK:
                        key_state["stop"] = True
                joy_prev_buttons[i] = curr

            if n_btns > XBOX_BTN_BACK and not joystick.get_button(XBOX_BTN_BACK):
                key_state["stop"] = False

except KeyboardInterrupt:
    print("\n\nStopping...")

# Cleanup
input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
if HAS_PYGAME:
    pygame.quit()

# Final stats
pos, _ = spot.robot.get_world_pose()
dist = np.sqrt(pos[0]**2 + pos[1]**2)
print("\n")
print("=" * 60)
print(f"  Session complete!")
print(f"  Final position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
print(f"  Distance from center: {dist:.2f}m")
print(f"  Simulation time: {sim_time[0]:.1f}s")
print("=" * 60)

simulation_app.close()
print("Done.")
