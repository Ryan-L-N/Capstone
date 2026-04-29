"""
Spot 100m Lava Arena - PPO Rough-Terrain Policy
=================================================

100m x 15m rectangular arena modeled after an aa lava field.
~800 random polyhedral rocks that shift, roll, and tumble when stepped on.
Now powered by the trained PPO rough-terrain policy (Phase B-easy) instead
of Isaac Sim's built-in SpotFlatTerrainPolicy.

ARENA LAYERS:
  1. Uneven base terrain (static heightfield mesh)
  2. Embedded boulders (60 static + 40 dynamic, partially buried)
  3. Loose surface rubble (~700 dynamic rocks in 3 size classes)
  4. Rectangular containment walls
  5. Lava cracks radiating across the arena floor
  6. Animated steam vents

KEYBOARD:
  W / S         Forward / Backward
  A / D         Turn left / Turn right
  SPACE         Emergency stop
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
  Back           Emergency stop

Backend: Isaac Lab AppLauncher + ManagerBasedRLEnv + RSL-RL PPO policy
Isaac Sim 5.1.0 + Isaac Lab 2.3.0
Created for AI2C Tech Capstone - MS for Autonomy, February 2026
"""

# ── 0. Parse args BEFORE any Isaac imports ──────────────────────────────
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spot 100m Lava Arena - PPO Policy")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel environments (default 1)")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to PPO model checkpoint (.pt)")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for arena generation")
parser.add_argument("--rock-count", type=int, default=200,
                    help="Number of loose rubble rocks (default: 200, max ~700 on H100)")
parser.add_argument("--mason", action="store_true", default=False,
                    help="Use Mason baseline architecture [512,256,128]")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 1. Imports (AFTER SimulationApp) ────────────────────────────────────
import math
import os
import time

import carb.input
import numpy as np
import omni
import omni.appwindow
import torch
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt
from scipy.spatial import ConvexHull

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab.sim as sim_utils

import isaaclab_tasks  # noqa: F401

# ── 2. Import our Spot PPO configs ──────────────────────────────────────
import quadruped_locomotion  # noqa: F401  — registers gym envs
from quadruped_locomotion.tasks.locomotion.config.spot.env_cfg import SpotLocomotionEnvCfg
from quadruped_locomotion.tasks.locomotion.config.spot.agents.rsl_rl_ppo_cfg import SpotPPORunnerCfg
from configs.arl_hybrid_env_cfg import SpotARLHybridEnvCfg
from configs.agents.rsl_rl_arl_hybrid_cfg import SpotARLHybridPPORunnerCfg

# ── 3. Optional pygame for Xbox controller ──────────────────────────────
HAS_PYGAME = False
try:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    HAS_PYGAME = True
except ImportError:
    pass


# ═════════════════════════════════════════════════════════════════════════
# DEFAULT CHECKPOINT
# ═════════════════════════════════════════════════════════════════════════

DEFAULT_CKPT = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "checkpoints", "spot_phase_b_easy_model_1600.pt"
))


# ═════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════

# Arena geometry — 100m x 15m rectangular
ARENA_LENGTH = 100.0
ARENA_WIDTH = 15.0
ARENA_CENTER_X = ARENA_LENGTH / 2.0
ARENA_CENTER_Y = ARENA_WIDTH / 2.0

# Robot start position
START_POS = np.array([5.0, ARENA_CENTER_Y, 0.8])

# Base terrain
BASE_RESOLUTION = 0.25
BASE_NOISE_AMPLITUDE = 0.08
BASE_NOISE_OCTAVES = 3
BASE_MOUND_COUNT = 30
BASE_MOUND_HEIGHT = 0.15
BASE_MOUND_WIDTH = 2.5

# Embedded boulders
N_BOULDERS_STATIC = 60
N_BOULDERS_DYNAMIC = 40
BOULDER_SIZE_RANGE = (0.30, 0.80)
BOULDER_MASS_RANGE = (50.0, 200.0)
BOULDER_BURY_FRACTION = (0.30, 0.50)

# Loose rubble
N_SMALL_RUBBLE = int(args_cli.rock_count * 0.60)
N_MEDIUM_RUBBLE = int(args_cli.rock_count * 0.30)
N_LARGE_RUBBLE = args_cli.rock_count - N_SMALL_RUBBLE - N_MEDIUM_RUBBLE

SMALL_SIZE_RANGE = (0.03, 0.08)
SMALL_MASS_RANGE = (0.1, 1.0)
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

# Containment walls
WALL_HEIGHT = 1.0
WALL_THICKNESS = 0.15

# Settling
SETTLING_STEPS = 200  # env.step() calls at 50Hz = 4 seconds

# Lava cracks
N_MAIN_CRACKS = 20
CRACK_WIDTH_RANGE = (0.06, 0.18)
CRACK_STEP = 0.3
CRACK_GLOW_INTENSITY = 5000.0
CRACK_GLOW_COLOR = (1.0, 0.15, 0.02)
CRACK_LIGHT_COUNT = 40
CRACK_LIGHT_INTENSITY = 5000.0
CRACK_LIGHT_RADIUS = 2.0

# Steam vents
N_STEAM_VENTS = 16
STEAM_VENT_HEIGHT = (1.0, 2.5)
STEAM_VENT_RADIUS = (0.08, 0.20)
STEAM_LIGHT_INTENSITY = 2000.0
STEAM_LIGHT_COLOR = (0.9, 0.85, 0.8)
STEAM_RISE_SPEED = 0.3
STEAM_CYCLE_PERIOD = 6.0
STEAM_OPACITY_BASE = 0.25
STEAM_OPACITY_TOP = 0.03
STEAM_ANIM_RATE = 10

# Rock material pool
N_BASALT_MATERIALS = 8
N_OXIDIZED_MATERIALS = 4

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

STABILIZE_TIME = 1.0
RECOVERY_STABILIZE = 1.5


# ═════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════

def quat_to_yaw(quat):
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def get_roll_pitch(quat):
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def is_rolled_over(quat):
    roll, pitch = get_roll_pitch(quat)
    return abs(roll) > np.radians(60) or abs(pitch) > np.radians(60)


def quat_forward_axis(quat):
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    fx = 1.0 - 2.0 * (y * y + z * z)
    fy = 2.0 * (x * y + w * z)
    fz = 2.0 * (x * z - w * y)
    return np.array([fx, fy, fz])


def apply_deadzone(value, dz=XBOX_DEADZONE):
    if abs(value) < dz:
        return 0.0
    sign = 1.0 if value >= 0 else -1.0
    return sign * (abs(value) - dz) / (1.0 - dz)


def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


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


# ═════════════════════════════════════════════════════════════════════════
# TERRAIN GENERATION: VALUE NOISE (numpy only)
# ═════════════════════════════════════════════════════════════════════════

def _value_noise_2d(x, y, seed=0):
    ix = int(np.floor(x))
    iy = int(np.floor(y))
    fx = x - ix
    fy = y - iy
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)

    def _hash(xi, yi):
        n = xi * 374761393 + yi * 668265263 + seed * 1013904223
        n = (n ^ (n >> 13)) * 1274126177
        n = n ^ (n >> 16)
        return (n & 0x7fffffff) / 0x7fffffff

    c00 = _hash(ix, iy)
    c10 = _hash(ix + 1, iy)
    c01 = _hash(ix, iy + 1)
    c11 = _hash(ix + 1, iy + 1)
    v0 = c00 * (1 - fx) + c10 * fx
    v1 = c01 * (1 - fx) + c11 * fx
    return v0 * (1 - fy) + v1 * fy


def fractal_noise(x, y, octaves=3, persistence=0.5, lacunarity=2.0, seed=0):
    total = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_val = 0.0
    for i in range(octaves):
        total += _value_noise_2d(x * frequency, y * frequency, seed + i * 7) * amplitude
        max_val += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return total / max_val


# ═════════════════════════════════════════════════════════════════════════
# ARENA CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════════

def _create_omnipbr_material(stage, path, color, roughness, friction,
                             emissive_color=None, emissive_intensity=0.0,
                             opacity=None, no_physics=False):
    mat = UsdShade.Material.Define(stage, path)

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
        phys_mat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        phys_mat.CreateStaticFrictionAttr(float(friction))
        phys_mat.CreateDynamicFrictionAttr(float(friction * 0.85))
        phys_mat.CreateRestitutionAttr(ROCK_RESTITUTION)
        physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
        physx_mat.CreateFrictionCombineModeAttr().Set("max")
        physx_mat.CreateRestitutionCombineModeAttr().Set("min")

    return mat


def create_rock_material_pool(stage, rng):
    print("  Creating rock material pool...")
    root = "/World/LavaArena/RockMaterials"
    UsdGeom.Xform.Define(stage, root)

    materials = []

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
    mats, weights, colors = zip(*material_pool)
    weights = np.array(weights)
    weights /= weights.sum()
    idx = rng.choice(len(mats), p=weights)
    return mats[idx], colors[idx]


def generate_random_rock(rng, size_min, size_max, vert_min, vert_max):
    size = rng.uniform(size_min, size_max)
    n_points = rng.randint(vert_min, min(vert_max + 1, 50))

    theta = rng.uniform(0, 2 * np.pi, n_points)
    phi = rng.uniform(0, np.pi, n_points)
    r = (size / 2) * (0.4 + 0.6 * rng.uniform(0, 1, n_points))

    sx = rng.uniform(0.6, 1.4)
    sy = rng.uniform(0.6, 1.4)
    sz = rng.uniform(0.4, 1.0)

    x = r * np.sin(phi) * np.cos(theta) * sx
    y = r * np.sin(phi) * np.sin(theta) * sy
    z = r * np.cos(phi) * sz

    points = np.column_stack([x, y, z])

    try:
        hull = ConvexHull(points)
        hull_verts = points[hull.vertices]
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(hull.vertices)}
        hull_faces = np.array([
            [old_to_new[f[0]], old_to_new[f[1]], old_to_new[f[2]]]
            for f in hull.simplices
        ])
        center = hull_verts.mean(axis=0)
        hull_verts -= center
        return hull_verts, hull_faces
    except Exception:
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
    if x < -1.0 or x > ARENA_LENGTH + 1.0 or y < -1.0 or y > ARENA_WIDTH + 1.0:
        return 0.0
    freq = 1.0 / 2.5
    h = fractal_noise(x * freq, y * freq, octaves=BASE_NOISE_OCTAVES, seed=seed)
    h = (h - 0.5) * 2.0 * BASE_NOISE_AMPLITUDE
    return h


def create_base_terrain(stage, rng):
    print("  Creating base terrain mesh (100m x 15m)...")

    res = BASE_RESOLUTION
    nx = int(ARENA_LENGTH / res) + 1
    ny = int(ARENA_WIDTH / res) + 1

    mound_positions = []
    for _ in range(BASE_MOUND_COUNT):
        mx = rng.uniform(5.0, ARENA_LENGTH - 5.0)
        my = rng.uniform(2.0, ARENA_WIDTH - 2.0)
        mound_positions.append((mx, my))

    vertices = []
    vertex_map = {}

    for iy in range(ny):
        for ix in range(nx):
            x = ix * res
            y = iy * res

            z = get_base_height(x, y, seed=args_cli.seed)

            for mx, my in mound_positions:
                md = np.sqrt((x - mx)**2 + (y - my)**2)
                if md < BASE_MOUND_WIDTH * 2:
                    z += BASE_MOUND_HEIGHT * np.exp(-md**2 / (BASE_MOUND_WIDTH**2))

            vertex_map[(ix, iy)] = len(vertices)
            vertices.append(Gf.Vec3f(float(x), float(y), float(z)))

    face_counts = []
    face_indices = []

    for iy in range(ny - 1):
        for ix in range(nx - 1):
            v00 = vertex_map.get((ix, iy))
            v10 = vertex_map.get((ix + 1, iy))
            v01 = vertex_map.get((ix, iy + 1))
            v11 = vertex_map.get((ix + 1, iy + 1))

            if v00 is not None and v10 is not None and v01 is not None and v11 is not None:
                face_counts.append(3)
                face_indices.extend([v00, v10, v11])
                face_counts.append(3)
                face_indices.extend([v00, v11, v01])

    mesh_path = "/World/LavaArena/BaseTerrain"
    UsdGeom.Xform.Define(stage, "/World/LavaArena")
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(vertices))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_counts))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_indices))
    mesh.GetSubdivisionSchemeAttr().Set("none")

    mesh.GetDisplayColorAttr().Set([Gf.Vec3f(0.07, 0.06, 0.05)])

    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    collision_api.GetApproximationAttr().Set("none")

    mat = _create_omnipbr_material(
        stage, "/World/LavaArena/BaseMaterial",
        color=Gf.Vec3f(0.07, 0.06, 0.05),
        roughness=0.92,
        friction=0.85)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

    print(f"    Vertices: {len(vertices)}, Faces: {len(face_counts)}")


def create_lava_rock(stage, path, vertices, faces, position, orientation,
                     mass, is_dynamic, material, color):
    mesh = UsdGeom.Mesh.Define(stage, path)

    points = Vt.Vec3fArray([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2]))
                            for v in vertices])
    mesh.GetPointsAttr().Set(points)

    face_counts_arr = Vt.IntArray([3] * len(faces))
    face_indices_arr = Vt.IntArray(faces.flatten().tolist())
    mesh.GetFaceVertexCountsAttr().Set(face_counts_arr)
    mesh.GetFaceVertexIndicesAttr().Set(face_indices_arr)
    mesh.GetSubdivisionSchemeAttr().Set("none")

    mesh.GetDisplayColorAttr().Set([Gf.Vec3f(color[0], color[1], color[2])])

    prim = mesh.GetPrim()
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(float(position[0]),
                                       float(position[1]),
                                       float(position[2])))
    xf.AddOrientOp().Set(orientation)

    UsdPhysics.CollisionAPI.Apply(prim)
    collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    collision_api.GetApproximationAttr().Set("convexHull")

    if is_dynamic:
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(float(mass))

        physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physx_rb.CreateLinearDampingAttr().Set(ROCK_LINEAR_DAMPING)
        physx_rb.CreateAngularDampingAttr().Set(ROCK_ANGULAR_DAMPING)
        physx_rb.CreateSleepThresholdAttr().Set(ROCK_SLEEP_THRESHOLD)

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def random_quat(rng):
    u1, u2, u3 = rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1)
    s1 = np.sqrt(1 - u1)
    s2 = np.sqrt(u1)
    w = s2 * np.cos(2 * np.pi * u3)
    x = s1 * np.sin(2 * np.pi * u2)
    y = s1 * np.cos(2 * np.pi * u2)
    z = s2 * np.sin(2 * np.pi * u3)
    return Gf.Quatf(float(w), float(x), float(y), float(z))


def random_point_in_rect(rng, margin=1.0):
    x = rng.uniform(margin, ARENA_LENGTH - margin)
    y = rng.uniform(margin, ARENA_WIDTH - margin)
    return x, y


def create_embedded_boulders(stage, rng, material_pool):
    print("  Creating embedded boulders...")
    root = "/World/LavaArena/Boulders"
    UsdGeom.Xform.Define(stage, root)

    n_total = N_BOULDERS_STATIC + N_BOULDERS_DYNAMIC

    for i in range(n_total):
        is_dynamic = (i >= N_BOULDERS_STATIC)

        x, y = random_point_in_rect(rng, margin=1.5)
        while x < 8.0:
            x, y = random_point_in_rect(rng, margin=1.5)

        verts, faces = generate_random_rock(
            rng, BOULDER_SIZE_RANGE[0], BOULDER_SIZE_RANGE[1], 20, 40)

        z_extent = verts[:, 2].max() - verts[:, 2].min()
        bury_frac = rng.uniform(BOULDER_BURY_FRACTION[0], BOULDER_BURY_FRACTION[1])
        z_base = get_base_height(x, y, seed=args_cli.seed)
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
            x, y = random_point_in_rect(rng, margin=0.5)

            verts, faces = generate_random_rock(
                rng, size_range[0], size_range[1], vert_range[0], vert_range[1])

            z_extent = verts[:, 2].max() - verts[:, 2].min()
            z_base = get_base_height(x, y, seed=args_cli.seed)
            z_pos = z_base + z_extent / 2 + 0.02

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
    print("  Creating lava cracks...")
    root = "/World/LavaArena/Cracks"
    UsdGeom.Xform.Define(stage, root)

    crack_paths = []
    for i in range(N_MAIN_CRACKS):
        src_x = rng.uniform(5.0, ARENA_LENGTH - 5.0)
        src_y = rng.uniform(2.0, ARENA_WIDTH - 2.0)
        angle = rng.uniform(0, 2 * np.pi)
        x, y = src_x, src_y
        path_pts = [(x, y)]

        length = rng.uniform(5.0, 25.0)
        n_steps = int(length / CRACK_STEP)

        for s in range(n_steps):
            angle += rng.uniform(-0.35, 0.35)
            x += CRACK_STEP * np.cos(angle)
            y += CRACK_STEP * np.sin(angle)
            if x < 0.5 or x > ARENA_LENGTH - 0.5 or y < 0.5 or y > ARENA_WIDTH - 0.5:
                break
            path_pts.append((x, y))

            if s > 3 and rng.uniform() < 0.15 and len(crack_paths) < 60:
                branch_angle = angle + rng.choice([-1, 1]) * rng.uniform(0.4, 1.0)
                branch = [(x, y)]
                bx, by = x, y
                for _ in range(rng.randint(3, 10)):
                    branch_angle += rng.uniform(-0.2, 0.2)
                    bx += CRACK_STEP * np.cos(branch_angle)
                    by += CRACK_STEP * np.sin(branch_angle)
                    if bx < 0.5 or bx > ARENA_LENGTH - 0.5 or by < 0.5 or by > ARENA_WIDTH - 0.5:
                        break
                    branch.append((bx, by))
                if len(branch) > 1:
                    crack_paths.append(branch)

        if len(path_pts) > 1:
            crack_paths.append(path_pts)

    all_verts = []
    all_face_counts = []
    all_face_indices = []
    vert_offset = 0
    light_positions = []

    for ci, crack in enumerate(crack_paths):
        width = rng.uniform(CRACK_WIDTH_RANGE[0], CRACK_WIDTH_RANGE[1])

        for j in range(len(crack) - 1):
            x0, y0 = crack[j]
            x1, y1 = crack[j + 1]

            dx = x1 - x0
            dy = y1 - y0
            seg_len = np.sqrt(dx**2 + dy**2)
            if seg_len < 0.001:
                continue
            nx, ny = -dy / seg_len, dx / seg_len

            w = width * rng.uniform(0.7, 1.3)

            z0 = get_base_height(x0, y0, seed=args_cli.seed) + 0.015
            z1 = get_base_height(x1, y1, seed=args_cli.seed) + 0.015

            all_verts.extend([
                Gf.Vec3f(float(x0 - nx * w), float(y0 - ny * w), float(z0)),
                Gf.Vec3f(float(x0 + nx * w), float(y0 + ny * w), float(z0)),
                Gf.Vec3f(float(x1 + nx * w), float(y1 + ny * w), float(z1)),
                Gf.Vec3f(float(x1 - nx * w), float(y1 - ny * w), float(z1)),
            ])

            all_face_counts.extend([3, 3])
            all_face_indices.extend([
                vert_offset, vert_offset + 1, vert_offset + 2,
                vert_offset, vert_offset + 2, vert_offset + 3,
            ])
            vert_offset += 4

        if len(crack) > 5:
            mid = len(crack) // 2
            mx, my = crack[mid]
            mz = get_base_height(mx, my, seed=args_cli.seed)
            light_positions.append((mx, my, mz))

    if not all_verts:
        print("    No crack geometry generated")
        return

    mesh = UsdGeom.Mesh.Define(stage, f"{root}/crack_mesh")
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(all_verts))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(all_face_counts))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(all_face_indices))
    mesh.GetSubdivisionSchemeAttr().Set("none")
    mesh.GetDoubleSidedAttr().Set(True)

    mesh.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.3, 0.05)])

    glow_color = Gf.Vec3f(CRACK_GLOW_COLOR[0], CRACK_GLOW_COLOR[1], CRACK_GLOW_COLOR[2])
    crack_mat = _create_omnipbr_material(
        stage, f"{root}/crack_material",
        color=Gf.Vec3f(1.0, 0.3, 0.05),
        roughness=0.3,
        friction=0.6,
        emissive_color=glow_color,
        emissive_intensity=CRACK_GLOW_INTENSITY)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(crack_mat)

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
    print("  Creating steam vents...")
    root = "/World/LavaArena/SteamVents"
    UsdGeom.Xform.Define(stage, root)

    all_puffs = []

    for i in range(N_STEAM_VENTS):
        x = rng.uniform(8.0, ARENA_LENGTH - 5.0)
        y = rng.uniform(2.0, ARENA_WIDTH - 2.0)
        z_base = get_base_height(x, y, seed=args_cli.seed)

        height = rng.uniform(STEAM_VENT_HEIGHT[0], STEAM_VENT_HEIGHT[1])
        vent_root = f"{root}/vent_{i}"
        UsdGeom.Xform.Define(stage, vent_root)

        n_puffs = rng.randint(5, 9)
        for p in range(n_puffs):
            phase = p / max(n_puffs - 1, 1)
            initial_t = phase

            puff_z = z_base + phase * height
            base_radius = 0.06 + rng.uniform(0.0, 0.04)
            offset_x = rng.uniform(-0.12, 0.12)
            offset_y = rng.uniform(-0.12, 0.12)

            puff_path = f"{vent_root}/puff_{p}"
            sphere = UsdGeom.Sphere.Define(stage, puff_path)
            sphere.GetRadiusAttr().Set(float(base_radius + phase * 0.30))

            grey = rng.uniform(0.75, 0.92)
            sphere.GetDisplayColorAttr().Set([Gf.Vec3f(grey, grey * 0.96, grey * 0.92)])

            puff_xf = UsdGeom.Xformable(sphere.GetPrim())
            puff_xf.AddTranslateOp().Set(Gf.Vec3d(
                float(x + offset_x), float(y + offset_y), float(puff_z)))

            all_puffs.append({
                "path": puff_path,
                "base_x": float(x),
                "base_y": float(y),
                "z_base": float(z_base),
                "height": float(height),
                "offset_x": float(offset_x),
                "offset_y": float(offset_y),
                "base_radius": float(base_radius),
                "phase": float(initial_t),
                "speed": float(rng.uniform(0.8, 1.2)),
            })

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


def create_containment_walls(stage):
    print("  Creating containment walls...")
    root = "/World/LavaArena/Walls"
    UsdGeom.Xform.Define(stage, root)

    wall_mat = _create_omnipbr_material(
        stage, "/World/LavaArena/WallMaterial",
        color=Gf.Vec3f(0.12, 0.10, 0.09),
        roughness=0.90,
        friction=0.80)

    walls = [
        ("north", ARENA_LENGTH / 2, ARENA_WIDTH + WALL_THICKNESS / 2,
         ARENA_LENGTH + 2 * WALL_THICKNESS, WALL_THICKNESS),
        ("south", ARENA_LENGTH / 2, -WALL_THICKNESS / 2,
         ARENA_LENGTH + 2 * WALL_THICKNESS, WALL_THICKNESS),
        ("east", ARENA_LENGTH + WALL_THICKNESS / 2, ARENA_WIDTH / 2,
         WALL_THICKNESS, ARENA_WIDTH),
        ("west", -WALL_THICKNESS / 2, ARENA_WIDTH / 2,
         WALL_THICKNESS, ARENA_WIDTH),
    ]

    for name, cx, cy, sx, sy in walls:
        path = f"{root}/{name}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(1.0)

        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(float(cx), float(cy),
                                           float(WALL_HEIGHT / 2)))
        xf.AddScaleOp().Set(Gf.Vec3d(float(sx), float(sy), float(WALL_HEIGHT)))

        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.10, 0.08, 0.07)])
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(wall_mat)

    print(f"    Walls: 4 segments, {ARENA_LENGTH}m x {ARENA_WIDTH}m, "
          f"height={WALL_HEIGHT}m")


def create_arena_lighting(stage):
    light_path = "/World/Lights"
    UsdGeom.Xform.Define(stage, light_path)

    dome = UsdLux.DomeLight.Define(stage, f"{light_path}/dome")
    dome.CreateIntensityAttr(120.0)
    dome.CreateColorAttr(Gf.Vec3f(0.6, 0.55, 0.65))

    haze_dome = UsdLux.DomeLight.Define(stage, f"{light_path}/haze_dome")
    haze_dome.CreateIntensityAttr(30.0)
    haze_dome.CreateColorAttr(Gf.Vec3f(1.0, 0.35, 0.15))

    sun = UsdLux.DistantLight.Define(stage, f"{light_path}/sun")
    sun.CreateIntensityAttr(1500.0)
    sun.CreateAngleAttr(2.0)
    sun_prim = stage.GetPrimAtPath(f"{light_path}/sun")
    UsdGeom.Xformable(sun_prim).AddRotateXYZOp().Set(Gf.Vec3f(-15, 25, 0))
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.75, 0.55))

    rim = UsdLux.DistantLight.Define(stage, f"{light_path}/lava_rim")
    rim.CreateIntensityAttr(400.0)
    rim.CreateAngleAttr(8.0)
    rim.CreateColorAttr(Gf.Vec3f(1.0, 0.25, 0.05))
    rim_prim = stage.GetPrimAtPath(f"{light_path}/lava_rim")
    UsdGeom.Xformable(rim_prim).AddRotateXYZOp().Set(Gf.Vec3f(80, 0, 0))

    print("  Lighting: dark volcanic (overcast dome + dim sun + lava rim)")


def create_volcanic_atmosphere():
    try:
        import carb.settings
        settings = carb.settings.get_settings()

        settings.set("/rtx/fog/enabled", True)
        settings.set("/rtx/fog/fogColor", [0.12, 0.08, 0.05])
        settings.set("/rtx/fog/fogColorIntensity", 0.6)
        settings.set("/rtx/fog/fogStartDist", 5.0)
        settings.set("/rtx/fog/fogEndDist", 40.0)

        settings.set("/rtx/post/tonemap/filmIso", 100.0)
        settings.set("/rtx/post/tonemap/whitepoint", 6.0)

        print("  Atmosphere: RTX fog + tone mapping (volcanic haze)")
    except Exception as e:
        print(f"  Atmosphere: Could not apply RTX fog ({e}), using lighting only")


def build_lava_arena(stage, rng):
    print()
    print("=" * 50)
    print("  BUILDING 100m LAVA ROCK ARENA")
    print("=" * 50)

    create_arena_lighting(stage)
    create_volcanic_atmosphere()
    rock_materials = create_rock_material_pool(stage, rng)
    create_base_terrain(stage, rng)
    create_lava_cracks(stage, rng)
    create_embedded_boulders(stage, rng, rock_materials)
    create_loose_rubble(stage, rng, rock_materials)
    steam_puffs = create_steam_vents(stage, rng)
    create_containment_walls(stage)

    total_rocks = (N_BOULDERS_STATIC + N_BOULDERS_DYNAMIC +
                   N_SMALL_RUBBLE + N_MEDIUM_RUBBLE + N_LARGE_RUBBLE)
    dynamic_rocks = (N_BOULDERS_DYNAMIC +
                     N_SMALL_RUBBLE + N_MEDIUM_RUBBLE + N_LARGE_RUBBLE)
    print()
    print(f"  Arena complete: {total_rocks} rocks ({dynamic_rocks} dynamic)")
    print(f"  + lava cracks, {N_STEAM_VENTS} steam vents, volcanic atmosphere")
    print(f"  Dimensions: {ARENA_LENGTH:.0f}m x {ARENA_WIDTH:.0f}m")
    print("=" * 50)
    print()
    return steam_puffs


# ═════════════════════════════════════════════════════════════════════════
# TELEOP ENV CONFIG
# ═════════════════════════════════════════════════════════════════════════

@configclass
class SpotLavaArenaEnvCfg(SpotLocomotionEnvCfg):
    """Single-env flat-terrain config for lava arena teleop."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 3600.0

        # Flat terrain — lava arena is built as USD prims on top
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            debug_vis=False,
        )

        # Reduce GPU PhysX buffers for laptop GPUs (8GB VRAM)
        self.sim.physx.gpu_collision_stack_size = 2**25    # 32MB (was 2GB)
        self.sim.physx.gpu_max_rigid_contact_count = 2**21  # 2M (was 16M)
        self.sim.physx.gpu_max_rigid_patch_count = 2**21    # 2M (was 16M)

        # Disable contact sensor — its GPU RigidContactView can't handle
        # dynamic rigid bodies (rocks) added after env initialization.
        # Contact sensor is only used by reward terms, not observations.
        self.scene.contact_forces = None

        # Disable all rewards (teleop = inference only, no training)
        self.rewards = None

        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.terminations.terrain_out_of_bounds = None
        self.curriculum = None


class SpotARLLavaArenaEnvCfg(SpotARLHybridEnvCfg):
    """Mason's env config adapted for lava arena (obs order: height_scan first)."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 3600.0

        # Flat terrain — lava arena is built as USD prims on top
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            debug_vis=False,
        )

        self.sim.physx.gpu_collision_stack_size = 2**25
        self.sim.physx.gpu_max_rigid_contact_count = 2**21
        self.sim.physx.gpu_max_rigid_patch_count = 2**21

        self.scene.contact_forces = None
        self.rewards = None
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        # Disable terminations that reference contact_forces sensor
        self.terminations.body_contact = None
        self.terminations.terrain_out_of_bounds = None
        self.curriculum = None


# ═════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL MUTABLE STATE
# ═════════════════════════════════════════════════════════════════════════

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

# Set in main()
viewport_api = None
fpv_cam_path = None
default_camera_path = None


# ═════════════════════════════════════════════════════════════════════════
# KEYBOARD HANDLER
# ═════════════════════════════════════════════════════════════════════════

def on_keyboard_event(event, *args_ev, **kwargs):
    global viewport_api, fpv_cam_path, default_camera_path

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
        if key == carb.input.KeyboardInput.T:
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
            # Position info — printed from main loop context
            pass

    return True


# ═════════════════════════════════════════════════════════════════════════
# SELFRIGHT HANDLER
# ═════════════════════════════════════════════════════════════════════════

def handle_selfright(env, dt):
    if not selfright_active[0]:
        return False

    robot = env.unwrapped.scene["robot"]
    quat = robot.data.root_quat_w[0].cpu().numpy()
    roll, pitch = get_roll_pitch(quat)
    abs_roll = abs(roll)

    if abs_roll < np.radians(SELFRIGHT_UPRIGHT_DEG) and abs(pitch) < np.radians(SELFRIGHT_UPRIGHT_DEG):
        selfright_upright_timer[0] += dt
        if selfright_upright_timer[0] >= SELFRIGHT_UPRIGHT_TIME:
            selfright_active[0] = False
            selfright_upright_timer[0] = 0.0
            smoother.reset()
            print(f"\n  >> SELFRIGHT COMPLETE!")
            return False
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
    current_w = robot.data.root_ang_vel_w[0].cpu().numpy()

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
        roll_impulse = forward_norm * roll_dir * accel * dt
        new_w = current_w + roll_impulse

        w_mag = np.linalg.norm(new_w)
        if w_mag > SELFRIGHT_MAX_ROLL_VEL:
            new_w = new_w * (SELFRIGHT_MAX_ROLL_VEL / w_mag)

        new_w_t = torch.tensor(new_w, dtype=torch.float32, device=env.unwrapped.device).unsqueeze(0)
        robot.write_root_velocity_to_sim(
            torch.cat([robot.data.root_lin_vel_w[:1], new_w_t], dim=-1)
        )

        if abs_roll > np.radians(45):
            lift_factor = min(1.0, (abs_roll - np.radians(45)) / np.radians(135))
            target_lift = SELFRIGHT_GROUND_LIFT * lift_factor
            current_lin_vel = robot.data.root_lin_vel_w[0].cpu().numpy()
            if current_lin_vel[2] < target_lift:
                current_lin_vel[2] = target_lift
                lin_t = torch.tensor(current_lin_vel, dtype=torch.float32, device=env.unwrapped.device).unsqueeze(0)
                robot.write_root_velocity_to_sim(
                    torch.cat([lin_t, new_w_t], dim=-1)
                )
    else:
        damping_factor = 1.0 - SELFRIGHT_DAMPING * dt
        damped_w = current_w * damping_factor
        damped_t = torch.tensor(damped_w, dtype=torch.float32, device=env.unwrapped.device).unsqueeze(0)
        robot.write_root_velocity_to_sim(
            torch.cat([robot.data.root_lin_vel_w[:1], damped_t], dim=-1)
        )

    return True


# ═════════════════════════════════════════════════════════════════════════
# STEAM ANIMATION
# ═════════════════════════════════════════════════════════════════════════

def setup_steam_animation(stage, steam_puff_data):
    """Cache xform ops for steam puffs."""
    steam_xforms = []
    for pd in steam_puff_data:
        prim = stage.GetPrimAtPath(pd["path"])
        xf = UsdGeom.Xformable(prim)
        ops = xf.GetOrderedXformOps()
        translate_op = ops[0] if ops else None
        steam_xforms.append((xf, translate_op, pd))
    return steam_xforms


steam_anim_step = [0]


def animate_steam(stage, steam_xforms, dt):
    steam_anim_step[0] += 1
    if steam_anim_step[0] % STEAM_ANIM_RATE != 0:
        return

    for _xf, _translate_op, pd in steam_xforms:
        if _translate_op is None:
            continue

        pd["phase"] += (dt * STEAM_ANIM_RATE * pd["speed"]) / STEAM_CYCLE_PERIOD
        if pd["phase"] >= 1.0:
            pd["phase"] -= 1.0

        t = pd["phase"]

        z = pd["z_base"] + t * pd["height"]
        drift_scale = 1.0 + t * 2.5
        dx = pd["offset_x"] * drift_scale
        dy = pd["offset_y"] * drift_scale

        _translate_op.Set(Gf.Vec3d(
            pd["base_x"] + dx,
            pd["base_y"] + dy,
            z))

        puff_prim = stage.GetPrimAtPath(pd["path"])
        radius = pd["base_radius"] + t * 0.35
        UsdGeom.Sphere(puff_prim).GetRadiusAttr().Set(float(radius))


# ═════════════════════════════════════════════════════════════════════════
# MAIN 50Hz TELEOP LOOP
# ═════════════════════════════════════════════════════════════════════════

def run_teleop(env, policy, policy_nn, obs, stage, joystick_ref, steam_xforms):
    global viewport_api, fpv_cam_path, default_camera_path

    device = env.unwrapped.device
    dt = env.unwrapped.step_dt
    sim_time = 0.0
    last_hud_time = 0.0
    last_command = np.array([0.0, 0.0, 0.0])
    recovery_timer = 0.0

    print(f"  Control dt: {dt:.4f}s ({1.0/dt:.0f} Hz)")
    print(f"\n  Controls active! Click on simulation window to capture keyboard.")
    print("=" * 60 + "\n")

    while simulation_app.is_running() and not key_state["exit"]:
        t0 = time.time()
        sim_time += dt

        # Animate steam
        animate_steam(stage, steam_xforms, dt)

        # ── Handle reset ────────────────────────────────────────────
        if key_state["reset"]:
            key_state["reset"] = False
            robot = env.unwrapped.scene["robot"]
            start_pos_t = torch.tensor([START_POS], dtype=torch.float32, device=device)
            start_quat_t = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
            zero_vel_t = torch.zeros(1, 6, dtype=torch.float32, device=device)
            robot.write_root_pose_to_sim(torch.cat([start_pos_t, start_quat_t], dim=-1))
            robot.write_root_velocity_to_sim(zero_vel_t)
            smoother.reset()
            auto_walk_active[0] = False
            recovery_timer = RECOVERY_STABILIZE
            print("\n  >> Robot reset to start!")

        # ── Selfright mode ──────────────────────────────────────────
        if handle_selfright(env, dt):
            zero_actions = torch.zeros(1, 12, dtype=torch.float32, device=device)
            obs, _, dones, _ = env.step(zero_actions)
            if dones.any():
                policy_nn.reset(dones)
        elif recovery_timer > 0:
            recovery_timer -= dt
            zero_actions = torch.zeros(1, 12, dtype=torch.float32, device=device)
            obs, _, dones, _ = env.step(zero_actions)
            if dones.any():
                policy_nn.reset(dones)
        else:
            # ── Get robot state ─────────────────────────────────────
            robot = env.unwrapped.scene["robot"]
            pos = robot.data.root_pos_w[0].cpu().numpy()
            quat = robot.data.root_quat_w[0].cpu().numpy()
            yaw = quat_to_yaw(quat)

            # ── Compute velocity command ────────────────────────────
            if auto_walk_active[0]:
                target_vx = 0.6
                desired_yaw = 0.0
                dy = ARENA_CENTER_Y - pos[1]
                target_wz_y = np.clip(dy * 0.3, -0.3, 0.3)
                yaw_error = normalize_angle(desired_yaw - yaw)
                target_wz = np.clip(yaw_error * 0.5 + target_wz_y, -0.5, 0.5)
                vx, wz = smoother.update(target_vx, target_wz, dt)
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

                vx, wz = smoother.update(target_vx, target_wz, dt)
                command = np.array([vx, 0.0, wz])

            last_command = command

            # ── Override velocity command in obs before policy ───
            cmd_t = torch.tensor([[command[0], command[1], command[2]]],
                                 dtype=torch.float32, device=device)
            # Mason obs: [scan(187)|lin(3)|ang(3)|grav(3)|cmd(3)|...]  → cmd at 196:199
            # Our obs:   [lin(3)|ang(3)|grav(3)|cmd(3)|...]            → cmd at 9:12
            cmd_idx = 196 if args_cli.mason else 9
            obs[:, cmd_idx:cmd_idx+3] = cmd_t

            # ── Run policy ──────────────────────────────────────
            with torch.inference_mode():
                actions = policy(obs)

            # ── Step environment ────────────────────────────────
            obs, _, dones, _ = env.step(actions)

            # ── Override velocity command AFTER step ─────────────
            vel_term = env.unwrapped.command_manager.get_term("base_velocity")
            vel_term.vel_command_b[:] = cmd_t
            obs[:, cmd_idx:cmd_idx+3] = cmd_t

            # ── Handle episode resets ───────────────────────────
            if dones.any():
                policy_nn.reset(dones)
                start_pos_t = torch.tensor([START_POS], dtype=torch.float32, device=device)
                start_quat_t = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
                zero_vel_t = torch.zeros(1, 6, dtype=torch.float32, device=device)
                robot.write_root_pose_to_sim(torch.cat([start_pos_t, start_quat_t], dim=-1))
                robot.write_root_velocity_to_sim(zero_vel_t)

        # ── Xbox controller polling ─────────────────────────────────
        if joystick_ref is not None:
            pygame.event.pump()

            raw_fwd = -joystick_ref.get_axis(XBOX_AXIS_FWD)
            raw_turn = -joystick_ref.get_axis(XBOX_AXIS_TURN)
            joy_analog["fwd"] = apply_deadzone(raw_fwd)
            joy_analog["turn"] = apply_deadzone(raw_turn)

            n_btns = joystick_ref.get_numbuttons()
            for i in range(min(n_btns, len(joy_prev_buttons))):
                curr = joystick_ref.get_button(i)
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
                    elif i == XBOX_BTN_BACK:
                        key_state["stop"] = True
                joy_prev_buttons[i] = curr

            if n_btns > XBOX_BTN_BACK and not joystick_ref.get_button(XBOX_BTN_BACK):
                key_state["stop"] = False

        # ── HUD (every 0.5s) ───────────────────────────────────────
        if sim_time - last_hud_time >= 0.5:
            last_hud_time = sim_time
            robot = env.unwrapped.scene["robot"]
            pos = robot.data.root_pos_w[0].cpu().numpy()
            quat = robot.data.root_quat_w[0].cpu().numpy()

            cam_str = "FPV" if fpv_active[0] else "ORB"
            walk_str = "AUTO" if auto_walk_active[0] else "WASD"
            progress = (pos[0] / ARENA_LENGTH) * 100.0
            roll, pitch = get_roll_pitch(quat)

            if selfright_active[0]:
                status = f"SELFRIGHT(R:{np.degrees(roll):+.0f})"
            elif is_rolled_over(quat):
                status = "FALLEN! (X=selfright)"
            else:
                status = "OK"

            print(f"\r  [{sim_time:6.1f}s] PPO {cam_str} {walk_str} | "
                  f"Pos:({pos[0]:5.1f},{pos[1]:5.1f}) Z:{pos[2]:.3f} "
                  f"Progress:{progress:5.1f}% | "
                  f"P:{np.degrees(pitch):+5.1f} R:{np.degrees(roll):+5.1f} | "
                  f"vx={last_command[0]:+.2f} wz={last_command[2]:+.2f} | "
                  f"{status}", end="     ")

        # ── Real-time pacing ────────────────────────────────────────
        elapsed = time.time() - t0
        if dt > elapsed:
            time.sleep(dt - elapsed)


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    global viewport_api, fpv_cam_path, default_camera_path
    global joystick, joy_prev_buttons

    # ── Xbox controller setup ───────────────────────────────────────
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

    # ── Print controls ──────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  SPOT 100m LAVA ROCK ARENA - PPO Rough-Terrain Policy")
    print("=" * 60)
    if joystick:
        print()
        print("  XBOX CONTROLLER:")
        print("    Left Stick     Forward/back + turn")
        print("    A              Toggle auto-walk")
        print("    B              Toggle selfright")
        print("    Y              Reset to start")
        print("    LB             Toggle FPV camera")
        print("    Back           Emergency stop")
    print()
    print("  KEYBOARD:")
    print("    W/S       Forward / Backward")
    print("    A/D       Turn left / Turn right")
    print("    SPACE     Emergency stop")
    print("    T         Toggle auto-walk (vx=0.6)")
    print("    M         Toggle FPV camera")
    print("    X         Selfright mode")
    print("    H         Show position / progress info")
    print("    R         Reset to start")
    print("    ESC       Exit")
    print()

    # ── Environment + Agent config ───────────────────────────────────
    if args_cli.mason:
        # Use Mason's env config (obs order: height_scan first) + smaller network
        env_cfg = SpotARLLavaArenaEnvCfg()
        agent_cfg = SpotARLHybridPPORunnerCfg()
        print("[INFO] Mason mode: HybridObservationsCfg (scan-first) + [512,256,128] network")
    else:
        env_cfg = SpotLavaArenaEnvCfg()
        agent_cfg = SpotPPORunnerCfg()
    env_cfg.scene.num_envs = args_cli.num_envs or 1
    env_cfg.seed = args_cli.seed

    # ── Checkpoint ──────────────────────────────────────────────────
    ckpt = args_cli.checkpoint or DEFAULT_CKPT
    if not os.path.exists(ckpt):
        print(f"[ERROR] Checkpoint not found: {ckpt}")
        print(f"  Use --checkpoint /path/to/model.pt to specify.")
        os._exit(1)
    print(f"[INFO] Checkpoint: {ckpt}")

    # ── Create environment ──────────────────────────────────────────
    env = gym.make(
        "Isaac-Velocity-Rough-Spot-v0",
        cfg=env_cfg,
    )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO] Environment created. Device: {env.unwrapped.device}")

    # ── Load trained policy ─────────────────────────────────────────
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device,
    )
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy
    print(f"[INFO] PPO policy loaded ({os.path.basename(ckpt)})")

    # ── Get initial observations ────────────────────────────────────
    obs = env.get_observations()
    print(f"[INFO] Observation shape: {obs.shape}")

    # ── Build lava arena on top of flat ground ──────────────────────
    stage = omni.usd.get_context().get_stage()
    device = env.unwrapped.device

    arena_rng = np.random.RandomState(args_cli.seed)
    steam_puff_data = build_lava_arena(stage, arena_rng)
    steam_xforms = setup_steam_animation(stage, steam_puff_data)

    # ── Settling: move robot up, let rocks fall ─────────────────────
    print(f"\n  Settling rocks for {SETTLING_STEPS * env.unwrapped.step_dt:.1f}s...")
    robot = env.unwrapped.scene["robot"]

    # Move robot high up out of the way
    suspend_pos = torch.tensor([[0.0, 0.0, 15.0]], dtype=torch.float32, device=device)
    suspend_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    zero_vel = torch.zeros(1, 6, dtype=torch.float32, device=device)
    robot.write_root_pose_to_sim(torch.cat([suspend_pos, suspend_quat], dim=-1))
    robot.write_root_velocity_to_sim(zero_vel)

    settle_start = time.time()
    zero_actions = torch.zeros(1, 12, dtype=torch.float32, device=device)
    for _ in range(SETTLING_STEPS):
        obs, _, dones, _ = env.step(zero_actions)
        if dones.any():
            policy_nn.reset(dones)
        # Keep robot suspended
        robot.write_root_pose_to_sim(torch.cat([suspend_pos, suspend_quat], dim=-1))
        robot.write_root_velocity_to_sim(zero_vel)
    settle_elapsed = time.time() - settle_start
    print(f"  Settling complete ({settle_elapsed:.1f}s wall time)")

    # ── Place robot at start ────────────────────────────────────────
    start_pos_t = torch.tensor([START_POS], dtype=torch.float32, device=device)
    start_quat_t = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    robot.write_root_pose_to_sim(torch.cat([start_pos_t, start_quat_t], dim=-1))
    robot.write_root_velocity_to_sim(zero_vel)

    # A few steps to let robot settle on terrain
    for _ in range(10):
        obs, _, dones, _ = env.step(zero_actions)
        if dones.any():
            policy_nn.reset(dones)
    obs = env.get_observations()
    print(f"  Spot placed at start: ({START_POS[0]:.1f}, {START_POS[1]:.1f}, {START_POS[2]:.1f})")

    # ── FPV camera ──────────────────────────────────────────────────
    fpv_cam_path = "/World/envs/env_0/Robot/body/fpv_camera"
    fpv_cam = UsdGeom.Camera.Define(stage, fpv_cam_path)
    fpv_cam.CreateFocalLengthAttr(18.0)
    fpv_cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
    cam_xform = UsdGeom.Xformable(fpv_cam.GetPrim())
    cam_xform.AddTranslateOp().Set(Gf.Vec3d(0.4, 0.0, 0.15))
    cam_xform.AddOrientOp().Set(Gf.Quatf(0.5, 0.5, -0.5, -0.5))

    viewport_api = get_active_viewport()
    default_camera_path = viewport_api.camera_path

    # ── Keyboard subscription ───────────────────────────────────────
    input_interface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    # ── Run teleop loop ─────────────────────────────────────────────
    try:
        run_teleop(env, policy, policy_nn, obs, stage, joystick, steam_xforms)
    except KeyboardInterrupt:
        print("\n\nStopping...")

    # ── Cleanup ─────────────────────────────────────────────────────
    input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
    if HAS_PYGAME:
        pygame.quit()

    robot = env.unwrapped.scene["robot"]
    pos = robot.data.root_pos_w[0].cpu().numpy()
    progress = (pos[0] / ARENA_LENGTH) * 100.0
    print("\n")
    print("=" * 60)
    print(f"  Session complete!")
    print(f"  Final position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    print(f"  Progress: {progress:.1f}% ({pos[0]:.1f}m / {ARENA_LENGTH:.0f}m)")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
    os._exit(0)
