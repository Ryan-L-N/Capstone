"""
Spot Gravel Slope Challenge -- 50deg Loose Gravel Incline
=======================================================

15m x 15m pitched surface at 50 degrees, covered in ~3 inches (7.5cm)
of loose dynamic gravel particles. Simulates a tarp/roof surface with
unstable gravel on top.

Physics:
  - Base surface: moderate friction (tarp/shingle grip underneath)
  - Gravel particles: low dynamic friction, high restitution
  - 3 size classes of gravel (1-4cm diameter)
  - Gravel shifts and slides under robot weight

KEYBOARD:
  W/S       Forward/backward
  A/D       Turn left/right
  G         Cycle gait (FLAT/ROUGH)
  T         Toggle auto-walk
  R         Reset robot to bottom of slope
  ESC       Exit

Isaac Sim 5.1.0 + Isaac Lab 2.3.0
"""

import numpy as np
import argparse
import sys
import os
import math
import time

parser = argparse.ArgumentParser(description="Spot Gravel Slope Challenge")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="PPO checkpoint for rough terrain policy")
parser.add_argument("--gravel-count", type=int, default=200,
                    help="Number of loose dynamic rocks on top (default: 200)")
parser.add_argument("--slope-deg", type=float, default=40.0,
                    help="Slope angle in degrees (default: 40)")
args, unknown = parser.parse_known_args()

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

import omni
from omni.isaac.core import World
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
from pxr import UsdGeom, Gf, UsdPhysics, UsdShade, PhysxSchema, UsdLux, Sdf
import carb.input

# Try to load rough policy
HAS_ROUGH_POLICY = False
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "4_env_test", "src"))
    from spot_rough_terrain_policy import SpotRoughTerrainPolicy
    HAS_ROUGH_POLICY = True
except ImportError:
    print("[WARNING] SpotRoughTerrainPolicy not available")

# =============================================================================
# CONSTANTS
# =============================================================================

SLOPE_DEG = args.slope_deg
SLOPE_RAD = math.radians(SLOPE_DEG)
ARENA_SIZE = 15.0  # meters
GRAVEL_COUNT = args.gravel_count
GRAVEL_DEPTH = 0.075  # 3 inches in meters

# Gravel size classes (diameter in meters) — smaller for 10K count
GRAVEL_SMALL = (0.008, 0.015)   # 0.8-1.5cm pebbles (60%)
GRAVEL_MEDIUM = (0.015, 0.025)  # 1.5-2.5cm stones (30%)
GRAVEL_LARGE = (0.025, 0.04)    # 2.5-4cm rocks (10%)

# Physics materials
TARP_STATIC_FRICTION = 0.45    # Tarp/shingle surface — moderate grip
TARP_DYNAMIC_FRICTION = 0.35
GRAVEL_STATIC_FRICTION = 0.25  # Gravel-on-gravel — shifts easily
GRAVEL_DYNAMIC_FRICTION = 0.15
GRAVEL_RESTITUTION = 0.1       # Low bounce
GRAVEL_DENSITY = 2500.0        # kg/m³ (granite gravel)

# Robot spawn on flat staging area, facing uphill (+X)
START_POS = np.array([-2.0, ARENA_SIZE / 2.0, 0.55])

# Control
PHYSICS_DT = 1.0 / 500.0
RENDER_DT = 1.0 / 60.0
VX_MAX = 1.0
WZ_MAX = 2.0


# =============================================================================
# ENVIRONMENT BUILDER
# =============================================================================

def create_gravel_slope(stage):
    """Build the 50-degree gravel slope arena."""

    # ── Root xform ────────────────────────────────────────────────────
    UsdGeom.Xform.Define(stage, "/World/GravelSlope")

    # ── 1. Physics scene ──────────────────────────────────────────────
    scene_path = "/World/PhysicsScene"
    if not stage.GetPrimAtPath(scene_path).IsValid():
        scene = UsdPhysics.Scene.Define(stage, scene_path)
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(9.81)

    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(scene_path))
    physx_scene.CreateEnableCCDAttr(True)
    physx_scene.CreateEnableStabilizationAttr(True)
    physx_scene.CreateGpuMaxNumPartitionsAttr(8)

    # ── 2. Physics materials ──────────────────────────────────────────
    mat_root = "/World/Physics/Materials"
    UsdGeom.Xform.Define(stage, "/World/Physics")
    UsdGeom.Xform.Define(stage, mat_root)

    # Tarp/base surface material
    tarp_mat_path = f"{mat_root}/TarpMaterial"
    UsdShade.Material.Define(stage, tarp_mat_path)
    tarp_prim = stage.GetPrimAtPath(tarp_mat_path)
    tarp_phys = UsdPhysics.MaterialAPI.Apply(tarp_prim)
    tarp_phys.CreateStaticFrictionAttr(TARP_STATIC_FRICTION)
    tarp_phys.CreateDynamicFrictionAttr(TARP_DYNAMIC_FRICTION)
    tarp_phys.CreateRestitutionAttr(0.05)
    tarp_physx = PhysxSchema.PhysxMaterialAPI.Apply(tarp_prim)
    tarp_physx.CreateFrictionCombineModeAttr().Set("multiply")

    # Gravel particle material
    gravel_mat_path = f"{mat_root}/GravelMaterial"
    UsdShade.Material.Define(stage, gravel_mat_path)
    gravel_prim = stage.GetPrimAtPath(gravel_mat_path)
    gravel_phys = UsdPhysics.MaterialAPI.Apply(gravel_prim)
    gravel_phys.CreateStaticFrictionAttr(GRAVEL_STATIC_FRICTION)
    gravel_phys.CreateDynamicFrictionAttr(GRAVEL_DYNAMIC_FRICTION)
    gravel_phys.CreateRestitutionAttr(GRAVEL_RESTITUTION)
    gravel_physx = PhysxSchema.PhysxMaterialAPI.Apply(gravel_prim)
    gravel_physx.CreateFrictionCombineModeAttr().Set("multiply")

    # ── 3. Sloped base surface — rough heightfield simulating gravel ─
    # Fine-grid mesh with random height noise (3" gravel texture) on a slope.
    # This gives the unstable footing feel without dynamic particles.
    slope_path = "/World/GravelSlope/SlopeSurface"
    slope_mesh = UsdGeom.Mesh.Define(stage, slope_path)

    cos_a = math.cos(SLOPE_RAD)
    sin_a = math.sin(SLOPE_RAD)

    # 10cm grid for gravel-like bumpiness (150x150 = 22.5K verts)
    cell_size = 0.10
    nx = int(ARENA_SIZE / cell_size) + 1
    ny = int(ARENA_SIZE / cell_size) + 1

    rng_surface = np.random.RandomState(123)

    verts = []
    for iy in range(ny):
        for ix in range(nx):
            u = ix * cell_size  # along slope
            v = iy * cell_size  # across slope

            # Random gravel-like height bump (0 to 7.5cm normal to surface)
            bump = rng_surface.uniform(0, GRAVEL_DEPTH)

            # Position on slope + bump in surface-normal direction
            x = u * cos_a - bump * sin_a
            z = u * sin_a + bump * cos_a
            y = v
            verts.append(Gf.Vec3f(x, y, z))

    # Build triangle faces
    face_counts = []
    face_indices = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            v00 = iy * nx + ix
            v10 = iy * nx + ix + 1
            v01 = (iy + 1) * nx + ix
            v11 = (iy + 1) * nx + ix + 1
            face_counts.extend([3, 3])
            face_indices.extend([v00, v10, v11, v00, v11, v01])

    print(f"  Slope mesh: {nx}x{ny} verts = {len(verts)}, {len(face_counts)} tris")

    slope_mesh.CreatePointsAttr(verts)
    slope_mesh.CreateFaceVertexCountsAttr(face_counts)
    slope_mesh.CreateFaceVertexIndicesAttr(face_indices)

    # Static collider — use triangle mesh for accurate bumpy collision
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(slope_path))
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(stage.GetPrimAtPath(slope_path))
    mesh_collision.CreateApproximationAttr("meshSimplification")

    # Bind gravel friction material (low friction — simulates loose surface)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(slope_path)).Bind(
        UsdShade.Material(stage.GetPrimAtPath(gravel_mat_path)),
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )

    # ── 4. Side walls (prevent robot falling off) ─────────────────────
    wall_height = 2.0
    wall_thickness = 0.1
    for side, y_pos in [("Left", 0.0), ("Right", ARENA_SIZE)]:
        wall_path = f"/World/GravelSlope/Wall{side}"
        wall = UsdGeom.Cube.Define(stage, wall_path)
        # Wall along the slope
        wall.AddTranslateOp().Set(Gf.Vec3d(
            ARENA_SIZE * cos_a / 2.0,
            y_pos,
            ARENA_SIZE * sin_a / 2.0 + wall_height / 2.0,
        ))
        wall.AddScaleOp().Set(Gf.Vec3d(ARENA_SIZE / 2.0, wall_thickness, wall_height / 2.0))
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(wall_path))

    # ── 5. Flat ground at bottom (staging area) — high friction ─────
    # Concrete-like surface so Spot can walk and approach the slope
    concrete_mat_path = f"{mat_root}/ConcreteMaterial"
    UsdShade.Material.Define(stage, concrete_mat_path)
    concrete_prim = stage.GetPrimAtPath(concrete_mat_path)
    concrete_phys = UsdPhysics.MaterialAPI.Apply(concrete_prim)
    concrete_phys.CreateStaticFrictionAttr(0.9)
    concrete_phys.CreateDynamicFrictionAttr(0.8)
    concrete_phys.CreateRestitutionAttr(0.05)

    ground_path = "/World/GravelSlope/Ground"
    ground = UsdGeom.Cube.Define(stage, ground_path)
    ground.AddTranslateOp().Set(Gf.Vec3d(0.0, ARENA_SIZE / 2.0, -0.05))
    ground.AddScaleOp().Set(Gf.Vec3d(5.0, ARENA_SIZE / 2.0, 0.05))
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(ground_path))
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(ground_path)).Bind(
        UsdShade.Material(stage.GetPrimAtPath(concrete_mat_path)),
        materialPurpose="physics",
    )

    # ── 6. Gravel particles ──────────────────────────────────────────
    print(f"Spawning {GRAVEL_COUNT} gravel particles on {SLOPE_DEG}deg slope...")

    gravel_root = "/World/GravelSlope/Gravel"
    UsdGeom.Xform.Define(stage, gravel_root)

    rng = np.random.RandomState(42)

    for i in range(GRAVEL_COUNT):
        # Size class selection
        r = rng.random()
        if r < 0.60:
            diam = rng.uniform(*GRAVEL_SMALL)
        elif r < 0.90:
            diam = rng.uniform(*GRAVEL_MEDIUM)
        else:
            diam = rng.uniform(*GRAVEL_LARGE)

        radius = diam / 2.0

        # Random position on the slope surface
        # u = position along slope (0=bottom, 1=top)
        # v = position across slope (0=left, 1=right)
        u = rng.uniform(0.05, 0.95)
        v = rng.uniform(0.05, 0.95)

        # Map to 3D coordinates on the slope
        x_slope = u * ARENA_SIZE * cos_a
        z_slope = u * ARENA_SIZE * sin_a
        y_slope = v * ARENA_SIZE

        # Offset slightly above surface for gravel depth layering
        layer = rng.uniform(0, GRAVEL_DEPTH)
        x_pos = x_slope - layer * sin_a  # offset normal to slope
        z_pos = z_slope + layer * cos_a + radius
        y_pos = y_slope

        gravel_path = f"{gravel_root}/rock_{i:04d}"
        rock = UsdGeom.Sphere.Define(stage, gravel_path)
        rock.CreateRadiusAttr(radius)
        rock.AddTranslateOp().Set(Gf.Vec3d(x_pos, y_pos, z_pos))

        # Random rotation for visual variety
        rock.AddRotateXYZOp().Set(Gf.Vec3d(
            rng.uniform(0, 360), rng.uniform(0, 360), rng.uniform(0, 360)
        ))

        # Physics — dynamic rigid body
        prim = stage.GetPrimAtPath(gravel_path)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.RigidBodyAPI.Apply(prim)

        # Mass from density
        volume = (4.0 / 3.0) * math.pi * radius ** 3
        mass = GRAVEL_DENSITY * volume

        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(mass)

        # Bind gravel material
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(stage.GetPrimAtPath(gravel_mat_path)),
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )

        # PhysX rigid body settings
        physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physx_rb.CreateSleepThresholdAttr(0.005)
        physx_rb.CreateLinearDampingAttr(0.5)
        physx_rb.CreateAngularDampingAttr(0.3)

    print(f"  Spawned {GRAVEL_COUNT} gravel particles")

    # ── 7. Lighting ──────────────────────────────────────────────────
    light_path = "/World/GravelSlope/DomeLight"
    dome = UsdLux.DomeLight.Define(stage, light_path)
    dome.CreateIntensityAttr(1500)

    dist_path = "/World/GravelSlope/DistantLight"
    dist = UsdLux.DistantLight.Define(stage, dist_path)
    dist.CreateIntensityAttr(3000)
    dist.AddRotateXYZOp().Set(Gf.Vec3d(-45, 30, 0))

    print(f"  Gravel slope arena built: {ARENA_SIZE}m x {ARENA_SIZE}m at {SLOPE_DEG}deg")


# =============================================================================
# MAIN
# =============================================================================

print("\n" + "=" * 60)
print(f"  GRAVEL SLOPE CHALLENGE")
print(f"  Slope: {SLOPE_DEG}deg | Size: {ARENA_SIZE}m x {ARENA_SIZE}m")
print(f"  Gravel: {GRAVEL_COUNT} particles, {GRAVEL_DEPTH*100:.1f}cm deep")
print(f"  Surface: tarp (us={TARP_STATIC_FRICTION}, ud={TARP_DYNAMIC_FRICTION})")
print(f"  Gravel:  (us={GRAVEL_STATIC_FRICTION}, ud={GRAVEL_DYNAMIC_FRICTION})")
print("=" * 60 + "\n")

# Create world
world = World(
    stage_units_in_meters=1.0,
    physics_dt=PHYSICS_DT,
    rendering_dt=RENDER_DT,
)
stage = omni.usd.get_context().get_stage()

# Build arena
create_gravel_slope(stage)

# Spawn Spot
print("Loading Spot...")
spot_flat = SpotFlatTerrainPolicy(
    prim_path="/World/Spot",
    name="Spot",
    position=START_POS,
)

spot_rough = None
if HAS_ROUGH_POLICY and args.checkpoint:
    try:
        spot_rough = SpotRoughTerrainPolicy(
            flat_policy=spot_flat, checkpoint_path=args.checkpoint, mason_baseline=True)
        print(f"[GAIT] Rough policy loaded from {args.checkpoint}")
    except Exception as e:
        print(f"[GAIT] Failed to load rough policy: {e}")

world.reset()
spot_flat.initialize()

# Warm up physics so articulation view is ready
print("Warming up physics...", flush=True)
for _ in range(20):
    spot_flat.forward(PHYSICS_DT, np.array([0.0, 0.0, 0.0]))
    world.step(render=True)

# NOW initialize rough policy (articulation view is ready)
if spot_rough is not None:
    spot_rough.initialize()
    print("[ROUGH] Policy initialized with solver/gains", flush=True)

# Stabilize with the rough policy so it builds internal state
stab_policy = spot_rough if spot_rough is not None else spot_flat
print("Stabilizing (letting gravel settle)...")
for i in range(200):
    stab_policy.forward(PHYSICS_DT, np.array([0.0, 0.0, 0.0]))
    world.step(render=True)
    if (i + 1) % 50 == 0:
        print(f"  Stabilize step {i+1}/200", flush=True)
print("Ready!", flush=True)

print("\n  Controls:")
print("    W/S     Forward/backward")
print("    A/D     Turn left/right")
print("    G       Cycle gait (FLAT/ROUGH)")
print("    T       Toggle auto-walk")
print("    R       Reset to bottom")
print("    ESC     Exit\n")

# Input
input_mgr = carb.input.acquire_input_interface()
keyboard = omni.appwindow.get_default_app_window().get_keyboard()
keys_pressed = set()

def on_key_event(event, *args, **kwargs):
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        keys_pressed.add(event.input)
    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        keys_pressed.discard(event.input)
    return True

input_mgr.subscribe_to_keyboard_events(keyboard, on_key_event)

# State — always use rough policy if available
use_rough = [spot_rough is not None]
auto_walk = [False]
active_policy = spot_rough if spot_rough is not None else spot_flat
if use_rough[0]:
    spot_rough.initialize()
    print("[GAIT] Starting in ROUGH (boulder V6 policy)")
else:
    print("[GAIT] No checkpoint — using FLAT")
step_size = PHYSICS_DT
sim_time = [0.0]

# Main loop
while simulation_app.is_running():
    # Check keys
    if carb.input.KeyboardInput.ESCAPE in keys_pressed:
        break

    if carb.input.KeyboardInput.G in keys_pressed:
        keys_pressed.discard(carb.input.KeyboardInput.G)
        if spot_rough is not None:
            use_rough[0] = not use_rough[0]
            active_policy = spot_rough if use_rough[0] else spot_flat
            print(f"\n  >> Gait: {'ROUGH' if use_rough[0] else 'FLAT'}")
        else:
            print("\n  >> No rough policy loaded")

    if carb.input.KeyboardInput.T in keys_pressed:
        keys_pressed.discard(carb.input.KeyboardInput.T)
        auto_walk[0] = not auto_walk[0]
        print(f"\n  >> Auto-walk: {'ON' if auto_walk[0] else 'OFF'}")

    if carb.input.KeyboardInput.R in keys_pressed:
        keys_pressed.discard(carb.input.KeyboardInput.R)
        spot_flat.robot.set_world_pose(
            position=np.array(START_POS),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        print("\n  >> Reset to bottom of slope")

    # Build velocity command
    vx, wz = 0.0, 0.0
    if auto_walk[0]:
        vx = 0.6
    else:
        if carb.input.KeyboardInput.W in keys_pressed:
            vx = VX_MAX
        if carb.input.KeyboardInput.S in keys_pressed:
            vx = -VX_MAX * 0.5
        if carb.input.KeyboardInput.A in keys_pressed:
            wz = WZ_MAX
        if carb.input.KeyboardInput.D in keys_pressed:
            wz = -WZ_MAX

    cmd = np.array([vx, 0.0, wz])

    # Step policy
    if use_rough[0] and spot_rough is not None:
        spot_rough.forward(step_size, cmd)
    else:
        spot_flat.forward(step_size, cmd)

    world.step(render=not args.headless)
    sim_time[0] += step_size

    # Status every 5 seconds
    if int(sim_time[0] * 10) % 50 == 0 and sim_time[0] > 0.1:
        pos, _ = spot_flat.robot.get_world_pose()
        pos = np.array(pos)
        # Distance up the slope
        slope_dist = pos[0] * math.cos(SLOPE_RAD) + pos[2] * math.sin(SLOPE_RAD)
        height = pos[2]
        gait = "ROUGH" if use_rough[0] else "FLAT"
        print(f"\r  [{sim_time[0]:6.1f}s] {gait:>5s} | "
              f"slope={slope_dist:.1f}m  height={height:.1f}m  "
              f"pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})", end="", flush=True)

print("\n\nExiting...")
os._exit(0)
