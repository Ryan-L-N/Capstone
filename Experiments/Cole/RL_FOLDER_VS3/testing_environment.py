"""
Simple Testing Environment for Spot RL Policies
================================================

A minimal environment for testing trained locomotion policies.
Spot navigates a circular arena with obstacles, no waypoint targets required.

Author: Cole (MS for Autonomy Project)
Date: March 2026
"""

import math
import argparse
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing BEFORE SimulationApp
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Simple Testing Environment for Spot RL")
parser.add_argument("--headless",  action="store_true", help="Run without GUI")
parser.add_argument("--episodes",  type=int, default=1,  help="Number of episodes")
parser.add_argument("--seed",      type=int, default=None, help="Random seed")
parser.add_argument("--model",     type=str, default=None, help="Path to trained model checkpoint (.pt)")
parser.add_argument("--use-model", action="store_true", help="Use model for control")
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
from pxr import UsdGeom, Gf, UsdPhysics, UsdLux
import torch

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Arena
ARENA_RADIUS         = 25.0
ARENA_CENTER_X       = 0.0
ARENA_CENTER_Y       = 0.0

# Spot
SPOT_START_X         = 0.0
SPOT_START_Y         = 0.0
SPOT_START_Z         = 0.7
SPOT_MASS_KG         = 32.7

# Environmental
FALL_HEIGHT_THRESHOLD = 0.25
EPISODE_DURATION = 600.0  # seconds
PHYSICS_DT = 1.0 / 500.0
RENDERING_DT = 10.0 / 500.0

# Obstacles
OBSTACLE_AREA_FRAC = 0.20
OBSTACLE_MIN_MASS = 0.227
OBSTACLE_MAX_MASS = 65.4
OBSTACLE_LIGHT_MAX = 0.45
OBSTACLE_MEDIUM_MAX = SPOT_MASS_KG

# Small obstacles
SMALL_OBSTACLE_MIN_SIZE = 0.043
SMALL_OBSTACLE_MAX_SIZE = 0.102
SMALL_OBSTACLE_COVERAGE = 0.10

# Physics
ARENA_AREA = math.pi * ARENA_RADIUS ** 2

# Colors
COLOR_LIGHT_OBSTACLE = Gf.Vec3f(1.0,  0.55, 0.0)
COLOR_HEAVY_OBSTACLE = Gf.Vec3f(0.27, 0.51, 0.71)
COLOR_SMALL_OBSTACLE = Gf.Vec3f(0.4, 0.4, 0.4)

# Model control settings
USE_MODEL_CONTROL = args.use_model
MODEL_PATH = args.model

print("=" * 72)
print("SIMPLE TESTING ENVIRONMENT FOR SPOT RL")
print(f"  Arena radius : {ARENA_RADIUS} m")
print(f"  Episode duration : {EPISODE_DURATION} s")
print("=" * 72)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def inside_arena(x: float, y: float, radius: float = ARENA_RADIUS, margin: float = 0.0) -> bool:
    """Check if point is inside arena."""
    return (x - ARENA_CENTER_X) ** 2 + (y - ARENA_CENTER_Y) ** 2 < (radius - margin) ** 2


def random_inside_arena(margin: float = 0.0, rng: np.random.Generator = None) -> np.ndarray:
    """Sample random position inside arena."""
    if rng is None:
        rng = np.random.default_rng()
    r_limit = ARENA_RADIUS - margin
    while True:
        x = rng.uniform(-r_limit, r_limit)
        y = rng.uniform(-r_limit, r_limit)
        if x ** 2 + y ** 2 < r_limit ** 2:
            return np.array([x, y])


def distance_2d(a, b) -> float:
    """2D Euclidean distance."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def quaternion_to_yaw(quat: np.ndarray) -> float:
    """Convert quaternion to yaw angle."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def apply_rigid_body_physics(stage, prim_path: str, mass_kg: float, friction: float = 0.5) -> None:
    """Apply rigid body physics to a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return

    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)

    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass_kg)

    physics_mat = UsdPhysics.MaterialAPI.Apply(prim)
    physics_mat.CreateStaticFrictionAttr(friction)
    physics_mat.CreateDynamicFrictionAttr(friction * 0.8)
    physics_mat.CreateRestitutionAttr(0.05)


# ─────────────────────────────────────────────────────────────────────────────
# OBSTACLES — simple mesh creation
# ─────────────────────────────────────────────────────────────────────────────

def create_sphere_mesh(stage, path: str, radius: float, color, segments: int = 16) -> None:
    """Create sphere mesh."""
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


def create_box_mesh(stage, path: str, w: float, d: float, h: float, color) -> None:
    """Create box mesh."""
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


# ─────────────────────────────────────────────────────────────────────────────
# OBSTACLE MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class ObstacleManager:
    """Manages obstacle spawning and tracking."""

    SHAPES = ["sphere", "box"]

    def __init__(self, stage, rng: np.random.Generator):
        self.stage = stage
        self.rng = rng
        self.obstacles = []

    def spawn_one(self, idx: int) -> None:
        """Spawn one random obstacle."""
        shape = self.rng.choice(self.SHAPES)
        
        # Random mass (light, medium, or heavy)
        mass_choice = self.rng.choice(["light", "medium", "heavy"])
        if mass_choice == "light":
            mass = self.rng.uniform(OBSTACLE_MIN_MASS, OBSTACLE_LIGHT_MAX)
            color = COLOR_LIGHT_OBSTACLE
        elif mass_choice == "medium":
            mass = self.rng.uniform(OBSTACLE_LIGHT_MAX, OBSTACLE_MEDIUM_MAX)
            color = Gf.Vec3f(0.8, 0.6, 0.2)  # bronze
        else:
            mass = self.rng.uniform(OBSTACLE_MEDIUM_MAX, OBSTACLE_MAX_MASS)
            color = COLOR_HEAVY_OBSTACLE

        pos_2d = random_inside_arena(margin=2.0, rng=self.rng)
        path = f"/World/Obstacles/Obs_{idx:03d}"

        # Create geometry
        if shape == "sphere":
            size = self.rng.uniform(SMALL_OBSTACLE_MIN_SIZE, SMALL_OBSTACLE_MAX_SIZE)
            create_sphere_mesh(self.stage, path, size, color)
            dims = (size * 2, size * 2)
        else:  # box
            w = self.rng.uniform(0.3, 1.0)
            d = self.rng.uniform(0.3, 1.0)
            h = self.rng.uniform(0.3, 1.0)
            create_box_mesh(self.stage, path, w, d, h, color)
            dims = (w, d)

        # Position and rotate
        prim = self.stage.GetPrimAtPath(path)
        if prim.IsValid():
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(pos_2d[0], pos_2d[1], dims[1] / 2 + 0.1))
            xform.AddRotateXYZOp().Set(Gf.Vec3d(0, 0, self.rng.uniform(0, 360)))

        # Apply physics
        apply_rigid_body_physics(self.stage, path, mass, friction=0.5)
        
        # Static if heavy
        if mass_choice == "heavy":
            rigid = UsdPhysics.RigidBodyAPI.Get(self.stage, path)
            if rigid:
                rigid.CreateRigidBodyEnabledAttr(False)

        self.obstacles.append({
            "path": path,
            "pos": pos_2d,
            "dims": dims,
            "mass": mass,
            "weight_class": mass_choice,
        })

    def populate(self, count: int = 15) -> None:
        """Spawn obstacles."""
        print(f"[INFO] Spawning {count} obstacles")
        for i in range(count):
            self.spawn_one(i)
        print(f"[OK] {len(self.obstacles)} obstacles spawned")

    def remove_prims(self) -> None:
        """Remove all obstacle prims."""
        for obs in self.obstacles:
            prim = self.stage.GetPrimAtPath(obs["path"])
            if prim.IsValid():
                self.stage.RemovePrim(obs["path"])
        self.obstacles.clear()


# ─────────────────────────────────────────────────────────────────────────────
# BUILD WORLD
# ─────────────────────────────────────────────────────────────────────────────

def build_world(world, stage) -> None:
    """Build arena scene."""
    # Ground plane
    world.scene.add_default_ground_plane(z_position=0.0, name="ground_plane")
    print("[OK] Ground plane created")

    # Lighting
    light = UsdLux.DistantLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(1000.0)
    light_xform = UsdGeom.Xformable(light)
    light_xform.AddRotateXYZOp().Set(Gf.Vec3d(45, 45, 0))
    print("[OK] Lighting created")

    # Create obstacle container
    obs_scope = UsdGeom.Scope.Define(stage, "/World/Obstacles")
    print("[OK] Obstacle container created")


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT CLASS
# ─────────────────────────────────────────────────────────────────────────────

class TestingEnv:
    """Simple testing environment."""

    def __init__(self, world, stage, rng: np.random.Generator, policy=None):
        self.world = world
        self.stage = stage
        self.rng = rng
        self.policy = policy

        self.spot = None
        self.obstacle_mgr = ObstacleManager(stage, rng)

        self.episode_start_time = 0.0
        self.episode_num = 0
        self.physics_ready = False
        self.last_status_print_time = 0.0
        self.episode_done = False

    def reset(self, episode: int) -> None:
        """Reset environment."""
        print(f"\n{'-' * 72}")
        print(f"[RESET] Episode {episode}")
        print(f"{'-' * 72}")

        self.episode_num = episode
        self.episode_done = False
        self.episode_start_time = 0.0
        self.physics_ready = False
        self.last_status_print_time = 0.0

        # Remove old obstacles
        self.obstacle_mgr.remove_prims()

        # Populate new obstacles
        self.obstacle_mgr.populate(count=15)

        # Reset Spot
        if self.spot is not None:
            self.spot.robot.set_world_pose(
                position=np.array([SPOT_START_X, SPOT_START_Y, SPOT_START_Z]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0])
            )
            self.spot.robot.set_joints_default_state(self.spot.default_pos)
            print(f"[OK] Spot reset to ({SPOT_START_X}, {SPOT_START_Y}, {SPOT_START_Z})")

        print(f"[OK] Episode {episode} initialized")

    def step(self, step_size: float) -> bool:
        """Execute one physics step. Return False to terminate."""
        if not self.physics_ready:
            self.episode_start_time = self.world.current_time
            self.physics_ready = True
            return True

        # Get Spot state
        spot_pos, spot_quat = self.spot.robot.get_world_pose()
        spot_x, spot_y, spot_z = spot_pos[0], spot_pos[1], spot_pos[2]

        # Check fall condition
        if spot_z < FALL_HEIGHT_THRESHOLD:
            print(f"[FALL] Spot fell (z={spot_z:.3f})")
            self.episode_done = True
            return False

        # Get velocity
        spot_vel_lin = self.spot.robot.get_linear_velocity()
        vel_x, vel_y, vel_z = spot_vel_lin[0], spot_vel_lin[1], spot_vel_lin[2]
        heading = quaternion_to_yaw(spot_quat)

        elapsed = self.world.current_time - self.episode_start_time

        # Status every second
        if elapsed - self.last_status_print_time >= 1.0:
            speed = math.sqrt(vel_x ** 2 + vel_y ** 2)
            heading_deg = math.degrees(heading)
            print(f"[STATUS] Elapsed: {elapsed:.1f}s | Speed: {speed:.2f} m/s | Heading: {heading_deg:.1f}°")
            self.last_status_print_time = elapsed

        # Simple locomotion control
        if self.policy is not None:
            # Model-based control
            observation = np.array([
                vel_x, vel_y, vel_z,
                heading, 0.0,
                0.0,
                np.sin(heading), np.cos(heading),
                0.0, 1.0
            ], dtype=np.float32)

            with torch.no_grad():
                obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)
                action = self.policy(obs_tensor).cpu().numpy()[0]

            forward_speed = np.clip(action[0], -1.0, 1.0) * 1.0
            strafe_speed = np.clip(action[1], -1.0, 1.0) * 0.5
            angular_velocity = np.clip(action[2], -1.0, 1.0) * 1.0
        else:
            # Manual control - just walk forward
            forward_speed = 0.3
            strafe_speed = 0.0
            angular_velocity = 0.0

        # Apply locomotion command via forward()
        command = np.array([forward_speed, strafe_speed, angular_velocity])
        self.spot.forward(step_size, command)

        # Check episode duration
        if elapsed >= EPISODE_DURATION:
            print(f"[TIMEOUT] Episode duration reached ({elapsed:.1f}s)")
            self.episode_done = True
            return False

        return True

    def close(self) -> None:
        """Clean up."""
        print(f"[OK] Testing complete")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Main loop."""
    print("\n" + "=" * 72)
    print("  SIMPLE TESTING ENVIRONMENT FOR SPOT RL")
    print("=" * 72)

    # Create world
    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)
    stage = omni.usd.get_context().get_stage()

    # Build environment
    build_world(world, stage)

    # Initialize RNG
    rng = np.random.default_rng(args.seed)
    print(f"[INFO] Random seed: {args.seed}")

    # Load policy if provided
    policy = None
    if USE_MODEL_CONTROL and MODEL_PATH:
        print(f"\n[MODEL] Loading policy from: {MODEL_PATH}")
        try:
            checkpoint = torch.load(MODEL_PATH, map_location='cpu')

            class SimplePolicy(torch.nn.Module):
                def __init__(self, obs_dim=10, action_dim=3, hidden_dim=128):
                    super().__init__()
                    self.policy_net = torch.nn.Sequential(
                        torch.nn.Linear(obs_dim, hidden_dim),
                        torch.nn.ReLU(),
                        torch.nn.Linear(hidden_dim, hidden_dim),
                        torch.nn.ReLU(),
                        torch.nn.Linear(hidden_dim, action_dim),
                        torch.nn.Tanh()
                    )

                def forward(self, obs):
                    return self.policy_net(obs)

            policy = SimplePolicy()

            if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
                policy.load_state_dict(checkpoint['policy_state_dict'])
            else:
                policy.load_state_dict(checkpoint)

            policy.eval()
            print(f"[OK] Policy loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load policy: {e}")
            print("[WARN] Falling back to manual control")
            policy = None

    # Create Spot
    spot = SpotFlatTerrainPolicy(
        prim_path="/World/Spot",
        name="Spot",
        position=np.array([SPOT_START_X, SPOT_START_Y, SPOT_START_Z])
    )
    print(f"[OK] Spot created")

    # Reset and initialize
    world.reset()
    print("[OK] World reset")

    spot.initialize()
    spot.robot.set_joints_default_state(spot.default_pos)
    print("[OK] Spot initialized")

    # Create environment
    env = TestingEnv(world, stage, rng, policy=policy)
    env.spot = spot

    # Print control mode
    if USE_MODEL_CONTROL and policy is not None:
        print("[MODE] Using MODEL-BASED CONTROL")
    else:
        print("[MODE] Using MANUAL CONTROL")

    # Physics callback
    def on_physics_step(step_size: float):
        env.step(step_size)

    world.add_physics_callback("spot_control", on_physics_step)
    print("[OK] Physics callback registered")

    # Episode loop
    print(f"\n{'=' * 72}")
    print(f"  RUNNING {args.episodes} EPISODES")
    print(f"{'=' * 72}\n")

    for episode in range(1, args.episodes + 1):
        env.reset(episode)

        while simulation_app.is_running():
            world.step(render=not args.headless)
            
            # Check if episode terminated
            if env.episode_done:
                break

        if not simulation_app.is_running():
            print("\n[EXIT] Simulation closed by user")
            break

    # Cleanup
    env.close()
    simulation_app.close()
    print("\n" + "=" * 72)
    print("  COMPLETE")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
