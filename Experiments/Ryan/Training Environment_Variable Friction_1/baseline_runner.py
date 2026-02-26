"""
Training Environment 1 — Baseline Runner
=========================================
Runs the stock SpotFlatTerrainPolicy with a fixed command [vx=1.5, vy=0, ω=0]
through all 7 friction surfaces. No learning occurs.

Outputs:
  logs/baseline_summary.csv  — one row per robot per episode
  logs/baseline_raw.npz      — per-step position/velocity arrays (optional)

Usage (Python 3.11 via IsaacSim venv):
  ./isaacSim_env/Scripts/python.exe baseline_runner.py
  ./isaacSim_env/Scripts/python.exe baseline_runner.py --headless
  ./isaacSim_env/Scripts/python.exe baseline_runner.py --headless --episodes_per_surface 5
"""

import numpy as np
import argparse
import os

# --- Parse args BEFORE SimulationApp ---
parser = argparse.ArgumentParser(description="Env1 Baseline Runner")
parser.add_argument("--headless",              action="store_true",   help="Run without GUI")
parser.add_argument("--episodes_per_surface",  type=int, default=20,  help="Episodes per friction surface")
args = parser.parse_args()

# --- Isaac Sim MUST be initialized before any other Isaac imports ---
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import omni
from omni.isaac.core import World
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
from pxr import UsdGeom, UsdLux, Gf, UsdPhysics

# --- Our modules (imported after SimulationApp is live) ---
from env_config import config, FRICTION_CONFIG, CURRICULUM_ORDER
from robot_state import RobotState
from metrics import MetricsLogger

os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

FIXED_CMD = np.array([1.5, 0.0, 0.0])   # Fixed command: vx=1.5, vy=0, yaw=0

print("=" * 70)
print("TRAINING ENV 1 — BASELINE MEASUREMENT (NO TRAINING)")
print(f"  Surfaces:            {CURRICULUM_ORDER}")
print(f"  Episodes/surface:    {args.episodes_per_surface}")
print(f"  Parallel robots:     {config.NUM_ROBOTS}")
print(f"  Fixed command:       vx={FIXED_CMD[0]}, vy={FIXED_CMD[1]}, yaw={FIXED_CMD[2]}")
total_runs = args.episodes_per_surface * len(CURRICULUM_ORDER) * config.NUM_ROBOTS
print(f"  Total robot-runs:    {total_runs}")
print("=" * 70)

# ===========================================================================
# World setup
# ===========================================================================

world = World(
    physics_dt=config.PHYSICS_DT,
    rendering_dt=config.RENDERING_DT,
    stage_units_in_meters=1.0,
)
stage = omni.usd.get_context().get_stage()

# Lighting
dome_light = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
dome_light.CreateIntensityAttr(1500.0)
dome_light.CreateColorAttr(Gf.Vec3f(0.95, 0.95, 1.0))

# Ground plane (initial friction = asphalt_dry; updated per surface)
world.scene.add_default_ground_plane(
    z_position=0,
    name="default_ground_plane",
    prim_path="/World/defaultGroundPlane",
    static_friction=0.75,
    dynamic_friction=0.65,
    restitution=0.01,
)
print("Ground plane added")

# Walls
L, W = config.ARENA_LENGTH, config.ARENA_WIDTH
T, H = config.WALL_THICKNESS, config.WALL_HEIGHT
WALL_COLOR = Gf.Vec3f(0.5, 0.5, 0.5)

for name, pos, scale in [
    ("North", (0,  W/2 + T/2, H/2), (L + 2*T, T, H)),
    ("South", (0, -W/2 - T/2, H/2), (L + 2*T, T, H)),
    ("East",  ( L/2 + T/2, 0, H/2), (T, W, H)),
    ("West",  (-L/2 - T/2, 0, H/2), (T, W, H)),
]:
    wall = UsdGeom.Cube.Define(stage, f"/World/Walls/{name}")
    xf   = UsdGeom.Xformable(wall.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddScaleOp().Set(Gf.Vec3f(*scale))
    wall.GetDisplayColorAttr().Set([WALL_COLOR])
    UsdPhysics.CollisionAPI.Apply(wall.GetPrim())

print(f"Walls created: {L}m × {W}m arena")


def set_ground_friction(static_mu: float, dynamic_mu: float) -> bool:
    """
    Update ground plane friction at runtime by scanning for friction attributes.

    Tries three strategies in order:
      1. Direct attribute write on prims that already have physics:staticFriction
      2. UsdPhysics.MaterialAPI scan
      3. PhysX schema attribute names (physxMaterial:staticFriction)

    If this prints a warning, run find_friction_prim.py to discover the correct
    prim path for this IsaacSim version, then hard-code it here.
    """
    # Strategy 1: scan for USD physics material attributes (most common)
    for prim in stage.Traverse():
        attr_static  = prim.GetAttribute("physics:staticFriction")
        attr_dynamic = prim.GetAttribute("physics:dynamicFriction")
        if attr_static.IsValid() and attr_dynamic.IsValid():
            attr_static.Set(float(static_mu))
            attr_dynamic.Set(float(dynamic_mu))
            print(f"[Friction] Strategy 1 OK  path={prim.GetPath()}  "
                  f"readback us={attr_static.Get():.4f}  uk={attr_dynamic.Get():.4f}")
            return True

    # Strategy 2: UsdPhysics.MaterialAPI applied to a prim
    for prim in stage.Traverse():
        try:
            if prim.HasAPI(UsdPhysics.MaterialAPI):
                mat = UsdPhysics.MaterialAPI(prim)
                sf  = mat.GetStaticFrictionAttr()
                df  = mat.GetDynamicFrictionAttr()
                if sf.IsValid() and df.IsValid():
                    sf.Set(float(static_mu))
                    df.Set(float(dynamic_mu))
                    print(f"[Friction] Strategy 2 OK  path={prim.GetPath()}  "
                          f"readback us={sf.Get():.4f}  uk={df.Get():.4f}")
                    return True
        except Exception:
            pass

    # Strategy 3: PhysX-specific schema attribute names
    for prim in stage.Traverse():
        attr_static  = prim.GetAttribute("physxMaterial:staticFriction")
        attr_dynamic = prim.GetAttribute("physxMaterial:dynamicFriction")
        if attr_static.IsValid() and attr_dynamic.IsValid():
            attr_static.Set(float(static_mu))
            attr_dynamic.Set(float(dynamic_mu))
            print(f"[Friction] Strategy 3 OK  path={prim.GetPath()}  "
                  f"readback us={attr_static.Get():.4f}  uk={attr_dynamic.Get():.4f}")
            return True

    print(
        "[WARNING] No friction prim found — friction unchanged. "
        "Run find_friction_prim.py to discover the correct prim path."
    )
    return False


# ===========================================================================
# Robots
# ===========================================================================

spots = []
for i in range(config.NUM_ROBOTS):
    y = config.START_Y_POSITIONS[i]
    spot = SpotFlatTerrainPolicy(
        prim_path=f"/World/Spot_{i}",
        name=f"Spot_{i}",
        position=np.array([config.START_X, y, config.START_Z]),
    )
    spots.append(spot)
    print(f"  Created Spot_{i} at ({config.START_X}, {y:.0f}, {config.START_Z})")

world.reset()
for spot in spots:
    spot.initialize()
    spot.robot.set_joints_default_state(spot.default_pos)

# Physics warm-up
for _ in range(20):
    world.step(render=False)
print(f"All {config.NUM_ROBOTS} robots initialized and physics warmed up")

# ===========================================================================
# Simulation state (mutable containers for callback access)
# ===========================================================================

sim_time       = [0.0]
physics_step_n = [0]
physics_ready  = [False]
episode_active = [True]

robot_states = [
    RobotState(i, spots[i], config.START_Y_POSITIONS[i], friction=0.75)
    for i in range(config.NUM_ROBOTS)
]

logger = MetricsLogger(
    log_path=os.path.join("logs", "baseline_summary.csv"),
    run_type="baseline",
)


def on_physics_step(step_size: float):
    """500 Hz callback: issue fixed forward command to all robots."""
    if not physics_ready[0]:
        physics_ready[0] = True
        return

    sim_time[0] += step_size
    physics_step_n[0] += 1

    if not episode_active[0]:
        return

    # Stabilization: send zero command until physics settles
    if sim_time[0] < config.STABILIZE_TIME:
        for spot in spots:
            spot.forward(step_size, np.zeros(3))
        return

    # Issue fixed command every physics step (500 Hz)
    # Record step data every CONTROL_DECIMATION steps (50 Hz)
    control_step = (physics_step_n[0] % config.CONTROL_DECIMATION == 0)
    dt_control = step_size * config.CONTROL_DECIMATION

    for spot, rs in zip(spots, robot_states):
        spot.forward(step_size, FIXED_CMD)
        if control_step:
            rs.record_step(
                cmd_vx=FIXED_CMD[0],
                cmd_vy=FIXED_CMD[1],
                cmd_yaw=FIXED_CMD[2],
                dt=dt_control,
            )


world.add_physics_callback("baseline_control", on_physics_step)
print("Physics callback registered\n")

# ===========================================================================
# Episode / surface loop
# ===========================================================================

episode_global = 0
all_summaries  = []

for surface_name in CURRICULUM_ORDER:
    friction    = FRICTION_CONFIG[surface_name]
    static_mu   = float(friction["static"])
    dynamic_mu  = float(friction["dynamic"])

    print(f"\n{'='*60}")
    print(f"Surface: {surface_name}  (μₛ={static_mu}, μₖ={dynamic_mu})")
    print(f"{'='*60}")

    set_ground_friction(static_mu, dynamic_mu)
    surface_summaries = []

    for ep in range(args.episodes_per_surface):
        episode_global += 1

        # Reset robot positions and episode state
        for rs in robot_states:
            rs.reset(friction=static_mu)

        sim_time[0]       = 0.0
        physics_step_n[0] = 0
        physics_ready[0]  = False
        episode_active[0] = True

        # Run episode
        while simulation_app.is_running():
            world.step(render=not args.headless)

            # Lap reset: teleport any robot that reaches the far end
            for rs in robot_states:
                if not rs.fell and not rs.out_of_bounds:
                    pos, _ = rs.spot.robot.get_world_pose()
                    if float(pos[0]) >= config.FAR_END_X:
                        rs.spot.robot.set_world_pose(
                            position=rs.start_pos,
                            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                        )
                        rs.spot.robot.set_linear_velocity(np.zeros(3))
                        rs.spot.robot.set_angular_velocity(np.zeros(3))

            # End episode when all robots are done
            if all(rs.is_done(sim_time[0])[0] for rs in robot_states):
                break

        episode_active[0] = False

        # Log episode results
        for rs in robot_states:
            summary = rs.get_episode_summary(surface=surface_name, sim_time=sim_time[0])
            logger.log_episode(summary, episode_global=episode_global)
            surface_summaries.append(summary)
            all_summaries.append(summary)

        # Console update every 5 episodes
        if (ep + 1) % 5 == 0:
            batch = surface_summaries[-(5 * config.NUM_ROBOTS):]
            logger.print_summary(batch, label=f"{surface_name} ep{ep+1:3d}")

    logger.print_summary(surface_summaries, label=f"{surface_name} FINAL")

# ===========================================================================
# Summary across all surfaces
# ===========================================================================

print("\n" + "=" * 70)
print("BASELINE COMPLETE — Per-Surface Summary")
print("=" * 70)

for surface_name in CURRICULUM_ORDER:
    surface_data = [s for s in all_summaries if s["surface"] == surface_name]
    logger.print_summary(surface_data, label=surface_name)

print(f"\nResults written to: logs/baseline_summary.csv")
print(f"Total robot-episodes logged: {len(all_summaries)}")
print("=" * 70)

logger.close()
simulation_app.close()
print("Done.")
