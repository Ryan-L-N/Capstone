"""
Training Environment 1 — RL Training Script
=============================================
Trains a Locomotion Command Policy on top of SpotFlatTerrainPolicy
using the Cross-Entropy Method (CEM). Runs a 4-phase friction curriculum.

Phase 1: asphalt_dry  (μ=0.75) — warm-up on easy surface
Phase 2: grass_dry    (μ=0.40) — transfer to moderate surface
Phase 3: snow         (μ=0.15) — transfer to low-friction surface
Phase 4: random       (μ=0.05–0.80) — generalization

Each phase loads the best checkpoint from the previous phase before continuing.

Usage (Python 3.11 via IsaacSim venv):
  ./isaacSim_env/Scripts/python.exe training_env_1.py --phase 1
  ./isaacSim_env/Scripts/python.exe training_env_1.py --phase 2 --headless
  ./isaacSim_env/Scripts/python.exe training_env_1.py --phase 1 --load checkpoints/phase1_best
"""

import numpy as np
import argparse
import os

# --- Parse args BEFORE SimulationApp ---
parser = argparse.ArgumentParser(description="Env1 RL Training")
parser.add_argument("--headless",              action="store_true")
parser.add_argument("--phase",                 type=int, default=1,
                    help="Training phase 1-4")
parser.add_argument("--load",                  type=str, default=None,
                    help="Checkpoint path to load (e.g. checkpoints/phase1_best)")
parser.add_argument("--generations",           type=int, default=50,
                    help="CEM generations to run")
parser.add_argument("--population",            type=int, default=20,
                    help="CEM population size")
parser.add_argument("--episodes_per_candidate", type=int, default=3,
                    help="Episodes per policy candidate for reward estimation")
args = parser.parse_args()

# --- Isaac Sim MUST be initialized first ---
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import omni
from omni.isaac.core import World
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
from pxr import UsdGeom, UsdLux, Gf, UsdPhysics

# --- Our modules ---
from env_config import config, FRICTION_CONFIG, CURRICULUM_ORDER
from robot_state import RobotState
from loco_policy import LocoPolicy, CEMTrainer
from metrics import MetricsLogger

os.makedirs("logs",        exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Phase configuration: surface(s) used for each training phase
PHASE_SURFACES = {
    1: ["asphalt_dry"],
    2: ["grass_dry"],
    3: ["snow"],
    4: "random",          # Sample friction uniformly each episode
}
PHASE_CHECKPOINT = {
    1: os.path.join("checkpoints", "phase1_best"),
    2: os.path.join("checkpoints", "phase2_best"),
    3: os.path.join("checkpoints", "phase3_best"),
    4: os.path.join("checkpoints", "phase4_best"),
}

print("=" * 70)
print(f"TRAINING ENV 1 — PHASE {args.phase}")
print(f"  Surface(s):          {PHASE_SURFACES[args.phase]}")
print(f"  CEM generations:     {args.generations}")
print(f"  Population size:     {args.population}")
print(f"  Episodes/candidate:  {args.episodes_per_candidate}")
print(f"  Parallel robots:     {config.NUM_ROBOTS}")
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

dome_light = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
dome_light.CreateIntensityAttr(1500.0)
dome_light.CreateColorAttr(Gf.Vec3f(0.95, 0.95, 1.0))

world.scene.add_default_ground_plane(
    z_position=0,
    name="default_ground_plane",
    prim_path="/World/defaultGroundPlane",
    static_friction=0.75,
    dynamic_friction=0.65,
    restitution=0.01,
)

L, W = config.ARENA_LENGTH, config.ARENA_WIDTH
T, H = config.WALL_THICKNESS, config.WALL_HEIGHT
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
    wall.GetDisplayColorAttr().Set([Gf.Vec3f(0.5, 0.5, 0.5)])
    UsdPhysics.CollisionAPI.Apply(wall.GetPrim())

print("World, ground plane, and walls created")


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

for _ in range(20):
    world.step(render=False)
print(f"All {config.NUM_ROBOTS} robots initialized")

# ===========================================================================
# Policy and CEM trainer
# ===========================================================================

policy = LocoPolicy(cfg=config)

if args.load:
    policy.load(args.load)
elif args.phase > 1:
    prev_ckpt = PHASE_CHECKPOINT[args.phase - 1]
    print(f"[Phase {args.phase}] Loading Phase {args.phase-1} checkpoint: {prev_ckpt}")
    policy.load(prev_ckpt)

cem = CEMTrainer(
    policy,
    population_size=args.population,
    elite_frac=0.2,
    noise_std=0.05,
    noise_decay=0.999,
    min_noise=0.01,
    seed=42,
)

print(f"Policy parameter dimension: {policy.param_dim}")

logger = MetricsLogger(
    log_path=os.path.join("logs", f"training_phase{args.phase}.csv"),
    run_type=f"phase{args.phase}",
)

# ===========================================================================
# Simulation state (mutable containers for callback access)
# ===========================================================================

sim_time        = [0.0]
physics_step_n  = [0]
physics_ready   = [False]
episode_active  = [True]
current_cmd     = [np.zeros((config.NUM_ROBOTS, 3))]  # Shape: [NUM_ROBOTS, 3]

robot_states = [
    RobotState(i, spots[i], config.START_Y_POSITIONS[i], friction=0.75)
    for i in range(config.NUM_ROBOTS)
]


def on_physics_step(step_size: float):
    """
    500 Hz callback: apply current commands to all robots.
    Policy is queried at 50 Hz (every CONTROL_DECIMATION steps).
    """
    if not physics_ready[0]:
        physics_ready[0] = True
        return

    sim_time[0]       += step_size
    physics_step_n[0] += 1

    if not episode_active[0]:
        return

    # Stabilization: zero command until physics settles
    if sim_time[0] < config.STABILIZE_TIME:
        for spot in spots:
            spot.forward(step_size, np.zeros(3))
        return

    control_step = (physics_step_n[0] % config.CONTROL_DECIMATION == 0)
    dt_control   = step_size * config.CONTROL_DECIMATION

    for i, (spot, rs) in enumerate(zip(spots, robot_states)):
        cmd = current_cmd[0][i]
        spot.forward(step_size, cmd)

        if control_step:
            # Query policy for new command using current observation
            obs     = rs.get_observation()
            new_cmd = policy.forward(obs)
            current_cmd[0][i] = new_cmd

            rs.record_step(
                cmd_vx=float(new_cmd[0]),
                cmd_vy=float(new_cmd[1]),
                cmd_yaw=float(new_cmd[2]),
                dt=dt_control,
            )


world.add_physics_callback("training_control", on_physics_step)
print("Physics callback registered\n")


def run_episode(surface_name: str, static_mu: float, dynamic_mu: float) -> list:
    """
    Run one full episode for all robots with the current policy.

    Returns
    -------
    list of episode summary dicts, one per robot
    """
    set_ground_friction(static_mu, dynamic_mu)

    for rs in robot_states:
        rs.reset(friction=static_mu)

    sim_time[0]       = 0.0
    physics_step_n[0] = 0
    physics_ready[0]  = False
    episode_active[0] = True
    current_cmd[0]    = np.zeros((config.NUM_ROBOTS, 3))

    while simulation_app.is_running():
        world.step(render=not args.headless)

        # Lap reset: teleport robots that reach the far end
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

        if all(rs.is_done(sim_time[0])[0] for rs in robot_states):
            break

    episode_active[0] = False
    return [rs.get_episode_summary(surface_name, sim_time[0]) for rs in robot_states]


# ===========================================================================
# CEM training loop
# ===========================================================================

rng = np.random.default_rng(42)
best_reward_ever = float("-inf")
episode_global   = 0
phase_surfaces   = PHASE_SURFACES[args.phase]

print(f"Starting Phase {args.phase} training ({args.generations} generations)...\n")

for gen in range(args.generations):
    logger.set_generation(gen + 1)
    population = cem.sample_population()

    # Determine surface and friction for this generation
    if phase_surfaces == "random":
        surface_name = rng.choice(CURRICULUM_ORDER)
        static_mu    = float(rng.uniform(0.05, 0.80))
        dynamic_mu   = static_mu * 0.85
    else:
        surface_name = phase_surfaces[gen % len(phase_surfaces)]
        static_mu    = float(FRICTION_CONFIG[surface_name]["static"])
        dynamic_mu   = float(FRICTION_CONFIG[surface_name]["dynamic"])

    pop_rewards = []

    for candidate_params in population:
        policy.set_params(candidate_params)

        candidate_summaries = []
        for _ in range(args.episodes_per_candidate):
            episode_global += 1
            summaries = run_episode(surface_name, static_mu, dynamic_mu)
            candidate_summaries.extend(summaries)
            for s in summaries:
                logger.log_episode(s, episode_global=episode_global)

        # Average reward across all episodes and robots for this candidate
        candidate_reward = float(np.mean([s["total_reward"] for s in candidate_summaries]))
        pop_rewards.append(candidate_reward)

    best_this_gen, mean_this_gen = cem.update(pop_rewards)

    # Save checkpoint if improved
    if best_this_gen > best_reward_ever:
        best_reward_ever = best_this_gen
        ckpt = PHASE_CHECKPOINT[args.phase]
        policy.save(ckpt)
        print(f"  -> New best! reward={best_reward_ever:.2f}  saved to {ckpt}.npz")

    # Print progress every 10 generations
    if (gen + 1) % 10 == 0:
        print(
            f"Gen {gen+1:4d}/{args.generations} | "
            f"best_ever={best_reward_ever:.2f} | "
            f"surface={surface_name} (μ={static_mu:.2f}) | "
            f"episodes={episode_global}"
        )

# ===========================================================================
# Done
# ===========================================================================

print("\n" + "=" * 70)
print(f"PHASE {args.phase} TRAINING COMPLETE")
print(f"  Best reward:  {best_reward_ever:.2f}")
print(f"  Checkpoint:   {PHASE_CHECKPOINT[args.phase]}.npz")
print(f"  Log:          logs/training_phase{args.phase}.csv")
print("=" * 70)

logger.close()
simulation_app.close()
print("Done.")
