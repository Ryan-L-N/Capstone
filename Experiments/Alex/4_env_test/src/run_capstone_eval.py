"""Main entry point for headless and rendered capstone evaluation.

Runs N episodes per (environment, policy) pair, collecting per-episode metrics
and optionally capturing video.

Usage (headless on H100):
    ./isaaclab.sh -p src/run_capstone_eval.py --headless \
        --num_episodes 1000 --policy flat --env friction \
        --output_dir results/

Usage (rendered with video):
    ./isaaclab.sh -p src/run_capstone_eval.py \
        --num_episodes 10 --policy rough --env stairs \
        --rendered --capture_video --output_dir results/rendered/

Usage (debug — 5 iterations):
    ./isaaclab.sh -p src/run_capstone_eval.py --headless \
        --num_episodes 5 --policy flat --env friction \
        --output_dir results/debug/

Reuses patterns from:
- ARL_DELIVERY/05_Training_Package/spot_rough_48h_cfg.py (AppLauncher, headless)
- ARL_DELIVERY/02_Obstacle_Course/spot_obstacle_course.py (standalone World API)
- ARL_DELIVERY/03_Rough_Terrain_Policy/spot_rough_terrain_policy.py (policy loading)
"""

# ── 0. Parse args BEFORE any Isaac imports ──────────────────────────────
import argparse
import sys
import os
import time

parser = argparse.ArgumentParser(description="Capstone 4-environment evaluation")
parser.add_argument("--env", type=str, required=True,
                    choices=["friction", "grass", "boulder", "stairs"],
                    help="Environment to evaluate")
parser.add_argument("--policy", type=str, required=True,
                    choices=["flat", "rough"],
                    help="Policy type to evaluate")
parser.add_argument("--num_episodes", type=int, default=1000,
                    help="Number of episodes to run (default: 1000)")
parser.add_argument("--headless", action="store_true", default=False,
                    help="Run without GUI rendering")
parser.add_argument("--rendered", action="store_true", default=False,
                    help="Force rendered mode (overrides --headless)")
parser.add_argument("--capture_video", action="store_true", default=False,
                    help="Capture video of episodes (requires rendered)")
parser.add_argument("--output_dir", type=str, default="results/",
                    help="Output directory for JSONL and video files")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")

args = parser.parse_args()

# Determine headless mode
headless = args.headless and not args.rendered

# ── 1. Create SimulationApp BEFORE any omni imports ─────────────────────
from isaacsim import SimulationApp

app_config = {
    "headless": headless,
    "width": 1920,
    "height": 1080,
}
simulation_app = SimulationApp(app_config)

# ── 2. Now safe to import Isaac and project modules ─────────────────────
import numpy as np
import omni
from omni.isaac.core import World
from pxr import UsdGeom, Gf, UsdPhysics

# Add src/ to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.eval_cfg import (
    PHYSICS_DT, RENDERING_DT, CONTROL_DT, DECIMATION,
    MAX_CONTROL_STEPS, FALL_THRESHOLD, COMPLETION_X,
    SPAWN_POSITION, STIFFNESS, DAMPING, ACTION_SCALE,
)
from configs.zone_params import ZONE_PARAMS
from envs import build_environment
from envs.base_arena import quat_to_yaw
from navigation.waypoint_follower import WaypointFollower
from metrics.collector import MetricsCollector

# Grass-specific drag scaling
if args.env == "grass":
    from envs.grass_env import get_velocity_scale


# ── Constants ───────────────────────────────────────────────────────────
STABILIZE_STEPS = 100  # 2 seconds at 50 Hz
SPAWN_POS = np.array(SPAWN_POSITION)
SPAWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # identity — facing +X


def main():
    print(f"\n{'='*60}")
    print(f"  Capstone Evaluation")
    print(f"  Environment: {args.env}")
    print(f"  Policy:      {args.policy}")
    print(f"  Episodes:    {args.num_episodes}")
    print(f"  Headless:    {headless}")
    print(f"  Output:      {args.output_dir}")
    print(f"{'='*60}\n")

    # ── 3. Create World ─────────────────────────────────────────────────
    world = World(
        physics_dt=PHYSICS_DT,
        rendering_dt=RENDERING_DT,
        stage_units_in_meters=1.0,
    )
    stage = omni.usd.get_context().get_stage()

    # ── 4. Build base ground (raw USD cube for GPU PhysX compatibility) ─
    ground = UsdGeom.Cube.Define(stage, "/World/GroundPlane")
    ground.GetSizeAttr().Set(1.0)
    ground.AddTranslateOp().Set(Gf.Vec3d(25.0, 15.0, -0.005))
    ground.AddScaleOp().Set(Gf.Vec3d(200.0, 200.0, 0.01))
    ground.GetDisplayColorAttr().Set([(0.5, 0.5, 0.5)])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    # ── 5. Build environment ────────────────────────────────────────────
    print(f"Building {args.env} environment...")
    build_environment(args.env, stage, None)

    # ── 6. Load policy and spawn robot ──────────────────────────────────
    print(f"Loading {args.policy} policy...")

    if args.policy == "flat":
        from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
        spot = SpotFlatTerrainPolicy(
            prim_path="/World/Spot",
            name="Spot",
            position=Gf.Vec3d(*SPAWN_POSITION),
        )
    else:
        from spot_rough_terrain_policy import SpotRoughTerrainPolicy
        # Rough needs flat as base for shared articulation
        from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
        spot_flat = SpotFlatTerrainPolicy(
            prim_path="/World/Spot",
            name="Spot",
            position=Gf.Vec3d(*SPAWN_POSITION),
        )
        world.scene.add(spot_flat)
        world.reset()
        spot_flat.initialize()
        spot = SpotRoughTerrainPolicy(flat_policy=spot_flat)
        spot.initialize()

    if args.policy == "flat":
        world.scene.add(spot)
        world.reset()
        spot.initialize()

    print("Robot loaded and initialized.")

    # ── 7. Create navigation and metrics ────────────────────────────────
    waypoint_follower = WaypointFollower()
    metrics_collector = MetricsCollector(args.output_dir, args.env, args.policy)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 8. Stabilization period ─────────────────────────────────────────
    print("Stabilizing robot...")
    for _ in range(STABILIZE_STEPS):
        spot.forward(PHYSICS_DT, np.array([0.0, 0.0, 0.0]))
        world.step(render=not headless)

    # ── 9. Episode loop ─────────────────────────────────────────────────
    total_start = time.time()

    for ep_idx in range(args.num_episodes):
        ep_id = f"{args.env}_{args.policy}_ep{ep_idx:04d}"
        ep_start = time.time()

        # Reset robot to spawn
        spot.robot.set_world_pose(
            position=SPAWN_POS,
            orientation=SPAWN_QUAT,
        )

        # Reset policy state
        if hasattr(spot, 'post_reset'):
            spot.post_reset()

        # Reset navigation and metrics
        waypoint_follower.reset()
        metrics_collector.start_episode(ep_id)

        # Brief stabilization after reset
        for _ in range(10):
            spot.forward(PHYSICS_DT, np.array([0.0, 0.0, 0.0]))
            world.step(render=not headless)

        # Step loop
        for step in range(MAX_CONTROL_STEPS):
            # Get robot state
            pos, quat = spot.robot.get_world_pose()
            lin_vel = spot.robot.get_linear_velocity()
            ang_vel = spot.robot.get_angular_velocity()

            pos_np = np.array(pos, dtype=np.float64)
            quat_np = np.array(quat, dtype=np.float64)
            lin_vel_np = np.array(lin_vel, dtype=np.float64)
            ang_vel_np = np.array(ang_vel, dtype=np.float64)

            yaw = quat_to_yaw(quat_np)
            sim_time = step * CONTROL_DT

            # Record metrics
            metrics_collector.step(
                root_pos=pos_np,
                root_quat=quat_np,
                root_lin_vel=lin_vel_np,
                root_ang_vel=ang_vel_np,
                sim_time=sim_time,
            )

            # Check termination (fall)
            if metrics_collector.episode_done():
                break

            # Compute navigation commands
            cmd = waypoint_follower.compute_commands(pos_np, yaw)

            # Apply grass drag if applicable
            if args.env == "grass":
                vscale = get_velocity_scale(float(pos_np[0]))
                cmd[0] *= vscale

            # Step policy and simulation
            spot.forward(PHYSICS_DT, cmd)
            world.step(render=not headless)

            # Check waypoint completion
            if waypoint_follower.is_done:
                break

        # End episode
        result = metrics_collector.end_episode()
        ep_time = time.time() - ep_start

        # Progress log
        status = "COMPLETE" if result.get("completion") else "FELL" if result.get("fall_detected") else "TIMEOUT"
        progress = result.get("progress", 0)
        zone = result.get("zone_reached", 0)
        print(f"  [{ep_idx+1:4d}/{args.num_episodes}] {ep_id}  "
              f"{status:8s}  progress={progress:5.1f}m  zone={zone}  "
              f"time={ep_time:.1f}s")

        # Save periodically (every 50 episodes)
        if (ep_idx + 1) % 50 == 0:
            metrics_collector.save()
            print(f"  >> Saved checkpoint at episode {ep_idx+1}")

    # ── 10. Final save and summary ──────────────────────────────────────
    metrics_collector.save()

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Evaluation complete!")
    print(f"  Total episodes: {args.num_episodes}")
    print(f"  Total time:     {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Avg per episode: {total_time/args.num_episodes:.1f}s")
    print(f"  Results saved to: {args.output_dir}")
    print(f"{'='*60}\n")

    # ── 11. Cleanup ─────────────────────────────────────────────────────
    simulation_app.close()


if __name__ == "__main__":
    main()
