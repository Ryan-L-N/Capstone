"""Keyboard teleop for the 5-ring (4-quadrant) gauntlet arena.

Mirrors `4_env_test/src/run_capstone_teleop.py` keyboard handler but uses
`5_ring_test/src/envs/ring_arena.create_quadrant_arena` for the scene.
Drive Spot through the friction/grass/boulder/stairs quadrants manually
to feel out the policy on the same arena that `run_ring_eval.py` benchmarks.

Controls (keyboard):
    W / S         forward / backward
    A / D         turn left / right
    G             cycle gait (FLAT <-> ROUGH)
    SHIFT         cycle drive mode (MANUAL / SMOOTH / PATROL)
    R             reset robot to spawn
    SPACE         e-stop toggle
    ESC           exit

Usage:
    python src/run_ring_teleop.py \\
        --checkpoint Experiments/Alex/PARKOUR_NAV/checkpoints/parkour_phase10b_19499.pt \\
        --mason --action_scale 0.3
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="5-ring (4-quadrant) teleop")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Rough policy ckpt. If omitted, flat policy only.")
parser.add_argument("--mason", action="store_true", default=False,
                    help="Mason obs order (height_scan first) for PARKOUR_NAV ckpts.")
parser.add_argument("--action_scale", type=float, default=None,
                    help="Override action_scale (e.g. 0.3 for PARKOUR_NAV).")
args = parser.parse_args()

# ── 1. SimulationApp before any Isaac imports ───────────────────────────
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({
    "headless": False,
    "width": 1920,
    "height": 1080,
})

# ── 2. Imports after SimulationApp ──────────────────────────────────────
import numpy as np  # noqa: E402
import omni  # noqa: E402
import carb.input  # noqa: E402
from omni.isaac.core import World  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.eval_cfg import (  # noqa: E402
    PHYSICS_DT, RENDERING_DT, SPAWN_POSITION,
)
from envs.base_arena import quat_to_yaw  # noqa: E402
from envs.ring_arena import create_quadrant_arena  # noqa: E402
from navigation.ring_follower import QuadrantFollower  # noqa: E402

SPAWN_POS = np.array(SPAWN_POSITION)
SPAWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
STABILIZE_STEPS = 100
GAIT_SWITCH_STABILIZE = 50

DRIVE_MODES = ["MANUAL", "SMOOTH", "PATROL"]
SPEED_PROFILES = {
    "MANUAL":  {"vx_max": 1.5, "wz_max": 2.0},
    "SMOOTH":  {"vx_max": 1.0, "wz_max": 1.5},
    "PATROL":  {"vx_max": 0.5, "wz_max": 1.0},
}


def main():
    print(f"\n{'='*60}", flush=True)
    print("  5-Ring Teleop — 4-Quadrant Gauntlet", flush=True)
    print(f"  Checkpoint: {args.checkpoint or '(flat policy only)'}", flush=True)
    print(f"  Mason: {args.mason}  action_scale: {args.action_scale}", flush=True)
    print("  Controls: WASD=move, G=gait, SHIFT=mode, R=reset, SPACE=e-stop, ESC=exit", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Build world
    world = World(physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT,
                  stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # Ground failsafe
    ground = UsdGeom.Cube.Define(stage, "/World/GroundPlane")
    ground.GetSizeAttr().Set(1.0)
    ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.005))
    ground.AddScaleOp().Set(Gf.Vec3d(200.0, 200.0, 0.01))
    ground.GetDisplayColorAttr().Set([(0.4, 0.4, 0.4)])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    # Lighting
    light_xf = UsdGeom.Xform.Define(stage, "/World/Lights")
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/dome")
    dome.CreateIntensityAttr(500.0)
    dome.CreateColorAttr(Gf.Vec3f(0.85, 0.90, 1.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/sun")
    sun.CreateIntensityAttr(3000.0)
    sun.CreateAngleAttr(1.0)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.85))
    UsdGeom.Xformable(stage.GetPrimAtPath("/World/Lights/sun")).AddRotateXYZOp().Set(
        Gf.Vec3d(-45, 30, 0)
    )

    # Use QuadrantFollower's default WPs to place stair pyramids
    follower = QuadrantFollower()
    stairs_wp_positions = follower.stairs_waypoint_positions()

    print("Building 4-quadrant arena...", flush=True)
    create_quadrant_arena(stage, stairs_wp_positions)

    # Spot (flat policy first; rough swapped in if --checkpoint)
    from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy  # noqa: E402

    flat_policy = SpotFlatTerrainPolicy(
        prim_path="/World/Robot",
        name="Robot",
        position=SPAWN_POS,
    )
    world.reset()
    world.step(render=True)
    flat_policy.initialize()
    flat_policy.post_reset()

    spot_rough = None
    if args.checkpoint:
        try:
            from spot_rough_terrain_policy import SpotRoughTerrainPolicy  # noqa: E402
            spot_rough = SpotRoughTerrainPolicy(
                flat_policy=flat_policy,
                checkpoint_path=os.path.abspath(args.checkpoint),
                ground_height_fn=None,
                mason_baseline=args.mason,
                action_scale=args.action_scale,
            )
            spot_rough.initialize()
            spot_rough.apply_gains()
            spot_rough._decimation = 1
            print("[GAIT] Rough policy loaded", flush=True)
        except Exception as e:
            print(f"[GAIT] Rough policy unavailable: {e}", flush=True)

    spot = spot_rough if spot_rough is not None else flat_policy
    gait_idx = 1 if spot_rough is not None else 0

    # State
    drive_mode_idx = 0
    estop = False
    running = True
    stabilize_counter = 0
    gait_switch_timer = 0
    key_state = {"forward": False, "backward": False, "left": False, "right": False, "reset": False}

    # Keyboard handler
    input_iface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()

    def on_key_event(event, *_args, **_kwargs):
        nonlocal gait_idx, drive_mode_idx, estop, running, gait_switch_timer
        if event.type not in (carb.input.KeyboardEventType.KEY_PRESS,
                              carb.input.KeyboardEventType.KEY_RELEASE):
            return True
        pressed = event.type == carb.input.KeyboardEventType.KEY_PRESS
        key = event.input

        if key == carb.input.KeyboardInput.W:
            key_state["forward"] = pressed
        elif key == carb.input.KeyboardInput.S:
            key_state["backward"] = pressed
        elif key == carb.input.KeyboardInput.A:
            key_state["left"] = pressed
        elif key == carb.input.KeyboardInput.D:
            key_state["right"] = pressed
        elif key == carb.input.KeyboardInput.G and pressed:
            if spot_rough is not None:
                gait_idx = 1 - gait_idx
                gait_switch_timer = GAIT_SWITCH_STABILIZE
                print(f"[GAIT] {'ROUGH' if gait_idx == 1 else 'FLAT'}", flush=True)
        elif key == carb.input.KeyboardInput.LEFT_SHIFT and pressed:
            drive_mode_idx = (drive_mode_idx + 1) % len(DRIVE_MODES)
            print(f"[MODE] {DRIVE_MODES[drive_mode_idx]}", flush=True)
        elif key == carb.input.KeyboardInput.R and pressed:
            key_state["reset"] = True
        elif key == carb.input.KeyboardInput.SPACE and pressed:
            estop = not estop
            print(f"[E-STOP] {'ENGAGED' if estop else 'RELEASED'}", flush=True)
        elif key == carb.input.KeyboardInput.ESCAPE and pressed:
            running = False
        return True

    input_iface.subscribe_to_keyboard_events(keyboard, on_key_event)

    def on_physics_step(step_size):
        nonlocal spot, gait_idx, gait_switch_timer, stabilize_counter
        # Stabilize at start
        if stabilize_counter < STABILIZE_STEPS:
            stabilize_counter += 1
            current = spot_rough if (gait_idx == 1 and spot_rough is not None) else flat_policy
            current.forward(step_size, np.array([0.0, 0.0, 0.0]))
            if stabilize_counter == STABILIZE_STEPS:
                print("Ready! WASD to move.", flush=True)
            return

        # Gait selection
        if gait_idx == 1 and spot_rough is not None:
            spot = spot_rough
        else:
            spot = flat_policy

        if gait_switch_timer > 0:
            gait_switch_timer -= 1
            spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
            return

        # Reset
        if key_state["reset"]:
            key_state["reset"] = False
            spot.robot.set_world_pose(position=SPAWN_POS, orientation=SPAWN_QUAT)
            if hasattr(spot, "post_reset"):
                spot.post_reset()
            spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
            return

        if estop:
            spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
            return

        # Read keyboard input → cmd
        vx_raw = (1.0 if key_state["forward"] else 0.0) - (1.0 if key_state["backward"] else 0.0)
        wz_raw = (1.0 if key_state["left"] else 0.0) - (1.0 if key_state["right"] else 0.0)

        profile = SPEED_PROFILES[DRIVE_MODES[drive_mode_idx]]
        cmd = np.array([vx_raw * profile["vx_max"], 0.0, wz_raw * profile["wz_max"]])
        spot.forward(step_size, cmd)

    world.add_physics_callback("teleop", on_physics_step)

    print("Starting simulation loop...", flush=True)
    try:
        while simulation_app.is_running() and running:
            world.step(render=True)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", flush=True)

    print("Closing simulation...", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
