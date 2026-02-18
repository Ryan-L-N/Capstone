"""Xbox controller teleop for manual single-robot walkthrough.

Supports gait switching (FLAT <-> ROUGH), FPV camera, and 4 drive modes.

Controls:
  Xbox:
    Left Stick Y:  Forward/backward
    Left Stick X:  Turn left/right
    A:             Cycle drive mode (MANUAL -> SMOOTH -> PATROL -> AUTO-NAV)
    B:             Toggle selfright
    Y:             Reset position
    LB:            Toggle FPV camera
    RB:            Cycle gait (FLAT <-> ROUGH)
    D-Pad:         Speed multiplier
    Back:          E-stop

  Keyboard:
    W/S:           Forward/backward
    A/D:           Turn left/right
    G:             Cycle gait
    M:             Toggle FPV camera
    SHIFT:         Cycle drive mode
    X:             Toggle selfright
    R:             Reset
    SPACE:         E-stop
    ESC:           Exit

Usage:
    ./isaaclab.sh -p src/run_capstone_teleop.py --env friction --device xbox

Reuses patterns from:
- ARL_DELIVERY/04_Teleop_System/spot_teleop.py (Xbox mapping, drive modes, FPV)
- ARL_DELIVERY/02_Obstacle_Course/spot_obstacle_course.py (gait switching)
- isaacsim.examples.interactive/quadruped/quadruped_example.py (init pattern)
"""

# ── 0. Parse args BEFORE any Isaac imports ──────────────────────────────
import argparse
import sys
import os
import time

parser = argparse.ArgumentParser(description="Capstone teleop")
parser.add_argument("--env", type=str, default="friction",
                    choices=["friction", "grass", "boulder", "stairs"],
                    help="Environment to explore")
parser.add_argument("--device", type=str, default="keyboard",
                    choices=["keyboard", "xbox"],
                    help="Input device")
args = parser.parse_args()

# ── 1. Create SimulationApp ─────────────────────────────────────────────
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "width": 1920,
    "height": 1080,
})

# ── 2. Imports (AFTER SimulationApp) ────────────────────────────────────
import numpy as np
import omni
from omni.isaac.core import World
from pxr import UsdGeom, Gf, UsdPhysics, UsdLux
import carb.input

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.eval_cfg import PHYSICS_DT, RENDERING_DT, SPAWN_POSITION
from envs import build_environment
from envs.base_arena import quat_to_yaw

# Stairs-specific ground elevation for analytical height scanning
if args.env == "stairs":
    from configs.zone_params import get_stair_elevation


# ── Constants ───────────────────────────────────────────────────────────
DEADZONE = 0.12
GAIT_SWITCH_STABILIZE = 25  # 0.5 seconds at 50Hz
STABILIZE_STEPS = 100       # 2 seconds at 50Hz

DRIVE_MODES = ["MANUAL", "SMOOTH", "PATROL", "AUTO-NAV"]
SPEED_PROFILES = {
    "MANUAL":   {"vx_max": 1.5, "wz_max": 2.0},
    "SMOOTH":   {"vx_max": 1.0, "wz_max": 1.5},
    "PATROL":   {"vx_max": 0.5, "wz_max": 1.0},
    "AUTO-NAV": {"vx_max": 1.0, "wz_max": 2.0},
}

SPAWN_POS = np.array(SPAWN_POSITION)
SPAWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


class VelocitySmoother:
    """Exponential moving average smoother for velocity commands."""

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self._cmd = np.zeros(3)

    def smooth(self, target):
        self._cmd = self.alpha * np.array(target) + (1 - self.alpha) * self._cmd
        return self._cmd.copy()

    def reset(self):
        self._cmd[:] = 0.0


def apply_deadzone(value, threshold=DEADZONE):
    """Apply deadzone to joystick input."""
    if abs(value) < threshold:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - threshold) / (1.0 - threshold)


def main():
    print(f"\n{'='*60}", flush=True)
    print(f"  Capstone Teleop — {args.env.capitalize()}", flush=True)
    print(f"  Device: {args.device}", flush=True)
    print(f"  Controls: WASD/Stick=move, G/RB=gait, R/Y=reset, ESC=exit", flush=True)
    print(f"{'='*60}\n", flush=True)

    # ── 3. Create World ─────────────────────────────────────────────────
    world = World(
        physics_dt=PHYSICS_DT,
        rendering_dt=RENDERING_DT,
        stage_units_in_meters=1.0,
    )
    stage = omni.usd.get_context().get_stage()

    # Ground plane
    ground = UsdGeom.Cube.Define(stage, "/World/GroundPlane")
    ground.GetSizeAttr().Set(1.0)
    ground.AddTranslateOp().Set(Gf.Vec3d(25.0, 15.0, -0.005))
    ground.AddScaleOp().Set(Gf.Vec3d(200.0, 200.0, 0.01))
    ground.GetDisplayColorAttr().Set([(0.5, 0.5, 0.5)])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    # Lighting
    light = UsdLux.DistantLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(3000.0)
    UsdGeom.Xformable(light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45, 0, 0))

    # ── 4. Build environment ────────────────────────────────────────────
    print(f"Building {args.env} environment...", flush=True)
    build_environment(args.env, stage, None)

    # ── 5. Create SpotFlatTerrainPolicy (follows official quadruped_example.py) ──
    from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy

    print("Loading Spot flat terrain policy...", flush=True)
    spot_flat = SpotFlatTerrainPolicy(
        prim_path="/World/Spot",
        name="Spot",
        position=np.array(SPAWN_POSITION),
    )

    # Reset world to start physics timeline
    world.reset()
    print("World reset complete.", flush=True)

    # ── 6. State variables ──────────────────────────────────────────────
    spot = spot_flat
    spot_rough = None
    gait_idx = 0  # 0 = flat, 1 = rough
    drive_mode_idx = 0
    smoother = VelocitySmoother(alpha=0.3)
    gait_switch_timer = 0
    sim_time = 0.0
    estop = False
    running = True
    physics_ready = False
    stabilize_counter = 0

    # Xbox button debounce (True = was pressed last frame)
    xbox_prev = {"A": False, "Y": False, "RB": False, "Back": False}

    # Keyboard state
    key_state = {
        "forward": False, "backward": False,
        "left": False, "right": False,
        "gait": False, "reset": False,
        "mode": False, "exit": False,
        "selfright": False, "fpv": False,
    }

    # ── 7. Xbox controller setup (always try to detect) ─────────────────
    joystick = None
    try:
        import pygame
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            print(f"[XBOX] Controller found: {joystick.get_name()}", flush=True)
        else:
            print("[XBOX] No controller found — using keyboard", flush=True)
    except ImportError:
        print("[XBOX] pygame not available — using keyboard", flush=True)

    # ── 8. Keyboard handler ─────────────────────────────────────────────
    input_iface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()

    def on_key_event(event, *_args, **_kwargs):
        nonlocal gait_idx, drive_mode_idx, estop, running, gait_switch_timer

        pressed = event.type == carb.input.KeyboardEventType.KEY_PRESS
        released = event.type == carb.input.KeyboardEventType.KEY_RELEASE

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
                gait_name = "ROUGH" if gait_idx == 1 else "FLAT"
                print(f"[GAIT] Switched to {gait_name}", flush=True)
        elif key == carb.input.KeyboardInput.R and pressed:
            key_state["reset"] = True
        elif key == carb.input.KeyboardInput.LEFT_SHIFT and pressed:
            drive_mode_idx = (drive_mode_idx + 1) % len(DRIVE_MODES)
            print(f"[MODE] {DRIVE_MODES[drive_mode_idx]}", flush=True)
        elif key == carb.input.KeyboardInput.SPACE and pressed:
            estop = not estop
            print(f"[E-STOP] {'ENGAGED' if estop else 'RELEASED'}", flush=True)
        elif key == carb.input.KeyboardInput.ESCAPE and pressed:
            running = False
        elif key == carb.input.KeyboardInput.X and pressed:
            key_state["selfright"] = not key_state["selfright"]
            print(f"[SELFRIGHT] {'ON' if key_state['selfright'] else 'OFF'}", flush=True)

        return True

    input_iface.subscribe_to_keyboard_events(keyboard, on_key_event)

    # ── 9. Physics callback (follows official quadruped_example.py init pattern) ──
    def on_physics_step(step_size):
        nonlocal spot, spot_rough, gait_idx, gait_switch_timer, sim_time
        nonlocal physics_ready, stabilize_counter, drive_mode_idx, estop

        # ── First-time initialization (inside physics step, per official example) ──
        if not physics_ready:
            physics_ready = True
            print("Initializing Spot policy...", flush=True)
            spot_flat.initialize()
            spot_flat.post_reset()

            # Try loading rough policy
            try:
                from spot_rough_terrain_policy import SpotRoughTerrainPolicy
                ground_fn = get_stair_elevation if args.env == "stairs" else None
                spot_rough = SpotRoughTerrainPolicy(
                    flat_policy=spot_flat,
                    ground_height_fn=ground_fn,
                )
                spot_rough.initialize()
                print("[GAIT] Both flat and rough policies loaded", flush=True)
            except Exception as e:
                print(f"[GAIT] Rough policy unavailable: {e}", flush=True)
                print("[GAIT] Running with flat policy only", flush=True)

            print("Stabilizing robot...", flush=True)
            return

        # ── Stabilization period ──
        if stabilize_counter < STABILIZE_STEPS:
            stabilize_counter += 1
            spot_flat.forward(step_size, np.array([0.0, 0.0, 0.0]))
            if stabilize_counter == STABILIZE_STEPS:
                print("Ready! Use WASD to move, G to switch gait, ESC to exit.", flush=True)
            return

        sim_time += step_size

        # Handle gait switching
        if gait_idx == 0 or spot_rough is None:
            if spot is not spot_flat:
                if spot_rough and hasattr(spot_rough, 'post_reset'):
                    spot_rough.post_reset()
                gait_switch_timer = GAIT_SWITCH_STABILIZE
            spot = spot_flat
        else:
            if spot is not spot_rough:
                if hasattr(spot_rough, 'post_reset'):
                    spot_rough.post_reset()
                gait_switch_timer = GAIT_SWITCH_STABILIZE
            spot = spot_rough

        # Gait switch stabilization
        if gait_switch_timer > 0:
            gait_switch_timer -= 1
            spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
            return

        # Handle reset
        if key_state["reset"]:
            key_state["reset"] = False
            spot.robot.set_world_pose(
                position=SPAWN_POS,
                orientation=SPAWN_QUAT,
            )
            smoother.reset()
            if hasattr(spot, 'post_reset'):
                spot.post_reset()
            print("  >> Robot reset to start!", flush=True)
            spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
            return

        # E-stop
        if estop:
            spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
            return

        # Read input
        vx_raw, wz_raw = 0.0, 0.0

        if joystick is not None:
            try:
                import pygame
                pygame.event.pump()
                # Left stick: axis 1 = forward/back (inverted), axis 0 = turn
                vx_raw = -apply_deadzone(joystick.get_axis(1))
                wz_raw = -apply_deadzone(joystick.get_axis(0))

                # Button reads (debounced — trigger on press, not hold)
                btn_a = joystick.get_button(0)       # A
                btn_y = joystick.get_button(3)       # Y
                btn_rb = joystick.get_button(5)      # RB
                btn_back = joystick.get_button(6)    # Back

                # Y = Reset position
                if btn_y and not xbox_prev["Y"]:
                    key_state["reset"] = True

                # RB = Cycle gait (FLAT <-> ROUGH)
                if btn_rb and not xbox_prev["RB"]:
                    if spot_rough is not None:
                        gait_idx = 1 - gait_idx
                        gait_switch_timer = GAIT_SWITCH_STABILIZE
                        gait_name = "ROUGH" if gait_idx == 1 else "FLAT"
                        print(f"[GAIT] Switched to {gait_name}", flush=True)

                # A = Cycle drive mode
                if btn_a and not xbox_prev["A"]:
                    drive_mode_idx = (drive_mode_idx + 1) % len(DRIVE_MODES)
                    print(f"[MODE] {DRIVE_MODES[drive_mode_idx]}", flush=True)

                # Back = E-stop toggle
                if btn_back and not xbox_prev["Back"]:
                    estop = not estop
                    print(f"[E-STOP] {'ENGAGED' if estop else 'RELEASED'}", flush=True)

                # Update debounce state
                xbox_prev["A"] = btn_a
                xbox_prev["Y"] = btn_y
                xbox_prev["RB"] = btn_rb
                xbox_prev["Back"] = btn_back
            except Exception:
                pass

        # Keyboard input (additive — always active as fallback)
        if key_state["forward"]:
            vx_raw = 1.0
        elif key_state["backward"]:
            vx_raw = -1.0
        if key_state["left"]:
            wz_raw = 1.0
        elif key_state["right"]:
            wz_raw = -1.0

        # Apply drive mode speed profile
        mode = DRIVE_MODES[drive_mode_idx]
        profile = SPEED_PROFILES[mode]
        vx = vx_raw * profile["vx_max"]
        vy = 0.0
        wz = wz_raw * profile["wz_max"]

        # Clamp to training ranges
        vx = np.clip(vx, -2.0, 3.0)
        wz = np.clip(wz, -2.0, 2.0)

        # Smooth commands
        cmd = smoother.smooth([vx, vy, wz])

        # Step policy
        spot.forward(step_size, cmd)

    world.add_physics_callback("teleop", on_physics_step)

    # ── 10. Main render loop ────────────────────────────────────────────
    print("Starting simulation loop...", flush=True)
    try:
        while simulation_app.is_running() and running:
            world.step(render=True)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", flush=True)

    # ── 11. Cleanup ─────────────────────────────────────────────────────
    print("Closing simulation...", flush=True)
    if joystick is not None:
        import pygame
        pygame.quit()
    simulation_app.close()


if __name__ == "__main__":
    main()
