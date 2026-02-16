"""
Spot Obstacle Course - Keyboard & Xbox Controller
===================================================

Drive Spot through a 100m x 15m obstacle course with 12 terrain segments:
  START -> WARM-UP (grass) -> GRASS+STONES -> BREAK -> STEPS (0.75m) ->
  FLAT -> RUBBLE POOL (-0.5m) -> FLAT -> LARGE BLOCKS -> FLAT ->
  INSTABILITY FIELD (120 bricks) -> FINISH

XBOX CONTROLLER:
  Left Stick Y       Forward / backward (analog)
  Left Stick X       Turn left / right (analog)
  A                  Cycle drive mode
  B                  Toggle selfright mode
  Y                  Reset to start
  LB                 Toggle FPV camera
  RB                 Cycle gait mode (FLAT <-> ROUGH)
  D-Pad Up/Down      Speed multiplier +/-
  Back               Emergency stop
  (In selfright: Left Stick X = roll direction)

KEYBOARD:
  W / S         Forward / Backward
  A / D         Turn left / Turn right
  SPACE         Emergency stop
  SHIFT         Cycle drive mode: MANUAL -> SMOOTH -> PATROL -> AUTO-NAV
  G             Cycle gait mode: FLAT <-> ROUGH
  M             Toggle FPV camera (first-person onboard view)
  X             Toggle selfright mode (A/D = roll, auto-exits when upright)
  H             Show current segment info
  UP / DOWN     Adjust speed multiplier
  R             Reset robot to start
  ESC           Exit

DRIVE MODES:
  MANUAL    - Instant response, arcade feel
  SMOOTH    - Velocity ramping, real robot inertia
  PATROL    - Slow & careful, optimized for rough terrain
  AUTO-NAV  - Autonomous waypoints, WASD overrides

Isaac Sim 5.1.0 + Isaac Lab 2.3.0
"""

import numpy as np
import argparse
import sys
import os
import torch

# Parse args BEFORE SimulationApp (required)
parser = argparse.ArgumentParser(description="Spot Obstacle Course")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
args = parser.parse_args()

# SimulationApp MUST be created before any omni.isaac imports
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": args.headless,
    "width": 1920,
    "height": 1080,
    "window_width": 1920,
    "window_height": 1080,
})

# Now safe to import Isaac modules
import omni
from omni.isaac.core import World
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
from pxr import UsdGeom, Gf, UsdPhysics, UsdShade, PhysxSchema, UsdLux
import carb.input
from omni.kit.viewport.utility import get_active_viewport

# Import rough terrain policy (if available - fallback to flat if not trained yet)
try:
    from spot_rough_terrain_policy import SpotRoughTerrainPolicy
    HAS_ROUGH_POLICY = True
except ImportError:
    HAS_ROUGH_POLICY = False
    print("[WARNING] SpotRoughTerrainPolicy not available - gait cycling disabled")

# Add parent paths so we can import from core/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Optional: pygame for Xbox controller / joystick support
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

# Course dimensions
COURSE_LENGTH = 100.0   # meters (X-axis)
COURSE_WIDTH = 15.0     # meters (Y-axis)
WALL_HEIGHT = 1.0       # meters (Z-axis)
WALL_THICKNESS = 0.15   # meters

# Robot start position (center of START zone, raised for settling)
START_POS = np.array([5.0, 7.5, 0.7])

# Segment definitions (ordered by X position)
SEGMENTS = [
    {"name": "START",       "x_min": 0,   "x_max": 10,  "color": (0.50, 0.50, 0.50), "desc": "Flat baseline"},
    {"name": "WARM-UP",     "x_min": 10,  "x_max": 20,  "color": (0.30, 0.60, 0.20), "desc": "Light grass (H1)"},
    {"name": "GRASS+STONES","x_min": 20,  "x_max": 30,  "color": (0.20, 0.50, 0.15), "desc": "Medium grass + stones"},
    {"name": "BREAK 1",     "x_min": 30,  "x_max": 35,  "color": (0.50, 0.50, 0.50), "desc": "Flat rest zone"},
    {"name": "STEPS",       "x_min": 35,  "x_max": 45,  "color": (0.60, 0.55, 0.40), "desc": "Steps up & down (0.75m)"},
    {"name": "FLAT 2",      "x_min": 45,  "x_max": 50,  "color": (0.50, 0.50, 0.50), "desc": "Flat"},
    {"name": "RUBBLE POOL", "x_min": 50,  "x_max": 60,  "color": (0.55, 0.40, 0.30), "desc": "Walled debris pool with rubble"},
    {"name": "FLAT 3",      "x_min": 60,  "x_max": 65,  "color": (0.50, 0.50, 0.50), "desc": "Flat"},
    {"name": "LARGE BLOCKS","x_min": 65,  "x_max": 75,  "color": (0.45, 0.42, 0.38), "desc": "Static block maze"},
    {"name": "FLAT 4",      "x_min": 75,  "x_max": 80,  "color": (0.50, 0.50, 0.50), "desc": "Flat"},
    {"name": "INSTABILITY", "x_min": 80,  "x_max": 90,  "color": (0.70, 0.50, 0.30), "desc": "120 dynamic bricks"},
    {"name": "FINISH",      "x_min": 90,  "x_max": 100, "color": (0.20, 0.70, 0.20), "desc": "Finish zone"},
]

# Grass height configs (used for WARM-UP and GRASS+STONES segments)
GRASS_HEIGHTS = {
    "H1": {"height": 0.1, "friction": 0.80, "description": "Ankle (0.1m)"},
    "H2": {"height": 0.3, "friction": 0.85, "description": "Knee (0.3m)"},
}

# AUTO-NAV waypoints along course centerline
NAV_WAYPOINTS = [
    (15.0, 7.5),    # WARM-UP
    (25.0, 7.5),    # GRASS+STONES
    (32.5, 7.5),    # BREAK 1
    (40.0, 7.5),    # STEPS
    (47.5, 7.5),    # FLAT 2
    (55.0, 7.5),    # RUBBLE POOL
    (62.5, 7.5),    # FLAT 3
    (70.0, 7.5),    # LARGE BLOCKS
    (77.5, 7.5),    # FLAT 4
    (85.0, 7.5),    # INSTABILITY
    (95.0, 7.5),    # FINISH
]

# Rubble piece definitions (reused for pool debris and instability bricks)
RUBBLE_PIECES = [
    # Small bricks
    {"name": "brick",  "half": (0.05, 0.025, 0.025), "mass": 1.0,
     "color": (0.65, 0.45, 0.30)},
    {"name": "chip",   "half": (0.04, 0.03,  0.015), "mass": 0.5,
     "color": (0.70, 0.55, 0.40)},
    # Medium blocks
    {"name": "block",  "half": (0.10, 0.05,  0.05),  "mass": 5.0,
     "color": (0.55, 0.50, 0.45)},
    {"name": "chunk",  "half": (0.12, 0.08,  0.06),  "mass": 7.0,
     "color": (0.50, 0.45, 0.40)},
    # Large blocks
    {"name": "slab",   "half": (0.15, 0.10,  0.075), "mass": 10.0,
     "color": (0.45, 0.42, 0.38)},
    {"name": "beam",   "half": (0.20, 0.06,  0.06),  "mass": 12.0,
     "color": (0.52, 0.48, 0.42)},
]

# Self-right (physics-based rollover recovery)
SELFRIGHT_ROLL_ACCEL = 12.0
SELFRIGHT_MAX_ROLL_VEL = 2.5
SELFRIGHT_GROUND_LIFT = 0.8
SELFRIGHT_DAMPING = 3.0
SELFRIGHT_UPRIGHT_DEG = 35.0
SELFRIGHT_UPRIGHT_TIME = 0.3

# Xbox controller mapping (XInput on Windows via pygame)
XBOX_AXIS_TURN = 0       # Left Stick X  (-1=left, +1=right)
XBOX_AXIS_FWD = 1        # Left Stick Y  (-1=up/fwd, +1=down/back)
XBOX_DEADZONE = 0.12     # 12% dead zone (Xbox sticks drift)
XBOX_BTN_A = 0           # Cycle drive mode
XBOX_BTN_B = 1           # Toggle selfright
XBOX_BTN_Y = 3           # Reset position
XBOX_BTN_LB = 4          # Toggle FPV camera
XBOX_BTN_RB = 5          # Cycle gait mode
XBOX_BTN_BACK = 6        # Emergency stop

# Drive mode configurations
DRIVE_MODES = [
    {
        "name": "MANUAL",
        "description": "Direct WASD (instant response)",
        "max_vx": 1.5,
        "max_vx_rev": 0.5,
        "max_wz": 1.0,
        "use_smoothing": False,
        "accel_rate": 999.0,
        "decel_rate": 999.0,
        "min_forward_for_turn": 0.0,
    },
    {
        "name": "SMOOTH",
        "description": "Real robot inertia",
        "max_vx": 1.2,
        "max_vx_rev": 0.4,
        "max_wz": 0.8,
        "use_smoothing": True,
        "accel_rate": 1.5,
        "decel_rate": 3.0,
        "min_forward_for_turn": 0.3,
    },
    {
        "name": "PATROL",
        "description": "Slow & careful (rough terrain)",
        "max_vx": 0.6,
        "max_vx_rev": 0.3,
        "max_wz": 0.5,
        "use_smoothing": True,
        "accel_rate": 1.0,
        "decel_rate": 2.5,
        "min_forward_for_turn": 0.3,
    },
    {
        "name": "AUTO-NAV",
        "description": "Autonomous + WASD override",
        "max_vx": 1.0,
        "max_vx_rev": 0.3,
        "max_wz": 0.8,
        "use_smoothing": True,
        "accel_rate": 1.2,
        "decel_rate": 3.0,
        "min_forward_for_turn": 0.5,
    },
]

# Gait mode configurations (locomotion policy switching)
GAIT_MODES = [
    {
        "name": "FLAT",
        "description": "Flat terrain policy (fast, efficient)",
        "policy_type": "flat",
    },
    {
        "name": "ROUGH",
        "description": "Rough terrain policy (stairs, rubble, obstacles)",
        "policy_type": "rough",
    },
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def quat_to_yaw(quat):
    """Convert quaternion [w, x, y, z] to yaw angle (ES-002)."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def distance_2d(pos1, pos2):
    """2D distance between two positions."""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def get_roll_pitch(quat):
    """Extract roll and pitch angles (radians) from quaternion [w,x,y,z]."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def is_rolled_over(quat):
    """Check if robot is rolled over (roll or pitch > 60 degrees)."""
    roll, pitch = get_roll_pitch(quat)
    return abs(roll) > np.radians(60) or abs(pitch) > np.radians(60)


def quat_forward_axis(quat):
    """Get body forward direction (X-axis) in world frame from quaternion [w,x,y,z]."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    fx = 1.0 - 2.0 * (y * y + z * z)
    fy = 2.0 * (x * y + w * z)
    fz = 2.0 * (x * z - w * y)
    return np.array([fx, fy, fz])


def get_current_segment(x_pos):
    """Return the segment dict for the given X position."""
    for seg in SEGMENTS:
        if seg["x_min"] <= x_pos < seg["x_max"]:
            return seg
    if x_pos < 0:
        return SEGMENTS[0]
    return SEGMENTS[-1]


def _to_np(x):
    """Convert torch tensor (any device) to numpy. Pass through if already numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


class NumpyRobotWrapper:
    """Wraps SingleArticulation to always return numpy arrays.

    GPU PhysX (device='cuda:0') forces torch backend, making all robot APIs
    return CUDA tensors. This wrapper transparently converts them to numpy
    so the existing numpy-based policy code works unchanged.
    """

    def __init__(self, robot):
        # Store the real robot in __dict__ to avoid __getattr__ recursion
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
        # GPU PhysX requires CUDA tensors — convert numpy to avoid silent drops
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


# =============================================================================
# VELOCITY SMOOTHER
# =============================================================================

class VelocitySmoother:
    """Smooths velocity with exponential ramping. Decel faster than accel."""

    def __init__(self, accel_rate=2.0, decel_rate=4.0):
        self.accel_rate = accel_rate
        self.decel_rate = decel_rate
        self.current_vx = 0.0
        self.current_wz = 0.0

    def update(self, target_vx, target_wz, dt):
        """Ramp current velocities toward targets. Returns (vx, wz)."""
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


# =============================================================================
# DRIVE CONTROLLER
# =============================================================================

class DriveController:
    """Manages drive modes, velocity smoothing, and auto-nav."""

    def __init__(self):
        self.mode_index = 0
        self.speed_multiplier = 1.0
        self.smoother = VelocitySmoother()

        # Auto-nav state
        self.nav_waypoints = NAV_WAYPOINTS
        self.nav_index = 0
        self.nav_active = False

    @property
    def mode(self):
        return DRIVE_MODES[self.mode_index]

    def cycle_mode(self):
        """Advance to next drive mode."""
        self.mode_index = (self.mode_index + 1) % len(DRIVE_MODES)
        mode = self.mode
        self.smoother = VelocitySmoother(
            accel_rate=mode["accel_rate"],
            decel_rate=mode["decel_rate"],
        )
        if mode["name"] == "AUTO-NAV":
            self.nav_active = True
            self.nav_index = 0
        else:
            self.nav_active = False
        return mode

    def adjust_speed(self, delta):
        """Adjust speed multiplier."""
        self.speed_multiplier = np.clip(self.speed_multiplier + delta, 0.2, 2.0)

    def compute_command(self, key_state, sim_time, position, yaw, dt,
                        joy_fwd=0.0, joy_turn=0.0):
        """Convert key/stick state + robot state into [vx, 0.0, wz] command."""
        mode = self.mode

        # Emergency stop
        if key_state["stop"]:
            self.smoother.reset()
            return np.array([0.0, 0.0, 0.0])

        # Check if any movement input present
        any_wasd = (key_state["forward"] or key_state["backward"] or
                    key_state["left"] or key_state["right"])
        any_joy = abs(joy_fwd) > 0.01 or abs(joy_turn) > 0.01
        any_input = any_wasd or any_joy

        # Get target velocities
        if mode["name"] == "AUTO-NAV" and not any_input and self.nav_active:
            target_vx, target_wz = self._auto_nav_command(position, yaw)
        else:
            target_vx, target_wz = self._input_to_velocity(
                key_state, mode, joy_fwd, joy_turn)

        # Apply speed multiplier
        target_vx *= self.speed_multiplier
        target_wz *= self.speed_multiplier

        # Apply smoothing
        if mode["use_smoothing"]:
            vx, wz = self.smoother.update(target_vx, target_wz, dt)
        else:
            vx, wz = target_vx, target_wz

        # ES-003: Minimum forward speed while turning
        if abs(wz) > 0.05 and mode["min_forward_for_turn"] > 0:
            if 0 < vx < mode["min_forward_for_turn"]:
                vx = mode["min_forward_for_turn"]
            elif -mode["min_forward_for_turn"] < vx < 0:
                vx = -mode["min_forward_for_turn"]
            elif vx == 0 and mode["min_forward_for_turn"] > 0:
                vx = mode["min_forward_for_turn"]

        # Clamp to mode speed limits
        vx = np.clip(vx, -mode["max_vx_rev"], mode["max_vx"])
        wz = np.clip(wz, -mode["max_wz"], mode["max_wz"])

        return np.array([vx, 0.0, wz])  # vy MUST always be 0

    def _input_to_velocity(self, key_state, mode, joy_fwd=0.0, joy_turn=0.0):
        """Map keyboard + analog stick to raw target velocity."""
        # Analog stick input (proportional)
        if abs(joy_fwd) > 0.01 or abs(joy_turn) > 0.01:
            if joy_fwd >= 0:
                target_vx = joy_fwd * mode["max_vx"]
            else:
                target_vx = joy_fwd * mode["max_vx_rev"]
            target_wz = joy_turn * mode["max_wz"]
            return target_vx, target_wz

        # Keyboard input (binary)
        target_vx = 0.0
        target_wz = 0.0

        if key_state["forward"]:
            target_vx = mode["max_vx"]
        elif key_state["backward"]:
            target_vx = -mode["max_vx_rev"]

        if key_state["left"]:
            target_wz = mode["max_wz"]
        elif key_state["right"]:
            target_wz = -mode["max_wz"]

        return target_vx, target_wz

    def _auto_nav_command(self, position, yaw):
        """Generate velocity command to reach current waypoint."""
        if self.nav_index >= len(self.nav_waypoints):
            self.nav_index = 0

        target = self.nav_waypoints[self.nav_index]
        dist = distance_2d(position, target)

        if dist < 0.8:
            self.nav_index = (self.nav_index + 1) % len(self.nav_waypoints)
            target = self.nav_waypoints[self.nav_index]
            dist = distance_2d(position, target)
            print(f"  >> Waypoint reached! Next: ({target[0]:.0f}, {target[1]:.0f})")

        dx = target[0] - position[0]
        dy = target[1] - position[1]
        desired_yaw = np.arctan2(dy, dx)
        yaw_error = normalize_angle(desired_yaw - yaw)

        if abs(yaw_error) < 0.1:
            wz = 0.0
        else:
            wz = np.clip(yaw_error * 0.8, -self.mode["max_wz"], self.mode["max_wz"])

        alignment = np.cos(yaw_error)
        vx = self.mode["max_vx"] * max(0.5, alignment)

        return vx, wz


# =============================================================================
# ENVIRONMENT CREATION
# =============================================================================

def create_physics_material(stage, name, friction, restitution=0.05):
    """Create a named physics material with given friction."""
    parent = "/World/Physics"
    if not stage.GetPrimAtPath(parent).IsValid():
        UsdGeom.Xform.Define(stage, parent)

    path = f"{parent}/{name}"
    if not stage.GetPrimAtPath(path).IsValid():
        UsdShade.Material.Define(stage, path)

    prim = stage.GetPrimAtPath(path)
    phys = UsdPhysics.MaterialAPI.Apply(prim)
    phys.CreateStaticFrictionAttr(friction)
    phys.CreateDynamicFrictionAttr(friction * 0.875)
    phys.CreateRestitutionAttr(restitution)
    physx = PhysxSchema.PhysxMaterialAPI.Apply(prim)
    physx.CreateFrictionCombineModeAttr().Set("average")
    physx.CreateRestitutionCombineModeAttr().Set("min")
    return path


def create_course_enclosure(stage):
    """Create 100m x 15m walled obstacle course enclosure."""
    root = "/World/Course"
    UsdGeom.Xform.Define(stage, root)

    L = COURSE_LENGTH
    W = COURSE_WIDTH
    H = WALL_HEIGHT
    T = WALL_THICKNESS

    wall_configs = [
        ("north_wall", (L / 2, W + T / 2, H / 2), (L + T * 2, T, H)),
        ("south_wall", (L / 2, -T / 2, H / 2),    (L + T * 2, T, H)),
        ("east_wall",  (L + T / 2, W / 2, H / 2),  (T, W, H)),
        ("west_wall",  (-T / 2, W / 2, H / 2),     (T, W, H)),
    ]

    for name, pos, scale in wall_configs:
        wall_path = f"{root}/{name}"
        wall = UsdGeom.Cube.Define(stage, wall_path)
        wall.GetSizeAttr().Set(1.0)
        wall.AddTranslateOp().Set(Gf.Vec3d(*pos))
        wall.AddScaleOp().Set(Gf.Vec3d(*scale))
        wall.GetDisplayColorAttr().Set([(0.75, 0.75, 0.75)])
        UsdPhysics.CollisionAPI.Apply(wall.GetPrim())

    print(f"  Enclosure: {L:.0f}m x {W:.0f}m x {H:.0f}m walls")


def create_ground_segments(stage):
    """No-op: using default ground plane instead of custom segments."""
    pass


def create_segment_markers(stage):
    """Create thin colored floor overlays marking each segment."""
    root = "/World/SegmentMarkers"
    UsdGeom.Xform.Define(stage, root)

    for seg in SEGMENTS:
        center_x = (seg["x_min"] + seg["x_max"]) / 2
        width = seg["x_max"] - seg["x_min"]
        safe_name = seg["name"].replace(" ", "_").replace("+", "_").replace("-", "_")

        path = f"{root}/{safe_name}"
        floor = UsdGeom.Cube.Define(stage, path)
        floor.GetSizeAttr().Set(1.0)
        floor.AddTranslateOp().Set(Gf.Vec3d(center_x, COURSE_WIDTH / 2, 0.001))
        floor.AddScaleOp().Set(Gf.Vec3d(width, COURSE_WIDTH, 0.001))
        floor.GetDisplayColorAttr().Set([seg["color"]])

    print(f"  Segment markers: {len(SEGMENTS)} colored zones")


def create_grass_segment(stage, x_min, x_max, height_key):
    """Create grass floor overlay with friction material for a segment."""
    config = GRASS_HEIGHTS[height_key]
    mat_path = create_physics_material(
        stage, f"GrassMat_{height_key}", config["friction"])

    center_x = (x_min + x_max) / 2
    width = x_max - x_min
    green = 0.6 - (config["height"] * 0.3)

    floor_path = f"/World/Course/Grass_{height_key}_{int(x_min)}"
    floor = UsdGeom.Cube.Define(stage, floor_path)
    floor.GetSizeAttr().Set(1.0)
    floor.AddTranslateOp().Set(Gf.Vec3d(center_x, COURSE_WIDTH / 2, 0.003))
    floor.AddScaleOp().Set(Gf.Vec3d(width, COURSE_WIDTH, 0.001))
    floor.GetDisplayColorAttr().Set([(0.15, green, 0.1)])

    prim = stage.GetPrimAtPath(floor_path)
    UsdPhysics.CollisionAPI.Apply(prim)
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(UsdShade.Material.Get(stage, mat_path))


def create_stone_obstacles(stage, x_min, x_max, count=15):
    """Create random static stone obstacles in the given X range."""
    root = "/World/Course/Stones"
    UsdGeom.Xform.Define(stage, root)
    rng = np.random.RandomState(42)

    for i in range(count):
        size = rng.uniform(0.15, 0.4)
        px = rng.uniform(x_min + 1.0, x_max - 1.0)
        py = rng.uniform(1.5, COURSE_WIDTH - 1.5)
        pz = size / 2

        path = f"{root}/stone_{i}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(px, py, pz))
        xf.AddRotateYOp().Set(float(rng.uniform(0, 360)))
        sx = size
        sy = size * rng.uniform(0.6, 1.0)
        sz = size * rng.uniform(0.5, 1.0)
        xf.AddScaleOp().Set(Gf.Vec3d(sx, sy, sz))
        r = rng.uniform(0.42, 0.55)
        g = rng.uniform(0.40, 0.52)
        b = rng.uniform(0.36, 0.48)
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(r, g, b)])

        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())  # Static only

    print(f"  Stones: {count} static rocks in X=[{x_min},{x_max}]")


def create_steps(stage, x_start=35.0, half_length=5.0, peak_height=0.75,
                 num_steps=5):
    """Create ascending then descending steps spanning full course width.

    Each step is a solid cube from Z=0 up to step_height, so there are
    no hollow gaps underneath. Steps use static collision (no RigidBody).
    """
    root = "/World/Course/Steps"
    UsdGeom.Xform.Define(stage, root)

    step_run = half_length / num_steps    # 1.0m per step
    step_rise = peak_height / num_steps   # 0.15m per step

    step_mat = create_physics_material(stage, "StepMat", friction=0.6)

    for i in range(num_steps):
        # Ascending step
        step_height = step_rise * (i + 1)
        cx = x_start + step_run * i + step_run / 2
        cz = step_height / 2  # Cube fills 0 to step_height

        path = f"{root}/step_up_{i}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(cx, COURSE_WIDTH / 2, cz))
        xf.AddScaleOp().Set(Gf.Vec3d(step_run, COURSE_WIDTH, step_height))
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.60, 0.55, 0.42)])

        prim = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(UsdShade.Material.Get(stage, step_mat))

    for i in range(num_steps):
        # Descending step (mirror)
        step_height = step_rise * (num_steps - i)
        cx = x_start + half_length + step_run * i + step_run / 2
        cz = step_height / 2

        path = f"{root}/step_down_{i}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(cx, COURSE_WIDTH / 2, cz))
        xf.AddScaleOp().Set(Gf.Vec3d(step_run, COURSE_WIDTH, step_height))
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.58, 0.53, 0.40)])

        prim = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(UsdShade.Material.Get(stage, step_mat))

    print(f"  Steps: {num_steps} up + {num_steps} down, peak {peak_height}m "
          f"(rise per step: {step_rise:.2f}m)")


def create_rubble_pool(stage, x_min=50.0, x_max=60.0, wall_height=0.3,
                       debris_count=40):
    """Create walled rubble zone filled with dynamic debris at ground level.

    Low containment walls (0.3m) keep debris from scattering. Spot walks
    over the wall lip and through the loose material.
    """
    root = "/World/Course/RubblePool"
    UsdGeom.Xform.Define(stage, root)

    pool_mat = create_physics_material(stage, "PoolMat", friction=0.5)
    pool_len = x_max - x_min
    pool_cx = (x_min + x_max) / 2
    wt = 0.15  # wall thickness

    # Containment walls (low, 0.3m — Spot steps over them)
    wall_configs = [
        ("wall_west",  (x_min - wt / 2, COURSE_WIDTH / 2, wall_height / 2),
         (wt, COURSE_WIDTH - 2.0, wall_height)),
        ("wall_east",  (x_max + wt / 2, COURSE_WIDTH / 2, wall_height / 2),
         (wt, COURSE_WIDTH - 2.0, wall_height)),
        ("wall_south", (pool_cx, 1.0, wall_height / 2),
         (pool_len, wt, wall_height)),
        ("wall_north", (pool_cx, COURSE_WIDTH - 1.0, wall_height / 2),
         (pool_len, wt, wall_height)),
    ]

    for name, pos, scale in wall_configs:
        path = f"{root}/{name}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xf.AddScaleOp().Set(Gf.Vec3d(*scale))
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.50, 0.42, 0.35)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    # Dynamic debris sitting on ground plane (Z=0)
    rng = np.random.RandomState(99)
    debris_pieces = RUBBLE_PIECES[:6]  # Small and medium only

    for i in range(debris_count):
        piece = debris_pieces[rng.randint(0, len(debris_pieces))]
        hx, hy, hz = piece["half"]

        px = rng.uniform(x_min + 0.5, x_max - 0.5)
        py = rng.uniform(1.5, COURSE_WIDTH - 1.5)
        pz = hz + 0.01  # Sit on ground
        yaw_deg = rng.uniform(0, 360)

        path = f"{root}/debris_{i}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(px, py, pz))
        xf.AddRotateYOp().Set(float(yaw_deg))
        xf.AddScaleOp().Set(Gf.Vec3d(hx * 2, hy * 2, hz * 2))
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(*piece["color"])])

        prim = stage.GetPrimAtPath(path)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(piece["mass"])

        mat_path = f"{path}/material"
        mat = UsdShade.Material.Define(stage, mat_path)
        phys_mat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        phys_mat.CreateStaticFrictionAttr(0.7)
        phys_mat.CreateDynamicFrictionAttr(0.5)
        phys_mat.CreateRestitutionAttr(0.1)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

    print(f"  Rubble pool: {debris_count} debris pieces in walled zone "
          f"X=[{x_min},{x_max}] (walls {wall_height}m)")


def create_large_blocks(stage, x_min=65.0, x_max=75.0, count=20):
    """Create large static blocks that Spot must navigate around or step over.

    A 2m corridor along the centerline (Y=7.5) is kept clear.
    """
    root = "/World/Course/LargeBlocks"
    UsdGeom.Xform.Define(stage, root)
    rng = np.random.RandomState(123)

    center_y = COURSE_WIDTH / 2
    clear_half = 1.0  # 2m total corridor

    placed = 0
    for i in range(count * 3):  # Extra attempts for placement
        if placed >= count:
            break

        sx = rng.uniform(0.3, 0.8)
        sy = rng.uniform(0.3, 0.8)
        sz = rng.uniform(0.1, 0.6)

        px = rng.uniform(x_min + 0.5, x_max - 0.5)
        py = rng.uniform(1.5, COURSE_WIDTH - 1.5)

        # Enforce clear corridor along centerline
        if abs(py - center_y) < clear_half + sy / 2:
            continue

        pz = sz / 2
        yaw = rng.uniform(0, 360)

        path = f"{root}/block_{placed}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(px, py, pz))
        xf.AddRotateYOp().Set(float(yaw))
        xf.AddScaleOp().Set(Gf.Vec3d(sx, sy, sz))

        r = rng.uniform(0.40, 0.55)
        g = rng.uniform(0.37, 0.50)
        b = rng.uniform(0.33, 0.45)
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(r, g, b)])

        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())  # Static only
        placed += 1

    print(f"  Large blocks: {placed} static blocks in X=[{x_min},{x_max}] "
          f"(2m center corridor clear)")


def create_instability_field(stage, x_min=80.0, x_max=90.0, count=120):
    """Create small dynamic bricks that shift underfoot.

    Non-uniform cube scaling approximates trapezoidal shapes.
    Low friction so they slide and tumble when stepped on.
    """
    root = "/World/Course/InstabilityField"
    UsdGeom.Xform.Define(stage, root)
    rng = np.random.RandomState(456)

    for i in range(count):
        # Trapezoidal approximation: non-uniform scale
        sx = rng.uniform(0.05, 0.15)
        sy = rng.uniform(0.05, 0.12)
        sz = rng.uniform(0.03, 0.08)
        mass = rng.uniform(0.5, 3.0)

        px = rng.uniform(x_min + 0.3, x_max - 0.3)
        py = rng.uniform(0.5, COURSE_WIDTH - 0.5)
        pz = sz + 0.005
        yaw = rng.uniform(0, 360)

        path = f"{root}/brick_{i}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(px, py, pz))
        xf.AddRotateYOp().Set(float(yaw))
        xf.AddScaleOp().Set(Gf.Vec3d(sx, sy, sz))

        r = rng.uniform(0.50, 0.70)
        g = rng.uniform(0.30, 0.45)
        b = rng.uniform(0.20, 0.35)
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(r, g, b)])

        prim = stage.GetPrimAtPath(path)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(float(mass))

        mat_path = f"{path}/material"
        mat = UsdShade.Material.Define(stage, mat_path)
        phys_mat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        phys_mat.CreateStaticFrictionAttr(0.6)
        phys_mat.CreateDynamicFrictionAttr(0.4)
        phys_mat.CreateRestitutionAttr(0.05)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

    print(f"  Instability field: {count} dynamic bricks in X=[{x_min},{x_max}]")


def create_finish_markers(stage):
    """Create finish line visual markers (green pillars)."""
    root = "/World/Course/FinishLine"
    UsdGeom.Xform.Define(stage, root)

    for i, y_pos in enumerate([2.0, COURSE_WIDTH - 2.0]):
        path = f"{root}/pillar_{i}"
        cyl = UsdGeom.Cylinder.Define(stage, path)
        cyl.GetRadiusAttr().Set(0.2)
        cyl.GetHeightAttr().Set(1.5)
        cyl.AddTranslateOp().Set(Gf.Vec3d(90.0, y_pos, 0.75))
        cyl.GetDisplayColorAttr().Set([(0.0, 0.8, 0.0)])
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())

    print("  Finish line markers created")


def create_course_lighting(stage):
    """Create outdoor lighting for the course."""
    light_path = "/World/Lights"
    UsdGeom.Xform.Define(stage, light_path)

    dome = UsdLux.DomeLight.Define(stage, f"{light_path}/dome")
    dome.CreateIntensityAttr(600)

    sun = UsdLux.DistantLight.Define(stage, f"{light_path}/sun")
    sun.CreateIntensityAttr(5000)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    print("  Lighting created (outdoor)")


def build_obstacle_course(stage):
    """Build the complete obstacle course."""
    print()
    print("Building obstacle course...")
    print("-" * 50)

    create_course_enclosure(stage)
    create_ground_segments(stage)
    create_segment_markers(stage)
    create_course_lighting(stage)

    # Grass segments
    create_grass_segment(stage, 10, 20, "H1")
    print("  WARM-UP: H1 grass (friction 0.80)")
    create_grass_segment(stage, 20, 30, "H2")
    create_stone_obstacles(stage, 20, 30, count=15)

    # Steps
    create_steps(stage, x_start=35, half_length=5.0, peak_height=0.75,
                 num_steps=5)

    # Rubble pool
    create_rubble_pool(stage, x_min=50, x_max=60, debris_count=40)

    # Large blocks
    create_large_blocks(stage, x_min=65, x_max=75, count=20)

    # Instability field
    create_instability_field(stage, x_min=80, x_max=90, count=120)

    # Finish markers
    create_finish_markers(stage)

    print("-" * 50)
    print("Obstacle course complete!")
    print()


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

# Shared state
fpv_active = [False]
selfright_active = [False]
selfright_upright_timer = [0.0]
drive_controller = DriveController()

# Analog stick values from Xbox controller
joy_analog = {"fwd": 0.0, "turn": 0.0}


def on_keyboard_event(event, *args_ev, **kwargs):
    """Handle keyboard events."""
    is_pressed = (event.type == carb.input.KeyboardEventType.KEY_PRESS or
                  event.type == carb.input.KeyboardEventType.KEY_REPEAT)
    key = event.input

    # Movement keys (track press/release)
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

    # One-shot keys (only on KEY_PRESS, not REPEAT)
    elif event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if key == carb.input.KeyboardInput.LEFT_SHIFT or key == carb.input.KeyboardInput.RIGHT_SHIFT:
            mode = drive_controller.cycle_mode()
            print(f"\n  >> Mode: {mode['name']} - {mode['description']}")
        elif key == carb.input.KeyboardInput.H:
            # Show current segment info
            try:
                pos, _ = spot.robot.get_world_pose()
                seg = get_current_segment(pos[0])
                progress = min(100.0, (pos[0] / COURSE_LENGTH) * 100)
                print(f"\n  >> Segment: {seg['name']} - {seg['desc']} "
                      f"(X={seg['x_min']}-{seg['x_max']}m) | "
                      f"Progress: {progress:.1f}%")
            except Exception:
                pass
        elif key == carb.input.KeyboardInput.M:
            fpv_active[0] = not fpv_active[0]
            if fpv_active[0]:
                viewport_api.camera_path = fpv_cam_path
                print(f"\n  >> Camera: FPV (first-person onboard view)")
            else:
                viewport_api.camera_path = default_camera_path
                print(f"\n  >> Camera: ORBIT (default view)")
        elif key == carb.input.KeyboardInput.X:
            selfright_active[0] = not selfright_active[0]
            selfright_upright_timer[0] = 0.0
            if selfright_active[0]:
                print(f"\n  >> SELFRIGHT MODE: A=roll left, D=roll right (X to cancel)")
            else:
                print(f"\n  >> SELFRIGHT MODE: cancelled")
        elif key == carb.input.KeyboardInput.R:
            key_state["reset"] = True
        elif key == carb.input.KeyboardInput.UP:
            drive_controller.adjust_speed(0.1)
            print(f"\n  >> Speed: {drive_controller.speed_multiplier:.1f}x")
        elif key == carb.input.KeyboardInput.DOWN:
            drive_controller.adjust_speed(-0.1)
            print(f"\n  >> Speed: {drive_controller.speed_multiplier:.1f}x")
        elif key == carb.input.KeyboardInput.G:
            # Cycle gait mode (FLAT <-> ROUGH)
            if spot_rough is not None:
                gait_idx[0] = (gait_idx[0] + 1) % len(GAIT_MODES)
                gait = GAIT_MODES[gait_idx[0]]
                # Apply matching actuator gains
                if gait["policy_type"] == "rough":
                    spot_rough.apply_gains()
                else:
                    # Restore ALL flat policy PhysX properties
                    try:
                        av = spot_flat.robot._articulation_view
                        if 'kps' in flat_saved_props:
                            av.set_gains(
                                kps=flat_saved_props['kps'],
                                kds=flat_saved_props['kds']
                            )
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
                        print(f"[GAIT] Flat properties restored")
                    except Exception as e:
                        print(f"[GAIT] Could not restore flat properties: {e}")
                print(f"\n  >> Gait: {gait['name']} - {gait['description']}")
            else:
                print(f"\n  >> Gait cycling disabled (rough policy not available)")

    return True


# =============================================================================
# XBOX CONTROLLER SETUP (pygame)
# =============================================================================
joystick = None
joy_prev_buttons = []

if HAS_PYGAME:
    pygame.init()
    pygame.joystick.init()
    n_joy = pygame.joystick.get_count()
    if n_joy > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joy_prev_buttons = [False] * joystick.get_numbuttons()
        print(f"  Controller: {joystick.get_name()}")
        print(f"  Axes: {joystick.get_numaxes()}, "
              f"Buttons: {joystick.get_numbuttons()}, "
              f"Hats: {joystick.get_numhats()}")
    else:
        print("  No controller detected - keyboard only")
else:
    print("  pygame not installed - keyboard only")
    print("  Install: C:\\miniconda3\\envs\\isaaclab311\\python.exe "
          "-m pip install pygame")

print()
print("=" * 60)
print("  SPOT OBSTACLE COURSE")
print("=" * 60)
if joystick:
    print()
    print("  XBOX CONTROLLER:")
    print("    Left Stick     Forward/back + turn (analog)")
    print("    A              Cycle drive mode")
    print("    B              Toggle selfright")
    print("    Y              Reset to start")
    print("    LB             Toggle FPV camera")
    print("    D-Pad Up/Down  Speed multiplier +/-")
    print("    Back           Emergency stop")
print()
print("  KEYBOARD:")
print("    W/S       Forward / Backward")
print("    A/D       Turn left / Turn right")
print("    SPACE     Emergency stop")
print("    SHIFT     Cycle drive mode")
print("    M         Toggle FPV camera (first-person view)")
print("    X         Selfright mode (A/D to roll, auto-exits when upright)")
print("    H         Show current segment info")
print("    UP/DOWN   Adjust speed multiplier")
print("    R         Reset robot to start")
print("    ESC       Exit")
print()

# =============================================================================
# MAIN SETUP
# =============================================================================

# Create world (500Hz physics, 50Hz render)
# GPU PhysX pipeline (device="cuda:0") is REQUIRED for the trained rough terrain
# policy. Isaac Lab trains with GPU broadphase + GPU dynamics + fabric. Without
# this, the policy falls within 2 seconds due to different constraint resolution
# dynamics between CPU and GPU PhysX.  See ROUGH_POLICY_DEBUG_HANDOFF.md §15.
world = World(
    physics_dt=1.0 / 500.0,
    rendering_dt=10.0 / 500.0,
    stage_units_in_meters=1.0,
    device="cuda:0",
)
stage = omni.usd.get_context().get_stage()

# Ground plane: raw USD collision cube (NOT scene-registered).
# GPU PhysX forces torch backend, which crashes scene-registered GroundPlane
# in post_reset() due to numpy→torch mismatch.  A raw USD cube avoids this.
ground = UsdGeom.Cube.Define(stage, "/World/GroundPlane")
ground.GetSizeAttr().Set(1.0)
ground.AddTranslateOp().Set(Gf.Vec3d(50.0, 7.5, -0.005))
ground.AddScaleOp().Set(Gf.Vec3d(200.0, 200.0, 0.01))
ground.GetDisplayColorAttr().Set([(0.5, 0.5, 0.5)])
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

# Build the obstacle course on top of the ground plane
build_obstacle_course(stage)

# Spawn Spot with dual gait support (flat + rough policies)
print("\n============================================================")
print("  INITIALIZING SPOT ROBOT")
print("============================================================")

# Create flat terrain policy (always available)
spot_flat = SpotFlatTerrainPolicy(
    prim_path="/World/Spot",
    name="Spot",
    position=START_POS,
)
print(f"[GAIT] Flat terrain policy loaded")

# Create rough terrain policy (if trained model available)
spot_rough = None
if HAS_ROUGH_POLICY:
    try:
        # Shares flat policy's robot articulation — no duplicate prim
        spot_rough = SpotRoughTerrainPolicy(flat_policy=spot_flat)
        print(f"[GAIT] Rough terrain policy loaded (trained model)")
    except Exception as e:
        print(f"[GAIT] Failed to load rough policy: {e}")
        print(f"[GAIT] Gait cycling disabled - only FLAT gait available")
        spot_rough = None
else:
    print(f"[GAIT] Rough policy not available - gait cycling disabled")

# Set active policy (start with flat)
spot = spot_flat
gait_idx = [0]  # Mutable for physics callback

print(f"Spot created at ({START_POS[0]:.1f}, {START_POS[1]:.1f})")

# --- GPU PhysX compatibility patches (MUST be applied BEFORE world.reset()) ---
# GPU PhysX forces torch backend. Several Isaac Sim internals assume all data
# is torch tensors but receive numpy arrays or Python lists. Patch before reset
# so the physics initialization and scene post_reset don't crash.

# Patch 1: Torch tensor backend functions that crash on numpy/list inputs.
# GPU PhysX forces torch backend, but many Isaac Sim internals pass numpy arrays
# or Python lists through these functions.  Patch them to auto-convert.
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
        # Scalar fallback
        return [data] if not hasattr(data, '__iter__') else list(data)

    def _patched_clone_tensor(data, device):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32, device=device)
        else:
            data = data.to(device=device)
        return torch.clone(data)

    # Apply patches at the submodule level (where functions are defined)
    _torch_tensor.move_data = _patched_move_data
    _torch_tensor.to_list = _patched_to_list
    _torch_tensor.clone_tensor = _patched_clone_tensor
    # Also patch at the package level (some code imports from parent)
    for _name, _fn in [('move_data', _patched_move_data),
                        ('to_list', _patched_to_list),
                        ('clone_tensor', _patched_clone_tensor)]:
        if hasattr(_torch_backend, _name):
            setattr(_torch_backend, _name, _fn)
    print("[GPU] Patched torch tensor backend (move_data, to_list, clone_tensor)")
except Exception as e:
    print(f"[GPU] Could not patch torch tensor backend: {e}")

# Patch 2: tf_matrices_from_poses() — scene post_reset() passes numpy orientations
# to torch backend's transform function which calls .detach() on them
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
    print("[GPU] Patched tf_matrices_from_poses for numpy compatibility")
except Exception as e:
    print(f"[GPU] Could not patch tf_matrices_from_poses: {e}")

# Patch 3: Remove any scene-registered GroundPlane before reset.
# GroundPlane.post_reset() stores numpy default state that crashes with torch backend.
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

# Wrap the robot for GPU PhysX compatibility.
# GPU pipeline forces torch backend → all robot APIs return CUDA tensors.
# NumpyRobotWrapper converts them back to numpy so existing code works unchanged.
_real_robot = spot_flat.robot
spot_flat.robot = NumpyRobotWrapper(_real_robot)
print(f"[GPU] Robot wrapped for numpy compatibility (torch backend active)")

# Update rough policy's robot reference to use the wrapper too
if spot_rough is not None:
    spot_rough.robot = spot_flat.robot

# Save flat policy's actuator properties so we can restore them on gait switch.
# CRITICAL: Keep as CUDA tensors — GPU PhysX set_gains() silently ignores numpy.
flat_saved_props = {}
_GPU_DEV = "cuda:0"

def _ensure_cuda(x):
    """Ensure data is a CUDA tensor for GPU PhysX ArticulationView setters."""
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
    efforts = av.get_max_efforts()
    flat_saved_props['efforts'] = _ensure_cuda(efforts)
    try:
        frictions = av.get_friction_coefficients()
        flat_saved_props['frictions'] = _ensure_cuda(frictions)
    except Exception:
        pass
    try:
        armatures = av.get_armatures()
        flat_saved_props['armatures'] = _ensure_cuda(armatures)
    except Exception:
        pass
    try:
        max_vels = av.get_max_joint_velocities()
        flat_saved_props['max_vels'] = _ensure_cuda(max_vels)
    except Exception:
        pass
    try:
        pos_iters = av.get_solver_position_iteration_counts()
        vel_iters = av.get_solver_velocity_iteration_counts()
        flat_saved_props['pos_iters'] = pos_iters.clone() if isinstance(pos_iters, torch.Tensor) else pos_iters
        flat_saved_props['vel_iters'] = vel_iters.clone() if isinstance(vel_iters, torch.Tensor) else vel_iters
    except Exception:
        pass
    kp_val = float(flat_saved_props['kps'].flatten()[0])
    eff_val = float(flat_saved_props['efforts'].flatten()[0])
    print(f"[GAIT] Flat props saved as CUDA tensors (Kp={kp_val:.1f}, effort={eff_val:.1f})")
except Exception as e:
    print(f"[GAIT] Could not save flat properties: {e}")

if spot_rough is not None:
    spot_rough.initialize()  # Sets training default_pos
    # Compare flat and rough default positions
    flat_dp = np.array(_to_np(spot_flat.default_pos))
    rough_dp = np.array(_to_np(spot_rough.default_pos))
    print(f"[GAIT] Flat  default_pos: {np.array2string(flat_dp, precision=3)}")
    print(f"[GAIT] Rough default_pos: {np.array2string(rough_dp, precision=3)}")
    diff = rough_dp - flat_dp
    print(f"[GAIT] Diff (rough-flat): {np.array2string(diff, precision=3)}")
print("Spot initialized")

# Create FPV camera attached to Spot's body (auto-follows robot)
fpv_cam_path = "/World/Spot/body/fpv_camera"
fpv_cam = UsdGeom.Camera.Define(stage, fpv_cam_path)
fpv_cam.CreateFocalLengthAttr(18.0)
fpv_cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
cam_xform = UsdGeom.Xformable(fpv_cam.GetPrim())
cam_xform.AddTranslateOp().Set(Gf.Vec3d(0.4, 0.0, 0.15))
cam_xform.AddOrientOp().Set(Gf.Quatf(0.5, 0.5, -0.5, -0.5))
print("FPV camera created")

# Store default viewport camera path for toggling back
viewport_api = get_active_viewport()
default_camera_path = viewport_api.camera_path

# Setup keyboard
input_interface = carb.input.acquire_input_interface()
keyboard = omni.appwindow.get_default_app_window().get_keyboard()
keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

# Physics state
physics_ready = [False]
sim_time = [0.0]
last_hud_time = [0.0]
last_command = [np.array([0.0, 0.0, 0.0])]
recovery_timer = [0.0]
rollover_detected = [False]
last_segment_name = ["START"]
gait_switch_timer = [0.0]  # Counts down after gait switch

STABILIZE_TIME = 1.0
RECOVERY_STABILIZE = 1.5
GAIT_SWITCH_STABILIZE = 0.5  # seconds of zero-command after gait switch

# Waypoint markers for AUTO-NAV
for i, wp in enumerate(NAV_WAYPOINTS):
    marker_path = f"/World/wp_{i}"
    if not stage.GetPrimAtPath(marker_path).IsValid():
        marker = UsdGeom.Sphere.Define(stage, marker_path)
        marker.GetRadiusAttr().Set(0.15)
        marker.AddTranslateOp().Set(Gf.Vec3d(wp[0], wp[1], 0.15))
        marker.GetDisplayColorAttr().Set([(0.2, 0.4, 1.0)])


# =============================================================================
# PHYSICS CALLBACK (500Hz)
# =============================================================================

def on_physics_step(step_size):
    """Called at 500Hz. Reads keyboard state, commands Spot."""
    global spot

    # ES-010B: Skip first callback
    if not physics_ready[0]:
        physics_ready[0] = True
        return

    # Switch active policy based on gait mode
    if gait_idx[0] == 0 or spot_rough is None:
        if spot is not spot_flat and spot is spot_rough:
            # Switching FROM rough TO flat — reset rough state
            spot_rough.post_reset()
            gait_switch_timer[0] = GAIT_SWITCH_STABILIZE
        spot = spot_flat
    else:
        if spot is not spot_rough:
            # Switching FROM flat TO rough — reset rough state for clean start
            spot_rough.post_reset()
            gait_switch_timer[0] = GAIT_SWITCH_STABILIZE
        spot = spot_rough

    sim_time[0] += step_size

    # Stabilization period
    if sim_time[0] < STABILIZE_TIME:
        spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
        return

    # Handle reset (to start position)
    if key_state["reset"]:
        key_state["reset"] = False
        spot.robot.set_world_pose(
            position=START_POS,
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        drive_controller.smoother.reset()
        recovery_timer[0] = RECOVERY_STABILIZE
        last_segment_name[0] = "START"
        print("\n  >> Robot reset to start!")
        spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
        return

    # Physics-based self-right mode
    if selfright_active[0]:
        pos, quat = spot.robot.get_world_pose()
        roll, pitch = get_roll_pitch(quat)
        abs_roll = abs(roll)

        if abs_roll < np.radians(SELFRIGHT_UPRIGHT_DEG) and abs(pitch) < np.radians(SELFRIGHT_UPRIGHT_DEG):
            selfright_upright_timer[0] += step_size
            if selfright_upright_timer[0] >= SELFRIGHT_UPRIGHT_TIME:
                selfright_active[0] = False
                selfright_upright_timer[0] = 0.0
                drive_controller.smoother.reset()
                recovery_timer[0] = RECOVERY_STABILIZE
                print(f"\n  >> SELFRIGHT COMPLETE! Stabilizing...")
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
                lift_factor = (abs_roll - np.radians(45)) / np.radians(135)
                lift_factor = min(1.0, lift_factor)
                target_lift = SELFRIGHT_GROUND_LIFT * lift_factor
                vel = spot.robot.get_linear_velocity()
                if vel[2] < target_lift:
                    vel[2] = target_lift
                    spot.robot.set_linear_velocity(vel)
        else:
            damping_factor = 1.0 - SELFRIGHT_DAMPING * step_size
            damped_w = current_w * damping_factor
            spot.robot.set_angular_velocity(damped_w)

        return

    # Recovery stabilization
    if recovery_timer[0] > 0:
        recovery_timer[0] -= step_size
        spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
        return

    # Gait switch stabilization (zero command while policy settles)
    if gait_switch_timer[0] > 0:
        gait_switch_timer[0] -= step_size
        spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
        return

    # Get robot state
    pos, quat = spot.robot.get_world_pose()
    yaw = quat_to_yaw(quat)

    # Track rollover status for HUD
    rollover_detected[0] = is_rolled_over(quat)

    # Segment transition detection
    seg = get_current_segment(pos[0])
    if seg["name"] != last_segment_name[0]:
        last_segment_name[0] = seg["name"]
        print(f"\n  >> ENTERING: {seg['name']} - {seg['desc']}")

    # Compute velocity command (keyboard + analog stick)
    command = drive_controller.compute_command(
        key_state, sim_time[0], pos, yaw, step_size,
        joy_fwd=joy_analog["fwd"], joy_turn=joy_analog["turn"],
    )
    last_command[0] = command

    # Send to robot
    spot.forward(step_size, command)

    # HUD (every 0.5s)
    if sim_time[0] - last_hud_time[0] >= 0.5:
        last_hud_time[0] = sim_time[0]
        mode = drive_controller.mode
        yaw_deg = np.degrees(yaw)
        progress = min(100.0, (pos[0] / COURSE_LENGTH) * 100)

        # Speed bar
        bar_max = mode["max_vx"]
        if bar_max > 0:
            pct = min(1.0, abs(command[0]) / bar_max)
        else:
            pct = 0
        bar_len = 10
        filled = int(pct * bar_len)
        bar = "#" * filled + "-" * (bar_len - filled)

        spd_str = f"{drive_controller.speed_multiplier:.1f}x"
        cam_str = "FPV" if fpv_active[0] else "ORB"
        input_str = "PAD" if (joystick and (abs(joy_analog["fwd"]) > 0.01 or abs(joy_analog["turn"]) > 0.01)) else "KEY"
        gait_str = GAIT_MODES[gait_idx[0]]['name'] if spot_rough is not None else "FLAT"

        # Selfright / rollover status
        if selfright_active[0]:
            roll_val, pitch_val = get_roll_pitch(quat)
            status_str = f" SELFRIGHT(R:{np.degrees(roll_val):+.0f} P:{np.degrees(pitch_val):+.0f}) A/D=roll"
        elif rollover_detected[0]:
            status_str = " ROLLED! (X=selfright)"
        else:
            status_str = ""

        print(f"\r  [{sim_time[0]:6.1f}s] {mode['name']:>8s} {gait_str} {cam_str} {input_str} | "
              f"{seg['name']:>14s} {spd_str} | "
              f"Pos:({pos[0]:5.1f},{pos[1]:5.1f}) "
              f"Yaw:{yaw_deg:4.0f} | "
              f"[{bar}] vx={command[0]:+.2f} wz={command[2]:+.2f} | "
              f"Progress:{progress:5.1f}%{status_str}", end="     ")


world.add_physics_callback("spot_obstacle_course", on_physics_step)


# =============================================================================
# MAIN LOOP
# =============================================================================

print(f"  Mode:  {drive_controller.mode['name']} - {drive_controller.mode['description']}")
print(f"\n  Stabilizing for {STABILIZE_TIME}s, then controls active...")
print("  Click on simulation window to capture keyboard input!")
print("=" * 60 + "\n")

# Print course overview
print("  COURSE SEGMENTS:")
for i, seg in enumerate(SEGMENTS):
    print(f"    {i+1:2d}. {seg['name']:>14s}  X=[{seg['x_min']:3.0f}-{seg['x_max']:3.0f}m]  {seg['desc']}")
print()


def apply_deadzone(value, dz=XBOX_DEADZONE):
    """Apply deadzone and rescale."""
    if abs(value) < dz:
        return 0.0
    sign = 1.0 if value >= 0 else -1.0
    return sign * (abs(value) - dz) / (1.0 - dz)


try:
    while simulation_app.is_running() and not key_state["exit"]:
        world.step(render=True)

        # --- Read Xbox controller each render frame ---
        if joystick is not None:
            pygame.event.pump()

            # Analog sticks
            raw_fwd = -joystick.get_axis(XBOX_AXIS_FWD)
            raw_turn = -joystick.get_axis(XBOX_AXIS_TURN)
            joy_analog["fwd"] = apply_deadzone(raw_fwd)
            joy_analog["turn"] = apply_deadzone(raw_turn)

            # Button edge detection
            n_btns = joystick.get_numbuttons()
            for i in range(min(n_btns, len(joy_prev_buttons))):
                curr = joystick.get_button(i)
                if curr and not joy_prev_buttons[i]:
                    if i == XBOX_BTN_A:
                        mode = drive_controller.cycle_mode()
                        print(f"\n  >> Mode: {mode['name']} - {mode['description']}")
                    elif i == XBOX_BTN_B:
                        selfright_active[0] = not selfright_active[0]
                        selfright_upright_timer[0] = 0.0
                        if selfright_active[0]:
                            print(f"\n  >> SELFRIGHT MODE: Stick/A/D=roll (B to cancel)")
                        else:
                            print(f"\n  >> SELFRIGHT MODE: cancelled")
                    elif i == XBOX_BTN_Y:
                        key_state["reset"] = True
                    elif i == XBOX_BTN_LB:
                        fpv_active[0] = not fpv_active[0]
                        if fpv_active[0]:
                            viewport_api.camera_path = fpv_cam_path
                            print(f"\n  >> Camera: FPV (first-person onboard view)")
                        else:
                            viewport_api.camera_path = default_camera_path
                            print(f"\n  >> Camera: ORBIT (default view)")
                    elif i == XBOX_BTN_RB:
                        # Cycle gait mode (FLAT <-> ROUGH)
                        if spot_rough is not None:
                            gait_idx[0] = (gait_idx[0] + 1) % len(GAIT_MODES)
                            gait = GAIT_MODES[gait_idx[0]]
                            # Apply matching actuator gains
                            if gait["policy_type"] == "rough":
                                spot_rough.apply_gains()
                            else:
                                try:
                                    av = spot_flat.robot._articulation_view
                                    if 'kps' in flat_saved_props:
                                        av.set_gains(
                                            kps=flat_saved_props['kps'],
                                            kds=flat_saved_props['kds']
                                        )
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
                                    print(f"[GAIT] Flat properties restored")
                                except Exception as e:
                                    print(f"[GAIT] Could not restore flat properties: {e}")
                            print(f"\n  >> Gait: {gait['name']} - {gait['description']}")
                        else:
                            print(f"\n  >> Gait cycling disabled (rough policy not available)")
                    elif i == XBOX_BTN_BACK:
                        key_state["stop"] = True
                joy_prev_buttons[i] = curr

            # Release emergency stop when Back button released
            if n_btns > XBOX_BTN_BACK and not joystick.get_button(XBOX_BTN_BACK):
                key_state["stop"] = False

            # D-Pad (hat) for speed multiplier
            if joystick.get_numhats() > 0:
                hat_x, hat_y = joystick.get_hat(0)
                if hat_y == 1:
                    drive_controller.adjust_speed(0.002)
                elif hat_y == -1:
                    drive_controller.adjust_speed(-0.002)

except KeyboardInterrupt:
    print("\n\nStopping...")

# Cleanup
input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
if HAS_PYGAME:
    pygame.quit()

# Final stats
pos, _ = spot.robot.get_world_pose()
seg = get_current_segment(pos[0])
progress = min(100.0, (pos[0] / COURSE_LENGTH) * 100)
print("\n")
print("=" * 60)
print(f"  Session complete!")
print(f"  Final position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
print(f"  Final segment: {seg['name']} - {seg['desc']}")
print(f"  Course progress: {progress:.1f}%")
print(f"  Simulation time: {sim_time[0]:.1f}s")
print("=" * 60)

simulation_app.close()
print("Done.")
