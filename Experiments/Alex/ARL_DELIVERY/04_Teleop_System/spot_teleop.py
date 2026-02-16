"""
Spot Teleoperation - Keyboard & Xbox Controller in Grass Terrain
=================================================================

Drive Spot with WASD keyboard or an Xbox controller through the grass
experiment room. Switch drive modes, rubble, grass, and camera views.

XBOX CONTROLLER:
  Left Stick Y       Forward / backward (analog)
  Left Stick X       Turn left / right (analog)
  A                  Cycle drive mode
  B                  Toggle selfright mode
  X                  Cycle rubble level
  Y                  Reset position
  LB                 Toggle FPV camera
  RB                 Cycle grass height
  D-Pad Up/Down      Speed multiplier +/-
  Back               Emergency stop
  (In selfright: Left Stick X = roll direction)

KEYBOARD:
  W / S         Forward / Backward
  A / D         Turn left / Turn right
  SPACE         Emergency stop
  SHIFT         Cycle drive mode: MANUAL -> SMOOTH -> PATROL -> AUTO-NAV
  H             Cycle rubble level: CLEAR -> LIGHT -> MODERATE -> HEAVY
  M             Toggle FPV camera (first-person onboard view)
  0-4           Switch grass height: 0=None, 1=H1, 2=H2, 3=H3, 4=H4
  UP / DOWN     Adjust speed multiplier
  X             Toggle selfright mode (physics-based rollover recovery)
                  In selfright mode: A/D = roll left/right, auto-exits when upright
  R             Reset robot position
  ESC           Exit

DRIVE MODES:
  MANUAL    - Instant response, arcade feel
  SMOOTH    - Velocity ramping, real robot inertia
  PATROL    - Slow & careful, optimized for tall grass
  AUTO-NAV  - Autonomous waypoints, WASD overrides

Isaac Sim 5.1.0 + Isaac Lab 2.3.0
"""

import numpy as np
import argparse
import sys
import os

# Parse args BEFORE SimulationApp (required)
parser = argparse.ArgumentParser(description="Spot Teleoperation in Grass Terrain")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--grass", choices=["H0", "H1", "H2", "H3", "H4"], default="H0",
                    help="Initial grass height (default: H0 = no grass)")
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

# Room (60ft x 30ft)
ROOM_LENGTH = 18.3   # meters (X-axis)
ROOM_WIDTH = 9.1     # meters (Y-axis)
ROOM_HEIGHT = 3.0    # meters (Z-axis)

# Grass zone (center of room)
GRASS_ZONE = {
    "x_min": 3.0,  "x_max": 15.3,
    "y_min": 2.0,  "y_max": 7.1,
}

# Grass height configs (from phase_2_friction_grass.py)
GRASS_HEIGHTS = {
    "H0": {"height": 0.0,  "friction": 0.80, "description": "No grass (baseline)"},
    "H1": {"height": 0.1,  "friction": 0.80, "description": "Ankle (0.1m)"},
    "H2": {"height": 0.3,  "friction": 0.85, "description": "Knee (0.3m)"},
    "H3": {"height": 0.5,  "friction": 0.90, "description": "Mid-body (0.5m)"},
    "H4": {"height": 0.7,  "friction": 0.95, "description": "Shoulder (0.7m)"},
}

# Robot start position
START_POS = np.array([1.0, 1.0, 0.7])

# AUTO-NAV waypoints (room corners + center)
NAV_WAYPOINTS = [
    (16.0, 7.5),
    (16.0, 1.5),
    (1.5, 7.5),
    (9.0, 4.5),
    (1.5, 1.5),
]

# Rubble piece definitions (shape, half-extents in meters, mass in kg, color)
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
    # Big boxes (~2x2ft = 0.6m, ~50 lbs = 23 kg)
    {"name": "crate",  "half": (0.30, 0.30,  0.15),  "mass": 23.0,
     "color": (0.40, 0.35, 0.30)},
    {"name": "pallet", "half": (0.25, 0.25,  0.10),  "mass": 18.0,
     "color": (0.58, 0.50, 0.35)},
]

# Rubble level configs: (level_name, description, piece_counts)
# piece_counts = (small, medium, large, big) - how many of each size class
RUBBLE_LEVELS = [
    ("CLEAR",    "No rubble",                (0,  0,  0,  0)),
    ("LIGHT",    "Scattered debris",          (6,  4,  2,  1)),
    ("MODERATE", "Cluttered terrain",         (12, 8,  5,  3)),
    ("HEAVY",    "Dense rubble field",        (20, 15, 10, 5)),
]

# Self-right (physics-based rollover recovery)
# Sim-to-real: phase-dependent torque simulating leg ground-reaction forces.
# Strong push when upside down (legs push off ground), gravity assists past 90 deg,
# gentle correction near upright, natural damping when no input.
SELFRIGHT_ROLL_ACCEL = 12.0       # rad/s^2 peak torque (near upside-down)
SELFRIGHT_MAX_ROLL_VEL = 2.5      # rad/s deliberate roll speed (~3s for 180 deg)
SELFRIGHT_GROUND_LIFT = 0.8       # m/s max upward force (only when heavily rolled)
SELFRIGHT_DAMPING = 3.0           # angular velocity damping coeff (ground friction)
SELFRIGHT_UPRIGHT_DEG = 35.0      # degrees - upright when roll & pitch < this
SELFRIGHT_UPRIGHT_TIME = 0.3      # seconds upright before auto-exiting selfright

# Xbox controller mapping (XInput on Windows via pygame)
XBOX_AXIS_TURN = 0       # Left Stick X  (-1=left, +1=right)
XBOX_AXIS_FWD = 1        # Left Stick Y  (-1=up/fwd, +1=down/back)
XBOX_DEADZONE = 0.12     # 12% dead zone (Xbox sticks drift)
XBOX_BTN_A = 0           # Cycle drive mode
XBOX_BTN_B = 1           # Toggle selfright
XBOX_BTN_X = 2           # Cycle rubble
XBOX_BTN_Y = 3           # Reset position
XBOX_BTN_LB = 4          # Toggle FPV camera
XBOX_BTN_RB = 5          # Cycle grass height
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
        "description": "Slow & careful (tall grass)",
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
    # Rotation matrix column 0 = body X-axis in world
    fx = 1.0 - 2.0 * (y * y + z * z)
    fy = 2.0 * (x * y + w * z)
    fz = 2.0 * (x * z - w * y)
    return np.array([fx, fy, fz])


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
        # Decel when moving toward zero, accel otherwise
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
        # Start auto-nav if entering that mode
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
        """Convert key/stick state + robot state into [vx, 0.0, wz] command.

        joy_fwd:  -1..1 analog forward/backward (from controller stick)
        joy_turn: -1..1 analog turn left(+)/right(-) (from controller stick)
        """
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
        """Map keyboard + analog stick to raw target velocity.

        Analog stick takes priority over keyboard when non-zero.
        """
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
            self.nav_index = 0  # Loop waypoints

        target = self.nav_waypoints[self.nav_index]
        dist = distance_2d(position, target)

        # Reached waypoint? Advance to next
        if dist < 0.8:
            self.nav_index = (self.nav_index + 1) % len(self.nav_waypoints)
            target = self.nav_waypoints[self.nav_index]
            dist = distance_2d(position, target)
            print(f"  >> Waypoint reached! Next: ({target[0]:.0f}, {target[1]:.0f})")

        # Calculate heading error
        dx = target[0] - position[0]
        dy = target[1] - position[1]
        desired_yaw = np.arctan2(dy, dx)
        yaw_error = normalize_angle(desired_yaw - yaw)

        # ES-005: Dead zone
        if abs(yaw_error) < 0.1:
            wz = 0.0
        else:
            wz = np.clip(yaw_error * 0.8, -self.mode["max_wz"], self.mode["max_wz"])

        # ES-003: Forward speed based on alignment
        alignment = np.cos(yaw_error)
        vx = self.mode["max_vx"] * max(0.5, alignment)

        return vx, wz


# =============================================================================
# ENVIRONMENT SETUP (from phase_2_friction_grass.py)
# =============================================================================

def create_room(stage):
    """Create 60ft x 30ft room with walls."""
    room_path = "/World/Room"
    if stage.GetPrimAtPath(room_path).IsValid():
        return room_path

    UsdGeom.Xform.Define(stage, room_path)

    wall_configs = [
        ("north_wall", (ROOM_LENGTH/2, ROOM_WIDTH, ROOM_HEIGHT/2), (ROOM_LENGTH, 0.1, ROOM_HEIGHT)),
        ("south_wall", (ROOM_LENGTH/2, 0, ROOM_HEIGHT/2), (ROOM_LENGTH, 0.1, ROOM_HEIGHT)),
        ("east_wall",  (ROOM_LENGTH, ROOM_WIDTH/2, ROOM_HEIGHT/2), (0.1, ROOM_WIDTH, ROOM_HEIGHT)),
        ("west_wall",  (0, ROOM_WIDTH/2, ROOM_HEIGHT/2), (0.1, ROOM_WIDTH, ROOM_HEIGHT)),
    ]

    for name, pos, scale in wall_configs:
        wall_path = f"{room_path}/{name}"
        wall = UsdGeom.Cube.Define(stage, wall_path)
        wall.AddTranslateOp().Set(Gf.Vec3d(*pos))
        wall.AddScaleOp().Set(Gf.Vec3d(*scale))
        wall.GetDisplayColorAttr().Set([(0.8, 0.8, 0.8)])

    print(f"Room created: {ROOM_LENGTH}m x {ROOM_WIDTH}m x {ROOM_HEIGHT}m")
    return room_path


def create_room_lighting(stage):
    """Create room lighting (dome + sun)."""
    light_path = "/World/Lights"
    if stage.GetPrimAtPath(light_path).IsValid():
        return light_path

    UsdGeom.Xform.Define(stage, light_path)

    dome = UsdLux.DomeLight.Define(stage, f"{light_path}/dome")
    dome.CreateIntensityAttr(500)

    sun = UsdLux.DistantLight.Define(stage, f"{light_path}/sun")
    sun.CreateIntensityAttr(3000)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    print("Lighting created")
    return light_path


def create_grass_material(stage, friction_value):
    """Create PhysX grass material with specified friction."""
    parent_path = "/World/Physics"
    material_path = f"{parent_path}/GrassMaterial"

    if not stage.GetPrimAtPath(parent_path).IsValid():
        UsdGeom.Xform.Define(stage, parent_path)

    if stage.GetPrimAtPath(material_path).IsValid():
        material_prim = stage.GetPrimAtPath(material_path)
    else:
        material = UsdShade.Material.Define(stage, material_path)
        material_prim = material.GetPrim()

    physics_material = UsdPhysics.MaterialAPI.Apply(material_prim)
    physics_material.CreateStaticFrictionAttr(friction_value)
    physics_material.CreateDynamicFrictionAttr(friction_value * 0.875)
    physics_material.CreateRestitutionAttr(0.05)

    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
    physx_material.CreateFrictionCombineModeAttr().Set("average")
    physx_material.CreateRestitutionCombineModeAttr().Set("min")

    return material_path


def create_grass_zone_visual(stage, height_key):
    """Create or update green floor overlay for grass zone."""
    grass_path = "/World/GrassZone"

    existing = stage.GetPrimAtPath(grass_path)
    if existing.IsValid():
        stage.RemovePrim(grass_path)

    if height_key == "H0":
        return None  # No grass visual for baseline

    UsdGeom.Xform.Define(stage, grass_path)

    x_min, x_max = GRASS_ZONE["x_min"], GRASS_ZONE["x_max"]
    y_min, y_max = GRASS_ZONE["y_min"], GRASS_ZONE["y_max"]
    width = x_max - x_min
    depth = y_max - y_min
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    floor_path = f"{grass_path}/floor"
    floor = UsdGeom.Cube.Define(stage, floor_path)
    floor.AddTranslateOp().Set(Gf.Vec3d(center_x, center_y, 0.002))
    floor.AddScaleOp().Set(Gf.Vec3d(width, depth, 0.001))

    # Darker green for taller grass
    height = GRASS_HEIGHTS[height_key]["height"]
    green_intensity = 0.6 - (height * 0.3)
    floor.GetDisplayColorAttr().Set([(0.15, green_intensity, 0.1)])

    # Bind grass physics material
    floor_prim = stage.GetPrimAtPath(floor_path)
    UsdPhysics.CollisionAPI.Apply(floor_prim)
    material_binding = UsdShade.MaterialBindingAPI.Apply(floor_prim)
    material_binding.Bind(UsdShade.Material.Get(stage, "/World/Physics/GrassMaterial"))

    return grass_path


def switch_grass(stage, height_key):
    """Switch grass height at runtime: update friction + visual."""
    config = GRASS_HEIGHTS[height_key]

    # Update material friction
    create_grass_material(stage, config["friction"])

    # Update visual
    create_grass_zone_visual(stage, height_key)

    print(f"  >> Grass: {height_key} - {config['description']} (friction={config['friction']:.2f})")


# =============================================================================
# RUBBLE SYSTEM
# =============================================================================

def clear_rubble(stage):
    """Remove all rubble pieces from the scene."""
    rubble_root = "/World/Rubble"
    prim = stage.GetPrimAtPath(rubble_root)
    if prim.IsValid():
        stage.RemovePrim(rubble_root)


def spawn_rubble(stage, level_index):
    """Spawn rubble at the given level. Clears existing rubble first."""
    clear_rubble(stage)

    level_name, description, counts = RUBBLE_LEVELS[level_index]
    if level_name == "CLEAR":
        print(f"\n  >> Rubble: CLEAR - No rubble")
        return

    rubble_root = "/World/Rubble"
    UsdGeom.Xform.Define(stage, rubble_root)

    # Size classes map to piece indices:
    #   small=0-1, medium=2-3, large=4-5, big=6-7
    size_classes = [
        (0, 2),   # small: indices 0-1
        (2, 4),   # medium: indices 2-3
        (4, 6),   # large: indices 4-5
        (6, 8),   # big: indices 6-7
    ]

    # Spawn area = grass zone with margin so pieces don't overlap walls
    x_min = GRASS_ZONE["x_min"] + 0.5
    x_max = GRASS_ZONE["x_max"] - 0.5
    y_min = GRASS_ZONE["y_min"] + 0.5
    y_max = GRASS_ZONE["y_max"] - 0.5

    # Keep rubble away from robot start position
    robot_clear_radius = 2.0

    piece_id = 0
    total_pieces = 0

    for class_idx, count in enumerate(counts):
        idx_lo, idx_hi = size_classes[class_idx]

        for _ in range(count):
            # Pick random piece from this size class
            piece = RUBBLE_PIECES[np.random.randint(idx_lo, idx_hi)]
            hx, hy, hz = piece["half"]

            # Random position (avoid robot start area)
            for _attempt in range(20):
                px = np.random.uniform(x_min, x_max)
                py = np.random.uniform(y_min, y_max)
                if distance_2d((px, py), START_POS[:2]) > robot_clear_radius:
                    break

            pz = hz + 0.01  # Sit on ground with tiny gap to prevent jitter

            # Random yaw rotation
            yaw_deg = np.random.uniform(0, 360)

            prim_path = f"{rubble_root}/piece_{piece_id}"
            piece_id += 1

            # Create cube geometry
            cube = UsdGeom.Cube.Define(stage, prim_path)
            cube.GetSizeAttr().Set(1.0)  # Unit cube, scaled by xformOps

            xform = UsdGeom.Xformable(cube.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3d(px, py, pz))
            xform.AddRotateYOp().Set(yaw_deg)
            xform.AddScaleOp().Set(Gf.Vec3d(hx * 2, hy * 2, hz * 2))

            # Color
            cube.GetDisplayColorAttr().Set([Gf.Vec3f(*piece["color"])])

            # Physics: rigid body + collision + mass
            prim_ref = stage.GetPrimAtPath(prim_path)
            UsdPhysics.RigidBodyAPI.Apply(prim_ref)
            UsdPhysics.CollisionAPI.Apply(prim_ref)

            mass_api = UsdPhysics.MassAPI.Apply(prim_ref)
            mass_api.CreateMassAttr(piece["mass"])

            # Friction so pieces don't slide forever
            mat_path = f"{prim_path}/material"
            mat = UsdShade.Material.Define(stage, mat_path)
            mat_prim = mat.GetPrim()
            phys_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
            phys_mat.CreateStaticFrictionAttr(0.7)
            phys_mat.CreateDynamicFrictionAttr(0.5)
            phys_mat.CreateRestitutionAttr(0.1)
            mat_bind = UsdShade.MaterialBindingAPI.Apply(prim_ref)
            mat_bind.Bind(mat)

            total_pieces += 1

    print(f"\n  >> Rubble: {level_name} - {description} ({total_pieces} pieces)")


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

# Shared state for grass, rubble & controller (mutated by keyboard handler)
current_grass = [args.grass]  # Mutable wrapper
current_rubble = [0]  # Index into RUBBLE_LEVELS (0 = CLEAR)
pending_rubble = [False]  # Flag to spawn rubble from physics callback
fpv_active = [False]  # FPV camera toggle state
selfright_active = [False]  # Physics-based selfright mode (X toggles)
selfright_upright_timer = [0.0]  # Time robot has been upright during selfright
drive_controller = DriveController()

# Analog stick values from Xbox controller (written in main loop, read in physics callback)
joy_analog = {"fwd": 0.0, "turn": 0.0}
# Grass height list for RB cycling
GRASS_KEYS = ["H0", "H1", "H2", "H3", "H4"]
current_grass_idx = [GRASS_KEYS.index(args.grass)]


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
            current_rubble[0] = (current_rubble[0] + 1) % len(RUBBLE_LEVELS)
            pending_rubble[0] = True
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
        # Grass height keys
        elif key == carb.input.KeyboardInput.KEY_0:
            current_grass[0] = "H0"
        elif key == carb.input.KeyboardInput.KEY_1:
            current_grass[0] = "H1"
        elif key == carb.input.KeyboardInput.KEY_2:
            current_grass[0] = "H2"
        elif key == carb.input.KeyboardInput.KEY_3:
            current_grass[0] = "H3"
        elif key == carb.input.KeyboardInput.KEY_4:
            current_grass[0] = "H4"

    return True


# =============================================================================
# MAIN SETUP
# =============================================================================

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
print("  SPOT TELEOPERATION - Grass Terrain")
print("=" * 60)
if joystick:
    print()
    print("  XBOX CONTROLLER:")
    print("    Left Stick     Forward/back + turn (analog)")
    print("    A              Cycle drive mode")
    print("    B              Toggle selfright")
    print("    X              Cycle rubble")
    print("    Y              Reset position")
    print("    LB             Toggle FPV camera")
    print("    RB             Cycle grass height")
    print("    D-Pad Up/Down  Speed multiplier +/-")
    print("    Back           Emergency stop")
print()
print("  KEYBOARD:")
print("    W/S       Forward / Backward")
print("    A/D       Turn left / Turn right")
print("    SPACE     Emergency stop")
print("    SHIFT     Cycle drive mode")
print("    H         Cycle rubble: CLEAR > LIGHT > MODERATE > HEAVY")
print("    M         Toggle FPV camera (first-person view)")
print("    X         Selfright mode (A/D to roll, auto-exits when upright)")
print("    0-4       Switch grass height")
print("    UP/DOWN   Adjust speed multiplier")
print("    R         Reset robot to start")
print("    ESC       Exit")
print()

# Create world (500Hz physics, 50Hz render)
world = World(
    physics_dt=1.0 / 500.0,
    rendering_dt=10.0 / 500.0,
    stage_units_in_meters=1.0,
)
stage = omni.usd.get_context().get_stage()

# Ground plane
world.scene.add_default_ground_plane(
    z_position=0,
    name="default_ground_plane",
    prim_path="/World/defaultGroundPlane",
    static_friction=0.5,
    dynamic_friction=0.5,
    restitution=0.01,
)

# Room, lighting, initial grass
create_room(stage)
create_room_lighting(stage)
create_grass_material(stage, GRASS_HEIGHTS[args.grass]["friction"])
if args.grass != "H0":
    create_grass_zone_visual(stage, args.grass)

# Spawn Spot
spot = SpotFlatTerrainPolicy(
    prim_path="/World/Spot",
    name="Spot",
    position=START_POS,
)
print(f"Spot created at ({START_POS[0]:.1f}, {START_POS[1]:.1f})")

# Initialize
world.reset()
spot.initialize()
spot.robot.set_joints_default_state(spot.default_pos)
print("Spot initialized")

# Create FPV camera attached to Spot's body (auto-follows robot)
fpv_cam_path = "/World/Spot/body/fpv_camera"
fpv_cam = UsdGeom.Camera.Define(stage, fpv_cam_path)
fpv_cam.CreateFocalLengthAttr(18.0)  # Wide FOV for immersive FPV
fpv_cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
cam_xform = UsdGeom.Xformable(fpv_cam.GetPrim())
cam_xform.AddTranslateOp().Set(Gf.Vec3d(0.4, 0.0, 0.15))  # Front of head, slightly up
# USD cameras look down -Z; rotate so camera looks along +X (forward) with +Z up
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
applied_grass = [args.grass]  # Track which grass is actually applied
recovery_timer = [0.0]  # Seconds remaining in self-right stabilization
rollover_detected = [False]  # Current rollover state for HUD

STABILIZE_TIME = 1.0
RECOVERY_STABILIZE = 1.5  # Seconds to stabilize after self-right

# Waypoint markers for AUTO-NAV
for i, wp in enumerate(NAV_WAYPOINTS):
    marker_path = f"/World/wp_{i}"
    if not stage.GetPrimAtPath(marker_path).IsValid():
        marker = UsdGeom.Sphere.Define(stage, marker_path)
        marker.GetRadiusAttr().Set(0.15)
        marker.AddTranslateOp().Set(Gf.Vec3d(wp[0], wp[1], 0.15))
        marker.GetDisplayColorAttr().Set([(0.2, 0.4, 1.0)])  # Blue


# =============================================================================
# PHYSICS CALLBACK (500Hz)
# =============================================================================

def on_physics_step(step_size):
    """Called at 500Hz. Reads keyboard state, commands Spot."""
    # ES-010B: Skip first callback
    if not physics_ready[0]:
        physics_ready[0] = True
        return

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
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # w,x,y,z facing +X
        )
        drive_controller.smoother.reset()
        recovery_timer[0] = RECOVERY_STABILIZE
        print("\n  >> Robot reset to start!")
        spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
        return

    # Physics-based self-right mode (sim-to-real)
    # Simulates leg ground-reaction forces: phase-dependent torque + natural damping.
    # Phase 1 (>120 deg): Strong push — legs pushing off ground to initiate roll
    # Phase 2 (60-120 deg): Moderate — gravity starting to assist
    # Phase 3 (30-60 deg): Light — gravity does most work, just guide it
    # Phase 4 (<30 deg): Brake — prevent over-rotation, settle upright
    if selfright_active[0]:
        pos, quat = spot.robot.get_world_pose()
        roll, pitch = get_roll_pitch(quat)
        abs_roll = abs(roll)

        # Check if robot is upright (roll & pitch below threshold)
        if abs_roll < np.radians(SELFRIGHT_UPRIGHT_DEG) and abs(pitch) < np.radians(SELFRIGHT_UPRIGHT_DEG):
            selfright_upright_timer[0] += step_size
            if selfright_upright_timer[0] >= SELFRIGHT_UPRIGHT_TIME:
                # Robot is upright and stable — exit selfright mode
                selfright_active[0] = False
                selfright_upright_timer[0] = 0.0
                drive_controller.smoother.reset()
                recovery_timer[0] = RECOVERY_STABILIZE
                print(f"\n  >> SELFRIGHT COMPLETE! Stabilizing...")
                spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
                return
        else:
            selfright_upright_timer[0] = 0.0

        # Determine roll direction from A/D keys or left stick X
        roll_dir = 0.0
        if abs(joy_analog["turn"]) > 0.1:
            roll_dir = joy_analog["turn"]  # Analog: proportional roll
        elif key_state["left"]:
            roll_dir = 1.0   # Roll left (positive around forward axis)
        elif key_state["right"]:
            roll_dir = -1.0  # Roll right (negative around forward axis)

        # Body forward axis in world frame (roll axis)
        forward = quat_forward_axis(quat)
        forward_norm = forward / (np.linalg.norm(forward) + 1e-8)
        current_w = spot.robot.get_angular_velocity()

        if roll_dir != 0.0:
            # Phase-dependent gain: strong when upside down, light near upright
            # Simulates how real legs generate more force when fully extended on ground
            if abs_roll > np.radians(120):
                phase_gain = 1.0    # Upside down: full push (legs against ground)
            elif abs_roll > np.radians(60):
                phase_gain = 0.6    # Mid-roll: gravity starting to help
            elif abs_roll > np.radians(30):
                phase_gain = 0.25   # Past tipping point: gravity does most work
            else:
                phase_gain = 0.1    # Nearly upright: gentle correction only

            # Incremental torque-like angular acceleration
            accel = SELFRIGHT_ROLL_ACCEL * phase_gain
            roll_impulse = forward_norm * roll_dir * accel * step_size
            new_w = current_w + roll_impulse

            # Cap roll speed for deliberate motion
            w_mag = np.linalg.norm(new_w)
            if w_mag > SELFRIGHT_MAX_ROLL_VEL:
                new_w = new_w * (SELFRIGHT_MAX_ROLL_VEL / w_mag)

            spot.robot.set_angular_velocity(new_w)

            # Upward ground-reaction force — only when heavily rolled
            # Simulates legs pushing down on ground creating upward reaction
            # Scales with roll angle: max when upside down, zero near upright
            if abs_roll > np.radians(45):
                lift_factor = (abs_roll - np.radians(45)) / np.radians(135)
                lift_factor = min(1.0, lift_factor)
                target_lift = SELFRIGHT_GROUND_LIFT * lift_factor
                vel = spot.robot.get_linear_velocity()
                if vel[2] < target_lift:
                    vel[2] = target_lift
                    spot.robot.set_linear_velocity(vel)

        else:
            # No A/D input — apply natural damping (ground friction slows the roll)
            # In reality, friction and gravity resist free rotation
            damping_factor = 1.0 - SELFRIGHT_DAMPING * step_size
            damped_w = current_w * damping_factor
            spot.robot.set_angular_velocity(damped_w)

        # Don't call spot.forward() during selfright — policy disabled
        return

    # Recovery stabilization (zero velocity while settling after selfright/reset)
    if recovery_timer[0] > 0:
        recovery_timer[0] -= step_size
        spot.forward(step_size, np.array([0.0, 0.0, 0.0]))
        return

    # Check if grass needs switching (deferred from keyboard handler)
    if current_grass[0] != applied_grass[0]:
        switch_grass(stage, current_grass[0])
        applied_grass[0] = current_grass[0]

    # Check if rubble needs spawning (deferred from keyboard handler)
    if pending_rubble[0]:
        pending_rubble[0] = False
        spawn_rubble(stage, current_rubble[0])

    # Get robot state
    pos, quat = spot.robot.get_world_pose()
    yaw = quat_to_yaw(quat)

    # Track rollover status for HUD
    rollover_detected[0] = is_rolled_over(quat)

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
        dist = distance_2d(pos, START_POS[:2])
        yaw_deg = np.degrees(yaw)

        # Speed bar
        bar_max = mode["max_vx"]
        if bar_max > 0:
            pct = min(1.0, abs(command[0]) / bar_max)
        else:
            pct = 0
        bar_len = 10
        filled = int(pct * bar_len)
        bar = "#" * filled + "-" * (bar_len - filled)

        grass_str = applied_grass[0]
        rubble_str = RUBBLE_LEVELS[current_rubble[0]][0]
        spd_str = f"{drive_controller.speed_multiplier:.1f}x"
        cam_str = "FPV" if fpv_active[0] else "ORB"
        input_str = "PAD" if (joystick and (abs(joy_analog["fwd"]) > 0.01 or abs(joy_analog["turn"]) > 0.01)) else "KEY"

        # Selfright / rollover status
        if selfright_active[0]:
            roll_val, pitch_val = get_roll_pitch(quat)
            status_str = f" SELFRIGHT(R:{np.degrees(roll_val):+.0f} P:{np.degrees(pitch_val):+.0f}) A/D=roll"
        elif rollover_detected[0]:
            status_str = " ROLLED! (X=selfright)"
        else:
            status_str = ""

        print(f"\r  [{sim_time[0]:6.1f}s] {mode['name']:>8s} {cam_str} {input_str} | "
              f"{grass_str} {rubble_str:>8s} {spd_str} | "
              f"Pos:({pos[0]:5.1f},{pos[1]:5.1f}) "
              f"Yaw:{yaw_deg:4.0f} | "
              f"[{bar}] vx={command[0]:+.2f} wz={command[2]:+.2f} | "
              f"Dist:{dist:5.1f}m{status_str}", end="     ")


world.add_physics_callback("spot_teleop", on_physics_step)


# =============================================================================
# MAIN LOOP
# =============================================================================

print(f"\n  Grass: {args.grass} - {GRASS_HEIGHTS[args.grass]['description']}")
print(f"  Mode:  {drive_controller.mode['name']} - {drive_controller.mode['description']}")
print(f"\n  Stabilizing for {STABILIZE_TIME}s, then controls active...")
print("  Click on simulation window to capture keyboard input!")
print("=" * 60 + "\n")

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

            # Analog sticks (written here, read in 500Hz physics callback)
            raw_fwd = -joystick.get_axis(XBOX_AXIS_FWD)   # Invert: stick up = forward
            raw_turn = -joystick.get_axis(XBOX_AXIS_TURN)  # Invert: stick left = positive turn
            joy_analog["fwd"] = apply_deadzone(raw_fwd)
            joy_analog["turn"] = apply_deadzone(raw_turn)

            # Button edge detection (press only, not hold)
            n_btns = joystick.get_numbuttons()
            for i in range(min(n_btns, len(joy_prev_buttons))):
                curr = joystick.get_button(i)
                if curr and not joy_prev_buttons[i]:
                    # A = cycle drive mode
                    if i == XBOX_BTN_A:
                        mode = drive_controller.cycle_mode()
                        print(f"\n  >> Mode: {mode['name']} - {mode['description']}")
                    # B = toggle selfright
                    elif i == XBOX_BTN_B:
                        selfright_active[0] = not selfright_active[0]
                        selfright_upright_timer[0] = 0.0
                        if selfright_active[0]:
                            print(f"\n  >> SELFRIGHT MODE: Stick/A/D=roll (B to cancel)")
                        else:
                            print(f"\n  >> SELFRIGHT MODE: cancelled")
                    # X = cycle rubble
                    elif i == XBOX_BTN_X:
                        current_rubble[0] = (current_rubble[0] + 1) % len(RUBBLE_LEVELS)
                        pending_rubble[0] = True
                    # Y = reset position
                    elif i == XBOX_BTN_Y:
                        key_state["reset"] = True
                    # LB = toggle FPV camera
                    elif i == XBOX_BTN_LB:
                        fpv_active[0] = not fpv_active[0]
                        if fpv_active[0]:
                            viewport_api.camera_path = fpv_cam_path
                            print(f"\n  >> Camera: FPV (first-person onboard view)")
                        else:
                            viewport_api.camera_path = default_camera_path
                            print(f"\n  >> Camera: ORBIT (default view)")
                    # RB = cycle grass height
                    elif i == XBOX_BTN_RB:
                        current_grass_idx[0] = (current_grass_idx[0] + 1) % len(GRASS_KEYS)
                        current_grass[0] = GRASS_KEYS[current_grass_idx[0]]
                    # Back = emergency stop
                    elif i == XBOX_BTN_BACK:
                        key_state["stop"] = True
                joy_prev_buttons[i] = curr

            # Release emergency stop when Back button released
            if n_btns > XBOX_BTN_BACK and not joystick.get_button(XBOX_BTN_BACK):
                key_state["stop"] = False

            # D-Pad (hat) for speed multiplier
            if joystick.get_numhats() > 0:
                hat_x, hat_y = joystick.get_hat(0)
                if hat_y == 1:   # D-Pad Up
                    drive_controller.adjust_speed(0.002)  # Gradual per frame
                elif hat_y == -1:  # D-Pad Down
                    drive_controller.adjust_speed(-0.002)

except KeyboardInterrupt:
    print("\n\nStopping...")

# Cleanup
input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
if HAS_PYGAME:
    pygame.quit()

# Final stats
pos, _ = spot.robot.get_world_pose()
dist = distance_2d(pos, START_POS[:2])
print("\n")
print("=" * 60)
print(f"  Session complete!")
print(f"  Final position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
print(f"  Distance from start: {dist:.2f}m")
print(f"  Simulation time: {sim_time[0]:.1f}s")
print("=" * 60)

simulation_app.close()
print("Done.")
