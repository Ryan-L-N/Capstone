"""
Navigation controller and utility functions for quadruped robots.

Consolidates NavigationController + utility functions from grass phases 1-5.
Includes all lessons learned:
  - ES-001: Uses spot.forward() (not advance())
  - ES-002: Quaternion format [w, x, y, z] (scalar-first)
  - ES-003: Maintains forward motion while turning
  - ES-005: Dead zone (0.1 rad) and lower turn gain (0.8)
  - Stall detection from phase_2_friction_grass.py

Usage:
    from core import NavigationController, quat_to_yaw, distance_2d

    nav = NavigationController(
        start_pos=(1.0, 1.0),
        target_pos=(17.3, 8.1),
        forward_speed=1.0,
    )

    # In physics callback:
    pos, quat = spot.robot.get_world_pose()
    yaw = quat_to_yaw(quat)
    command = nav.update(sim_time, pos, yaw)
    spot.forward(step_size, np.array(command))
"""

import numpy as np


# =============================================================================
# Utility Functions
# =============================================================================

def quat_to_yaw(quat):
    """
    Convert quaternion [w, x, y, z] to yaw angle (rotation around Z).

    Isaac Sim 5.1.0 returns quaternions in [w, x, y, z] format (scalar-first).
    See LESSONS_LEARNED.md ES-002.
    """
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
    """Calculate 2D Euclidean distance between two positions."""
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def calculate_path_efficiency(path_length, optimal_length):
    """Calculate path efficiency as percentage (100% = perfectly straight)."""
    if path_length <= 0:
        return 0.0
    return min(100.0, (optimal_length / path_length) * 100.0)


# =============================================================================
# Navigation Controller
# =============================================================================

class NavigationController:
    """
    Point-to-point navigation with dead zone and proportional steering.

    State machine: STABILIZE -> NAVIGATE -> COMPLETE or FAILED

    All parameters are configurable via constructor kwargs with defaults
    matching the validated grass experiment configuration.
    """

    STATE_STABILIZE = 0
    STATE_NAVIGATE = 1
    STATE_COMPLETE = 2
    STATE_FAILED = 3

    def __init__(
        self,
        start_pos,
        target_pos,
        goal_threshold=0.5,
        stabilize_time=2.0,
        forward_speed=1.0,
        turn_rate=1.0,
        dead_zone=0.1,
        turn_gain=0.8,
        min_forward_ratio=0.5,
        timeout=120.0,
        fall_height=0.3,
        stall_timeout=None,
        stall_threshold=0.1,
    ):
        """
        Args:
            start_pos: Starting position (x, y) or (x, y, z)
            target_pos: Target position (x, y) or (x, y, z)
            goal_threshold: Distance to consider goal reached (meters)
            stabilize_time: Wait time before navigation starts (seconds)
            forward_speed: Maximum forward speed (m/s)
            turn_rate: Maximum turn rate (rad/s)
            dead_zone: Heading error below which no turning (radians, ~5.7 deg)
            turn_gain: Proportional gain for turning (0.8 prevents overshoot)
            min_forward_ratio: Minimum forward speed as ratio of max (0.5 = 50%)
            timeout: Maximum time before failure (seconds)
            fall_height: Robot height below which = fallen (meters)
            stall_timeout: Seconds without progress before stall failure (None=disabled)
            stall_threshold: Minimum forward progress to reset stall timer (meters)
        """
        self.start_pos = np.array(start_pos[:2], dtype=float)
        self.target_pos = np.array(target_pos[:2], dtype=float)
        self.goal_threshold = goal_threshold
        self.stabilize_time = stabilize_time
        self.forward_speed = forward_speed
        self.turn_rate = turn_rate
        self.dead_zone = dead_zone
        self.turn_gain = turn_gain
        self.min_forward_ratio = min_forward_ratio
        self.timeout = timeout
        self.fall_height = fall_height
        self.stall_timeout = stall_timeout
        self.stall_threshold = stall_threshold

        # State
        self.state = self.STATE_STABILIZE
        self.state_start_time = 0.0
        self.nav_start_time = None
        self.failure_reason = None

        # Path tracking
        self.path_length = 0.0
        self.last_pos = None
        self.optimal_distance = distance_2d(self.start_pos, self.target_pos)

        # Stall detection
        self._last_progress_pos = self.start_pos.copy()
        self._last_progress_time = 0.0

    def update(self, sim_time, position, yaw):
        """
        Update navigation and return velocity command.

        Args:
            sim_time: Current simulation time (seconds)
            position: Robot position (array-like, at least x,y)
            yaw: Robot heading (radians)

        Returns:
            [vx, vy, wz] command. vy is always 0 (quadruped constraint).
        """
        pos = np.array([position[0], position[1]])

        # Track path length
        if self.last_pos is not None:
            self.path_length += distance_2d(pos, self.last_pos)
        self.last_pos = pos.copy()

        # --- STABILIZE ---
        if self.state == self.STATE_STABILIZE:
            if sim_time - self.state_start_time >= self.stabilize_time:
                self.state = self.STATE_NAVIGATE
                self.nav_start_time = sim_time
                self.path_length = 0.0
                self._last_progress_pos = pos.copy()
                self._last_progress_time = sim_time
                print(f"[{sim_time:.2f}s] Navigation started")
            return [0.0, 0.0, 0.0]

        # --- COMPLETE / FAILED ---
        if self.state in (self.STATE_COMPLETE, self.STATE_FAILED):
            return [0.0, 0.0, 0.0]

        # --- NAVIGATE ---
        dist_to_goal = distance_2d(pos, self.target_pos)

        # Check goal reached
        if dist_to_goal < self.goal_threshold:
            self.state = self.STATE_COMPLETE
            print(f"[{sim_time:.2f}s] Goal reached! Distance: {dist_to_goal:.2f}m")
            return [0.0, 0.0, 0.0]

        # Check timeout
        if sim_time - self.nav_start_time >= self.timeout:
            self.mark_failed("Timeout")
            return [0.0, 0.0, 0.0]

        # Check stall
        if self.stall_timeout is not None:
            progress = distance_2d(pos, self._last_progress_pos)
            if progress > self.stall_threshold:
                self._last_progress_pos = pos.copy()
                self._last_progress_time = sim_time
            elif sim_time - self._last_progress_time > self.stall_timeout:
                self.mark_failed("Stalled")
                return [0.0, 0.0, 0.0]

        # Calculate desired heading
        dx = self.target_pos[0] - pos[0]
        dy = self.target_pos[1] - pos[1]
        desired_yaw = np.arctan2(dy, dx)
        yaw_error = normalize_angle(desired_yaw - yaw)

        # ES-005: Dead zone to prevent oscillation
        if abs(yaw_error) < self.dead_zone:
            wz = 0.0
        else:
            wz = np.clip(yaw_error * self.turn_gain, -self.turn_rate, self.turn_rate)

        # ES-003: Maintain forward motion while turning (quadrupeds need momentum)
        alignment = np.cos(yaw_error)
        vx = self.forward_speed * max(self.min_forward_ratio, alignment)

        return [vx, 0.0, wz]

    def check_fall(self, position):
        """Check if robot has fallen. Call from main loop with full 3D position."""
        if len(position) >= 3 and position[2] < self.fall_height:
            self.mark_failed("Robot fell")
            return True
        return False

    def mark_failed(self, reason):
        """Mark navigation as failed with a reason."""
        self.state = self.STATE_FAILED
        self.failure_reason = reason
        print(f"  [!] Navigation failed: {reason}")

    def is_complete(self):
        return self.state == self.STATE_COMPLETE

    def is_failed(self):
        return self.state == self.STATE_FAILED

    def is_done(self):
        return self.state in (self.STATE_COMPLETE, self.STATE_FAILED)

    def get_navigation_time(self, current_time):
        """Get time spent navigating (excluding stabilization)."""
        if self.nav_start_time is None:
            return 0.0
        return current_time - self.nav_start_time

    def get_path_efficiency(self):
        """Calculate path efficiency (100% = perfectly straight)."""
        return calculate_path_efficiency(self.path_length, self.optimal_distance)
