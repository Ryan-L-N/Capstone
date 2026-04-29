"""Skill-Nav Lite — waypoint -> [vx, vy, wz] P-controller.

Skill-Nav (arxiv 2506.21853) has no public code release. This adapter captures
its interface philosophy: consume a 2D base-frame waypoint, emit a 3-dim velocity
command. The low-level policy (SpotFlatTerrainPolicy / SpotRoughTerrainPolicy)
handles locomotion — no new RL training required.

Replaces ActorCriticCNN in cole_arena_eval.py at the nav-step boundary.
"""

import math
import numpy as np


def _normalize_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class SkillNavLiteNavigator:
    def __init__(
        self,
        waypoints,
        reach_threshold=0.5,
        kp_lin=1.0,
        kp_ang=2.0,
        vx_range=(-1.0, 3.0),
        vy_range=(-1.5, 1.5),
        wz_range=(-2.0, 2.0),
        max_lin_speed=1.5,
        yaw_first_threshold=math.radians(45),
        yaw_first_vx=0.3,
        cmd_smooth=0.3,
        lateral_damp=0.5,
        obstacles=None,
        obstacle_influence_radius=1.2,
        obstacle_repulse_gain=0.8,
        obstacle_tangent_bias=0.6,
        robot_radius=0.45,
        stuck_window=30,
        stuck_dist_threshold=0.3,
        escape_duration=15,
        escape_vx=1.5,
        escape_wz=1.5,
        skip_wp_after_escapes=3,
    ):
        self.waypoints = waypoints
        self.wp_idx = 0
        self.reach_threshold = reach_threshold
        self.kp_lin = kp_lin
        self.kp_ang = kp_ang
        self.vx_range = vx_range
        self.vy_range = vy_range
        self.wz_range = wz_range
        self.max_lin_speed = max_lin_speed
        self.yaw_first_threshold = yaw_first_threshold
        self.yaw_first_vx = yaw_first_vx
        self.cmd_smooth = cmd_smooth
        self.lateral_damp = lateral_damp
        self.obstacles = obstacles or []
        self.obstacle_influence_radius = obstacle_influence_radius
        self.obstacle_repulse_gain = obstacle_repulse_gain
        self.obstacle_tangent_bias = obstacle_tangent_bias
        self.robot_radius = robot_radius
        self._prev_cmd = np.zeros(3, dtype=np.float32)
        self.stuck_window = int(stuck_window)
        self.stuck_dist_threshold = float(stuck_dist_threshold)
        self.escape_duration = int(escape_duration)
        self.escape_vx = float(escape_vx)
        self.escape_wz = float(escape_wz)
        self._pos_history = []
        self._escape_ticks_left = 0
        self._escape_wz_sign = 1.0
        self.skip_wp_after_escapes = int(skip_wp_after_escapes)
        self._escape_count_for_wp = 0
        self._last_wp_for_escape = -1

    def set_obstacles(self, obstacles):
        self.obstacles = list(obstacles) if obstacles else []

    def _repulsive_force_world(self, robot_xy, goal_unit):
        fx, fy = 0.0, 0.0
        if not self.obstacles:
            return fx, fy
        R = self.obstacle_influence_radius
        gx, gy = goal_unit
        for ox, oy, osize in self.obstacles:
            ohalf = osize * 0.5
            dx = robot_xy[0] - ox
            dy = robot_xy[1] - oy
            d = math.sqrt(dx * dx + dy * dy)
            clearance = d - ohalf - self.robot_radius
            if clearance >= R or d < 1e-6:
                continue
            eff = max(clearance, 0.05)
            strength = self.obstacle_repulse_gain * (1.0 / eff - 1.0 / R) / (eff * eff)
            rx, ry = dx / d, dy / d
            tx1, ty1 = -ry, rx
            if tx1 * gx + ty1 * gy < 0:
                tx1, ty1 = ry, -rx
            t_bias = self.obstacle_tangent_bias
            fx += strength * ((1.0 - t_bias) * rx + t_bias * tx1)
            fy += strength * ((1.0 - t_bias) * ry + t_bias * ty1)
        return fx, fy

    def reset(self, waypoints=None):
        if waypoints is not None:
            self.waypoints = waypoints
        self.wp_idx = 0
        self._prev_cmd[:] = 0.0
        self._pos_history = []
        self._escape_ticks_left = 0
        self._escape_wz_sign = 1.0
        self._escape_count_for_wp = 0
        self._last_wp_for_escape = -1

    def replace_waypoints(self, new_waypoints):
        """Swap waypoint list mid-episode (for online replanning). Resets only
        the per-waypoint escape counters and stuck history; preserves the
        velocity-smoothing state so the robot doesn't jerk on a replan."""
        self.waypoints = list(new_waypoints)
        self.wp_idx = 0
        self._pos_history = []
        self._escape_ticks_left = 0
        self._escape_count_for_wp = 0
        self._last_wp_for_escape = -1

    @property
    def done(self):
        return self.wp_idx >= len(self.waypoints)

    @property
    def current_waypoint(self):
        if self.done:
            return None
        return self.waypoints[self.wp_idx]["pos"]

    @property
    def current_label(self):
        if self.done:
            return None
        return self.waypoints[self.wp_idx]["label"]

    def check_reached(self, robot_xy):
        if self.done:
            return False
        wp = self.waypoints[self.wp_idx]["pos"]
        d = math.sqrt((robot_xy[0] - wp[0]) ** 2 + (robot_xy[1] - wp[1]) ** 2)
        if d < self.reach_threshold:
            self.wp_idx += 1
            return True
        return False

    def get_velocity_command(self, robot_xy, robot_yaw):
        if self.done:
            cmd = np.zeros(3, dtype=np.float32)
            self._prev_cmd = cmd
            return cmd

        self._pos_history.append((float(robot_xy[0]), float(robot_xy[1])))
        if len(self._pos_history) > self.stuck_window:
            self._pos_history.pop(0)

        wp = self.waypoints[self.wp_idx]["pos"]

        if self._escape_ticks_left > 0:
            self._escape_ticks_left -= 1
            heading_to_wp = math.atan2(wp[1] - robot_xy[1], wp[0] - robot_xy[0])
            err = _normalize_angle(heading_to_wp - robot_yaw)
            wz_esc = self._escape_wz_sign * self.escape_wz
            cmd = np.array(
                [self.escape_vx, 0.0, float(np.clip(wz_esc, *self.wz_range))],
                dtype=np.float32,
            )
            cmd = self.cmd_smooth * cmd + (1.0 - self.cmd_smooth) * self._prev_cmd
            self._prev_cmd = cmd
            return cmd

        if len(self._pos_history) >= self.stuck_window:
            x0, y0 = self._pos_history[0]
            max_d = max(
                math.sqrt((x - x0) ** 2 + (y - y0) ** 2)
                for (x, y) in self._pos_history
            )
            if max_d < self.stuck_dist_threshold:
                if self._last_wp_for_escape != self.wp_idx:
                    self._last_wp_for_escape = self.wp_idx
                    self._escape_count_for_wp = 0
                self._escape_count_for_wp += 1
                if self._escape_count_for_wp > self.skip_wp_after_escapes:
                    print(f"[NAV] Skipping unreachable WP {self.current_label} after "
                          f"{self._escape_count_for_wp} failed escapes")
                    self.wp_idx += 1
                    self._escape_count_for_wp = 0
                    self._pos_history = []
                    if self.done:
                        cmd = np.zeros(3, dtype=np.float32)
                        self._prev_cmd = cmd
                        return cmd
                    wp = self.waypoints[self.wp_idx]["pos"]
                else:
                    self._escape_ticks_left = self.escape_duration
                    heading_to_wp = math.atan2(wp[1] - robot_xy[1], wp[0] - robot_xy[0])
                    err = _normalize_angle(heading_to_wp - robot_yaw)
                    self._escape_wz_sign = -1.0 if err > 0 else 1.0
                    self._pos_history = []

        dx_w = wp[0] - robot_xy[0]
        dy_w = wp[1] - robot_xy[1]

        goal_norm = math.sqrt(dx_w * dx_w + dy_w * dy_w)
        gx_att = (dx_w / goal_norm) if goal_norm > 1e-6 else 0.0
        gy_att = (dy_w / goal_norm) if goal_norm > 1e-6 else 0.0
        fx_rep, fy_rep = self._repulsive_force_world(robot_xy, (gx_att, gy_att))
        blend_x = gx_att + fx_rep
        blend_y = gy_att + fy_rep
        blend_norm = math.sqrt(blend_x * blend_x + blend_y * blend_y)
        if blend_norm > 1e-6:
            ux_w = blend_x / blend_norm
            uy_w = blend_y / blend_norm
        else:
            ux_w, uy_w = gx_att, gy_att

        cos_y = math.cos(-robot_yaw)
        sin_y = math.sin(-robot_yaw)
        ux_b = cos_y * ux_w - sin_y * uy_w
        uy_b = sin_y * ux_w + cos_y * uy_w

        target_heading = math.atan2(blend_y, blend_x)
        heading_err = _normalize_angle(target_heading - robot_yaw)

        desired_speed = min(self.max_lin_speed, self.kp_lin * goal_norm)
        heading_scale = max(0.2, math.cos(heading_err))
        if abs(heading_err) > self.yaw_first_threshold:
            vx = self.yaw_first_vx
            vy = 0.0
            wz = self.kp_ang * heading_err
        else:
            vx = desired_speed * ux_b * heading_scale
            vy = desired_speed * uy_b * self.lateral_damp
            wz = self.kp_ang * heading_err

        vx = float(np.clip(vx, *self.vx_range))
        vy = float(np.clip(vy, *self.vy_range))
        wz = float(np.clip(wz, *self.wz_range))

        cmd = np.array([vx, vy, wz], dtype=np.float32)
        cmd = self.cmd_smooth * cmd + (1.0 - self.cmd_smooth) * self._prev_cmd
        self._prev_cmd = cmd
        return cmd
