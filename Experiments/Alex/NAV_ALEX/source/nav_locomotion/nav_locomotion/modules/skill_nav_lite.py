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
        cmd_smooth=0.3,
        lateral_damp=0.5,
        obstacles=None,
        obstacle_influence_radius=1.2,
        obstacle_repulse_gain=0.8,
        obstacle_tangent_bias=0.6,
        robot_radius=0.45,
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
        self.cmd_smooth = cmd_smooth
        self.lateral_damp = lateral_damp
        self.obstacles = obstacles or []
        self.obstacle_influence_radius = obstacle_influence_radius
        self.obstacle_repulse_gain = obstacle_repulse_gain
        self.obstacle_tangent_bias = obstacle_tangent_bias
        self.robot_radius = robot_radius
        self._prev_cmd = np.zeros(3, dtype=np.float32)

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

        wp = self.waypoints[self.wp_idx]["pos"]
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
            vx = 0.3
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
