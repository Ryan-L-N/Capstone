"""
Navigation Environment
======================
Hierarchical RL environment for Spot waypoint navigation.
High-level policy outputs velocity commands to SpotFlatTerrainPolicy.

Author: Cole (MS for Autonomy Project)  
Date: March 2026
"""

import sys
import os
import math
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class NavigationEnvironment:
    """
    Navigation environment for hierarchical RL.
    
    Integrates with:
    - Isaac Sim World
    - SpotFlatTerrainPolicy (low-level locomotion)
    - Obstacle spawning from Baseline_Environment.py
    """
    
    def __init__(self, world, stage, spot, config_path: str):
        """
        Initialize environment.
        
        Args:
            world: Isaac Sim World instance
            stage: USD stage
            spot: SpotFlatTerrainPolicy instance
            config_path: path to nav_config.yaml
        """
        self.world = world
        self.stage = stage
        self.spot = spot
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Arena geometry
        self.arena_radius = self.config['arena']['radius']
        self.arena_center = np.array(self.config['arena']['center'])
        self.obstacle_buffer = self.config['arena']['obstacle_buffer']
        
        # Robot settings
        self.start_pos = np.array(self.config['robot']['start_position'])
        self.start_ori = np.array(self.config['robot']['start_orientation'])
        self.fall_threshold = self.config['robot']['fall_threshold']
        
        # Action ranges
        self.vx_range = tuple(self.config['action']['vx_range'])
        self.vy_range = tuple(self.config['action']['vy_range'])
        self.omega_range = tuple(self.config['action']['omega_range'])
        
        # Observation settings
        self.num_obstacle_rays = self.config['observation']['obstacle_rays']
        self.ray_length = self.config['observation']['obstacle_ray_length']
        
        # Waypoint settings
        self.total_waypoints = self.config['waypoints']['total_count']
        self.capture_radius = self.config['waypoints']['capture_radius']
        self.boundary_margin = self.config['waypoints']['boundary_margin']
        
        # Scoring
        self.initial_points = self.config['scoring']['initial_points']
        self.time_decay = self.config['scoring']['time_decay_per_sec']
        self.waypoint_bonus_points = self.config['scoring']['waypoint_bonus']
        
        # Reward weights
        self.reward_time_penalty = self.config['reward']['time_penalty']
        self.reward_waypoint = self.config['reward']['waypoint_capture']
        self.reward_timeout = self.config['reward'].get('timeout_penalty', -50.0)
        self.reward_progress_alpha = self.config['reward']['progress_shaping']
        self.reward_distance_weight = self.config['reward'].get('distance_reward', 0.0)
        self.reward_heading_weight = self.config['reward'].get('heading_reward', 0.0)
        self.reward_speed_weight = self.config['reward'].get('speed_reward', 0.0)
        self.reward_wrong_direction_penalty = self.config['reward'].get('wrong_direction_penalty', 0.0)
        self.reward_fall = self.config['reward']['fall_penalty']
        self.reward_boundary = self.config['reward']['boundary_penalty']
        
        # Physics
        self.control_dt = 1.0 / self.config['physics']['control_hz']
        
        # Curriculum
        self.curriculum_stages = self.config['curriculum']['stages']
        self.success_window = self.config['curriculum']['success_window']
        self.success_threshold = self.config['curriculum']['success_threshold']
        
        # Episode state
        self.current_stage = 0
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.score = self.initial_points
        self.episode_time = 0.0
        self.episode_start_time = 0.0
        self.waypoints_captured = 0
        self.last_waypoint_distance = None
        self.episode_active = False
        
        # Curriculum tracking
        self.episode_history = []  # Last N episodes for success calculation
        self.total_episodes = 0
        
        # Obstacle manager - simplified for now
        # TODO: Integrate full ObstacleManager from Baseline_Environment
        self.obstacle_mgr = None
        
        # Control timing
        self.last_control_time = 0.0
        
        # Current command for physics callback
        self.current_command = np.zeros(3, dtype=np.float32)  # [vx, vy, omega]
        
    def reset(self, stage_id: Optional[int] = None) -> np.ndarray:
        """
        Reset environment for new episode.
        
        Args:
            stage_id: curriculum stage to use (None = current stage)
        
        Returns:
            initial observation
        """
        if stage_id is not None:
            self.current_stage = stage_id
        
        stage = self.curriculum_stages[self.current_stage]
        
        # Reset robot
        self.spot.robot.set_world_pose(self.start_pos, self.start_ori)
        self.spot.robot.set_linear_velocity(np.zeros(3))
        self.spot.robot.set_angular_velocity(np.zeros(3))
        self.spot.robot.set_joints_default_state(self.spot.default_pos)
        
        # Reset episode state
        self.score = self.initial_points
        self.episode_time = 0.0
        self.episode_start_time = self.world.current_time
        self.waypoints_captured = 0
        self.current_waypoint_idx = 0
        self.last_waypoint_distance = None
        self.episode_active = True
        self.last_control_time = self.world.current_time
        self.current_command = np.zeros(3, dtype=np.float32)  # Reset command
        
        # Generate waypoints (if stage has them)
        self.waypoints = []
        if stage['waypoint_distance_first'] is not None:
            self.waypoints = self._generate_waypoints(
                first_dist=stage['waypoint_distance_first'],
                subsequent_dist=stage['waypoint_distance_subsequent']
            )
        
        # Spawn obstacles (stages 6-8 only)
        # Temporarily disabled until obstacle system integrated
        if self.obstacle_mgr is not None:
            self.obstacle_mgr.remove_prims()
            total_coverage = (stage['obstacles_light_coverage'] + 
                             stage['obstacles_heavy_coverage'] + 
                             stage['obstacles_small_coverage'])
            if total_coverage > 0:
                self._spawn_obstacles(stage)
        
        # Get initial observation
        obs = self._get_observation()
        
        return obs
    
    def apply_command(self, dt: float):
        """
        Apply current command to Spot (called by physics callback).
        
        Args:
            dt: physics timestep (typically 0.002s at 500Hz)
        """
        if self.episode_active:
            self.spot.forward(dt, self.current_command)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one control step.
        
        Args:
            action: normalized action in [-1, 1], shape (3,) [vx, vy, omega]
        
        Returns:
            obs: next observation
            reward: reward for this step
            done: episode termination flag
            info: additional information
        """
        # Scale action to velocity command and store for physics callback
        self.current_command = self._scale_action(action)
        
        # Step physics sim until control time elapsed
        # Physics callback will apply self.current_command at each physics step
        target_time = self.last_control_time + self.control_dt
        while self.world.current_time < target_time:
            self.world.step(render=False)
        
        self.last_control_time = self.world.current_time
        
        # Update episode time
        self.episode_time = self.world.current_time - self.episode_start_time
        
        # Update score
        self.score -= self.time_decay * self.control_dt
        
        # Get robot state
        pos, ori = self.spot.robot.get_world_pose()
        robot_x, robot_y, robot_z = pos[0], pos[1], pos[2]
        
        # Check termination conditions
        done = False
        fall = robot_z < self.fall_threshold
        out_of_bounds = not self._inside_arena(robot_x, robot_y)
        timeout = False
        
        stage = self.curriculum_stages[self.current_stage]
        if stage['max_time'] is not None:
            timeout = self.episode_time >= stage['max_time']
        else:
            timeout = self.score <= 0
        
        # Calculate reward
        reward = self.reward_time_penalty  # constant time penalty
        
        # Speed reward - encourage fast movement when path is clear
        if self.reward_speed_weight > 0:
            vel = self.spot.robot.get_linear_velocity()
            forward_speed = np.linalg.norm(vel[:2])  # horizontal velocity magnitude
            # Only reward speed if no close obstacles (> 2m clear ahead)
            # Get robot heading for obstacle check
            yaw = self._quat_to_yaw(ori)
            obstacle_dists = self._raycast_obstacles(robot_x, robot_y, yaw)
            # Check forward-facing obstacle rays (indices around front)
            min_obstacle_dist = min(obstacle_dists[6:10]) if len(obstacle_dists) >= 10 else 5.0  # front 4 rays
            if min_obstacle_dist > 2.0:  # path is clear
                # Reward proportional to speed, scaled 0-1 for speeds 0-5 m/s
                speed_reward = self.reward_speed_weight * min(forward_speed / 5.0, 1.0)
                reward += speed_reward
        
        # Check waypoint capture
        waypoint_captured = False
        if self.waypoints and self.current_waypoint_idx < len(self.waypoints):
            wp = self.waypoints[self.current_waypoint_idx]
            dist_to_wp = np.sqrt((robot_x - wp[0])**2 + (robot_y - wp[1])**2)
            
            if dist_to_wp < self.capture_radius:
                # Waypoint captured!
                waypoint_captured = True
                self.waypoints_captured += 1
                self.current_waypoint_idx += 1
                self.score += self.waypoint_bonus_points
                reward += self.reward_waypoint
                self.last_waypoint_distance = None
                
                # Check if all waypoints captured
                if self.current_waypoint_idx >= len(self.waypoints):
                    done = True
            else:
                # Enhanced reward shaping (Stages 2-5 only)
                if stage['use_progress_shaping']:
                    # 1. Progress shaping - reward for getting closer, PENALTY for moving away
                    if self.last_waypoint_distance is not None:
                        progress = self.last_waypoint_distance - dist_to_wp
                        progress_reward = self.reward_progress_alpha * progress
                        # Apply extra penalty if moving in wrong direction
                        if progress < 0 and self.reward_wrong_direction_penalty > 0:
                            progress_reward *= self.reward_wrong_direction_penalty
                        reward += progress_reward
                    
                    # 2. Distance-based reward - continuous guidance (inverse distance)
                    # Reward decreases with distance, max at 1m, zero at 20m
                    distance_reward = self.reward_distance_weight * max(0, 1.0 - dist_to_wp / 20.0)
                    reward += distance_reward
                    
                    # 3. Heading reward - reward for facing toward waypoint
                    if self.reward_heading_weight > 0:
                        # Get robot heading
                        vel = self.spot.robot.get_linear_velocity()
                        if np.linalg.norm(vel[:2]) > 0.1:  # only if moving
                            vel_direction = vel[:2] / np.linalg.norm(vel[:2])
                            waypoint_direction = np.array([wp[0] - robot_x, wp[1] - robot_y])
                            waypoint_direction = waypoint_direction / (dist_to_wp + 1e-6)
                            heading_alignment = np.dot(vel_direction, waypoint_direction)
                            heading_reward = self.reward_heading_weight * max(0, heading_alignment)
                            reward += heading_reward
                
                self.last_waypoint_distance = dist_to_wp
        
        # Fall penalty
        if fall:
            reward += self.reward_fall
            done = True
            self.score = 0
        
        # Boundary penalty
        if out_of_bounds:
            reward += self.reward_boundary
            done = True
        
        # Timeout penalty (ran out of score points)
        if timeout and self.score <= 0:
            reward += self.reward_timeout
            done = True
        elif timeout:
            done = True
        
        # Get next observation
        obs = self._get_observation()
        
        # Info dict
        info = {
            'stage_id': self.current_stage,
            'score': self.score,
            'waypoints_captured': self.waypoints_captured,
            'total_waypoints': len(self.waypoints),
            'episode_time': self.episode_time,
            'fall': fall,
            'boundary': out_of_bounds,
            'timeout': timeout,
            'waypoint_captured_this_step': waypoint_captured,
            'success': (self.waypoints_captured >= len(self.waypoints)) if self.waypoints else (self.episode_time >= stage['max_time'] and not fall)
        }
        
        # Track episode completion
        if done:
            self.episode_active = False
            self.total_episodes += 1
            self.episode_history.append(info['success'])
            if len(self.episode_history) > self.success_window:
                self.episode_history.pop(0)
        
        return obs, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get observation vector (32 dimensions).
        
        Components:
        - Base velocity (vx, vy, omega): 3
        - Heading (sin, cos): 2
        - Waypoint info (dx, dy, distance): 3
        - Obstacle distances (16 rays): 16
        - Stage encoding (one-hot): 8
        
        Returns:
            observation vector, shape (32,)
        """
        obs = []
        
        # Get robot state
        pos, ori = self.spot.robot.get_world_pose()
        vel = self.spot.robot.get_linear_velocity()
        angvel = self.spot.robot.get_angular_velocity()
        
        robot_x, robot_y = pos[0], pos[1]
        yaw = self._quat_to_yaw(ori)
        
        # Base velocity (3)
        obs.extend([vel[0], vel[1], angvel[2]])
        
        # Heading (2)
        obs.extend([np.sin(yaw), np.cos(yaw)])
        
        # Waypoint info (3)
        if self.waypoints and self.current_waypoint_idx < len(self.waypoints):
            wp = self.waypoints[self.current_waypoint_idx]
            # World frame
            dx_world = wp[0] - robot_x
            dy_world = wp[1] - robot_y
            # Rotate to robot frame
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            dx_robot = dx_world * cos_yaw + dy_world * sin_yaw
            dy_robot = -dx_world * sin_yaw + dy_world * cos_yaw
            distance = np.sqrt(dx_world**2 + dy_world**2)
            obs.extend([dx_robot, dy_robot, distance])
        else:
            obs.extend([0.0, 0.0, 0.0])
        
        # Obstacle distances (16 rays)
        obstacle_dists = self._raycast_obstacles(robot_x, robot_y, yaw)
        obs.extend(obstacle_dists)
        
        # Stage encoding (8 - one-hot)
        stage_encoding = [0.0] * 8
        stage_encoding[self.current_stage] = 1.0
        obs.extend(stage_encoding)
        
        return np.array(obs, dtype=np.float32)
    
    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale action from [-1, 1] to velocity command ranges."""
        vx = action[0] * (self.vx_range[1] if action[0] > 0 else -self.vx_range[0])
        vy = action[1] * (self.vy_range[1] if action[1] > 0 else -self.vy_range[0])
        omega = action[2] * (self.omega_range[1] if action[2] > 0 else -self.omega_range[0])
        return np.array([vx, vy, omega], dtype=np.float32)
    
    def _generate_waypoints(self, first_dist: float, subsequent_dist: float) -> List[np.ndarray]:
        """
        Generate waypoints with given distances.
        
        Args:
            first_dist: distance for first waypoint from start
            subsequent_dist: distance for subsequent waypoints from previous
        
        Returns:
            list of waypoint positions [x, y]
        """
        waypoints = []
        prev_pos = np.array([0.0, 0.0])  # start position
        
        for i in range(self.total_waypoints):
            dist = first_dist if i == 0 else subsequent_dist
            
            # Try random angles until we find one that fits in arena
            max_attempts = 100
            for _ in range(max_attempts):
                angle = np.random.uniform(0, 2 * np.pi)
                x = prev_pos[0] + dist * np.cos(angle)
                y = prev_pos[1] + dist * np.sin(angle)
                
                if self._inside_arena(x, y, margin=self.boundary_margin):
                    waypoints.append(np.array([x, y]))
                    prev_pos = np.array([x, y])
                    break
            else:
                # Couldn't find valid position, stop generating
                break
        
        return waypoints
    
    def _spawn_obstacles(self, stage: Dict):
        """Spawn obstacles based on stage configuration."""
        # This would integrate with ObstacleManager from Baseline_Environment.py
        # For now, simplified version - you'll need to adapt the full logic
        pass
    
    def _raycast_obstacles(self, robot_x: float, robot_y: float, yaw: float) -> List[float]:
        """
        Raycast in multiple directions to detect obstacles.
        
        Args:
            robot_x, robot_y: robot position
            yaw: robot heading
        
        Returns:
            list of distances (normalized to [0, 1]), length num_obstacle_rays
        """
        distances = []
        
        for i in range(self.num_obstacle_rays):
            # Ray angle in robot frame  
            ray_angle = 2 * np.pi * i / self.num_obstacle_rays
            # Convert to world frame
            world_angle = yaw + ray_angle
            
            # Ray endpoint
            ray_x = robot_x + self.ray_length * np.cos(world_angle)
            ray_y = robot_y + self.ray_length * np.sin(world_angle)
            
            # Check collision with obstacles
            # Simplified: check distance to nearest obstacle in this direction
            # In full implementation, use physx raycasting
            dist = self.ray_length  # default: no obstacle
            
            # Check arena boundary
            boundary_dist = self._ray_to_boundary(robot_x, robot_y, world_angle)
            dist = min(dist, boundary_dist)
            
            # Normalize to [0, 1]
            distances.append(dist / self.ray_length)
        
        return distances
    
    def _ray_to_boundary(self, x: float, y: float, angle: float) -> float:
        """Calculate distance to arena boundary along ray."""
        # Ray: (x, y) + t * (cos(angle), sin(angle))
        # Circle: x^2 + y^2 = R^2
        # Solve for t
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        a = cos_a**2 + sin_a**2
        b = 2 * (x * cos_a + y * sin_a)
        c = x**2 + y**2 - self.arena_radius**2
        
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return self.ray_length
        
        t = (-b + np.sqrt(discriminant)) / (2 * a)
        return max(0, t)
    
    def _inside_arena(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Check if position is inside arena."""
        dist_sq = (x - self.arena_center[0])**2 + (y - self.arena_center[1])**2
        return dist_sq < (self.arena_radius - margin)**2
    
    def _quat_to_yaw(self, quat: np.ndarray) -> float:
        """Convert quaternion [w, x, y, z] to yaw angle."""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)
    
    def get_success_rate(self) -> float:
        """Get success rate over last N episodes."""
        if len(self.episode_history) == 0:
            return 0.0
        return sum(self.episode_history) / len(self.episode_history)
    
    def should_advance_curriculum(self) -> bool:
        """Check if should advance to next curriculum stage."""
        if len(self.episode_history) < self.success_window:
            return False
        return self.get_success_rate() >= self.success_threshold
    
    def advance_stage(self) -> bool:
        """
        Advance to next curriculum stage.
        
        Returns:
            True if advanced, False if already at final stage
        """
        if self.current_stage < len(self.curriculum_stages) - 1:
            self.current_stage += 1
            self.episode_history.clear()  # Reset history for new stage
            return True
        return False
