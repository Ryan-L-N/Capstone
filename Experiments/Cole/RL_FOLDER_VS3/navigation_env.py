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
        # Multi-layer raycast heights (relative to robot center ~0.5m)
        self.raycast_heights = [-0.2, 0.0, 0.2]  # low, center, high
        # Total rays: 16 rays × 3 heights = 48 obstacle observations
        
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
        self.reward_stillness = self.config['reward'].get('stillness_penalty', 0.0)  # NEW: penalty for not moving
        self.reward_stillness = self.config['reward'].get('stillness_penalty', 0.0)  # NEW: penalty for not moving
        
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
        
        # Force-based inference for obstacle classification
        self.recent_contact_force = 0.0  # Rolling average of contact force magnitude
        self.recent_joint_effort = 0.0   # Estimated joint torque/effort
        self.push_detected = False       # Whether robot is currently pushing
        self.push_efficiency = 0.0       # Movement gained per unit effort
        self.contact_force_history = []  # Last N contact forces for trending
        self.max_contact_history = 10    # Window size for force history
        self._last_speed = 0.0           # Previous speed for acceleration calculation
        
        # Curriculum tracking
        self.episode_history = []  # Last N episodes for success calculation
        self.total_episodes = 0
        
        # Obstacle manager - simplified for now
        # TODO: Integrate full ObstacleManager from Baseline_Environment
        self.obstacle_mgr = None
        
        # Stage 3: Object Pushing Training tracking
        self.objects_pushed = {}  # Dict: object_id -> total_distance_pushed
        self.lightweight_objects_found = 0  # Count of unique light objects pushed >= 1m
        self.stage_3_push_threshold = 1.0  # meters - minimum push distance to count success
        self.current_object_id = 0  # Unique ID for current object being pushed
        self.last_contact_time = 0.0  # Timestamp of last contact with obstacle
        self.contact_loss_threshold = 0.5  # seconds - treat as new object if contact lost this long
        self.push_exploration_weight = 0.5  # Reward for initial contact with pushable objects
        self.push_sustained_weight = 0.8  # Reward for sustained pushing momentum
        self.push_success_weight = 15.0  # Bonus for each successful 1m+ push
        self.push_wasted_effort_weight = -0.3  # Penalty for high effort without progress
        
        # Control timing
        self.last_control_time = 0.0
        
        # Current command for physics callback
        self.current_command = np.zeros(3, dtype=np.float32)  # [vx, vy, omega]
        
        # Velocity tracking for acceleration calculation
        self._last_vel = np.zeros(3)
        self._last_vx = 0.0
        self._last_vy = 0.0
        
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
        
        # Reset Stage 3 (Object Pushing) tracking
        if self.current_stage == 2:  # Stage 3 is index 2 (0-indexed)
            self.objects_pushed = {}
            self.lightweight_objects_found = 0
            self.current_object_id = 0  # Start with first object
            self.last_contact_time = self.world.current_time
        
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
        
        # Track movement for stagnation penalty (Stages 1-2)
        if not hasattr(self, '_last_robot_pos'):
            self._last_robot_pos = pos.copy()
        
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
        
        # Stagnation penalty for Stages 1-3 (stability + pushing training)
        if self.current_stage < 3:  # Stages 1, 2, 3 (0-indexed: 0, 1, 2)
            dist_moved = np.linalg.norm(pos[:2] - self._last_robot_pos[:2])
            min_movement_threshold = 0.05  # meters per timestep
            if dist_moved < min_movement_threshold and self.reward_stillness < 0:
                stagnation_penalty = self.reward_stillness * (self.control_dt / 0.05)  # scale to standard timestep
                reward += stagnation_penalty
            self._last_robot_pos = pos.copy()
        
        # Speed reward - encourage fast movement when path is clear AND far from waypoint
        if self.reward_speed_weight > 0:
            vel = self.spot.robot.get_linear_velocity()
            forward_speed = np.linalg.norm(vel[:2])  # horizontal velocity magnitude
            
            # Check distance to current waypoint - don't reward speed when close
            dist_to_wp = 999.0
            if self.waypoints and self.current_waypoint_idx < len(self.waypoints):
                wp = self.waypoints[self.current_waypoint_idx]
                dist_to_wp = np.sqrt((robot_x - wp[0])**2 + (robot_y - wp[1])**2)
            
            # Only reward speed if no close obstacles (> 2m clear ahead) AND far from waypoint (>3m)
            # Get robot heading for obstacle check
            yaw = self._quat_to_yaw(ori)
            obstacle_dists = self._raycast_obstacles(robot_x, robot_y, yaw)
            # Check forward-facing obstacle rays (indices around front)
            min_obstacle_dist = min(obstacle_dists[6:10]) if len(obstacle_dists) >= 10 else 5.0  # front 4 rays
            if min_obstacle_dist > 2.0 and dist_to_wp > 3.0:  # path is clear AND not near waypoint
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
                    # 1. Progress shaping - reward for getting closer to waypoint
                    if self.last_waypoint_distance is not None:
                        progress = self.last_waypoint_distance - dist_to_wp
                        progress_reward = self.reward_progress_alpha * progress
                        # Simple reward: positive for progress, zero or small negative for moving away
                        reward += progress_reward
                    
                    # 2. Heading reward - reward for facing toward waypoint (applies throughout approach)
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
                    
                    # 3. Proximity deceleration reward - encourage slowing near waypoint
                    # Reward low speed when within 2.5m of waypoint to enable smooth capture
                    if dist_to_wp < 2.5:
                        vel = self.spot.robot.get_linear_velocity()
                        horizontal_speed = np.linalg.norm(vel[:2])
                        # Ideal speed scales with distance: 0.5 m/s at 2.5m, 0 m/s at capture
                        ideal_speed = 0.2 * dist_to_wp  # linear scaling
                        speed_error = abs(horizontal_speed - ideal_speed)
                        # Reward being close to ideal speed (inverse error)
                        decel_reward = 0.15 * max(0, 1.0 - speed_error)
                        reward += decel_reward
                
                self.last_waypoint_distance = dist_to_wp
        
        # Force-based pushing rewards and penalties (all stages, but scaled by stage priority)
        # Scales: Stages 1-2 get 0.2x, Stage 3 gets 1.0x, Stages 4-7 get 0.4x
        if self.current_stage < 2:
            push_scale = 0.2  # Stages 1-2: low priority (focus on stability)
        elif self.current_stage == 2:
            push_scale = 1.0  # Stage 3: full rewards (dedicated pushing training)
        else:
            push_scale = 0.4  # Stages 4-7: reduced (focus on navigation)
        
        contact_force = self._calculate_contact_force()
        joint_effort = self._calculate_joint_effort()
        vel = self.spot.robot.get_linear_velocity()
        speed = np.linalg.norm(vel[:2])
        
        # Reward contact with light objects (high contact force + low joint effort)
        if contact_force > 0.4 and joint_effort < 0.7:
            # Light contact detected - reward exploration and pushing
            push_exploration_reward = push_scale * self.push_exploration_weight * contact_force * (1.0 - joint_effort)
            reward += push_exploration_reward
            
            # Bonus reward for sustained pushing (moving while in contact)
            if speed > 0.2:
                sustained_push_reward = push_scale * self.push_sustained_weight * contact_force
                reward += sustained_push_reward
            else:
                # Penalty for high contact without moving (stuck pushing attempt)
                stuck_penalty = push_scale * self.push_wasted_effort_weight * contact_force
                reward += stuck_penalty
        elif contact_force > 0.3 and joint_effort > 0.8:
            # High effort with moderate contact but likely not pushing effectively
            wasted_push_effort = push_scale * self.push_wasted_effort_weight * joint_effort
            reward += wasted_push_effort
        
        # Track cumulative distance traveled while pushing (Stage 3 unique objects)
        if self.current_stage == 2:  # Stage 3 only
            # Detect contact loss - if no contact for loss threshold, next contact = new object
            if contact_force < 0.2:
                time_since_contact = self.world.current_time - self.last_contact_time
                if time_since_contact > self.contact_loss_threshold and self.lightweight_objects_found < 5:
                    # Transition to next object on next contact
                    pass  # Will increment when contact resumes
            elif contact_force > 0.3 and speed > 0.1:
                # In active contact and moving: accumulate distance
                current_obj_str = str(self.current_object_id)
                if current_obj_str not in self.objects_pushed:
                    self.objects_pushed[current_obj_str] = 0.0
                
                dist_traveled = speed * self.control_dt
                self.objects_pushed[current_obj_str] += dist_traveled
                self.last_contact_time = self.world.current_time
                
                # Check if this object has reached the 1m threshold
                if self.objects_pushed[current_obj_str] >= self.stage_3_push_threshold:
                    if self.lightweight_objects_found < 5:
                        # Reward each successful unique push
                        push_success_reward = self.push_success_weight
                        reward += push_success_reward
                        self.lightweight_objects_found += 1
                        # Move to next object ID
                        self.current_object_id += 1
                        self.objects_pushed[current_obj_str] = 0.0  # Reset this object
            elif contact_force > 0.2:
                # Light contact but not moving - just update timestamp
                self.last_contact_time = self.world.current_time
        
        # Fall penalty
        if fall:
            reward += self.reward_fall
            done = True
            self.score = 0
        
        # Boundary penalty (soft penalty - episode continues to allow learning)
        if out_of_bounds:
            reward += self.reward_boundary
            # Do NOT terminate episode - let agent learn to avoid boundaries through penalty
        
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
            'objects_pushed_count': self.lightweight_objects_found if self.current_stage == 2 else 0,
        }
        
        # Determine success based on stage
        if self.current_stage == 2:  # Stage 3: Object Pushing
            info['success'] = self.lightweight_objects_found >= 5
            if info['success']:
                done = True
        else:  # Stages 1-2 and 4-7: by waypoint capture or time limit
            info['success'] = (self.waypoints_captured >= len(self.waypoints)) if self.waypoints else (self.episode_time >= stage['max_time'] and not fall)
        
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
        Get observation vector with multi-sensor fusion.
        
        Components:
        - Base velocity (vx, vy, omega): 3
        - Heading (sin, cos): 2
        - IMU data (roll, pitch, accel_x, accel_y): 4
        - Waypoint info (dx, dy, distance): 3
        - Multi-layer obstacle distances (16 rays × 3 heights): 48
        - Foot contact (4 feet): 4
        - Leg joint summary (4 legs avg angle): 4
        - Stage encoding (one-hot): 7
        Total: 75 dimensions
        
        Returns:
            observation vector, shape (75,)
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
        
        # IMU data - extract roll and pitch from quaternion (4)
        w, x, y, z = ori[0], ori[1], ori[2], ori[3]
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(2*(w*y - z*x))
        # Linear acceleration (approximate from velocity change)
        accel_magnitude = np.linalg.norm(vel - getattr(self, '_last_vel', vel)) / self.control_dt if hasattr(self, '_last_vel') else 0.0
        accel_x = vel[0] - getattr(self, '_last_vx', 0.0)
        accel_y = vel[1] - getattr(self, '_last_vy', 0.0)
        obs.extend([roll, pitch, accel_x, accel_y])
        self._last_vel = vel.copy()
        self._last_vx = vel[0]
        self._last_vy = vel[1]
        
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
        
        # Multi-layer obstacle distances (48 rays: 16 rays × 3 heights)
        for height_offset in self.raycast_heights:
            obstacle_dists = self._raycast_obstacles(robot_x, robot_y, yaw, height_offset=height_offset)
            obs.extend(obstacle_dists)
        
        # Foot contact feedback (4 feet)
        foot_contacts = self._get_foot_contacts()
        obs.extend(foot_contacts)
        
        # Leg joint summary (4 legs, average joint angle per leg)
        leg_joint_angles = self._get_leg_joint_summary()
        obs.extend(leg_joint_angles)
        
        # Stage encoding (7 - one-hot for current curriculum)
        stage_encoding = [0.0] * 7
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
        if not hasattr(self, 'rng'):
            self.rng = np.random.default_rng(seed=42)
        
        # Draw arena boundary (circular boundary)
        self._draw_arena_boundary()
        
        # Calculate number of obstacles to spawn
        light_coverage = stage['obstacles_light_coverage']
        heavy_coverage = stage['obstacles_heavy_coverage']
        small_coverage = stage['obstacles_small_coverage']
        
        # Estimate arena area and spawn counts
        arena_area = np.pi * (self.arena_radius ** 2)
        
        # Spawn light obstacles
        num_light = max(1, int(arena_area * light_coverage / 0.1))
        for i in range(num_light):
            self._spawn_random_obstacle(i, mass_range=(0.1, 0.45))
        
        # Spawn heavy obstacles
        num_heavy = max(1, int(arena_area * heavy_coverage / 0.1))
        for i in range(num_heavy + num_light, num_heavy + num_light + num_heavy):
            self._spawn_random_obstacle(i, mass_range=(65.4, 100.0))
        
        # Spawn small static obstacles
        num_small = max(1, int(arena_area * small_coverage / 0.1))
        for i in range(num_heavy + num_light + num_heavy, num_heavy + num_light + num_heavy + num_small):
            self._spawn_small_obstacle(i)
    
    def _draw_arena_boundary(self):
        """Draw the arena boundary as a visual ring."""
        from omni.isaac.core.prims import GeometryPrim
        import omni.kit.commands as commands
        
        # Create a torus to represent the arena boundary
        boundary_path = "/World/ArenaRim"
        major_radius = self.arena_radius
        minor_radius = 0.2  # 20cm thickness for visibility
        
        try:
            commands.execute('CreatePrimWithDefaultXformFromStore',
                prim_type="Torus",
                prim_path=boundary_path)
            
            torus = self.stage.GetPrimAtPath(boundary_path)
            if torus:
                # Set appearance
                torus_geom = torus.GetChild("Torus")
                if torus_geom:
                    material_path = "/World/Looks/BoundaryMaterial"
                    commands.execute('CreateMaterial',
                        material_path=material_path,
                        material_name="BoundaryMaterial")
        except:
            pass  # Fallback - arena still exists even if boundary visual fails
    
    def _spawn_random_obstacle(self, idx: int, mass_range=(0.1, 0.45)):
        """Spawn a random obstacle inside the arena."""
        from pxr import UsdGeom, Gf
        
        max_attempts = 20
        for _ in range(max_attempts):
            # Random position inside arena
            angle = self.rng.uniform(0, 2 * np.pi)
            r = self.rng.uniform(2.0, self.arena_radius - 2.0)  # Keep 2m from boundary
            x = self.arena_center[0] + r * np.cos(angle)
            y = self.arena_center[1] + r * np.sin(angle)
            
            # Check distance from start position
            if np.sqrt((x - 0)**2 + (y - 0)**2) < 3.0:  # 3m buffer from start
                continue
            
            break
        else:
            return  # Could not find position
        
        mass = self.rng.uniform(mass_range[0], mass_range[1])
        size = self.rng.uniform(0.2, 0.5)
        
        # Choose color based on mass
        if mass < 0.5:
            color = Gf.Vec3f(0.2, 0.8, 0.2)  # Green for light
        else:
            color = Gf.Vec3f(0.8, 0.2, 0.2)  # Red for heavy
        
        prim_path = f"/World/Obstacles/Obst_{idx:03d}"
        
        # Create a simple cube obstacle
        cube = UsdGeom.Cube.Define(self.stage, prim_path)
        cube.GetSizeAttr().Set(size)
        cube.AddTranslateOp().Set((x, y, size/2))
        
        # Add physics
        try:
            from pxr import UsdPhysics, PhysxSchema
            
            rigid_body = UsdPhysics.RigidBodyAPI.Apply(self.stage.GetPrimAtPath(prim_path))
            mass_api = UsdPhysics.MassAPI.Apply(self.stage.GetPrimAtPath(prim_path))
            mass_api.GetMassAttr().Set(mass)
            
            collider = UsdPhysics.CollisionAPI.Apply(self.stage.GetPrimAtPath(prim_path))
        except:
            pass
    
    def _spawn_small_obstacle(self, idx: int):
        """Spawn a small static obstacle."""
        from pxr import UsdGeom, Gf
        
        max_attempts = 20
        for _ in range(max_attempts):
            # Random position inside arena
            angle = self.rng.uniform(0, 2 * np.pi)
            r = self.rng.uniform(2.0, self.arena_radius - 2.0)
            x = self.arena_center[0] + r * np.cos(angle)
            y = self.arena_center[1] + r * np.sin(angle)
            
            if np.sqrt((x - 0)**2 + (y - 0)**2) < 3.0:
                continue
            
            break
        else:
            return
        
        size = self.rng.uniform(0.04, 0.15)
        prim_path = f"/World/SmallObstacles/Small_{idx:03d}"
        
        # Create a small sphere
        sphere = UsdGeom.Sphere.Define(self.stage, prim_path)
        sphere.GetRadiusAttr().Set(size/2)
        sphere.AddTranslateOp().Set((x, y, size/2))
        
        try:
            from pxr import UsdPhysics
            collider = UsdPhysics.CollisionAPI.Apply(self.stage.GetPrimAtPath(prim_path))
        except:
            pass
    
    def _raycast_obstacles(self, robot_x: float, robot_y: float, yaw: float, height_offset: float = 0.0) -> List[float]:
        """
        Raycast in multiple directions to detect obstacles at specified height.
        
        Args:
            robot_x, robot_y: robot position
            yaw: robot heading
            height_offset: height above robot center for this raycast layer (positive = up)
        
        Returns:
            list of distances (normalized to [0, 1]), length num_obstacle_rays
        """
        distances = []
        
        for i in range(self.num_obstacle_rays):
            # Ray angle in robot frame  
            ray_angle = 2 * np.pi * i / self.num_obstacle_rays
            # Convert to world frame
            world_angle = yaw + ray_angle
            
            # Ray endpoint (2D, height handled separately for visualization)
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
    
    def _get_foot_contacts(self) -> List[float]:
        """
        Get foot contact feedback for all 4 feet.
        
        Returns:
            list of 4 values (0.0-1.0), one per foot indicating contact pressure
        """
        contacts = []
        # Simplified: return [1.0, 1.0, 1.0, 1.0] if robot is not falling
        # In full implementation, query actual contact forces from physics engine
        pos = self.spot.robot.get_world_pose()[0]
        is_falling = pos[2] < self.fall_threshold
        contact_value = 0.0 if is_falling else 1.0
        return [contact_value, contact_value, contact_value, contact_value]
    
    def _get_leg_joint_summary(self) -> List[float]:
        """
        Get summary of leg joint angles.
        
        Returns:
            list of 4 values, one per leg (average joint angle normalized)
        """
        leg_angles = []
        # Simplified: return neutral positions (0.5 normalized)
        # In full implementation, query actual joint states from Spot
        # Spot has 3 joints per leg (hip_x, hip_y, knee), so 12 total
        # For now, return average posture per leg
        for i in range(4):
            # Neutral leg posture = 0.5 (middle of range)
            leg_angles.append(0.5)
        return leg_angles
    
    def _calculate_contact_force(self) -> float:
        """
        Estimate contact force magnitude from foot contacts.
        
        Returns:
            contact_force: normalized contact force (0.0 to 1.0)
        """
        # Get foot contact feedback
        foot_contacts = self._get_foot_contacts()
        
        # Average contact across all feet (0.0-1.0 per foot)
        avg_contact = np.mean(foot_contacts) if foot_contacts else 0.0
        
        # Store in history for trending
        self.contact_force_history.append(avg_contact)
        if len(self.contact_force_history) > self.max_contact_history:
            self.contact_force_history.pop(0)
        
        # Update rolling average
        self.recent_contact_force = np.mean(self.contact_force_history) if self.contact_force_history else 0.0
        
        return self.recent_contact_force
    
    def _calculate_joint_effort(self) -> float:
        """
        Estimate joint effort from velocity and acceleration changes.
        
        Returns:
            joint_effort: normalized effort level (0.0 to 1.0)
        """
        # Get current velocity
        vel = self.spot.robot.get_linear_velocity()
        current_speed = np.linalg.norm(vel[:2])
        
        # Estimate effort from acceleration (velocity change per timestep)
        accel = abs(current_speed - getattr(self, '_last_speed', current_speed)) / (self.control_dt + 1e-6)
        self._last_speed = current_speed
        
        # High acceleration/deceleration indicates high joint effort
        # Normalize to 0-1 range (assuming max effort produces ~5 m/s^2)
        joint_effort = min(1.0, accel / 5.0)
        self.recent_joint_effort = joint_effort
        
        return joint_effort
    
    def _calculate_push_efficiency(self, robot_vel: np.ndarray, waypoint_dir: np.ndarray) -> float:
        """
        Calculate push efficiency: how effectively is effort translating to waypoint progress?
        
        Args:
            robot_vel: current robot velocity [vx, vy]
            waypoint_dir: normalized direction to waypoint
        
        Returns:
            push_efficiency: 0.0 (wasted effort) to 1.0 (perfect alignment)
        """
        if np.linalg.norm(robot_vel[:2]) < 0.05:
            return 0.0  # Not moving
        
        vel_direction = robot_vel[:2] / (np.linalg.norm(robot_vel[:2]) + 1e-6)
        
        # Dot product: how aligned is velocity with waypoint direction?
        # Positive = moving toward waypoint (efficient)
        # Negative = moving away (inefficient)
        alignment = np.dot(vel_direction, waypoint_dir)
        
        # Convert from [-1, 1] to [0, 1]
        efficiency = max(0.0, alignment)
        self.push_efficiency = efficiency
        
        return efficiency
    
    def _detect_push_resistance(self) -> bool:
        """
        Detect if robot is encountering high resistance (heavy obstacle).
        
        Returns:
            high_resistance: True if pushing into obstacle, False otherwise
        """
        # High resistance indicated by:
        # 1. High joint effort AND
        # 2. Low velocity (not making progress despite effort)
        effort = self.recent_joint_effort
        contact = self.recent_contact_force
        vel = self.spot.robot.get_linear_velocity()
        speed = np.linalg.norm(vel[:2])
        
        # Detection: high effort + high contact + low speed = obstacle resistance
        high_resistance = effort > 0.6 and contact > 0.7 and speed < 0.5
        self.push_detected = high_resistance
        
        return high_resistance
    
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
