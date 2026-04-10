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


class HierarchicalNavigationController:
    """
    Hierarchical behavior controller for Spot navigation.
    
    Manages two behavioral modes:
    - Turn mode: Robot aligns with waypoint (vx=0, omega allowed)
    - Approach mode: Robot moves toward waypoint (full vx, vy, omega allowed)
    
    Mode transitions based on heading error to waypoint.
    """
    
    def __init__(self, turn_threshold_rad: float = 0.3):
        """
        Args:
            turn_threshold_rad: Angle error threshold (radians).
                               If |heading_error| > threshold, enter turn mode.
        """
        self.turn_threshold = turn_threshold_rad
        self.current_mode = "approach"  # Start in approach mode
        
    def get_mode(self, heading_error: float) -> str:
        """
        Determine current behavior mode based on heading error.
        
        Args:
            heading_error: Angle error to waypoint (radians, normalized to [-π, π])
        
        Returns:
            "turn" if |heading_error| > threshold, else "approach"
        """
        if abs(heading_error) > self.turn_threshold:
            return "turn"
        else:
            return "approach"
    
    def constrain_action(self, action: np.ndarray, mode: str) -> np.ndarray:
        """
        Apply mode-specific constraints to raw policy output.
        
        Args:
            action: Raw action from policy [vx, vy, omega]
            mode: Current behavior mode ("turn" or "approach")
        
        Returns:
            Constrained action [vx_constrained, vy_constrained, omega]
        """
        vx, vy, omega = action[0], action[1], action[2]
        
        if mode == "turn":
            # In turn mode: suppress forward movement, allow full turning
            # This forces the robot to rotate in place toward waypoint
            return np.array([0.0, 0.0, omega], dtype=np.float32)
        else:  # approach mode
            # In approach mode: allow full movement
            return action
    
    def get_mode_encoding(self, mode: str) -> List[float]:
        """
        Convert mode to one-hot encoding for observation.
        
        Args:
            mode: Current mode ("turn" or "approach")
        
        Returns:
            [is_turning, is_approaching] one-hot vector
        """
        if mode == "turn":
            return [1.0, 0.0]
        else:
            return [0.0, 1.0]


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
        # Total rays: 18 rays × 3 heights = 54 obstacle observations
        
        # Initialize hierarchical controller
        turn_threshold = self.config['observation'].get('turn_threshold_rad', 0.3)
        self.hierarchical_controller = HierarchicalNavigationController(turn_threshold_rad=turn_threshold)
        self.current_mode = "approach"
        
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
        # Distance-based speed reward tiers (NEW)
        self.speed_reward_tier_1_threshold = self.config['reward'].get('speed_reward_tier_1_threshold', 0.9)
        self.speed_reward_tier_2_threshold = self.config['reward'].get('speed_reward_tier_2_threshold', 1.79)
        self.speed_reward_tier_2_per_meter = self.config['reward'].get('speed_reward_tier_2_per_meter', 0.25)
        self.speed_reward_tier_3_per_meter = self.config['reward'].get('speed_reward_tier_3_per_meter', 0.50)
        self.reward_lateral_velocity_penalty = self.config['reward'].get('lateral_velocity_penalty', 0.0)
        self.reward_cross_track_error_weight = self.config['reward'].get('cross_track_error_weight', 0.0)
        self.reward_static_turn_reward = self.config['reward'].get('static_turn_reward', 0.0)
        self.reward_fall = self.config['reward']['fall_penalty']
        self.reward_boundary = self.config['reward']['boundary_penalty']
        
        # LiDAR-based obstacle avoidance (Stages with obstacles)
        self.collision_penalty = self.config['reward'].get('collision_penalty', -100.0)
        self.min_safe_distance = self.config['reward'].get('min_safe_distance', 0.5)
        self.obstacle_clearance_penalty = self.config['reward'].get('obstacle_clearance_penalty', -1.0)
        self.contact_threshold = 0.1  # meters - distance at which robot is considered in contact with obstacle
        
        # Efficiency bonus - reward for completing waypoints before score expires
        self.efficiency_bonus_weight = self.config['reward'].get('efficiency_bonus_weight', 0.5)
        
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
        
        # Episode reward breakdown tracking (for logging)
        self.episode_reward_components = {
            'waypoint_capture': 0.0,
            'time_penalty': 0.0,
            'progress_shaping': 0.0,
            'distance_reward': 0.0,
            'heading_reward': 0.0,
            'speed_reward': 0.0,
            'lateral_penalty': 0.0,
            'cross_track_error': 0.0,
            'static_turn_reward': 0.0,
            'stagnation_penalty': 0.0,
            'decel_reward': 0.0,
            'fall_penalty': 0.0,
            'boundary_penalty': 0.0,
            'timeout_penalty': 0.0,
            'efficiency_bonus': 0.0,
            'collision_penalty': 0.0,
            'obstacle_clearance_penalty': 0.0
        }
        
        # Episode score breakdown tracking (for logging)
        self.episode_score_components = {
            'initial_points': 0.0,
            'time_decay': 0.0,
            'waypoint_bonus': 0.0,
            'stage_success_bonus': 0.0,
            'fall_penalty': 0.0
        }
        
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
        
        # Velocity tracking for acceleration calculation
        self._last_vel = np.zeros(3)
        self._last_vx = 0.0
        self._last_vy = 0.0
        
        # Position tracking for distance-based speed reward
        self._last_robot_pos_for_speed_reward = np.zeros(3)  # [x, y, z]
        
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
            # Reset episode history when advancing to a new stage
            self.episode_history = []
        
        stage = self.curriculum_stages[self.current_stage]
        
        # Update success criteria from stage config if specified (otherwise use global defaults)
        if 'success_window' in stage:
            self.success_window = stage['success_window']
        if 'success_threshold' in stage:
            self.success_threshold = stage['success_threshold']
        
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
        self._last_robot_pos_for_speed_reward = self.start_pos.copy()  # Initialize position tracking for speed reward
        
        # Reset reward and score breakdowns
        for key in self.episode_reward_components:
            self.episode_reward_components[key] = 0.0
        for key in self.episode_score_components:
            self.episode_score_components[key] = 0.0
        self.episode_score_components['initial_points'] = self.initial_points  # Track initial points
        
        # Generate waypoints (if stage has them)
        self.waypoints = []
        if stage['waypoint_distance_first'] is not None:
            self.waypoints = self._generate_waypoints(
                first_dist=stage['waypoint_distance_first'],
                subsequent_dist=stage['waypoint_distance_subsequent']
            )
        
        # Spawn obstacles (stage 6 only)
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
        Apply current command to Spot with hierarchical mode constraint (called by physics callback).
        
        The hierarchical controller intercepts raw policy outputs and constrains
        them based on the current behavioral mode:
        - Turn mode: suppresses forward motion (vx=0), enables turning (omega allowed)
        - Approach mode: allows full movement (vx, vy, omega all allowed)
        
        Args:
            dt: physics timestep (typically 0.002s at 500Hz)
        """
        if self.episode_active:
            # Get current heading error for mode decision
            heading_error = self.compute_heading_error()
            
            # Determine current mode and constrain action
            mode = self.hierarchical_controller.get_mode(heading_error)
            constrained_command = self.hierarchical_controller.constrain_action(
                self.current_command, 
                mode
            )
            
            # Apply constrained command to SpotFlatTerrainPolicy
            self.spot.forward(dt, constrained_command)
    
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
        
        # Update score - time decay applies to ALL stages (300 points, -1/sec)
        time_decay_amount = self.time_decay * self.control_dt
        self.score -= time_decay_amount
        self.episode_score_components['time_decay'] -= time_decay_amount
        
        # Get robot state
        pos, ori = self.spot.robot.get_world_pose()
        robot_x, robot_y, robot_z = pos[0], pos[1], pos[2]
        
        # Check termination conditions
        done = False
        fall = robot_z < self.fall_threshold
        out_of_bounds = not self._inside_arena(robot_x, robot_y)
        timeout = False
        stage_success = False  # Track if stage-specific success criterion met
        
        stage = self.curriculum_stages[self.current_stage]
        if stage['max_time'] is not None:
            timeout = self.episode_time >= stage['max_time']
            # Award bonus for stages 1-2 when reaching max_time without falling
            if timeout and not fall and self.current_stage < 2:  # Stages 1-2 (0-indexed: 0-1)
                # Stage 1: +100 bonus, Stage 2+: +50 bonus
                bonus = 100.0 if self.current_stage == 0 else 50.0
                self.score += bonus
                self.episode_score_components['stage_success_bonus'] += bonus
                stage_success = True
            if timeout:
                done = True
        else:
            timeout = self.score <= 0
            if timeout:
                done = True
        
        # Calculate reward
        # NO TIME PENALTY for RL reward in stages 1-3 (let them focus on learning stability)
        if self.current_stage >= 3:
            reward = self.reward_time_penalty  # constant time penalty (stages 4+)
            self.episode_reward_components['time_penalty'] += self.reward_time_penalty
        else:
            reward = 0.0  # no time penalty for stages 1-3
        
        # Stage success bonus for stages 1-2 (completing without falling)
        if stage_success:
            # Stage 1: +100 bonus, Stage 2+: +50 bonus
            bonus = 100.0 if self.current_stage == 0 else 50.0
            reward += bonus
        
        # Get robot yaw and obstacle distances (used for multiple reward calculations)
        yaw = self._quat_to_yaw(ori)
        obstacle_dists = self._raycast_obstacles(robot_x, robot_y, yaw)
        
        # LiDAR-based obstacle avoidance penalties
        # Penalize robot for getting too close to obstacles (Stages 2+)
        if self.current_stage >= 1:  # Enable from Stage 2 onward (0-indexed: 1+)
            if obstacle_dists:
                min_obs_dist = min(obstacle_dists)
                
                # Check for collision (robot in contact with obstacle)
                if min_obs_dist < self.contact_threshold:
                    collision_penalty = self.collision_penalty
                    reward += collision_penalty
                    self.episode_reward_components['collision_penalty'] += collision_penalty
                
                # Check for clearance violation (obstacle too close but not touching)
                elif min_obs_dist < self.min_safe_distance:
                    # Penalty scales with how far below min_safe_distance the robot is
                    clearance_violation = self.min_safe_distance - min_obs_dist
                    clearance_penalty = self.obstacle_clearance_penalty * clearance_violation
                    reward += clearance_penalty
                    self.episode_reward_components['obstacle_clearance_penalty'] += clearance_penalty
        
        # Speed reward - distance-based tier system (replaces old linear speed_reward)
        # Tier 1: <0.9 m/s (2 mph) = 0 reward
        # Tier 2: 0.9-1.79 m/s (2-4 mph) = 0.25 per meter
        # Tier 3: ≥1.79 m/s (4+ mph) = 0.50 per meter
        vel = self.spot.robot.get_linear_velocity()
        forward_speed = np.linalg.norm(vel[:2])  # horizontal velocity magnitude (m/s)
        
        # Calculate distance traveled this frame
        pos = self.spot.robot.get_world_pose()[0]
        distance_traveled = np.linalg.norm(pos[:2] - self._last_robot_pos_for_speed_reward[:2])
        
        # Check if path is clear (obstacle check) and not near waypoint
        dist_to_wp = 999.0
        if self.waypoints and self.current_waypoint_idx < len(self.waypoints):
            wp = self.waypoints[self.current_waypoint_idx]
            dist_to_wp = np.sqrt((robot_x - wp[0])**2 + (robot_y - wp[1])**2)
        
        min_obstacle_dist = min(obstacle_dists[6:10]) if len(obstacle_dists) >= 10 else 5.0  # front rays
        
        # Apply speed reward if path is clear AND not near waypoint
        if min_obstacle_dist > 2.0 and dist_to_wp > 3.0:
            speed_reward = 0.0
            if forward_speed >= self.speed_reward_tier_2_threshold:  # Tier 3 (≥1.79 m/s)
                speed_reward = self.speed_reward_tier_3_per_meter * distance_traveled
            elif forward_speed >= self.speed_reward_tier_1_threshold:  # Tier 2 (0.9-1.79 m/s)
                speed_reward = self.speed_reward_tier_2_per_meter * distance_traveled
            # Tier 1 (<0.9 m/s) yields 0 reward (no bonus for slow movement)
            
            if speed_reward > 0:
                reward += speed_reward
                self.episode_reward_components['speed_reward'] += speed_reward
        
        # Update position for next frame
        self._last_robot_pos_for_speed_reward = pos.copy()
        
        # Lateral velocity penalty - discourage unnecessary strafing to prevent drift
        if self.reward_lateral_velocity_penalty > 0:
            vel = self.spot.robot.get_linear_velocity()
            lateral_speed = abs(vel[1])  # |vy| - absolute value for left/right strafe
            # Penalty proportional to lateral speed, scaled 0-1 for speeds 0-0.5 m/s
            lateral_penalty = self.reward_lateral_velocity_penalty * min(lateral_speed / 0.5, 1.0)
            reward -= lateral_penalty
            self.episode_reward_components['lateral_penalty'] -= lateral_penalty
        
        # Static turn reward - reward turning in place (high omega, low vx)
        if self.reward_static_turn_reward > 0:
            vel = self.spot.robot.get_linear_velocity()
            angvel = self.spot.robot.get_angular_velocity()
            vx = vel[0]
            omega = angvel[2]  # yaw rate
            
            # If forward speed is low (< 0.3 m/s) but turning fast (> 0.3 rad/s), reward it
            if abs(vx) < 0.3 and abs(omega) > 0.3:
                # Reward proportional to turn rate, scaled 0-1 for rotations 0.3-2.0 rad/s
                turn_reward = self.reward_static_turn_reward * min(abs(omega) / 2.0, 1.0)
                reward += turn_reward
                self.episode_reward_components['static_turn_reward'] += turn_reward
        
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
                self.episode_score_components['waypoint_bonus'] += self.waypoint_bonus_points
                reward += self.reward_waypoint
                self.episode_reward_components['waypoint_capture'] += self.reward_waypoint
                self.last_waypoint_distance = None
                
                # Check if all waypoints captured
                if self.current_waypoint_idx >= len(self.waypoints):
                    done = True
                    # Efficiency bonus: reward for completing with points remaining
                    if self.score > 0:
                        efficiency_bonus = self.efficiency_bonus_weight * self.score
                        reward += efficiency_bonus
                        self.episode_reward_components['efficiency_bonus'] += efficiency_bonus
            else:
                # Enhanced reward shaping (enabled per stage via use_progress_shaping flag)
                if stage['use_progress_shaping']:
                    # 1. Progress shaping - reward for getting closer, PENALTY for moving away
                    if self.last_waypoint_distance is not None:
                        progress = self.last_waypoint_distance - dist_to_wp
                        progress_reward = self.reward_progress_alpha * progress
                        reward += progress_reward
                        self.episode_reward_components['progress_shaping'] += progress_reward
                    
                    # 2. Distance-based reward - continuous guidance (inverse distance)
                    # Reward decreases with distance, max at 1m, zero at 20m
                    distance_reward = self.reward_distance_weight * max(0, 1.0 - dist_to_wp / 20.0)
                    reward += distance_reward
                    self.episode_reward_components['distance_reward'] += distance_reward
                    
                    # 3. Heading reward - reward for facing toward waypoint (applies throughout approach)
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
                            self.episode_reward_components['heading_reward'] += heading_reward
                    
                    # 3b. Cross-track error penalty - penalize deviating left/right from ideal path
                    if self.reward_cross_track_error_weight > 0 and dist_to_wp > 0.5:
                        vel = self.spot.robot.get_linear_velocity()
                        if np.linalg.norm(vel[:2]) > 0.1:  # only if moving
                            # Ideal direction toward waypoint
                            waypoint_direction = np.array([wp[0] - robot_x, wp[1] - robot_y])
                            waypoint_direction = waypoint_direction / (dist_to_wp + 1e-6)
                            
                            # Perpendicular vector (cross-track direction - pure left/right)
                            perp_direction = np.array([-waypoint_direction[1], waypoint_direction[0]])
                            
                            # Robot velocity direction
                            vel_direction = vel[:2] / np.linalg.norm(vel[:2])
                            
                            # Cross-track error = how much velocity is perpendicular to ideal path
                            cross_track_component = abs(np.dot(vel_direction, perp_direction))
                            
                            # Penalty proportional to perpendicular motion
                            cross_track_penalty = self.reward_cross_track_error_weight * cross_track_component
                            reward -= cross_track_penalty
                            self.episode_reward_components['cross_track_error'] -= cross_track_penalty
                    
                    # 4. Proximity deceleration reward - encourage slowing near waypoint
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
                        self.episode_reward_components['decel_reward'] += decel_reward
                
                self.last_waypoint_distance = dist_to_wp
        
        # Fall penalty
        if fall:
            reward += self.reward_fall
            self.episode_reward_components['fall_penalty'] += self.reward_fall
            self.episode_score_components['fall_penalty'] += self.reward_fall
            done = True  # Terminate immediately on fall
        
        # Boundary penalty (per second, scaled to control timestep)
        # RL reward only - does NOT affect score points
        if out_of_bounds:
            boundary_penalty_per_step = self.reward_boundary * self.control_dt
            reward += boundary_penalty_per_step
            self.episode_reward_components['boundary_penalty'] += boundary_penalty_per_step
            # Episode continues (no done = True)
        
        # Timeout penalty (ran out of score points)
        if timeout and self.score <= 0:
            reward += self.reward_timeout
            self.episode_reward_components['timeout_penalty'] += self.reward_timeout
        
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
            'success': (self.waypoints_captured >= len(self.waypoints)) if self.waypoints else (self.episode_time >= stage['max_time'] and not fall),
            'reward_breakdown': self.episode_reward_components.copy(),
            'score_breakdown': self.episode_score_components.copy()
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
        Get observation vector with multi-sensor fusion.
        
        Components:
        - Base velocity (vx, vy, omega): 3
        - Heading (sin, cos): 2
        - IMU data (roll, pitch, accel_x, accel_y): 4
        - Waypoint info (dx, dy, distance): 3
        - Multi-layer obstacle distances (18 rays × 3 heights): 54
        - Foot contact (4 feet): 4
        - Contact force (normalized): 1
        - Joint effort (normalized): 1
        - Leg joint summary (4 legs avg angle): 4
        - Stage encoding (one-hot): 6
        Total: 82 dimensions
        
        Returns:
            observation vector, shape (82,)
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
        
        # Multi-layer obstacle distances (54 rays: 18 rays × 3 heights)
        for height_offset in self.raycast_heights:
            obstacle_dists = self._raycast_obstacles(robot_x, robot_y, yaw, height_offset=height_offset)
            obs.extend(obstacle_dists)
        
        # Foot contact feedback (4 feet)
        foot_contacts = self._get_foot_contacts()
        obs.extend(foot_contacts)
        
        # Contact force estimate (1 value)
        contact_force = self._calculate_contact_force()
        obs.append(contact_force)
        
        # Joint effort estimate (1 value)
        joint_effort = self._calculate_joint_effort()
        obs.append(joint_effort)
        
        # Leg joint summary (4 legs, average joint angle per leg)
        leg_joint_angles = self._get_leg_joint_summary()
        obs.extend(leg_joint_angles)
        
        # Stage encoding (6 - one-hot)
        stage_encoding = [0.0] * 6
        stage_encoding[self.current_stage] = 1.0
        obs.extend(stage_encoding)
        
        return np.array(obs, dtype=np.float32)
    
    def get_network_observation(self) -> np.ndarray:
        """
        Extract network-compatible observation (34 dimensions).
        
        This is the simplified observation used by NavigationPolicy (34-dim):
        - Base velocity: [vx, vy, omega] (3)
        - Heading: [sin(yaw), cos(yaw)] (2)
        - Waypoint info: [dx, dy, distance] (3)
        - Obstacle distances: 18 raycasts (single layer) (18)
        - Stage encoding: one-hot (6)
        - Mode encoding: [is_turning, is_approaching] (2)
        Total: 3 + 2 + 3 + 18 + 6 + 2 = 34 dimensions
        
        Returns:
            observation vector, shape (34,)
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
        
        # Obstacle distances - single layer (18 rays)
        obstacle_dists = self._raycast_obstacles(robot_x, robot_y, yaw, height_offset=self.raycast_heights[0])
        obs.extend(obstacle_dists)
        
        # Stage encoding (6 - one-hot)
        stage_encoding = [0.0] * 6
        stage_encoding[self.current_stage] = 1.0
        obs.extend(stage_encoding)
        
        # Get current behavioral mode and add mode encoding (2 - one-hot)
        heading_error = self.compute_heading_error()
        mode = self.hierarchical_controller.get_mode(heading_error)
        self.current_mode = mode  # Track for logging
        mode_encoding = self.hierarchical_controller.get_mode_encoding(mode)
        obs.extend(mode_encoding)
        
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
    
    def compute_heading_error(self) -> float:
        """
        Compute heading error from robot facing direction to waypoint.
        
        Uses current robot orientation and waypoint direction to compute
        angle error in [-π, π] range.
        
        Returns:
            Heading error in radians. Positive = counterclockwise to waypoint.
        """
        # Get robot's current state
        pos, ori = self.spot.robot.get_world_pose()
        robot_x, robot_y = pos[0], pos[1]
        robot_yaw = self._quat_to_yaw(ori)
        
        # Get waypoint direction
        if self.waypoints and self.current_waypoint_idx < len(self.waypoints):
            wp = self.waypoints[self.current_waypoint_idx]
            waypoint_vec = np.array([wp[0] - robot_x, wp[1] - robot_y])
            waypoint_angle = np.arctan2(waypoint_vec[1], waypoint_vec[0])
        else:
            # No waypoint available - return 0 error
            return 0.0
        
        # Compute error and normalize to [-π, π]
        heading_error = waypoint_angle - robot_yaw
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        return heading_error
    
    def _get_foot_contacts(self) -> List[float]:
        """
        Get foot contact feedback for all 4 feet.
        
        Returns:
            list of 4 values (0.0-1.0), one per foot indicating contact pressure
        """
        pos = self.spot.robot.get_world_pose()[0]
        is_falling = pos[2] < self.fall_threshold
        contact_value = 0.0 if is_falling else 1.0
        
        return [contact_value, contact_value, contact_value, contact_value]
    
    def _calculate_contact_force(self) -> float:
        """
        Calculate contact force estimate from foot contacts.
        
        Returns:
            normalized contact force in [0, 1]
        """
        pos = self.spot.robot.get_world_pose()[0]
        return 0.0 if pos[2] < self.fall_threshold else 1.0
    
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
    
    def get_success_rate(self) -> float:
        """Get success rate over last N episodes."""
        if len(self.episode_history) == 0:
            return 0.0
        return sum(self.episode_history) / len(self.episode_history)
    
    def should_advance_curriculum(self, iterations_on_stage: int = 0) -> bool:
        """
        Check if should advance to next curriculum stage.
        
        Requires BOTH conditions:
        - Minimum 50 iterations on current stage
        - Success rate meets threshold (80% for stages 1-6)
        
        Args:
            iterations_on_stage: number of iterations completed on current stage
        
        Returns:
            True if both conditions met, False otherwise
        """
        # Must complete at least 50 iterations on this stage
        if iterations_on_stage < 50:
            return False
        
        # Must have enough episodes in history for success calculation
        stage = self.curriculum_stages[self.current_stage]
        stage_window = stage.get('success_window', self.success_window)
        stage_threshold = stage.get('success_threshold', self.success_threshold)
        
        if len(self.episode_history) < stage_window:
            return False
        
        # Check if success rate meets stage-specific threshold
        return self.get_success_rate() >= stage_threshold
    
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
        
        return joint_effort
