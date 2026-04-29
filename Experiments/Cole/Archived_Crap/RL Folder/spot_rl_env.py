"""
Spot RL Environment for Obstacle-Aware Navigation
==================================================
Reinforcement Learning environment for Boston Dynamics Spot quadruped.
Extends baseline circular waypoint navigation with obstacle interaction learning.

RL Objectives:
--------------
1. Forward-dominant locomotion with natural gait
2. Intelligent obstacle interaction (nudge vs. bypass)
3. Stable, energy-efficient movement
4. Incremental heading adjustments while walking

Observation Space:
------------------
- Robot state: joint positions/velocities, base orientation, linear/angular velocity
- Waypoint info: distance, heading error, progress
- Obstacle sensing: nearest obstacles with mass, friction, shape, distance
- Contact forces: foot contacts, collision magnitudes
- Stability metrics: roll, pitch, height

Action Space:
-------------
- Linear velocity (forward/backward)
- Angular velocity (turning)
- (Optional) Lateral velocity for bypass maneuvers

Reward Components:
------------------
+ Waypoint progress (distance reduction)
+ Waypoint reached (large bonus)
+ Forward locomotion (encourage natural gait)
+ Stability (upright orientation)
+ Energy efficiency (smooth actions)
+ Successful nudging (light obstacles pushed)
+ Smart bypass (when nudging fails)
- Falling
- Excessive lateral/backward motion
- High contact forces (unsafe nudging)
- Time penalty

Author: Cole (MS for Autonomy Project)
Date: February 2026
"""

import csv
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

# Import after SimulationApp initialization (handled by caller)
from omni.isaac.core import World
from omni.isaac.quadruped.robots import Spot
from pxr import Gf, UsdGeom, UsdPhysics


# ═════════════════════════════════════════════════════════════════════════════
# OBSERVATION AND ACTION SPACE DIMENSIONS
# ═════════════════════════════════════════════════════════════════════════════

# Robot proprioceptive state
OBS_JOINT_POS = 12          # 4 legs × 3 joints
OBS_JOINT_VEL = 12
OBS_BASE_ORIENTATION = 4    # Quaternion [w, x, y, z]
OBS_BASE_LIN_VEL = 3        # [vx, vy, vz]
OBS_BASE_ANG_VEL = 3        # [wx, wy, wz]
OBS_BASE_HEIGHT = 1

# Waypoint navigation
OBS_WAYPOINT_DIST = 1
OBS_WAYPOINT_HEADING = 1
OBS_WAYPOINT_PROGRESS = 1   # Fraction of waypoints completed

# Obstacle sensing (5 nearest obstacles)
OBS_NUM_OBSTACLES = 5
OBS_PER_OBSTACLE = 7        # [rel_x, rel_y, distance, mass_norm, friction, shape_id, is_static]
OBS_OBSTACLES_DIM = OBS_NUM_OBSTACLES * OBS_PER_OBSTACLE

# Contact and stability
OBS_FOOT_CONTACTS = 4       # Binary contact for each foot
OBS_COLLISION_MAGNITUDE = 1 # Total collision force magnitude
OBS_ROLL = 1
OBS_PITCH = 1

# Action history (for temporal awareness)
OBS_PREV_ACTIONS = 3        # [prev_vx, prev_vy, prev_omega]

# Total observation dimension
OBS_DIM = (OBS_JOINT_POS + OBS_JOINT_VEL + OBS_BASE_ORIENTATION + 
           OBS_BASE_LIN_VEL + OBS_BASE_ANG_VEL + OBS_BASE_HEIGHT +
           OBS_WAYPOINT_DIST + OBS_WAYPOINT_HEADING + OBS_WAYPOINT_PROGRESS +
           OBS_OBSTACLES_DIM + OBS_FOOT_CONTACTS + OBS_COLLISION_MAGNITUDE +
           OBS_ROLL + OBS_PITCH + OBS_PREV_ACTIONS)

# Action space: [forward_vel, lateral_vel, angular_vel]
ACTION_DIM = 3


# ═════════════════════════════════════════════════════════════════════════════
# REWARD WEIGHTS (Tunable hyperparameters)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RewardWeights:
    """Reward function hyperparameters."""
    
    # Progress rewards
    waypoint_reached: float = 100.0          # Large bonus for reaching waypoint
    distance_reduction: float = 2.0          # Reward per meter closer to waypoint
    
    # Locomotion quality
    forward_locomotion: float = 1.0          # Encourage forward movement
    lateral_penalty: float = 0.5             # Penalize sideways motion
    backward_penalty: float = 1.0            # Penalize backing up
    
    # Stability
    stability_reward: float = 0.2            # Upright orientation
    height_deviation: float = 1.0            # Penalize height changes
    
    # Obstacle interaction
    successful_nudge: float = 5.0            # Light obstacle pushed successfully
    failed_nudge_penalty: float = 2.0        # Stuck pushing immovable object
    smart_bypass: float = 3.0                # Efficiently navigating around obstacle
    collision_penalty: float = 0.5           # Per unit contact force
    
    # Energy efficiency
    action_smoothness: float = 0.1           # Penalize jerky movements
    energy_penalty: float = 0.05             # Penalize high action magnitudes
    
    # Terminal conditions
    fall_penalty: float = 100.0              # Fell over
    timeout_penalty: float = 10.0            # Ran out of time
    
    # Time penalty
    time_penalty: float = 0.01               # Small per-step penalty


# ═════════════════════════════════════════════════════════════════════════════
# CONTACT EVENT TRACKER
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ContactEvent:
    """Records a single obstacle contact event."""
    timestamp: float
    obstacle_id: str
    obstacle_mass: float
    contact_force: float
    spot_velocity: float
    result: str  # "nudged", "bypassed", "stuck", "failed"


class ContactTracker:
    """Tracks obstacle contacts and nudge/bypass behaviors."""
    
    def __init__(self, history_size: int = 100):
        self.events: deque = deque(maxlen=history_size)
        self.current_contacts: Dict[str, ContactEvent] = {}
        
    def register_contact(self, timestamp: float, obstacle_id: str, 
                         obstacle_mass: float, contact_force: float, 
                         spot_velocity: float):
        """Register new contact with obstacle."""
        if obstacle_id not in self.current_contacts:
            event = ContactEvent(
                timestamp=timestamp,
                obstacle_id=obstacle_id,
                obstacle_mass=obstacle_mass,
                contact_force=contact_force,
                spot_velocity=spot_velocity,
                result="ongoing"
            )
            self.current_contacts[obstacle_id] = event
    
    def resolve_contact(self, obstacle_id: str, result: str):
        """Resolve contact outcome: nudged, bypassed, stuck, failed."""
        if obstacle_id in self.current_contacts:
            event = self.current_contacts.pop(obstacle_id)
            event.result = result
            self.events.append(event)
            return event
        return None
    
    def get_recent_events(self, count: int = 10) -> List[ContactEvent]:
        """Get most recent contact events."""
        return list(self.events)[-count:]
    
    def clear(self):
        """Clear all tracked contacts."""
        self.events.clear()
        self.current_contacts.clear()


# ═════════════════════════════════════════════════════════════════════════════
# SPOT RL ENVIRONMENT
# ═════════════════════════════════════════════════════════════════════════════

class SpotRLEnv:
    """
    RL Environment for Spot obstacle-aware navigation.
    
    Wraps Isaac Sim physics with RL observation/action/reward interface.
    Designed for RSL-RL training framework.
    """
    
    def __init__(self, world: World, stage, obstacle_mgr, waypoint_mgr,
                 reward_weights: Optional[RewardWeights] = None):
        """
        Initialize RL environment.
        
        Args:
            world: Isaac Sim World instance
            stage: USD stage
            obstacle_mgr: ObstacleManager instance
            waypoint_mgr: Waypoint management system
            reward_weights: Reward function hyperparameters
        """
        self.world = world
        self.stage = stage
        self.obstacle_mgr = obstacle_mgr
        self.waypoint_mgr = waypoint_mgr
        self.rewards = reward_weights or RewardWeights()
        
        # Robot
        self.spot: Optional[Spot] = None
        
        # Environment state
        self.current_waypoint_idx = 0
        self.waypoints_reached = 0
        self.episode_num = 0
        self.episode_start_time = 0.0
        self.episode_length = 0
        
        # Previous state tracking
        self.prev_waypoint_dist = None
        self.prev_actions = np.zeros(ACTION_DIM)
        self.prev_position = np.zeros(3)
        
        # Contact tracking
        self.contact_tracker = ContactTracker()
        
        # Episode statistics
        self.episode_reward = 0.0
        self.episode_stats = {
            'waypoints_reached': 0,
            'obstacles_nudged': 0,
            'obstacles_bypassed': 0,
            'total_contacts': 0,
            'avg_stability': 0.0,
            'total_distance': 0.0,
            'termination_reason': None
        }
        
        # Logging
        self.episode_log: List[Dict] = []
        
    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVATION SPACE
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_observations(self) -> np.ndarray:
        """
        Construct observation vector for RL policy.
        
        Returns:
            np.ndarray: Observation vector of shape (OBS_DIM,)
        """
        obs = []
        
        # Get Spot state
        spot_pos, spot_quat = self.spot.get_world_pose()
        spot_lin_vel = self.spot.get_linear_velocity()
        spot_ang_vel = self.spot.get_angular_velocity()
        
        # Joint states (12 positions + 12 velocities)
        joint_positions = self.spot.get_joint_positions()
        joint_velocities = self.spot.get_joint_velocities()
        obs.extend(joint_positions)
        obs.extend(joint_velocities)
        
        # Base orientation (quaternion)
        obs.extend(spot_quat)  # [w, x, y, z]
        
        # Base velocity
        obs.extend(spot_lin_vel)
        obs.extend(spot_ang_vel)
        
        # Base height
        obs.append(spot_pos[2])
        
        # Waypoint navigation info
        if self.current_waypoint_idx < len(self.waypoint_mgr.waypoints):
            wp = self.waypoint_mgr.waypoints[self.current_waypoint_idx]
            wp_pos = wp["pos"]
            
            # Distance to waypoint
            dist = self._distance_2d(spot_pos[:2], wp_pos)
            obs.append(dist / 50.0)  # Normalize by arena diameter
            
            # Heading error
            heading_error = self._calculate_heading_error(spot_pos, spot_quat, wp_pos)
            obs.append(heading_error / math.pi)  # Normalize to [-1, 1]
            
            # Progress (waypoints completed)
            progress = self.waypoints_reached / len(self.waypoint_mgr.waypoints)
            obs.append(progress)
        else:
            obs.extend([0.0, 0.0, 1.0])  # No waypoint, full progress
        
        # Obstacle sensing (5 nearest obstacles)
        obstacles_obs = self._get_nearest_obstacles(spot_pos, n=OBS_NUM_OBSTACLES)
        obs.extend(obstacles_obs)
        
        # Foot contacts (4 binary values)
        foot_contacts = self._get_foot_contacts()
        obs.extend(foot_contacts)
        
        # Collision magnitude
        collision_force = self._get_collision_magnitude()
        obs.append(np.clip(collision_force / 100.0, 0, 1))  # Normalize
        
        # Stability: roll and pitch angles
        roll, pitch = self._get_roll_pitch(spot_quat)
        obs.append(roll / math.pi)  # Normalize
        obs.append(pitch / math.pi)
        
        # Previous actions
        obs.extend(self.prev_actions)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_nearest_obstacles(self, spot_pos: np.ndarray, n: int) -> List[float]:
        """Get observations for n nearest obstacles."""
        obs = []
        
        # Get all obstacles with distances
        obstacles_with_dist = []
        for obst in self.obstacle_mgr.obstacles:
            obst_pos = obst['pos']
            dist = self._distance_2d(spot_pos[:2], obst_pos)
            obstacles_with_dist.append((dist, obst))
        
        # Sort by distance
        obstacles_with_dist.sort(key=lambda x: x[0])
        
        # Take n nearest
        for i in range(n):
            if i < len(obstacles_with_dist):
                dist, obst = obstacles_with_dist[i]
                
                # Relative position
                rel_x = (obst['pos'][0] - spot_pos[0]) / 50.0  # Normalize
                rel_y = (obst['pos'][1] - spot_pos[1]) / 50.0
                
                # Distance (normalized)
                dist_norm = dist / 50.0
                
                # Mass (normalized by Spot's mass)
                mass_norm = obst['mass'] / 32.7  # Spot mass = 32.7 kg
                
                # Friction
                friction = obst.get('friction', 0.5)
                
                # Shape ID (encoded as float)
                shape_id = self._encode_shape(obst['shape'])
                
                # Is static (heavy obstacle)
                is_static = 1.0 if obst['mass'] > 32.7 else 0.0
                
                obs.extend([rel_x, rel_y, dist_norm, mass_norm, friction, shape_id, is_static])
            else:
                # No obstacle at this index
                obs.extend([0.0] * OBS_PER_OBSTACLE)
        
        return obs
    
    def _encode_shape(self, shape: str) -> float:
        """Encode shape as normalized float."""
        shapes = ["rectangle", "square", "trapezoid", "sphere", "diamond", "oval", "cylinder"]
        return shapes.index(shape) / len(shapes) if shape in shapes else 0.0
    
    def _get_foot_contacts(self) -> List[float]:
        """Get binary contact state for each foot."""
        # TODO: Implement foot contact sensing via Isaac Sim contact sensors
        # For now, return dummy data
        return [0.0, 0.0, 0.0, 0.0]
    
    def _get_collision_magnitude(self) -> float:
        """Get total collision force magnitude."""
        # TODO: Implement collision force reading from PhysX
        # For now, return dummy data
        return 0.0
    
    def _get_roll_pitch(self, quat: np.ndarray) -> Tuple[float, float]:
        """Convert quaternion to roll and pitch angles."""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        sinp = max(-1.0, min(1.0, sinp))  # Clamp
        pitch = math.asin(sinp)
        
        return roll, pitch
    
    def _distance_2d(self, a: np.ndarray, b: np.ndarray) -> float:
        """2D Euclidean distance."""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def _calculate_heading_error(self, pos: np.ndarray, quat: np.ndarray, 
                                   target_pos: np.ndarray) -> float:
        """Calculate heading error to target."""
        # Desired heading
        dx = target_pos[0] - pos[0]
        dy = target_pos[1] - pos[1]
        desired_yaw = math.atan2(dy, dx)
        
        # Current heading from quaternion
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        current_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Heading error (normalized to [-π, π])
        error = desired_yaw - current_yaw
        while error > math.pi:
            error -= 2 * math.pi
        while error < -math.pi:
            error += 2 * math.pi
        
        return error
    
    # ─────────────────────────────────────────────────────────────────────────
    # REWARD FUNCTION
    # ─────────────────────────────────────────────────────────────────────────
    
    def calculate_reward(self, action: np.ndarray, terminated: bool, 
                         info: Dict) -> float:
        """
        Calculate reward for current step.
        
        Args:
            action: Action taken [vx, vy, omega]
            terminated: Whether episode ended
            info: Additional information dict
        
        Returns:
            float: Total reward for this step
        """
        reward = 0.0
        spot_pos, spot_quat = self.spot.get_world_pose()
        
        # ─────────────────────────────────────────────────────────────────────
        # WAYPOINT PROGRESS
        # ─────────────────────────────────────────────────────────────────────
        if self.current_waypoint_idx < len(self.waypoint_mgr.waypoints):
            wp = self.waypoint_mgr.waypoints[self.current_waypoint_idx]
            current_dist = self._distance_2d(spot_pos[:2], wp["pos"])
            
            # Reward for reducing distance
            if self.prev_waypoint_dist is not None:
                dist_reduction = self.prev_waypoint_dist - current_dist
                reward += self.rewards.distance_reduction * dist_reduction
            
            self.prev_waypoint_dist = current_dist
            
            # Large bonus for reaching waypoint
            if info.get('waypoint_reached', False):
                reward += self.rewards.waypoint_reached
        
        # ─────────────────────────────────────────────────────────────────────
        # LOCOMOTION QUALITY
        # ─────────────────────────────────────────────────────────────────────
        vx, vy, omega = action[0], action[1], action[2]
        
        # Encourage forward locomotion
        if vx > 0:
            reward += self.rewards.forward_locomotion * vx
        else:
            reward -= self.rewards.backward_penalty * abs(vx)
        
        # Penalize excessive lateral motion
        reward -= self.rewards.lateral_penalty * abs(vy)
        
        # ─────────────────────────────────────────────────────────────────────
        # STABILITY
        # ─────────────────────────────────────────────────────────────────────
        roll, pitch = self._get_roll_pitch(spot_quat)
        tilt = max(abs(roll), abs(pitch))
        
        # Reward upright orientation
        if tilt < 0.1:  # < ~6 degrees
            reward += self.rewards.stability_reward
        else:
            reward -= self.rewards.stability_reward * (tilt / math.pi)
        
        # Height stability
        target_height = 0.7  # Spot normal standing height
        height_error = abs(spot_pos[2] - target_height)
        reward -= self.rewards.height_deviation * height_error
        
        # ─────────────────────────────────────────────────────────────────────
        # OBSTACLE INTERACTION
        # ─────────────────────────────────────────────────────────────────────
        if info.get('nudged_obstacle', False):
            reward += self.rewards.successful_nudge
        
        if info.get('bypassed_obstacle', False):
            reward += self.rewards.smart_bypass
        
        if info.get('stuck_pushing', False):
            reward -= self.rewards.failed_nudge_penalty
        
        # Collision penalty
        collision_force = self._get_collision_magnitude()
        reward -= self.rewards.collision_penalty * collision_force
        
        # ─────────────────────────────────────────────────────────────────────
        # ENERGY EFFICIENCY
        # ─────────────────────────────────────────────────────────────────────
        # Penalize jerky movements
        action_change = np.linalg.norm(action - self.prev_actions)
        reward -= self.rewards.action_smoothness * action_change
        
        # Penalize high action magnitudes
        action_magnitude = np.linalg.norm(action)
        reward -= self.rewards.energy_penalty * action_magnitude
        
        # ─────────────────────────────────────────────────────────────────────
        # TERMINAL CONDITIONS
        # ─────────────────────────────────────────────────────────────────────
        if terminated:
            if info.get('fell', False):
                reward -= self.rewards.fall_penalty
            elif info.get('timeout', False):
                reward -= self.rewards.timeout_penalty
        
        # ─────────────────────────────────────────────────────────────────────
        # TIME PENALTY (encourage efficiency)
        # ─────────────────────────────────────────────────────────────────────
        reward -= self.rewards.time_penalty
        
        return reward
    
    # ─────────────────────────────────────────────────────────────────────────
    # RESET AND STEP
    # ─────────────────────────────────────────────────────────────────────────
    
    def reset(self) -> np.ndarray:
        """
        Reset environment for new episode.
        
        Returns:
            np.ndarray: Initial observation
        """
        self.episode_num += 1
        self.current_waypoint_idx = 0
        self.waypoints_reached = 0
        self.episode_start_time = self.world.current_time
        self.episode_length = 0
        self.episode_reward = 0.0
        
        self.prev_waypoint_dist = None
        self.prev_actions = np.zeros(ACTION_DIM)
        self.prev_position = np.zeros(3)
        
        self.contact_tracker.clear()
        
        self.episode_stats = {
            'waypoints_reached': 0,
            'obstacles_nudged': 0,
            'obstacles_bypassed': 0,
            'total_contacts': 0,
            'avg_stability': 0.0,
            'total_distance': 0.0,
            'termination_reason': None
        }
        
        # Reset Spot
        if self.spot:
            self.spot.post_reset()
        
        # Get initial observation
        obs = self.get_observations()
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Action vector [vx, vy, omega]
        
        Returns:
            observation: Next state observation
            reward: Reward for this step
            terminated: Whether episode ended
            info: Additional information dict
        """
        self.episode_length += 1
        info = {}
        
        # Apply action to Spot
        # TODO: Convert RL action to Spot controller command
        # For now, pass directly
        command = action  # [vx, vy, omega]
        self.spot.advance(self.world.get_physics_dt(), command)
        
        # Step physics
        self.world.step(render=True)
        
        # Get current state
        spot_pos, spot_quat = self.spot.get_world_pose()
        
        # Check termination conditions
        terminated = False
        
        # Fall detection
        if spot_pos[2] < 0.3:  # Below safe height threshold
            terminated = True
            info['fell'] = True
            self.episode_stats['termination_reason'] = 'fell'
        
        # Timeout (max steps per episode)
        if self.episode_length >= 3000:  # ~5 minutes at 10Hz
            terminated = True
            info['timeout'] = True
            self.episode_stats['termination_reason'] = 'timeout'
        
        # Waypoint progress
        if self.current_waypoint_idx < len(self.waypoint_mgr.waypoints):
            wp = self.waypoint_mgr.waypoints[self.current_waypoint_idx]
            dist = self._distance_2d(spot_pos[:2], wp["pos"])
            
            if dist < 3.0:  # Waypoint reached
                self.waypoints_reached += 1
                self.current_waypoint_idx += 1
                info['waypoint_reached'] = True
                self.episode_stats['waypoints_reached'] = self.waypoints_reached
                
                # Check completion
                if self.current_waypoint_idx >= len(self.waypoint_mgr.waypoints):
                    terminated = True
                    info['completed'] = True
                    self.episode_stats['termination_reason'] = 'completed'
        
        # Calculate reward
        reward = self.calculate_reward(action, terminated, info)
        self.episode_reward += reward
        
        # Get next observation
        obs = self.get_observations()
        
        # Update previous state
        self.prev_actions = action.copy()
        self.prev_position = spot_pos.copy()
        
        # Log episode if terminated
        if terminated:
            self._log_episode()
        
        return obs, reward, terminated, info
    
    # ─────────────────────────────────────────────────────────────────────────
    # LOGGING
    # ─────────────────────────────────────────────────────────────────────────
    
    def _log_episode(self):
        """Log episode statistics."""
        episode_data = {
            'episode': self.episode_num,
            'total_reward': self.episode_reward,
            'waypoints_reached': self.episode_stats['waypoints_reached'],
            'obstacles_nudged': self.episode_stats['obstacles_nudged'],
            'obstacles_bypassed': self.episode_stats['obstacles_bypassed'],
            'total_contacts': self.episode_stats['total_contacts'],
            'episode_length': self.episode_length,
            'termination': self.episode_stats['termination_reason']
        }
        self.episode_log.append(episode_data)
        
        print(f"[RL] Episode {self.episode_num} | Reward: {self.episode_reward:.2f} | "
              f"Waypoints: {self.episode_stats['waypoints_reached']} | "
              f"Length: {self.episode_length} | "
              f"Term: {self.episode_stats['termination_reason']}")
    
    def save_episode_log(self, filepath: str):
        """Save episode log to CSV."""
        if not self.episode_log:
            return
        
        keys = self.episode_log[0].keys()
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.episode_log)
        
        print(f"[RL] Episode log saved: {filepath}")
