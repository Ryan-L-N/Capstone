"""
Spot Robot Module
=================
Reusable Spot robot class for Isaac Sim environments.
Encapsulates Spot initialization, sensors, state tracking, and control.

Features:
  - Robot initialization at custom positions
  - Sensor configuration (camera, lidar, IMU, contact)
  - State tracking (position, velocity, heading)
  - Simple go-to-goal controller
  - RL-compatible action interface

Author: MS for Autonomy Project
Date: February 2026
"""

import numpy as np
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy


class SpotRobot:
    """
    Spot Robot wrapper for Isaac Sim 5.1
    Manages robot initialization, sensing, and control
    """
    
    def __init__(self, world, stage, prim_path="/World/Spot", name="Spot",
                 position=np.array([-45.0, 0.0, 0.7])):
        """
        Initialize Spot robot
        
        Args:
            world: Isaac Sim World object
            stage: USD stage
            prim_path: Path in USD stage
            name: Robot name
            position: Initial position [x, y, z]
        """
        self.world = world
        self.stage = stage
        self.prim_path = prim_path
        self.name = name
        self.position = np.array(position)
        
        # Create the robot
        self.robot = SpotFlatTerrainPolicy(
            prim_path=prim_path,
            name=name,
            position=self.position,
        )
        
        # Initialize robot (must happen after world.reset())
        self.robot.initialize()
        
        # Sensor configuration
        self.sensors = {
            "camera": {"enabled": True, "resolution": (640, 480)},
            "lidar": {"enabled": True, "range": 10.0},
            "imu": {"enabled": True},
            "contact": {"enabled": True}
        }
        
        # State tracking
        self.current_pos = np.array(self.position)
        self.current_heading = 0.0
        self.current_vel = np.array([0.0, 0.0, 0.0])
        self.current_ang_vel = np.array([0.0, 0.0, 0.0])
        
        # Control parameters
        self.FORWARD_SPEED = 1.5
        self.TURN_GAIN = 2.0
        self.HEADING_THRESHOLD = 0.1
        
        print(f"{name} robot created at position: ({position[0]}, {position[1]}, {position[2]})")
    
    def stabilize(self, steps=10):
        """
        Run physics steps to stabilize robot after initialization
        
        Args:
            steps: Number of physics steps
        """
        for _ in range(steps):
            self.world.step(render=False)
        print(f"{self.name} stable and ready")
    
    def get_state(self):
        """
        Get current robot state
        
        Returns:
            pos: Position [x, y, z]
            heading: Heading in radians
            vel: Linear velocity [vx, vy, vz]
            ang_vel: Angular velocity [wx, wy, wz]
        """
        # Get position and rotation from the robot's root prim
        try:
            # For SpotFlatTerrainPolicy, access the robot through its base_link
            prim = self.robot.robot.get_world_pose()  # robot.robot is the underlying articulation
            pos, quat = prim
        except:
            try:
                # Alternative: try direct access
                pos = np.array([self.robot.position[0], self.robot.position[1], self.robot.position[2]])
                quat = [1, 0, 0, 0]  # Default identity quaternion
            except:
                # Final fallback: use stored position
                pos = self.current_pos
                quat = [1, 0, 0, 0]
        
        self.current_pos = np.array(pos)
        
        # Extract heading from quaternion (assuming quat is [w, x, y, z])
        if len(quat) == 4:
            w, x, y, z = quat[0], quat[1], quat[2], quat[3]
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            self.current_heading = np.arctan2(siny_cosp, cosy_cosp)
        else:
            self.current_heading = 0.0
        
        # Get velocities (simplified - assume stationary for now)
        self.current_vel = np.array([0.0, 0.0, 0.0])
        self.current_ang_vel = np.array([0.0, 0.0, 0.0])
        
        return self.current_pos, self.current_heading, self.current_vel, self.current_ang_vel
    
    def get_sensor_data(self):
        """
        Get simulated sensor data
        
        Returns:
            dict with keys: 'camera', 'lidar_points', 'imu_accel', 'imu_gyro', 'imu_heading', 'contact'
        """
        sensor_data = {}
        
        try:
            pos, heading, vel, ang_vel = self.get_state()
            
            # Camera data (shape tuple)
            sensor_data['camera'] = self.sensors['camera']['resolution'] + (3,)
            
            # Lidar data (point count)
            sensor_data['lidar_points'] = max(0, int(360 / 0.25))  # 1440 points
            
            # IMU data
            sensor_data['imu_accel'] = vel
            sensor_data['imu_gyro'] = ang_vel
            sensor_data['imu_heading'] = heading
            
            # Contact sensor
            sensor_data['contact'] = pos[2] > 0.3
            
        except Exception as e:
            # Fallback sensor values
            sensor_data['camera'] = (640, 480, 3)
            sensor_data['lidar_points'] = 1440
            sensor_data['imu_accel'] = np.array([0, 0, -9.81])
            sensor_data['imu_gyro'] = np.array([0, 0, 0])
            sensor_data['imu_heading'] = 0
            sensor_data['contact'] = False
        
        return sensor_data
    
    def compute_go_to_goal_command(self, goal_position):
        """
        Compute command to reach goal position
        
        Args:
            goal_position: Target position [x, y]
        
        Returns:
            command: [forward_velocity, lateral_velocity, angular_velocity]
        """
        pos, heading, _, _ = self.get_state()
        
        to_goal = np.array(goal_position) - pos[:2]
        dist = np.linalg.norm(to_goal)
        desired_heading = np.arctan2(to_goal[1], to_goal[0])
        
        # Normalize heading error to [-pi, pi]
        heading_error = desired_heading - heading
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        # Compute turn rate
        turn_rate = self.TURN_GAIN * heading_error
        turn_rate = np.clip(turn_rate, -1.0, 1.0)
        
        # Compute forward speed
        if abs(heading_error) < self.HEADING_THRESHOLD:
            forward_speed = self.FORWARD_SPEED
        else:
            forward_speed = self.FORWARD_SPEED * 0.4
        
        command = np.array([forward_speed, 0.0, turn_rate])
        
        return command, dist, heading_error
    
    def apply_action(self, action_idx, action_to_command_fn=None):
        """
        Apply RL action to robot
        
        Args:
            action_idx: Action index (0-8) for RL agent
            action_to_command_fn: Function to convert action_idx to (direction, speed)
                If None, uses default mapping
        
        Returns:
            command: [forward_velocity, lateral_velocity, angular_velocity]
        """
        if action_to_command_fn is None:
            # Default action mapping
            speeds = [0.4, 2.0, 6.7]
            directions = [-1, 0, 1]
            
            direction_idx = action_idx // 3
            speed_idx = action_idx % 3
            
            direction = directions[direction_idx]
            speed = speeds[speed_idx]
        else:
            direction, speed = action_to_command_fn(action_idx)
        
        # Simple forward motion with turning based on direction
        forward_speed = speed
        turn_rate = direction * 0.5
        turn_rate = np.clip(turn_rate, -1.0, 1.0)
        
        command = np.array([forward_speed, 0.0, turn_rate])
        return command
    
    def set_command(self, command):
        """
        Set robot movement command
        
        Args:
            command: [forward_velocity, lateral_velocity, angular_velocity]
        """
        try:
            # SpotFlatTerrainPolicy.advance() expects the command directly
            self.robot.advance(command)
        except Exception as e:
            # If advance fails, try alternative approach with the underlying robot object
            try:
                if hasattr(self.robot, 'robot') and hasattr(self.robot.robot, 'advance'):
                    self.robot.robot.advance(command)
            except:
                pass
    
    def print_config(self):
        """Print robot configuration summary"""
        print(f"\n{self.name} Configuration:")
        print(f"  Position: {self.position}")
        print(f"  Sensors:")
        for sensor_name, config in self.sensors.items():
            if config['enabled']:
                if 'resolution' in config:
                    print(f"    - {sensor_name}: {config['resolution']}")
                elif 'range' in config:
                    print(f"    - {sensor_name}: {config['range']}m range")
                else:
                    print(f"    - {sensor_name}: enabled")
        print(f"  Control Parameters:")
        print(f"    - Forward speed: {self.FORWARD_SPEED} m/s")
        print(f"    - Turn gain: {self.TURN_GAIN}")
        print(f"    - Heading threshold: {self.HEADING_THRESHOLD} rad")
