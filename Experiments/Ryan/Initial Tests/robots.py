
spots = []
for i in range(config.NUM_ROBOTS):
    y = config.START_Y_POSITIONS[i]
    spot = SpotFlatTerrainPolicy(
        prim_path=f"/World/Spot_{i}",
        name=f"Spot_{i}",
        position=np.array([config.START_X, y, config.START_Z]),
    )
    spots.append(spot)
    print(f"Created Spot_{i} at ({config.START_X}, {y}, {config.START_Z})")

# CRITICAL: Reset world BEFORE accessing spot properties
world.reset()

# Initialize all robots
for i, spot in enumerate(spots):
    spot.initialize()
    spot.robot.set_joints_default_state(spot.default_pos)
print(f"All {config.NUM_ROBOTS} robots initialized")

# Run physics steps for stability
for _ in range(20):
    world.step(render=False)
print("Robots stable and ready")

# =============================================================================
# NAVIGATION POLICY
# =============================================================================

class NavigationPolicy:
    """Simple policy for goal-directed navigation.
    
    Uses a proportional controller plus learned adjustments.
    """
    
    def __init__(self):
        # Learning parameters
        self.forward_gain = 1.5   # Base forward speed when aligned
        self.turn_gain = 2.0      # Proportional gain for turning
        self.lateral_gain = 0.3   # Small lateral corrections
        
        # Learned adjustments (start at 0)
        self.forward_bias = 0.0
        self.turn_bias = 0.0
        
        # Experience for learning
        self.episode_progress = []
        self.lr = 0.01
    
    def get_action(self, goal_dir, heading_error, dist_to_goal):
        """Compute velocity command to reach goal.
        
        Args:
            goal_dir: Normalized direction to goal [x, y]
            heading_error: Angle error to goal (radians)
            dist_to_goal: Distance to goal (meters)
        
        Returns:
            [forward_vel, lateral_vel, yaw_rate]
        """
        # Proportional turn control - turn toward goal
        turn_rate = self.turn_gain * heading_error + self.turn_bias
        turn_rate = np.clip(turn_rate, -config.MAX_YAW_RATE, config.MAX_YAW_RATE)
        
        # Forward speed: full speed if aligned, reduced if turning
        alignment = np.cos(heading_error)  # 1 when aligned, -1 when backwards
        if abs(heading_error) < 0.2:  # Well aligned (< ~11 degrees)
            forward_speed = config.MAX_FORWARD_VEL * (self.forward_gain + self.forward_bias)
        elif abs(heading_error) < 0.5:  # Somewhat aligned
            forward_speed = config.MAX_FORWARD_VEL * 0.7
        else:  # Need to turn significantly
            forward_speed = config.MAX_FORWARD_VEL * 0.3
        
        forward_speed = np.clip(forward_speed, 0, config.MAX_FORWARD_VEL)
        
        # Small lateral correction toward goal
        lateral = self.lateral_gain * goal_dir[1] * alignment
        lateral = np.clip(lateral, -config.MAX_LATERAL_VEL, config.MAX_LATERAL_VEL)
        
        return np.array([forward_speed, lateral, turn_rate])
    
    def update(self, progress):
        """Update policy based on episode progress."""
        self.episode_progress.append(progress)
        if len(self.episode_progress) >= 5:
            recent = np.mean(self.episode_progress[-5:])
            if recent < 30:  # Not making good progress
                self.forward_gain = min(2.0, self.forward_gain + self.lr)
            self.episode_progress = self.episode_progress[-10:]
    
    def save(self, path):
        np.savez(path, forward_gain=self.forward_gain, turn_gain=self.turn_gain,
                 forward_bias=self.forward_bias, turn_bias=self.turn_bias)
        print(f"Policy saved to {path}")
    
    def load(self, path):
        if os.path.exists(path):
            data = np.load(path)
            self.forward_gain = float(data['forward_gain'])
            self.turn_gain = float(data['turn_gain'])
            print(f"Policy loaded from {path}")

# Create policy
policy = NavigationPolicy()
if args.load_checkpoint:
    policy.load(args.load_checkpoint)

# =============================================================================
# ROBOT STATE TRACKING
# =============================================================================

class RobotState:
    def __init__(self, robot_id, spot, start_y):
        self.id = robot_id
        self.spot = spot
        self.start_pos = np.array([config.START_X, start_y, config.START_Z])
        self.goal_pos = np.array([config.GOAL_X, start_y])
        self.prev_dist = config.GOAL_X - config.START_X
        self.episode_reward = 0.0
        self.episode_progress = 0.0
        self.goal_reached = False
        self.fell = False
    
    def get_state(self):
        """Get position, heading, velocity."""
        pos, quat = self.spot.robot.get_world_pose()
        vel = self.spot.robot.get_linear_velocity()
        
        # Extract yaw from quaternion
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        heading = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array(pos), heading, np.array(vel)
    
    def compute_goal_info(self):
        """Compute goal direction and heading error."""
        pos, heading, vel = self.get_state()
        
        to_goal = self.goal_pos - pos[:2]
        dist = np.linalg.norm(to_goal)
        goal_dir = to_goal / (dist + 1e-6)
        
        # Heading error (desired - actual)
        desired_heading = np.arctan2(to_goal[1], to_goal[0])
        heading_error = desired_heading - heading
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        return goal_dir, heading_error, dist, pos, heading, vel
    
    def compute_reward(self, dist, pos, heading, vel, action):
        """Compute reward for this step."""
        reward = 0.0
        
        # Progress toward goal (X direction)
        progress = self.prev_dist - dist
        reward += config.PROGRESS_SCALE * progress
        
        # Velocity toward goal
        goal_dir = (self.goal_pos - pos[:2]) / (np.linalg.norm(self.goal_pos - pos[:2]) + 1e-6)
        vel_toward_goal = np.dot(vel[:2], goal_dir)
        reward += config.VELOCITY_SCALE * max(0, vel_toward_goal)
        
        # Facing goal
        desired_heading = np.arctan2(goal_dir[1], goal_dir[0])
        heading_error = abs(desired_heading - heading)
        if heading_error > np.pi:
            heading_error = 2 * np.pi - heading_error
        facing_bonus = 1.0 - heading_error / np.pi
        reward += config.ORIENTATION_SCALE * facing_bonus
        
        # Alive
        if pos[2] > 0.4:
            reward += config.ALIVE_REWARD
        
        # Action cost
        reward -= config.ACTION_COST * np.sum(action**2)
        
        self.prev_dist = dist
        self.episode_reward += reward
        return reward
    
    def reset(self):
        """Reset robot to starting position."""
        self.spot.robot.set_world_pose(
            position=self.start_pos,
            orientation=np.array([1.0, 0.0, 0.0, 0.0])  # Facing +X
        )
        self.spot.robot.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
        self.spot.robot.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
        self.prev_dist = config.GOAL_X - config.START_X
        self.episode_progress = 0.0
        self.episode_reward = 0.0
        self.goal_reached = False
        self.fell = False

# Create robot states
robot_states = [
    RobotState(i, spots[i], config.START_Y_POSITIONS[i])
    for i in range(config.NUM_ROBOTS)
]
