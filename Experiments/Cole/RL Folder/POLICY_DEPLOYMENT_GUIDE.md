"""
POLICY DEPLOYMENT GUIDE
=======================

This guide explains how to capture and use trained RL policies in testing environments.

TABLE OF CONTENTS
-----------------
1. Understanding Checkpoints
2. Loading a Trained Policy
3. Using Policy in Testing Environments
4. Checkpoint Management


================================================================================
1. UNDERSTANDING CHECKPOINTS
================================================================================

During training, three types of checkpoints are saved:

A. STAGE CHECKPOINTS (after completing each curriculum stage)
   - stage_1_complete.pt  → Completed 10 consecutive 25-waypoint episodes (no obstacles)
   - stage_2_complete.pt  → Completed Stage 2 (10% light/medium obstacles)
   - stage_3_complete.pt  → Completed Stage 3 (20% total obstacles)
   - stage_4_complete.pt  → Completed Stage 4 (30% total obstacles)

B. FINAL POLICY (after all stages complete)
   - FULLY_TRAINED_POLICY.pt  → Ready for deployment! 🎓

C. PERIODIC CHECKPOINTS (every N iterations)
   - model_0.pt, model_100.pt, model_200.pt, etc.
   - Used for recovery if training crashes

Location: checkpoints/spot_rl_curriculum/


Checkpoint Contents:
--------------------
All checkpoints contain:
  • actor_state_dict       → Policy network weights (THIS IS WHAT YOU NEED!)
  • critic_state_dict      → Value function weights (only needed for continued training)
  • optimizer_state_dict   → Adam optimizer state (only for continued training)
  • iteration              → Training iteration when saved
  • stage                  → Curriculum stage (1-4)
  • consecutive_successes  → How many successful episodes at that stage

FULLY_TRAINED_POLICY.pt also includes:
  • obs_dim               → Observation space dimension (for reconstructing network)
  • action_dim            → Action space dimension
  • config                → Full training configuration


================================================================================
2. LOADING A TRAINED POLICY
================================================================================

Method 1: Use the provided utility function (RECOMMENDED)
----------------------------------------------------------

from load_trained_policy import load_trained_policy

# Load the fully trained policy
actor, info = load_trained_policy(
    checkpoint_path="checkpoints/spot_rl_curriculum/FULLY_TRAINED_POLICY.pt",
    device="cuda"  # or "cpu"
)

print(f"Loaded policy from iteration {info['iteration']}")
print(f"Observation dim: {info['obs_dim']}")
print(f"Action dim: {info['action_dim']}")


Method 2: Manual loading (if you need more control)
----------------------------------------------------

import torch
from load_trained_policy import ActorNetwork

# Load checkpoint
checkpoint = torch.load("checkpoints/spot_rl_curriculum/FULLY_TRAINED_POLICY.pt")

# Create network with same architecture
actor = ActorNetwork(
    obs_dim=checkpoint['obs_dim'],
    action_dim=checkpoint['action_dim']
).to('cuda')

# Load weights
actor.load_state_dict(checkpoint['actor_state_dict'])
actor.eval()  # IMPORTANT: Set to evaluation mode!


Method 3: Find available checkpoints
-------------------------------------

from load_trained_policy import list_available_checkpoints

# List all checkpoints in directory
checkpoints = list_available_checkpoints("checkpoints/spot_rl_curriculum")


================================================================================
3. USING POLICY IN TESTING ENVIRONMENTS
================================================================================

Option A: Use with existing RL_Development.py environment
----------------------------------------------------------

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from RL_Development import CircularWaypointEnv
from load_trained_policy import load_trained_policy
import torch

# 1. Load trained policy
actor, info = load_trained_policy(
    checkpoint_path="checkpoints/spot_rl_curriculum/FULLY_TRAINED_POLICY.pt"
)

# 2. Create environment (set to Stage 4 for full obstacles)
env = CircularWaypointEnv(
    num_robots=16,
    training_stage=4,  # Use hardest stage for testing
    headless=False
)

env.reset()

# 3. Run policy
print("Running trained policy...")
for step in range(10000):
    # Get current observations
    observations = env.get_observations()  # Shape: (16, obs_dim)
    
    # Get actions from policy (NO GRADIENT needed for inference)
    with torch.no_grad():
        actions = actor(observations)  # Shape: (16, action_dim)
    
    # Apply actions
    env.apply_rl_action(actions)
    
    # Step simulation
    env.world.step(render=True)
    
    # Optional: Check episode completion
    dones = env._compute_dones()
    if dones.any():
        print(f"Step {step}: Some episodes finished!")

simulation_app.close()


Option B: Integrate into your own testing environment
------------------------------------------------------

# In your Testing_Environment_X.py file:

from load_trained_policy import load_trained_policy
import torch

# Load policy once at startup
ACTOR, policy_info = load_trained_policy(
    checkpoint_path="../checkpoints/spot_rl_curriculum/FULLY_TRAINED_POLICY.pt"
)

# In your simulation loop:
def run_testing():
    for step in range(10000):
        # Get observations from your Spot robots
        # (You need to compute these - see RL_Development.py for reference)
        observations = compute_spot_observations()  # Shape: (num_robots, 71)
        
        # Get actions from trained policy
        with torch.no_grad():
            actions = ACTOR(observations)  # Shape: (num_robots, 12)
        
        # Apply actions to your robots
        apply_actions_to_spots(actions)
        
        # Step simulation
        world.step(render=True)


IMPORTANT: Observation Space
-----------------------------
Your testing environment MUST provide the same observations as training!

Expected observation vector (71 dimensions):
  • Base linear velocity (3)       → spot.get_linear_velocity()
  • Base angular velocity (3)      → spot.get_angular_velocity()
  • Projected gravity (3)          → Gravity in robot's local frame
  • Joint positions (12)           → spot.get_joint_positions()
  • Joint velocities (12)          → spot.get_joint_velocities()
  • Previous actions (12)          → Store from last step
  • Target waypoint direction (2)  → [dx, dy] to next waypoint
  • Distance to waypoint (1)       → Euclidean distance
  • Waypoints completed (1)        → How many reached so far
  • Commands (3)                   → [forward, lateral, yaw_rate] targets
  • Feet contact states (4)        → [FL, FR, RL, RR] boolean
  • Feet air time (4)              → Time since last contact
  • Collision flag (1)             → 1.0 if collision, 0.0 otherwise
  • Obstacle clearance (1)         → Minimum distance to obstacles
  • Gait phase (4)                 → Sin/cos encoding
  • Feet positions (12)            → [x,y,z] × 4 feet
  • Feet velocities (12)           → [vx,vy,vz] × 4 feet

Total: 71 dimensions


Action Space
------------
Policy outputs 12 actions (joint position targets):
  • Front Left  [Hip, Thigh, Calf]  → actions[0:3]
  • Front Right [Hip, Thigh, Calf]  → actions[3:6]
  • Rear Left   [Hip, Thigh, Calf]  → actions[6:9]
  • Rear Right  [Hip, Thigh, Calf]  → actions[9:12]

Actions are in range [-1, 1] and must be scaled to joint limits.


================================================================================
4. CHECKPOINT MANAGEMENT
================================================================================

Finding the Best Checkpoint
----------------------------
If training was interrupted or you want to test intermediate stages:

1. Check training logs for highest rewards:
   - Look at "Mean Episode Reward" in logs/spot_rl_curriculum/training.log

2. Use stage checkpoints for progressive testing:
   - stage_1_complete.pt → Can navigate to waypoints without obstacles
   - stage_2_complete.pt → Can avoid light/medium moveable obstacles
   - stage_3_complete.pt → Can handle heavy immovable obstacles
   - stage_4_complete.pt → Can navigate through all obstacle types

3. Use FULLY_TRAINED_POLICY.pt for final deployment


Resuming Training
-----------------
To continue training from a checkpoint:

# In train_spot_ppo.py, modify the train() function:

resume_checkpoint = "checkpoints/spot_rl_curriculum/stage_2_complete.pt"
checkpoint = torch.load(resume_checkpoint)

# Load states
actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_iteration = checkpoint['iteration']
current_stage = checkpoint['stage']
consecutive_successes = checkpoint.get('consecutive_successes', 0)


Checkpoint Backups
------------------
IMPORTANT: Back up your checkpoints regularly!

# Copy to safe location:
cp -r checkpoints/spot_rl_curriculum/ ~/backups/spot_rl_$(date +%Y%m%d)/

Or use version control:
git lfs track "*.pt"
git add checkpoints/
git commit -m "Save trained policy - Stage 4 complete"


================================================================================
QUICK REFERENCE COMMANDS
================================================================================

# List available checkpoints
python load_trained_policy.py

# Test loading a checkpoint
python -c "from load_trained_policy import load_trained_policy; actor, info = load_trained_policy('checkpoints/spot_rl_curriculum/FULLY_TRAINED_POLICY.pt'); print(info)"

# Deploy in testing environment
python Testing_Environment_X.py --use-trained-policy --checkpoint FULLY_TRAINED_POLICY.pt


================================================================================
TROUBLESHOOTING
================================================================================

Problem: "Checkpoint not found"
Solution: Check that training has completed at least one stage.
          Checkpoints are saved in: checkpoints/spot_rl_curriculum/

Problem: "Observation dimension mismatch"
Solution: Ensure your testing environment computes the exact same 71-dimensional
          observation vector as RL_Development.py

Problem: "Robot falls immediately with trained policy"
Solution: 1. Check if you're using FULLY_TRAINED_POLICY.pt (all stages complete)
          2. Verify observation computation matches training environment
          3. Ensure action scaling is correct (-1 to 1 range)
          4. Check if PD gains match training configuration

Problem: "Actions seem random"
Solution: Did you call actor.eval() and use torch.no_grad()?
          Training mode includes exploration noise!

Problem: "Want to test Stage 2 policy only"
Solution: Load stage_2_complete.pt instead of FULLY_TRAINED_POLICY.pt
          Use env.set_training_stage(2) to match training conditions


================================================================================
NEXT STEPS
================================================================================

After loading your trained policy:

1. ✅ Verify it works in RL_Development.py with Stage 4 obstacles
2. ✅ Test in your specific testing environments  
3. ✅ Benchmark performance metrics (waypoints/minute, collision rate, etc.)
4. ✅ Compare Stage 1/2/3/4 checkpoints to see progression
5. ✅ Fine-tune on new terrain types if needed (transfer learning)


Good luck deploying your trained Spot policy! 🐕🤖
"""