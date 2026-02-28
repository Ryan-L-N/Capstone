# How to Train Your RAWDOG

### A Field Guide to Teaching Robots to Walk (and Fall, and Walk Again)

*AI2C Tech Capstone -- MS for Autonomy, Carnegie Mellon University, February 2026*

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [The 30-Second Version](#2-the-30-second-version)
3. [The Cast of Characters](#3-the-cast-of-characters)
4. [How the Codebase Is Wired Together](#4-how-the-codebase-is-wired-together)
5. [The Technology Stack (and Why Each Piece Exists)](#5-the-technology-stack-and-why-each-piece-exists)
6. [How Training Actually Works](#6-how-training-actually-works)
7. [The Three-Phase Training Pipeline](#7-the-three-phase-training-pipeline)
8. [The Evaluation Gauntlet](#8-the-evaluation-gauntlet)
9. [The Bug Museum: Every Disaster and How We Survived](#9-the-bug-museum-every-disaster-and-how-we-survived)
10. [Lessons That Will Save Your Future Self](#10-lessons-that-will-save-your-future-self)
11. [How Good Engineers Think](#11-how-good-engineers-think)
12. [Training Run Log](#12-training-run-log)
13. [Quick Reference: Commands and Paths](#13-quick-reference-commands-and-paths)

---

## 1. What Is This Project?

We taught two quadruped robots -- Boston Dynamics **Spot** and Ghost Robotics **Vision60** -- to walk across brutal terrain using reinforcement learning. Not just "walk on a flat floor" walk, but "traverse ice, climb stairs, wade through knee-high grass, and stumble over boulders without face-planting" walk.

The robots don't get a manual. They don't get pre-programmed gaits. They get dropped into a simulated world with 20,000 copies of themselves, and they learn by trial and error -- roughly 10 billion trials per training run. The ones that walk well get rewarded. The ones that face-plant get penalized. After a few days on an NVIDIA H100 GPU, what emerges is a neural network that can handle terrain it has never seen before.

This document explains how the whole system works, how the code is structured, what went wrong (a lot), and what we learned from it.

---

## 2. The 30-Second Version

```
You have two robots.
You have 12 types of terrain.
You have 19 reward signals telling the robot what "good walking" means.
You simulate 10,000+ robots in parallel on a GPU.
Each robot gets 30 seconds per attempt.
You run 30,000+ attempts.
After ~24 hours of GPU time, the robots can walk.

That's it. Everything else is details about how to do this
without the whole thing catching fire.
```

---

## 3. The Cast of Characters

### The Robots

| | Boston Dynamics Spot | Ghost Robotics Vision60 |
|---|---|---|
| **Weight** | ~32 kg | ~13.6 kg |
| **Standing Height** | 0.42 m | 0.55 m |
| **Degrees of Freedom** | 12 (3 per leg) | 12 (3 per leg) |
| **Joint Stiffness (Kp)** | 60.0 | 80.0 |
| **Joint Damping (Kd)** | 1.5 | 2.0 |
| **Personality** | Tank. Low center of gravity, stable. | Giraffe. Tall, light, lanky. |
| **Foot Names (in code)** | `".*_foot"` | `"lower.*"` |

These two robots are different enough that you can't just copy-paste a policy from one to the other. Spot is a 32kg bulldog -- low, heavy, hard to knock over. Vision60 is a 13kg greyhound -- tall, light, and prone to toppling if you look at it funny. Same training framework, different parameters, different failure modes.

### The Terrains (12 Types)

Think of it as an obstacle course designed by someone who really doesn't like robots:

**Geometric Challenges (40% of training)**
- Pyramid stairs up/down -- standard staircases, 5-25cm risers
- Random grid boxes -- imagine a floor made of mismatched Lego bricks
- Stepping stones -- the "floor is lava" challenge
- Gaps -- literal holes in the ground

**Surface Challenges (35% of training)**
- Random rough ground -- bumpy gravel, noise everywhere
- Slopes up/down -- smooth inclines up to 27 degrees
- Wave terrain -- undulating ground like a frozen ocean
- Friction plane -- pure ice (friction as low as 0.05)
- Vegetation plane -- grass that grabs your feet like quicksand

**Compound Challenges (25% of training)**
- High-frequency stairs -- coarser, noisier versions of stairs
- Discrete obstacles -- scattered random blocks
- Repeated boxes -- regular obstacle patterns

All 12 types are arranged in a 10x40 grid of 8m patches -- 400 patches total. The rows represent difficulty (0 = easy, 9 = nightmare). Robots get promoted to harder rows when they perform well, and demoted when they eat dirt. It's an automatic curriculum: the simulation figures out each robot's skill level and gives it appropriately hard challenges.

### The Rewards (19 Signals)

The reward function is the secret sauce. It's a weighted sum of 19 terms that collectively define what "good locomotion" looks like. Think of it as a 19-dimensional report card that the robot gets every single timestep (every 0.02 seconds).

**Positive Rewards (7 terms) -- "Do more of this"**
| Term | Weight | What It Rewards |
|------|--------|----------------|
| `base_linear_velocity` | +12.0 | Moving at the commanded speed |
| `gait` | +15.0 | Maintaining a proper diagonal trot |
| `air_time` | +3.0 | Swinging feet through the air (not dragging) |
| `foot_clearance` | +2.0 | Lifting feet high enough to clear obstacles |
| `velocity_modulation` | +2.0 | Slowing down on hard terrain (smart!) |
| `base_angular_velocity` | +1.0 | Turning when commanded |
| `body_height_tracking` | +1.0 | Maintaining proper standing height |

**Negative Rewards (12 terms) -- "Stop doing this"**
| Term | Weight | What It Penalizes |
|------|--------|------------------|
| `base_orientation` | -5.0 | Tilting/rolling the body |
| `base_motion` | -4.0 | Unwanted vertical/lateral body motion |
| `foot_slip` | -3.0 | Feet sliding on the ground |
| `joint_pos` | -2.0 | Extreme joint angles |
| `dof_pos_limits` | -10.0 | Hitting mechanical joint limits |
| `joint_vel` | -0.05 | Jerky joint movements |
| `joint_torques` | -0.005 | Wasting motor effort |
| `action_smoothness` | -0.5 | Erratic control signals |
| `stumble` | -0.3 | Tripping on obstacles at knee height |
| `body_contact` | -10.0 | Body/leg touching the ground (falling) |
| `contact_force_smoothness` | -0.02 | Slamming feet down hard |
| `vegetation_drag` | -0.001 | Moving through vegetation (physics signal) |

Getting these weights right is an art. Too much penalty and the robot learns to stand still (the safest strategy). Too much reward and it flails wildly. We spent days tuning these, and Section 9 has the full story of what went wrong.

---

## 4. How the Codebase Is Wired Together

### Directory Map

```
multi_robot_training/
    |
    |-- shared/                     <-- Robot-agnostic utilities
    |   |-- terrain_cfg.py          <-- 12-terrain curriculum definition
    |   |-- scratch_terrain_cfg.py  <-- 7-terrain warmup (for testing)
    |   |-- reward_terms.py         <-- 5 custom reward functions
    |   |-- lr_schedule.py          <-- Cosine learning rate annealing
    |   |-- dr_schedule.py          <-- Progressive domain randomization
    |   |-- training_utils.py       <-- TF32 config, noise clamping
    |
    |-- configs/                    <-- Robot-specific configurations
    |   |-- robot_params.py         <-- RobotParams dataclass
    |   |-- spot_params.py          <-- Spot-specific constants
    |   |-- vision60_params.py      <-- Vision60-specific constants
    |   |-- spot_ppo_env_cfg.py     <-- Spot environment config (Phase 1)
    |   |-- vision60_ppo_env_cfg.py <-- Vision60 environment config (Phase 1)
    |   |-- spot_ppo_cfg.py         <-- Spot PPO runner config
    |   |-- vision60_ppo_cfg.py     <-- Vision60 PPO runner config
    |   |-- spot_teacher_env_cfg.py <-- Spot teacher config (Phase 2a)
    |   |-- vision60_teacher_env_cfg.py
    |   |-- distill_ppo_cfg.py      <-- Shared distillation config (Phase 2b)
    |
    |-- train_ppo.py                <-- Phase 1: Train a policy from scratch
    |-- train_teacher.py            <-- Phase 2a: Train teacher w/ privileged info
    |-- train_distill.py            <-- Phase 2b: Distill teacher into student
    |-- play.py                     <-- Visualize a trained policy
    |
    |-- eval/                       <-- Deployment wrappers
    |   |-- vision60_rough_terrain_policy.py
    |
    |-- INTEGRATION_PLAN.md         <-- Full technical integration spec
    |-- HOW_TO_TRAIN_YOUR_RAWDOG.md <-- You are here
```

### The Data Flow (How Everything Connects)

Here's the signal chain from "I want to train a robot" to "robot walks":

```
 [You]
   |
   | python train_ppo.py --robot spot --num_envs 10000
   |
   v
 [train_ppo.py]
   |
   |-- loads configs/spot_ppo_env_cfg.py     (what the world looks like)
   |-- loads configs/spot_ppo_cfg.py          (how training is configured)
   |-- loads shared/terrain_cfg.py            (12 terrain types)
   |-- loads shared/reward_terms.py           (5 custom rewards)
   |
   v
 [Isaac Lab]  <-- NVIDIA's robot simulation framework
   |
   |-- Creates 10,000 parallel environments on GPU
   |-- Each env has: robot + terrain + physics
   |-- Runs at 500 Hz physics, 50 Hz control (decimation = 10)
   |
   v
 [RSL-RL OnPolicyRunner]  <-- ETH Zurich's PPO implementation
   |
   |-- Collects 32 steps of experience from all 10K envs
   |-- That's 320,000 transitions per iteration
   |-- Computes advantages (GAE), updates policy via PPO
   |
   v
 [train_ppo.py's monkey-patch]  <-- Our custom additions
   |
   |-- Cosine LR annealing      (shared/lr_schedule.py)
   |-- Progressive DR            (shared/dr_schedule.py, Vision60 only)
   |-- Noise std clamping         (shared/training_utils.py)
   |-- W&B / TensorBoard logging
   |
   v
 [Checkpoint]  model_XXXXX.pt
   |
   |-- Can be evaluated in 4_env_test/ gauntlet
   |-- Can be fed into Phase 2 (teacher-student distillation)
   |-- Can be visualized with play.py
```

### Why "Shared" vs "Robot-Specific"?

The key architectural decision: **separate the physics of what makes terrain hard from the physics of what makes each robot different.**

Terrain is terrain. A staircase doesn't care if it's Spot or Vision60 climbing it. The 12-terrain curriculum, the vegetation drag physics, the learning rate schedule -- these are universal. They live in `shared/`.

But each robot has different feet, different weight, different joint limits, different failure modes. Spot's feet are called `".*_foot"` in the URDF; Vision60's are `"lower.*"`. Spot weighs 32kg and has a low center of gravity; Vision60 weighs 13.6kg and stands tall. These differences live in `configs/` as separate files.

The training script (`train_ppo.py`) is a single file parameterized by `--robot spot|vision60`. It dynamically imports the right config, plugs in the right body names, and runs the same training loop. One codebase, two robots, shared infrastructure.

This is good engineering. Here's why:

1. **Bug fixes propagate.** Fix a terrain config bug, and both robots get the fix.
2. **Experiments are comparable.** Both robots train on identical terrain, so performance differences reflect the robot, not the setup.
3. **Adding a third robot** means writing one new params file and one new env config -- not rewriting the whole training pipeline.

---

## 5. The Technology Stack (and Why Each Piece Exists)

### NVIDIA Isaac Lab + PhysX (The Simulator)

**What it does:** Simulates thousands of robots and terrain patches on the GPU simultaneously.

**Why this and not something else:** Isaac Lab runs physics directly on GPU memory (GPU PhysX). This means simulating 20,000 robots costs only ~3x what simulating 100 costs, because the GPU cores work in parallel. Traditional CPU physics (PyBullet, MuJoCo) would take 200x longer.

**The catch:** Everything must stay on the GPU. The moment you move data to CPU (`tensor.cpu()`), you create a synchronization bottleneck. Isaac Lab handles this by keeping observations, rewards, and actions as CUDA tensors throughout.

**Key config:** `@configclass` decorator from `isaaclab.utils`. These are dataclasses-on-steroids that Isaac Lab uses for type-safe, serializable configuration. Every environment config (`SpotPPOEnvCfg`, etc.) inherits from `ManagerBasedRLEnvCfg` and defines scenes, observations, actions, rewards, termination conditions, and domain randomization events.

### RSL-RL (The Training Algorithm)

**What it does:** Implements Proximal Policy Optimization (PPO) -- the RL algorithm that actually trains the neural network.

**Why PPO:** PPO is the workhorse of robot RL. It's stable (won't diverge catastrophically like vanilla policy gradient), sample-efficient enough for GPU-parallel simulation, and well-tested on locomotion tasks. It's not cutting-edge, but it works. In engineering, "boring and reliable" beats "novel and fragile."

**How it works in 60 seconds:**
1. Run 10,000 robots for 32 steps each. Collect all the observations, actions, rewards.
2. Compute "advantages" -- how much better was each action than average?
3. Update the neural network to make high-advantage actions more likely.
4. The "proximal" part: don't update too much at once (clip ratio to 0.2). Big updates destabilize training.
5. Repeat 30,000 times.

**Our additions (monkey-patching):** RSL-RL's `OnPolicyRunner` doesn't natively support cosine learning rate annealing or progressive domain randomization. Instead of forking the library, we monkey-patch the `update()` method:

```python
original_update = runner.alg.update
def update_with_schedule(*args, **kwargs):
    # Set learning rate (cosine annealing)
    lr = cosine_annealing_lr(iteration, max_iters, lr_max, lr_min, warmup)
    set_learning_rate(runner, lr)
    # Expand DR ranges (Vision60 only)
    if robot == "vision60":
        update_dr_params(env, iteration, expansion_iters)
    # Call original PPO update
    result = original_update(*args, **kwargs)
    # Clamp noise (safety)
    clamp_noise_std(runner.alg.policy, min_std=0.3, max_std=2.0)
    return result
runner.alg.update = update_with_schedule
```

This is a pragmatic pattern: inject your logic around the library's core loop without modifying the library itself. It means you can update RSL-RL without merge conflicts, and your changes are clearly isolated in one place.

### PyTorch (The Neural Network)

**The policy network:**
```
235 inputs --> [1024 neurons, ELU] --> [512 neurons, ELU] --> [256 neurons, ELU] --> 12 outputs
```

**What are the 235 inputs?**
- `[0:3]` Base linear velocity (how fast am I moving, in body frame?)
- `[3:6]` Base angular velocity (how fast am I rotating?)
- `[6:9]` Projected gravity vector (which way is "down" relative to my body?)
- `[9:12]` Velocity commands (what speed am I being told to go?)
- `[12:24]` Joint positions relative to default stance (where are my legs?)
- `[24:36]` Joint velocities (how fast are my legs moving?)
- `[36:48]` Previous actions (what did I do last timestep?)
- `[48:235]` Height scan -- 187 ray-cast measurements (what does the ground look like around me?)

**What are the 12 outputs?** Joint position offsets. Each output says "move this joint X radians from its default position." Multiply by action_scale (0.25) to get the actual target. The PD controller at each joint then tracks this target.

**TF32:** We enable TF32 (TensorFloat-32) for matrix multiplications on the H100. This uses the H100's tensor cores for ~2-3x faster training with negligible precision loss. Two lines of code, massive speedup:
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### Cosine Learning Rate Annealing

The learning rate controls how big each gradient step is. Too high and training oscillates. Too low and it never converges. The cosine schedule is elegant:

```
Phase 1 (warmup): LR ramps linearly from 1e-5 to 1e-3 over 500 iterations
Phase 2 (cosine): LR decays smoothly from 1e-3 back to 1e-5 following a cosine curve
```

Why cosine? It's aggressive early (large steps to escape bad regions) and gentle late (small steps to fine-tune). Empirically, it outperforms constant LR and linear decay for locomotion tasks. The warmup prevents the optimizer from making huge steps before the value function is calibrated.

### Progressive Domain Randomization (Vision60 Only)

Domain randomization means "randomize the simulation parameters so the policy generalizes to the real world." Friction, mass, external pushes, joint noise -- all randomized.

The problem: if you randomize everything to the max from iteration 0, the robot can't learn anything. Imagine trying to learn to ride a bike while someone randomly changes the tire friction, adds 10 pounds to one side, and pushes you every 6 seconds. You'd never get past standing up.

Progressive DR solves this by starting with mild randomization and linearly expanding to full difficulty over 15,000 iterations:

| Parameter | Start (easy) | End (hard) |
|-----------|-------------|-----------|
| Friction range | [0.3, 1.3] | [0.1, 1.5] |
| Push velocity | +/-0.5 m/s | +/-1.0 m/s |
| External force | +/-3.0 N | +/-6.0 N |
| Mass offset | +/-5.0 kg | +/-7.0 kg |

The robot learns to walk first, then learns to walk while being messed with. Order matters.

**Why only Vision60?** Spot uses fixed aggressive DR from iteration 0. Spot is heavier and has a lower center of gravity, so it can handle the chaos. Vision60 is light and tall -- it tips over from a stiff breeze at iteration 0 if you apply full DR.

### Weights & Biases (Experiment Tracking)

TensorBoard works for one person staring at one run. W&B works for a team comparing runs across robots:

- Side-by-side dashboards: Spot vs Vision60 reward curves on the same plot
- No SSH tunnel: accessible from any browser
- Hyperparameter tracking: which reward weights produced the best gait?
- System metrics: GPU temperature, VRAM usage, training throughput

RSL-RL has native W&B support -- just set `logger = "wandb"` in the PPO config. We add custom metric logging (learning rate, DR progress, per-reward-term breakdown) via `wandb.log()` in our monkey-patched update function.

### Paramiko (Remote Deployment)

The H100 GPU server is at `172.24.254.24`. We deploy code via SFTP using Python's `paramiko` library:

```python
# deploy_multi_robot.py -- uploads the entire directory tree
sftp.put(local_path, remote_path)  # per file
```

We chose Paramiko over `scp` or `rsync` because it's pure Python (no external dependencies), works on Windows without WSL, and gives us programmatic control (skip `__pycache__`, verify uploads, run remote commands).

### Screen (Persistent Sessions)

Training runs take 10-70 hours. SSH connections don't last that long. `screen` creates a persistent terminal session on the server that survives disconnection:

```bash
screen -dmS spot_train bash -c 'source conda... && python train_ppo.py ...'
screen -r spot_train      # reattach to see output
Ctrl-A, D                  # detach (training continues)
```

We learned the hard way (Section 9) that you must NEVER kill a screen session while Isaac Sim is running. Always detach cleanly or send Ctrl-C.

---

## 6. How Training Actually Works

### One Training Iteration (The Inner Loop)

Every ~10 seconds on the H100, this happens:

1. **Collect experience**: All 10,000 robots take 32 steps each (320,000 transitions). Each step: observe the world (235 dims) -> neural network picks an action (12 dims) -> simulation advances 10 physics steps (0.02s) -> get a reward (sum of 19 terms) -> record everything.

2. **Compute advantages**: For each transition, ask "how much better was this outcome than expected?" This uses Generalized Advantage Estimation (GAE), which balances bias and variance by looking ahead with a discount factor (gamma=0.99) and a smoothing parameter (lambda=0.95).

3. **PPO update**: Divide the 320K transitions into 64 mini-batches. For each mini-batch, compute the policy gradient and update the neural network. Do this 8 times (learning epochs). The clipping mechanism (epsilon=0.2) prevents any single update from changing the policy too much.

4. **Our additions**: Adjust the learning rate (cosine schedule), expand DR ranges (Vision60), clamp noise std (safety net), log metrics.

5. **Checkpoint**: Every 500 iterations, save `model_XXXXX.pt`.

### Episode Lifecycle

Each robot runs independently in its own terrain patch:

```
[Spawn]
  Robot materializes 0.5m (Spot) or 0.6m (Vision60) above ground
  Random initial joint velocities, random terrain patch
  Velocity command sampled: vx in [0.5, 1.5], vy in [-0.3, 0.3], omega in [-0.5, 0.5]

[Live]
  Up to 30 seconds of walking (1500 control steps at 50 Hz)
  Every step: observe -> act -> reward
  Terrain curriculum may promote/demote to harder/easier row

[Death]
  Body touches ground? -> Terminated (body_contact penalty kicks in)
  30 seconds elapsed? -> Timeout (survived!)

[Reset]
  New terrain patch, new velocity command, new DR parameters
  Life begins again
```

### What the Robot "Sees" (Observation Vector)

The 235-dimensional observation vector is the robot's entire perception of reality. It has no cameras, no LIDAR, no GPS. It has:

**Proprioception (48 dims):** "How does my body feel?"
- Am I moving forward/sideways/up? (3 dims)
- Am I rotating? (3 dims)
- Which way is gravity pulling me? (3 dims -- this tells it if it's on a slope)
- What speed am I supposed to go? (3 dims)
- Where are my joints? (12 dims)
- How fast are my joints moving? (12 dims)
- What did I do last time? (12 dims -- action history for smoothness)

**Height Scan (187 dims):** "What does the ground look like?"
A 17x11 grid of ray-casts centered on the robot, each 0.1m apart, covering a 1.6m x 1.0m window. Each value is "how far below my body is the ground at this point?" This is how the robot "sees" stairs, boulders, and gaps -- not with eyes, but with a grid of virtual measuring sticks pointing down.

The height scan is the most fragile part of the observation. Get the fill value wrong (1.0 instead of 0.0 for flat ground) and the policy collapses instantly. More on that in Section 9.

---

## 7. The Three-Phase Training Pipeline

### Phase 1: Direct PPO (This Is What We're Running)

The bread and butter. Train a policy from scratch using PPO on 12 terrain types with 19 reward terms.

```bash
# On H100
cd ~/IsaacLab
./isaaclab.sh -p ~/multi_robot_training/train_ppo.py --headless \
    --robot spot --num_envs 10000 --max_iterations 30000
```

This produces a policy that can walk on diverse terrain. It takes 10-70 hours depending on environment count and iteration count.

### Phase 2a: Teacher Training (Optional Upgrade)

The Phase 1 policy only gets height scans. What if we gave it *cheats* -- exact friction coefficients, terrain type labels, clean contact forces? These are "privileged observations" that exist in simulation but not on a real robot.

The teacher network gets 235 standard dims + 19 privileged dims = 254 total inputs. It uses the extra info to learn better terrain adaptation.

**Weight surgery:** The teacher starts from the Phase 1 checkpoint, but its input layer is wider (254 vs 235). We surgically extend the first layer's weight matrix:

```
Original: [1024, 235]  (Phase 1 checkpoint)
Extended: [1024, 254]  (zero-padded columns for privileged dims)
```

The zero-initialization means the teacher initially ignores the privileged inputs (they contribute nothing to the first hidden layer), then gradually learns to use them as training progresses.

### Phase 2b: Student Distillation (Deployment-Ready)

The teacher can't run on a real robot -- it needs privileged info that doesn't exist outside simulation. So we train a *student* that matches the teacher's behavior using only standard observations.

Combined loss:
```
loss = (1 - bc_coef) * PPO_loss + bc_coef * BC_loss
where BC_loss = MSE(student_action, teacher_action)
bc_coef anneals: 0.8 -> 0.2 over 10K iterations
```

Start by mostly imitating the teacher (bc_coef = 0.8), then gradually shift to RL self-improvement (bc_coef = 0.2). The student learns the teacher's terrain-adaptive tricks without needing privileged observations.

---

## 8. The Evaluation Gauntlet

### The Four Arenas

After training, every policy gets tested in four custom arena environments, each 50 meters long with 5 zones of increasing difficulty:

**Friction Arena:** Sandpaper -> Dry rubber -> Wet concrete -> Wet ice -> Oiled steel
- Tests: Can you walk when the ground stops cooperating?

**Grass Arena:** Empty -> Thin grass -> Medium lawn -> Thick grass -> Dense brush
- Tests: Can you push through vegetation that grabs your feet?

**Boulder Arena:** Gravel -> River rocks -> Large rocks -> Small boulders -> Large boulders
- Tests: Can you pick your way through an unstructured rock field?

**Stairs Arena:** 3cm risers -> 8cm -> 13cm -> 18cm -> 23cm
- Tests: Can you climb increasingly aggressive staircases?

Each robot runs 1000 episodes per arena. We measure completion rate, distance traveled, fall rate, stability score, and energy consumption. Results are saved as JSONL (one JSON object per episode) for statistical analysis.

### The Waypoint Follower

The robot doesn't navigate autonomously. A simple waypoint follower gives it commands:

```
"Walk forward at 1.0 m/s, correct heading toward the next waypoint"
```

Six waypoints along the arena centerline, 10m apart. The heading controller is proportional-only (Kp = 2.0). This is intentionally simple -- we're testing the locomotion policy, not navigation intelligence.

---

## 9. The Bug Museum: Every Disaster and How We Survived

This section is the most valuable part of this document. Every bug here cost us hours or days. Learn from our pain.

### Bug #1: "The 100-Hour Run That Learned Nothing"

**What happened:** Our first serious training attempt ran for 100 hours on the H100 (10,000 iterations, 17.6 billion timesteps). The robot never learned to walk. Not even stand. It just flopped around and died in 7 seconds, every episode, for four straight days.

**Root cause (five problems stacked):**

1. **Architecture mismatch:** We used a bigger network `[1024, 512, 256]` instead of the working `[512, 256, 128]`. This meant we couldn't load the working checkpoint (weight shapes didn't match), so we trained from scratch.

2. **From scratch on max difficulty:** The randomly initialized network had to simultaneously learn to stand, balance, walk, and navigate 12 terrain types with extreme domain randomization (friction as low as 0.05). That's like teaching a newborn to run a military obstacle course.

3. **Maximum DR from iteration 0:** Friction down to 0.05 (oil on polished steel), push velocities of 1.5 m/s, random forces of 8 N. Even a *working* policy would struggle. A random one never stood a chance.

4. **Contradictory gradients:** On ice (friction 0.05), the robot needs slow, careful movements. On sandpaper (friction 1.5), aggressive gaits work best. The gradient update tried to optimize for both simultaneously, resulting in incoherent learning.

5. **No progressive DR:** The terrain had curriculum (auto-difficulty), but DR was at maximum from the start. The robot kept falling due to extreme randomization, not terrain difficulty, so the curriculum never advanced.

**The fix:** Everything in this project's architecture was designed to prevent this failure:
- Use the same network architecture as the checkpoint you're loading from
- Progressive DR (start mild, expand over 15K iterations)
- Warm start from a working policy instead of random initialization
- Cosine LR annealing (gentle start, aggressive middle, gentle end)

**The lesson:** Difficulty must be *earned*, not imposed. If you wouldn't throw a first-day ski student down a black diamond, don't throw a random neural network into oil-on-steel terrain with 8N random pushes.

---

### Bug #2: "The Critic That Poisoned Everything" (Value Function Mismatch)

**What happened:** We loaded a working 48-hour checkpoint and started fine-tuning with 19 reward terms (the checkpoint was trained with 14). Within 2,900 iterations, the policy went from competent walking to 100% fall rate on flat ground.

**The TensorBoard death spiral:**
```
Iteration 0:     terrain_level=3.44  fall_rate=7.5%   entropy=11.89
Iteration 2,900: terrain_level=0.00  fall_rate=100%   entropy=-5.44
                  Value function loss at iteration 0: 5,298 (!!!)
```

**Root cause:** PPO has two networks: the *actor* (picks actions) and the *critic* (estimates how good a state is). We loaded both from the checkpoint. The actor was fine -- 235 inputs, 12 outputs, same as before. But the critic was catastrophically wrong.

The critic was trained to predict rewards from 14 terms. We changed to 19 terms with different magnitudes. Its predictions were off by orders of magnitude, which corrupted every advantage estimate, which sent garbage gradients to the actor, which triggered the adaptive KL schedule to kill the learning rate and collapse exploration noise.

**Analogy:** Imagine you're a financial advisor (the critic) trained to value tech stocks. Your client (the actor) asks you to evaluate a portfolio that now includes crypto, commodities, and real estate. Your valuations are wildly off, so your client makes terrible trades based on your advice, loses confidence, and stops trading entirely.

**The fix (three changes, applied together):**
1. **Actor-only loading:** Load only the actor weights. Initialize the critic from scratch. Fresh critic, no stale predictions.
2. **Critic warmup:** Freeze the actor for 1,000 iterations while the new critic calibrates to the new reward landscape.
3. **Noise floor:** Clamp exploration noise >= 0.4 so the adaptive KL can't collapse it to zero.

**The lesson:** When you change the reward function, the critic becomes a liar. Either reset it or let it recalibrate before trusting its advice.

---

### Bug #3: "The Do-Nothing Policy" (Reward Imbalance)

**What happened:** Both Spot and Vision60 converged to a policy that simply stood still and waited for the episode to end. body_contact = 1.0 (100% fall rate from standing still too long), but the robot never even *tried* to walk.

**Root cause:** The penalty terms were overwhelmingly stronger than the rewards. Check the math:

```
stumble penalty:    -2.0 weight * ~25 magnitude = -50 per step
gait reward:        +10.0 weight * ~0.1 magnitude = +1 per step
```

The stumble penalty alone was 50x larger than the gait reward. The robot quickly learned that moving produces enormous penalties (stumbling, slipping, force spikes), while standing still produces small penalties (just the velocity tracking error). Standing still was the *rational* strategy.

**Analogy:** Imagine a job where you get paid $1 for every widget you build, but fined $50 every time you accidentally drop a tool. The optimal strategy is to not touch any tools.

**The fix:**
```
stumble:                  -2.0  -->  -0.3
contact_force_smoothness: -0.5  -->  -0.02
action_smoothness:        -2.0  -->  -0.5
base_linear_velocity:     +7.0  -->  +12.0
gait:                     +10.0 -->  +15.0
warmup_iters:             3000  -->  500
```

**The lesson:** Always check the *actual magnitudes* of your reward terms, not just the weights. A term with weight 1.0 that outputs values of 100.0 dominates a term with weight 10.0 that outputs values of 0.01. Monitor per-term reward breakdown in TensorBoard from iteration 1.

---

### Bug #4: "Spot's Instant Death" (Overly Aggressive Termination)

**What happened:** After fixing the reward imbalance, Vision60 started learning beautifully (31% episodes surviving, curriculum advancing). Spot was still at 100% fall rate with average episode length of 4.12 steps (0.08 seconds).

**Root cause:** Spot's termination condition included leg segments:
```python
body_names=["body", ".*leg"]  # Spot
body_names=["body"]            # Vision60
```

For Spot, *any* leg touching the ground was flagged as a fall. Since quadrupeds frequently make brief leg-ground contact during normal locomotion (especially on rough terrain), every episode was terminated almost immediately.

**The fix:** Changed Spot's termination to body-only, matching Vision60:
```python
body_names=["body"]  # Both robots now
```

**The lesson:** Compare your robots' configs side by side when one is learning and the other isn't. The answer is often hiding in the difference. This is literally `diff spot_cfg.py vision60_cfg.py`.

---

### Bug #5: "The Folding Legs" (Missing Joint Limits)

**What happened:** Vision60's legs would fold inward during training -- joints hyperextending past their mechanical limits, causing the robot to collapse into a pretzel shape.

**Root cause:** No penalty for exceeding joint position limits. The policy found that extreme joint angles sometimes reduced other penalties (less motion = less slip = less force spikes), so it contorted the legs into physically impossible configurations.

**The fix:** Added the `dof_pos_limits` penalty from Isaac Lab's built-in `mdp` module:
```python
dof_pos_limits = RewardTermCfg(
    func=mdp.joint_pos_limits,
    weight=-10.0,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
)
```

Also increased the general `joint_pos` penalty from -1.0 to -2.0 as a softer version that activates before the hard limits.

**The lesson:** Neural networks will exploit any gap in your reward function. If you don't penalize physically impossible states, the optimizer *will* find them. Think adversarially about what a degenerate policy could do.

---

### Bug #6: "The Zombie Apocalypse" (GPU Process Leaks)

**What happened:** After killing training runs (via `screen -X quit` or `Ctrl-C` at the wrong time), Isaac Sim left zombie processes in "D-state" (uninterruptible kernel sleep). These zombies held 4-80 GB of GPU memory, were immune to `kill -9`, blocked `nvidia-smi --gpu-reset`, and even caused `sudo reboot` to hang indefinitely.

**Root cause:** NVIDIA PhysX/CUDA driver teardown deadlocks in the Linux kernel. When you call `simulation_app.close()` (or the process is killed while the GPU driver is active), the cleanup sequence sometimes enters a kernel-level deadlock that no userspace signal can break.

**The fix (3-layer defense):**

1. **Never call `simulation_app.close()`.** Use `os._exit(0)` instead. This skips Python cleanup and driver teardown entirely. All training data is already saved to disk, so there's nothing to lose.

2. **Signal handler for Ctrl-C:**
```python
def _graceful_shutdown(signum, frame):
    metrics_collector.save()  # Save any pending data
    os._exit(0)               # Skip driver cleanup
signal.signal(signal.SIGINT, _graceful_shutdown)
```

3. **Shell script timeouts:** Wrap every training command in `timeout` and follow with `pkill -f` cleanup.

**Recovery:** When zombies are already holding the GPU, the only fix is a **physical power cycle** (IPMI/BMC or power button). Software reboot hangs because the kernel can't release the GPU. We bricked the server twice before learning this.

**The lesson:** GPU drivers are kernel code. Kernel code can deadlock. Always assume your cleanup code might not run, and design for it: save data eagerly, never rely on graceful shutdown, and use `os._exit(0)` when you're done.

---

### Bug #7: "The Height Scan That Broke Everything"

**What happened:** The rough terrain policy worked perfectly in training but collapsed immediately during standalone evaluation. The robot would take one step and fall over, action norms at 7.42 (should be ~3.0).

**Root cause:** During evaluation, we filled the 187 height scan dimensions with `1.0` for flat ground (thinking "1.0 = ground is 1m below, seems reasonable"). But during training, the height scan values ranged from -0.000002 to 0.148. A fill value of 1.0 was 7x larger than anything the policy ever saw during training.

The neural network saw 187 inputs screaming "THERE IS A 1-METER CLIFF BELOW YOU IN EVERY DIRECTION" and responded with maximum panic actions.

**The fix:** Fill with `0.0` for flat ground:
```python
HEIGHT_SCAN_FILL = 0.0  # NOT 1.0
```

**The lesson:** Neural networks are not robust to out-of-distribution inputs. A value that seems "reasonable" to a human can be catastrophically wrong for a network trained on a specific distribution. Always check what the training data actually looks like (`print(obs.min(), obs.max(), obs.mean())`) and match it exactly.

---

### Bug #8: "Import Order: The Silent Killer"

**What happened:** Script crashes with opaque errors about missing modules or CUDA not being initialized.

**Root cause:** Isaac Lab requires a very specific import order:

```python
# CORRECT (and mandatory):
from isaaclab.app import AppLauncher     # 1. Import launcher
app_launcher = AppLauncher(args)         # 2. Create launcher (starts SimulationApp)
simulation_app = app_launcher.app        # 3. Get app handle

# NOW you can import everything else:
import torch
from isaaclab.envs import ...
from rsl_rl.runners import ...
```

Importing `omni.isaac.core`, `isaaclab.envs`, or even `torch.cuda` before `AppLauncher` causes silent failures, wrong device selection, or immediate crashes.

**The lesson:** Some frameworks have strict initialization order requirements. When the docs say "import X before Y," they mean it. Put a big comment block in your code:
```python
# ── 0. Parse args BEFORE any Isaac imports ──────────
# ── 1. Imports (AFTER SimulationApp) ─────────────────
```

---

### Bug #9: "Two Isaac Sims Walk Into a GPU..."

**What happened:** We tried running Spot and Vision60 training simultaneously on the same H100 to save time. The second process crashed with `CUDA illegal memory access`.

**Root cause:** Two Isaac Sim instances running GPU PhysX on the same GPU conflict over the PhysX backend. The GPU's physics engine is essentially a singleton -- two processes writing to the same PhysX buffers causes memory corruption.

**The fix:** Run sequentially (one after the other) or use `screen` sessions where only one is actively using the GPU at a time. On a multi-GPU machine, you could use `CUDA_VISIBLE_DEVICES` to isolate them.

**The irony:** This bug was non-deterministic. Sometimes both processes ran fine for hours. Sometimes it crashed in 30 seconds. The worst kind of bug -- intermittent, hardware-dependent, and non-reproducible.

**The lesson:** GPU physics engines often have singleton assumptions. Test parallel execution explicitly before assuming it works. "It worked once" is not validation.

---

### Bug #10: "CRLF Will Ruin Your Day"

**What happened:** Shell scripts uploaded from Windows to the Linux H100 server fail with `bash: '\r': command not found`.

**Root cause:** Windows uses `\r\n` (CRLF) line endings. Linux expects `\n` (LF). The carriage return (`\r`) is interpreted as part of the command.

**The fix:** `sed -i "s/\r$//" scripts/*.sh` on the server, or set `git config core.autocrlf input`.

**The lesson:** This bug is older than most of us reading this. It will never stop happening. Add `dos2unix` to your deployment script or use `.gitattributes` to force LF for shell scripts.

---

### Bug #11: "The Hand-Tuned Weights That Were 2-6x Too Harsh"

**What happened:** After fixing Bugs #3 (do-nothing policy) and #4 (Spot's instant death), we pushed reward weights in the *opposite* direction -- boosting positive rewards and adjusting penalties based on intuition. This seemed to fix the do-nothing problem, but Spot still couldn't learn. By iteration 1,750 the training metrics told a familiar story:

```
body_contact termination:  12.6% (iter 0) --> 99.9% (iter 1750)
mean episode length:       27.0 steps     --> 3.9 steps
terrain_levels:            3.39           --> 0.009
```

The robot was falling *faster* as training progressed. Not just failing to learn -- actively getting worse.

**Root cause:** Our "intuitive" reward weights were significantly harsher than the paper's empirically validated coefficients. Here's the side-by-side comparison:

| Term | Our Weight | Paper Weight | How Wrong |
|------|-----------|-------------|-----------|
| `base_linear_velocity` | +12.0 | +5.0 | 2.4x too high |
| `gait` | +15.0 | +5.0 | 3x too high |
| `air_time` | +3.0 | +5.0 | 0.6x (too low) |
| `foot_clearance` | +3.5 | +0.75 | 4.7x too high |
| `base_orientation` | -5.0 | -3.0 | 1.7x too harsh |
| `base_motion` | -4.0 | -2.0 | 2x too harsh |
| `foot_slip` | -3.0 | -0.5 | 6x too harsh |
| `joint_pos` | -2.0 | -0.7 | 2.9x too harsh |
| `joint_acc` | -5e-4 | -1e-4 | 5x too harsh |
| `joint_torques` | -2e-3 | -5e-4 | 4x too harsh |
| `joint_vel` | -5e-2 | -1e-2 | 5x too harsh |
| `dof_pos_limits` | -10.0 | -5.0 | 2x too harsh |
| `body_height_tracking` | -2.0 | -1.0 | 2x too harsh |
| `stumble` | -0.3 | -0.1 | 3x too harsh |

**The subtle trap:** After fixing Bug #3, we overcorrected. We cranked up `base_linear_velocity` to +12.0 and `gait` to +15.0, thinking bigger positive rewards would overpower the penalties. But we also left the penalties much higher than the paper's values. The result was a chaotic reward landscape where the robot was being *pulled* toward movement by huge rewards and simultaneously *punished* for every aspect of actually moving. The gradient was incoherent -- not "do nothing" this time, but "do everything at once and crash."

**What made it worse:** We also had parameter mismatches beyond just weights:
- `mode_time`: 0.3 (ours) vs 0.2 (paper) -- longer required air time made gait harder to achieve
- `velocity_threshold`: 0.5 (ours) vs 0.25 (paper) -- required faster speed before rewarding gait
- `target_height` (foot clearance): 0.10 (ours) vs 0.125 (paper) -- different clearance expectation
- `joint_names` for `joint_acc`/`joint_vel`: `".*_h[xy]"` (ours, hip joints only) vs `".*"` (paper, all joints)

These parameter differences meant the reward functions weren't just scaled wrong -- they were *shaped* differently.

**The fix:** We went back to the paper's source code (`Robust_RL/quadruped_locomotion/.../spot_reward_env_cfg.py`) and copied every single coefficient and parameter verbatim. No interpretation, no "but I think this should be higher." Just copy-paste the numbers that took the paper's authors months to validate.

**The lesson:** When you have a published paper with working coefficients, **use them exactly.** Human intuition about reward magnitudes is terrible. A penalty of -0.5 "feels" too small for foot slip, so you bump it to -3.0 -- but the paper's authors ran dozens of experiments to find that -0.5 is exactly right for the reward *landscape* they designed. Each term interacts with every other term. Changing one coefficient ripples through the entire optimization. Unless you have time to run your own hyperparameter sweep, trust the people who already did.

---

### Bug #12: "GPU Contention -- Two Isaac Sims Walk, Both Crawl"

**What happened:** We launched Spot and Vision60 training simultaneously on the same H100 (96GB). Both processes ran -- no crashes this time (unlike Bug #9 which was a total crash). But each iteration took ~27 seconds instead of the expected ~15 seconds. Both robots were training at half speed.

**Root cause:** Two Isaac Sim instances running GPU PhysX on the same device don't crash (if you're lucky), but they time-slice the GPU. With both active:
- GPU utilization: 83% at only 139W/400W (low for H100)
- Each process got roughly half the compute bandwidth
- Collection phase (PhysX stepping) doubled from ~10s to ~21s per iteration

**The math that killed our timeline:**
```
Parallel (both at 27s/iter):  V60 is bottleneck at 44s/iter = ~15 days total
Sequential (one at ~16s/iter): Spot 4.4d + V60 7.4d = ~12 days total
```

Sequential training is 3 days *faster* than parallel on a single GPU. The parallelism overhead cost more than it saved.

**The fix:** Kill Vision60, let Spot have the full GPU. Spot's iteration time immediately dropped from 27s to 17.2s (collection: 20.8s to 10.3s, fps: 10,921 to 18,588). Run them sequentially.

**Bonus horror:** When we killed Vision60, it became a D-state zombie holding 20.7 GB of VRAM (Bug #6 strikes again). The zombie wasn't computing anymore, so Spot got the compute bandwidth back, but the VRAM was still locked. Eventually required a GPU reset.

**The lesson:** On a single GPU, sequential beats parallel for heavy simulation workloads. The GPU can't meaningfully parallelize two independent PhysX simulations -- it just context-switches between them. Save the "run both at once" strategy for multi-GPU machines with `CUDA_VISIBLE_DEVICES` isolation.

---

## 10. Lessons That Will Save Your Future Self

### On Reward Engineering

**1. Penalties are more powerful than rewards.** A penalty of -2.0 on a term that fires every step creates a massive negative gradient. A reward of +10.0 on a term that only fires when the robot does something right creates a sparse positive gradient. The penalty always wins. Start penalties small and increase them only if the unwanted behavior persists.

**2. Monitor per-term reward magnitudes, not just weights.** A weight of 1.0 on a term that outputs 100.0 is ten times stronger than a weight of 10.0 on a term that outputs 1.0. Log every reward term separately in TensorBoard/W&B from the first iteration.

**3. The "do nothing" policy is always a local minimum.** If standing still scores better than walking-but-sometimes-stumbling, the robot will learn to stand still. Make sure your velocity reward dominates your stumble/slip/force penalties by at least 2-3x in the first 1000 iterations.

**4. Neural networks will exploit your reward function like a tax loophole.** If folding legs into a pretzel reduces foot slip penalty, the robot will fold its legs. If hovering at zero velocity avoids all contact penalties, it'll hover. Think like an adversary: what's the stupidest, most degenerate strategy that maximizes this reward?

### On Training

**5. Warm starts beat cold starts.** Loading a working checkpoint and fine-tuning beats training from scratch almost every time. Even if the fine-tuning task is very different, the actor network retains useful features (balance, gait timing) that transfer.

**6. But the critic doesn't transfer across reward changes.** If you change your reward function, reset the critic. Load actor-only. The critic's value predictions are specific to the exact reward landscape it was trained on.

**7. Difficulty must be progressive.** Curriculum for terrain difficulty. Progressive schedule for domain randomization. Linear warmup for learning rate. In every case, the pattern is the same: start easy, get harder as the agent gets stronger.

**8. Noise clamping is a safety net, not a feature.** Clamping the policy's noise standard deviation (floor 0.3, ceiling 2.0) prevents two catastrophic failure modes: exploration collapse (noise -> 0, policy locks into one behavior) and noise explosion (noise -> infinity, random actions). These take hours to recover from without the clamp.

### On Engineering

**9. Deploy scripts are code, not afterthoughts.** `deploy_multi_robot.py` is 63 lines that recursively SFTP an entire directory to the H100. It skips `__pycache__`, creates remote directories, and verifies the upload. We ran this script 30+ times during development. The 30 minutes spent writing it saved days of manual `scp` commands.

**10. Always save eagerly, exit rudely.** Save your metrics/checkpoints to disk *before* you need them. Exit with `os._exit(0)` instead of trusting cleanup code to run. GPU drivers can deadlock. SSH can drop. Power can fail. If your data is on disk, none of these matter.

**11. Separate what changes from what doesn't.** Terrain configs don't change between robots -- put them in `shared/`. Body names change -- put them in robot-specific configs. Training loop structure doesn't change -- put it in one script with a `--robot` flag. This is the Single Responsibility Principle applied to ML infrastructure.

**12. Compare configs when one thing works and another doesn't.** When Vision60 was learning and Spot wasn't, the answer was literally in the diff between their configs. `termination_body_names: ["body", ".*leg"]` vs `["body"]`. The most powerful debugging tool is often `diff`.

### On GPU Computing

**13. GPU processes are not just processes.** They hold kernel-level resources that survive `kill -9`. A zombie GPU process can brick your server until you physically power cycle it. Design your shutdown path to avoid kernel driver cleanup entirely.

**14. TF32 is free performance.** Two lines of code. 2-3x speedup on H100. Zero accuracy loss for RL. There is no reason not to enable it.

**15. More parallel environments != more time per iteration.** Going from 8K to 16K envs costs ~60% more time per iteration but gives 2x the data. The gradient quality improves because you sample more terrain types per batch. The sweet spot is where your GPU is well-utilized but not overheating (~70-80% utilization, <70C).

### On Working with Isaac Lab

**16. The import order is non-negotiable.** `AppLauncher` before everything. Put a big visible comment. Future you will thank present you.

**17. `@configclass` is your friend.** Isaac Lab's config system is verbose, but it catches type errors at initialization, serializes to YAML for reproducibility, and makes experiment configs diffable. Embrace the verbosity.

**18. Monkey-patching is pragmatic.** Forking RSL-RL to add cosine LR would create merge conflicts on every upstream update. Wrapping `runner.alg.update()` achieves the same thing with zero library modifications and clear isolation of your changes.

### On Trusting Published Work

**19. Use the paper's exact coefficients. Not "inspired by." Exact.** We spent days hand-tuning reward weights based on intuition. Every attempt failed -- too harsh, too lenient, reward conflicts, degenerate policies. The paper's authors spent *months* finding the right balance across all 19 terms. Their coefficients encode a working equilibrium that isn't obvious from reading the equations. Copy the numbers exactly. Tune later if needed, but start from a known-working baseline.

**20. Reward weights are not independent knobs.** Changing `foot_slip` from -0.5 to -3.0 doesn't just make slip 6x more penalized. It changes the relative gradient contribution of every other term. The policy will now over-optimize for slip avoidance at the expense of velocity tracking, gait timing, and everything else. Reward engineering is system design, not parameter tuning.

**21. Check parameters, not just weights.** Two reward functions with the same weight can behave completely differently if their internal parameters differ. Our `mode_time=0.3` vs the paper's `mode_time=0.2` changed when the gait reward activated. Our `velocity_threshold=0.5` vs the paper's `0.25` changed what counted as "moving." These aren't cosmetic differences -- they reshape the entire reward surface.

### On GPU Resource Management

**22. Sequential beats parallel on a single GPU for heavy simulations.** Two Isaac Sim instances on one H100 each ran at half speed. Sequential was 3 days faster over the full training horizon. GPU PhysX can't meaningfully parallelize independent simulations -- it just context-switches.

**23. Zombie processes hold VRAM but not compute.** A D-state zombie from a killed Isaac Sim holds GPU memory (preventing new allocations) but doesn't consume compute cycles. If you have headroom, the remaining process will speed up. But eventually you need to reset the GPU to reclaim the memory.

---

## 11. How Good Engineers Think

This project taught us as much about engineering process as it did about RL. Here are the meta-lessons:

### "Make It Work, Make It Right, Make It Fast"

Our codebase evolved through three distinct phases:

1. **Make it work:** `100hr_env_run/` and `vision60_training/` were standalone scripts -- one per robot, duplicated code, hardcoded parameters. They *worked* (eventually), but maintaining two separate training pipelines was painful.

2. **Make it right:** `multi_robot_training/` unified everything. Shared terrain configs, parameterized reward functions, one training script for both robots. The architecture is clean, extensible, and maintainable.

3. **Make it fast:** TF32 matmul, progressive DR (less wasted computation on early iterations), cosine LR (converge faster by being aggressive in the middle).

Each phase builds on the previous one. We didn't try to build the perfect architecture on day one -- we built the simplest thing that could work, identified the pain points, and refactored.

### Debug by Bisection

When Spot wasn't learning and Vision60 was, we didn't stare at the code hoping for insight. We systematically compared every difference:

- Same terrain? Yes.
- Same rewards? Yes (after our fixes).
- Same network? Yes.
- Same DR? Yes.
- Same termination? **No.** Spot had `["body", ".*leg"]`, Vision60 had `["body"]`.

One difference. That was the bug. Bisection (systematically eliminating possibilities) beats intuition for debugging complex systems.

### Log Everything, Trust Nothing

Every training run logs: learning rate, per-reward-term breakdown, noise std, terrain levels, episode lengths, DR progress. We save configs to YAML alongside checkpoints so we can reproduce any run.

When the "do nothing" convergence happened, we diagnosed it in minutes because we could see that `stumble_penalty = -51` vs `gait_reward = +1.05`. Without per-term logging, we would have been guessing for days.

### Invest in Deployment Tooling Early

We wrote `deploy_multi_robot.py` before our first training run. It took 30 minutes. We used it 30+ times over two weeks. That's 30 minutes of investment saving hours of manual file copying, typos in `scp` commands, and forgotten files.

Similarly, the deploy-verify-check pattern (upload files -> verify directory structure on server -> run smoke test) caught problems before they wasted GPU hours. A 2-minute smoke test (64 envs, 10 iterations) catches 90% of bugs that would otherwise only surface after 10 hours of training.

### Know Your Failure Modes

Before every training run, we now ask:
- What's the degenerate policy? (Stand still? Fold legs? Vibrate in place?)
- What will the reward landscape look like at iteration 0?
- What's the worst thing that can happen to the GPU process?
- If this crashes at hour 10, what data have we saved?

Thinking about failure modes before they happen is the difference between a 2-minute fix and a 2-day investigation.

---

## 12. Training Run Log

This is the chronological record of every significant training attempt, what happened, and what we changed. This is the raw empirical history of the project.

### Trial 1: First Multi-Robot Launch (Feb 27, 2026)

**Config:** Custom-tuned weights (Bug #3 fix: boosted positive rewards, reduced some penalties)
**Envs:** 10,000 per robot | **Hardware:** H100 96GB (shared between both robots)
**Setup:** Spot + Vision60 launched simultaneously in separate screen sessions

**What happened:**
- Both processes launched successfully (no PhysX crash this time -- Bug #9 was intermittent)
- Both running at half speed due to GPU contention (~27s/iter instead of ~15s)
- Vision60 was progressing faster than Spot in early iterations
- Decision: Kill Vision60 to give Spot full GPU (Spot is priority)
- V60 became D-state zombie holding 20.7 GB VRAM (Bug #6)
- Spot sped up to 17.2s/iter with solo GPU access

**Spot results after ~1,750 iterations:**
```
body_contact termination:  12.6% --> 99.9%
mean episode length:       27.0  --> 3.9 steps
terrain_levels:            3.39  --> 0.009
```

**Diagnosis:** Reward weights 2-6x harsher than paper (Bug #11). Robot getting worse, not better.

**Outcome:** FAILED -- killed training, deployed paper-matched coefficients.

---

### Trial 2: Paper-Matched Coefficients (Feb 28, 2026) -- IN PROGRESS

**Config:** Exact paper coefficients from `Robust_RL/quadruped_locomotion/.../spot_reward_env_cfg.py`
**Envs:** 10,000 Spot only | **Hardware:** H100 96GB (solo, clean GPU after reset)
**Log dir:** `spot_robust_ppo/2026-02-28_08-52-14/`

**Key coefficient changes from Trial 1:**
```
base_linear_velocity:  +12.0 --> +5.0    (paper)
gait:                  +15.0 --> +5.0    (paper)
air_time:              +3.0  --> +5.0    (paper)
foot_clearance:        +3.5  --> +0.75   (paper)
foot_slip:             -3.0  --> -0.5    (paper)
base_orientation:      -5.0  --> -3.0    (paper)
base_motion:           -4.0  --> -2.0    (paper)
joint_pos:             -2.0  --> -0.7    (paper)
joint_acc:             -5e-4 --> -1e-4   (paper)
joint_torques:         -2e-3 --> -5e-4   (paper)
joint_vel:             -5e-2 --> -1e-2   (paper)
dof_pos_limits:        -10.0 --> -5.0    (paper)
body_height:           -2.0  --> -1.0    (paper)
stumble:               -0.3  --> -0.1    (paper)
```

Also fixed parameter mismatches:
- `mode_time`: 0.3 --> 0.2 (paper)
- `velocity_threshold`: 0.5 --> 0.25 (paper)
- `target_height`: 0.10 --> 0.125 (paper)
- `joint_names` for acc/vel: `".*_h[xy]"` --> `".*"` (paper, all joints not just hips)

**Performance:** ~16s/iter, ~18,500 fps, 49C GPU temp

**Early metrics (first iterations):**
- body_contact: 99.98% (expected -- fresh random weights, robot hasn't learned yet)
- Penalty magnitudes noticeably smaller than Trial 1
- ETA: ~16.5 hours for 30,000 iterations

**Status:** RUNNING. Check TensorBoard at `http://172.24.254.24:6006`

**What to watch for:**
- body_contact should start dropping below 90% by iter ~500 (sign of learning)
- episode_length should climb above 10 by iter ~1000
- terrain_levels should start advancing by iter ~2000
- If none of these happen by iter 3000, the config still needs work

---

### Trial 3: Vision60 (PLANNED)

After Spot's Trial 2 completes (or reaches a good checkpoint), relaunch Vision60 with:
- Paper-matched coefficients adapted for V60 body parameters
- V60-specific adjustments: reduced air_time, foot_slip, base_motion per heavier robot dynamics
- Progressive DR (mild to aggressive over 15K iterations)
- Full solo GPU

---

## 13. Quick Reference: Commands and Paths

### Training Commands (H100)

```bash
# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
export OMNI_KIT_ACCEPT_EULA=YES && export PYTHONUNBUFFERED=1

# Phase 1: Spot (full scale)
cd ~/IsaacLab
screen -dmS spot_train bash -c '
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab &&
    export OMNI_KIT_ACCEPT_EULA=YES && export PYTHONUNBUFFERED=1 &&
    ./isaaclab.sh -p ~/multi_robot_training/train_ppo.py --headless \
        --robot spot --num_envs 10000 --max_iterations 30000 --no_wandb
'

# Phase 1: Vision60 (full scale)
screen -dmS v60_train bash -c '
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab &&
    export OMNI_KIT_ACCEPT_EULA=YES && export PYTHONUNBUFFERED=1 &&
    ./isaaclab.sh -p ~/multi_robot_training/train_ppo.py --headless \
        --robot vision60 --num_envs 10000 --max_iterations 30000 --no_wandb
'

# Local smoke test (Windows)
isaaclab.bat -p path/to/train_ppo.py --headless \
    --robot spot --num_envs 64 --max_iterations 10 --no_wandb
```

### TensorBoard

```bash
# Spot only (port 6006)
screen -dmS tb_spot bash -c '
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab &&
    tensorboard --logdir ~/IsaacLab/logs/rsl_rl/spot_robust_ppo/LATEST_RUN/ \
        --port 6006 --bind_all
'

# Vision60 only (port 6007)
screen -dmS tb_v60 bash -c '
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab &&
    tensorboard --logdir ~/IsaacLab/logs/rsl_rl/vision60_robust_ppo/LATEST_RUN/ \
        --port 6007 --bind_all
'
```

### Key File Paths

| What | Local Path |
|------|-----------|
| Training script | `multi_robot_training/train_ppo.py` |
| Spot env config | `multi_robot_training/configs/spot_ppo_env_cfg.py` |
| Vision60 env config | `multi_robot_training/configs/vision60_ppo_env_cfg.py` |
| 12-terrain definition | `multi_robot_training/shared/terrain_cfg.py` |
| Custom rewards | `multi_robot_training/shared/reward_terms.py` |
| LR schedule | `multi_robot_training/shared/lr_schedule.py` |
| DR schedule | `multi_robot_training/shared/dr_schedule.py` |
| Evaluation pipeline | `4_env_test/src/run_capstone_eval.py` |
| Evaluation configs | `4_env_test/src/configs/eval_cfg.py` |
| Deploy script | `deploy_multi_robot.py` |
| Lessons learned | `4_env_test/LESSONS_LEARNED.md` |
| Integration plan | `multi_robot_training/INTEGRATION_PLAN.md` |

### Observation Vector Cheat Sheet

| Index | Dims | What |
|-------|------|------|
| 0:3 | 3 | Base linear velocity (body frame) |
| 3:6 | 3 | Base angular velocity |
| 6:9 | 3 | Projected gravity |
| 9:12 | 3 | Velocity commands (vx, vy, omega_z) |
| 12:24 | 12 | Joint positions (offset from default) |
| 24:36 | 12 | Joint velocities |
| 36:48 | 12 | Previous actions |
| 48:235 | 187 | Height scan (17x11 grid, 0.1m resolution) |

### The Deployment Checklist

Before every training run:
- [ ] Deploy latest code: `python deploy_multi_robot.py`
- [ ] Verify file structure on server
- [ ] Run smoke test: 64 envs, 10 iterations, both robots
- [ ] Check: all 19 reward terms produce non-zero values
- [ ] Check: terrain_levels advance (not stuck at 0)
- [ ] Check: body_contact < 1.0 after 100 iterations
- [ ] Launch TensorBoard before full run
- [ ] Verify no zombie processes (`nvidia-smi`)

---

*"In theory, there is no difference between theory and practice. In practice, there is."*
*-- Yogi Berra (possibly)*

*This document was written after spending two weeks discovering that quadrupeds, like people, learn best when you don't start by throwing them off a cliff.*
