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

**FAQ: "Do the robots collide with each other?"** No. Each of the 20,480 robots exists in its own isolated physics instance (`replicate_physics=True`). They share the same GPU simulation kernel for performance, but they cannot physically interact -- no collisions, no contact forces, no shared objects. Each robot is assigned to its own 8m x 8m terrain patch by the curriculum system. The `collision_group=-1` setting on the terrain mesh means the ground collides with every robot, but robots from different environments are invisible to each other. What looks like 20,000 robots sharing a world is actually 20,000 independent physics simulations running in parallel. In `play.py`, robots appear close together (`env_spacing=2.5`) for visualization purposes only -- during headless training they're spread across a much larger terrain grid.

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

### Bug #13: "The Termination That Should Never Have Been" (Body Contact Kill)

**What happened:** Even after matching the paper's exact reward coefficients (Trial 2), Spot still couldn't learn. body_contact termination climbed to 99.3%, episode length shrank to 4.68 steps. Same death spiral, just slower.

**Root cause:** We did a **full config comparison** -- not just rewards, but every section -- between our config and the paper's actual Spot terrain training config (`spot_env_cfg.py`). The smoking gun:

```python
# Paper's config (spot_env_cfg.py, lines 528-532):
# # Base contact termination
# body_contact = DoneTerm(
#     func=mdp.illegal_contact,
#     params={"sensor_cfg": ..., "threshold": 1.0},
# )
# ^^^ COMMENTED OUT. Deliberately disabled.

# Our config:
body_contact = DoneTerm(
    func=mdp.illegal_contact,
    params={"sensor_cfg": ..., "threshold": 1.0},
)
# ^^^ ACTIVE. Killing 99.3% of episodes.
```

The paper's authors discovered that Spot's low body geometry inevitably makes brief ground contact during normal locomotion -- stepping off a curb, recovering from a push, traversing rough terrain. A 1-Newton threshold is triggered by the slightest brush. With this termination active, the robot can never survive more than a few steps, so it never gets enough experience to learn.

**What the paper does instead:**
1. **`body_flip_over`** termination at 150 degrees -- only kills if the robot is nearly upside down
2. **`undesired_contacts`** reward penalty at -2.0 -- teaches the robot to *avoid* body contact via gradient signal, not death

**Additional findings from the full config comparison:**
- Our friction minimums (0.02-0.05) were near-zero ice. Paper uses 0.3 minimum.
- Our action scale (0.25) was 25% larger than the paper's (0.2), causing more violent random movements.
- Paper sets `disable_contact_processing = True` in PhysX. We didn't.

**The fix (4 changes):**
1. Replace `body_contact` termination with `body_flip_over` (bad_orientation, 150 deg)
2. Add `undesired_contacts` reward penalty (-2.0)
3. Raise friction minimums to 0.3
4. Reduce action scale to 0.2

**The result:** At iteration 0 of Trial 3, episode length jumped from 27.0 to 31.77. Only 0.24% flip-over terminations vs 12.6% body_contact. The robot is surviving full episodes from the very first iteration.

**The lesson:** Reward coefficients are only half the story. Termination conditions, physics parameters, friction ranges, and action scaling are equally critical. When matching a paper, compare **every single section** of the config -- not just the reward table. The paper's authors commented out body_contact for a reason. That reason cost us two failed trials and 15+ hours of GPU time.

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

### Bug #14: "The Noise That Ate the Robot" (Exploration Death Spiral)

**What happened:** After fixing Bug #13 (body contact termination), Trial 3 started beautifully -- 0.24% flip-over at iteration 0. By iteration 100: 97.8% flip-over. The robot hadn't learned to do anything except fall over more aggressively.

**Root cause:** `init_noise_std=1.0` with `action_scale=0.2` produces action noise of ~N(0, 0.2) radians -- that's ±11 degrees of random joint movement per step. For a top-heavy robot like Spot, this is violent enough to flip it over. Without the hard `body_contact` kill (removed in Bug #13 fix), the 150-degree flip-over threshold is lenient -- the robot can flail for many steps before triggering it. The result:

1. Random policy (noise=1.0) flips robot in most episodes
2. No useful gradient signal (everything ends the same way)
3. PPO increases noise_std to "explore more"
4. Higher noise = more violent actions = more flips = even less signal
5. Positive feedback loop until noise hits ceiling (1.68 in Trial 3, 2.0 in Trial 4)

This ran for **8,745 iterations** in Trial 3 (~40 hours of GPU time) before the value function exploded to infinity. The robot never learned a single thing.

**The fix:** Lower `init_noise_std` to 0.5 (actions ±6° instead of ±11°) and cap `max_noise_std` at 1.0. Conservative enough to maintain balance while still discovering gaits.

**The lesson:** Removing a termination condition doesn't just change what kills the robot -- it changes the entire exploration landscape. With body_contact killing episodes in <5 steps, high noise was immediately punished (short episodes = low returns). Without it, high noise episodes survive longer (reaching the lenient 150° threshold slowly), so the return signal is *noisier* (pun intended), and PPO interprets this as "need more exploration." The termination was also serving as an implicit noise regularizer. When you remove it, you need explicit noise control.

---

### Bug #15: "The LR Schedule That Wasn't" (Adaptive Overrides Cosine)

**What happened:** Training script applied cosine LR annealing before each PPO update. TensorBoard showed the LR collapsing to near-zero for 8,700 iterations, then spiking to 0.01 at the end. The cosine schedule was being ignored.

**Root cause:** RSL-RL's `schedule="adaptive"` adjusts the learning rate *inside* `PPO.update()` based on KL divergence vs `desired_kl`. Our cosine annealing sets the optimizer LR *before* calling `update()`. Then the adaptive schedule immediately overwrites it:

```python
# Our code (before update):
optimizer.param_groups[0]["lr"] = cosine_lr(iter)  # e.g., 5e-4

# RSL-RL's PPO.update() (inside update):
if kl > 2 * desired_kl:
    optimizer.param_groups[0]["lr"] /= 1.5  # overwritten to 3.3e-4
```

With the robot not learning (Bug #14), KL stayed near zero for thousands of iterations. The adaptive schedule kept multiplying by 1.5 (low KL = "not exploring enough"). But our cosine kept resetting to a lower value each iteration. The net effect was chaos -- LR crushed to ~0 for most of training, then a sudden spike when conditions changed.

**The fix:** Set `schedule="fixed"` in the PPO config. Let the cosine annealing monkey-patch be the sole LR controller.

**The lesson:** When overriding internal optimizer behavior from the outside, make sure the framework isn't also overriding it from the inside. Read the source code of `PPO.update()` to understand what `schedule="adaptive"` actually does before combining it with your own schedule.

---

### Bug #16: "Domain Randomization Before Locomotion" (Aggressive DR on Untrained Robot)

**What happened:** Trial 3 used ±8N external forces, ±3Nm torques, ±1.5 m/s pushes, and ±8kg mass randomization. The Isaac Lab reference config for Spot uses: zero forces, zero torques, ±0.5 m/s pushes, ±5kg mass.

**Root cause:** We copied the "aggressive DR" from the 100hr config paper without realizing that config was designed for a *pre-trained* robot being fine-tuned for robustness. For a robot learning from scratch, these perturbations are catastrophic. Getting hit with 8N of force every reset while trying to learn basic balance is like teaching a baby to walk by repeatedly pushing them over.

**The fix (Trial 4):** Match the reference config exactly: zero external forces/torques, ±0.5 pushes, ±5kg mass, (10,15)s push interval. This delayed the death spiral by ~70 iterations (flip_over at iter 8 was 76% vs Trial 3's 97.8% at iter 100).

**The lesson:** DR is for robustness, not learning. The robot needs to learn basic locomotion first with gentle conditions, THEN you can gradually increase perturbations. Start from the reference config's DR values, prove the robot walks, then dial up.

---

### Bug #18: "The Warmup That Never Ended" (LR Climbing for Entire Run)

**What happened:** Trial 6 trained beautifully for 273 iterations -- reward hit 384, robot walking perfectly, only 5.4% flip-over. Then value_loss jumped from 0.2 to 1,233 to 3.3 billion to infinity in 5 iterations. The entire policy was lost because save_interval was 500 (no intermediate checkpoints).

**Root cause:** `warmup_iters=500` on a `max_iterations=500` run. The LR linearly increased from 1e-5 toward 1e-3 for the *entire* training run -- it never transitioned to cosine decay. By iter 273, LR was ~5.5e-4 and still climbing. Combined with noise growing to 0.77, the value function couldn't handle updates that large.

**The fix (Trial 7):** `warmup_iters=50` (quick warmup, then cosine decay) and `save_interval=50` (checkpoint every 50 iterations). But `lr_max=1e-3` was still too high -- Trial 7 exploded at iter 102 when LR peaked near 9.7e-4.

**The real fix (Trial 7b):** `lr_max=3e-4`. The value function stayed stable for the entire 500-iteration run. Noise decreased from 0.52 to 0.38. Policy converged cleanly.

**The lesson:** LR must match training stability. A warmup run with a fragile early policy needs a lower LR than a mature training run. Save checkpoints frequently during experimental runs -- disk is cheap, lost training is not.

---

### Bug #22: "The Height That Wasn't" (World-Frame Z on Rough Terrain)

**What happened:** Trial 10b disabled the action smoothness explosion (by capping terrain difficulty) but a new killer emerged: `body_height_tracking = -52.45`. The robot was being massively penalized just for standing on top of stairs.

**Root cause:** The `body_height_tracking_penalty` uses `root_pos_w[:, 2]` -- the robot's Z position in **world frame** -- with a target of 0.42m. On flat terrain at z=0, a robot at z=0.42 gets error=0 (perfect). On stairs where the ground is at z=1.0, the robot at z=1.42 gets error = (1.42-0.42)² = 1.0. On a 2m pyramid: error = (2.42-0.42)² = 4.0. The penalty grows **quadratically with terrain elevation**.

This bug hid for 9 trials because flat terrain has ground at z=0, making world-frame Z identical to height-above-ground. The moment we introduced real terrain elevation, the penalty exploded.

**The fix:** `env_cfg.rewards.body_height_tracking.weight = 0.0` for any non-flat terrain. A proper fix would use height above local terrain surface (from the height scan), but disabling it works for now.

**The lesson:** Any reward term that uses absolute world-frame position will break on non-flat terrain. Always check whether your rewards are terrain-relative or world-frame. This includes height tracking, base position tracking, and any distance-from-origin measurements.

---

### Bug #23: "The Value Function Whiplash" (LR Too High for Terrain Transitions)

**What happened:** Trial 10c fixed the height bug but used `lr_max=3e-4` (same as Phase A and A.5). Value loss went 31 → 101 → 4,670 in 25 iterations. Crashed.

**Root cause:** The value function was trained on transition terrain (50% flat + gentle rough). On robust_easy (0% flat, 12 terrain types), its predictions are completely wrong -- it expects flat-terrain returns but sees rough-terrain returns. At lr=3e-4, each gradient update massively overcorrects the value function, causing oscillation that amplifies each iteration.

**The fix:** Progressive LR reduction. Trial 10d (`lr_max=1e-4`) survived to iter ~1319 before NaN crash. Trial 10g (`lr_max=1e-4` + NaN fix) survived to iter ~1134 before value explosion to 2.4×10²¹. Trial 10h (`lr_max=5e-5`) is stable past iter 1608 with value_loss at 7-16.

**The lesson:** Every terrain transition is a distribution shift for the value function. The safe learning rate decreases with each transition because the value predictions become less accurate. Phase A (flat): lr=3e-4 works. Phase A.5 (transition): lr=3e-4 works. Phase B-easy (robust): need lr=5e-5. The pattern: when you change the terrain, cut the LR aggressively — halving is not enough, you may need 6x reduction.

---

### Bug #21: "The Action Smoothness Landmine" (Squared Penalties + Chaotic Falls = Infinity)

**What happened:** Trial 10 resumed from the transition-trained model_998.pt onto full robust terrain. After ~15 iterations, action_smoothness exploded to -103 trillion, noise std went negative, crash.

**Root cause:** The action smoothness penalty computes the squared difference between consecutive actions. When the robot falls chaotically on unfamiliar hard terrain, its actions become extreme and erratic. Squaring those differences creates exponential penalty growth: if the action difference is 100x normal, the squared penalty is 10,000x normal. This cascading overflow corrupted the entire policy in a single update step.

**The fix (Trial 10b):** `--terrain robust_easy` with `num_rows=3` instead of 10. Same 12 terrain types but capped difficulty. The robot falls less chaotically → action differences stay bounded → smoothness penalty stays finite. 8.6% flip_over vs 63%.

**The lesson:** Squared penalties (action smoothness, joint acceleration) are landmines during terrain transitions. They amplify chaotic behavior exponentially. When introducing new terrain types, cap the difficulty low enough that the robot's failures are "graceful stumbles" not "catastrophic flips." The penalty scaling handles the rest.

---

### Bug #20: "Terrain Shock" (Flat to Robust Is Too Big a Jump)

**What happened:** Trial 8 resumed from the flat-trained model_498.pt (99.3% survival, noise 0.38) onto the full 12-type robust terrain. Within 40 iterations: flip_over 96.5%, noise at ceiling (1.00), value_loss 482,431, reward -101 and falling. Same death spiral as Trials 3-5.

**Root cause:** The curriculum controls difficulty *within* each terrain type (row 0 = easy stairs, row 9 = hard stairs). But even easy stairs are qualitatively different from flat ground -- the robot has never experienced height changes under its feet. With 12 terrain types, the robot encounters stairs, gaps, slopes, stepping stones, and obstacles all at once. It fails on ~95% of patches and gets no useful gradient.

**The fix (Trial 9):** Added `--terrain transition` -- an intermediate phase with 50% flat + gentle versions of 5 terrain types (slopes at half angle, stairs at half height, roughness at half amplitude). The flat-trained policy survives immediately (1.2% flip_over) while learning new skills.

**The lesson:** Curriculum steps must be small enough that the policy can transfer without catastrophic failure. If survival drops below ~70% on the new terrain, the step is too big. The number of new terrain *types* matters as much as the difficulty within each type.

---

### Bug #24: "The Clamp That Didn't" (NaN Passes Through clamp_)

**What happened:** Trial 10d crashed overnight with `RuntimeError: normal expects all elements of std >= 0.0`. We added a safety clamp that runs `policy.std.clamp_(min=0.3, max=1.0)` before every `policy.act()` call. Trial 10e crashed with the *exact same error* -- the clamp was visibly in the stack trace but didn't prevent the crash.

**Root cause:** `torch.clamp_()` does **not** fix NaN values. `NaN.clamp_(min=0.3)` returns NaN, not 0.3. When the optimizer pushes std parameters to NaN (via gradient explosion from a value function blowup), clamping is useless -- NaN is not less than 0.3, not greater than 1.0, not anything. It passes through every comparison unchanged.

```python
# This does NOT work:
policy.std.clamp_(min=0.3, max=1.0)  # NaN stays NaN

# This works:
bad = torch.isnan(param.data) | torch.isinf(param.data) | (param.data < 0)
param.data[bad] = min_val  # Replace NaN/Inf/negative with safe value
param.data.clamp_(min=min_val, max=max_val)  # Then clamp
```

**The fix:** `_sanitize_std()` in `shared/training_utils.py` -- explicitly detects and replaces NaN, Inf, and negative values before clamping. Registered via `register_std_safety_clamp()` which monkey-patches `policy.act()` to sanitize before every forward pass.

**The lesson:** Never assume `clamp_()` handles pathological values. NaN is not a number -- it's a black hole that swallows every operation. When guarding against gradient explosions, always check for NaN/Inf explicitly. This applies to any safety mechanism in training: if your safety net assumes finite values, NaN will walk right through it.

---

### Bug #25: "The Slow Bleed" (Value Loss Oscillation Cascade)

**What happened:** Trial 10h (lr_max=5e-5) looked stable past iter 1608 where 1e-4 had crashed. Reward climbed to ~155 at iter 1900-2200, then *slowly* declined: 155 → 142 → 112 → 58 → 8 over 2000 iterations. At iter 4037, value_function_loss went 7.6 → 411K → 170M → 8.5B → NaN. The NaN sanitizer (`_sanitize_std`) did not trigger because the NaN originated in the value function, not in the action noise std.

**Root cause:** Same disease as Bug #23 (value function whiplash from terrain distribution shift), but in slow motion. At lr=5e-5, the value loss didn't explode instantly -- instead it *oscillated wildly* even during the "good" period: 10 → 56 → 193 → 210 → 973 → 28 → 8 → 5792 → 11734. Each spike partially damaged the policy. Over thousands of iterations, these accumulated damages degraded performance until the value function finally exploded to NaN. Lowering LR (3e-4 → 1e-4 → 5e-5) only delays the crash; it doesn't break the oscillation → amplification cycle.

**The fix:** Added a **value loss watchdog** in `train_ppo.py`'s `update_with_schedule` wrapper. After each PPO update, it checks `result["value_function"]`. If value_loss exceeds a threshold (100.0), the effective LR is halved for 50 iterations. This breaks the amplification cycle: a spike triggers reduced LR, which prevents the overcorrection, which prevents the next spike. Also lowered lr_max to 3e-5 for additional margin.

```python
# Value loss watchdog (in update_with_schedule):
vl = result.get("value_function", 0.0)
if vl > _VL_THRESHOLD:  # 100.0
    _vl_penalty[0] = 0.5
    _vl_cooldown[0] = 50  # halve LR for 50 iters
```

**The lesson:** Safety mechanisms must guard the actual failure point. The NaN sanitizer guarded std (the *symptom*), but the disease was in the value function. When a training failure has a slow onset (hours of gradual decline before catastrophic collapse), look for oscillating metrics -- they're the early warning sign that a cascade is building. A reactive guard (detect spike → reduce LR) is more robust than a fixed LR because it adapts to the actual instability rather than guessing how low the LR needs to be.

---

### Bug #26: "The Noisy Ceiling" (Curriculum Stall from Forced Exploration)

**What happened:** Trial 10h reached reward ~155 with terrain_levels stuck at ~0.8 for thousands of iterations. The curriculum never advanced past the easy/medium boundary. More training made things *worse* — reward declined from 155 to 8 over 2000 iterations before NaN crash.

**Root cause:** `max_noise_std=1.0` forced the policy to maintain maximum exploration. With `entropy_coef=0.01`, the optimizer kept pushing std upward and the clamp caught it every iteration. On easy terrain (row 0-1), the robot is robust enough to walk despite noisy actions. On hard terrain (row 2), the noise causes falls — 23% flip rate, stable for thousands of iterations. The curriculum creates an equilibrium: robots get promoted from easy terrain (walk >4m easily) then demoted from hard terrain (flip and walk nowhere). The policy can't learn to survive hard terrain because it's *forced to be noisy*.

**The fix:** Lower `--max_noise_std` from 1.0 to 0.7. This immediately lets the policy take more precise actions on hard terrain → fewer random falls → lower flip rate → curriculum advances. In Trial 10j, reward at iter 2025 was already 130 (vs 12 with std=1.0 in Trial 10i) and survival was 41% (vs 22%).

**The lesson:** Exploration (high noise std) and exploitation (low noise std) must be balanced by phase. Phase A on flat terrain benefits from high exploration — the robot needs to discover walking. Phase B on hard terrain needs exploitation — the robot already knows how to walk and needs to do it *precisely*. If the curriculum stalls, check whether noise_std is at its ceiling. A clamped std means the policy wants to explore more than it should, and the clamp is the only thing stopping it. Lower the ceiling to match the phase.

---

### Bug #19: "The Learning Rate Ceiling" (1e-3 Is Too Hot for Early Training)

**What happened:** Trial 7 fixed the warmup (50 iters) but kept `lr_max=1e-3`. The robot was learning well (reward 133, ep_len 1,181 at iter 100), then noise suddenly spiked from 0.70 to 0.86 and value_loss went from 0.61 to 3.4e24 in a single iteration.

**Root cause:** After the cosine warmup completed at iter 50, the LR was near its peak of 1e-3. At this rate, a single noisy batch can produce a catastrophically large update to the value function. The noise spike (0.70 → 0.86) created one such batch, and the value function never recovered.

**The pattern:** Trial 6 exploded at iter 274 when LR reached ~5.5e-4. Trial 7 exploded at iter 102 when LR was ~9.7e-4. Both times: growing noise + high LR → value explosion. The safe LR ceiling appears to be around 3-5e-4 for early-stage training.

**The fix (Trial 7b):** `lr_max=3e-4`. Stayed stable for 500 iterations with monotonically decreasing noise. The value function converged smoothly to 0.09.

**The lesson:** The maximum safe learning rate depends on the training phase. Early training (unstable policy, high noise) needs lower LR. Mature training (stable policy, low noise) can tolerate higher LR. When in doubt, use 3e-4 and let cosine annealing handle the schedule.

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

**23. Compare the FULL config, not just the rewards.** We matched the paper's reward coefficients exactly and still failed -- because the body_contact termination was killing 99% of episodes before the rewards could do anything. Termination conditions, friction ranges, action scale, and physics settings are just as important as reward weights. When reproducing a paper, diff every section of the config.

**24. Hard terminations are training killers.** A hard termination (instant episode death) gives zero gradient information -- the policy just learns "don't be in that state" without learning what to do instead. A reward penalty (-2.0 for body contact) gives a smooth gradient that teaches the policy to gradually avoid the behavior. Replace hard terminations with soft penalties whenever possible, especially during early training.

**25. Zombie processes hold VRAM but not compute.** A D-state zombie from a killed Isaac Sim holds GPU memory (preventing new allocations) but doesn't consume compute cycles. If you have headroom, the remaining process will speed up. But eventually you need to reset the GPU to reclaim the memory.

### On Curriculum Learning

**26. Flat terrain first, always.** Five consecutive trials (Trials 1-5) failed trying to teach locomotion and terrain traversal simultaneously. Trial 6 proved the robot could learn to walk in 200 iterations on flat terrain. The gradient on flat terrain is 100% locomotion signal. On rough terrain, it's dominated by "don't trip" noise. Master the prerequisite skill before adding complexity.

**27. Save checkpoints like your training depends on it (because it does).** Trial 6 trained perfectly for 273 iterations, then exploded and we lost everything because `save_interval=500`. Trial 7 saved every 50 iterations and we recovered from `model_50.pt` to eventually complete the run. 21MB per checkpoint × 10 checkpoints = 210MB. That's nothing compared to the 1.7 hours of GPU time each run costs.

**28. Learning rate must match training maturity.** `lr_max=1e-3` works for a mature policy on a long training run. It's catastrophic for an early-training fragile policy on a short warmup. Trial 7 exploded at iter 102 with LR near 1e-3. Trial 7b ran to completion with `lr_max=3e-4`. The safe ceiling during Phase A warmup is ~3-5e-4. Phase B with a stable policy may tolerate higher.

**29. Noise trend is your canary.** In every failed trial, noise_std grew monotonically (0.5 → 0.77 → 0.86). In the successful Trial 7b, noise *decreased* (0.52 → 0.38). If noise is growing, the policy isn't converging -- it's becoming more uncertain. This is the earliest warning sign, often visible 50-100 iterations before a crash.

**30. When the terrain changes, re-evaluate your penalties.** A penalty that makes sense on flat terrain can be catastrophic on rough terrain. `undesired_contacts=-5.0` was fine for Phase A (flat terrain, body contact = the robot fell). On Phase B's 12-terrain curriculum, the robot WILL contact obstacles with its body -- that's physics, not failure. We lowered it to -1.5 and added a targeted `body_scraping` term that penalizes *sustained* belly contact while moving, not momentary bumps. The principle: prefer targeted penalties over blanket punishments, and always ask "is this penalty still valid in the new environment?"

**31. Curriculum steps must be survivable.** Trial 8 proved that flat --> 12-type robust is too large a jump (96.5% flip_over). Trial 9's transition terrain (50% flat + gentle rough) achieved 1.2% flip_over from iteration 1. The rule of thumb: if the policy's survival rate drops below ~70% on the new terrain, the curriculum step is too large. The number of new terrain *types* matters as much as difficulty -- stairs, gaps, and slopes are qualitatively different from flat, regardless of how "easy" they are. Bridge the gap with an intermediate phase.

**32. Terrain complexity has two axes: type novelty and difficulty.** Trial 10 showed that even after transition training, full robust (12 types × 10 difficulty rows) crashed in 15 iterations with 63% flip_over. Trial 10b uses the same 12 types but only 3 difficulty rows and it's stable at 8.6% flip_over. The robot can learn many new terrain types if difficulty is capped, or hard difficulty on familiar types -- but not both at once. The four-phase curriculum separates these: flat (walk) --> transition (gentle types) --> robust_easy (all types, low difficulty) --> robust (all types, full difficulty).

**33. Squared penalties are exponential landmines.** Action smoothness and joint acceleration penalties compute squared differences. When the robot falls chaotically, actions become 100x normal -- the squared penalty becomes 10,000x normal. Trial 10's action_smoothness hit -103 trillion from one bad iteration. These terms are safe during stable training but become weapons during terrain transitions. Cap terrain difficulty low enough that failures are "stumbles" not "catastrophic flips."

**34. World-frame rewards are terrain-specific.** `body_height_tracking` uses absolute Z position, not height above terrain. It worked perfectly on flat terrain (ground at z=0) and hid for 9 trials. The moment we added real elevation, it penalized the robot for standing on stairs. Check every reward term: does it use world-frame coordinates? If yes, it will break on rough terrain. Disable or rewrite before terrain transitions.

**35. Cut the learning rate aggressively at every terrain transition.** Each terrain change is a distribution shift for the value function. Phase A to A.5: lr=3e-4 was fine. Phase A.5 to B-easy: lr=3e-4 exploded in 25 iters, lr=1e-4 survived ~300 iters then exploded, lr=5e-5 is stable past iter 1600+. The value function's predictions become less accurate with each transition, and "halving the LR" isn't enough — you may need a 6x reduction. The pattern: 3e-4 → 3e-4 → 5e-5 across A → A.5 → B-easy.

**36. NaN propagates through clamp_().** `torch.clamp_(min=0.3, max=1.0)` does NOT fix NaN values — NaN.clamp_(min=0.3) returns NaN. When gradient explosions push policy std to NaN during PPO mini-batch updates, you must explicitly detect and replace bad values with `torch.isnan() | torch.isinf() | (data < 0)` before clamping. See `_sanitize_std()` in `training_utils.py`.

**37. save_interval=100 prevents catastrophic progress loss.** At 20,480 envs, each iteration is ~655K steps. save_interval=500 means ~328M steps between checkpoints — if the run crashes, you lose hours of training. save_interval=100 (~65M steps) costs only ~2.1GB per 1000 iters and lets you recover from within 33 minutes of the crash point.

**38. Lower max_noise_std for later training phases.** In Phase A (flat), high exploration (std=1.0) helps discover gaits. In Phase B (rough terrain), the robot already knows how to walk and needs *precision*. If noise_std is pinned at its ceiling every iteration, the entropy bonus is overwhelming the policy gradient's desire to exploit. Lower the ceiling: 1.0 for Phase A, 0.7 for Phase B-easy. A stalled curriculum (terrain_levels stuck) is often a sign that noise is too high — the robot can't survive hard terrain because its actions are too noisy.

**39. Value loss oscillation is the early warning sign.** Before a NaN crash, the value loss oscillates by orders of magnitude (10 → 200 → 970 → 5800 → 11700). Each spike partially damages the policy. A value loss watchdog that detects spikes and temporarily reduces LR breaks the amplification cycle. Look for oscillating metrics — they mean a cascade is building.

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

### Trial 2: Paper-Matched Coefficients (Feb 28, 2026) -- FAILED

**Config:** Exact paper coefficients from `Robust_RL/quadruped_locomotion/.../spot_reward_env_cfg.py`
**Envs:** 10,000 Spot only | **Hardware:** H100 96GB (solo, clean GPU after reset)
**Log dir:** `spot_robust_ppo/2026-02-28_08-52-14/`

**Key coefficient changes from Trial 1:**
```
base_linear_velocity:  +12.0 --> +5.0    gait:  +15.0 --> +5.0
foot_clearance:        +3.5  --> +0.75   foot_slip:  -3.0  --> -0.5
base_orientation:      -5.0  --> -3.0    base_motion: -4.0  --> -2.0
joint_pos:             -2.0  --> -0.7    joint_acc:   -5e-4 --> -1e-4
joint_torques:         -2e-3 --> -5e-4   joint_vel:   -5e-2 --> -1e-2
dof_pos_limits:        -10.0 --> -5.0    stumble:     -0.3  --> -0.1
```

**Results after 1,471 iterations (~7 hours):**
```
body_contact termination:  12.6% --> 99.3%
mean episode length:       27.0  --> 4.68 steps
terrain_levels:            3.39  --> 0.017
time_out:                  1.0%  --> 0.7% (peaked at 3.1% mid-run, then fell back)
mean_reward:               -4.09 --> +0.20 (positive but stagnant)
```

**Diagnosis:** Reward weights were correct but the **termination condition was the real problem** (Bug #13). A thorough comparison of our full config against the paper's actual Spot terrain config (`spot_env_cfg.py`) revealed that the paper **deliberately disabled body_contact termination** for Spot and replaced it with a flip-over check + reward penalty. Our robot was dying in 4.68 steps from momentary body-ground contact during normal locomotion.

**Outcome:** FAILED -- same trajectory as Trial 1 (just slower). Killed training, applied structural fixes.

---

### Trial 3: Structural Fixes — No Body Contact Kill (Feb 28, 2026) -- FAILED

**Config:** Paper-matched coefficients + 5 structural fixes from full config comparison
**Envs:** 10,000 Spot only | **Hardware:** H100 96GB (solo, clean GPU after reset)
**Log dir:** `spot_robust_ppo/2026-02-28_16-05-11/`

**What changed from Trial 2 (5 fixes):**

1. **REMOVED body_contact termination** → Added `body_flip_over` (bad_orientation at 150 deg)
2. **Added `undesired_contacts` reward** (weight=-2.0) — soft penalty replaces hard kill
3. **Raised friction minimums** — static (0.05→0.3), dynamic (0.02→0.3)
4. **Reduced action scale** — 0.25 → 0.2
5. **Added `disable_contact_processing = True`**

**What happened:** Ran for 8,745 iterations (~40 hours) before crashing. By iteration 100, body_flip_over hit 97.8% — the *exact same death spiral* as Trials 1 and 2, just with a different termination name. The robot was flipping over instead of making body contact, but just as dead.

```
iter 0:    flip_over=0.24%   ep_len=31.8   noise_std=1.0   value_loss=13.8
iter 100:  flip_over=97.8%   ep_len=21.7   noise_std=1.68  value_loss=1.8
iter 8000: flip_over=96.4%   ep_len=45.2   noise_std=1.39  value_loss=1.78
iter 8745: flip_over=96.0%   ep_len=21.2   noise_std=1.79  value_loss=INF  <-- CRASH
```

**Two bugs discovered (Bug #14 and Bug #15):**

1. **Bug #14 — Exploration Noise Death Spiral:** `init_noise_std=1.0` with `action_scale=0.2` produces actions ~N(0, 0.2) rad (±11°). Violent enough to flip Spot. Without hard termination to instantly punish flipping, the 150° threshold is too lenient. Most actions flip the robot → no useful gradient → noise_std *increases* → more violent actions → more flips. Classic positive feedback loop. Noise went 1.0 → 1.68 by iter 100.

2. **Bug #15 — LR Schedule Override:** RSL-RL's `schedule="adaptive"` overrides our cosine annealing inside `PPO.update()`. The adaptive schedule crushed LR to ~0 for 8,700 iterations (no learning), then spiked to 0.01 → value function → inf → crash.

**Also discovered:** DR was 3-16x more aggressive than the Isaac Lab reference config (±8N forces vs zero, ±3Nm torques vs zero, ±1.5 push vs ±0.5). See Bug #16.

---

### Trial 4: DR + LR + Reward Fixes (Mar 2, 2026) -- FAILED

**Config:** Reference-matched DR + LR fix + reward boost
**Envs:** 10,000 Spot only | **Hardware:** H100 96GB
**Log dir:** `spot_robust_ppo/2026-03-02_10-39-33/`

**8 changes from Trial 3:**
1. Zeroed external forces (±8N → 0) and torques (±3Nm → 0)
2. Reduced mass DR: ±8kg → ±5kg
3. Reduced push velocity: ±1.5 → ±0.5 m/s
4. Increased push interval: (5,12) → (10,15)s
5. Reduced joint reset velocity: ±3.0 → ±2.5
6. Increased gait weight: 5.0 → 10.0
7. Increased foot_clearance: 0.75 → 2.0
8. Fixed LR: `schedule="adaptive"` → `schedule="fixed"` (cosine annealing works correctly)

**What happened:** Crashed at iteration ~80. DR fixes delayed the death spiral by ~70 iterations — 76% flip_over at iter 8 vs Trial 3's 97.8% at iter 100. Episode length peaked at 118 steps (4x Trial 3's peak). But the fundamental noise spiral (Bug #14) still killed it.

```
iter 0:   flip_over=0.24%  ep_len=31.9   noise_std=1.0   value_loss=12.5
iter 8:   flip_over=76.1%  ep_len=82.3   noise_std=1.02  value_loss=8.6
iter 77:  flip_over=95.5%  ep_len=117.9  noise_std=2.0   value_loss=212,568  <-- CRASH
```

**Diagnosis:** DR fixes were necessary but not sufficient. The exploration noise is the root cause. With `init_noise_std=1.0`, the random actions are too violent for Spot's balance regardless of DR settings.

---

### Trial 5: Noise Stabilization (Mar 2, 2026) -- FAILED

**Config:** Trial 4 config + noise fixes targeting Bug #14
**Envs:** 10,000 Spot only | **Hardware:** H100 96GB
**Log dir:** `spot_robust_ppo/2026-03-02_12-15-09/`

**3 changes from Trial 4:**
1. `init_noise_std`: 1.0 → 0.5
2. `max_noise_std`: 2.0 → 1.0
3. `undesired_contacts` weight: -2.0 → -5.0

**What happened:** Best early metrics yet — 21.6% flip_over at iter 1, episode length peaked at 216 steps at iter 26, value loss stayed stable at 3.71 (vs 212K crash in Trial 4). But by iter 101, flip_over climbed back to 96.2% and noise hit the 1.0 ceiling. The noise ceiling prevented a crash but the robot still couldn't learn.

```
iter 0:   flip_over=0.18%  ep_len=31.9   noise=0.50  value_loss=6.2
iter 1:   flip_over=21.6%  ep_len=62.6   noise=0.50  value_loss=6.1
iter 26:  flip_over=80.6%  ep_len=216.2  noise=0.58  value_loss=7.9
iter 101: flip_over=96.2%  ep_len=54.2   noise=1.00  value_loss=3.7
```

**Diagnosis (Bug #17):** The robot is on 12-type rough terrain from step 0. Even with gentle noise, it trips on stairs/gaps/slopes and flips. The gradient is dominated by "don't trip on terrain" instead of "learn to walk." Five trials have now proven that learning locomotion and terrain traversal simultaneously doesn't work for Spot. The robot needs to learn to walk on flat terrain first.

---

### Trial 6: Flat Terrain Warmup (Mar 2, 2026) -- FAILED (value explosion at iter 274)

**Config:** Trial 5 config + flat terrain warmup (two-phase training)
**Envs:** 10,000 Spot only | **Hardware:** H100 96GB
**Log dir:** `spot_robust_ppo/2026-03-02_13-01-42/`

The flat terrain approach *worked beautifully* for 273 iterations -- reward hit 384, episode length maxed at 1,500, only 5.4% flip-over. The robot actually learned to walk! Then the value function exploded to infinity and the entire run was lost.

| Iter | Reward | Ep Length | Flip Over | Noise Std | Value Loss |
|------|--------|-----------|-----------|-----------|------------|
| 2 | 2.87 | 70 | 50.8% | 0.50 | 0.397 |
| 199 | 374 | 1,500 | 5.4% | 0.58 | 0.216 |
| 273 | 318 | 1,279 | 3.8% | 0.77 | 1,233 |
| 274 | -8,844 | — | — | 0.77 | **inf** |

**Diagnosis (Bug #18):** `warmup_iters=500` on a 500-iteration run = LR never stopped climbing (5.5e-4 at crash). `save_interval=500` = zero intermediate checkpoints. Entire policy lost.

---

### Trial 7: Lower LR Warmup (Mar 2, 2026) -- FAILED (value explosion at iter 102)

**Config:** Trial 6 + `warmup_iters=50`, `save_interval=50`
**Envs:** 10,000 Spot only | **Hardware:** H100 96GB
**Log dir:** `spot_robust_ppo/2026-03-02_15-21-03/`

Fixed the warmup but kept `lr_max=1e-3`. The robot was learning well (reward 133, ep_len 1,181 at iter 100), then noise spiked 0.70 → 0.86 and the value function detonated at iter 102.

**Diagnosis:** `lr_max=1e-3` is too high for a fragile early-training policy. At iter 100 the cosine schedule had LR at ~9.7e-4. In Trial 6, the same explosion happened when LR reached ~5.5e-4 at iter 274. The threshold is somewhere around 5e-4.

**Saved:** `model_50.pt` (clean, noise=0.59) -- the save_interval fix worked!

---

### Trial 7b: Lower LR + Resume (Mar 2, 2026) -- SUCCESS (Phase A COMPLETE)

**Config:** Resume from Trial 7 `model_50.pt` + `lr_max=3e-4`
**Envs:** 10,000 Spot only | **Hardware:** H100 96GB
**Log dir:** `spot_robust_ppo/2026-03-02_15-55-25/`

The winning combination. Lower learning rate kept the value function stable for the entire run.

| Iter | Reward | Ep Length | Flip Over | Time Out | Noise Std | Value Loss |
|------|--------|-----------|-----------|----------|-----------|------------|
| 51 (resume) | 1.6 | 30 | 0.03% | 1.0% | 0.52 | 0.975 |
| 110 | 141 | 1,155 | 28.1% | 71.9% | 0.54 | 0.220 |
| 199 | 375 | 1,500 | 5.4% | 94.6% | 0.58 | 0.216 |
| 300 | 520 | 1,500 | 1.5% | 98.5% | 0.42 | 0.100 |
| **498** | **567** | **1,500** | **0.7%** | **99.3%** | **0.38** | **0.09** |

**This is the first successful training run in the entire project.** Key results:
- 99.3% of episodes survive the full length (0.7% flip over)
- Noise *decreased* from 0.52 to 0.38 -- the policy learned precision
- Value loss converged to 0.09 -- no explosion
- 10 checkpoints saved across the run
- 143M timesteps, 1.7 hours wall time, 24K steps/sec

**Checkpoint:** `model_498.pt` (20.6 MB) -- flat terrain walking policy, ready for Phase B.

**What we learned:** Flat terrain + low LR + frequent checkpoints = stable training. The robot has 7 trials of failures behind it and finally knows how to walk.

---

### Trial 8: Phase B -- Direct Rough Terrain Transfer (Mar 2, 2026) -- FAILED

**Config:** Phase A checkpoint (model_498.pt) + 2 reward changes for rough terrain
**Envs:** 20,480 Spot | **Hardware:** H100 96GB (solo, clean GPU)
**Log dir:** `spot_robust_ppo/2026-03-02_19-33-05/`
**Resume from:** `spot_robust_ppo/2026-03-02_15-55-25/model_498.pt`

**What changed from Phase A:**
```
undesired_contacts:  -5.0 --> -1.5   (rough terrain = body bumps are unavoidable)
body_scraping:       NEW  --> -2.0   (penalize belly-dragging at speed, allow momentary bumps)
terrain:             flat --> robust  (12-type curriculum, 400 patches)
```

**Results (~40 iterations after resume):**
```
flip_over:       95.0% --> 96.5%      (not improving at all)
noise_std:       0.50  --> 1.00       (hit ceiling -- death spiral)
value_loss:      ---   --> 482,431    (exploding)
reward:          -54   --> -101       (getting worse)
terrain_levels:  1.30  --> 1.18       (stuck at bottom)
```

**Diagnosis (Bug #20):** The jump from 100% flat to 12-type rough terrain was too large. Even the easiest curriculum rows have stairs, gaps, and slopes -- terrain *types* the robot has never seen. The flat-trained policy flips on 96.5% of patches, gets no gradient signal, noise grows to ceiling, same death spiral as Trials 3-5. The curriculum controls *difficulty* within each type, but can't make stairs feel like flat ground.

**Outcome:** FAILED -- killed at iter 538. Added `--terrain transition` option.

---

### Trial 9: Phase A.5 -- Transition Terrain (Mar 2, 2026) -- IN PROGRESS

**Config:** Phase A checkpoint (model_498.pt) + transition terrain (50% flat + gentle rough)
**Envs:** 10,000 Spot | **Hardware:** H100 96GB (solo, clean GPU)
**Resume from:** `spot_robust_ppo/2026-03-02_15-55-25/model_498.pt`

**The idea:** Bridge the gap between flat and robust. Instead of 12 harsh terrain types, use 6 gentle ones:
```
50% flat               (safe zone -- policy can keep practicing walking)
15% gentle slopes      (max 14 degrees, half of robust's 29 degrees)
10% slight roughness   (noise 0.01-0.06, half of robust's 0.02-0.15)
10% gentle stairs      (max 0.10m step, half of robust's 0.25m)
10% wave terrain       (amplitude 0.02-0.08, half of robust's 0.05-0.20)
5%  vegetation plane   (drag training)
```

Only 5 difficulty rows (vs robust's 10), 20 columns. The robot gets to practice walking on mostly-flat terrain while gradually encountering gentle obstacles.

**Early results (iter 502, ~4 iterations after resume):**

| Metric | Failed Phase B | **Phase A.5** |
|--------|---------------|---------------|
| flip_over | 96.5% | **1.2%** |
| noise_std | 1.00 (ceiling) | **0.39** (stable!) |
| value_loss | 482,431 | **8.5** |
| reward | -101 | **+10.35** |
| terrain_levels | 1.18 (stuck) | **2.29** (climbing!) |
| time_out | 2.8% | **10.1%** |

Night and day. The transition terrain is gentle enough that the flat-trained policy can survive while learning new skills. Stumble penalty (-1.12) is the biggest hit -- the robot is learning to deal with obstacles for the first time.

**Plan:** 1000 iterations on transition terrain, then proceed to full robust terrain (Phase B).

---

### Trial 10: Phase B -- Full Robust Terrain (Mar 2, 2026) -- FAILED

**Config:** Phase A.5 checkpoint (model_998.pt) + full ROBUST_TERRAINS_CFG (12 types, 10 rows)
**Envs:** 20,480 Spot | **Hardware:** H100 96GB (solo, clean GPU)
**Log dir:** `spot_robust_ppo/2026-03-02_21-58-52/`

**What happened:** Crashed after ~15 iterations.
```
action_smoothness:  -103,329,900,265,472  (negative 103 TRILLION)
flip_over:          63%
crash:              RuntimeError: normal expects all elements of std >= 0.0
```

The action smoothness term exploded because the robot was falling chaotically on unfamiliar hard terrain -- actions became extreme, squared differences went to infinity, corrupted the entire policy. Noise std went negative (NaN), crash.

**Diagnosis (Bug #21):** Transition terrain (50% flat) to full robust (~10% flat) is still too steep. 63% flip_over is better than Trial 8's 95% (so transition training helped) but not enough. The action smoothness penalty is a landmine -- it squares action differences, so chaotic falling creates exponential penalty growth.

**Key insight:** The robot needs to see all 12 terrain types at LOW difficulty before seeing them at FULL difficulty. Two independent axes: terrain type novelty × terrain difficulty. Can't crank both at once.

---

### Trial 10b: Phase B -- Robust Easy (Mar 2, 2026) -- IN PROGRESS (overnight)

**Config:** Phase A.5 checkpoint (model_998.pt) + all 12 terrain types, `num_rows=3` (capped difficulty)
**Envs:** 20,480 Spot | **Hardware:** H100 96GB (solo, clean GPU)
**Resume from:** `spot_robust_ppo/2026-03-02_19-53-10/model_998.pt`

**The idea:** Same 12 terrain types as full robust, but only 3 difficulty rows instead of 10. The robot sees stairs, gaps, slopes, stepping stones -- everything -- but only the easy-to-medium versions. Max stair ~0.10m instead of 0.25m, max gap ~0.20m instead of 0.50m.

**First iteration results:**
```
flip_over:          8.6%      (vs 63% on full robust -- stable!)
action_smoothness:  -0.35     (vs -103 trillion -- normal!)
value_loss:         39.7      (elevated but stable, not exploding)
ep_length:          89.8      (short but surviving)
reward:             -26.6     (negative but learning)
```

Running overnight. 30K iterations, ~21 hour ETA. Checkpoints every 500 iterations.

---

### Trial 10c: Phase B -- Robust Easy, Height Fix (Mar 2, 2026) -- FAILED

**Config:** Phase A.5 checkpoint + `body_height_tracking` disabled (Bug #22 fix) + `lr_max=3e-4`
**Envs:** 20,480 Spot | **Hardware:** H100 96GB
**Log dir:** `spot_robust_ppo/2026-03-02_22-34-19/`

**What happened:** Fixed the height tracking bug but LR was still too high for the terrain distribution shift. Value loss went 31 → 101 → 4,670 in 25 iterations. Flip_over climbed 15% → 39% → 59%. Crashed.

**Diagnosis:** The value function was trained on transition terrain (50% flat). On robust_easy (0% flat, 12 types), its predictions are wildly wrong. At lr=3e-4, each update overcorrects, causing oscillation → explosion. Need lower LR to let the value function converge gradually.

---

### Trial 10d: Phase B -- Robust Easy, Lower LR (Mar 2, 2026) -- FAILED (NaN crash at iter ~1319)

**Config:** Phase A.5 checkpoint + height tracking disabled + `lr_max=1e-4`
**Envs:** 20,480 Spot | **Hardware:** H100 96GB
**Log dir:** `spot_robust_ppo/2026-03-02_22-50-53/`
**Log file:** `~/phase_b_easy3.log`
**Resume from:** `spot_robust_ppo/2026-03-02_19-53-10/model_998.pt`

**The journey to get here:**
```
Trial 8:   flat --> robust (12 types, 10 rows)      = 95% flip, terrain shock
Trial 9:   flat --> transition (50% flat, gentle)    = SUCCESS (93% survival)
Trial 10:  transition --> robust (12 types, 10 rows) = 63% flip, action_smooth=-103T, crash
Trial 10b: transition --> robust_easy (12 types, 3 rows) = 52% flip, height_tracking=-52, crash
Trial 10c: same + height tracking disabled           = 59% flip, value explosion at lr=3e-4
Trial 10d: same + lr=1e-4                            = 71% survival, then NaN crash at ~1319
Trial 10e: same + NaN clamp fix + lr=3e-4 (mistake)  = clamp doesn't fix NaN, crash
Trial 10f: same + NaN sanitizer + lr=3e-4 (mistake)  = NaN-zombie, all metrics NaN
Trial 10g: NaN sanitizer + lr=1e-4                    = value explosion at ~1134
Trial 10h: NaN sanitizer + lr=5e-5                    = STABLE at iter 1608+ <<<
```

Ten attempts to get from flat terrain to rough terrain. Each failure taught us something:
- Trial 8: need intermediate terrain phase
- Trial 9: transition terrain works
- Trial 10: need to cap terrain difficulty
- Trial 10b: world-frame height rewards break on rough terrain
- Trial 10c: LR must decrease at each terrain transition
- Trial 10d: lr=1e-4 helps but NaN std crashes mid-update
- Trial 10e: clamp_() does NOT fix NaN values (Bug #24)
- Trial 10f: NaN sanitizer works but lr=3e-4 corrupts policy
- Trial 10g: lr=1e-4 still too high — value explosion at ~1134
- Trial 10h: lr=5e-5 stable past 1600 but curriculum stalled at 0.8, then value loss cascade to NaN at 4037
- Trial 10j: lr=3e-5 + max_noise_std=0.7 + value loss watchdog — IN PROGRESS

**Results (Trial 10h — peaked then collapsed):**

| Iter | Reward | Ep Length | Flip Over | Time Out | Value Loss | Terrain Levels |
|------|--------|-----------|-----------|----------|------------|----------------|
| 1000 (resume) | -16 | 91 | 7.9% | 5.0% | 39 | 0.3 |
| ~1025 (danger zone) | +8 | 430 | 49.4% | 23.0% | 967 | 0.5 |
| 1100 (recovery) | +117 | 1,099 | 42.3% | 55.8% | 53 | 0.6 |
| 1400 | +145 | 1,127 | 24.0% | 74.0% | 7 | 0.8 |
| **2000 (peak)** | **+155** | **1,180** | **23.0%** | **75.0%** | **16** | **0.8** |
| 2800 (declining) | +112 | 905 | 22% | 76% | 193 | 0.79 |
| 3500 (death spiral) | +24 | 195 | 26% | 72% | 11 | 0.74 |
| 4037 (NaN crash) | NaN | 1,500 | 0% | 100% | NaN | 0.70 |

The curriculum stalled at terrain_levels ~0.8 because noise_std was pinned at 1.0 (max), causing too many random falls on harder terrain. The value loss oscillated by orders of magnitude (10 → 193 → 973 → 5792 → 11734) throughout, eventually cascading to NaN. More training made things worse, not better.

**Crashed overnight at ~iter 1319** (1h43m elapsed, 209M timesteps) with:
```
RuntimeError: normal expects all elements of std >= 0.0
```

The noise std was at the ceiling (1.00) for most of the run. During a PPO mini-batch update, an optimizer step pushed the `policy.std` parameter to NaN. The post-update `clamp_noise_std()` never got to run because the crash happened mid-update. See Bug #24.

**Only checkpoint:** `model_1000.pt` (saved at iter 1000, before any B-easy learning). All 319 iterations of progress were lost.

**Lesson:** Save more frequently during dangerous terrain transitions. `save_interval=100` would have preserved progress.

---

### Trial 10e: Phase B -- Robust Easy, NaN Clamp Fix (Mar 3, 2026) -- FAILED (NaN crash at iter ~1035)

**Config:** Trial 10d config + `register_std_safety_clamp()` monkey-patch on `policy.act()` + **lr_max=3e-4** (mistake)
**Envs:** 20,480 Spot | **Hardware:** H100 96GB
**Log dir:** `spot_robust_ppo/2026-03-03_07-59-37/`
**Log file:** `~/phase_b_easy4.log`
**Resume from:** `spot_robust_ppo/2026-03-02_22-50-53/model_1000.pt`

**The fix attempt:** Added a monkey-patch on `policy.act()` to clamp `policy.std` to [0.3, 1.0] before every forward pass, not just after the full PPO update. This should catch the std going negative between mini-batches.

**What happened:** Same crash at iter ~1035. The safety clamp DID execute (visible in stack trace), but `clamp_()` does **not fix NaN values** -- `NaN.clamp_(min=0.3)` returns NaN. The optimizer pushed std to NaN (not just negative), and our clamp passed NaN through unchanged. See Bug #24.

Also used `lr_max=3e-4` instead of 10d's working `1e-4`. This was a mistake -- the curriculum template says 3e-4 but the empirical evidence from 10c and 10d clearly shows 1e-4 is needed for B-easy. Two bugs stacked: wrong LR + insufficient NaN handling.

---

### Trial 10f: Phase B -- Robust Easy, NaN Sanitizer (Mar 3, 2026) -- FAILED (NaN-corrupted policy)

**Config:** Trial 10e config + `_sanitize_std()` replacing clamp with NaN/Inf/negative detection + **lr_max=3e-4** (same mistake)
**Envs:** 20,480 Spot | **Hardware:** H100 96GB
**Log dir:** `spot_robust_ppo/2026-03-03_08-15-23/`
**Log file:** `~/phase_b_easy5.log`
**Resume from:** `spot_robust_ppo/2026-03-02_22-50-53/model_1000.pt`

**The fix:** Replaced `policy.std.clamp_()` with a proper sanitizer:
```python
bad = torch.isnan(param.data) | torch.isinf(param.data) | (param.data < 0)
if bad.any():
    param.data[bad] = min_val
param.data.clamp_(min=min_val, max=max_val)
```

**What happened:** The sanitizer prevented the crash! Training kept running. But at `lr_max=3e-4`, the policy weights NaN-corrupted within ~35 iterations. The sanitizer kept std valid so `Normal.sample()` didn't crash, but the rest of the network (actor, critic) was producing NaN outputs. By iter ~1035: noise_std=NaN, value_loss=NaN, most rewards=NaN, flip_over=81.7%.

The training was a zombie -- alive but brain-dead.

**Diagnosis:** The NaN sanitizer is necessary (prevents crashes) but not sufficient. The root cause is `lr_max=3e-4` causing value explosion that corrupts all network weights, not just std. Once the actor weights are NaN, sanitizing std is like putting a band-aid on a corpse.

---

### Trial 10g: Phase B -- Robust Easy, Full Fix (Mar 3, 2026) -- FAILED (value explosion at iter ~1134)

**Config:** NaN sanitizer + **lr_max=1e-4** (matching Trial 10d's working config)
**Envs:** 20,480 Spot | **Hardware:** H100 96GB (fresh reboot, clean GPU, zero zombies)
**Log dir:** `spot_robust_ppo/2026-03-03_08-50-56/`
**Log file:** `~/phase_b_easy7.log`
**Resume from:** `spot_robust_ppo/2026-03-02_22-50-53/model_1000.pt`

**What happened:** NaN sanitizer prevented the std crash (Bug #24 fix worked), but value_loss oscillated wildly (400 → 7,500 → 668 → 828) and then exploded to 2.4×10²¹ around iter ~1134. The lr_max=1e-4 is still too aggressive for the terrain transition — the value function can't track the new terrain distribution fast enough without overshooting.

**Conclusion:** `lr_max=1e-4` is the *minimum* for stable flat/transition training but *too high* for B-easy terrain transitions. Need to go lower.

---

### Trial 10h: Phase B -- Robust Easy, Lower LR (Mar 3, 2026) -- FAILED (value loss cascade at iter ~4037)

**Config:** NaN sanitizer + **lr_max=5e-5** + **save_interval=100**
**Envs:** 20,480 Spot | **Hardware:** H100 96GB
**Log dir:** `spot_robust_ppo/2026-03-03_09-59-28/`
**Log file:** `~/phase_b_easy8.log`
**Resume from:** `spot_robust_ppo/2026-03-02_22-50-53/model_1000.pt`

**What happened:** Reward climbed to ~155 at iter 1900-2200, then slowly declined to 8 over 2000 iterations. Value loss oscillated wildly throughout (10 → 193 → 973 → 5792 → 11734) before finally exploding to NaN at iter 4037. Curriculum stalled at terrain_levels ~0.8 — the 23% flip rate on harder terrain (caused by noise_std pinned at 1.0) created a promotion/demotion equilibrium that more training could not break.

**Root cause (Bug #25 + Bug #26):** Two compounding issues:
1. Value loss oscillation cascade — same as Bug #23 but slower at lr=5e-5. NaN sanitizer only guards std, not the value function.
2. Curriculum stall — max_noise_std=1.0 forced maximum exploration, causing too many random falls on harder terrain. The policy couldn't learn precision.

**Peak checkpoint:** `model_2000.pt` (avg reward ~155, 75% survival, terrain_levels 0.8). 43 checkpoints saved (model_1000 through model_5200).

---

### Trial 10j: Phase B -- Robust Easy, Value Loss Watchdog + Lower Noise (Mar 4, 2026) -- IN PROGRESS

**Config:** NaN sanitizer + value loss watchdog + **lr_max=3e-5** + **max_noise_std=0.7** + save_interval=100
**Envs:** 20,480 Spot | **Hardware:** H100 96GB
**Log dir:** `spot_robust_ppo/2026-03-04_10-30-37/`
**Log file:** `~/phase_b_easy10.log`
**Resume from:** `spot_robust_ppo/2026-03-03_09-59-28/model_2000.pt`

**What's different from 10h:**
1. `lr_max=3e-5` (reduced from 5e-5) — gentler value function updates
2. `max_noise_std=0.7` (reduced from 1.0) — lets policy be precise on hard terrain (Bug #26 fix)
3. Value loss watchdog (Bug #25 fix) — halves LR for 50 iters when value_loss > 100

**Early results (iter ~2025, 7 minutes elapsed):**
- Reward: 130.6 (vs 11.8 with std=1.0 in aborted Trial 10i)
- Ep Length: 710 (vs 371)
- Survival: 41.3% (vs 21.6%)
- Flip Rate: 17.6%
- Terrain Levels: 0.95
- Value loss watchdog caught 4 spikes in first 25 iters, all recovered

**TensorBoard:** `http://172.24.254.24:6006` (this run only, dir `2026-03-04_10-30-37`).

---

### Trial 11: Phase B -- Full Robust (PLANNED)

Resume from Trial 10j best checkpoint with `--terrain robust` (10 difficulty rows). `lr_max=3e-5`, `max_noise_std` TBD (likely 0.5-0.7).

---

### Trial 12: Vision60 (PLANNED)

After Spot succeeds -- same four-phase approach (flat --> transition --> robust_easy --> robust).

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
