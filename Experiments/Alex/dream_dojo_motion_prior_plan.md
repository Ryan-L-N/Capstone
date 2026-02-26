# Dream Dojo: Reverse-Engineering Spot Locomotion from Video

## Overview

Use Boston Dynamics' official YouTube footage as a free, high-quality motion capture dataset to reverse-engineer Spot's gaits. Extract joint trajectories from video, build motion priors, and train RL policies in Isaac Sim that produce naturalistic, BD-quality locomotion — starting with minimal data.

This builds on top of the terrain-aware auto-gait switching system (see `auto_gait_switching_plan.md`) by replacing the current RL-only policies with **motion-prior-guided policies** that move like the real Spot.

---

## Why This Works

1. **BD YouTube = Gold Standard Data** — Professionally shot, multiple angles, controlled environments, showcasing gaits BD spent years tuning
2. **Known Kinematics** — Spot's URDF is in the SDK (`BD_SPOT_GIT/protos/`), so imperfect video pose estimation can be corrected with inverse kinematics
3. **Small Data is Enough** — AMP (Adversarial Motion Priors) works with **5-10 seconds of reference motion per gait type**
4. **Multi-Angle Coverage** — BD often shows the same movement from multiple camera angles in one video, enabling 3D reconstruction

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1: VIDEO → MOTION                  │
│                                                             │
│  BD YouTube ──► Frame Extraction ──► Keypoint Detection     │
│  (ffmpeg)        (30fps clips)       (DeepLabCut/ViTPose)   │
│                                           │                 │
│                                           ▼                 │
│                                    2D Joint Positions       │
│                                           │                 │
│                         ┌─────────────────┤                 │
│                         ▼                 ▼                 │
│                   Multi-View         Monocular + IK         │
│                   Triangulation      Constraint Solve       │
│                         │                 │                 │
│                         └────────┬────────┘                 │
│                                  ▼                          │
│                         3D Joint Angles                     │
│                     (matched to Spot URDF)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   PHASE 2: MOTION PRIORS                    │
│                                                             │
│  Joint Trajectories ──► AMP Discriminator Training          │
│  (5-10s per gait)       (learns "what real Spot looks like")│
│                                  │                          │
│                                  ▼                          │
│                         Motion Prior Network                │
│                    (scores how "Spot-like" a motion is)     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               PHASE 3: PRIOR-GUIDED RL TRAINING             │
│                                                             │
│  Isaac Lab + RSL-RL (PPO)                                   │
│                                                             │
│  reward = task_reward + style_reward                        │
│           ▲               ▲                                 │
│           │               │                                 │
│    (velocity tracking,    (AMP discriminator score:         │
│     survival, energy)      "does this look like real Spot?")│
│                                                             │
│  Trains across: flat, rubble, stairs, friction arenas       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 4: DEPLOYMENT & REFINEMENT               │
│                                                             │
│  Auto-gait switching (already implemented) selects the      │
│  best motion-prior policy based on terrain difficulty.      │
│  Fine-tune thresholds with naturalistic gaits in the loop.  │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Video to Motion Data

### 1.1 Target Videos from BD YouTube

Minimum viable dataset — **4-5 clips, ~5-10s each**:

| Gait Type | Video Source | What to Extract | Duration |
|-----------|-------------|-----------------|----------|
| Flat walk/trot | "Spot Launch" / "Spot Robot Testing" | Steady-state walking at ~1 m/s | 5-8s |
| Rough terrain | "Spot Autonomous Navigation" | Walking over rocks/debris | 5-8s |
| Stair climb | "Spot Climbing Stairs" | Ascending + descending | 5-8s per direction |
| Recovery/push | "Spot Robot - Testing Robustness" | Push recovery, stumble correction | 3-5s |
| Parkour/jump | "New Moves" / Atlas+Spot compilations | Dynamic transitions, leaps | 3-5s |

**Selection criteria:**
- Clear side/diagonal view of all 4 legs
- Minimal motion blur
- Known or estimable camera distance
- Bonus: same sequence shown from multiple angles

### 1.2 Frame Extraction

```bash
# Extract frames at 30fps from a BD YouTube clip
yt-dlp -f "bestvideo[height>=1080]" -o "spot_walk.mp4" <URL>
ffmpeg -i spot_walk.mp4 -vf "fps=30,crop=1920:1080:0:0" -q:v 2 frames/walk_%04d.jpg
```

Trim to the useful segment only — we need 150-240 frames per gait (5-8s at 30fps).

### 1.3 Keypoint Detection

**Option A: DeepLabCut (recommended for quadrupeds)**
- Pre-trained on quadruped skeletons, fine-tune on ~50 hand-labeled Spot frames
- Outputs 2D pixel coordinates for: 4x (hip, knee, foot) + body center + head = 14 keypoints
- Label mapping to Spot joints:
  ```
  fl_hx, fl_hy, fl_kn  →  front-left hip_x, hip_y, knee
  fr_hx, fr_hy, fr_kn  →  front-right hip_x, hip_y, knee
  hl_hx, hl_hy, hl_kn  →  hind-left hip_x, hip_y, knee
  hr_hx, hr_hy, hr_kn  →  hind-right hip_x, hip_y, knee
  ```

**Option B: ViTPose / MMPose**
- More general-purpose, may need custom skeleton definition
- Better for single-frame accuracy, worse for temporal consistency

**Option C: SAM2 + Tracking (fallback)**
- Segment Spot's body with SAM2, track joint regions across frames
- Less precise but requires zero labeled data

### 1.4 2D to 3D Lifting

**If multi-view available** (same movement, different angles in the video):
- Camera calibration from known Spot dimensions (body length = 1.1m, height = 0.5m standing)
- Triangulate 2D keypoints → 3D world coordinates
- Direct geometric solution, most accurate

**If monocular only** (single camera angle):
- Use Spot URDF as a kinematic constraint
- For each frame: find joint angles `q` that minimize:
  ```
  argmin_q  ||project(FK(q)) - detected_2D_keypoints||^2
            + lambda * ||q - q_prev||^2           (temporal smoothness)
            + mu * joint_limit_penalty(q)          (valid range)
  ```
- FK = forward kinematics from URDF, project = camera projection
- Spot's known link lengths make this well-constrained even from a single view

### 1.5 Output Format

```python
# Per-gait motion clip: dictionary of numpy arrays
motion_clip = {
    "joint_positions": np.array,   # (T, 12) joint angles in radians
    "joint_velocities": np.array,  # (T, 12) finite-difference velocities
    "base_linear_vel": np.array,   # (T, 3) body linear velocity
    "base_angular_vel": np.array,  # (T, 3) body angular velocity
    "base_orientation": np.array,  # (T, 4) body quaternion [w,x,y,z]
    "dt": float,                   # timestep (1/30 = 0.033s)
    "gait_label": str,             # "walk", "trot", "stair", "recovery"
}
# Save as .npz or .pkl for AMP training
```

---

## Phase 2: Adversarial Motion Priors (AMP)

### 2.1 Concept

AMP trains a **discriminator** that distinguishes "real Spot motion" (from video) vs "policy-generated motion" (from RL). The RL policy then gets a **style reward** for producing motions that fool the discriminator.

```
Discriminator: D(s_t, s_{t+1}) → [0, 1]
  - 1 = "this transition looks like real Spot"
  - 0 = "this looks like RL jitter/unnatural motion"

Style reward: r_style = -log(1 - D(s_t, s_{t+1}))
```

### 2.2 Discriminator Architecture

```
Input: state transition (s_t, s_{t+1}) — concatenated
  s = [joint_pos(12), joint_vel(12), base_ang_vel(3), base_lin_vel(3)]
  → input dim = 2 * 30 = 60

Network: 60 → 256 (ReLU) → 128 (ReLU) → 1 (sigmoid)

Training:
  - Positive samples: transitions from extracted BD video clips
  - Negative samples: transitions from current RL policy rollouts
  - Loss: standard GAN binary cross-entropy
  - Gradient penalty (GP) for stability
```

### 2.3 Integration with RSL-RL (PPO)

The existing training pipeline uses RSL-RL with PPO. AMP adds a style reward term:

```python
# In reward function (rewards/reward_terms.py)
total_reward = (
    w_task * task_reward          # velocity tracking, survival, energy
    + w_style * style_reward      # AMP discriminator score
)

# Typical weights:
# w_task = 0.5, w_style = 0.5 (equal emphasis)
# Start with w_style = 0.7 to strongly enforce style, then anneal
```

### 2.4 Training Data Requirements

| Gait | Clip Duration | Transitions at 50Hz | Enough? |
|------|--------------|---------------------|---------|
| Walk | 5s | 250 | Yes — AMP paper used 200-500 |
| Rough terrain | 5s | 250 | Yes |
| Stair climb | 8s | 400 | Yes |
| Recovery | 3s | 150 | Marginal — augment with mirroring |

**Data augmentation:**
- Mirror left/right (doubles data for free)
- Small temporal jitter (shift by ±1-2 frames)
- Additive Gaussian noise on joint angles (σ = 0.01 rad)

---

## Phase 3: Prior-Guided RL Training

### 3.1 Modified Training Config

```python
@configclass
class SpotAMPEnvCfg(SpotRoughEnvCfg):
    """Spot environment with AMP motion prior."""

    # AMP discriminator
    amp_motion_files = [
        "motion_clips/spot_walk.npz",
        "motion_clips/spot_rough.npz",
        "motion_clips/spot_stairs.npz",
        "motion_clips/spot_recovery.npz",
    ]
    amp_reward_weight = 0.5        # style reward weight
    amp_disc_update_freq = 2       # update discriminator every N policy updates
    amp_replay_buffer_size = 100000
    amp_gradient_penalty = 5.0     # GP coefficient for discriminator stability
```

### 3.2 Training Phases

**Phase 3a: Style Imitation (first 5k iterations)**
- `w_style = 0.8`, `w_task = 0.2`
- Policy learns to produce Spot-like motions regardless of task
- Discriminator converges on what "real" motion looks like

**Phase 3b: Task + Style (5k-20k iterations)**
- Anneal to `w_style = 0.5`, `w_task = 0.5`
- Policy learns to achieve velocity targets while maintaining style
- Train across all 4 arenas (flat, rubble, stairs, friction)

**Phase 3c: Task Emphasis (20k-30k iterations)**
- `w_style = 0.3`, `w_task = 0.7`
- Fine-tune task performance while keeping naturalistic motion
- Domain randomization cranked up for robustness

### 3.3 Expected Outcome vs Current Policies

| Aspect | Current RL-Only | With AMP Motion Prior |
|--------|----------------|----------------------|
| Gait naturalness | Robotic, jerky at transitions | Smooth, BD-like cadence |
| Foot placement | Optimal but unnatural | Natural stepping pattern |
| Body posture | Whatever minimizes reward | Maintains Spot's characteristic stance |
| Energy efficiency | May exploit physics | Closer to real-world efficiency |
| Recovery style | Flailing | BD-style recovery motions |

---

## Phase 4: Integration with Auto-Gait System

The terrain-aware auto-gait switching (already implemented in `spot_lava_arena.py`) becomes the deployment layer:

```
TerrainDifficultyAssessor
         │
         ▼
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  AMP-FLAT    │     │  AMP-ROUGH   │     │  AMP-PARKOUR │
  │  (walk clip  │     │  (rough clip │     │  (stair/jump │
  │   prior)     │     │   prior)     │     │   prior)     │
  └──────────────┘     └──────────────┘     └──────────────┘
         ▲                    ▲                    ▲
         │                    │                    │
    D < 0.02            0.04 < D < 0.12        D > 0.12
```

Each gait mode now uses a policy trained with the corresponding motion prior, so transitions between gaits also look natural.

---

## Tool & Dependency Stack

| Tool | Purpose | Install |
|------|---------|---------|
| **yt-dlp** | Download BD YouTube videos | `pip install yt-dlp` |
| **ffmpeg** | Frame extraction from video | System install |
| **DeepLabCut** | 2D keypoint detection on quadrupeds | `pip install deeplabcut` |
| **ViTPose** (alt) | General-purpose pose estimation | `pip install mmpose` |
| **SAM2** (alt) | Zero-shot segmentation + tracking | `pip install segment-anything-2` |
| **OpenCV** | Camera calibration, image processing | `pip install opencv-python` |
| **Pinocchio** | Inverse kinematics with URDF constraints | `pip install pin` |
| **Isaac Lab + RSL-RL** | RL training with PPO | Already installed (`isaaclab311`) |
| **PyTorch** | Discriminator network training | Already installed |

---

## Spot URDF & Kinematic Reference

From the cloned SDK (`BD_SPOT_GIT/`):

```
BD_SPOT_GIT/
├── protos/bosdyn/api/
│   ├── spot/robot_command.proto    # Joint names, limits, command format
│   ├── geometry.proto              # SE3Pose, Quaternion, Vec3
│   └── hazard_avoidance.proto      # Terrain classification (already used)
└── python/examples/
    └── spot_cam/                   # Camera intrinsics reference
```

**Spot joint configuration (12 DOF):**
```
Joint Name    | Type  | Range (rad)     | Default (standing)
──────────────┼───────┼─────────────────┼───────────────────
fl_hx (hip X) | Roll  | [-0.78, 0.78]   | +0.1
fl_hy (hip Y) | Pitch | [-0.89, 2.79]   | +0.9
fl_kn (knee)  | Pitch | [-2.79, -0.09]  | -1.5
fr_hx         | Roll  | [-0.78, 0.78]   | -0.1
fr_hy         | Pitch | [-0.89, 2.79]   | +0.9
fr_kn         | Pitch | [-2.79, -0.09]  | -1.5
hl_hx         | Roll  | [-0.78, 0.78]   | +0.1
hl_hy         | Pitch | [-0.89, 2.79]   | +1.1
hl_kn         | Pitch | [-2.79, -0.09]  | -1.5
hr_hx         | Roll  | [-0.78, 0.78]   | -0.1
hr_hy         | Pitch | [-0.89, 2.79]   | +1.1
hr_kn         | Pitch | [-2.79, -0.09]  | -1.5
```

**Body dimensions (for camera calibration):**
- Body length: ~1.1m (hip-to-hip)
- Body width: ~0.5m (left-right hip spacing)
- Standing height: ~0.5m (hip to ground)
- Total height: ~0.84m (top of body)

---

## Minimum Viable Pipeline (Quick Start)

For a first proof-of-concept with the least effort:

1. **1 video** — Download BD "Spot - Testing Robustness" (has walking + push recovery, multiple angles)
2. **DeepLabCut** — Label 30-50 frames by hand, train keypoint detector (~2 hours)
3. **Monocular IK** — Use Spot URDF + detected 2D keypoints → joint angles (no multi-view needed)
4. **1 motion clip** — Extract 5s of walking (~150 transitions)
5. **Simple discriminator** — Train on walking clip, add style reward to existing rough terrain training
6. **Compare** — Run AMP policy vs current policy in lava arena, compare gait naturalness

**Estimated time to first result:** 1-2 days of focused work (labeling is the bottleneck).

---

## Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Poor keypoint detection on Spot's dark body | Noisy joint trajectories | Use high-contrast frames, temporal smoothing, IK constraint solving |
| Single-view depth ambiguity | Incorrect 3D reconstruction | Spot's known kinematics heavily constrain the solution; multi-view when available |
| Discriminator overfits to small dataset | Style reward becomes uninformative | Data augmentation (mirroring, noise), gradient penalty, early stopping |
| AMP reward conflicts with task reward | Policy can't achieve both | Anneal style weight from 0.8 → 0.3 over training; task always wins eventually |
| Video frame rate (30fps) vs sim rate (50Hz) | Temporal mismatch | Interpolate motion clips to 50Hz with cubic spline before AMP training |

---

## References

- **AMP**: Peng et al., "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Animation" (SIGGRAPH 2021)
- **ASE**: Peng et al., "ASE: Large-Scale Reusable Adversarial Skill Embeddings" (SIGGRAPH 2022)
- **DreamWaQ**: Locomotion from limited demonstrations for quadrupeds
- **DeepLabCut**: Mathis et al., "DeepLabCut: markerless pose estimation" (Nature Neuroscience 2018)
- **Isaac Lab AMP**: NVIDIA Isaac Lab includes AMP implementation in `omni.isaac.lab_tasks`
