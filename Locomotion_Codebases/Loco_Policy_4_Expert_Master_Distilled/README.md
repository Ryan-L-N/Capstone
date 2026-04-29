# Multi-Expert Distillation for Spot Locomotion

Distills two specialist expert policies into a single generalist student that handles all terrain types. Instead of training one policy to be good at everything (which leads to conflicting gradients), we let each expert master its terrain, then teach the student to use both.

## The Problem

Training a single policy on mixed terrain forces trade-offs:
- Loose penalties (good for boulders/stairs) → sloppy flat-ground walking
- Tight penalties (good for friction/grass) → robot can't climb obstacles

We spent days fighting this — every time boulder performance improved, friction/grass regressed.

## The Solution

**Two experts, one student.**

| Expert | Checkpoint | Terrain Specialty |
|--------|-----------|-------------------|
| Friction/Grass | `mason_hybrid_best_33200.pt` | Smooth surfaces, drag terrain |
| Boulder/Stairs | `obstacle_best.pt` | Rock fields, step climbing |

The student learns WHEN to use each expert's behavior by reading the height scan.

## Architecture

```
Observations (235 dims)
    ├── Height Scan (187 dims) ──→ [Roughness = variance] ──→ [Sigmoid Gate]
    │                                                              │
    │                                              gate=0: flat    gate=1: rough
    │                                                │                  │
    │                                        [Friction Expert]  [Obstacle Expert]
    │                                              │                  │
    │                                         friction_action    obstacle_action
    │                                                  \            /
    │                                            [Blended Expert Target]
    │                                                       │
    └──→ [Student Actor] ──→ student_action ──→ [MSE + KL Loss]
                                                       │
                              Total Loss = (1-α)·PPO + α·Distillation
```

**Terrain routing:** The height scan (first 187 obs dims) encodes terrain geometry. Flat terrain has near-zero variance. Boulders/stairs have high variance. A sigmoid gate routes each environment to the appropriate expert.

**DAgger-style:** The student acts in the environment (collects real experience), then experts label what they would have done. This avoids distribution shift from pure behavior cloning.

**Alpha annealing:** Starts at 0.8 (mostly copy experts) → decays to 0.2 (mostly PPO). The student absorbs expert knowledge first, then adapts with its own reward signal.

## Usage

### H100 Training

```bash
# SSH to H100
ssh t2user@172.24.254.24

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# Copy files to H100
scp -r multi_expert_distillation/ t2user@172.24.254.24:~/

# Run distillation
cd ~/multi_expert_distillation
screen -S distill
python distill.py --headless \
    --friction_expert ~/checkpoints/mason_hybrid_best_33200.pt \
    --obstacle_expert ~/checkpoints/obstacle_best.pt \
    --num_envs 4096 \
    --max_iterations 5000 \
    --save_interval 100 \
    --no_wandb
```

### Local Debug

```bash
python distill.py --headless \
    --friction_expert checkpoints/mason_hybrid_best_33200.pt \
    --obstacle_expert checkpoints/obstacle_best.pt \
    --num_envs 64 \
    --max_iterations 10 \
    --no_wandb
```

### Evaluate with 4-Env Gauntlet

The distilled student saves standard RSL-RL checkpoints — directly compatible with the existing eval system:

```bash
python 4_env_test/src/run_capstone_eval.py \
    --robot spot --policy rough --env friction --mason \
    --checkpoint logs/rsl_rl/spot_distill/<run>/model_5000.pt
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--alpha_start` | 0.8 | Initial distillation weight (high = trust experts) |
| `--alpha_end` | 0.2 | Final distillation weight (low = trust PPO) |
| `--kl_weight` | 0.1 | KL vs MSE balance in distillation loss |
| `--roughness_threshold` | 0.005 | Height scan variance gate for routing |
| `--routing_temperature` | 0.005 | Sigmoid sharpness (lower = harder gate) |
| `--init_from` | friction | Initialize student from: friction, obstacle, or scratch |
| `--critic_warmup_iters` | 300 | Freeze actor while critic calibrates |
| `--distill_batch_size` | 8192 | Samples per distillation gradient step |

## File Structure

```
multi_expert_distillation/
├── README.md              # This file
├── config.py              # All hyperparameters (dataclass)
├── expert_router.py       # Height-scan-based terrain routing
├── distillation_loss.py   # MSE + KL loss between student and expert
└── distill.py             # Main training script
```

## How It Works (Step by Step)

1. **Load experts:** Both expert checkpoints are loaded and frozen. No gradients flow through them.

2. **Initialize student:** Student actor weights copied from friction expert (best general gait). Critic left fresh since reward landscape changes with distillation loss.

3. **Critic warmup (300 iters):** Actor is frozen while the critic learns the new value landscape. This prevents the actor from making bad updates before the critic calibrates.

4. **Training loop (each iteration):**
   - Student collects rollout in the environment (standard PPO collection)
   - PPO update runs normally (surrogate loss + value loss)
   - Post-hoc distillation step:
     - Sample 8192 observations from rollout buffer
     - Query both experts on those observations
     - Compute routing gate from height scan variance
     - Blend expert actions based on gate
     - MSE + KL loss between student and blended expert
     - Separate gradient step with alpha scaling

5. **Alpha decay:** Over training, the distillation weight decreases from 0.8 to 0.2. Early on, the student mostly copies experts. Later, it mostly follows its own PPO reward signal.

## Expected Training Time

- **~3,000–5,000 iterations** to match both experts
- **~6–8 hours** on H100 with 4096 envs
- Much faster than training a single policy from scratch (which took 19,000+ iterations and still had trade-offs)

## Monitoring

TensorBoard logs include:
- `Distill/alpha` — current distillation weight
- `Distill/mse_loss` — action MSE between student and expert
- `Distill/kl_loss` — distribution KL divergence
- `Distill/mean_gate` — average routing gate (0=all friction, 1=all obstacle)
- Standard PPO metrics (reward, terrain_levels, flip_rate, etc.)

## Design Decisions

**Why not a mixture-of-experts architecture?**
The student is a standard [512,256,128] MLP — same as both experts. This means the distilled checkpoint is directly compatible with play.py, the 4-env gauntlet, and any existing eval infrastructure. No architecture changes needed.

**Why soft gating instead of hard routing?**
At terrain boundaries (e.g., transition from flat to boulders), the robot needs a smooth blend of behaviors. Hard routing would cause jerky transitions. The sigmoid gate with tunable temperature allows this.

**Why post-hoc distillation instead of inline?**
RSL-RL's PPO update zeroes gradients per mini-batch internally. Injecting distillation loss inline would require patching RSL-RL internals. The post-hoc approach (run PPO update, then do a separate distillation gradient step) is cleaner, simpler, and adds only ~10-15% overhead.

**Why initialize from friction expert?**
The friction expert has the best general locomotion gait (0% flip, 49.5m on smooth terrain). Starting from it means the student already knows how to walk — it just needs to learn when to switch to obstacle behavior. Starting from scratch would waste thousands of iterations relearning basic locomotion.

## Credits

Created for AI2C Tech Capstone — MS for Autonomy, March 2026.
Multi-expert distillation approach inspired by DAgger (Ross et al., 2011) and
multi-expert locomotion policies (Lee et al., 2020).
