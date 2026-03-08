# Spot RL Training - Quick Reference

## Essential Commands

### Training

```bash
# Start training with default settings (4096 envs, 5000 iterations)
python train_spot.py

# Debug mode (16 envs, 100 iterations, for testing)
python train_spot.py --config debug

# Fast training (1024 envs, aggressive learning)
python train_spot.py --config fast

# High-performance training (16k envs, for GPU clusters)
python train_spot.py --config high_performance

# Custom configuration
python train_spot.py --config default --num_envs 2048 --iterations 10000

# Resume from checkpoint
python train_spot.py --resume logs/checkpoints/model_002000.pt

# Training with rendering (slow, for debugging)
python train_spot.py --config debug --render
```

### Monitoring

```bash
# Launch TensorBoard
tensorboard --logdir logs/tensorboard

# View in browser
# http://localhost:6006
```

### Evaluation

```bash
# Evaluate with rendering (10 episodes)
python eval_policy.py --checkpoint logs/checkpoints/model_005000.pt --episodes 10 --render

# Statistical evaluation (100 episodes, reproducible)
python eval_policy.py --checkpoint logs/checkpoints/model_005000.pt --episodes 100 --seed 42

# Generate plots
python eval_policy.py --checkpoint logs/checkpoints/model_005000.pt --episodes 50 --plot

# Compare multiple checkpoints
python eval_policy.py --checkpoint "logs/checkpoints/model_*.pt" --episodes 50 --compare
```

## File Structure

```
RL Folder/
├── spot_rl_env.py           # Core RL environment (700+ lines)
├── training_config.py       # Training configs & hyperparameters (400+ lines)
├── train_spot.py            # Training script (600+ lines)
├── eval_policy.py           # Evaluation script (500+ lines)
├── README_RL.md             # Complete documentation
├── QUICK_REFERENCE.md       # This file
└── logs/                    # Generated during training
    ├── checkpoints/         # Model checkpoints (.pt files)
    ├── tensorboard/         # TensorBoard logs
    └── episodes/            # Episode logs (CSV files)
```

## Key Metrics

### Training Targets
- **Waypoints**: > 15/25 (baseline: 1.68)
- **Success Rate**: > 50% (baseline: 0%)
- **Fall Rate**: < 5% (baseline: 35%)
- **Episode Length**: > 800 steps (baseline: varies)

### Observation Space (92-dim)
- 37-dim: Proprioception (joints, base state, roll/pitch)
- 3-dim: Navigation (waypoint distance/heading, progress)
- 35-dim: Obstacle sensing (5 nearest × 7 features)
- 5-dim: Contact & forces
- 3-dim: Action history

### Action Space (3-dim)
- Forward velocity: [-1.5, 1.5] m/s
- Lateral velocity: [-0.5, 0.5] m/s
- Angular velocity: [-1.0, 1.0] rad/s

### Reward Components
- Waypoint reached: +100
- Distance reduction: +2.0/m
- Forward locomotion: +1.0
- Successful nudge: +5.0
- Smart bypass: +3.0
- Failed nudge: -2.0
- Fall: -100
- Time penalty: -0.01/step

## Training Profiles

| Profile | Envs | Iterations | Use Case | Est. Time |
|---------|------|------------|----------|-----------|
| debug | 16 | 100 | Quick testing | 5-10 min |
| fast | 1024 | 5000 | Rapid prototyping | 2-4 hours |
| default | 4096 | 5000 | Standard training | 4-8 hours |
| stable | 8192 | 5000 | Conservative | 6-12 hours |
| high_perf | 16384 | 10000 | GPU clusters | 12-24 hours |

## Hyperparameter Tuning

### Common Adjustments

**Training is unstable (reward drops, erratic behavior):**
```python
config.ppo.clip_param = 0.15          # More conservative (default: 0.2)
config.ppo.learning_rate = 1e-4       # Slower learning (default: 3e-4)
config.ppo.max_grad_norm = 0.5        # Stricter clipping (default: 1.0)
```

**Training is too slow (minimal improvement):**
```python
config.ppo.learning_rate = 1e-3       # Faster learning (default: 3e-4)
config.env.num_envs = 8192            # More parallel experience (default: 4096)
config.ppo.ppo_epoch = 4              # Fewer epochs (default: 8)
```

**Not reaching waypoints:**
```python
config.rewards.waypoint_reached = 150.0       # Increase (default: 100.0)
config.rewards.distance_reduction = 3.0       # Increase (default: 2.0)
config.rewards.forward_locomotion = 1.5       # Increase (default: 1.0)
```

**Too many collisions:**
```python
config.rewards.collision_penalty = 1.0        # Increase (default: 0.5)
config.rewards.failed_nudge_penalty = 3.0     # Increase (default: 2.0)
config.rewards.smart_bypass = 5.0             # Increase (default: 3.0)
```

**Falling too often:**
```python
config.rewards.stability_reward = 0.5         # Increase (default: 0.2)
config.rewards.height_penalty = 2.0           # Increase (default: 1.0)
config.rewards.fall_penalty = 150.0           # Increase (default: 100.0)
```

## Workflow

### 1. Initial Training
```bash
# Start with debug profile to verify setup
python train_spot.py --config debug

# If successful, run full training
python train_spot.py --config default
```

### 2. Monitor Progress
```bash
# Launch TensorBoard in separate terminal
tensorboard --logdir logs/tensorboard

# Watch key metrics:
# - Mean episode reward (should increase)
# - Waypoints per episode (target > 15)
# - Success rate (target > 50%)
# - Policy/value loss (should stabilize)
```

### 3. Evaluate Checkpoints
```bash
# Quick evaluation of latest checkpoint
python eval_policy.py --checkpoint logs/checkpoints/model_005000.pt --episodes 10 --render

# Statistical evaluation
python eval_policy.py --checkpoint logs/checkpoints/model_005000.pt --episodes 100 --plot
```

### 4. Fine-Tune
```bash
# If performance plateaus, try adjusting hyperparameters
# Edit training_config.py or use command-line overrides

# Resume training with new settings
python train_spot.py --resume logs/checkpoints/model_005000.pt --iterations 10000
```

### 5. Compare Models
```bash
# Compare all saved checkpoints
python eval_policy.py --checkpoint "logs/checkpoints/model_*.pt" --episodes 50 --compare
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce number of environments
python train_spot.py --num_envs 1024

# Or use smaller mini-batches (edit training_config.py)
config.ppo.num_mini_batches = 2  # Default: 4
```

### Isaac Sim Crashes
```bash
# Run in headless mode (no rendering)
python train_spot.py  # Default is headless

# Reduce physics complexity (edit training_config.py)
config.env.num_obstacles = 20  # Default: 40
```

### Policy Not Learning
```bash
# Check observation scaling (should be normalized)
# Check reward magnitudes (should be balanced)
# Try debug mode with rendering to see behavior
python train_spot.py --config debug --render

# Verify baseline locomotion works before adding obstacles
```

### Training Too Slow
```bash
# Use fast profile
python train_spot.py --config fast

# Reduce logging frequency (edit train_spot.py)
config.ppo.log_interval = 100  # Default: 10
config.ppo.save_interval = 500  # Default: 100
```

## Performance Benchmarks

### Baseline (SpotFlatTerrainPolicy)
- Waypoints: 1.68 avg
- Success: 0%
- Falls: 35%
- Timeouts: 65%

### Expected RL Performance (after 5000 iterations)
- Waypoints: 12-18 avg
- Success: 40-60%
- Falls: < 10%
- Nudge success: 60-80%

### Target Performance (well-tuned)
- Waypoints: > 20 avg
- Success: > 70%
- Falls: < 5%
- Nudge success: > 80%

## Checkpoint Management

### Automatic Saves
- Every 100 iterations (default)
- Final model at end of training
- Format: `model_XXXXXX.pt` (6-digit iteration number)

### Manual Save/Load
```python
# In training_config.py, adjust:
config.ppo.save_interval = 50  # Save every 50 iterations

# Resume from any checkpoint:
python train_spot.py --resume logs/checkpoints/model_003500.pt
```

### Keep Best Models
- Evaluate multiple checkpoints with `--compare`
- Identify best performing iteration
- Archive best model separately

## Next Steps

1. **Test setup**: Run debug training (5-10 min)
2. **Full training**: Run default profile (4-8 hours)
3. **Monitor**: Watch TensorBoard for progress
4. **Evaluate**: Test best checkpoint on 100 episodes
5. **Fine-tune**: Adjust hyperparameters based on results
6. **Deploy**: Use best policy for navigation tasks

## Support

- Full documentation: [README_RL.md](README_RL.md)
- Environment details: [spot_rl_env.py](spot_rl_env.py)
- Config reference: [training_config.py](training_config.py)

---

Author: Cole | Date: February 2026
