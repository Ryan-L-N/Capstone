# Hierarchical RL Navigation Policy for Spot

High-level navigation policy training with curriculum learning. The low-level locomotion is handled by SpotFlatTerrainPolicy (frozen), while the high-level policy learns velocity commands to reach waypoints and avoid obstacles.

## Architecture

### 1. **Hierarchical Control**
- **Low-level**: SpotFlatTerrainPolicy (pretrained, frozen) - handles balance, gait, joint control
- **High-level**: NavigationPolicy (learned) - outputs velocity commands `[vx, vy, omega]`

### 2. **Observation Space (32 dimensions)**
- Base velocity: `[vx, vy, omega]` (3)
- Heading: `[sin(yaw), cos(yaw)]` (2)
- Waypoint info: `[dx, dy, distance]` in robot frame (3)
- Obstacle distances: 16 raycasts around robot, 5m range (16)
- Stage encoding: one-hot vector for curriculum stage (8)

### 3. **Action Space**
- Normalized actions in `[-1, 1]` mapped to:
  - `vx`: forward/back velocity `[-0.5, 2.0]` m/s
  - `vy`: left/right strafe `[-0.5, 0.5]` m/s
  - `omega`: turning rate `[-1.5, 1.5]` rad/s

### 4. **Reward Structure**
- **Time penalty**: -0.01 per step (encourages efficiency)
- **Waypoint capture**: +10.0
- **Progress shaping**: +/- based on distance change (Stages 2-5 only)
- **Fall penalty**: -50.0 (episode terminates)
- **Boundary penalty**: -5.0 if leaving arena

### 5. **Scoring System** (internal game logic)
- Start: 300 points
- Time decay: -1 point/second
- Waypoint bonus: +15 points
- Episode ends when: points ≤ 0 OR all waypoints captured OR fall

## Curriculum (8 Stages)

Success criterion: 80% success rate over last 100 episodes

| Stage | Name | Waypoint Spacing | Obstacles | Goal |
|-------|------|------------------|-----------|------|
| 1 | Random Walking | None | None | No falls for 60s |
| 2 | Waypoints 5m | 5m → 5m | None | Capture all 25 waypoints |
| 3 | Waypoints 10m | 10m → 10m | None | Capture all 25 waypoints |
| 4 | Waypoints 20m | 20m → 20m | None | Capture all 25 waypoints |
| 5 | Waypoints 20m→40m | 20m → 40m | None | Capture all 25 waypoints |
| 6 | Light Obstacles | 20m → 40m | 10% light (pushable) | Capture all 25 waypoints |
| 7 | Heavy Obstacles | 20m → 40m | 5% light + 5% heavy (immovable) | Capture all 25 waypoints |
| 8 | Small Static | 20m → 40m | 5% light + 5% heavy + 10% small | **TRAINING COMPLETE** |

## Files

```
RL_Folder_VS2/
├── nav_config.yaml          # Configuration (arena, rewards, curriculum)
├── navigation_policy.py     # MLP policy network (exportable)
├── navigation_env.py        # Environment (Isaac Sim integration)
├── ppo_trainer.py          # PPO algorithm
├── train_navigation.py     # Main training script
└── checkpoints/            # Saved models
    └── navigation_policy/
        ├── checkpoint_50.pt
        ├── best_model.pt
        ├── stage_X_complete.pt
        └── final_model.pt
```

## Usage

### Training from Scratch

```bash
cd "Experiments/Cole/RL_Folder_VS2"
C:\isaac-sim\python.bat train_navigation.py --headless --iterations 10000
```

### Resume from Checkpoint

```bash
C:\isaac-sim\python.bat train_navigation.py --headless --checkpoint checkpoints/navigation_policy/checkpoint_500.pt
```

### Start from Specific Stage

```bash
C:\isaac-sim\python.bat train_navigation.py --headless --stage 3
```

## Training Progression

**Stage 1** (Random Walking):
- Goal: Prove policy doesn't destabilize SpotFlatTerrainPolicy
- Success: Robot walks for 60s without falling
- Expected: ~100-500 iterations

**Stage 2-5** (Waypoint Navigation):
- Progressively larger waypoint spacing
- Uses progress shaping reward to guide learning
- Each stage: ~500-2000 iterations

**Stage 6-8** (Obstacle Avoidance):
- Removes progress shaping (must learn from captures only)
- Adds obstacles of increasing complexity
- Each stage: ~1000-3000 iterations

**Total expected training time**: ~20-40 hours on H100

## Monitoring Training

Watch the log file:
```bash
Get-Content "checkpoints/navigation_policy/training_log.txt" -Wait
```

Key metrics to watch:
- **Success rate**: Should approach 80% before advancing stages
- **Mean score**: Should increase over time (max 300 + 25*15 = 675 for perfect run)
- **Policy loss**: Should stabilize (not diverge)
- **Entropy**: Should gradually decrease (policy becoming more confident)

## Exporting Trained Policy

The NavigationPolicy is a pure PyTorch MLP and can be exported:

### TorchScript
```python
policy = NavigationPolicy(obs_dim=32, action_dim=3)
policy.load_state_dict(torch.load("checkpoints/navigation_policy/final_model.pt")['policy_state_dict'])
scripted = torch.jit.script(policy)
scripted.save("spot_navigation_policy.pt")
```

### ONNX
```python
dummy_obs = torch.randn(1, 32)
torch.onnx.export(policy, dummy_obs, "spot_navigation_policy.onnx")
```

## Deployment

To use in another simulation or on real Spot:

1. **Replicate observation construction**:
   - Get robot velocity, heading, waypoint position
   - Raycast for 16 obstacle distances
   - Encode current stage

2. **Run policy**:
   ```python
   obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
   action, _, _ = policy.get_action(obs_tensor, deterministic=True)
   ```

3. **Scale action to velocity command**:
   ```python
   from navigation_policy import scale_action
   command = scale_action(action.numpy()[0], vx_range, vy_range, omega_range)
   ```

4. **Send to low-level controller**:
   ```python
   spot.forward(dt, command)  # For SpotFlatTerrainPolicy
   # OR
   spot.set_velocity(command)  # For real Spot API
   ```

## Configuration

Edit `nav_config.yaml` to customize:
- Arena size
- Waypoint count and spacing
- Obstacle densities
- Reward weights
- PPO hyperparameters
- Network architecture

## Troubleshooting

**Robot falls immediately in Stage 1**:
- Reduce learning rate: `ppo.learning_rate: 1.0e-4`
- Increase exploration: `network.action_std: 0.5`
- Check robot starts on ground with correct height

**Stuck on a stage**:
- Lower success threshold: `curriculum.success_threshold: 0.70`
- Increase success window: `curriculum.success_window: 150`
- Adjust reward weights in `nav_config.yaml`

**Training too slow**:
- Reduce `training.steps_per_iteration: 1000`
- Use `--headless` flag
- Check GPU utilization

**Policy not improving**:
- Check entropy (should be > 0.1 for exploration)
- Verify observations are normalized properly
- Try different random seed

## Notes

- **Observation normalization**: Distances are normalized to `[0, 1]`, velocities kept in physical units
- **Progress shaping**: Only active in Stages 2-5, removed for obstacle stages to force pure waypoint-seeking
- **Deterministic evaluation**: Use `deterministic=True` in `get_action()` for deployment
- **Curriculum automation**: System automatically advances stages when success threshold met

## Future Improvements

1. **Parallel environments**: Train with multiple Spot instances simultaneously
2. **Domain randomization**: Vary robot parameters, friction, etc.
3. **Recurrent policy**: Add LSTM for partial observability
4. **Hierarchical goals**: Multi-level waypoint planning
5. **Real-world transfer**: Add noise, latency, imperfect observations
