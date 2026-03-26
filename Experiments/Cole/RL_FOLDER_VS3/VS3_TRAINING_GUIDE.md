# VS3 Training Runs - Checkpoint & Physics Configuration Guide

## Directory Structure

```
RL_FOLDER_VS3/
├── VS3_checkpoints/
│   ├── conservative/        # Run with conservative learning rate (5.0e-5)
│   ├── moderate/           # Run with moderate learning rate (1.0e-4)
│   └── aggressive/         # Run with aggressive learning rate (2.0e-4)
├── nav_config_conservative.yaml
├── nav_config_moderate.yaml
├── nav_config_aggressive.yaml
├── physics_diagnostics.py   # Static physics engine diagnostics
├── physics_monitor.py       # Runtime physics monitoring
├── navigation_env.py
├── navigation_policy.py
├── train_navigation.py
└── ppo_trainer.py
```

## Three Training Runs

### 1. Conservative Run (5.0e-4 learning rate)
- **Purpose**: Slow, stable learning - good for debugging and early curriculum
- **Checkpoint Directory**: `VS3_checkpoints/conservative/`
- **Config File**: `nav_config_conservative.yaml`
- **Learning Rate**: 5.0e-5 (0.5x baseline)
- **Expected Behavior**: Slower convergence, higher stability, smoother training curves

**Launch Command:**
```powershell
cd Experiments\Cole\RL_FOLDER_VS3
$env:KMP_DUPLICATE_LIB_OK="TRUE"
C:\isaac-sim\python.bat train_navigation.py `
  --headless --iterations 15000 `
  --config nav_config_conservative.yaml `
  --checkpoint-dir VS3_checkpoints/conservative `
  --stop-at-stage-complete --stage 0
```

### 2. Moderate Run (1.0e-4 learning rate)
- **Purpose**: Balanced learning - baseline approach
- **Checkpoint Directory**: `VS3_checkpoints/moderate/`
- **Config File**: `nav_config_moderate.yaml`
- **Learning Rate**: 1.0e-4 (1.0x baseline)
- **Expected Behavior**: Standard convergence, good stability-speed tradeoff

**Launch Command:**
```powershell
cd Experiments\Cole\RL_FOLDER_VS3
$env:KMP_DUPLICATE_LIB_OK="TRUE"
C:\isaac-sim\python.bat train_navigation.py `
  --headless --iterations 15000 `
  --config nav_config_moderate.yaml `
  --checkpoint-dir VS3_checkpoints/moderate `
  --stop-at-stage-complete --stage 0
```

### 3. Aggressive Run (2.0e-4 learning rate)
- **Purpose**: Fast learning - good for later curriculum stages
- **Checkpoint Directory**: `VS3_checkpoints/aggressive/`
- **Config File**: `nav_config_aggressive.yaml`
- **Learning Rate**: 2.0e-4 (2.0x baseline)
- **Expected Behavior**: Faster convergence, higher variance, potential instability in early stages

**Launch Command:**
```powershell
cd Experiments\Cole\RL_FOLDER_VS3
$env:KMP_DUPLICATE_LIB_OK="TRUE"
C:\isaac-sim\python.bat train_navigation.py `
  --headless --iterations 15000 `
  --config nav_config_aggressive.yaml `
  --checkpoint-dir VS3_checkpoints/aggressive `
  --stop-at-stage-complete --stage 0
```

## Physics Engine Diagnostics

### Static Physics Diagnostics

Query Isaac Sim's physics engine settings without running training:

```powershell
C:\isaac-sim\python.bat physics_diagnostics.py --diagnostic
```

**Available Options:**

| Command | Purpose |
|---------|---------|
| `--diagnostic` | Full physics engine report |
| `--gravity` | Current gravity vector and magnitude |
| `--contact` | Contact offset, rest offset, depenetration settings |
| `--timestep` | Simulation frequency and timestep (dt) |
| `--constraints` | Constraint solver parameters |
| `--adjust-contact-offset <value>` | Set contact offset (meters) |
| `--adjust-rest-offset <value>` | Set rest offset (meters) |
| `--adjust-depenetration <value>` | Set max depenetration velocity (m/s) |

**Example: Check current physics settings**
```powershell
C:\isaac-sim\python.bat physics_diagnostics.py --all
```

**Example: Adjust contact properties**
```powershell
C:\isaac-sim\python.bat physics_diagnostics.py `
  --adjust-contact-offset 0.001 `
  --adjust-rest-offset 0.0005 `
  --adjust-depenetration 10.0
```

### Runtime Physics Monitoring

Monitor physics properties during training (integrated approach):

```powershell
C:\isaac-sim\python.bat physics_monitor.py --all
```

**Available Options:**

| Command | Purpose |
|---------|---------|
| `--joints` | Query Spot's joint properties, limits, and current states |
| `--contacts` | Get contact forces and foot contact states |
| `--constraints` | Show contact constraint parameters |
| `--accuracy` | Force estimation accuracy and limitations |
| `--all` | All information (default) |

**Example: Check robot joint configuration**
```powershell
C:\isaac-sim\python.bat physics_monitor.py --joints
```

**Example: Check contact properties**
```powershell
C:\isaac-sim\python.bat physics_monitor.py --contacts
```

## Key Physics Parameters

### Current Configuration

**Gravity**: 9.81 m/s² (Earth standard)

**Contact Settings**:
- Contact Offset: ~0.001 m (typically)
- Rest Offset: ~0.0005 m
- Max Depenetration Velocity: ~10.0 m/s

**Timestep**:
- Simulation Frequency: 500 Hz
- Timestep (dt): 0.002 seconds (2ms)
- Policy Control Frequency: 20 Hz

**Joint Limits** (Spot quadruped):
- Hip AA (abduction/adduction): ±0.5 rad
- Hip FE (flexion/extension): ±2.0 rad
- Knee (flexion): ~0 to 2.5 rad
- Effort Limits: ~100-150 N⋅m per joint

### Adjustable Constraints

**Contact-Related**:
- `contact_offset`: Tune for stability vs. penetration sensitivity
  - Decrease to improve precision (risk: numerical instability)
  - Increase for more stable but less precise contacts
- `rest_offset`: Fine threshold for contact activation
- `max_depenetration_velocity`: Max speed to separate overlapping objects

**Solver-Related** (PhysX):
- Substeps: Increase for accuracy (slower simulation)
- Solver iterations: More iterations = better constraint satisfaction

**Gravity**:
- Can adjust for different planet simulations
- Default: 9.81 m/s² (Earth)

## Tuning Recommendations

### For Pushing (Stage 3) Improvements

1. **Increase Contact Precision**: Lower `contact_offset` to 0.0005-0.0008m
2. **Reduce Joint Flexibility**: Increase joint damping in URDF
3. **Improve Contact Forces**: Ensure `rest_offset` is properly tuned
4. **Monitor Force Stability**: Use physics_monitor.py to watch contact data

### For Overall Stability Improvements

1. **Check Timestep**: Ensure 500Hz is sufficient (monitor energy conservation)
2. **Verify Gravity**: Confirm 9.81 m/s² is active
3. **Test Substeps**: Increase to 2-4 for sensitive terrain interactions
4. **Monitor Joint Torques**: Watch for saturation in early stages

### For Learning Rate Selection

- **Conservative (5.0e-5)**: Use for debugging Stage 1-2, then switch
- **Moderate (1.0e-4)**: Default; good across all stages
- **Aggressive (2.0e-4)**: Use after Stage 2 for faster convergence

## Expected Training Flow

### Recommended Sequence

1. **Launch Conservative Run** (1-2 hours):
   - Stage 1: 15-30 min to 80% success
   - Stage 2: 20-40 min to 80% success
   - Test policy after Stage 2

2. **Launch Moderate Run** (3-4 hours):
   - Stages 3-4: Object pushing and short navigation
   - Monitor pushing success rate closely
   - Adjust contact parameters if needed

3. **Launch Aggressive Run** (2-3 hours):
   - Stages 5-7: Long-range navigation
   - Faster convergence expected
   - Monitor for training instability

## Performance Monitoring

### Key Metrics to Watch

```
Training Log Analysis:
├── Success Rate (should reach 80%)
├── Mean Waypoints Captured (increases with distance in later stages)
├── Mean Reward (should increase over iterations)
├── Policy Loss (should decrease)
└── Contact Forces (should stabilize in Stage 3)
```

### Data Collection for Analysis

Checkpoints include:
- `best_model.pt`: Highest success rate model
- `stage_X_complete.pt`: Model at stage transition
- `training_log.txt`: Complete iteration-by-iteration metrics

## Troubleshooting Physics Issues

### Problem: Robot sinking through terrain
**Solution**: Lower contact_offset or increase substeps

### Problem: Jerky/unstable movements
**Solution**: Check timestep (should be 0.002s), verify gravity setting

### Problem: Pushing not detected in Stage 3
**Solution**: Run `physics_monitor.py --accuracy` to check force estimation

### Problem: Learning rate too slow/volatile
**Solution**: Adjust learning_rate in config (5.0e-5 to 2.0e-4)

## Files Reference

| File | Purpose |
|------|---------|
| `nav_config_conservative.yaml` | 5.0e-5 learning rate config |
| `nav_config_moderate.yaml` | 1.0e-4 learning rate config |
| `nav_config_aggressive.yaml` | 2.0e-4 learning rate config |
| `physics_diagnostics.py` | Static physics engine query tool |
| `physics_monitor.py` | Runtime physics monitoring tool |
| `train_navigation.py` | Main training script with testing mode |
| `navigation_env.py` | Environment with force inference |
| `navigation_policy.py` | PPO policy network |
| `ppo_trainer.py` | PPO algorithm implementation |

## Next Steps

1. **Run physics diagnostics** to verify current settings:
   ```powershell
   C:\isaac-sim\python.bat physics_diagnostics.py --all
   ```

2. **Choose starting run** (recommended: conservative first):
   ```powershell
   C:\isaac-sim\python.bat train_navigation.py --headless --iterations 15000 --config nav_config_conservative.yaml --checkpoint-dir VS3_checkpoints/conservative --stop-at-stage-complete --stage 0
   ```

3. **Monitor checkpoint directories** for saving intermediate models:
   - `VS3_checkpoints/conservative/stage_1_complete.pt`
   - `VS3_checkpoints/conservative/stage_2_complete.pt`
   - etc.

4. **Review logs** after each stage completes for metrics analysis
