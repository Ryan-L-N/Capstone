# RL_FOLDER_VS3 - Three Learning Rate Training Variants

## Overview

This folder contains a parallel training setup with three different learning rates to evaluate learning dynamics:

| Variant | Learning Rate | Characteristics |
|---------|---------------|-----------------|
| **Conservative** | 5.0e-5 | Stable, slow convergence, low risk of instability |
| **Moderate** | 1.0e-4 | Balanced, medium convergence speed |
| **Aggressive** | 3.0e-4 | Fast convergence, high risk of policy collapse |

## Folder Structure

```
RL_FOLDER_VS3/
├── README.md                        # This file
├── conservative/
│   ├── nav_config.yaml             # Config file (LR=5.0e-5, checkpoint_freq=25)
│   └── checkpoints/                # Populated during training (stage_X_complete.pt)
├── moderate/
│   ├── nav_config.yaml             # Config file (LR=1.0e-4, checkpoint_freq=25)
│   └── checkpoints/                # Populated during training (stage_X_complete.pt)
└── aggressive/
    ├── nav_config.yaml             # Config file (LR=3.0e-4, checkpoint_freq=25)
    └── checkpoints/                # Populated during training (stage_X_complete.pt)
```

## Training Configuration

All three variants use:
- **Starting Stage**: Stage 1 "Waypoints 10m without obstacles" (0% obstacles)
- **Max Iterations**: 500 (or until success threshold reached after 50 iterations minimum)
- **Success Threshold**: 80% success rate to advance to next stage
- **Checkpoint Frequency**: Every 25 iterations
- **Stage Completion Format**: `stage_X_complete.pt` (e.g., stage_1_complete.pt)

### Curriculum Stages

1. **Stage 1**: "Waypoints 10m without obstacles" (0% obstacles)
2. **Stage 2**: "Waypoints 10m with 1% coverage" (1% light + 1% heavy + 1% small obstacles)
3. **Stage 3**: "Waypoints 20m with 2% coverage" (2% light + 2% heavy + 2% small obstacles)
4. **Stage 4**: "Waypoints 20m then 40m 3% coverage" (3% light + 3% heavy + 3% small obstacles)
5. **Stage 5**: "Waypoints 20m then 40m 4% total obstacles ground coverage" (4% light + 4% heavy + 4% small obstacles)
6. **Stage 6**: "Waypoints 20m then 40m 5% total obstacles ground coverage" (5% light + 5% heavy + 5% small obstacles, final)

## How to Launch Training

Run each variant from the RL_FOLDER_VS3 root directory using `cmd /c`:

Conservative:
```powershell
cmd /c "cd C:\Users\user\Desktop\Capstone_vs_1.2\Immersive-Modeling-and-Simulation-for-Autonomy\Experiments\Cole\RL_FOLDER_VS3 && C:\isaac-sim\python.bat train_navigation.py --config conservative/nav_config.yaml --checkpoint-dir conservative/checkpoints --headless"
```

Moderate:
```powershell
cmd /c "cd C:\Users\user\Desktop\Capstone_vs_1.2\Immersive-Modeling-and-Simulation-for-Autonomy\Experiments\Cole\RL_FOLDER_VS3 && C:\isaac-sim\python.bat train_navigation.py --config moderate/nav_config.yaml --checkpoint-dir moderate/checkpoints --headless"
```

Aggressive:
```powershell
cmd /c "cd C:\Users\user\Desktop\Capstone_vs_1.2\Immersive-Modeling-and-Simulation-for-Autonomy\Experiments\Cole\RL_FOLDER_VS3 && C:\isaac-sim\python.bat train_navigation.py --config aggressive/nav_config.yaml --checkpoint-dir aggressive/checkpoints --headless"
```

To resume from a checkpoint at a specific stage (e.g., Stage 2 with aggressive):
```powershell
cmd /c "cd C:\Users\user\Desktop\Capstone_vs_1.2\Immersive-Modeling-and-Simulation-for-Autonomy\Experiments\Cole\RL_FOLDER_VS3 && C:\isaac-sim\python.bat train_navigation.py --config aggressive/nav_config.yaml --checkpoint-dir aggressive/checkpoints --checkpoint aggressive/checkpoints/stage_1_complete.pt --stage 2 --headless"
```

## Expected Behavior

### Conservative (5.0e-5 LR)
- **Pros**: Most stable, less risk of policy collapse
- **Cons**: Slower convergence, may not reach final stages within 500 iterations
- **Expected**: Good Stage 1-2 performance, potential early plateau

### Moderate (1.0e-4 LR)
- **Pros**: Balanced learning, likely to reach Stage 3-4
- **Cons**: Moderate risk of instability
- **Expected**: Steady progression through stages

### Aggressive (3.0e-4 LR)
- **Pros**: Fastest convergence potential, reaches later stages
- **Cons**: High risk of policy collapse or divergence
- **Expected**: Rapid early progress, possible instability at later stages

## Stage Naming Convention

When requesting to start a specific stage, use the following naming:
- **Stage 1** = "Waypoints 10m without obstacles"
- **Stage 2** = "Waypoints 10m with 1% coverage"
- **Stage 3** = "Waypoints 20m with 2% coverage"
- **Stage 4** = "Waypoints 20m then 40m 3% coverage"
- **Stage 5** = "Waypoints 20m then 40m 4% total obstacles ground coverage"
- **Stage 6** = "Waypoints 20m then 40m 5% total obstacles ground coverage"

Example: To start training from Stage 2, use `--stage 2`

## Checkpoints

Checkpoints are saved in two ways:

1. **Regular Checkpoints**: Every 25 iterations (from `checkpoint_frequency`)
2. **Stage Completion Checkpoints**: Automatically saved as `stage_X_complete.pt` when stage completes

## Comparing Results

After all three finish:

1. Compare completion stages (how far each variant progressed)
2. Compare success rates within each stage
3. Analyze convergence speed (iterations/stage)
4. Evaluate stability (training logs for divergence indicators)

## Troubleshooting

### Training Hangs on Startup
- Check Isaac Sim installation: `C:\isaac-sim\python.bat --version`
- Verify config file exists: `.\nav_config.yaml`
- Check GPU availability and VRAM

### Out of Memory Issues
- Reduce `steps_per_iteration` in config (currently 6000)
- Reduce network `hidden_dims` (currently [256, 256, 128])
- Run fewer variants in parallel (e.g., conservative + moderate first)

### Config Files Not Found
- Ensure you're in the correct directory before running scripts
- Verify config files exist in each folder: 
  - `conservative/nav_config.yaml`
  - `moderate/nav_config.yaml`
  - `aggressive/nav_config.yaml`

## Next Steps

1. **Launch Training**: Run each variant using the `cmd /c` commands above
2. **Monitor Progress**: Check logs every 5-10 minutes
3. **Stop if Needed**: Close the terminal running the variant
4. **Analyze Results**: Compare checkpoints and success rates after completion
5. **Select Best Variant**: Choose the variant with best convergence for deployment

---

**Last Updated**: [Date]
**Status**: Ready for training
