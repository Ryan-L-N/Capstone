# Policy Checkpoints

Place policy checkpoint files in this directory before running evaluations.

## Required Checkpoints

### 1. Custom Rough Terrain Policy — `model_29999.pt`

- **Size:** 6.6 MB
- **Source:** 48h H100 training run (30,000 iterations, 8,192 parallel envs)
- **Observation space:** 235 dims (48 proprioceptive + 187 height scan)
- **Network:** MLP [512, 256, 128] with ELU activation
- **PD gains:** Kp=60, Kd=1.5
- **Action scale:** 0.25

**Location on H100 server:**
```bash
~/IsaacLab/logs/rsl_rl/spot_rough/*/model_29999.pt
```

**Copy to this directory:**
```bash
scp t2user@ai2ct2:~/IsaacLab/logs/rsl_rl/spot_rough/*/model_29999.pt ./checkpoints/
```

### 2. NVIDIA Flat Terrain Baseline

- **Source:** Isaac Lab pre-trained model (bundled with Isaac Lab 2.3.0)
- **Observation space:** 48 dims (proprioceptive only, no height scan)
- **Network:** MLP [512, 256, 128] with ELU activation

**Location in Isaac Lab installation:**
```bash
# The flat policy is loaded via Isaac Lab's SpotFlatTerrainPolicy class
# No separate checkpoint file needed — it's built into the Isaac Lab package
```

If using the RSL-RL flat checkpoint directly:
```bash
~/IsaacLab/logs/rsl_rl/spot_flat/*/model_*.pt
```

## Verification

After placing checkpoints, verify:
```bash
ls -la checkpoints/*.pt
# Should show model_29999.pt (~6.6 MB)

python -c "import torch; m = torch.load('checkpoints/model_29999.pt', map_location='cpu'); print(f'Keys: {list(m.keys())}')"
```

## Notes

- Checkpoint files (*.pt) are gitignored — they must be manually placed
- Do NOT commit checkpoint files to git (they are large binary files)
- The flat baseline may not need a separate .pt file if using SpotFlatTerrainPolicy directly
