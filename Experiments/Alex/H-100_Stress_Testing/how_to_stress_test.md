# Running Isaac Lab Training in Detached Sessions

**Quick guide for running long training sessions that survive SSH disconnects**

---

## Method 1: Using Screen (Recommended)

Screen is simple and lets you reattach to see live output.

### Start a Training Session

```bash
# SSH into the server
ssh t2user@172.24.254.24

# Activate the environment
conda activate env_isaaclab

# Create a named screen session
screen -S my_training

# Inside screen, run your training
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Velocity-Rough-Anymal-C-v0 \
    --headless \
    --num_envs 4096
```

### Detach from Screen

**Press: `Ctrl+A`, then `D`**

You can now safely disconnect from SSH - the training continues running.

### Reconnect Later

```bash
# SSH back in
ssh t2user@172.24.254.24

# List all screen sessions
screen -ls

# Reattach to your training session
screen -r my_training
```

### Common Screen Commands

| Command | Action |
|---------|--------|
| `screen -S name` | Create new named session |
| `screen -ls` | List all sessions |
| `screen -r name` | Reattach to session |
| `Ctrl+A, D` | Detach (leave running) |
| `Ctrl+A, K` | Kill session |
| `exit` | Exit session (stops training!) |

---

## Method 2: Using nohup (Fire and Forget)

Good for scripts you don't need to monitor interactively.

### Start Training

```bash
# SSH into the server
ssh t2user@172.24.254.24

# Activate environment
conda activate env_isaaclab
cd ~/IsaacLab

# Run with nohup (output goes to log file)
nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Velocity-Rough-Anymal-C-v0 \
    --headless \
    --num_envs 4096 \
    > training_output.log 2>&1 &
```

The `&` at the end runs it in background. You can disconnect immediately.

### Monitor Progress

```bash
# View the log file (updates in real-time)
tail -f training_output.log

# Press Ctrl+C to stop viewing (doesn't stop training)
```

### Check if Training is Still Running

```bash
# See GPU usage
nvidia-smi

# Find the training process
ps aux | grep train.py
```

---

## Method 3: Running Custom Scripts

### For Your Custom Scripts (like stress_test.sh)

**With screen:**
```bash
screen -S stress_test
bash ~/stress_test.sh --headless
# Ctrl+A, D to detach
```

**With nohup:**
```bash
nohup bash ~/stress_test.sh --headless > stress_test_output.log 2>&1 &
tail -f stress_test_output.log
```

---

## Monitoring GPU During Training

### One-time Check
```bash
nvidia-smi
```

### Live Monitoring (updates every second)
```bash
watch -n 1 nvidia-smi
# Press Ctrl+C to exit
```

### In a Separate Screen Window
```bash
# While attached to a screen session:
# Ctrl+A, C - creates new window
# Ctrl+A, N - next window
# Ctrl+A, P - previous window

# This lets you monitor nvidia-smi while training runs in another window
```

---

## Best Practices

### ✅ DO:
- Use `screen` for interactive sessions you want to check on
- Use `nohup` for scripts that log everything to a file
- Name your screen sessions descriptively (`-S quadruped_training`)
- Redirect output to log files (`> training.log 2>&1`)
- Check GPU usage with `nvidia-smi` to verify training started

### ❌ DON'T:
- Open multiple SSH sessions simultaneously (causes server freeze)
- Run training without `--headless` flag
- Forget to activate `conda activate env_isaaclab` first
- Use `exit` to leave screen (this kills the session - use Ctrl+A, D instead)

---

## Quick Reference Card

```bash
# === START TRAINING ===
ssh t2user@172.24.254.24
conda activate env_isaaclab
screen -S training
cd ~/IsaacLab
./isaaclab.sh -p scripts/.../train.py --task=... --headless
# Ctrl+A, D to detach

# === CHECK STATUS ===
screen -ls                    # List sessions
nvidia-smi                    # Check GPU
tail -f ~/IsaacLab/logs/...   # View logs

# === RECONNECT ===
screen -r training            # Reattach
# Ctrl+A, D to detach again

# === STOP TRAINING ===
screen -r training            # Reattach
# Ctrl+C to stop training
exit                          # Close screen session
```

---

## Troubleshooting

**Q: "There is no screen to be resumed matching training"**
- Check `screen -ls` to see actual session names
- Session might have crashed - check logs

**Q: Training stopped unexpectedly**
- Check `dmesg | tail` for out-of-memory errors
- Reduce `--num_envs` if OOM
- Check screen session still exists with `screen -ls`

**Q: Can't SSH back in**
- Wait 5 minutes - server might be busy
- Don't open multiple SSH sessions
- Hard reboot may be needed (physical access required)

**Q: GPU shows 0% utilization**
- Training might still be initializing (first run takes 10+ min)
- Check process with `ps aux | grep python`
- Verify training didn't crash - check logs

---

**Last Updated:** February 20, 2026
