# Isaac Sim Cleanup — Automatic Zombie Process Killer

When Isaac Sim or Isaac Lab training crashes or exits abruptly, it often leaves behind orphaned `kit/python` and `omni.isaac` subprocesses that hold GPU memory. These block the next training run until the H100 is physically rebooted.

`isaac_cleanup.sh` solves this by registering a shell trap that fires on any exit — normal finish, crash, or Ctrl+C — and kills all related processes automatically.

---

## How to Add It to Your Training Script

Two lines. Source the script, then replace your `python ...` call with `launch_training python ...`.

```bash
#!/usr/bin/env bash
set -e

# 1. Source the cleanup script (adjust relative path to match your script's location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../Colby/ExitKillScript/isaac_cleanup.sh"

# 2. Use launch_training instead of calling python directly
launch_training python train.py \
    --headless \
    --num_envs 2048 \
    --max_iterations 30000
```

That's all. The cleanup runs automatically — you don't call it yourself.

---

## Adjusting the Relative Path

The `source` path depends on where your script lives relative to this folder.

| Your script location | Source path |
|---|---|
| `Experiments/Colby/CombinedPolicyTraining/` | `../../Colby/ExitKillScript/isaac_cleanup.sh` |
| `Experiments/Alex/NAV_ALEX/` | `../../Colby/ExitKillScript/isaac_cleanup.sh` |
| `Experiments/Cole/RL_Folder_VS2/` | `../../Colby/ExitKillScript/isaac_cleanup.sh` |
| `Experiments/Ryan/` | `../Colby/ExitKillScript/isaac_cleanup.sh` |

Or just use an absolute path anchored to the repo root:

```bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # adjust depth
source "$REPO_ROOT/Experiments/Colby/ExitKillScript/isaac_cleanup.sh"
```

---

## What It Cleans Up

| Target | How |
|---|---|
| The training process itself | `kill -- -$PID` (kills entire process group) |
| `kit/python` subprocesses | `pkill -u $(whoami) -f "kit/python"` |
| `omni.isaac` worker threads | `pkill -u $(whoami) -f "omni.isaac"` |
| `omni.kit` subprocesses | `pkill -u $(whoami) -f "omni.kit"` |
| Anything still alive after 2s | `pkill -9` (force kill) |

After cleanup it runs `nvidia-smi` and prints whether the GPU is clear or if any processes are still holding memory.

---

## What Triggers Cleanup

The trap fires on:
- **Normal exit** — training finishes successfully
- **Crash** — Python throws an unhandled exception
- **Ctrl+C** — you interrupt the run manually
- **SIGTERM** — the process is killed externally (e.g. `kill <pid>`)

---

## Important Notes

- **Do not call `simulation_app.close()`** in your Python script. It causes a CUDA deadlock (D-state kernel zombie). Use `os._exit(0)` instead — the shell trap handles process cleanup from the outside.
- The cleanup only kills processes owned by the **current user** (`$(whoami)`), so it won't interfere with teammates' runs on a shared machine.
- `launch_training` uses `setsid` to put Python in its own process group, which is what makes `kill -- -$PID` (group kill) work reliably.
