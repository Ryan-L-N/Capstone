"""CLI tool for the live training control panel.

Run in a second SSH terminal to send commands to a running training session.

Usage:
    # Auto-detect latest run directory
    python -m control_panel.cli status
    python -m control_panel.cli set action_smoothness -0.5
    python -m control_panel.cli set foot_clearance 2.5 joint_pos -0.3
    python -m control_panel.cli lr --max 5e-4
    python -m control_panel.cli noise --max 0.4
    python -m control_panel.cli s2r max_dropout_rate 0.08
    python -m control_panel.cli save --label "pre_experiment"
    python -m control_panel.cli stop --reason "diverging"
    python -m control_panel.cli freeze foot_slip
    python -m control_panel.cli unfreeze foot_slip
    python -m control_panel.cli history --last 20

    # Explicit run directory
    python -m control_panel.cli --run /path/to/run/dir status

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime

try:
    import yaml
    _USE_YAML = True
except ImportError:
    _USE_YAML = False


def _find_latest_run() -> str:
    """Auto-detect the most recent training run directory."""
    base = os.path.join(os.path.dirname(__file__), "..", "logs", "rsl_rl")
    base = os.path.abspath(base)
    candidates = []
    for exp_dir in glob.glob(os.path.join(base, "*")):
        ctrl = os.path.join(exp_dir, "control.yaml")
        if os.path.exists(ctrl):
            candidates.append((os.path.getmtime(ctrl), exp_dir))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _read_control(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = f.read()
    if _USE_YAML:
        return yaml.safe_load(raw) or {}
    return json.loads(raw) if raw.strip() else {}


def _write_control(path: str, data: dict):
    tmp = path + ".tmp"
    if _USE_YAML:
        content = yaml.dump(data, default_flow_style=False, sort_keys=False)
    else:
        content = json.dumps(data, indent=2)
    with open(tmp, "w") as f:
        f.write(content)
    os.replace(tmp, path)


def _add_command(ctrl_path: str, command: dict):
    """Add a command to pending_commands in the control file."""
    command["timestamp"] = datetime.now().isoformat(timespec="seconds")
    data = _read_control(ctrl_path)
    if "pending_commands" not in data:
        data["pending_commands"] = []
    data["pending_commands"].append(command)
    _write_control(ctrl_path, data)
    print(f"Command queued: {command['type']}")
    if "name" in command:
        print(f"  {command['name']} = {command.get('value', '')}")


def cmd_status(args, ctrl_path):
    data = _read_control(ctrl_path)
    if not data:
        print("No control.yaml found or empty.")
        return

    s = data.get("status", {})
    print(f"\n{'='*60}")
    print(f"  LIVE TRAINING STATUS")
    print(f"{'='*60}")
    print(f"  Iteration:     {s.get('iteration', '?')}")
    print(f"  Terrain Level: {s.get('terrain_level', '?')}")
    print(f"  Learning Rate: {s.get('lr', '?')}")
    print(f"  Noise Std:     {s.get('noise_std', '?')}")
    print(f"  S2R Scale:     {s.get('s2r_scale', '?')}")
    print(f"  Last Poll:     {s.get('last_poll', '?')}")

    lr_b = data.get("lr_bounds", {})
    noise_b = data.get("noise_bounds", {})
    print(f"\n  LR Bounds:     [{lr_b.get('min', '?')}, {lr_b.get('max', '?')}]")
    print(f"  Noise Bounds:  [{noise_b.get('min', '?')}, {noise_b.get('max', '?')}]")

    weights = data.get("current_weights", {})
    frozen = set(data.get("frozen_weights", []))
    if weights:
        print(f"\n  {'REWARD WEIGHTS':^56}")
        print(f"  {'-'*56}")
        # Task rewards
        print(f"  {'Task Rewards:':<30}")
        for name in ["air_time", "base_angular_velocity", "base_linear_velocity",
                      "foot_clearance", "gait"]:
            if name in weights:
                lock = " [FROZEN]" if name in frozen else ""
                print(f"    {name:<28} {weights[name]:>8.4f}{lock}")
        # Penalties
        print(f"  {'Penalties:':<30}")
        for name in ["action_smoothness", "air_time_variance", "base_motion",
                      "base_orientation", "base_pitch", "base_roll",
                      "foot_slip", "joint_acc", "joint_pos",
                      "joint_torques", "joint_vel", "terrain_relative_height",
                      "dof_pos_limits", "undesired_contacts", "motor_power",
                      "torque_limit"]:
            if name in weights:
                lock = " [FROZEN]" if name in frozen else ""
                print(f"    {name:<28} {weights[name]:>8.4f}{lock}")
        # Frozen
        for name in ["stumble", "body_height_tracking"]:
            if name in weights:
                print(f"    {name:<28} {weights[name]:>8.4f} [HARD-FROZEN]")
    print(f"{'='*60}\n")


def cmd_set(args, ctrl_path):
    pairs = args.pairs
    if len(pairs) % 2 != 0:
        print("ERROR: set requires name-value pairs (e.g., set foot_clearance 2.5)")
        return
    for i in range(0, len(pairs), 2):
        name = pairs[i]
        try:
            value = float(pairs[i + 1])
        except ValueError:
            print(f"ERROR: invalid value '{pairs[i + 1]}' for {name}")
            continue
        _add_command(ctrl_path, {
            "type": "set_weight",
            "name": name,
            "value": value,
            "force": args.force,
        })


def cmd_lr(args, ctrl_path):
    cmd = {"type": "set_lr_bounds"}
    if args.max is not None:
        cmd["lr_max"] = args.max
    if args.min is not None:
        cmd["lr_min"] = args.min
    if len(cmd) == 1:
        print("ERROR: specify --max and/or --min")
        return
    _add_command(ctrl_path, cmd)


def cmd_noise(args, ctrl_path):
    cmd = {"type": "set_noise_bounds"}
    if args.max is not None:
        cmd["max_std"] = args.max
    if args.min is not None:
        cmd["min_std"] = args.min
    if len(cmd) == 1:
        print("ERROR: specify --max and/or --min")
        return
    _add_command(ctrl_path, cmd)


def cmd_s2r(args, ctrl_path):
    _add_command(ctrl_path, {
        "type": "set_s2r_param",
        "param": args.param,
        "value": float(args.value),
    })


def cmd_save(args, ctrl_path):
    _add_command(ctrl_path, {
        "type": "save_checkpoint",
        "label": args.label or "manual",
    })


def cmd_stop(args, ctrl_path):
    _add_command(ctrl_path, {
        "type": "emergency_stop",
        "reason": args.reason or "manual stop via CLI",
    })


def cmd_freeze(args, ctrl_path):
    _add_command(ctrl_path, {"type": "freeze_weight", "name": args.name})


def cmd_unfreeze(args, ctrl_path):
    _add_command(ctrl_path, {"type": "unfreeze_weight", "name": args.name})


def cmd_history(args, ctrl_path):
    log_path = os.path.join(os.path.dirname(ctrl_path), "control_panel_changes.jsonl")
    if not os.path.exists(log_path):
        print("No change history yet.")
        return
    with open(log_path) as f:
        lines = f.readlines()
    n = args.last or 20
    print(f"\nLast {min(n, len(lines))} changes:")
    print("-" * 70)
    for line in lines[-n:]:
        try:
            entry = json.loads(line.strip())
            ts = entry.get("timestamp", "?")
            it = entry.get("iteration", "?")
            ct = entry.get("command_type", "?")
            applied = entry.get("applied", {})
            print(f"  {ts}  iter {it:>6}  {ct:<20}  {json.dumps(applied)[:60]}")
        except json.JSONDecodeError:
            continue
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Live Training Control Panel CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status                              Show current training state
  %(prog)s set action_smoothness -0.5          Change one weight
  %(prog)s set foot_clearance 2.5 joint_pos -0.3   Change multiple weights
  %(prog)s set joint_pos -0.1 --force          Override 50%% delta limit
  %(prog)s lr --max 5e-4                       Change LR schedule
  %(prog)s noise --max 0.4 --min 0.25          Change noise bounds
  %(prog)s s2r max_dropout_rate 0.08           Change S2R param
  %(prog)s save --label "pre_experiment"       Save checkpoint now
  %(prog)s stop --reason "diverging"           Emergency stop
  %(prog)s freeze foot_slip                    Lock a weight
  %(prog)s history --last 30                   View change history
        """,
    )
    parser.add_argument("--run", type=str, default=None,
                        help="Run directory (auto-detect if omitted)")

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show current training state")

    p_set = sub.add_parser("set", help="Change reward weight(s)")
    p_set.add_argument("pairs", nargs="+", help="name value [name value ...]")
    p_set.add_argument("--force", action="store_true", help="Override 50%% delta guard")

    p_lr = sub.add_parser("lr", help="Change LR schedule bounds")
    p_lr.add_argument("--max", type=float, default=None)
    p_lr.add_argument("--min", type=float, default=None)

    p_noise = sub.add_parser("noise", help="Change noise bounds")
    p_noise.add_argument("--max", type=float, default=None)
    p_noise.add_argument("--min", type=float, default=None)

    p_s2r = sub.add_parser("s2r", help="Change S2R wrapper param")
    p_s2r.add_argument("param", help="S2R param name")
    p_s2r.add_argument("value", help="New value")

    p_save = sub.add_parser("save", help="Save checkpoint immediately")
    p_save.add_argument("--label", type=str, default=None)

    p_stop = sub.add_parser("stop", help="Emergency stop training")
    p_stop.add_argument("--reason", type=str, default=None)

    p_freeze = sub.add_parser("freeze", help="Freeze a weight")
    p_freeze.add_argument("name", help="Weight name to freeze")

    p_unfreeze = sub.add_parser("unfreeze", help="Unfreeze a weight")
    p_unfreeze.add_argument("name", help="Weight name to unfreeze")

    p_hist = sub.add_parser("history", help="View change history")
    p_hist.add_argument("--last", type=int, default=20)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    # Find run directory
    run_dir = args.run
    if not run_dir:
        run_dir = _find_latest_run()
    if not run_dir:
        print("ERROR: No run directory found. Use --run /path/to/run/dir")
        sys.exit(1)

    ctrl_path = os.path.join(run_dir, "control.yaml")
    if not os.path.exists(ctrl_path) and args.command != "status":
        print(f"WARNING: {ctrl_path} does not exist yet. Training may not have started.")

    # Dispatch
    dispatch = {
        "status": cmd_status, "set": cmd_set, "lr": cmd_lr,
        "noise": cmd_noise, "s2r": cmd_s2r, "save": cmd_save,
        "stop": cmd_stop, "freeze": cmd_freeze, "unfreeze": cmd_unfreeze,
        "history": cmd_history,
    }
    dispatch[args.command](args, ctrl_path)


if __name__ == "__main__":
    main()
