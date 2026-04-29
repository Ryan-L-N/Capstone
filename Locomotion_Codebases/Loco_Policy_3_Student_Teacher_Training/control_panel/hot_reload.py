"""Hot-reload engine — polls a YAML control file and applies changes to the live training.

The training loop calls `poll_and_apply()` once per iteration (~0.3s).
A CLI tool or web dashboard writes commands to the same YAML file.
Atomic file operations (write-to-tmp + rename) prevent race conditions.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

try:
    import yaml
    _USE_YAML = True
except ImportError:
    _USE_YAML = False

from . import guardrails
from .change_log import ChangeLog


def _read_control(path: str) -> dict:
    """Read the control file (YAML or JSON fallback)."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            raw = f.read()
        if _USE_YAML:
            return yaml.safe_load(raw) or {}
        else:
            return json.loads(raw) if raw.strip() else {}
    except Exception:
        return {}


def _write_control(path: str, data: dict):
    """Atomic write: write to .tmp then rename."""
    tmp = path + ".tmp"
    try:
        if _USE_YAML:
            content = yaml.dump(data, default_flow_style=False, sort_keys=False)
        else:
            content = json.dumps(data, indent=2)
        with open(tmp, "w") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[CONTROL] Warning: failed to write {path}: {e}", flush=True)


class HotReloader:
    """Polls a YAML control file and applies parameter changes to a live training run.

    Usage:
        reloader = HotReloader(log_dir, env, runner, s2r_wrapper)
        reloader.write_initial_state(...)

        # Inside update_with_schedule(), every iteration:
        changes = reloader.poll_and_apply(iteration, lr, terrain_level)
    """

    # Stale command threshold (seconds)
    STALE_TIMEOUT = 120

    def __init__(self, log_dir: str, env, runner, s2r_wrapper=None,
                 frozen_weights: set = None,
                 noise_bounds: dict = None,
                 lr_bounds: dict = None):
        """
        Args:
            log_dir: Active run log directory (where control.yaml lives).
            env: Gymnasium env (for reward_manager access via env.unwrapped).
            runner: RSL-RL OnPolicyRunner (for optimizer/checkpoint access).
            s2r_wrapper: Optional ProgressiveS2RWrapper instance.
            frozen_weights: Set of user-frozen weight names.
            noise_bounds: Mutable dict {"min": float, "max": float} shared with clamp.
            lr_bounds: Mutable dict {"max": float, "min": float} for LR schedule.
        """
        self.log_dir = log_dir
        self.env = env
        self.runner = runner
        self.s2r_wrapper = s2r_wrapper

        self.frozen_weights = (frozen_weights or set()) | guardrails.HARD_FROZEN
        self.noise_bounds = noise_bounds or {"min": 0.3, "max": 0.5}
        self.lr_bounds = lr_bounds or {"max": 3e-5, "min": 1e-6}

        self.control_path = os.path.join(log_dir, "control.yaml")
        self.changelog = ChangeLog(log_dir)

        self._last_mtime = 0.0

    # ------------------------------------------------------------------
    # Reward manager helpers
    # ------------------------------------------------------------------

    def _get_reward_manager(self):
        """Navigate wrapper chain to get the IsaacLab reward manager."""
        e = self.env
        while hasattr(e, "env"):
            e = e.env
        if hasattr(e, "unwrapped"):
            e = e.unwrapped
        if hasattr(e, "reward_manager"):
            return e.reward_manager
        return None

    def _get_weight(self, rm, name: str):
        """Read current weight value from reward manager."""
        if hasattr(rm, "_term_names") and hasattr(rm, "_term_cfgs"):
            if name in rm._term_names:
                idx = rm._term_names.index(name)
                return rm._term_cfgs[idx].weight
        return None

    def _set_weight(self, rm, name: str, value: float) -> float | None:
        """Set weight and return old value, or None if not found."""
        if hasattr(rm, "_term_names") and hasattr(rm, "_term_cfgs"):
            if name in rm._term_names:
                idx = rm._term_names.index(name)
                old = rm._term_cfgs[idx].weight
                rm._term_cfgs[idx].weight = value
                return old
        return None

    def _get_all_weights(self, rm) -> dict:
        """Read all reward weights as a dict."""
        weights = {}
        if hasattr(rm, "_term_names") and hasattr(rm, "_term_cfgs"):
            for i, name in enumerate(rm._term_names):
                weights[name] = rm._term_cfgs[i].weight
        return weights

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    def write_initial_state(self, iteration: int, lr: float,
                            noise_std: float, terrain_level: float):
        """Write initial control.yaml with current training state."""
        rm = self._get_reward_manager()
        weights = self._get_all_weights(rm) if rm else {}

        data = {
            "status": {
                "iteration": iteration,
                "lr": lr,
                "noise_std": noise_std,
                "terrain_level": terrain_level,
                "s2r_scale": self.s2r_wrapper.scale if self.s2r_wrapper else 0.0,
                "last_poll": datetime.now().isoformat(timespec="seconds"),
            },
            "current_weights": weights,
            "lr_bounds": dict(self.lr_bounds),
            "noise_bounds": dict(self.noise_bounds),
            "frozen_weights": sorted(self.frozen_weights),
            "pending_commands": [],
        }
        _write_control(self.control_path, data)
        self._last_mtime = os.path.getmtime(self.control_path)
        print(f"[CONTROL] Live control panel ready: {self.control_path}", flush=True)

    # ------------------------------------------------------------------
    # Per-iteration poll
    # ------------------------------------------------------------------

    def poll_and_apply(self, iteration: int, current_lr: float,
                       terrain_level: float) -> list:
        """Check for pending commands and apply them. Returns list of (type, detail) strings."""
        # Quick mtime check — skip read if file hasn't changed
        try:
            mtime = os.path.getmtime(self.control_path)
        except OSError:
            return []
        if mtime <= self._last_mtime:
            # Still update status periodically (every 50 iters)
            if iteration % 50 == 0:
                self._update_status(iteration, current_lr, terrain_level)
            return []

        self._last_mtime = mtime

        data = _read_control(self.control_path)
        if not data:
            return []

        pending = data.get("pending_commands", [])
        if not pending:
            self._update_status(iteration, current_lr, terrain_level)
            return []

        # Process commands
        applied_changes = []
        rm = self._get_reward_manager()
        now = time.time()

        for cmd in pending:
            cmd_type = cmd.get("type", "")
            cmd_ts = cmd.get("timestamp", "")
            force = cmd.get("force", False)

            # Stale check
            try:
                ts = datetime.fromisoformat(cmd_ts).timestamp()
                if now - ts > self.STALE_TIMEOUT:
                    applied_changes.append(("STALE", f"Skipped {cmd_type} from {cmd_ts}"))
                    continue
            except (ValueError, TypeError):
                pass  # No timestamp or bad format — process anyway

            if cmd_type == "emergency_stop":
                reason = cmd.get("reason", "manual")
                print(f"[CONTROL] EMERGENCY STOP: {reason}", flush=True)
                ckpt_path = os.path.join(self.log_dir, f"model_{iteration}_emergency.pt")
                self.runner.save(ckpt_path)
                print(f"[CONTROL] Saved {ckpt_path}", flush=True)
                self.changelog.record(iteration, "emergency_stop",
                                      cmd, {"saved": ckpt_path})
                os._exit(0)

            elif cmd_type == "save_checkpoint":
                label = cmd.get("label", "manual")
                ckpt_path = os.path.join(self.log_dir, f"model_{iteration}_{label}.pt")
                self.runner.save(ckpt_path)
                applied_changes.append(("SAVED", ckpt_path))
                self.changelog.record(iteration, "save_checkpoint", cmd,
                                      {"path": ckpt_path})

            elif cmd_type == "set_weight" and rm:
                name = cmd.get("name", "")
                value = cmd.get("value")
                if value is None:
                    continue
                current = self._get_weight(rm, name)
                if current is None:
                    applied_changes.append(("REJECTED", f"Unknown weight: {name}"))
                    continue
                validated, msgs = guardrails.validate_weight_change(
                    name, float(value), current, self.frozen_weights, force)
                if validated is not None:
                    old = self._set_weight(rm, name, validated)
                    applied_changes.append(("WEIGHT", f"{name}: {old} -> {validated}"))
                    self.changelog.record(iteration, "set_weight", cmd,
                                          {name: {"old": old, "new": validated}},
                                          msgs)
                else:
                    applied_changes.append(("REJECTED", " | ".join(msgs)))
                    self.changelog.record(iteration, "set_weight_rejected",
                                          cmd, {}, msgs)

            elif cmd_type == "set_lr_bounds":
                new_max = cmd.get("lr_max")
                new_min = cmd.get("lr_min")
                v_max, v_min, msgs = guardrails.validate_lr(
                    new_max, new_min,
                    self.lr_bounds.get("max"), self.lr_bounds.get("min"))
                if v_max is not None or v_min is not None:
                    old = dict(self.lr_bounds)
                    if v_max is not None:
                        self.lr_bounds["max"] = v_max
                    if v_min is not None:
                        self.lr_bounds["min"] = v_min
                    applied_changes.append(("LR", f"{old} -> {dict(self.lr_bounds)}"))
                    self.changelog.record(iteration, "set_lr_bounds", cmd,
                                          {"old": old, "new": dict(self.lr_bounds)}, msgs)
                else:
                    applied_changes.append(("REJECTED", " | ".join(msgs)))

            elif cmd_type == "set_noise_bounds":
                new_max = cmd.get("max_std")
                new_min = cmd.get("min_std")
                v_max, v_min, msgs = guardrails.validate_noise(new_max, new_min)
                if v_max is not None or v_min is not None:
                    old = dict(self.noise_bounds)
                    if v_max is not None:
                        self.noise_bounds["max"] = v_max
                    if v_min is not None:
                        self.noise_bounds["min"] = v_min
                    applied_changes.append(("NOISE", f"{old} -> {dict(self.noise_bounds)}"))
                    self.changelog.record(iteration, "set_noise_bounds", cmd,
                                          {"old": old, "new": dict(self.noise_bounds)}, msgs)
                else:
                    applied_changes.append(("REJECTED", " | ".join(msgs)))

            elif cmd_type == "set_s2r_param" and self.s2r_wrapper:
                param = cmd.get("param", "")
                value = cmd.get("value")
                if value is None:
                    continue
                validated, msgs = guardrails.validate_s2r_param(param, float(value))
                if validated is not None:
                    old = getattr(self.s2r_wrapper, param, None)
                    setattr(self.s2r_wrapper, param, validated)
                    applied_changes.append(("S2R", f"{param}: {old} -> {validated}"))
                    self.changelog.record(iteration, "set_s2r_param", cmd,
                                          {param: {"old": old, "new": validated}}, msgs)
                else:
                    applied_changes.append(("REJECTED", " | ".join(msgs)))

            elif cmd_type == "freeze_weight":
                name = cmd.get("name", "")
                self.frozen_weights.add(name)
                applied_changes.append(("FROZEN", name))
                self.changelog.record(iteration, "freeze_weight", cmd, {"frozen": name})

            elif cmd_type == "unfreeze_weight":
                name = cmd.get("name", "")
                if name in guardrails.HARD_FROZEN:
                    applied_changes.append(("REJECTED", f"Cannot unfreeze hard-frozen '{name}'"))
                else:
                    self.frozen_weights.discard(name)
                    applied_changes.append(("UNFROZEN", name))
                    self.changelog.record(iteration, "unfreeze_weight", cmd, {"unfrozen": name})

        # Clear pending commands and update status
        data["pending_commands"] = []
        self._write_status(data, iteration, current_lr, terrain_level)
        _write_control(self.control_path, data)

        return applied_changes

    # ------------------------------------------------------------------
    # Status updates
    # ------------------------------------------------------------------

    def _update_status(self, iteration: int, lr: float, terrain_level: float):
        """Update status section in control.yaml."""
        data = _read_control(self.control_path)
        if not data:
            return
        self._write_status(data, iteration, lr, terrain_level)
        _write_control(self.control_path, data)

    def _write_status(self, data: dict, iteration: int, lr: float,
                      terrain_level: float):
        """Write status fields into a data dict."""
        rm = self._get_reward_manager()
        noise = 0.0
        try:
            noise = self.runner.alg.policy.std.mean().item()
        except Exception:
            pass

        data["status"] = {
            "iteration": iteration,
            "lr": lr,
            "noise_std": round(noise, 4),
            "terrain_level": round(terrain_level, 4) if isinstance(terrain_level, float) else terrain_level,
            "s2r_scale": round(self.s2r_wrapper.scale, 3) if self.s2r_wrapper else 0.0,
            "last_poll": datetime.now().isoformat(timespec="seconds"),
        }
        data["current_weights"] = self._get_all_weights(rm) if rm else {}
        data["lr_bounds"] = dict(self.lr_bounds)
        data["noise_bounds"] = dict(self.noise_bounds)
        data["frozen_weights"] = sorted(self.frozen_weights)
        if self.s2r_wrapper:
            data["s2r_params"] = {
                "max_dropout_rate": self.s2r_wrapper.max_dropout_rate,
                "max_drift_rate": self.s2r_wrapper.max_drift_rate,
                "max_spike_prob": self.s2r_wrapper.max_spike_prob,
                "max_action_delay": self.s2r_wrapper.max_action_delay,
                "max_obs_delay": self.s2r_wrapper.max_obs_delay,
            }
