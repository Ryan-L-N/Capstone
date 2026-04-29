"""Unified eval launcher for Final Capstone Policy teacher/student policies.

Wraps the two existing eval entry points with the Loco_Policy_5_Final_Capstone_Policy-specific flags:
  - 4_env_test (friction/grass/boulder/stairs):
        run_capstone_eval.py --mason --action_scale 0.3 [--env {...}]
  - Cole skill-nav-lite:
        cole_arena_skillnav_lite.py --loco_checkpoint <ckpt> --loco_action_scale 0.3
        + the Cole-quarter-density recipe from skill_nav_lite_integration.md
          (moveable 0.25, nonmoveable 0.25, small 0.0, apf_radius 1.5,
           apf_gain 0.9, apf_tangent 0.8, max_lin_speed 2.4, waypoint_reach 0.9)

The wrapper runs one suite at a time (shared Isaac Sim GPU lock).

Usage (teacher quick smoke after Phase 2, rendered — user can watch):
    python scripts/eval.py --checkpoint logs/.../model_8000.pt \
        --target 4_env --envs friction,stairs

Usage (full battery on H100 post-training, headless):
    python scripts/eval.py --checkpoint logs/.../model_8000.pt \
        --target both --headless --num_episodes 3

Usage (Cole quarter density only, 3-seed):
    python scripts/eval.py --checkpoint logs/.../model_8000.pt \
        --target cole --cole_density quarter --seeds 42,123,7
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_ALEX_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CAPSTONE_EVAL = os.path.join(
    _ALEX_ROOT, "4_env_test", "src", "run_capstone_eval.py",
)
_COLE_EVAL = os.path.join(
    _ALEX_ROOT, "NAV_ALEX", "scripts", "cole_arena_skillnav_lite.py",
)
_PYTHON = sys.executable  # same conda env as caller

FINAL_CAPSTONE_POLICY_ACTION_SCALE = 0.3
DEFAULT_ENVS = ["friction", "grass", "boulder", "stairs"]

# Cole recipes cribbed from memory/skill_nav_lite_integration.md
COLE_RECIPES = {
    "quarter": dict(
        moveable_pct=0.25,
        nonmoveable_pct=0.25,
        small_static_pct=0.0,
        apf_radius=1.5,
        apf_gain=0.9,
        apf_tangent=0.8,
        max_lin_speed=2.4,
        waypoint_reach=0.9,
    ),
    "max": dict(
        moveable_pct=1.0,
        nonmoveable_pct=1.0,
        small_static_pct=0.0,
        apf_radius=1.5,
        apf_gain=0.9,
        apf_tangent=0.8,
        max_lin_speed=2.4,
        waypoint_reach=0.9,
    ),
}


def run_4env(args) -> int:
    envs = args.envs.split(",") if args.envs else DEFAULT_ENVS
    out_dir = os.path.abspath(args.output_dir or "results/parkour_nav_eval")
    os.makedirs(out_dir, exist_ok=True)

    failures = 0
    for env in envs:
        cmd = [
            _PYTHON, _CAPSTONE_EVAL,
            "--robot", "spot", "--policy", "rough", "--env", env,
            "--mason", "--action_scale", str(FINAL_CAPSTONE_POLICY_ACTION_SCALE),
            "--num_episodes", str(args.num_episodes),
            "--checkpoint", os.path.abspath(args.checkpoint),
            "--output_dir", out_dir,
            "--seed", str(args.seeds.split(",")[0]),
        ]
        cmd.append("--headless" if args.headless else "--rendered")
        print(f"\n[EVAL 4_env] {env} -> {out_dir}\n  {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[EVAL 4_env] {env} FAILED rc={rc}", flush=True)
            failures += 1
    return failures


def run_cole(args) -> int:
    if args.cole_density not in COLE_RECIPES:
        raise ValueError(f"--cole_density must be one of {list(COLE_RECIPES)}")
    recipe = COLE_RECIPES[args.cole_density]

    failures = 0
    for seed in args.seeds.split(","):
        seed = seed.strip()
        cmd = [
            _PYTHON, _COLE_EVAL,
            "--loco_checkpoint", os.path.abspath(args.checkpoint),
            "--loco_action_scale", str(FINAL_CAPSTONE_POLICY_ACTION_SCALE),
            "--loco_decimation", str(args.cole_decimation),
            "--cole_arena", "--rough_heightscan",
            "--episodes", str(args.num_episodes),
            "--seed", seed,
            "--moveable_pct", str(recipe["moveable_pct"]),
            "--nonmoveable_pct", str(recipe["nonmoveable_pct"]),
            "--small_static_pct", str(recipe["small_static_pct"]),
            "--apf_radius", str(recipe["apf_radius"]),
            "--apf_gain", str(recipe["apf_gain"]),
            "--apf_tangent", str(recipe["apf_tangent"]),
            "--max_lin_speed", str(recipe["max_lin_speed"]),
            "--waypoint_reach", str(recipe["waypoint_reach"]),
        ]
        if args.headless:
            cmd.append("--headless")
        else:
            cmd.append("--rendered")
        print(f"\n[EVAL cole] density={args.cole_density} seed={seed}\n  "
              f"{' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[EVAL cole] seed={seed} FAILED rc={rc}", flush=True)
            failures += 1
    return failures


def main():
    p = argparse.ArgumentParser(description="Loco_Policy_5_Final_Capstone_Policy eval launcher")
    p.add_argument("--checkpoint", required=True,
                   help="Path to Final Capstone Policy teacher or student .pt")
    p.add_argument("--target", choices=["4_env", "cole", "both"], default="both")
    p.add_argument("--envs", type=str, default=None,
                   help="Comma-list of 4_env arenas (default: all 4). "
                        "Ignored if --target cole.")
    p.add_argument("--num_episodes", type=int, default=1,
                   help="Episodes per env (4_env) or per seed (cole).")
    p.add_argument("--seeds", type=str, default="42",
                   help="Comma-list of seeds for Cole. 4_env uses the first.")
    p.add_argument("--cole_density", choices=["quarter", "max"], default="quarter")
    p.add_argument("--cole_decimation", type=int, default=1,
                   help="Decimation for Cole arena locomotion. Final Capstone Policy teacher "
                        "trains @ 50Hz — dec=1 with world.step()=10 substeps keeps "
                        "policy at 50Hz (matches rough-policy Cole integration).")
    p.add_argument("--headless", action="store_true",
                   help="Headless for both suites. Default = rendered (user watch).")
    p.add_argument("--output_dir", type=str, default=None,
                   help="4_env results dir (default: results/parkour_nav_eval).")
    args = p.parse_args()

    if not os.path.isfile(os.path.abspath(args.checkpoint)):
        print(f"[ERROR] checkpoint not found: {args.checkpoint}", flush=True)
        sys.exit(1)

    total_fail = 0
    if args.target in ("4_env", "both"):
        total_fail += run_4env(args)
    if args.target in ("cole", "both"):
        total_fail += run_cole(args)

    print(f"\n[EVAL] DONE — failures={total_fail}", flush=True)
    sys.exit(1 if total_fail else 0)


if __name__ == "__main__":
    main()
