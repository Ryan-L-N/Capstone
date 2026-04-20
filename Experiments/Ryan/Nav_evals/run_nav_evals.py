"""Batch nav-eval runner: 100 seeds x 4 policies x 2 densities.

Runs `cole_arena_skillnav_lite.py` headless for each (policy, density, seed) combo
and aggregates the results into a CSV. Logs are written per-run under ./logs.

Ryan: edit `CONFIG` below so PYTHON, NAV_ALEX_DIR, and the checkpoint paths
point at the real locations on your machine, then run:

    python run_nav_evals.py --seeds 100

Partial runs are safe to resume — already-finished (policy, density, seed) combos
are detected by a completed log line and skipped.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ============================================================================
# CONFIG — Ryan: edit these paths to match your machine
# ============================================================================
CONFIG = {
    # Python interpreter with Isaac Lab 2.3 / Isaac Sim 5.1 installed
    "PYTHON": r"C:/miniconda3/envs/isaaclab311/python.exe",

    # Path to Alex's NAV_ALEX directory (contains scripts/cole_arena_skillnav_lite.py).
    # If you clone the Capstone repo, this is .../Capstone/Experiments/Alex/NAV_ALEX
    "NAV_ALEX_DIR": r"C:/Users/Gabriel Santiago/OneDrive/Desktop/Capstone Project/Capstone/Experiments/Alex/NAV_ALEX",

    # Checkpoint paths. Absolute paths are easiest.
    "CKPT_BOULDER_V6": r"C:/Users/Gabriel Santiago/OneDrive/Desktop/Capstone Project/Capstone/Experiments/Alex/NAV_ALEX/checkpoints/boulder_v6_expert_4500.pt",
    "CKPT_MASON_BASELINE": r"C:/Users/Gabriel Santiago/OneDrive/Desktop/Capstone Project/Capstone/Experiments/Alex/multi_robot_training/checkpoints/mason_baseline_final_19999.pt",
    "CKPT_MASON_HYBRID": r"C:/Users/Gabriel Santiago/OneDrive/Desktop/Capstone Project/Capstone/Experiments/Alex/multi_robot_training/checkpoints/mason_hybrid_best_33200.pt",

    # Per-episode wall-clock cap (seconds). Headless full run ~5 min; kill runaway at 15 min.
    "PER_RUN_TIMEOUT_SEC": 900,
}
# ============================================================================

POLICIES = {
    # label -> extra CLI args appended to the base command
    "stock_flat":     ["--stock_flat"],
    "boulder_v6":     ["--loco_checkpoint", CONFIG["CKPT_BOULDER_V6"],
                       "--rough_heightscan", "--loco_decimation", "1"],
    "mason_baseline": ["--loco_checkpoint", CONFIG["CKPT_MASON_BASELINE"],
                       "--rough_heightscan", "--loco_decimation", "1"],
    "mason_hybrid":   ["--loco_checkpoint", CONFIG["CKPT_MASON_HYBRID"],
                       "--rough_heightscan", "--loco_decimation", "1"],
}

DENSITIES = {
    "full": [],  # default: full Cole density (~662 obstacles, 6.1% coverage)
    "30pct": ["--moveable_pct", "0.9",
              "--nonmoveable_pct", "0.9",
              "--small_static_pct", "0.09"],
}

BASE_ARGS = [
    "--episodes", "1",
    "--cole_arena",
    "--depth_sensor",
    "--planner_inflate", "0.7",
    "--replan_period_sec", "3.0",
    "--headless",
]

RESULT_RE = re.compile(
    r"Result:\s*(?P<status>\S+)\s*\|\s*WP:\s*(?P<wp>\d+)/(?P<wp_total>\d+)"
    r"\s*\|\s*Dist:\s*(?P<dist>[\d.]+)m\s*\|\s*Score:\s*(?P<score>-?[\d.]+)"
    r"\s*\|\s*Time:\s*(?P<time>[\d.]+)s"
)


def parse_log(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(errors="ignore")
    except Exception:
        return None
    m = RESULT_RE.search(text)
    if not m:
        return None
    return {
        "status": m.group("status"),
        "wp_reached": int(m.group("wp")),
        "wp_total": int(m.group("wp_total")),
        "dist_m": float(m.group("dist")),
        "score": float(m.group("score")),
        "time_s": float(m.group("time")),
    }


def run_one(policy: str, density: str, seed: int, logs_dir: Path) -> dict:
    log_path = logs_dir / f"{policy}_{density}_seed{seed:04d}.log"

    cached = parse_log(log_path)
    if cached is not None:
        cached.update(policy=policy, density=density, seed=seed,
                      log=str(log_path), cached=True)
        return cached

    cmd = [CONFIG["PYTHON"], "-u",
           str(Path(CONFIG["NAV_ALEX_DIR"]) / "scripts" / "cole_arena_skillnav_lite.py"),
           *BASE_ARGS,
           "--seed", str(seed),
           *DENSITIES[density],
           *POLICIES[policy]]

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"# cmd: {' '.join(cmd)}\n")
        fh.flush()
        try:
            subprocess.run(
                cmd,
                cwd=CONFIG["NAV_ALEX_DIR"],
                stdout=fh,
                stderr=subprocess.STDOUT,
                timeout=CONFIG["PER_RUN_TIMEOUT_SEC"],
                check=False,
            )
        except subprocess.TimeoutExpired:
            fh.write("\n# TIMEOUT\n")
    dur = time.time() - t0

    parsed = parse_log(log_path)
    if parsed is None:
        return {"policy": policy, "density": density, "seed": seed,
                "status": "PARSE_FAIL", "wp_reached": 0, "wp_total": 25,
                "dist_m": 0.0, "score": 0.0, "time_s": 0.0,
                "log": str(log_path), "wall_s": dur, "cached": False}
    parsed.update(policy=policy, density=density, seed=seed,
                  log=str(log_path), wall_s=dur, cached=False)
    return parsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100,
                    help="number of seeds per (policy, density) — default 100")
    ap.add_argument("--seed_start", type=int, default=0,
                    help="first seed id — default 0")
    ap.add_argument("--policies", nargs="+", default=list(POLICIES.keys()),
                    choices=list(POLICIES.keys()))
    ap.add_argument("--densities", nargs="+", default=list(DENSITIES.keys()),
                    choices=list(DENSITIES.keys()))
    ap.add_argument("--out_dir", default=None,
                    help="override output dir (default: ./results/run_<ts>)")
    args = ap.parse_args()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else HERE / "results" / f"run_{ts}"
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    plan = [(p, d, s) for p in args.policies
                      for d in args.densities
                      for s in range(args.seed_start, args.seed_start + args.seeds)]
    total = len(plan)
    print(f"[run_nav_evals] plan: {total} runs -> {out_dir}")
    (out_dir / "config_snapshot.json").write_text(json.dumps({
        "config": CONFIG, "policies": list(args.policies),
        "densities": list(args.densities), "seeds": args.seeds,
        "seed_start": args.seed_start,
    }, indent=2))

    csv_path = out_dir / "results.csv"
    fieldnames = ["policy", "density", "seed", "status", "wp_reached", "wp_total",
                  "dist_m", "score", "time_s", "wall_s", "cached", "log"]
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            writer.writeheader()

        for i, (policy, density, seed) in enumerate(plan, 1):
            print(f"[{i}/{total}] {policy} / {density} / seed={seed}")
            rec = run_one(policy, density, seed, logs_dir)
            writer.writerow({**rec, "wall_s": rec.get("wall_s", 0.0)})
            cf.flush()
            tag = "CACHED" if rec.get("cached") else "RAN"
            print(f"    {tag} {rec['status']:<16} wp={rec['wp_reached']}/{rec['wp_total']} "
                  f"dist={rec['dist_m']:.1f}m time={rec['time_s']:.1f}s")

    print(f"[run_nav_evals] done. csv: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
