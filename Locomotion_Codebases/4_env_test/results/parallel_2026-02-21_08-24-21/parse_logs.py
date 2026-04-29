"""
Parse *_flat.log files → *_flat_episodes.jsonl
Expected by visualize_flat_policy.py.

Missing fields approximated:
  mean_velocity  = progress / time  (m/s)
  stability_score = 0.05 for TIMEOUT, 1.0 for FELL  (lower = more stable)
  completion      = True only if status == COMPLETE
  fall_zone       = zone if FELL, else None
"""

import re
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
ENVS = ["friction", "grass", "boulder", "stairs"]

LINE_RE = re.compile(
    r"\[\s*(\d+)/100\]\s+\S+\s+(TIMEOUT|FELL|COMPLETE)\s+"
    r"progress=\s*([\d.]+)m\s+zone=(\d+)\s+time=([\d.]+)s"
)

for env in ENVS:
    log_path  = os.path.join(BASE, f"{env}_flat.log")
    jsonl_path = os.path.join(BASE, f"{env}_flat_episodes.jsonl")

    if not os.path.exists(log_path):
        print(f"[SKIP] {log_path} not found")
        continue

    records = []
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            _ep_num, status, progress, zone, time_s = m.groups()
            progress = float(progress)
            zone     = int(zone)
            time_s   = float(time_s)

            fell       = status == "FELL"
            completion = status == "COMPLETE"
            mean_vel   = progress / time_s if time_s > 0 else 0.0
            stability  = 1.0 if fell else 0.05

            records.append({
                "progress":        progress,
                "fall_detected":   fell,
                "completion":      completion,
                "zone_reached":    zone,
                "fall_zone":       zone if fell else None,
                "mean_velocity":   round(mean_vel, 4),
                "stability_score": stability,
                "episode_time":    time_s,
            })

    with open(jsonl_path, "w") as out:
        for r in records:
            out.write(json.dumps(r) + "\n")

    falls    = sum(1 for r in records if r["fall_detected"])
    timeouts = sum(1 for r in records if not r["fall_detected"] and not r["completion"])
    complete = sum(1 for r in records if r["completion"])
    print(f"{env:10s}  {len(records):3d} eps  fell={falls}  timeout={timeouts}  complete={complete}  -> {os.path.basename(jsonl_path)}")

print("\nDone. Now run:  python visualize_flat_policy.py")
