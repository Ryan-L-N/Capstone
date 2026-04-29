"""Aggregate results.csv from run_nav_evals.py into per-(policy,density) stats.

Usage:
    python aggregate_results.py results/run_<ts>/results.csv
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="path to results.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv_path, encoding="utf-8")))
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_group[(r["policy"], r["density"])].append(r)

    hdr = f"{'policy':<16} {'density':<6} {'n':>4} {'fell':>5} {'compl':>6} {'scoredepl':>10} "
    hdr += f"{'wp_mean':>8} {'wp_med':>7} {'wp_max':>7} {'dist_mean':>10} {'time_mean':>10}"
    print(hdr)
    print("-" * len(hdr))

    for (pol, den), runs in sorted(by_group.items()):
        wps = [int(r["wp_reached"]) for r in runs]
        dists = [float(r["dist_m"]) for r in runs]
        times = [float(r["time_s"]) for r in runs]
        statuses = [r["status"] for r in runs]
        n = len(runs)
        fell = sum(1 for s in statuses if s == "FELL")
        compl = sum(1 for s in statuses if s == "COMPLETE")
        depl = sum(1 for s in statuses if s == "SCORE_DEPLETED")
        print(f"{pol:<16} {den:<6} {n:>4} {fell:>5} {compl:>6} {depl:>10} "
              f"{statistics.mean(wps):>8.2f} {statistics.median(wps):>7.1f} "
              f"{max(wps):>7d} {statistics.mean(dists):>10.1f} "
              f"{statistics.mean(times):>10.1f}")


if __name__ == "__main__":
    main()
