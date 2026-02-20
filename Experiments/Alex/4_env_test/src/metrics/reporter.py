"""Summary statistics and visualization generator.

Reads JSONL episode files and produces:
  - summary.csv: aggregate statistics per policy per environment
  - Plots: completion rates, progress boxplots, fall heatmaps, stability curves

Usage:
    python src/metrics/reporter.py --input results/
"""

import csv
import json
import os
from collections import defaultdict

import numpy as np


def load_episodes(input_dir):
    """Read all *.jsonl files from input_dir.

    Returns:
        list of dicts, each representing one episode.
    """
    episodes = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".jsonl"):
            continue
        filepath = os.path.join(input_dir, fname)
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
    return episodes


def _group_episodes(episodes):
    """Group episodes by (environment, policy).

    Returns:
        dict of (env, policy) -> list of episode dicts
    """
    groups = defaultdict(list)
    for ep in episodes:
        key = (ep["environment"], ep["policy"])
        groups[key].append(ep)
    return dict(groups)


def compute_summary_statistics(episodes):
    """Compute aggregate statistics per (environment, policy) group.

    Returns:
        list of dicts with summary metrics per group.
    """
    groups = _group_episodes(episodes)
    summaries = []

    for (env, policy), eps in sorted(groups.items()):
        n = len(eps)
        completions = [e["completion"] for e in eps]
        progresses = [e["progress"] for e in eps]
        stabilities = [e["stability_score"] for e in eps]
        falls = [e["fall_detected"] for e in eps]
        velocities = [e["mean_velocity"] for e in eps]
        zones = [e["zone_reached"] for e in eps]

        completion_rate = sum(completions) / n if n > 0 else 0.0

        summary = {
            "environment": env,
            "policy": policy,
            "num_episodes": n,
            "completion_rate": round(completion_rate, 4),
            "mean_progress": round(float(np.mean(progresses)), 3),
            "std_progress": round(float(np.std(progresses)), 3),
            "median_progress": round(float(np.median(progresses)), 3),
            "mean_stability": round(float(np.mean(stabilities)), 6),
            "std_stability": round(float(np.std(stabilities)), 6),
            "fall_rate": round(sum(falls) / n, 4) if n > 0 else 0.0,
            "mean_velocity": round(float(np.mean(velocities)), 4),
            "mean_zone_reached": round(float(np.mean(zones)), 2),
        }

        # Zone distribution
        zone_counts = [0] * 5
        for z in zones:
            zone_counts[min(z, 5) - 1] += 1
        for i in range(5):
            summary[f"zone_{i+1}_count"] = zone_counts[i]

        summaries.append(summary)

    return summaries


def run_statistical_tests(episodes):
    """Run statistical tests comparing flat vs rough for each environment.

    Returns:
        list of dicts with test results per environment.
    """
    try:
        from scipy import stats
    except ImportError:
        print("scipy not available — skipping statistical tests")
        return []

    groups = _group_episodes(episodes)
    results = []

    envs = set(ep["environment"] for ep in episodes)
    for env in sorted(envs):
        flat_eps = groups.get((env, "flat"), [])
        rough_eps = groups.get((env, "rough"), [])

        if not flat_eps or not rough_eps:
            continue

        flat_prog = [e["progress"] for e in flat_eps]
        rough_prog = [e["progress"] for e in rough_eps]

        # Welch's t-test for progress
        t_stat, p_value = stats.ttest_ind(flat_prog, rough_prog, equal_var=False)

        # Cohen's d effect size
        pooled_std = np.sqrt((np.var(flat_prog) + np.var(rough_prog)) / 2)
        cohens_d = (np.mean(rough_prog) - np.mean(flat_prog)) / pooled_std if pooled_std > 0 else 0.0

        # Two-proportion z-test for completion rates
        n_flat, n_rough = len(flat_eps), len(rough_eps)
        p_flat = sum(e["completion"] for e in flat_eps) / n_flat
        p_rough = sum(e["completion"] for e in rough_eps) / n_rough
        p_pool = (p_flat * n_flat + p_rough * n_rough) / (n_flat + n_rough)

        if p_pool > 0 and p_pool < 1:
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n_flat + 1/n_rough))
            z_stat = (p_rough - p_flat) / se if se > 0 else 0.0
            z_pval = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat, z_pval = 0.0, 1.0

        results.append({
            "environment": env,
            "progress_t_stat": round(t_stat, 4),
            "progress_p_value": round(p_value, 6),
            "progress_cohens_d": round(cohens_d, 4),
            "completion_z_stat": round(z_stat, 4),
            "completion_p_value": round(z_pval, 6),
        })

    return results


def _save_csv(filepath, rows):
    """Write list of dicts to CSV."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_completion_rates(episodes, output_dir):
    """Bar chart: completion rate per environment, grouped by policy."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    groups = _group_episodes(episodes)
    envs = sorted(set(ep["environment"] for ep in episodes))
    policies = sorted(set(ep["policy"] for ep in episodes))

    x = np.arange(len(envs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, policy in enumerate(policies):
        rates = []
        for env in envs:
            eps = groups.get((env, policy), [])
            rate = sum(e["completion"] for e in eps) / len(eps) if eps else 0
            rates.append(rate * 100)
        offset = (i - 0.5) * width + width / 2
        ax.bar(x + offset, rates, width, label=policy.capitalize())

    ax.set_ylabel("Completion Rate (%)")
    ax.set_title("Completion Rate by Environment and Policy")
    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in envs])
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "completion_rates.png"), dpi=150)
    plt.close()


def plot_progress_boxplots(episodes, output_dir):
    """Box plot: progress distribution per environment, split by policy."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    groups = _group_episodes(episodes)
    envs = sorted(set(ep["environment"] for ep in episodes))
    policies = sorted(set(ep["policy"] for ep in episodes))

    fig, axes = plt.subplots(1, len(envs), figsize=(4 * len(envs), 6), sharey=True)
    if len(envs) == 1:
        axes = [axes]

    for ax, env in zip(axes, envs):
        data = []
        labels = []
        for policy in policies:
            eps = groups.get((env, policy), [])
            data.append([e["progress"] for e in eps])
            labels.append(policy.capitalize())
        ax.boxplot(data, tick_labels=labels)
        ax.set_title(env.capitalize())
        ax.set_ylabel("Progress (m)")

    plt.suptitle("Progress Distribution by Environment", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "progress_boxplot.png"), dpi=150)
    plt.close()


def plot_fall_heatmap(episodes, output_dir):
    """Heatmap: fall zone distribution per environment and policy."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    groups = _group_episodes(episodes)
    envs = sorted(set(ep["environment"] for ep in episodes))
    policies = sorted(set(ep["policy"] for ep in episodes))

    rows = []
    row_labels = []
    for env in envs:
        for policy in policies:
            eps = groups.get((env, policy), [])
            zone_falls = [0] * 5
            for e in eps:
                if e["fall_detected"] and e["fall_zone"] is not None:
                    idx = max(0, min(4, e["fall_zone"] - 1))
                    zone_falls[idx] += 1
            total = len(eps) if eps else 1
            rows.append([c / total * 100 for c in zone_falls])
            row_labels.append(f"{env}-{policy}")

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(rows) * 0.5 + 2)))
    im = ax.imshow(rows, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"Zone {i+1}" for i in range(5)])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Fall Rate by Zone (%)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fall_heatmap.png"), dpi=150)
    plt.close()


def plot_stability_by_zone(episodes, output_dir):
    """Line plot: mean stability score vs zone for each env/policy."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    groups = _group_episodes(episodes)
    envs = sorted(set(ep["environment"] for ep in episodes))
    policies = sorted(set(ep["policy"] for ep in episodes))

    fig, ax = plt.subplots(figsize=(10, 6))
    zones = np.arange(1, 6)

    for env in envs:
        for policy in policies:
            eps = groups.get((env, policy), [])
            if not eps:
                continue
            # Average stability for episodes that reached each zone
            zone_stab = []
            for z in range(1, 6):
                zone_eps = [e for e in eps if e["zone_reached"] >= z]
                if zone_eps:
                    zone_stab.append(np.mean([e["stability_score"] for e in zone_eps]))
                else:
                    zone_stab.append(np.nan)
            ax.plot(zones, zone_stab, marker="o", label=f"{env}-{policy}")

    ax.set_xlabel("Zone")
    ax.set_ylabel("Mean Stability Score (lower = better)")
    ax.set_title("Stability Score by Zone Reached")
    ax.legend(fontsize=8)
    ax.set_xticks(zones)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stability_by_zone.png"), dpi=150)
    plt.close()


def generate_report(input_dir, output_dir=None):
    """Generate summary CSV and plots from JSONL episode data."""
    output_dir = output_dir or input_dir
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Load data
    episodes = load_episodes(input_dir)
    if not episodes:
        print(f"No JSONL episode files found in {input_dir}")
        return

    print(f"Loaded {len(episodes)} episodes from {input_dir}")

    # Summary statistics
    summaries = compute_summary_statistics(episodes)
    _save_csv(os.path.join(output_dir, "summary.csv"), summaries)
    print(f"Saved summary.csv ({len(summaries)} groups)")

    # Print summary table
    print("\n--- Summary ---")
    for s in summaries:
        print(f"  {s['environment']:10s} {s['policy']:6s}  "
              f"completion={s['completion_rate']:.1%}  "
              f"progress={s['mean_progress']:.1f}m  "
              f"stability={s['mean_stability']:.4f}  "
              f"falls={s['fall_rate']:.1%}")

    # Statistical tests
    test_results = run_statistical_tests(episodes)
    if test_results:
        _save_csv(os.path.join(output_dir, "statistical_tests.csv"), test_results)
        print("\n--- Statistical Tests (flat vs rough) ---")
        for r in test_results:
            sig = "*" if r["progress_p_value"] < 0.05 else ""
            print(f"  {r['environment']:10s}  "
                  f"progress p={r['progress_p_value']:.4f}{sig}  "
                  f"d={r['progress_cohens_d']:.3f}  "
                  f"completion p={r['completion_p_value']:.4f}")

    # Plots
    plot_completion_rates(episodes, plots_dir)
    plot_progress_boxplots(episodes, plots_dir)
    plot_fall_heatmap(episodes, plots_dir)
    plot_stability_by_zone(episodes, plots_dir)
    print(f"\nPlots saved to {plots_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("--input", required=True, help="Directory containing JSONL episode files")
    parser.add_argument("--output", default=None, help="Output directory (default: same as input)")
    args = parser.parse_args()
    generate_report(args.input, args.output or args.input)
