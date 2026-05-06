"""
22100 (Final Capstone Policy) vs Flat Policy — Poster visualization.

Mirrors the toolkit used in `parallel_2026-02-21_08-24-21/visualize_flat_policy.py`
and `plots/Rough_v_Flat_19_feb/` so plots match the Phase-2 deliverable style.

Inputs (100 episodes per env, schema matches episode_schema.json):
  - flat:  4_env_test/results/parallel_2026-02-21_08-24-21/{env}_flat_episodes.jsonl
  - 22100: Experiments/Ryan/22100 Final Eval 100/{env}_rough_episodes.jsonl

Outputs (plots_22100_vs_flat/):
  - summary.csv, statistical_tests.csv
  - completion_rates.png, fall_rates.png, mean_progress.png
  - progress_boxplot.png, mean_velocity.png, zone_distribution.png
  - fall_heatmap.png, stall_heatmap.png, stability_by_zone.png
  - composite_4panel.png (PPT widescreen 13.33 x 7.5)
  - per-env pies + headline panel
"""

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sstats

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent.resolve()
FLAT_DIR = (
    HERE.parent.parent.parent.parent  # eval_100ep -> Final_Capstone_Policy_22100 -> Ryan -> Experiments -> Capstone
    / "Locomotion_Codebases"
    / "4_env_test"
    / "results"
    / "parallel_2026-02-21_08-24-21"
)
OUT_DIR = HERE / "plots_22100_vs_flat"
OUT_DIR.mkdir(exist_ok=True)

ENVIRONMENTS = ["friction", "grass", "boulder", "stairs"]
ENV_LABELS = {"friction": "Friction", "grass": "Grass", "boulder": "Boulder", "stairs": "Stairs"}
ENV_COLORS = {"friction": "#2196F3", "grass": "#4CAF50", "boulder": "#FF9800", "stairs": "#F44336"}

POLICIES = ["flat", "22100"]
POLICY_LABELS = {"flat": "Flat Policy", "22100": "Final (22100)"}
POLICY_COLORS = {"flat": "#2196F3", "22100": "#FF5722"}


# ── Loaders ──────────────────────────────────────────────────────────────────
def load_jsonl(path: Path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_episodes():
    eps = {p: {} for p in POLICIES}
    for env in ENVIRONMENTS:
        eps["flat"][env] = load_jsonl(FLAT_DIR / f"{env}_flat_episodes.jsonl")
        eps["22100"][env] = load_jsonl(HERE / f"{env}_rough_episodes.jsonl")
    return eps


def summarize(eps):
    out = {}
    for policy in POLICIES:
        out[policy] = {}
        for env in ENVIRONMENTS:
            arr = eps[policy][env]
            progress = np.array([e["progress"] for e in arr])
            stability = np.array([e["stability_score"] for e in arr])
            velocity = np.array([e["mean_velocity"] for e in arr])
            zones = np.array([e["zone_reached"] for e in arr])
            falls = np.array([1 if e["fall_detected"] else 0 for e in arr])
            completions = np.array([1 if e["completion"] else 0 for e in arr])
            stalls = np.array(
                [1 if (not e["fall_detected"] and not e["completion"]) else 0 for e in arr]
            )
            zone_counts = [int(np.sum(zones == z)) for z in range(1, 6)]
            out[policy][env] = dict(
                n=len(arr),
                completion_rate=float(completions.mean()),
                fall_rate=float(falls.mean()),
                stall_rate=float(stalls.mean()),
                mean_progress=float(progress.mean()),
                std_progress=float(progress.std(ddof=1)) if len(progress) > 1 else 0.0,
                median_progress=float(np.median(progress)),
                mean_stability=float(stability.mean()),
                std_stability=float(stability.std(ddof=1)) if len(stability) > 1 else 0.0,
                mean_velocity=float(velocity.mean()),
                mean_zone=float(zones.mean()),
                zone_counts=zone_counts,
                progress=progress,
                stability=stability,
                velocity=velocity,
                zones=zones,
                falls=falls,
                completions=completions,
                stalls=stalls,
                eps=arr,
            )
    return out


# ── Stats ────────────────────────────────────────────────────────────────────
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def two_proportion_z(p1, n1, p2, n2):
    x1, x2 = p1 * n1, p2 * n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p_val = 2 * (1 - sstats.norm.cdf(abs(z)))
    return float(z), float(p_val)


def stat_table(stats):
    """Welch's t on progress, two-prop-z on completion + fall rates per env."""
    rows = []
    for env in ENVIRONMENTS:
        a = stats["22100"][env]["progress"]
        b = stats["flat"][env]["progress"]
        t_stat, t_p = sstats.ttest_ind(a, b, equal_var=False)
        d = cohens_d(a, b)  # positive => 22100 > flat
        n = stats["flat"][env]["n"]
        cp_22 = stats["22100"][env]["completion_rate"]
        cp_fl = stats["flat"][env]["completion_rate"]
        z_c, p_c = two_proportion_z(cp_22, n, cp_fl, n)
        fp_22 = stats["22100"][env]["fall_rate"]
        fp_fl = stats["flat"][env]["fall_rate"]
        z_f, p_f = two_proportion_z(fp_22, n, fp_fl, n)
        rows.append(
            dict(
                environment=env,
                progress_t_stat=float(t_stat),
                progress_p_value=float(t_p),
                progress_cohens_d=d,
                completion_z_stat=z_c,
                completion_p_value=p_c,
                fall_z_stat=z_f,
                fall_p_value=p_f,
                progress_22100_minus_flat=float(a.mean() - b.mean()),
                completion_22100_minus_flat=float(cp_22 - cp_fl),
                fall_22100_minus_flat=float(fp_22 - fp_fl),
            )
        )
    return rows


def sig_label(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ── CSV writers ──────────────────────────────────────────────────────────────
def write_summary_csv(stats, path):
    fields = [
        "environment", "policy", "num_episodes", "completion_rate", "fall_rate",
        "stall_rate", "mean_progress", "std_progress", "median_progress",
        "mean_stability", "std_stability", "mean_velocity", "mean_zone_reached",
        "zone_1_count", "zone_2_count", "zone_3_count", "zone_4_count", "zone_5_count",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for env in ENVIRONMENTS:
            for policy in POLICIES:
                s = stats[policy][env]
                row = {
                    "environment": env,
                    "policy": policy,
                    "num_episodes": s["n"],
                    "completion_rate": round(s["completion_rate"], 3),
                    "fall_rate": round(s["fall_rate"], 3),
                    "stall_rate": round(s["stall_rate"], 3),
                    "mean_progress": round(s["mean_progress"], 3),
                    "std_progress": round(s["std_progress"], 3),
                    "median_progress": round(s["median_progress"], 3),
                    "mean_stability": round(s["mean_stability"], 6),
                    "std_stability": round(s["std_stability"], 6),
                    "mean_velocity": round(s["mean_velocity"], 4),
                    "mean_zone_reached": round(s["mean_zone"], 2),
                }
                for i, c in enumerate(s["zone_counts"], start=1):
                    row[f"zone_{i}_count"] = c
                w.writerow(row)


def write_stats_csv(rows, path):
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
            )


# ── Plot helpers ─────────────────────────────────────────────────────────────
def save(fig, name):
    p = OUT_DIR / name
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved {name}")


def grouped_bar(ax, env_keys, vals, errs=None, ylabel="", title="", ymax=None,
                 fmt="{:.1f}", suffix="", show_legend=True):
    x = np.arange(len(env_keys))
    width = 0.38
    for i, policy in enumerate(POLICIES):
        offset = (i - 0.5) * width
        vs = [vals[policy][e] for e in env_keys]
        es = [errs[policy][e] for e in env_keys] if errs else None
        bars = ax.bar(
            x + offset, vs, width, color=POLICY_COLORS[policy], alpha=0.88,
            zorder=3, label=POLICY_LABELS[policy],
            yerr=es, capsize=4, error_kw={"elinewidth": 1.1},
        )
        for bar, v in zip(bars, vs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (ymax * 0.015 if ymax else 0.02 * max(vs)),
                fmt.format(v) + suffix,
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABELS[e] for e in env_keys], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    if show_legend:
        ax.legend(fontsize=9, loc="upper right")


def annotate_sig(ax, x_pos, y_top, label):
    ax.text(x_pos, y_top, label, ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#444")


# ── Plots ────────────────────────────────────────────────────────────────────
def plot_completion_rates(stats, sig_rows):
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    vals = {p: {e: stats[p][e]["completion_rate"] * 100 for e in ENVIRONMENTS}
            for p in POLICIES}
    grouped_bar(ax, ENVIRONMENTS, vals,
                ylabel="Completion Rate (%)",
                title="Course Completion Rate — 22100 vs Flat (n=100/env)",
                ymax=110, fmt="{:.0f}", suffix="%")
    sig_map = {r["environment"]: sig_label(r["completion_p_value"]) for r in sig_rows}
    for i, env in enumerate(ENVIRONMENTS):
        top = max(vals["flat"][env], vals["22100"][env]) + 6
        annotate_sig(ax, i, top, sig_map[env])
    fig.tight_layout()
    save(fig, "completion_rates.png")


def plot_fall_rates(stats, sig_rows):
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    vals = {p: {e: stats[p][e]["fall_rate"] * 100 for e in ENVIRONMENTS}
            for p in POLICIES}
    grouped_bar(ax, ENVIRONMENTS, vals,
                ylabel="Fall Rate (%)",
                title="Fall Rate — 22100 vs Flat (n=100/env)",
                ymax=115, fmt="{:.0f}", suffix="%")
    ax.axhline(50, color="red", linestyle="--", linewidth=1, alpha=0.5, label="50% threshold")
    sig_map = {r["environment"]: sig_label(r["fall_p_value"]) for r in sig_rows}
    for i, env in enumerate(ENVIRONMENTS):
        top = max(vals["flat"][env], vals["22100"][env]) + 6
        annotate_sig(ax, i, top, sig_map[env])
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    save(fig, "fall_rates.png")


def plot_mean_progress(stats, sig_rows):
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    vals = {p: {e: stats[p][e]["mean_progress"] for e in ENVIRONMENTS} for p in POLICIES}
    errs = {p: {e: stats[p][e]["std_progress"] for e in ENVIRONMENTS} for p in POLICIES}
    grouped_bar(ax, ENVIRONMENTS, vals, errs=errs,
                ylabel="Mean Progress (m)",
                title="Mean Forward Progress — 22100 vs Flat (n=100/env)",
                ymax=60, fmt="{:.1f}", suffix="m")
    ax.axhline(49, color="green", linestyle="--", linewidth=1, alpha=0.5, label="Course goal (49m)")
    sig_map = {r["environment"]: sig_label(r["progress_p_value"]) for r in sig_rows}
    for i, env in enumerate(ENVIRONMENTS):
        top = max(vals["flat"][env] + errs["flat"][env],
                  vals["22100"][env] + errs["22100"][env]) + 4
        annotate_sig(ax, i, top, sig_map[env])
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    save(fig, "mean_progress.png")


def plot_progress_boxplot(stats):
    fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
    positions = []
    box_data = []
    box_colors = []
    box_labels = []
    width = 0.34
    for i, env in enumerate(ENVIRONMENTS):
        for j, policy in enumerate(POLICIES):
            positions.append(i + (j - 0.5) * (width + 0.06))
            box_data.append(stats[policy][env]["progress"])
            box_colors.append(POLICY_COLORS[policy])
            box_labels.append(f"{ENV_LABELS[env]}\n{POLICY_LABELS[policy]}")
    bp = ax.boxplot(
        box_data, positions=positions, widths=width, patch_artist=True,
        medianprops={"color": "black", "linewidth": 2.0},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
    )
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.78)
    ax.set_xticks(np.arange(len(ENVIRONMENTS)))
    ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=12)
    ax.set_ylabel("Forward Progress per Episode (m)", fontsize=12)
    ax.set_title("Episode Progress Distribution — 22100 vs Flat", fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.axhline(49, color="green", linestyle="--", linewidth=1, alpha=0.5, label="Course goal (49m)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=POLICY_COLORS[p], alpha=0.78,
                             label=POLICY_LABELS[p]) for p in POLICIES]
    ax.legend(handles=handles + [plt.Line2D([0], [0], color="green", linestyle="--",
                                            label="Course goal (49m)")],
              fontsize=10, loc="upper right")
    fig.tight_layout()
    save(fig, "progress_boxplot.png")


def plot_mean_velocity(stats):
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    vals = {p: {e: stats[p][e]["mean_velocity"] for e in ENVIRONMENTS} for p in POLICIES}
    grouped_bar(ax, ENVIRONMENTS, vals,
                ylabel="Mean Velocity (m/s)",
                title="Mean Forward Velocity — 22100 vs Flat",
                ymax=2.5, fmt="{:.2f}")
    ax.axhline(2.235, color="red", linestyle="--", linewidth=1.2,
               alpha=0.6, label="Spot Max (2.235 m/s)")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    save(fig, "mean_velocity.png")


def plot_zone_distribution(stats):
    zone_colors = ["#EF5350", "#FF7043", "#FFA726", "#66BB6A", "#42A5F5"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white", sharey=True)
    for ax, policy in zip(axes, POLICIES):
        bottom = np.zeros(len(ENVIRONMENTS))
        x = np.arange(len(ENVIRONMENTS))
        for z in range(5):
            counts = np.array([stats[policy][e]["zone_counts"][z] for e in ENVIRONMENTS])
            ax.bar(x, counts, bottom=bottom, color=zone_colors[z], alpha=0.92,
                   label=f"Zone {z + 1}", zorder=3)
            for i, c in enumerate(counts):
                if c >= 6:
                    ax.text(i, bottom[i] + c / 2, str(int(c)),
                            ha="center", va="center", fontsize=9, color="white",
                            fontweight="bold")
            bottom += counts
        ax.set_xticks(x)
        ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=11)
        ax.set_title(POLICY_LABELS[policy], fontsize=13, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    axes[0].set_ylabel("Number of Episodes", fontsize=12)
    axes[1].legend(fontsize=9, loc="lower right", title="Highest zone reached")
    fig.suptitle("Highest Zone Reached — 22100 vs Flat (100 ep each)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "zone_distribution.png")


def fall_matrix(stats, policy):
    m = np.zeros((len(ENVIRONMENTS), 5))
    for i, env in enumerate(ENVIRONMENTS):
        for ep in stats[policy][env]["eps"]:
            if ep["fall_detected"] and ep.get("fall_zone") is not None:
                z = int(ep["fall_zone"]) - 1
                if 0 <= z < 5:
                    m[i, z] += 1
    return m


def stall_matrix(stats, policy):
    m = np.zeros((len(ENVIRONMENTS), 5))
    for i, env in enumerate(ENVIRONMENTS):
        for ep in stats[policy][env]["eps"]:
            if not ep["fall_detected"] and not ep["completion"]:
                z = max(0, min(4, int(ep["progress"] / 10)))
                m[i, z] += 1
    return m


def plot_heatmap_pair(name, mat_flat, mat_22100, cmap, title, cbar_label):
    zone_labels = ["Z1\n(0-10m)", "Z2\n(10-20m)", "Z3\n(20-30m)",
                   "Z4\n(30-40m)", "Z5\n(40-50m)"]
    zone_flipped = zone_labels[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), facecolor="white")
    vmax = max(mat_flat.max(), mat_22100.max(), 1)
    for ax, mat, policy in zip(axes, [mat_flat, mat_22100], POLICIES):
        T = mat.T[::-1]
        im = ax.imshow(T, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
        for i in range(5):
            for j in range(len(ENVIRONMENTS)):
                v = int(T[i, j])
                tc = "white" if v > vmax * 0.55 else "black"
                if v > 0:
                    ax.text(j, i, str(v), ha="center", va="center",
                            fontsize=12, fontweight="bold", color=tc)
        ax.set_xticks(np.arange(len(ENVIRONMENTS)))
        ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=11)
        ax.set_yticks(np.arange(5))
        ax.set_yticklabels(zone_flipped, fontsize=9)
        ax.set_title(POLICY_LABELS[policy], fontsize=13, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label(cbar_label, fontsize=11)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    save(fig, name)


def plot_fall_heatmap(stats):
    plot_heatmap_pair(
        "fall_heatmap.png",
        fall_matrix(stats, "flat"), fall_matrix(stats, "22100"),
        cmap="YlOrRd",
        title="Fall Locations by Zone — 22100 vs Flat (100 ep / env)",
        cbar_label="Falls",
    )


def plot_stall_heatmap(stats):
    plot_heatmap_pair(
        "stall_heatmap.png",
        stall_matrix(stats, "flat"), stall_matrix(stats, "22100"),
        cmap="YlOrBr",
        title="Stall Locations by Zone — 22100 vs Flat\n(timeouts: alive but didn't complete)",
        cbar_label="Stalls",
    )


def plot_stability_by_zone(stats):
    fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
    markers = {"friction": "o", "grass": "^", "boulder": "s", "stairs": "D"}
    for policy in POLICIES:
        for env in ENVIRONMENTS:
            arr = stats[policy][env]["eps"]
            zones_present = sorted({e["zone_reached"] for e in arr})
            xs, ys = [], []
            for z in zones_present:
                vals = [e["stability_score"] for e in arr if e["zone_reached"] == z]
                if vals:
                    xs.append(z)
                    ys.append(float(np.mean(vals)))
            ls = "-" if policy == "22100" else "--"
            ax.plot(xs, ys, ls, marker=markers[env], color=ENV_COLORS[env],
                    linewidth=2 if policy == "22100" else 1.4,
                    markersize=8 if policy == "22100" else 6,
                    alpha=0.95 if policy == "22100" else 0.6,
                    label=f"{ENV_LABELS[env]} ({POLICY_LABELS[policy]})")
    ax.set_xlabel("Highest Zone Reached", fontsize=12)
    ax.set_ylabel("Mean Stability Score (lower = more stable)", fontsize=12)
    ax.set_title("Stability Degradation by Zone — 22100 vs Flat",
                 fontsize=14, fontweight="bold")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    save(fig, "stability_by_zone.png")


def plot_outcome_pies(stats):
    fig, axes = plt.subplots(2, 4, figsize=(15, 9), facecolor="white")
    for col, env in enumerate(ENVIRONMENTS):
        for row, policy in enumerate(POLICIES):
            s = stats[policy][env]
            n = s["n"]
            n_fall = int(s["falls"].sum())
            n_complete = int(s["completions"].sum())
            n_stall = int(s["stalls"].sum())
            sizes, labels, colors, explode = [], [], [], []
            if n_fall:
                sizes.append(n_fall)
                labels.append(f"Falls\n({n_fall}/{n})")
                colors.append("#F44336")
                explode.append(0.04)
            if n_stall:
                sizes.append(n_stall)
                labels.append(f"Stalls\n({n_stall}/{n})")
                colors.append("#FF9800")
                explode.append(0.04)
            if n_complete:
                sizes.append(n_complete)
                labels.append(f"Complete\n({n_complete}/{n})")
                colors.append("#4CAF50")
                explode.append(0.04)
            ax = axes[row, col]
            if not sizes:
                ax.axis("off")
                continue
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, colors=colors, explode=explode,
                autopct="%1.0f%%", startangle=90,
                textprops={"fontsize": 10},
                wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
            )
            for t in autotexts:
                t.set_fontweight("bold")
                t.set_fontsize(11)
            ax.set_title(f"{ENV_LABELS[env]} — {POLICY_LABELS[policy]}",
                         fontsize=12, fontweight="bold")
    fig.suptitle("Episode Outcomes — 22100 vs Flat (100 ep each)",
                 fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout()
    save(fig, "outcome_pies_4x2.png")


def plot_composite_4panel(stats, sig_rows):
    """PPT-widescreen 13.33 x 7.5 — Progress | Fall Heatmap (22100) | Fall Rate | Completion Rate."""
    fig = plt.figure(figsize=(13.33, 7.5), facecolor="white")
    fig.suptitle(
        "Final Capstone Policy (22100) vs Flat Policy — 100-Episode Eval, 4 Environments",
        fontsize=18, fontweight="bold", y=0.97,
    )
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1.2, 1, 1], wspace=0.40,
                          left=0.05, right=0.97, top=0.85, bottom=0.13)
    x = np.arange(len(ENVIRONMENTS))
    width = 0.38

    # Panel 1: Mean Progress
    ax = fig.add_subplot(gs[0])
    for j, policy in enumerate(POLICIES):
        means = [stats[policy][e]["mean_progress"] for e in ENVIRONMENTS]
        stds = [stats[policy][e]["std_progress"] for e in ENVIRONMENTS]
        offset = (j - 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                      color=POLICY_COLORS[policy], alpha=0.88,
                      label=POLICY_LABELS[policy], zorder=3,
                      error_kw={"elinewidth": 1.0})
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                    f"{m:.0f}", ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold")
    sig_map = {r["environment"]: sig_label(r["progress_p_value"]) for r in sig_rows}
    for i, env in enumerate(ENVIRONMENTS):
        top = max(stats["flat"][env]["mean_progress"] + stats["flat"][env]["std_progress"],
                  stats["22100"][env]["mean_progress"] + stats["22100"][env]["std_progress"]) + 3
        ax.text(i, top, sig_map[env], ha="center", fontsize=10, fontweight="bold")
    ax.axhline(49, color="green", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=10, rotation=30, ha="right")
    ax.set_ylabel("Mean Progress (m)", fontsize=10)
    ax.set_title("Mean Progress", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 60)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(fontsize=8, loc="upper left")
    ax.tick_params(axis="y", labelsize=9)

    # Panel 2: Fall heatmap (22100)
    ax = fig.add_subplot(gs[1])
    fall22 = fall_matrix(stats, "22100")
    fallfl = fall_matrix(stats, "flat")
    vmax_fall = max(fall22.max(), fallfl.max(), 1)
    T = fall22.T[::-1]
    im = ax.imshow(T, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax_fall)
    for i in range(5):
        for j in range(len(ENVIRONMENTS)):
            v = int(T[i, j])
            tc = "white" if v > vmax_fall * 0.55 else "black"
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=10, fontweight="bold", color=tc)
    ax.set_xticks(np.arange(len(ENVIRONMENTS)))
    ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS],
                       fontsize=10, rotation=30, ha="right")
    ax.set_yticks(np.arange(5))
    ax.set_yticklabels(["Z5", "Z4", "Z3", "Z2", "Z1"], fontsize=9)
    ax.set_title("22100 — Fall Locations", fontsize=12, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
    cbar.set_label("Falls", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Panel 3: Fall Rate
    ax = fig.add_subplot(gs[2])
    for j, policy in enumerate(POLICIES):
        fr = [stats[policy][e]["fall_rate"] * 100 for e in ENVIRONMENTS]
        offset = (j - 0.5) * width
        bars = ax.bar(x + offset, fr, width,
                      color=POLICY_COLORS[policy], alpha=0.88, zorder=3,
                      label=POLICY_LABELS[policy])
        for bar, v in zip(bars, fr):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold")
    ax.axhline(50, color="red", linestyle="--", linewidth=1.0, alpha=0.55, label="50%")
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS],
                       fontsize=10, rotation=30, ha="right")
    ax.set_ylabel("Fall Rate (%)", fontsize=10)
    ax.set_title("Fall Rate", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(axis="y", labelsize=9)

    # Panel 4: Completion Rate
    ax = fig.add_subplot(gs[3])
    for j, policy in enumerate(POLICIES):
        cr = [stats[policy][e]["completion_rate"] * 100 for e in ENVIRONMENTS]
        offset = (j - 0.5) * width
        bars = ax.bar(x + offset, cr, width,
                      color=POLICY_COLORS[policy], alpha=0.88, zorder=3,
                      label=POLICY_LABELS[policy])
        for bar, v in zip(bars, cr):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS],
                       fontsize=10, rotation=30, ha="right")
    ax.set_ylabel("Completion Rate (%)", fontsize=10)
    ax.set_title("Course Completion (≥49 m)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(axis="y", labelsize=9)

    fig.text(0.5, 0.025,
             "Significance markers (Welch's t on progress): *** p<0.001  ** p<0.01  * p<0.05  ns p≥0.05.  "
             "100 episodes per policy / environment.",
             ha="center", va="center", fontsize=9.5,
             fontstyle="italic", color="#444")
    save(fig, "composite_4panel.png")


def plot_headline_panel(stats):
    """Single hero figure for the poster — biggest delta first."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white",
                             gridspec_kw={"width_ratios": [1.1, 1]})

    # Left: completion-rate bullet
    ax = axes[0]
    x = np.arange(len(ENVIRONMENTS))
    width = 0.38
    for j, policy in enumerate(POLICIES):
        cr = [stats[policy][e]["completion_rate"] * 100 for e in ENVIRONMENTS]
        offset = (j - 0.5) * width
        bars = ax.bar(x + offset, cr, width, color=POLICY_COLORS[policy],
                      alpha=0.9, zorder=3, label=POLICY_LABELS[policy])
        for bar, v in zip(bars, cr):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{v:.0f}%", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=11)
    ax.set_ylabel("Completion Rate (%)", fontsize=12)
    ax.set_title("Course Completion (≥49 m)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(fontsize=10, loc="upper right")

    # Right: progress with error bars
    ax = axes[1]
    for j, policy in enumerate(POLICIES):
        means = [stats[policy][e]["mean_progress"] for e in ENVIRONMENTS]
        stds = [stats[policy][e]["std_progress"] for e in ENVIRONMENTS]
        offset = (j - 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4,
                      color=POLICY_COLORS[policy], alpha=0.9, zorder=3,
                      label=POLICY_LABELS[policy], error_kw={"elinewidth": 1.0})
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{m:.1f}", ha="center", va="bottom", fontsize=10,
                    fontweight="bold")
    ax.axhline(49, color="green", linestyle="--", linewidth=1, alpha=0.6,
               label="Goal (49 m)")
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=11)
    ax.set_ylabel("Mean Forward Progress (m)", fontsize=12)
    ax.set_title("Forward Progress (mean ± SD)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 60)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(fontsize=10, loc="upper right")

    fig.suptitle(
        "Headline: 22100 lifts Friction completion 0%→96%, Grass 0%→75%",
        fontsize=15, fontweight="bold", y=0.995,
    )
    fig.tight_layout()
    save(fig, "headline_panel.png")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading episodes…")
    eps = load_episodes()
    for p in POLICIES:
        for e in ENVIRONMENTS:
            n = len(eps[p][e])
            assert n == 100, f"expected 100 ep, got {n} for {p}/{e}"
    stats = summarize(eps)

    print("Computing statistics…")
    sig_rows = stat_table(stats)

    print("Writing CSVs…")
    write_summary_csv(stats, OUT_DIR / "summary.csv")
    write_stats_csv(sig_rows, OUT_DIR / "statistical_tests.csv")

    print("Rendering plots…")
    plot_completion_rates(stats, sig_rows)
    plot_fall_rates(stats, sig_rows)
    plot_mean_progress(stats, sig_rows)
    plot_progress_boxplot(stats)
    plot_mean_velocity(stats)
    plot_zone_distribution(stats)
    plot_fall_heatmap(stats)
    plot_stall_heatmap(stats)
    plot_stability_by_zone(stats)
    plot_outcome_pies(stats)
    plot_composite_4panel(stats, sig_rows)
    plot_headline_panel(stats)

    print("\n" + "=" * 78)
    print(f"  22100 (Final Capstone Policy) vs Flat — 100-EPISODE COMPARISON")
    print("=" * 78)
    header = (
        f"  {'Env':<10} {'Policy':<8} {'Compl':>7} {'Fall':>6} {'Stall':>6} "
        f"{'Prog (mean±SD)':>20} {'Vel':>6} {'Zone':>5}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for env in ENVIRONMENTS:
        for policy in POLICIES:
            s = stats[policy][env]
            print(
                f"  {ENV_LABELS[env]:<10} {POLICY_LABELS[policy]:<8} "
                f"{s['completion_rate'] * 100:>6.0f}% {s['fall_rate'] * 100:>5.0f}% "
                f"{s['stall_rate'] * 100:>5.0f}% "
                f"{s['mean_progress']:>7.1f}±{s['std_progress']:>5.1f} m   "
                f"{s['mean_velocity']:>5.2f} {s['mean_zone']:>5.1f}"
            )
    print("=" * 78)
    print("  Significance (Welch's t on progress, two-prop-z on completion + fall):")
    print("  " + "-" * 76)
    print(f"  {'Env':<10} {'dProg (m)':>10} {'t':>7} {'p-progress':>12} "
          f"{'d':>7} {'dCompl':>9} {'p-compl':>10} {'dFall':>9} {'p-fall':>10}")
    for r in sig_rows:
        env = r["environment"]
        d = r["progress_cohens_d"]
        winner = "22100" if d > 0 else "flat"
        print(
            f"  {ENV_LABELS[env]:<10} {r['progress_22100_minus_flat']:>+10.2f} "
            f"{r['progress_t_stat']:>+7.2f} {r['progress_p_value']:>12.2e} "
            f"{d:>+7.2f} {r['completion_22100_minus_flat']*100:>+8.0f}% "
            f"{r['completion_p_value']:>10.2e} "
            f"{r['fall_22100_minus_flat']*100:>+8.0f}% "
            f"{r['fall_p_value']:>10.2e}  ({sig_label(r['progress_p_value'])} -> {winner})"
        )
    print("=" * 78)
    print(f"\n  Outputs in: {OUT_DIR}")
    print("  Done.")


if __name__ == "__main__":
    main()
