"""
Cole Nav Eval — No Policy vs MPPI Policy on Flat Terrain.

Mirrors the toolkit and poster style used in
`Experiments/Ryan/Final_Capstone_Policy_22100/eval_100ep/visualize_22100_vs_flat.py`
so this slide drops cleanly into the capstone deliverable.

Inputs (this dir):
  - flat_no_policy_episodes.csv     (100 ep, no nav policy)
  - flat_mppi_policy_episodes.csv   (100 ep, MPPI nav policy)

Outputs (plots_nav_no_policy_vs_mppi/):
  - summary.csv, statistical_tests.csv
  - completion_rates.png, fall_rates.png, mean_waypoints.png
  - waypoints_boxplot.png
  - outcome_pies.png, segment_distribution.png
  - composite_4panel.png  (PPT widescreen 13.33 x 7.5)
  - headline_panel.png
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sstats

# Paths
HERE = Path(__file__).parent.resolve()
OUT_DIR = HERE / "plots_nav_no_policy_vs_mppi"
OUT_DIR.mkdir(exist_ok=True)

# Single environment, two policies (mirrors POLICY_COLORS scheme from 22100 plot).
ENVIRONMENTS = ["flat"]
ENV_LABELS = {"flat": "Flat"}

POLICIES = ["no_policy", "mppi"]
POLICY_LABELS = {"no_policy": "No Policy", "mppi": "MPPI Policy"}
POLICY_COLORS = {"no_policy": "#2196F3", "mppi": "#FF5722"}

WAYPOINT_GOAL = 25
NUM_SEGMENTS = 5  # 5 segments of 5 waypoints (mirrors 5-zone poster style)


# Loader
def load_csv(path: Path):
    """Read the per-episode rows from one of Cole's CSVs (skips header block)."""
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        in_data = False
        for r in reader:
            if not r or r[0].startswith("#"):
                continue
            if r[0] == "Episode":
                in_data = True
                continue
            if not in_data:
                continue
            ep, wp, reason, score = r[0], r[1], r[2], r[3]
            rows.append({
                "episode": int(ep),
                "waypoints": int(wp),
                "reason": reason.strip(),
                "score": float(score),
            })
    return rows


def load_all():
    eps = {
        "no_policy": load_csv(HERE / "flat_no_policy_episodes.csv"),
        "mppi": load_csv(HERE / "flat_mppi_policy_episodes.csv"),
    }
    return eps


def summarize(eps):
    out = {p: {} for p in POLICIES}
    for policy in POLICIES:
        arr = eps[policy]
        wps = np.array([e["waypoints"] for e in arr])
        scores = np.array([e["score"] for e in arr])
        falls = np.array([1 if e["reason"] == "Fell Over" else 0 for e in arr])
        completions = np.array(
            [1 if e["reason"] == "Completed All Waypoints" else 0 for e in arr]
        )
        stalls = np.array(
            [1 if e["reason"] == "Ran Out of Points" else 0 for e in arr]
        )
        # Bin highest waypoint reached into 5 segments of 5 waypoints
        # Segment 1 = [0,5), Segment 2 = [5,10), ..., Segment 5 = [20,25].
        seg = np.clip(wps // 5 + (wps == WAYPOINT_GOAL).astype(int) * 0, 0, 4)
        # Special case: 25 waypoints belongs to segment 5 (index 4)
        seg = np.where(wps >= 20, 4, np.clip(wps // 5, 0, 3))
        seg_counts = [int(np.sum(seg == s)) for s in range(NUM_SEGMENTS)]
        out[policy]["flat"] = dict(
            n=len(arr),
            completion_rate=float(completions.mean()),
            fall_rate=float(falls.mean()),
            stall_rate=float(stalls.mean()),
            mean_waypoints=float(wps.mean()),
            std_waypoints=float(wps.std(ddof=1)) if len(wps) > 1 else 0.0,
            median_waypoints=float(np.median(wps)),
            max_waypoints=int(wps.max()),
            mean_score=float(scores.mean()),
            mean_completed_score=float(scores[completions == 1].mean())
                if completions.sum() > 0 else 0.0,
            seg_counts=seg_counts,
            waypoints=wps,
            scores=scores,
            falls=falls,
            completions=completions,
            stalls=stalls,
            eps=arr,
        )
    return out


# Stats
def cohens_d(a, b):
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
    rows = []
    for env in ENVIRONMENTS:
        a = stats["mppi"][env]["waypoints"]
        b = stats["no_policy"][env]["waypoints"]
        t_stat, t_p = sstats.ttest_ind(a, b, equal_var=False)
        d = cohens_d(a, b)  # positive => MPPI > no_policy
        n = stats["no_policy"][env]["n"]
        cp_m = stats["mppi"][env]["completion_rate"]
        cp_n = stats["no_policy"][env]["completion_rate"]
        z_c, p_c = two_proportion_z(cp_m, n, cp_n, n)
        fp_m = stats["mppi"][env]["fall_rate"]
        fp_n = stats["no_policy"][env]["fall_rate"]
        z_f, p_f = two_proportion_z(fp_m, n, fp_n, n)
        rows.append(dict(
            environment=env,
            waypoints_t_stat=float(t_stat),
            waypoints_p_value=float(t_p),
            waypoints_cohens_d=d,
            completion_z_stat=z_c,
            completion_p_value=p_c,
            fall_z_stat=z_f,
            fall_p_value=p_f,
            waypoints_mppi_minus_nopolicy=float(a.mean() - b.mean()),
            completion_mppi_minus_nopolicy=float(cp_m - cp_n),
            fall_mppi_minus_nopolicy=float(fp_m - fp_n),
        ))
    return rows


def sig_label(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# CSV writers
def write_summary_csv(stats, path):
    fields = [
        "environment", "policy", "num_episodes",
        "completion_rate", "fall_rate", "stall_rate",
        "mean_waypoints", "std_waypoints", "median_waypoints", "max_waypoints",
        "mean_final_score", "mean_completed_score",
        "seg_1_count", "seg_2_count", "seg_3_count", "seg_4_count", "seg_5_count",
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
                    "mean_waypoints": round(s["mean_waypoints"], 3),
                    "std_waypoints": round(s["std_waypoints"], 3),
                    "median_waypoints": round(s["median_waypoints"], 3),
                    "max_waypoints": s["max_waypoints"],
                    "mean_final_score": round(s["mean_score"], 3),
                    "mean_completed_score": round(s["mean_completed_score"], 3),
                }
                for i, c in enumerate(s["seg_counts"], start=1):
                    row[f"seg_{i}_count"] = c
                w.writerow(row)


def write_stats_csv(rows, path):
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in r.items()})


# Plot helpers
def save(fig, name):
    p = OUT_DIR / name
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved {name}")


def grouped_bar_single_env(ax, vals, errs=None, ylabel="", title="",
                            ymax=None, fmt="{:.1f}", suffix=""):
    """Single env, two-policy side-by-side bars (mirrors 22100 grouped bar)."""
    x = np.arange(len(POLICIES))
    width = 0.55
    bars_all = []
    for i, policy in enumerate(POLICIES):
        v = vals[policy]
        e = errs[policy] if errs else None
        bar = ax.bar(
            x[i], v, width, color=POLICY_COLORS[policy], alpha=0.88, zorder=3,
            label=POLICY_LABELS[policy],
            yerr=[e] if e is not None else None,
            capsize=5, error_kw={"elinewidth": 1.2},
        )
        bars_all.append((bar, v))
        ax.text(
            x[i], v + (ymax * 0.02 if ymax else 0.03 * max(v, 1)),
            fmt.format(v) + suffix,
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in POLICIES], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    if ymax is not None:
        ax.set_ylim(0, ymax)


# Plots
def plot_completion_rates(stats, sig_rows):
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="white")
    vals = {p: stats[p]["flat"]["completion_rate"] * 100 for p in POLICIES}
    grouped_bar_single_env(
        ax, vals,
        ylabel="Completion Rate (%)",
        title="Course Completion Rate — Flat (n=100/policy)",
        ymax=110, fmt="{:.0f}", suffix="%",
    )
    sig = sig_label(sig_rows[0]["completion_p_value"])
    top = max(vals.values()) + 8
    ax.text(0.5, top, sig, ha="center", fontsize=13, fontweight="bold", color="#444")
    ax.plot([0, 1], [top - 2, top - 2], "k-", linewidth=1)
    fig.tight_layout()
    save(fig, "completion_rates.png")


def plot_fall_rates(stats, sig_rows):
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="white")
    vals = {p: stats[p]["flat"]["fall_rate"] * 100 for p in POLICIES}
    grouped_bar_single_env(
        ax, vals,
        ylabel="Fall Rate (%)",
        title="Fall Rate — Flat (n=100/policy)",
        ymax=60, fmt="{:.0f}", suffix="%",
    )
    ax.axhline(50, color="red", linestyle="--", linewidth=1,
               alpha=0.5, label="50% threshold")
    sig = sig_label(sig_rows[0]["fall_p_value"])
    top = max(vals.values()) + 8
    ax.text(0.5, top, sig, ha="center", fontsize=13, fontweight="bold", color="#444")
    ax.plot([0, 1], [top - 1.5, top - 1.5], "k-", linewidth=1)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    save(fig, "fall_rates.png")


def plot_mean_waypoints(stats, sig_rows):
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="white")
    vals = {p: stats[p]["flat"]["mean_waypoints"] for p in POLICIES}
    errs = {p: stats[p]["flat"]["std_waypoints"] for p in POLICIES}
    grouped_bar_single_env(
        ax, vals, errs=errs,
        ylabel="Mean Waypoints Reached",
        title="Mean Waypoints Reached — Flat (n=100/policy)",
        ymax=30, fmt="{:.2f}",
    )
    ax.axhline(WAYPOINT_GOAL, color="green", linestyle="--", linewidth=1,
               alpha=0.55, label=f"Course goal ({WAYPOINT_GOAL} WPs)")
    sig = sig_label(sig_rows[0]["waypoints_p_value"])
    top = max(vals[p] + errs[p] for p in POLICIES) + 3
    ax.text(0.5, top, sig, ha="center", fontsize=13, fontweight="bold", color="#444")
    ax.plot([0, 1], [top - 0.5, top - 0.5], "k-", linewidth=1)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    save(fig, "mean_waypoints.png")


def plot_waypoints_boxplot(stats):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    data = [stats[p]["flat"]["waypoints"] for p in POLICIES]
    colors = [POLICY_COLORS[p] for p in POLICIES]
    bp = ax.boxplot(
        data, positions=[0, 1], widths=0.5, patch_artist=True,
        medianprops={"color": "black", "linewidth": 2.0},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.78)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([POLICY_LABELS[p] for p in POLICIES], fontsize=12)
    ax.set_ylabel("Waypoints Reached per Episode", fontsize=12)
    ax.set_title("Per-Episode Waypoint Distribution — Flat",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.axhline(WAYPOINT_GOAL, color="green", linestyle="--", linewidth=1,
               alpha=0.55, label=f"Course goal ({WAYPOINT_GOAL} WPs)")
    ax.set_ylim(-1, 28)
    handles = [plt.Rectangle((0, 0), 1, 1, color=POLICY_COLORS[p], alpha=0.78,
                             label=POLICY_LABELS[p]) for p in POLICIES]
    handles.append(plt.Line2D([0], [0], color="green", linestyle="--",
                              label=f"Course goal ({WAYPOINT_GOAL} WPs)"))
    ax.legend(handles=handles, fontsize=10, loc="upper left")
    fig.tight_layout()
    save(fig, "waypoints_boxplot.png")


def plot_outcome_pies(stats):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), facecolor="white")
    for ax, policy in zip(axes, POLICIES):
        s = stats[policy]["flat"]
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
            labels.append(f"Ran Out of Points\n({n_stall}/{n})")
            colors.append("#FF9800")
            explode.append(0.04)
        if n_complete:
            sizes.append(n_complete)
            labels.append(f"Completed\n({n_complete}/{n})")
            colors.append("#4CAF50")
            explode.append(0.04)
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, explode=explode,
            autopct="%1.0f%%", startangle=90,
            textprops={"fontsize": 11},
            wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
        )
        for t in autotexts:
            t.set_fontweight("bold")
            t.set_fontsize(12)
        ax.set_title(f"Flat — {POLICY_LABELS[policy]}",
                     fontsize=13, fontweight="bold")
    fig.suptitle("Episode Outcomes — No Policy vs MPPI (100 ep each)",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout()
    save(fig, "outcome_pies.png")


def plot_segment_distribution(stats):
    """Stacked bars of highest waypoint segment reached. Mirrors zone_distribution.png."""
    seg_colors = ["#EF5350", "#FF7043", "#FFA726", "#66BB6A", "#42A5F5"]
    seg_labels = ["WP 0-4", "WP 5-9", "WP 10-14", "WP 15-19", "WP 20-25"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             facecolor="white", sharey=True)
    for ax, policy in zip(axes, POLICIES):
        s = stats[policy]["flat"]
        bottom = 0
        for i, (cnt, lbl) in enumerate(zip(s["seg_counts"], seg_labels)):
            ax.bar(0, cnt, bottom=bottom, color=seg_colors[i], alpha=0.92,
                   label=lbl, zorder=3, width=0.6)
            if cnt >= 4:
                ax.text(0, bottom + cnt / 2, str(cnt),
                        ha="center", va="center",
                        fontsize=11, color="white", fontweight="bold")
            bottom += cnt
        ax.set_xticks([0])
        ax.set_xticklabels([POLICY_LABELS[policy]], fontsize=12)
        ax.set_title(POLICY_LABELS[policy], fontsize=13, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("Number of Episodes", fontsize=12)
    axes[1].legend(fontsize=9, loc="lower right",
                   title="Highest waypoint segment")
    fig.suptitle("Highest Waypoint Segment Reached — No Policy vs MPPI",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "segment_distribution.png")


def plot_composite_4panel(stats, sig_rows):
    """PPT-widescreen 13.33 x 7.5 — Mean WPs | Outcome stack | Fall Rate | Completion."""
    fig = plt.figure(figsize=(13.33, 7.5), facecolor="white")
    fig.suptitle(
        "Cole Nav Eval — MPPI Policy vs No Policy (Flat Terrain, 100 ep each)",
        fontsize=18, fontweight="bold", y=0.97,
    )
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1.1, 1, 1], wspace=0.50,
                          left=0.06, right=0.97, top=0.85, bottom=0.13)
    x = np.arange(len(POLICIES))
    width = 0.55
    sig = sig_rows[0]

    # Panel 1: Mean Waypoints
    ax = fig.add_subplot(gs[0])
    means = [stats[p]["flat"]["mean_waypoints"] for p in POLICIES]
    stds = [stats[p]["flat"]["std_waypoints"] for p in POLICIES]
    for i, p in enumerate(POLICIES):
        ax.bar(x[i], means[i], width, yerr=stds[i], capsize=4,
               color=POLICY_COLORS[p], alpha=0.9, zorder=3,
               label=POLICY_LABELS[p], error_kw={"elinewidth": 1.0})
        ax.text(x[i], means[i] + 1.5, f"{means[i]:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    top = max(m + s for m, s in zip(means, stds)) + 3
    ax.text(0.5, top, sig_label(sig["waypoints_p_value"]),
            ha="center", fontsize=11, fontweight="bold")
    ax.plot([0, 1], [top - 0.5, top - 0.5], "k-", linewidth=1)
    ax.axhline(WAYPOINT_GOAL, color="green", linestyle="--",
               linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in POLICIES],
                       fontsize=10, rotation=20, ha="right")
    ax.set_ylabel("Mean Waypoints", fontsize=10)
    ax.set_title("Mean Waypoints", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 30)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.tick_params(axis="y", labelsize=9)

    # Panel 2: Outcome stack (per policy)
    ax = fig.add_subplot(gs[1])
    cat_colors = {"complete": "#4CAF50", "stall": "#FF9800", "fall": "#F44336"}
    for i, p in enumerate(POLICIES):
        s = stats[p]["flat"]
        c, st, f = int(s["completions"].sum()), int(s["stalls"].sum()), int(s["falls"].sum())
        bottom = 0
        for cnt, lbl, col in [(c, "Complete", cat_colors["complete"]),
                              (st, "Ran Out", cat_colors["stall"]),
                              (f, "Fell Over", cat_colors["fall"])]:
            ax.bar(x[i], cnt, width, bottom=bottom, color=col, alpha=0.92,
                   zorder=3, label=lbl if i == 0 else None)
            if cnt >= 6:
                ax.text(x[i], bottom + cnt / 2, str(cnt),
                        ha="center", va="center", fontsize=10,
                        color="white", fontweight="bold")
            bottom += cnt
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in POLICIES],
                       fontsize=10, rotation=20, ha="right")
    ax.set_ylabel("Episodes (out of 100)", fontsize=10)
    ax.set_title("Episode Outcomes", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(fontsize=8, loc="upper left")
    ax.tick_params(axis="y", labelsize=9)

    # Panel 3: Fall Rate
    ax = fig.add_subplot(gs[2])
    fr = [stats[p]["flat"]["fall_rate"] * 100 for p in POLICIES]
    for i, p in enumerate(POLICIES):
        ax.bar(x[i], fr[i], width, color=POLICY_COLORS[p], alpha=0.9, zorder=3,
               label=POLICY_LABELS[p])
        ax.text(x[i], fr[i] + 1.5, f"{fr[i]:.0f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    top = max(fr) + 8
    ax.text(0.5, top, sig_label(sig["fall_p_value"]),
            ha="center", fontsize=11, fontweight="bold")
    ax.plot([0, 1], [top - 1.5, top - 1.5], "k-", linewidth=1)
    ax.axhline(50, color="red", linestyle="--", linewidth=1.0, alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in POLICIES],
                       fontsize=10, rotation=20, ha="right")
    ax.set_ylabel("Fall Rate (%)", fontsize=10)
    ax.set_title("Fall Rate", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 60)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.tick_params(axis="y", labelsize=9)

    # Panel 4: Completion Rate
    ax = fig.add_subplot(gs[3])
    cr = [stats[p]["flat"]["completion_rate"] * 100 for p in POLICIES]
    for i, p in enumerate(POLICIES):
        ax.bar(x[i], cr[i], width, color=POLICY_COLORS[p], alpha=0.9, zorder=3,
               label=POLICY_LABELS[p])
        ax.text(x[i], cr[i] + 1.5, f"{cr[i]:.0f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    top = max(cr) + 8
    ax.text(0.5, top, sig_label(sig["completion_p_value"]),
            ha="center", fontsize=11, fontweight="bold")
    ax.plot([0, 1], [top - 1.5, top - 1.5], "k-", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in POLICIES],
                       fontsize=10, rotation=20, ha="right")
    ax.set_ylabel("Completion Rate (%)", fontsize=10)
    ax.set_title("Course Completion (25 WPs)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.tick_params(axis="y", labelsize=9)

    fig.text(0.5, 0.025,
             "Significance markers: *** p<0.001  ** p<0.01  * p<0.05  ns p>=0.05.  "
             "Welch's t on waypoints, two-prop-z on completion + fall rates. "
             "100 episodes per policy.",
             ha="center", va="center", fontsize=9.5,
             fontstyle="italic", color="#444")
    save(fig, "composite_4panel.png")


def plot_headline_panel(stats, sig_rows):
    """Single hero figure mirroring 22100 headline_panel.png."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white",
                             gridspec_kw={"width_ratios": [1.1, 1]})
    sig = sig_rows[0]

    # Left: completion-rate
    ax = axes[0]
    x = np.arange(len(POLICIES))
    width = 0.55
    cr = [stats[p]["flat"]["completion_rate"] * 100 for p in POLICIES]
    for i, p in enumerate(POLICIES):
        ax.bar(x[i], cr[i], width, color=POLICY_COLORS[p], alpha=0.9,
               zorder=3, label=POLICY_LABELS[p])
        ax.text(x[i], cr[i] + 1.5, f"{cr[i]:.0f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    top = max(cr) + 9
    ax.text(0.5, top, sig_label(sig["completion_p_value"]),
            ha="center", fontsize=13, fontweight="bold")
    ax.plot([0, 1], [top - 1.5, top - 1.5], "k-", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in POLICIES], fontsize=11)
    ax.set_ylabel("Completion Rate (%)", fontsize=12)
    ax.set_title("Course Completion (25 WPs)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(fontsize=10, loc="upper left")

    # Right: mean waypoints with error bars
    ax = axes[1]
    means = [stats[p]["flat"]["mean_waypoints"] for p in POLICIES]
    stds = [stats[p]["flat"]["std_waypoints"] for p in POLICIES]
    for i, p in enumerate(POLICIES):
        ax.bar(x[i], means[i], width, yerr=stds[i], capsize=4,
               color=POLICY_COLORS[p], alpha=0.9, zorder=3,
               label=POLICY_LABELS[p], error_kw={"elinewidth": 1.0})
        ax.text(x[i], means[i] + 1.0, f"{means[i]:.1f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    top = max(m + s for m, s in zip(means, stds)) + 3
    ax.text(0.5, top, sig_label(sig["waypoints_p_value"]),
            ha="center", fontsize=13, fontweight="bold")
    ax.plot([0, 1], [top - 0.5, top - 0.5], "k-", linewidth=1)
    ax.axhline(WAYPOINT_GOAL, color="green", linestyle="--",
               linewidth=1, alpha=0.6, label=f"Goal ({WAYPOINT_GOAL} WPs)")
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in POLICIES], fontsize=11)
    ax.set_ylabel("Mean Waypoints (mean +/- SD)", fontsize=12)
    ax.set_title("Waypoints Reached (mean +/- SD)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 30)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(fontsize=10, loc="upper left")

    fig.suptitle(
        "Headline: MPPI lifts Flat completion 0% -> 52%, mean WPs 0.65 -> 19.52",
        fontsize=15, fontweight="bold", y=0.995,
    )
    fig.tight_layout()
    save(fig, "headline_panel.png")


# Main
def main():
    print("Loading episodes...")
    eps = load_all()
    for p in POLICIES:
        n = len(eps[p])
        assert n == 100, f"expected 100 ep, got {n} for {p}"
    stats = summarize(eps)

    print("Computing statistics...")
    sig_rows = stat_table(stats)

    print("Writing CSVs...")
    write_summary_csv(stats, OUT_DIR / "summary.csv")
    write_stats_csv(sig_rows, OUT_DIR / "statistical_tests.csv")

    print("Rendering plots...")
    plot_completion_rates(stats, sig_rows)
    plot_fall_rates(stats, sig_rows)
    plot_mean_waypoints(stats, sig_rows)
    plot_waypoints_boxplot(stats)
    plot_outcome_pies(stats)
    plot_segment_distribution(stats)
    plot_composite_4panel(stats, sig_rows)
    plot_headline_panel(stats, sig_rows)

    # Console table
    print("\n" + "=" * 78)
    print("  Cole Nav Eval — MPPI vs No Policy (Flat, 100 ep each)")
    print("=" * 78)
    header = (f"  {'Policy':<12} {'Compl':>7} {'Fall':>6} {'ROOP':>6} "
              f"{'WPs (mean+/-SD)':>18} {'Median':>7} {'Max':>5}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for p in POLICIES:
        s = stats[p]["flat"]
        print(
            f"  {POLICY_LABELS[p]:<12} "
            f"{s['completion_rate'] * 100:>6.0f}% "
            f"{s['fall_rate'] * 100:>5.0f}% "
            f"{s['stall_rate'] * 100:>5.0f}% "
            f"{s['mean_waypoints']:>6.2f}+/-{s['std_waypoints']:>5.2f}   "
            f"{s['median_waypoints']:>6.1f} {s['max_waypoints']:>5d}"
        )
    print("=" * 78)
    print("  Significance:")
    r = sig_rows[0]
    d = r["waypoints_cohens_d"]
    winner = "MPPI" if d > 0 else "No Policy"
    print(f"  Waypoints  Welch's t = {r['waypoints_t_stat']:+.2f}, "
          f"p = {r['waypoints_p_value']:.2e}, d = {d:+.2f}  "
          f"({sig_label(r['waypoints_p_value'])} -> {winner})")
    print(f"  Completion 2-prop z = {r['completion_z_stat']:+.2f}, "
          f"p = {r['completion_p_value']:.2e}, "
          f"diff = {r['completion_mppi_minus_nopolicy']*100:+.0f}%  "
          f"({sig_label(r['completion_p_value'])})")
    print(f"  Fall rate  2-prop z = {r['fall_z_stat']:+.2f}, "
          f"p = {r['fall_p_value']:.2e}, "
          f"diff = {r['fall_mppi_minus_nopolicy']*100:+.0f}%  "
          f"({sig_label(r['fall_p_value'])})")
    print("=" * 78)
    print(f"\n  Outputs in: {OUT_DIR}")
    print("  Done.")


if __name__ == "__main__":
    main()
