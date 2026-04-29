"""
Visualization script for 4-environment debug test results.
Run: python visualize_results.py
Outputs saved to: report/plots/
"""

import json
import os
import csv
import math
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for all systems
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(BASE_DIR, "report")
PLOTS_DIR  = os.path.join(REPORT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

ENVIRONMENTS = ["boulder", "friction", "grass", "stairs"]
POLICIES     = ["flat", "rough"]
COLORS       = {"flat": "#2196F3", "rough": "#FF5722"}
ENV_LABELS   = {"boulder": "Boulder", "friction": "Friction", "grass": "Grass", "stairs": "Stairs"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_jsonl(env, policy):
    path = os.path.join(BASE_DIR, f"{env}_{policy}_episodes.jsonl")
    records = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records

def load_csv(filename):
    path = os.path.join(REPORT_DIR, filename)
    rows = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows

def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: report/plots/{name}")

# ── Load all episode data ─────────────────────────────────────────────────────
all_episodes = {}
for env in ENVIRONMENTS:
    all_episodes[env] = {}
    for pol in POLICIES:
        all_episodes[env][pol] = load_jsonl(env, pol)

summary_rows = load_csv("summary.csv")
stats_rows   = load_csv("statistical_tests.csv")

# Build quick lookup from summary
summary = {}
for row in summary_rows:
    key = (row["environment"], row["policy"])
    summary[key] = {k: float(v) if v not in ("", "nan") else float("nan")
                    for k, v in row.items() if k not in ("environment", "policy")}

# ── FIGURE 1: Mean Progress (grouped bar) ─────────────────────────────────────
print("Generating Figure 1: Mean Progress...")
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(ENVIRONMENTS))
width = 0.35
for i, pol in enumerate(POLICIES):
    means = [summary.get((env, pol), {}).get("mean_progress", 0) for env in ENVIRONMENTS]
    stds  = [summary.get((env, pol), {}).get("std_progress",  0) for env in ENVIRONMENTS]
    bars = ax.bar(x + (i - 0.5) * width, means, width,
                  label=f"{pol.capitalize()} Policy",
                  color=COLORS[pol], alpha=0.85, zorder=3,
                  yerr=stds, capsize=4, error_kw={"elinewidth": 1.2})
ax.set_xticks(x)
ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=11)
ax.set_ylabel("Mean Progress (m)", fontsize=11)
ax.set_title("Mean Episode Progress by Environment & Policy", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0, max(summary.get((e, p), {}).get("mean_progress", 0)
                   for e in ENVIRONMENTS for p in POLICIES) * 1.25)
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
ax.axhline(0, color="black", linewidth=0.8)
fig.tight_layout()
save(fig, "fig1_mean_progress.png")

# ── FIGURE 2: Per-Episode Progress Box Plot ───────────────────────────────────
print("Generating Figure 2: Per-Episode Progress Distribution...")
fig, ax = plt.subplots(figsize=(11, 5))
positions = []
data_list = []
tick_pos  = []
tick_lbl  = []
col_list  = []
gap = 0.45
group_width = len(POLICIES) * gap
spacing = group_width + 0.6
pos = 1.0
for i, env in enumerate(ENVIRONMENTS):
    center = pos + (len(POLICIES) - 1) * gap / 2
    tick_pos.append(center)
    tick_lbl.append(ENV_LABELS[env])
    for j, pol in enumerate(POLICIES):
        eps = all_episodes[env][pol]
        data_list.append([e["progress"] for e in eps])
        positions.append(pos)
        col_list.append(COLORS[pol])
        pos += gap
    pos += 0.6

bp = ax.boxplot(data_list, positions=positions, widths=0.32, patch_artist=True,
                medianprops={"color": "black", "linewidth": 2},
                whiskerprops={"linewidth": 1.2}, capprops={"linewidth": 1.2})
for patch, col in zip(bp["boxes"], col_list):
    patch.set_facecolor(col)
    patch.set_alpha(0.8)

ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lbl, fontsize=11)
ax.set_ylabel("Progress per Episode (m)", fontsize=11)
ax.set_title("Episode Progress Distribution — Flat vs Rough Policy", fontsize=13, fontweight="bold")
patches = [mpatches.Patch(color=COLORS[p], label=f"{p.capitalize()} Policy") for p in POLICIES]
ax.legend(handles=patches, fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.tight_layout()
save(fig, "fig2_progress_boxplot.png")

# ── FIGURE 3: Mean Stability Score ───────────────────────────────────────────
print("Generating Figure 3: Stability Score...")
fig, ax = plt.subplots(figsize=(9, 5))
for i, pol in enumerate(POLICIES):
    means = [summary.get((env, pol), {}).get("mean_stability", 0) for env in ENVIRONMENTS]
    stds  = [summary.get((env, pol), {}).get("std_stability",  0) for env in ENVIRONMENTS]
    ax.bar(x + (i - 0.5) * width, means, width,
           label=f"{pol.capitalize()} Policy",
           color=COLORS[pol], alpha=0.85, zorder=3,
           yerr=stds, capsize=4, error_kw={"elinewidth": 1.2})
ax.set_xticks(x)
ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=11)
ax.set_ylabel("Mean Stability Score", fontsize=11)
ax.set_title("Mean Stability Score by Environment & Policy", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
fig.tight_layout()
save(fig, "fig3_stability_score.png")

# ── FIGURE 4: Mean Velocity ───────────────────────────────────────────────────
print("Generating Figure 4: Mean Velocity...")
fig, ax = plt.subplots(figsize=(9, 5))
for i, pol in enumerate(POLICIES):
    means = [summary.get((env, pol), {}).get("mean_velocity", 0) for env in ENVIRONMENTS]
    ax.bar(x + (i - 0.5) * width, means, width,
           label=f"{pol.capitalize()} Policy",
           color=COLORS[pol], alpha=0.85, zorder=3)
ax.axhline(2.235, color="red", linestyle="--", linewidth=1.2, label="Spot Max Speed (2.235 m/s)")
ax.set_xticks(x)
ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=11)
ax.set_ylabel("Mean Velocity (m/s)", fontsize=11)
ax.set_title("Mean Velocity by Environment & Policy", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0, 2.6)
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
fig.tight_layout()
save(fig, "fig4_mean_velocity.png")

# ── FIGURE 5: Zone Reached Distribution (stacked bar) ─────────────────────────
print("Generating Figure 5: Zone Distribution...")
zone_cols = ["zone_1_count", "zone_2_count", "zone_3_count", "zone_4_count", "zone_5_count"]
zone_colors = ["#EF5350", "#FF7043", "#FFA726", "#66BB6A", "#42A5F5"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, pol in zip(axes, POLICIES):
    bottom = np.zeros(len(ENVIRONMENTS))
    for z, (zcol, zcolor) in enumerate(zip(zone_cols, zone_colors)):
        vals = np.array([summary.get((env, pol), {}).get(zcol, 0) for env in ENVIRONMENTS])
        ax.bar(ENVIRONMENTS, vals, bottom=bottom, color=zcolor, label=f"Zone {z+1}", alpha=0.9)
        bottom += vals
    ax.set_title(f"{pol.capitalize()} Policy — Zone Reached", fontsize=12, fontweight="bold")
    ax.set_ylabel("Episode Count", fontsize=10)
    ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
axes[0].legend(title="Zone", fontsize=9, title_fontsize=9, loc="upper right")
fig.suptitle("Zone Reached Distribution per Policy", fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "fig5_zone_distribution.png")

# ── FIGURE 6: Statistical Test Results ────────────────────────────────────────
print("Generating Figure 6: Statistical Tests...")
stats_lookup = {row["environment"]: row for row in stats_rows}
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Subplot A: Cohen's d (effect size)
ax = axes[0]
cohens_d = [float(stats_lookup.get(env, {}).get("progress_cohens_d", 0) or 0)
            for env in ENVIRONMENTS]
bar_colors = ["#EF5350" if d < 0 else "#66BB6A" for d in cohens_d]
bars = ax.bar([ENV_LABELS[e] for e in ENVIRONMENTS], cohens_d, color=bar_colors, alpha=0.85, zorder=3)
ax.axhline(0, color="black", linewidth=0.8)
ax.axhline( 0.8, color="green",  linestyle="--", linewidth=1, alpha=0.6, label="Large effect (+0.8)")
ax.axhline(-0.8, color="red",    linestyle="--", linewidth=1, alpha=0.6, label="Large effect (-0.8)")
ax.axhline( 0.2, color="gray",   linestyle=":",  linewidth=1, alpha=0.5, label="Small effect (±0.2)")
ax.axhline(-0.2, color="gray",   linestyle=":",  linewidth=1, alpha=0.5)
ax.set_ylabel("Cohen's d (Flat - Rough)", fontsize=10)
ax.set_title("Effect Size: Flat vs Rough Progress\n(Positive = Flat better)", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

# Subplot B: p-values with significance threshold
ax = axes[1]
p_vals = [float(stats_lookup.get(env, {}).get("progress_p_value", 1) or 1)
          for env in ENVIRONMENTS]
bar_colors_p = ["#42A5F5" if p < 0.05 else "#BDBDBD" for p in p_vals]
ax.bar([ENV_LABELS[e] for e in ENVIRONMENTS], p_vals, color=bar_colors_p, alpha=0.85, zorder=3)
ax.axhline(0.05, color="red", linestyle="--", linewidth=1.5, label="p = 0.05 threshold")
ax.set_ylabel("p-value (t-test)", fontsize=10)
ax.set_title("Statistical Significance of Progress\nFlat vs Rough Policy", fontsize=11, fontweight="bold")
ax.set_ylim(0, max(p_vals) * 1.2)
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
for i, p in enumerate(p_vals):
    label = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    ax.text(i, p + 0.005, label, ha="center", fontsize=10, fontweight="bold",
            color="red" if p < 0.05 else "gray")

fig.suptitle("Statistical Analysis: Policy Comparison", fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "fig6_statistical_tests.png")

# ── FIGURE 7: Summary Dashboard (2×3 grid) ────────────────────────────────────
print("Generating Figure 7: Summary Dashboard...")
fig = plt.figure(figsize=(16, 10))
fig.suptitle("4-Environment Debug Test — Results Dashboard\n(local_debug_2026-02-19_16-01-11)",
             fontsize=14, fontweight="bold", y=0.98)
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

metrics = [
    ("mean_progress",   "Mean Progress (m)",         "Mean Progress",   False),
    ("mean_stability",  "Stability Score",            "Mean Stability",  False),
    ("mean_velocity",   "Mean Velocity (m/s)",        "Mean Velocity",   False),
    ("fall_rate",       "Fall Rate (fraction)",       "Fall Rate",       False),
    ("completion_rate", "Completion Rate (fraction)", "Completion Rate", False),
    ("mean_zone_reached","Mean Zone Reached",         "Mean Zone",       False),
]

for idx, (metric, ylabel, title, _) in enumerate(metrics):
    row, col = divmod(idx, 3)
    ax = fig.add_subplot(gs[row, col])
    for i, pol in enumerate(POLICIES):
        vals = [summary.get((env, pol), {}).get(metric, 0) for env in ENVIRONMENTS]
        ax.bar(x + (i - 0.5) * width, vals, width,
               label=f"{pol.capitalize()}", color=COLORS[pol], alpha=0.85, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([e[:4].capitalize() for e in ENVIRONMENTS], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    if idx == 0:
        ax.legend(fontsize=8, loc="upper right")

save(fig, "fig7_dashboard.png")

# ── FIGURE 8: Stability vs Progress scatter ───────────────────────────────────
print("Generating Figure 8: Stability vs Progress scatter...")
fig, ax = plt.subplots(figsize=(8, 6))
markers = {"boulder": "o", "friction": "s", "grass": "^", "stairs": "D"}
for env in ENVIRONMENTS:
    for pol in POLICIES:
        eps = all_episodes[env][pol]
        xs = [e["progress"]        for e in eps]
        ys = [e["stability_score"] for e in eps]
        ax.scatter(xs, ys, c=COLORS[pol], marker=markers[env], s=60, alpha=0.8,
                   label=f"{ENV_LABELS[env]} / {pol.capitalize()}")
ax.set_xlabel("Progress (m)", fontsize=11)
ax.set_ylabel("Stability Score", fontsize=11)
ax.set_title("Stability vs Progress — All Conditions", fontsize=13, fontweight="bold")
# Deduplicate legend
handles, labels = ax.get_legend_handles_labels()
seen = {}
for h, l in zip(handles, labels):
    if l not in seen:
        seen[l] = h
ax.legend(seen.values(), seen.keys(), fontsize=8, ncol=2)
ax.grid(linestyle="--", alpha=0.4)
fig.tight_layout()
save(fig, "fig8_stability_vs_progress.png")

print("\nAll figures saved to:", PLOTS_DIR)
print("Done.")
