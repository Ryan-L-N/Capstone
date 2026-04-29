"""
Flat Policy Visualization — 100-episode eval across 4 environments.
Generates individual plots + a 3-panel composite "one-shot" figure.

Run:  python visualize_flat_policy.py
Output: report/flat_policy_plots/
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR  = os.path.join(BASE_DIR, "report", "flat_policy_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

ENVIRONMENTS = ["friction", "grass", "boulder", "stairs"]
ENV_LABELS   = {"friction": "Friction", "grass": "Grass", "boulder": "Boulder", "stairs": "Stairs"}
ENV_COLORS   = {"friction": "#2196F3", "grass": "#4CAF50", "boulder": "#FF9800", "stairs": "#F44336"}

# ── Load data ─────────────────────────────────────────────────────────────────
def load_jsonl(env):
    path = os.path.join(BASE_DIR, f"{env}_flat_episodes.jsonl")
    records = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records

episodes = {env: load_jsonl(env) for env in ENVIRONMENTS}

# Quick stats
stats = {}
for env in ENVIRONMENTS:
    eps = episodes[env]
    progress = [e["progress"] for e in eps]
    falls = sum(1 for e in eps if e["fall_detected"])
    stability = [e["stability_score"] for e in eps]
    velocity = [e["mean_velocity"] for e in eps]
    zones = [e["zone_reached"] for e in eps]
    stats[env] = {
        "n": len(eps),
        "mean_progress": np.mean(progress),
        "std_progress": np.std(progress),
        "median_progress": np.median(progress),
        "fall_rate": falls / len(eps) if eps else 0,
        "falls": falls,
        "mean_stability": np.mean(stability),
        "std_stability": np.std(stability),
        "mean_velocity": np.mean(velocity),
        "mean_zone": np.mean(zones),
        "progress": progress,
        "stability": stability,
        "velocity": velocity,
        "zones": zones,
    }

def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")

# ══════════════════════════════════════════════════════════════════════════════
#  INDIVIDUAL PLOTS
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Mean Progress Bar Chart ────────────────────────────────────────────────
print("Generating: Mean Progress...")
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(ENVIRONMENTS))
means = [stats[e]["mean_progress"] for e in ENVIRONMENTS]
stds  = [stats[e]["std_progress"] for e in ENVIRONMENTS]
colors = [ENV_COLORS[e] for e in ENVIRONMENTS]
bars = ax.bar(x, means, color=colors, alpha=0.85, zorder=3,
              yerr=stds, capsize=5, error_kw={"elinewidth": 1.5})
# Value labels on bars
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{m:.1f}m", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=12)
ax.set_ylabel("Mean Progress (m)", fontsize=12)
ax.set_title("Flat Policy — Mean Progress by Environment", fontsize=14, fontweight="bold")
ax.set_ylim(0, max(means) * 1.35)
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
fig.tight_layout()
save(fig, "flat_mean_progress.png")

# ── 2. Progress Box Plot ─────────────────────────────────────────────────────
print("Generating: Progress Distribution...")
fig, ax = plt.subplots(figsize=(8, 5))
data = [stats[e]["progress"] for e in ENVIRONMENTS]
bp = ax.boxplot(data, positions=x, widths=0.5, patch_artist=True,
                medianprops={"color": "black", "linewidth": 2},
                whiskerprops={"linewidth": 1.2}, capprops={"linewidth": 1.2},
                flierprops={"marker": "o", "markersize": 4, "alpha": 0.5})
for patch, col in zip(bp["boxes"], colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.75)
ax.set_xticks(x)
ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=12)
ax.set_ylabel("Progress per Episode (m)", fontsize=12)
ax.set_title("Flat Policy — Episode Progress Distribution", fontsize=14, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.tight_layout()
save(fig, "flat_progress_boxplot.png")

# ── 3. Fall Rate Bar Chart ───────────────────────────────────────────────────
print("Generating: Fall Rate...")
fig, ax = plt.subplots(figsize=(8, 5))
fall_rates = [stats[e]["fall_rate"] * 100 for e in ENVIRONMENTS]
bars = ax.bar(x, fall_rates, color=colors, alpha=0.85, zorder=3)
for bar, fr, n_falls in zip(bars, fall_rates, [stats[e]["falls"] for e in ENVIRONMENTS]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{fr:.0f}%\n({n_falls}/100)", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=12)
ax.set_ylabel("Fall Rate (%)", fontsize=12)
ax.set_title("Flat Policy — Fall Rate by Environment", fontsize=14, fontweight="bold")
ax.set_ylim(0, 120)
ax.axhline(50, color="red", linestyle="--", linewidth=1, alpha=0.6, label="50% threshold")
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
ax.legend(fontsize=10)
fig.tight_layout()
save(fig, "flat_fall_rate.png")

# ── 4. Mean Velocity ─────────────────────────────────────────────────────────
print("Generating: Mean Velocity...")
fig, ax = plt.subplots(figsize=(8, 5))
vels = [stats[e]["mean_velocity"] for e in ENVIRONMENTS]
bars = ax.bar(x, vels, color=colors, alpha=0.85, zorder=3)
for bar, v in zip(bars, vels):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f"{v:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.axhline(2.235, color="red", linestyle="--", linewidth=1.2, label="Spot Max (2.235 m/s)")
ax.set_xticks(x)
ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=12)
ax.set_ylabel("Mean Velocity (m/s)", fontsize=12)
ax.set_title("Flat Policy — Mean Velocity by Environment", fontsize=14, fontweight="bold")
ax.set_ylim(0, 2.5)
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
fig.tight_layout()
save(fig, "flat_mean_velocity.png")

# ── 5. Zone Distribution (stacked bar) ───────────────────────────────────────
print("Generating: Zone Distribution...")
zone_colors = ["#EF5350", "#FF7043", "#FFA726", "#66BB6A", "#42A5F5"]
zone_labels = ["Zone 1\n(0-10m)", "Zone 2\n(10-20m)", "Zone 3\n(20-30m)", "Zone 4\n(30-40m)", "Zone 5\n(40-50m)"]
fig, ax = plt.subplots(figsize=(8, 5))
bottom = np.zeros(len(ENVIRONMENTS))
for z in range(1, 6):
    counts = np.array([sum(1 for ep in episodes[env] if ep["zone_reached"] == z)
                       for env in ENVIRONMENTS])
    ax.bar(x, counts, bottom=bottom, color=zone_colors[z-1],
           label=zone_labels[z-1], alpha=0.9, zorder=3)
    # Label counts > 0
    for i, c in enumerate(counts):
        if c >= 5:
            ax.text(i, bottom[i] + c/2, str(int(c)), ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")
    bottom += counts
ax.set_xticks(x)
ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=12)
ax.set_ylabel("Number of Episodes", fontsize=12)
ax.set_title("Flat Policy — Zone Reached Distribution", fontsize=14, fontweight="bold")
ax.legend(fontsize=8, loc="upper right", ncol=2)
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
fig.tight_layout()
save(fig, "flat_zone_distribution.png")

# ── 6. Stability vs Progress Scatter ─────────────────────────────────────────
print("Generating: Stability vs Progress...")
fig, ax = plt.subplots(figsize=(8, 6))
markers = {"friction": "o", "grass": "^", "boulder": "s", "stairs": "D"}
for env in ENVIRONMENTS:
    eps = episodes[env]
    xs = [e["progress"] for e in eps]
    ys = [e["stability_score"] for e in eps]
    fell = [e["fall_detected"] for e in eps]
    # Plot non-falls
    ax.scatter([x for x, f in zip(xs, fell) if not f],
               [y for y, f in zip(ys, fell) if not f],
               c=ENV_COLORS[env], marker=markers[env], s=50, alpha=0.7,
               label=f"{ENV_LABELS[env]}", edgecolors="white", linewidth=0.5)
    # Plot falls with red edge
    ax.scatter([x for x, f in zip(xs, fell) if f],
               [y for y, f in zip(ys, fell) if f],
               c=ENV_COLORS[env], marker=markers[env], s=50, alpha=0.7,
               edgecolors="red", linewidth=1.5)
ax.set_xlabel("Progress (m)", fontsize=12)
ax.set_ylabel("Stability Score (lower = better)", fontsize=12)
ax.set_title("Flat Policy — Stability vs Progress\n(red border = fell)", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="upper left")
ax.grid(linestyle="--", alpha=0.4)
fig.tight_layout()
save(fig, "flat_stability_vs_progress.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FOUR-PANEL COMPOSITE ("ONE-SHOT") — PPT Widescreen (13.33" x 7.5")
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating: 4-Panel Composite (PPT size)...")

# Build fall heatmap data: rows = environments, cols = zones 1-5
fall_matrix = np.zeros((len(ENVIRONMENTS), 5))
for i, env in enumerate(ENVIRONMENTS):
    for ep in episodes[env]:
        if ep["fall_detected"] and ep.get("fall_zone") is not None:
            z = int(ep["fall_zone"]) - 1  # 0-indexed
            if 0 <= z < 5:
                fall_matrix[i, z] += 1

# Build stall heatmap data
stall_matrix_comp = np.zeros((len(ENVIRONMENTS), 5))
for i, env in enumerate(ENVIRONMENTS):
    for ep in episodes[env]:
        if not ep["fall_detected"] and not ep["completion"]:
            z = max(0, min(4, int(ep["progress"] / 10)))
            stall_matrix_comp[i, z] += 1

zone_labels_short = ["Zone 1\n(0-10m)", "Zone 2\n(10-20m)", "Zone 3\n(20-30m)", "Zone 4\n(30-40m)", "Zone 5\n(40-50m)"]
zone_labels_flipped = zone_labels_short[::-1]
env_labels_list = [ENV_LABELS[e] for e in ENVIRONMENTS]

# PPT widescreen: 13.33 x 7.5 inches
fig = plt.figure(figsize=(13.33, 7.5), facecolor="white")
fig.suptitle("Flat Policy — 100-Episode Evaluation Across 4 Environments",
             fontsize=20, fontweight="bold", y=0.97)

# 4 panels: Progress | Stall Heatmap | Fall Rate | Fall Heatmap
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1.2, 1, 1.2], wspace=0.35,
                      left=0.05, right=0.97, top=0.85, bottom=0.12)

# ── Panel 1: Mean Progress with error bars ──
ax = fig.add_subplot(gs[0])
bars = ax.bar(x, means, color=colors, alpha=0.85, zorder=3,
              yerr=stds, capsize=4, error_kw={"elinewidth": 1.2})
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{m:.1f}m", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=11, rotation=30, ha="right")
ax.set_ylabel("Mean Progress (m)", fontsize=11)
ax.set_title("Mean Progress", fontsize=13, fontweight="bold")
ax.set_ylim(0, max(means) * 1.35)
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
ax.tick_params(axis="y", labelsize=9)

# ── Panel 2: Stall Location Heatmap ──
ax = fig.add_subplot(gs[1])
stall_T = stall_matrix_comp.T[::-1]
im_stall = ax.imshow(stall_T, cmap="YlOrBr", aspect="auto", vmin=0,
                      vmax=max(stall_T.max(), 1))
for i in range(5):
    for j in range(len(ENVIRONMENTS)):
        val = int(stall_T[i, j])
        text_color = "white" if val > stall_T.max() * 0.55 else "black"
        ax.text(j, i, str(val) if val > 0 else "",
                ha="center", va="center", fontsize=12, fontweight="bold", color=text_color)
ax.set_xticks(np.arange(len(ENVIRONMENTS)))
ax.set_xticklabels(env_labels_list, fontsize=11, rotation=30, ha="right")
ax.set_yticks(np.arange(5))
ax.set_yticklabels(zone_labels_flipped, fontsize=8)
ax.set_title("Stall Locations", fontsize=13, fontweight="bold")
cbar_s = fig.colorbar(im_stall, ax=ax, shrink=0.7, pad=0.03)
cbar_s.set_label("Stalls", fontsize=9)
cbar_s.ax.tick_params(labelsize=8)

# ── Panel 3: Fall Rate ──
ax = fig.add_subplot(gs[2])
bars = ax.bar(x, fall_rates, color=colors, alpha=0.85, zorder=3)
for bar, fr in zip(bars, fall_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{fr:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.axhline(50, color="red", linestyle="--", linewidth=1.2, alpha=0.6, label="50%")
ax.set_xticks(x)
ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=11, rotation=30, ha="right")
ax.set_ylabel("Fall Rate (%)", fontsize=11)
ax.set_title("Fall Rate", fontsize=13, fontweight="bold")
ax.set_ylim(0, 115)
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
ax.legend(fontsize=8, loc="upper left")
ax.tick_params(axis="y", labelsize=9)

# ── Panel 4: Fall Location Heatmap ──
ax = fig.add_subplot(gs[3])
fall_T = fall_matrix.T[::-1]
im_fall = ax.imshow(fall_T, cmap="YlOrRd", aspect="auto", vmin=0,
                     vmax=max(fall_T.max(), 1))
for i in range(5):
    for j in range(len(ENVIRONMENTS)):
        val = int(fall_T[i, j])
        text_color = "white" if val > fall_T.max() * 0.6 else "black"
        ax.text(j, i, str(val) if val > 0 else "",
                ha="center", va="center", fontsize=12, fontweight="bold", color=text_color)
ax.set_xticks(np.arange(len(ENVIRONMENTS)))
ax.set_xticklabels(env_labels_list, fontsize=11, rotation=30, ha="right")
ax.set_yticks(np.arange(5))
ax.set_yticklabels(zone_labels_flipped, fontsize=8)
ax.set_title("Fall Locations", fontsize=13, fontweight="bold")
cbar_f = fig.colorbar(im_fall, ax=ax, shrink=0.7, pad=0.03)
cbar_f.set_label("Falls", fontsize=9)
cbar_f.ax.tick_params(labelsize=8)

# Footnote
fig.text(0.5, 0.02,
         "*Stall = timed out after 600s without falling or completing 50m. "
         "0 of 400 episodes completed the course.",
         ha="center", va="center", fontsize=10, fontstyle="italic", color="#555555")

save(fig, "flat_policy_composite.png")


# ══════════════════════════════════════════════════════════════════════════════
#  STALL LOCATION HEATMAP — PPT Widescreen
# ══════════════════════════════════════════════════════════════════════════════
print("Generating: Stall Location Heatmap...")

# A "stall" = timeout episode (no fall, no completion). The robot was alive
# but couldn't advance past its max progress. Stall zone = zone of max progress.
stall_matrix = np.zeros((len(ENVIRONMENTS), 5))
stall_counts = {}
for i, env in enumerate(ENVIRONMENTS):
    timeout_eps = [e for e in episodes[env]
                   if not e["fall_detected"] and not e["completion"]]
    stall_counts[env] = len(timeout_eps)
    for ep in timeout_eps:
        z = max(0, min(4, int(ep["progress"] / 10)))  # zone 0-4
        stall_matrix[i, z] += 1

# Transpose & flip so zones go bottom-to-top, environments match x-axis
stall_matrix_T = stall_matrix.T[::-1]
zone_labels_flipped_stall = ["Zone 5\n(40-50m)", "Zone 4\n(30-40m)", "Zone 3\n(20-30m)",
                              "Zone 2\n(10-20m)", "Zone 1\n(0-10m)"]

fig = plt.figure(figsize=(13.33, 7.5), facecolor="white")
fig.suptitle("Flat Policy — Stall Locations Across 4 Environments",
             fontsize=20, fontweight="bold", y=0.95)

ax = fig.add_axes([0.12, 0.15, 0.65, 0.70])  # [left, bottom, width, height]

im = ax.imshow(stall_matrix_T, cmap="YlOrBr", aspect="auto", vmin=0,
               vmax=max(stall_matrix_T.max(), 1))

# Annotate cells
for i in range(5):
    for j in range(len(ENVIRONMENTS)):
        val = int(stall_matrix_T[i, j])
        text_color = "white" if val > stall_matrix_T.max() * 0.55 else "black"
        ax.text(j, i, str(val) if val > 0 else "",
                ha="center", va="center", fontsize=18, fontweight="bold",
                color=text_color)

ax.set_xticks(np.arange(len(ENVIRONMENTS)))
ax.set_xticklabels([f"{ENV_LABELS[e]}\n({stall_counts[e]} stalls)" for e in ENVIRONMENTS],
                    fontsize=13)
ax.set_yticks(np.arange(5))
ax.set_yticklabels(zone_labels_flipped_stall, fontsize=11)
ax.set_title("Zone Where Robot Stalled (Timeout Without Falling)", fontsize=14, fontweight="bold")

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
cbar.set_label("Number of Stalled Episodes", fontsize=12)
cbar.ax.tick_params(labelsize=11)

fig.text(0.5, 0.03,
         "*Stall = episode timed out after 600 seconds without falling or completing the 50m course.\n"
         "The robot reached its max progress but could not advance further.",
         ha="center", va="center", fontsize=11, fontstyle="italic", color="#555555")

save(fig, "flat_stall_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FRICTION-ONLY 4-PANEL COMPOSITE — Same style as main composite
# ══════════════════════════════════════════════════════════════════════════════
print("Generating: Friction Flat Composite...")

FRIC_ENVS = ["friction"]
fric_eps_list = episodes["friction"]
fric_n_fell = sum(1 for e in fric_eps_list if e["fall_detected"])
fric_n_stall = sum(1 for e in fric_eps_list if not e["fall_detected"] and not e["completion"])
fric_mean_prog = np.mean([e["progress"] for e in fric_eps_list])
fric_std_prog = np.std([e["progress"] for e in fric_eps_list])
fric_fall_rate = fric_n_fell / len(fric_eps_list) * 100

# Build fall matrix (1 env x 5 zones)
fric_fall_mat = np.zeros((1, 5))
for ep in fric_eps_list:
    if ep["fall_detected"] and ep.get("fall_zone") is not None:
        z = int(ep["fall_zone"]) - 1
        if 0 <= z < 5:
            fric_fall_mat[0, z] += 1

# Build stall matrix (1 env x 5 zones)
fric_stall_mat = np.zeros((1, 5))
for ep in fric_eps_list:
    if not ep["fall_detected"] and not ep["completion"]:
        z = max(0, min(4, int(ep["progress"] / 10)))
        fric_stall_mat[0, z] += 1

fric_zone_labels = ["Zone 1\n(0-10m)", "Zone 2\n(10-20m)", "Zone 3\n(20-30m)", "Zone 4\n(30-40m)", "Zone 5\n(40-50m)"]
fric_zone_flipped = fric_zone_labels[::-1]
fx = np.arange(1)

fig = plt.figure(figsize=(13.33, 7.5), facecolor="white")
fig.suptitle("Flat Policy / Friction — 100-Episode Evaluation",
             fontsize=20, fontweight="bold", y=0.97)

gs = fig.add_gridspec(1, 4, width_ratios=[1, 1.2, 1, 1.2], wspace=0.35,
                      left=0.05, right=0.97, top=0.85, bottom=0.12)

# ── Panel 1: Mean Progress ──
ax = fig.add_subplot(gs[0])
bars = ax.bar(fx, [fric_mean_prog], color=["#2196F3"], alpha=0.85, zorder=3,
              yerr=[fric_std_prog], capsize=5, error_kw={"elinewidth": 1.5})
ax.text(0, fric_mean_prog + fric_std_prog + 1.5, f"{fric_mean_prog:.1f}m",
        ha="center", va="bottom", fontsize=13, fontweight="bold")
ax.set_xticks(fx)
ax.set_xticklabels(["Friction"], fontsize=13)
ax.set_ylabel("Mean Progress (m)", fontsize=13)
ax.set_title("Mean Progress", fontsize=15, fontweight="bold")
ax.set_ylim(0, 52)
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
ax.tick_params(axis="y", labelsize=11)

# ── Panel 2: Stall Location Heatmap ──
ax = fig.add_subplot(gs[1])
stall_T_fric = fric_stall_mat.T[::-1]  # (5 zones, 1 env)
im_s = ax.imshow(stall_T_fric, cmap="YlOrBr", aspect="auto", vmin=0,
                  vmax=max(stall_T_fric.max(), 1))
for i in range(5):
    val = int(stall_T_fric[i, 0])
    text_color = "white" if val > stall_T_fric.max() * 0.55 else "black"
    ax.text(0, i, str(val) if val > 0 else "",
            ha="center", va="center", fontsize=14, fontweight="bold", color=text_color)
ax.set_xticks([0])
ax.set_xticklabels(["Friction"], fontsize=13)
ax.set_yticks(np.arange(5))
ax.set_yticklabels(fric_zone_flipped, fontsize=10)
ax.set_title("Stall Locations", fontsize=15, fontweight="bold")
cbar_s = fig.colorbar(im_s, ax=ax, shrink=0.7, pad=0.04)
cbar_s.set_label("Stalls", fontsize=11)
cbar_s.ax.tick_params(labelsize=9)

# ── Panel 3: Fall Rate ──
ax = fig.add_subplot(gs[2])
bars = ax.bar(fx, [fric_fall_rate], color=["#2196F3"], alpha=0.85, zorder=3)
ax.text(0, fric_fall_rate + 2, f"{fric_fall_rate:.0f}%",
        ha="center", va="bottom", fontsize=13, fontweight="bold")
ax.axhline(50, color="red", linestyle="--", linewidth=1.2, alpha=0.6, label="50%")
ax.set_xticks(fx)
ax.set_xticklabels(["Friction"], fontsize=13)
ax.set_ylabel("Fall Rate (%)", fontsize=13)
ax.set_title("Fall Rate", fontsize=15, fontweight="bold")
ax.set_ylim(0, 115)
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
ax.legend(fontsize=10, loc="upper left")
ax.tick_params(axis="y", labelsize=11)

# ── Panel 4: Fall Location Heatmap ──
ax = fig.add_subplot(gs[3])
fall_T_fric = fric_fall_mat.T[::-1]  # (5 zones, 1 env)
im_f = ax.imshow(fall_T_fric, cmap="YlOrRd", aspect="auto", vmin=0,
                  vmax=max(fall_T_fric.max(), 1))
for i in range(5):
    val = int(fall_T_fric[i, 0])
    text_color = "white" if val > fall_T_fric.max() * 0.6 else "black"
    ax.text(0, i, str(val) if val > 0 else "",
            ha="center", va="center", fontsize=14, fontweight="bold", color=text_color)
ax.set_xticks([0])
ax.set_xticklabels(["Friction"], fontsize=13)
ax.set_yticks(np.arange(5))
ax.set_yticklabels(fric_zone_flipped, fontsize=10)
ax.set_title("Fall Locations", fontsize=15, fontweight="bold")
cbar_f = fig.colorbar(im_f, ax=ax, shrink=0.7, pad=0.04)
cbar_f.set_label("Falls", fontsize=11)
cbar_f.ax.tick_params(labelsize=9)

# Footnote
fig.text(0.5, 0.02,
         f"*{fric_n_stall} stalled (timeout after 600s), {fric_n_fell} fell, 0 completed. "
         "0 of 100 episodes completed the full 50m course.",
         ha="center", va="center", fontsize=10, fontstyle="italic", color="#555555")

save(fig, "flat_friction_composite.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FRICTION PIE CHART — Falls vs Stalls (vs Completions)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating: Friction Pie Chart...")

fric_n_complete = sum(1 for e in fric_eps_list if e["completion"])
fric_total = len(fric_eps_list)

pie_sizes  = []
pie_labels = []
pie_colors = []
pie_explode = []

if fric_n_fell > 0:
    pie_sizes.append(fric_n_fell)
    pie_labels.append(f"Falls\n({fric_n_fell}/{fric_total})")
    pie_colors.append("#F44336")
    pie_explode.append(0.04)

if fric_n_stall > 0:
    pie_sizes.append(fric_n_stall)
    pie_labels.append(f"Stalls\n({fric_n_stall}/{fric_total})")
    pie_colors.append("#FF9800")
    pie_explode.append(0.04)

if fric_n_complete > 0:
    pie_sizes.append(fric_n_complete)
    pie_labels.append(f"Completed\n({fric_n_complete}/{fric_total})")
    pie_colors.append("#4CAF50")
    pie_explode.append(0.04)

fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
wedges, texts, autotexts = ax.pie(
    pie_sizes,
    labels=pie_labels,
    colors=pie_colors,
    explode=pie_explode,
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 13},
    wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
)
for at in autotexts:
    at.set_fontsize(14)
    at.set_fontweight("bold")
ax.set_title("Friction Environment — Episode Outcomes\n(Flat Policy, 100 Episodes)",
             fontsize=15, fontweight="bold", pad=18)
fig.tight_layout()
save(fig, "flat_friction_pie.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FRICTION STANDALONE: Stall Location Heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("Generating: Friction Stall Location (standalone)...")

fig, ax = plt.subplots(figsize=(6, 7), facecolor="white")
stall_T_fric = fric_stall_mat.T[::-1]
im = ax.imshow(stall_T_fric, cmap="YlOrBr", aspect="auto", vmin=0,
               vmax=max(stall_T_fric.max(), 1))
for i in range(5):
    val = int(stall_T_fric[i, 0])
    text_color = "white" if val > stall_T_fric.max() * 0.55 else "black"
    ax.text(0, i, str(val) if val > 0 else "",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color=text_color)
ax.set_xticks([0])
ax.set_xticklabels([f"Friction\n({fric_n_stall} stalls)"], fontsize=13)
ax.set_yticks(np.arange(5))
ax.set_yticklabels(fric_zone_flipped, fontsize=12)
ax.set_title("Stall Locations\n(Timeout — robot alive but didn't advance)",
             fontsize=14, fontweight="bold")
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
cbar.set_label("Number of Stalled Episodes", fontsize=11)
cbar.ax.tick_params(labelsize=10)
fig.tight_layout()
save(fig, "flat_friction_stall_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FRICTION STANDALONE: Fall Location Heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("Generating: Friction Fall Location (standalone)...")

fig, ax = plt.subplots(figsize=(6, 7), facecolor="white")
fall_T_fric = fric_fall_mat.T[::-1]
im = ax.imshow(fall_T_fric, cmap="YlOrRd", aspect="auto", vmin=0,
               vmax=max(fall_T_fric.max(), 1))
for i in range(5):
    val = int(fall_T_fric[i, 0])
    text_color = "white" if val > fall_T_fric.max() * 0.6 else "black"
    ax.text(0, i, str(val) if val > 0 else "",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color=text_color)
ax.set_xticks([0])
ax.set_xticklabels([f"Friction\n({fric_n_fell} falls)"], fontsize=13)
ax.set_yticks(np.arange(5))
ax.set_yticklabels(fric_zone_flipped, fontsize=12)
ax.set_title("Fall Locations\n(Zone where robot fell)",
             fontsize=14, fontweight="bold")
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
cbar.set_label("Number of Falls", fontsize=11)
cbar.ax.tick_params(labelsize=10)
fig.tight_layout()
save(fig, "flat_friction_fall_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
#  GRASS PIE CHART — Falls vs Stalls (vs Completions)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating: Grass Pie Chart...")

grass_eps_list  = episodes["grass"]
grass_n_fell    = sum(1 for e in grass_eps_list if e["fall_detected"])
grass_n_stall   = sum(1 for e in grass_eps_list if not e["fall_detected"] and not e["completion"])
grass_n_complete = sum(1 for e in grass_eps_list if e["completion"])
grass_total     = len(grass_eps_list)

# Build fall matrix (1 env x 5 zones)
grass_fall_mat = np.zeros((1, 5))
for ep in grass_eps_list:
    if ep["fall_detected"] and ep.get("fall_zone") is not None:
        z = int(ep["fall_zone"]) - 1
        if 0 <= z < 5:
            grass_fall_mat[0, z] += 1

# Build stall matrix (1 env x 5 zones)
grass_stall_mat = np.zeros((1, 5))
for ep in grass_eps_list:
    if not ep["fall_detected"] and not ep["completion"]:
        z = max(0, min(4, int(ep["progress"] / 10)))
        grass_stall_mat[0, z] += 1

grass_zone_labels  = ["Zone 1\n(0-10m)\nLight fluid",
                      "Zone 2\n(10-20m)\nThin grass",
                      "Zone 3\n(20-30m)\nMedium lawn",
                      "Zone 4\n(30-40m)\nThick grass",
                      "Zone 5\n(40-50m)\nDense brush"]
grass_zone_flipped = grass_zone_labels[::-1]

gpie_sizes   = []
gpie_labels  = []
gpie_colors  = []
gpie_explode = []

if grass_n_fell > 0:
    gpie_sizes.append(grass_n_fell)
    gpie_labels.append(f"Falls\n({grass_n_fell}/{grass_total})")
    gpie_colors.append("#F44336")
    gpie_explode.append(0.04)

if grass_n_stall > 0:
    gpie_sizes.append(grass_n_stall)
    gpie_labels.append(f"Stalls\n({grass_n_stall}/{grass_total})")
    gpie_colors.append("#FF9800")
    gpie_explode.append(0.04)

if grass_n_complete > 0:
    gpie_sizes.append(grass_n_complete)
    gpie_labels.append(f"Completed\n({grass_n_complete}/{grass_total})")
    gpie_colors.append("#4CAF50")
    gpie_explode.append(0.04)

fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
wedges, texts, autotexts = ax.pie(
    gpie_sizes,
    labels=gpie_labels,
    colors=gpie_colors,
    explode=gpie_explode,
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 13},
    wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
)
for at in autotexts:
    at.set_fontsize(14)
    at.set_fontweight("bold")
ax.set_title("Grass Environment — Episode Outcomes\n(Flat Policy, 100 Episodes)",
             fontsize=15, fontweight="bold", pad=18)
fig.tight_layout()
save(fig, "flat_grass_pie.png")


# ══════════════════════════════════════════════════════════════════════════════
#  GRASS STANDALONE: Stall Location Heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("Generating: Grass Stall Location (standalone)...")

fig, ax = plt.subplots(figsize=(6, 7), facecolor="white")
grass_stall_T = grass_stall_mat.T[::-1]
im = ax.imshow(grass_stall_T, cmap="YlOrBr", aspect="auto", vmin=0,
               vmax=max(grass_stall_T.max(), 1))
for i in range(5):
    val = int(grass_stall_T[i, 0])
    text_color = "white" if val > grass_stall_T.max() * 0.55 else "black"
    ax.text(0, i, str(val) if val > 0 else "",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color=text_color)
ax.set_xticks([0])
ax.set_xticklabels([f"Grass\n({grass_n_stall} stalls)"], fontsize=13)
ax.set_yticks(np.arange(5))
ax.set_yticklabels(grass_zone_flipped, fontsize=10)
ax.set_title("Grass — Stall Locations\n(Timeout — robot alive but didn't advance)",
             fontsize=14, fontweight="bold")
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
cbar.set_label("Number of Stalled Episodes", fontsize=11)
cbar.ax.tick_params(labelsize=10)
fig.tight_layout()
save(fig, "flat_grass_stall_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
#  GRASS STANDALONE: Fall Location Heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("Generating: Grass Fall Location (standalone)...")

fig, ax = plt.subplots(figsize=(6, 7), facecolor="white")
grass_fall_T = grass_fall_mat.T[::-1]
im = ax.imshow(grass_fall_T, cmap="YlOrRd", aspect="auto", vmin=0,
               vmax=max(grass_fall_T.max(), 1))
for i in range(5):
    val = int(grass_fall_T[i, 0])
    text_color = "white" if val > grass_fall_T.max() * 0.6 else "black"
    ax.text(0, i, str(val) if val > 0 else "",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color=text_color)
ax.set_xticks([0])
ax.set_xticklabels([f"Grass\n({grass_n_fell} falls)"], fontsize=13)
ax.set_yticks(np.arange(5))
ax.set_yticklabels(grass_zone_flipped, fontsize=10)
ax.set_title("Grass — Fall Locations\n(Zone where robot fell)",
             fontsize=14, fontweight="bold")
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
cbar.set_label("Number of Falls", fontsize=11)
cbar.ax.tick_params(labelsize=10)
fig.tight_layout()
save(fig, "flat_grass_fall_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
#  BOULDER PIE CHART — Falls vs Stalls (vs Completions)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating: Boulder Pie Chart...")

boulder_eps_list   = episodes["boulder"]
boulder_n_fell     = sum(1 for e in boulder_eps_list if e["fall_detected"])
boulder_n_stall    = sum(1 for e in boulder_eps_list if not e["fall_detected"] and not e["completion"])
boulder_n_complete = sum(1 for e in boulder_eps_list if e["completion"])
boulder_total      = len(boulder_eps_list)

# Build fall matrix (1 env x 5 zones)
boulder_fall_mat = np.zeros((1, 5))
for ep in boulder_eps_list:
    if ep["fall_detected"] and ep.get("fall_zone") is not None:
        z = int(ep["fall_zone"]) - 1
        if 0 <= z < 5:
            boulder_fall_mat[0, z] += 1

# Build stall matrix (1 env x 5 zones)
boulder_stall_mat = np.zeros((1, 5))
for ep in boulder_eps_list:
    if not ep["fall_detected"] and not ep["completion"]:
        z = max(0, min(4, int(ep["progress"] / 10)))
        boulder_stall_mat[0, z] += 1

boulder_zone_labels  = ["Zone 1\n(0-10m)\nGravel",
                         "Zone 2\n(10-20m)\nRiver rocks",
                         "Zone 3\n(20-30m)\nLarge rocks",
                         "Zone 4\n(30-40m)\nSmall boulders",
                         "Zone 5\n(40-50m)\nLarge boulders"]
boulder_zone_flipped = boulder_zone_labels[::-1]

bpie_sizes   = []
bpie_labels  = []
bpie_colors  = []
bpie_explode = []

if boulder_n_fell > 0:
    bpie_sizes.append(boulder_n_fell)
    bpie_labels.append(f"Falls\n({boulder_n_fell}/{boulder_total})")
    bpie_colors.append("#F44336")
    bpie_explode.append(0.04)

if boulder_n_stall > 0:
    bpie_sizes.append(boulder_n_stall)
    bpie_labels.append(f"Stalls\n({boulder_n_stall}/{boulder_total})")
    bpie_colors.append("#FF9800")
    bpie_explode.append(0.04)

if boulder_n_complete > 0:
    bpie_sizes.append(boulder_n_complete)
    bpie_labels.append(f"Completed\n({boulder_n_complete}/{boulder_total})")
    bpie_colors.append("#4CAF50")
    bpie_explode.append(0.04)

fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
wedges, texts, autotexts = ax.pie(
    bpie_sizes,
    labels=bpie_labels,
    colors=bpie_colors,
    explode=bpie_explode,
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 13},
    wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
)
for at in autotexts:
    at.set_fontsize(14)
    at.set_fontweight("bold")
ax.set_title("Boulder Environment — Episode Outcomes\n(Flat Policy, 100 Episodes)",
             fontsize=15, fontweight="bold", pad=18)
fig.tight_layout()
save(fig, "flat_boulder_pie.png")


# ══════════════════════════════════════════════════════════════════════════════
#  BOULDER STANDALONE: Stall Location Heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("Generating: Boulder Stall Location (standalone)...")

fig, ax = plt.subplots(figsize=(6, 7), facecolor="white")
boulder_stall_T = boulder_stall_mat.T[::-1]
im = ax.imshow(boulder_stall_T, cmap="YlOrBr", aspect="auto", vmin=0,
               vmax=max(boulder_stall_T.max(), 1))
for i in range(5):
    val = int(boulder_stall_T[i, 0])
    text_color = "white" if val > boulder_stall_T.max() * 0.55 else "black"
    ax.text(0, i, str(val) if val > 0 else "",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color=text_color)
ax.set_xticks([0])
ax.set_xticklabels([f"Boulder\n({boulder_n_stall} stalls)"], fontsize=13)
ax.set_yticks(np.arange(5))
ax.set_yticklabels(boulder_zone_flipped, fontsize=10)
ax.set_title("Boulder — Stall Locations\n(Timeout — robot alive but didn't advance)",
             fontsize=14, fontweight="bold")
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
cbar.set_label("Number of Stalled Episodes", fontsize=11)
cbar.ax.tick_params(labelsize=10)
fig.tight_layout()
save(fig, "flat_boulder_stall_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
#  BOULDER STANDALONE: Fall Location Heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("Generating: Boulder Fall Location (standalone)...")

fig, ax = plt.subplots(figsize=(6, 7), facecolor="white")
boulder_fall_T = boulder_fall_mat.T[::-1]
im = ax.imshow(boulder_fall_T, cmap="YlOrRd", aspect="auto", vmin=0,
               vmax=max(boulder_fall_T.max(), 1))
for i in range(5):
    val = int(boulder_fall_T[i, 0])
    text_color = "white" if val > boulder_fall_T.max() * 0.6 else "black"
    ax.text(0, i, str(val) if val > 0 else "",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color=text_color)
ax.set_xticks([0])
ax.set_xticklabels([f"Boulder\n({boulder_n_fell} falls)"], fontsize=13)
ax.set_yticks(np.arange(5))
ax.set_yticklabels(boulder_zone_flipped, fontsize=10)
ax.set_title("Boulder — Fall Locations\n(Zone where robot fell)",
             fontsize=14, fontweight="bold")
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
cbar.set_label("Number of Falls", fontsize=11)
cbar.ax.tick_params(labelsize=10)
fig.tight_layout()
save(fig, "flat_boulder_fall_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
#  STAIRS PIE CHART — Falls vs Stalls (vs Completions)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating: Stairs Pie Chart...")

stairs_eps_list   = episodes["stairs"]
stairs_n_fell     = sum(1 for e in stairs_eps_list if e["fall_detected"])
stairs_n_stall    = sum(1 for e in stairs_eps_list if not e["fall_detected"] and not e["completion"])
stairs_n_complete = sum(1 for e in stairs_eps_list if e["completion"])
stairs_total      = len(stairs_eps_list)

# Build fall matrix (1 env x 5 zones)
stairs_fall_mat = np.zeros((1, 5))
for ep in stairs_eps_list:
    if ep["fall_detected"] and ep.get("fall_zone") is not None:
        z = int(ep["fall_zone"]) - 1
        if 0 <= z < 5:
            stairs_fall_mat[0, z] += 1

# Build stall matrix (1 env x 5 zones)
stairs_stall_mat = np.zeros((1, 5))
for ep in stairs_eps_list:
    if not ep["fall_detected"] and not ep["completion"]:
        z = max(0, min(4, int(ep["progress"] / 10)))
        stairs_stall_mat[0, z] += 1

stairs_zone_labels  = ["Zone 1\n(0-10m)\nAccess ramp\n(3cm steps)",
                        "Zone 2\n(10-20m)\nLow residential\n(8cm steps)",
                        "Zone 3\n(20-30m)\nStd residential\n(13cm steps)",
                        "Zone 4\n(30-40m)\nSteep commercial\n(18cm steps)",
                        "Zone 5\n(40-50m)\nMax challenge\n(23cm steps)"]
stairs_zone_flipped = stairs_zone_labels[::-1]

spie_sizes   = []
spie_labels  = []
spie_colors  = []
spie_explode = []

if stairs_n_fell > 0:
    spie_sizes.append(stairs_n_fell)
    spie_labels.append(f"Falls\n({stairs_n_fell}/{stairs_total})")
    spie_colors.append("#F44336")
    spie_explode.append(0.04)

if stairs_n_stall > 0:
    spie_sizes.append(stairs_n_stall)
    spie_labels.append(f"Stalls\n({stairs_n_stall}/{stairs_total})")
    spie_colors.append("#FF9800")
    spie_explode.append(0.04)

if stairs_n_complete > 0:
    spie_sizes.append(stairs_n_complete)
    spie_labels.append(f"Completed\n({stairs_n_complete}/{stairs_total})")
    spie_colors.append("#4CAF50")
    spie_explode.append(0.04)

fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
wedges, texts, autotexts = ax.pie(
    spie_sizes,
    labels=spie_labels,
    colors=spie_colors,
    explode=spie_explode,
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 13},
    wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
)
for at in autotexts:
    at.set_fontsize(14)
    at.set_fontweight("bold")
ax.set_title("Stairs Environment — Episode Outcomes\n(Flat Policy, 100 Episodes)",
             fontsize=15, fontweight="bold", pad=18)
fig.tight_layout()
save(fig, "flat_stairs_pie.png")


# ══════════════════════════════════════════════════════════════════════════════
#  STAIRS STANDALONE: Stall Location Heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("Generating: Stairs Stall Location (standalone)...")

fig, ax = plt.subplots(figsize=(6, 7), facecolor="white")
stairs_stall_T = stairs_stall_mat.T[::-1]
im = ax.imshow(stairs_stall_T, cmap="YlOrBr", aspect="auto", vmin=0,
               vmax=max(stairs_stall_T.max(), 1))
for i in range(5):
    val = int(stairs_stall_T[i, 0])
    text_color = "white" if val > stairs_stall_T.max() * 0.55 else "black"
    ax.text(0, i, str(val) if val > 0 else "",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color=text_color)
ax.set_xticks([0])
ax.set_xticklabels([f"Stairs\n({stairs_n_stall} stalls)"], fontsize=13)
ax.set_yticks(np.arange(5))
ax.set_yticklabels(stairs_zone_flipped, fontsize=9)
ax.set_title("Stairs — Stall Locations\n(Timeout — robot alive but didn't advance)",
             fontsize=14, fontweight="bold")
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
cbar.set_label("Number of Stalled Episodes", fontsize=11)
cbar.ax.tick_params(labelsize=10)
fig.tight_layout()
save(fig, "flat_stairs_stall_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
#  STAIRS STANDALONE: Fall Location Heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("Generating: Stairs Fall Location (standalone)...")

fig, ax = plt.subplots(figsize=(6, 7), facecolor="white")
stairs_fall_T = stairs_fall_mat.T[::-1]
im = ax.imshow(stairs_fall_T, cmap="YlOrRd", aspect="auto", vmin=0,
               vmax=max(stairs_fall_T.max(), 1))
for i in range(5):
    val = int(stairs_fall_T[i, 0])
    text_color = "white" if val > stairs_fall_T.max() * 0.6 else "black"
    ax.text(0, i, str(val) if val > 0 else "",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color=text_color)
ax.set_xticks([0])
ax.set_xticklabels([f"Stairs\n({stairs_n_fell} falls)"], fontsize=13)
ax.set_yticks(np.arange(5))
ax.set_yticklabels(stairs_zone_flipped, fontsize=9)
ax.set_title("Stairs — Fall Locations\n(Zone where robot fell)",
             fontsize=14, fontweight="bold")
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
cbar.set_label("Number of Falls", fontsize=11)
cbar.ax.tick_params(labelsize=10)
fig.tight_layout()
save(fig, "flat_stairs_fall_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FRICTION V2 PIE CHART — Falls vs Stalls (vs Completions)
# ══════════════════════════════════════════════════════════════════════════════
fv2_eps_list = load_jsonl("friction_v2")

if not fv2_eps_list:
    print("Skipping Friction V2 plots — friction_v2_flat_episodes.jsonl not found.")
else:
    print("Generating: Friction V2 Pie Chart...")

    fv2_n_fell    = sum(1 for e in fv2_eps_list if e["fall_detected"])
    fv2_n_stall   = sum(1 for e in fv2_eps_list if not e["fall_detected"] and not e["completion"])
    fv2_n_complete = sum(1 for e in fv2_eps_list if e["completion"])
    fv2_total     = len(fv2_eps_list)

    # Build fall matrix (1 env x 5 zones)
    fv2_fall_mat = np.zeros((1, 5))
    for ep in fv2_eps_list:
        if ep["fall_detected"] and ep.get("fall_zone") is not None:
            z = int(ep["fall_zone"]) - 1
            if 0 <= z < 5:
                fv2_fall_mat[0, z] += 1

    # Build stall matrix (1 env x 5 zones)
    fv2_stall_mat = np.zeros((1, 5))
    for ep in fv2_eps_list:
        if not ep["fall_detected"] and not ep["completion"]:
            z = max(0, min(4, int(ep["progress"] / 10)))
            fv2_stall_mat[0, z] += 1

    fv2_zone_labels  = ["Zone 1\n(0-10m)\nWet asphalt\n(μ=0.38)",
                        "Zone 2\n(10-20m)\nWet grass\n(μ=0.25)",
                        "Zone 3\n(20-30m)\nPacked snow\n(μ=0.15)",
                        "Zone 4\n(30-40m)\nLoose snow\n(μ=0.11)",
                        "Zone 5\n(40-50m)\nGlaze ice\n(μ=0.08)"]
    fv2_zone_flipped = fv2_zone_labels[::-1]

    fv2pie_sizes   = []
    fv2pie_labels  = []
    fv2pie_colors  = []
    fv2pie_explode = []

    if fv2_n_fell > 0:
        fv2pie_sizes.append(fv2_n_fell)
        fv2pie_labels.append(f"Falls\n({fv2_n_fell}/{fv2_total})")
        fv2pie_colors.append("#F44336")
        fv2pie_explode.append(0.04)

    if fv2_n_stall > 0:
        fv2pie_sizes.append(fv2_n_stall)
        fv2pie_labels.append(f"Stalls\n({fv2_n_stall}/{fv2_total})")
        fv2pie_colors.append("#FF9800")
        fv2pie_explode.append(0.04)

    if fv2_n_complete > 0:
        fv2pie_sizes.append(fv2_n_complete)
        fv2pie_labels.append(f"Completed\n({fv2_n_complete}/{fv2_total})")
        fv2pie_colors.append("#4CAF50")
        fv2pie_explode.append(0.04)

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
    wedges, texts, autotexts = ax.pie(
        fv2pie_sizes,
        labels=fv2pie_labels,
        colors=fv2pie_colors,
        explode=fv2pie_explode,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 13},
        wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontsize(14)
        at.set_fontweight("bold")
    ax.set_title("Friction V2 Environment — Episode Outcomes\n(Flat Policy, 100 Episodes)",
                 fontsize=15, fontweight="bold", pad=18)
    fig.tight_layout()
    save(fig, "flat_friction_v2_pie.png")


    # ══════════════════════════════════════════════════════════════════════════════
    #  FRICTION V2 STANDALONE: Stall Location Heatmap
    # ══════════════════════════════════════════════════════════════════════════════
    print("Generating: Friction V2 Stall Location (standalone)...")

    fig, ax = plt.subplots(figsize=(6, 7), facecolor="white")
    fv2_stall_T = fv2_stall_mat.T[::-1]
    im = ax.imshow(fv2_stall_T, cmap="YlOrBr", aspect="auto", vmin=0,
                   vmax=max(fv2_stall_T.max(), 1))
    for i in range(5):
        val = int(fv2_stall_T[i, 0])
        text_color = "white" if val > fv2_stall_T.max() * 0.55 else "black"
        ax.text(0, i, str(val) if val > 0 else "",
                ha="center", va="center", fontsize=20, fontweight="bold",
                color=text_color)
    ax.set_xticks([0])
    ax.set_xticklabels([f"Friction V2\n({fv2_n_stall} stalls)"], fontsize=13)
    ax.set_yticks(np.arange(5))
    ax.set_yticklabels(fv2_zone_flipped, fontsize=9)
    ax.set_title("Friction V2 — Stall Locations\n(Timeout — robot alive but didn't advance)",
                 fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
    cbar.set_label("Number of Stalled Episodes", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    fig.tight_layout()
    save(fig, "flat_friction_v2_stall_heatmap.png")


    # ══════════════════════════════════════════════════════════════════════════════
    #  FRICTION V2 STANDALONE: Fall Location Heatmap
    # ══════════════════════════════════════════════════════════════════════════════
    print("Generating: Friction V2 Fall Location (standalone)...")

    fig, ax = plt.subplots(figsize=(6, 7), facecolor="white")
    fv2_fall_T = fv2_fall_mat.T[::-1]
    im = ax.imshow(fv2_fall_T, cmap="YlOrRd", aspect="auto", vmin=0,
                   vmax=max(fv2_fall_T.max(), 1))
    for i in range(5):
        val = int(fv2_fall_T[i, 0])
        text_color = "white" if val > fv2_fall_T.max() * 0.6 else "black"
        ax.text(0, i, str(val) if val > 0 else "",
                ha="center", va="center", fontsize=20, fontweight="bold",
                color=text_color)
    ax.set_xticks([0])
    ax.set_xticklabels([f"Friction V2\n({fv2_n_fell} falls)"], fontsize=13)
    ax.set_yticks(np.arange(5))
    ax.set_yticklabels(fv2_zone_flipped, fontsize=9)
    ax.set_title("Friction V2 — Fall Locations\n(Zone where robot fell)",
                 fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
    cbar.set_label("Number of Falls", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    fig.tight_layout()
    save(fig, "flat_friction_v2_fall_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE (printed)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  FLAT POLICY — 100-EPISODE RESULTS SUMMARY")
print("="*70)
print(f"  {'Environment':<12} {'Progress':>10} {'Fall Rate':>10} {'Velocity':>10} {'Stability':>10} {'Zone':>6}")
print(f"  {'-'*12:<12} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10} {'-'*6:>6}")
for env in ENVIRONMENTS:
    s = stats[env]
    print(f"  {ENV_LABELS[env]:<12} {s['mean_progress']:>8.1f}m {s['fall_rate']*100:>8.0f}% "
          f"{s['mean_velocity']:>8.2f} {s['mean_stability']:>10.3f} {s['mean_zone']:>5.1f}")
print("="*70)
print(f"\nAll plots saved to: {PLOTS_DIR}")
print("Done.")
