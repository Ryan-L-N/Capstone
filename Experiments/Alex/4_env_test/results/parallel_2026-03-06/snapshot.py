"""
Quick-snapshot visualization for the 2026-03-06 flat-policy rerun.
Highlights key trends across the 4 environments using the full schema data.

Output: report/snapshot.png
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(BASE_DIR, "report")
os.makedirs(OUT_DIR, exist_ok=True)

ENVIRONMENTS = ["friction", "grass", "boulder", "stairs"]
LABELS       = ["Friction", "Grass", "Boulder", "Stairs"]
COLORS       = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
X            = np.arange(len(ENVIRONMENTS))

# ── Load data ─────────────────────────────────────────────────────────────────
def load(env):
    path = os.path.join(BASE_DIR, f"{env}_flat_episodes.jsonl")
    return [json.loads(l) for l in open(path)]

data = {env: load(env) for env in ENVIRONMENTS}

def arr(env, key):
    return np.array([r[key] for r in data[env]])

# ── Pre-compute stats ─────────────────────────────────────────────────────────
prog      = {e: arr(e, "progress")        for e in ENVIRONMENTS}
roll      = {e: arr(e, "mean_roll")       for e in ENVIRONMENTS}
pitch     = {e: arr(e, "mean_pitch")      for e in ENVIRONMENTS}
ang_vel   = {e: arr(e, "mean_ang_vel")    for e in ENVIRONMENTS}
hvar      = {e: arr(e, "height_variance") for e in ENVIRONMENTS}
vel       = {e: arr(e, "mean_velocity")   for e in ENVIRONMENTS}
ep_len    = {e: arr(e, "episode_length")  for e in ENVIRONMENTS}

fall_rate   = [np.mean(arr(e, "fall_detected")) * 100  for e in ENVIRONMENTS]
mean_prog   = [np.mean(prog[e])   for e in ENVIRONMENTS]
std_prog    = [np.std(prog[e])    for e in ENVIRONMENTS]
mean_roll_v = [np.mean(roll[e])   for e in ENVIRONMENTS]
mean_pit_v  = [np.mean(pitch[e])  for e in ENVIRONMENTS]
mean_ang_v  = [np.mean(ang_vel[e]) for e in ENVIRONMENTS]
mean_hvar   = [np.mean(hvar[e])   for e in ENVIRONMENTS]
mean_vel    = [np.mean(vel[e])    for e in ENVIRONMENTS]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10), facecolor="white")
fig.suptitle(
    "Flat Policy — Trend Snapshot Across 4 Environments (100 episodes each)",
    fontsize=18, fontweight="bold", y=0.98
)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35,
                       left=0.06, right=0.97, top=0.91, bottom=0.09)

def bar_ax(ax, values, ylabel, title, fmt=".1f", pct=False):
    bars = ax.bar(X, values, color=COLORS, alpha=0.88, zorder=3)
    for bar, v in zip(bars, values):
        label = f"{v:{fmt}}{'%' if pct else ''}"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.03,
                label, ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(X)
    ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.28)
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.tick_params(axis="y", labelsize=9)

# ── Panel 1: Mean Progress ────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(X, mean_prog, color=COLORS, alpha=0.88, zorder=3,
               yerr=std_prog, capsize=5, error_kw={"elinewidth": 1.4})
for bar, v in zip(bars, mean_prog):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + max(mean_prog) * 0.05,
             f"{v:.1f}m", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax1.set_xticks(X)
ax1.set_xticklabels(LABELS, fontsize=11)
ax1.set_ylabel("Mean Progress (m)", fontsize=11)
ax1.set_title("Mean Progress", fontsize=13, fontweight="bold")
ax1.set_ylim(0, max(mean_prog) * 1.3)
ax1.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
ax1.tick_params(axis="y", labelsize=9)

# ── Panel 2: Fall Rate ────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
bar_ax(ax2, fall_rate, "Fall Rate (%)", "Fall Rate", fmt=".0f", pct=True)
ax2.axhline(50, color="red", linestyle="--", linewidth=1.2, alpha=0.6, label="50%")
ax2.legend(fontsize=9, loc="upper left")
ax2.set_ylim(0, 115)

# ── Panel 3: Episode Length boxplot ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
bp = ax3.boxplot([ep_len[e] for e in ENVIRONMENTS], positions=X, widths=0.5,
                 patch_artist=True,
                 medianprops={"color": "black", "linewidth": 2},
                 whiskerprops={"linewidth": 1.2}, capprops={"linewidth": 1.2},
                 flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
for patch, col in zip(bp["boxes"], COLORS):
    patch.set_facecolor(col)
    patch.set_alpha(0.75)
ax3.set_xticks(X)
ax3.set_xticklabels(LABELS, fontsize=11)
ax3.set_ylabel("Episode Length (s)", fontsize=11)
ax3.set_title("Episode Length Distribution", fontsize=13, fontweight="bold")
ax3.grid(axis="y", linestyle="--", alpha=0.45)
ax3.tick_params(axis="y", labelsize=9)

# ── Panel 4: Stability — Roll / Pitch / Ang Vel (grouped bars) ───────────────
ax4 = fig.add_subplot(gs[1, 0])
w = 0.25
x2 = np.arange(len(ENVIRONMENTS))
ax4.bar(x2 - w, mean_roll_v,  width=w, color="#E53935", alpha=0.85,
        label="Mean Roll (rad)",    zorder=3)
ax4.bar(x2,     mean_pit_v,   width=w, color="#FB8C00", alpha=0.85,
        label="Mean Pitch (rad)",   zorder=3)
ax4.bar(x2 + w, mean_ang_v,   width=w, color="#8E24AA", alpha=0.85,
        label="Mean Ang Vel (rad/s)", zorder=3)
ax4.set_xticks(x2)
ax4.set_xticklabels(LABELS, fontsize=11)
ax4.set_ylabel("Magnitude", fontsize=11)
ax4.set_title("Rotational Stability Metrics", fontsize=13, fontweight="bold")
ax4.legend(fontsize=8, loc="upper left")
ax4.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
ax4.tick_params(axis="y", labelsize=9)

# ── Panel 5: Height Variance ──────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
bar_ax(ax5, [v * 1000 for v in mean_hvar],
       "Height Variance (×10⁻³ m²)", "Body Height Variance", fmt=".2f")

# ── Panel 6: Mean Forward Velocity ───────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
bar_ax(ax6, mean_vel, "Mean Velocity (m/s)", "Mean Forward Velocity", fmt=".2f")
ax6.axhline(2.235, color="red", linestyle="--", linewidth=1.2,
            alpha=0.6, label="Spot max (2.235 m/s)")
ax6.legend(fontsize=8, loc="upper right")

out = os.path.join(OUT_DIR, "snapshot.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
