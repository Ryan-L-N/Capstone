"""
Visualization script for Hybrid No-Coach 100-episode 4-environment eval.
Model: model_19999.pt (spot_hybrid_ppo, MH-2a, [512,256,128])
Run: 2026-03-16, H100 parallel eval

Outputs saved to: plots/
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

ENVIRONMENTS = ["friction", "grass", "boulder", "stairs"]
ENV_LABELS   = {"friction": "Friction", "grass": "Grass", "boulder": "Boulder", "stairs": "Stairs"}
ENV_COLORS   = {"friction": "#42A5F5", "grass": "#66BB6A", "boulder": "#FF7043", "stairs": "#AB47BC"}
ZONE_COLORS  = ["#EF5350", "#FF7043", "#FFA726", "#66BB6A", "#42A5F5"]
COURSE_LENGTH = 49.5  # meters

# ── Load data ─────────────────────────────────────────────────────────────────
data = {}
for env in ENVIRONMENTS:
    path = os.path.join(BASE_DIR, f"{env}_rough_episodes.jsonl")
    with open(path) as f:
        data[env] = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {env}: {len(data[env])} episodes")

# ── Compute stats ─────────────────────────────────────────────────────────────
stats = {}
for env in ENVIRONMENTS:
    eps = data[env]
    progress = np.array([e["progress"] for e in eps])
    zones    = np.array([e["zone_reached"] for e in eps])
    times    = np.array([e.get("time_to_complete") or e.get("episode_length", 0) or 0 for e in eps])
    velocity = np.array([e.get("mean_velocity", 0) or 0 for e in eps])
    stability= np.array([e.get("stability_score", 0) or 0 for e in eps])
    completions = sum(1 for e in eps if e["completion"])
    falls       = sum(1 for e in eps if e["fall_detected"])

    stats[env] = {
        "progress_mean": np.mean(progress),
        "progress_std": np.std(progress),
        "progress_min": np.min(progress),
        "progress_max": np.max(progress),
        "zone_mean": np.mean(zones),
        "zone_counts": Counter(int(z) for z in zones),
        "time_mean": np.mean(times),
        "time_std": np.std(times),
        "velocity_mean": np.mean(velocity),
        "velocity_std": np.std(velocity),
        "stability_mean": np.mean(stability),
        "stability_std": np.std(stability),
        "completion_rate": completions / len(eps),
        "fall_rate": falls / len(eps),
        "n_episodes": len(eps),
    }

# Print summary table
print("\n" + "="*80)
print("HYBRID NO-COACH (model_19999.pt) — 100-Episode 4-Environment Evaluation")
print("="*80)
print(f"{'Env':<12} {'Progress':>12} {'Zone':>8} {'Complete':>10} {'Falls':>8} {'Velocity':>12} {'Stability':>12}")
print("-"*80)
for env in ENVIRONMENTS:
    s = stats[env]
    print(f"{ENV_LABELS[env]:<12} {s['progress_mean']:>8.1f}±{s['progress_std']:<4.1f}"
          f" {s['zone_mean']:>6.1f}  {s['completion_rate']:>9.0%}  {s['fall_rate']:>7.0%}"
          f" {s['velocity_mean']:>8.3f}±{s['velocity_std']:<4.3f}"
          f" {s['stability_mean']:>8.3f}±{s['stability_std']:<4.3f}")
print("="*80)


def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: plots/{name}")


# ── FIGURE 1: Mean Progress Bar Chart ─────────────────────────────────────────
print("\nGenerating Figure 1: Mean Progress...")
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(ENVIRONMENTS))
means = [stats[e]["progress_mean"] for e in ENVIRONMENTS]
stds  = [stats[e]["progress_std"]  for e in ENVIRONMENTS]
colors = [ENV_COLORS[e] for e in ENVIRONMENTS]

bars = ax.bar(x, means, color=colors, alpha=0.85, zorder=3,
              yerr=stds, capsize=5, error_kw={"elinewidth": 1.5})
ax.axhline(COURSE_LENGTH, color="red", linestyle="--", linewidth=1.5,
           label=f"Course Length ({COURSE_LENGTH}m)", alpha=0.7)

# Add value labels on bars
for i, (bar, m) in enumerate(zip(bars, means)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 1,
            f"{m:.1f}m", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=12)
ax.set_ylabel("Mean Progress (m)", fontsize=12)
ax.set_title("Hybrid No-Coach — Mean Progress by Environment\n(100 episodes each, 49.5m course)",
             fontsize=13, fontweight="bold")
ax.set_ylim(0, 60)
ax.legend(fontsize=10, loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
fig.tight_layout()
save(fig, "fig1_mean_progress.png")


# ── FIGURE 2: Progress Distribution Box + Strip ──────────────────────────────
print("Generating Figure 2: Progress Distribution...")
fig, ax = plt.subplots(figsize=(9, 5))
progress_data = [np.array([e["progress"] for e in data[env]]) for env in ENVIRONMENTS]

bp = ax.boxplot(progress_data, positions=x, widths=0.5, patch_artist=True,
                medianprops={"color": "black", "linewidth": 2},
                whiskerprops={"linewidth": 1.2}, capprops={"linewidth": 1.2},
                showfliers=False)
for patch, col in zip(bp["boxes"], colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.6)

# Overlay individual points (jittered)
for i, env in enumerate(ENVIRONMENTS):
    vals = [e["progress"] for e in data[env]]
    jitter = np.random.normal(0, 0.06, len(vals))
    ax.scatter(x[i] + jitter, vals, c=ENV_COLORS[env], s=15, alpha=0.5, zorder=4)

ax.axhline(COURSE_LENGTH, color="red", linestyle="--", linewidth=1.2, alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=12)
ax.set_ylabel("Progress per Episode (m)", fontsize=12)
ax.set_title("Episode Progress Distribution — Hybrid No-Coach",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
fig.tight_layout()
save(fig, "fig2_progress_distribution.png")


# ── FIGURE 3: Zone Distribution Stacked Bar ──────────────────────────────────
print("Generating Figure 3: Zone Distribution...")
fig, ax = plt.subplots(figsize=(9, 5))
bottom = np.zeros(len(ENVIRONMENTS))
for zone in range(1, 6):
    counts = np.array([stats[env]["zone_counts"].get(zone, 0) for env in ENVIRONMENTS])
    ax.bar(x, counts, bottom=bottom, color=ZONE_COLORS[zone-1],
           label=f"Zone {zone}", alpha=0.9, zorder=3)
    # Add count labels on non-zero segments
    for i, c in enumerate(counts):
        if c > 3:
            ax.text(i, bottom[i] + c/2, str(c), ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")
    bottom += counts

ax.set_xticks(x)
ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=12)
ax.set_ylabel("Episode Count", fontsize=12)
ax.set_title("Zone Reached Distribution — Hybrid No-Coach\n(100 episodes per environment)",
             fontsize=13, fontweight="bold")
ax.legend(title="Zone", fontsize=9, title_fontsize=10, loc="upper right")
ax.set_ylim(0, 110)
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
fig.tight_layout()
save(fig, "fig3_zone_distribution.png")


# ── FIGURE 4: Velocity by Environment ────────────────────────────────────────
print("Generating Figure 4: Mean Velocity...")
fig, ax = plt.subplots(figsize=(9, 5))
vel_means = [stats[e]["velocity_mean"] for e in ENVIRONMENTS]
vel_stds  = [stats[e]["velocity_std"]  for e in ENVIRONMENTS]

bars = ax.bar(x, vel_means, color=colors, alpha=0.85, zorder=3,
              yerr=vel_stds, capsize=5, error_kw={"elinewidth": 1.5})
ax.axhline(2.235, color="red", linestyle="--", linewidth=1.2,
           label="Spot Max Speed (2.235 m/s)", alpha=0.7)

for bar, m in zip(bars, vel_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f"{m:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=12)
ax.set_ylabel("Mean Velocity (m/s)", fontsize=12)
ax.set_title("Mean Velocity by Environment — Hybrid No-Coach",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0, 2.6)
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
fig.tight_layout()
save(fig, "fig4_mean_velocity.png")


# ── FIGURE 5: Stability by Environment ───────────────────────────────────────
print("Generating Figure 5: Stability Score...")
fig, ax = plt.subplots(figsize=(9, 5))
stab_data = [np.array([e["stability_score"] for e in data[env]]) for env in ENVIRONMENTS]

bp = ax.boxplot(stab_data, positions=x, widths=0.5, patch_artist=True,
                medianprops={"color": "black", "linewidth": 2},
                whiskerprops={"linewidth": 1.2}, capprops={"linewidth": 1.2})
for patch, col in zip(bp["boxes"], colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.6)

ax.set_xticks(x)
ax.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=12)
ax.set_ylabel("Stability Score (lower = more stable)", fontsize=12)
ax.set_title("Stability Score Distribution — Hybrid No-Coach",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
fig.tight_layout()
save(fig, "fig5_stability_distribution.png")


# ── FIGURE 6: Progress Over Episodes (time series) ───────────────────────────
print("Generating Figure 6: Progress Over Episodes...")
fig, ax = plt.subplots(figsize=(12, 5))
for env in ENVIRONMENTS:
    progress = [e["progress"] for e in data[env]]
    episodes = range(len(progress))
    ax.plot(episodes, progress, color=ENV_COLORS[env], alpha=0.7,
            linewidth=1.5, label=ENV_LABELS[env])
    # Rolling average
    if len(progress) > 10:
        rolling = np.convolve(progress, np.ones(10)/10, mode='valid')
        ax.plot(range(9, len(progress)), rolling, color=ENV_COLORS[env],
                linewidth=2.5, linestyle="--", alpha=0.9)

ax.axhline(COURSE_LENGTH, color="red", linestyle=":", linewidth=1, alpha=0.5)
ax.set_xlabel("Episode Number", fontsize=12)
ax.set_ylabel("Progress (m)", fontsize=12)
ax.set_title("Progress Over Episodes — Hybrid No-Coach\n(solid = raw, dashed = 10-ep rolling avg)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="center right")
ax.set_ylim(0, 55)
ax.grid(linestyle="--", alpha=0.4)
fig.tight_layout()
save(fig, "fig6_progress_over_episodes.png")


# ── FIGURE 7: Completion & Fall Rate Summary ─────────────────────────────────
print("Generating Figure 7: Completion & Fall Rate...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Completion rate
comp_rates = [stats[e]["completion_rate"] * 100 for e in ENVIRONMENTS]
bars1 = ax1.bar(x, comp_rates, color=colors, alpha=0.85, zorder=3)
for bar, v in zip(bars1, comp_rates):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{v:.0f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=11)
ax1.set_ylabel("Completion Rate (%)", fontsize=11)
ax1.set_title("Course Completion Rate", fontsize=12, fontweight="bold")
ax1.set_ylim(0, 115)
ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

# Fall rate
fall_rates = [stats[e]["fall_rate"] * 100 for e in ENVIRONMENTS]
bars2 = ax2.bar(x, fall_rates, color=colors, alpha=0.85, zorder=3)
for bar, v in zip(bars2, fall_rates):
    ax2.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0) + 0.5,
             f"{v:.0f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=11)
ax2.set_ylabel("Fall Rate (%)", fontsize=11)
ax2.set_title("Fall Rate", fontsize=12, fontweight="bold")
ax2.set_ylim(0, max(fall_rates) * 1.5 + 5 if max(fall_rates) > 0 else 10)
ax2.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

fig.suptitle("Hybrid No-Coach — Safety Metrics (100 episodes each)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "fig7_completion_fall_rate.png")


# ── FIGURE 8: Summary Dashboard ──────────────────────────────────────────────
print("Generating Figure 8: Summary Dashboard...")
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Hybrid No-Coach (model_19999.pt) — 4-Environment Evaluation Dashboard\n"
             "100 episodes per environment | 49.5m course | [512, 256, 128] architecture",
             fontsize=14, fontweight="bold", y=0.98)
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

# Panel 1: Progress
ax = fig.add_subplot(gs[0, 0])
ax.bar(x, [stats[e]["progress_mean"] for e in ENVIRONMENTS], color=colors, alpha=0.85)
ax.axhline(COURSE_LENGTH, color="red", linestyle="--", linewidth=1, alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels([e[:5].title() for e in ENVIRONMENTS], fontsize=9)
ax.set_title("Mean Progress (m)", fontsize=11, fontweight="bold")
ax.set_ylim(0, 55)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# Panel 2: Zone
ax = fig.add_subplot(gs[0, 1])
ax.bar(x, [stats[e]["zone_mean"] for e in ENVIRONMENTS], color=colors, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([e[:5].title() for e in ENVIRONMENTS], fontsize=9)
ax.set_title("Mean Zone Reached", fontsize=11, fontweight="bold")
ax.set_ylim(0, 5.5)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# Panel 3: Velocity
ax = fig.add_subplot(gs[0, 2])
ax.bar(x, [stats[e]["velocity_mean"] for e in ENVIRONMENTS], color=colors, alpha=0.85)
ax.axhline(2.235, color="red", linestyle="--", linewidth=1, alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels([e[:5].title() for e in ENVIRONMENTS], fontsize=9)
ax.set_title("Mean Velocity (m/s)", fontsize=11, fontweight="bold")
ax.set_ylim(0, 2.5)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# Panel 4: Completion rate
ax = fig.add_subplot(gs[1, 0])
ax.bar(x, [stats[e]["completion_rate"]*100 for e in ENVIRONMENTS], color=colors, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([e[:5].title() for e in ENVIRONMENTS], fontsize=9)
ax.set_title("Completion Rate (%)", fontsize=11, fontweight="bold")
ax.set_ylim(0, 110)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# Panel 5: Stability
ax = fig.add_subplot(gs[1, 1])
ax.bar(x, [stats[e]["stability_mean"] for e in ENVIRONMENTS], color=colors, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([e[:5].title() for e in ENVIRONMENTS], fontsize=9)
ax.set_title("Mean Stability Score", fontsize=11, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.3)

# Panel 6: Time per episode
ax = fig.add_subplot(gs[1, 2])
ax.bar(x, [stats[e]["time_mean"] for e in ENVIRONMENTS], color=colors, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([e[:5].title() for e in ENVIRONMENTS], fontsize=9)
ax.set_title("Mean Episode Time (s)", fontsize=11, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.3)

save(fig, "fig8_dashboard.png")


# ── FIGURE 9: Stability vs Progress Scatter ──────────────────────────────────
print("Generating Figure 9: Stability vs Progress...")
fig, ax = plt.subplots(figsize=(9, 6))
markers = {"friction": "o", "grass": "^", "boulder": "s", "stairs": "D"}
for env in ENVIRONMENTS:
    xs = [e["progress"]        for e in data[env]]
    ys = [e["stability_score"] for e in data[env]]
    ax.scatter(xs, ys, c=ENV_COLORS[env], marker=markers[env], s=40, alpha=0.6,
               label=ENV_LABELS[env], edgecolors="white", linewidths=0.3)
ax.set_xlabel("Progress (m)", fontsize=12)
ax.set_ylabel("Stability Score (lower = more stable)", fontsize=12)
ax.set_title("Stability vs Progress — Hybrid No-Coach", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(linestyle="--", alpha=0.4)
fig.tight_layout()
save(fig, "fig9_stability_vs_progress.png")


print(f"\nAll figures saved to: {PLOTS_DIR}")
print("Done.")
