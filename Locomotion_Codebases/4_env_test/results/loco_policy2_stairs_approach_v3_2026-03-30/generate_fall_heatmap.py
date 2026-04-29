"""
Fall Locations heatmap for Locomotion Policy 2 — stairs_approach v3.
Output: report/fall_heatmap.png
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "report")
os.makedirs(PLOTS_DIR, exist_ok=True)

recs = [json.loads(l) for l in open(os.path.join(BASE_DIR, "stairs_approach_rough_episodes.jsonl"))]

fall_counts = np.zeros(5)
for ep in recs:
    if ep["fall_detected"] and ep.get("fall_zone") is not None:
        z = int(ep["fall_zone"]) - 1
        if 0 <= z < 5:
            fall_counts[z] += 1

zone_labels = ["Zone 1\n(0-10m)", "Zone 2\n(10-20m)", "Zone 3\n(20-30m)",
               "Zone 4\n(30-40m)", "Zone 5\n(40-50m)"]

fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")

data = fall_counts.reshape(1, 5)
im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(fall_counts.max(), 1))

for j in range(5):
    val = int(fall_counts[j])
    text_color = "white" if val > fall_counts.max() * 0.6 else "black"
    ax.text(j, 0, str(val) if val > 0 else "",
            ha="center", va="center", fontsize=14, fontweight="bold", color=text_color)

ax.set_xticks(np.arange(5))
ax.set_xticklabels(zone_labels, fontsize=10)
ax.set_yticks([])
ax.set_title("Fall Locations — Loco Policy 2 Stairs Approach (v3)", fontsize=13, fontweight="bold")

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
cbar.set_label("Falls", fontsize=10)
cbar.ax.tick_params(labelsize=9)

fig.tight_layout()
out = os.path.join(PLOTS_DIR, "fall_heatmap.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
