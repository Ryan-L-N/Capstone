"""
Fall Locations heatmap for Locomotion Policy 2 (Mason Hybrid).
Uses stairs_approach data for the stairs column.
Output: report/mason_hybrid_fall_heatmap_approach.png
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

STAIRS_APPROACH_DIR = os.path.join(
    os.path.dirname(BASE_DIR),
    "loco_policy2_stairs_approach_2026-03-30"
)

ENVIRONMENTS = ["friction", "grass", "boulder", "stairs"]
ENV_LABELS   = {"friction": "Friction", "grass": "Grass", "boulder": "Boulder", "stairs": "Stairs"}

def load_jsonl(env):
    if env == "stairs":
        path = os.path.join(STAIRS_APPROACH_DIR, "stairs_approach_rough_episodes.jsonl")
    else:
        path = os.path.join(BASE_DIR, f"{env}_rough_episodes.jsonl")
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

episodes = {env: load_jsonl(env) for env in ENVIRONMENTS}

fall_matrix = np.zeros((len(ENVIRONMENTS), 5))
for i, env in enumerate(ENVIRONMENTS):
    for ep in episodes[env]:
        if ep["fall_detected"] and ep.get("fall_zone") is not None:
            z = int(ep["fall_zone"]) - 1
            if 0 <= z < 5:
                fall_matrix[i, z] += 1

zone_labels_flipped = [
    "Zone 5\n(40-50m)", "Zone 4\n(30-40m)", "Zone 3\n(20-30m)",
    "Zone 2\n(10-20m)", "Zone 1\n(0-10m)"
]
env_labels_list = [ENV_LABELS[e] for e in ENVIRONMENTS]

fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")

fall_T = fall_matrix.T[::-1]
im = ax.imshow(fall_T, cmap="YlOrRd", aspect="auto", vmin=0,
               vmax=max(fall_T.max(), 1))

for i in range(5):
    for j in range(len(ENVIRONMENTS)):
        val = int(fall_T[i, j])
        text_color = "white" if val > fall_T.max() * 0.6 else "black"
        ax.text(j, i, str(val) if val > 0 else "",
                ha="center", va="center", fontsize=13, fontweight="bold", color=text_color)

ax.set_xticks(np.arange(len(ENVIRONMENTS)))
ax.set_xticklabels(env_labels_list, fontsize=12)
ax.set_yticks(np.arange(5))
ax.set_yticklabels(zone_labels_flipped, fontsize=9)
ax.set_title("Fall Locations", fontsize=15, fontweight="bold")

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
cbar.set_label("Falls", fontsize=10)
cbar.ax.tick_params(labelsize=9)

fig.tight_layout()
out = os.path.join(PLOTS_DIR, "mason_hybrid_fall_heatmap_approach.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
