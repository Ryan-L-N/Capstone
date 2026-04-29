"""Inspect Final World staircase USDs for dimensions.

Prints bounding box, mesh count, and inferred riser height + run depth.
Use these numbers to configure MeshPyramidStairsTerrainCfg in Loco_Policy_5_Final_Capstone_Policy
for Phase-7 training.

Run with isaaclab311 conda env:
    python inspect_fw_stairs.py
"""

import os
from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "width": 64, "height": 64})

from pxr import Usd, UsdGeom, Gf
import numpy as np

ROOT = r"C:\Users\Gabriel Santiago\OneDrive\Desktop\Collected_Final_World\SubUSDs"
TARGETS = [
    "SM_Staircase_01.usd",
    "SM_Staircase_02.usd",
    "SM_StaircaseHalf_01.usd",
    "SM_StaircaseHalf_02.usd",
]


def measure_stair(usd_path):
    print(f"\n{'='*60}")
    print(f"  {os.path.basename(usd_path)}")
    print(f"{'='*60}")
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        print(f"  ERROR: could not open {usd_path}")
        return

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                               [UsdGeom.Tokens.default_])

    # Total bbox
    root = stage.GetPseudoRoot()
    total_bbox = cache.ComputeWorldBound(root).ComputeAlignedRange()
    sz = total_bbox.GetSize()
    print(f"  Total bbox (m):   {sz[0]:.3f} W x {sz[1]:.3f} D x {sz[2]:.3f} H")
    print(f"    min: ({total_bbox.GetMin()[0]:.2f}, {total_bbox.GetMin()[1]:.2f}, {total_bbox.GetMin()[2]:.2f})")
    print(f"    max: ({total_bbox.GetMax()[0]:.2f}, {total_bbox.GetMax()[1]:.2f}, {total_bbox.GetMax()[2]:.2f})")

    # Collect all mesh prim z-extents to infer step pattern
    mesh_z_centers = []
    mesh_count = 0
    for p in stage.Traverse():
        if p.IsA(UsdGeom.Mesh):
            mesh_count += 1
            bbox = cache.ComputeWorldBound(p).ComputeAlignedRange()
            cz = (bbox.GetMin()[2] + bbox.GetMax()[2]) * 0.5
            mesh_z_centers.append(cz)

    print(f"  Mesh count: {mesh_count}")

    if mesh_z_centers:
        zs = sorted(mesh_z_centers)
        # Histogram bins of 5cm — likely each stair tread is one mesh
        unique_z = []
        last_z = -1e9
        for z in zs:
            if z - last_z > 0.03:  # 3cm dedup tolerance
                unique_z.append(z)
                last_z = z
        if len(unique_z) > 2:
            steps = np.diff(unique_z)
            avg_riser = float(np.mean(steps))
            print(f"  Unique z-levels: {len(unique_z)} ({unique_z[0]:.3f} → {unique_z[-1]:.3f})")
            print(f"  Inferred avg riser height (z spacing): {avg_riser:.3f} m")
            print(f"  Inferred total rise: {unique_z[-1] - unique_z[0]:.3f} m")

    # Infer run depth from longer of x/y bbox dims
    run_dim = max(sz[0], sz[1])
    n_steps = max(1, int(round(sz[2] / 0.18)))  # assume ~18cm risers
    inferred_run = run_dim / n_steps
    print(f"  Inferred run depth (assuming {n_steps} steps): {inferred_run:.3f} m")


for f in TARGETS:
    p = os.path.join(ROOT, f)
    if os.path.exists(p):
        measure_stair(p)
    else:
        print(f"NOT FOUND: {p}")

app.close()
