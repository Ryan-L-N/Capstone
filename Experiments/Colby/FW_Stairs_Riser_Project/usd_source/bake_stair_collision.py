"""Bake CollisionAPI + MeshCollisionAPI(triangleMesh) into all SM_Staircase USDs.

Modifies the source USDs in place so that ANY scene referencing them gets
proper stair-step collision automatically. triangleMesh approximation
preserves the exact stair geometry (vs convex hull which would round the
risers off into a ramp).

Run once with the isaaclab311 conda env:
    python bake_stair_collision.py

Each USD gets backed up to *.usd.bak before modification.
"""
import os
import shutil
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 64, "height": 64})

from pxr import Usd, UsdPhysics, UsdGeom

ROOT = r"C:\Users\Gabriel Santiago\OneDrive\Desktop\Collected_Final_World\SubUSDs"
TARGETS = [
    "SM_Staircase_01.usd",
    "SM_Staircase_02.usd",
    "SM_StaircaseHalf_01.usd",
    "SM_StaircaseHalf_02.usd",
]


def bake_collision(usd_path):
    print(f"\n=== {os.path.basename(usd_path)} ===", flush=True)

    # Backup
    backup = usd_path + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(usd_path, backup)
        print(f"  Backup -> {os.path.basename(backup)}", flush=True)
    else:
        print(f"  (existing backup at {os.path.basename(backup)})", flush=True)

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        print(f"  ERROR: open failed", flush=True)
        return False

    n_meshes = 0
    n_added = 0
    for p in stage.Traverse():
        if p.IsA(UsdGeom.Mesh):
            n_meshes += 1
            # Apply CollisionAPI (basic collision marker)
            if not p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(p)
                n_added += 1
            # Apply MeshCollisionAPI with triangleMesh approximation —
            # exact geometry, preserves stair-step ridges. Convex hull
            # would round the steps into a smooth ramp.
            if not p.HasAPI(UsdPhysics.MeshCollisionAPI):
                mc = UsdPhysics.MeshCollisionAPI.Apply(p)
                mc.CreateApproximationAttr().Set("none")  # "none" = exact triangle mesh
            else:
                mc = UsdPhysics.MeshCollisionAPI(p)
                cur = mc.GetApproximationAttr().Get()
                if cur != "none":
                    mc.GetApproximationAttr().Set("none")

    print(f"  Meshes processed: {n_meshes}, CollisionAPI added: {n_added}", flush=True)

    if n_meshes == 0:
        print("  WARN: no meshes found, nothing to bake", flush=True)
        return False

    # Save in place — overwrites the binary USDC file
    stage.GetRootLayer().Save()
    print(f"  Saved {os.path.basename(usd_path)}", flush=True)
    return True


with open(r"C:\Users\Gabriel Santiago\stair_bake_report.txt", "w") as out:
    success = 0
    for t in TARGETS:
        ok = bake_collision(os.path.join(ROOT, t))
        out.write(f"{t}: {'OK' if ok else 'FAILED'}\n")
        if ok:
            success += 1
    out.write(f"\n{success}/{len(TARGETS)} files updated\n")

app.close()
