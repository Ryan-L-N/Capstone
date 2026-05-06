"""Quick check that the RiserCheat group is in the modified USD with collision."""
import os
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 64, "height": 64})

from pxr import Usd, UsdGeom, UsdPhysics

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TARGETS = ["SM_Staircase_02.usd", "SM_Staircase_01.usd",
           "SM_StaircaseHalf_01.usd", "SM_StaircaseHalf_02.usd"]

for t in TARGETS:
    p = os.path.join(THIS_DIR, t)
    print(f"\n=== {t} ===", flush=True)
    s = Usd.Stage.Open(p)
    if s is None:
        print("  open failed"); continue
    riser_count = 0
    for prim in s.Traverse():
        path = str(prim.GetPath())
        if "Flight_" in path or "RiserCheat" in path:
            vis = prim.GetAttribute("visibility")
            vis_val = vis.Get() if vis else None
            has_coll = prim.HasAPI(UsdPhysics.CollisionAPI)
            has_mesh_coll = prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            approx = None
            if has_mesh_coll:
                approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            ptype = prim.GetTypeName()
            print(f"  {path}  type={ptype}  vis={vis_val}  CollisionAPI={has_coll}  MeshColl={has_mesh_coll} approx={approx}", flush=True)
            if "Flight_" in path and prim.IsA(UsdGeom.Mesh):
                m = UsdGeom.Mesh(prim)
                pts = m.GetPointsAttr().Get()
                if pts:
                    print(f"    points (cm): {[tuple(p) for p in pts]}", flush=True)
            riser_count += 1
    print(f"  riser-related prims: {riser_count}", flush=True)

app.close()
