from pxr import Usd, UsdGeom

for name in ["SM_Staircase_02.usd", "SM_Staircase_02.usd.bak"]:
    try:
        stage = Usd.Stage.Open(name)
        for p in stage.Traverse():
            if p.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(p)
                pts = mesh.GetPointsAttr().Get()
                fc  = mesh.GetFaceVertexCountsAttr().Get()
                print(f"{name}: verts={len(pts)}, faces={len(fc)}")
    except Exception as e:
        print(f"{name}: ERROR — {e}")
