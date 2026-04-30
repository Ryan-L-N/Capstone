import shutil
from pxr import Usd, UsdGeom, UsdPhysics

src = "SM_Staircase_02.usd"
dst = r"C:\Users\Colby\Documents\AI2C\Class\Capstone\Collected_Final_World\SubUSDs\SM_Staircase_02.usd"

stage = Usd.Stage.Open(src)
for p in stage.Traverse():
    if p.IsA(UsdGeom.Mesh):
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(p)
        if not p.HasAPI(UsdPhysics.MeshCollisionAPI):
            mc = UsdPhysics.MeshCollisionAPI.Apply(p)
            mc.CreateApproximationAttr().Set("none")
        else:
            UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Set("none")
        print("Done:", p.GetPath())

stage.GetRootLayer().Save()
shutil.copy2(src, dst)
print("Copied to SubUSDs")
