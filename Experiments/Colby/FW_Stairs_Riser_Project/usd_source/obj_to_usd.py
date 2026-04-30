"""
Convert an OBJ exported from Blender to USD and apply CollisionAPI.
Usage:
    python obj_to_usd.py path\to\SM_Staircase_02.obj
"""
import sys
import shutil
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Vt


def parse_obj(obj_path):
    verts = []
    faces = []
    with open(obj_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == 'f':
                # OBJ face indices are 1-based; strip texture/normal refs
                indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                if len(indices) == 3:
                    faces.append(indices)
                elif len(indices) == 4:
                    # Triangulate quad
                    faces.append([indices[0], indices[1], indices[2]])
                    faces.append([indices[0], indices[2], indices[3]])
    return verts, faces


def main():
    if len(sys.argv) < 2:
        print("Usage: python obj_to_usd.py <path_to.obj>")
        sys.exit(1)

    obj_path = Path(sys.argv[1])
    out_usd  = Path(__file__).parent / "SM_Staircase_02.usd"
    dst_usd  = Path(r"C:\Users\Colby\Documents\AI2C\Class\Capstone\Collected_Final_World\SubUSDs\SM_Staircase_02.usd")

    print(f"Parsing {obj_path.name}...")
    verts, faces = parse_obj(obj_path)
    print(f"  verts={len(verts)}  tri_faces={len(faces)}")

    stage = Usd.Stage.CreateNew(str(out_usd))
    stage.SetMetadata("upAxis", "Z")
    stage.SetMetadata("metersPerUnit", 0.01)

    xform = UsdGeom.Xform.Define(stage, "/SM_Staircase_02")
    mesh_prim = UsdGeom.Mesh.Define(stage, "/SM_Staircase_02/Mesh")

    mesh_prim.GetPointsAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(*v) for v in verts])
    )
    mesh_prim.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(faces)))
    mesh_prim.GetFaceVertexIndicesAttr().Set(
        Vt.IntArray([i for tri in faces for i in tri])
    )

    # Collision
    p = mesh_prim.GetPrim()
    UsdPhysics.CollisionAPI.Apply(p)
    mc = UsdPhysics.MeshCollisionAPI.Apply(p)
    mc.CreateApproximationAttr().Set("none")

    stage.GetRootLayer().Save()
    print(f"Saved → {out_usd}")

    shutil.copy2(str(out_usd), str(dst_usd))
    print(f"Copied → SubUSDs/")


if __name__ == "__main__":
    main()
