"""
add_risers.py
=============
Injects solid vertical riser faces into the SM_Staircase* USD meshes.

Uses face normals to identify real horizontal tread surfaces, ignoring
structural beams, railings, and other vertical elements. For each
consecutive tread pair, places a riser quad at the step edge.

Usage:
    conda activate isaacLab
    cd Experiments/Colby/FW_Stairs_Riser_Project/usd_source
    python add_risers.py
"""

import shutil
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

THIS_DIR   = Path(__file__).resolve().parent
SUBUSD_DIR = THIS_DIR.parents[4] / "Collected_Final_World" / "SubUSDs"

TARGETS = [
    "SM_Staircase_02.usd",
    "SM_StaircaseHalf_01.usd",
    "SM_StaircaseHalf_02.usd",
    "SM_Staircase_01.usd",
]

HORIZONTAL_THRESHOLD = 0.85   # |normal_z| must exceed this to count as a tread face
MIN_FACE_AREA        = 50.0   # cm² — skip tiny faces (rivets, edge artefacts)
MIN_TREAD_SPAN       = 15.0   # cm — skip tread faces narrow in either dimension (structural beams)
Z_GROUP_TOL          = 1.5    # cm — merge tread faces within this Z range
MIN_RISER_HEIGHT     = 8.0    # cm — skip tread pairs closer than this (structural noise)
MAX_RISER_HEIGHT     = 35.0   # cm — skip tread pairs taller than this (landing transitions)
MIN_RISER_SPAN       = 10.0   # cm — skip risers narrower than this


# ── Geometry helpers ──────────────────────────────────────────────────────────

def triangle_normal_and_area(v0, v1, v2):
    e1 = v1 - v0
    e2 = v2 - v0
    cross = np.cross(e1, e2)
    area = np.linalg.norm(cross) / 2.0
    if area < 1e-12:
        return np.array([0, 0, 1]), 0.0
    return cross / (2.0 * area), area


def get_tread_faces(pts, fc, fi):
    """Return list of face dicts for horizontal (tread) faces only."""
    treads = []
    cursor = 0
    for count in fc:
        idx = list(fi[cursor:cursor + count])
        cursor += count
        if count < 3:
            continue
        verts = pts[idx]
        # Compute normal from first triangle of the face
        n, area = triangle_normal_and_area(verts[0], verts[1], verts[2])
        if area < MIN_FACE_AREA:
            continue
        if abs(n[2]) < HORIZONTAL_THRESHOLD:
            continue
        x_span = float(verts[:, 0].max() - verts[:, 0].min())
        y_span = float(verts[:, 1].max() - verts[:, 1].min())
        if min(x_span, y_span) < MIN_TREAD_SPAN:
            continue
        treads.append({
            "z":  float(verts[:, 2].mean()),
            "x0": float(verts[:, 0].min()), "x1": float(verts[:, 0].max()),
            "y0": float(verts[:, 1].min()), "y1": float(verts[:, 1].max()),
            "cx": float(verts[:, 0].mean()), "cy": float(verts[:, 1].mean()),
        })
    return treads


def group_tread_faces(faces, tol=Z_GROUP_TOL):
    """Merge tread faces that share the same Z level into one bounding box."""
    if not faces:
        return []
    faces = sorted(faces, key=lambda f: f["z"])
    groups = []
    grp = [faces[0]]
    z_ref = faces[0]["z"]
    for f in faces[1:]:
        if f["z"] - z_ref <= tol:
            grp.append(f)
        else:
            groups.append(grp)
            grp = [f]
            z_ref = f["z"]
    groups.append(grp)

    merged = []
    for grp in groups:
        merged.append({
            "z":     float(np.mean([f["z"]  for f in grp])),
            "z_max": float(max(f["z"]        for f in grp)),
            "x0":    float(min(f["x0"] for f in grp)),
            "x1":    float(max(f["x1"] for f in grp)),
            "y0":    float(min(f["y0"] for f in grp)),
            "y1":    float(max(f["y1"] for f in grp)),
            "cx":    float(np.mean([f["cx"] for f in grp])),
            "cy":    float(np.mean([f["cy"] for f in grp])),
        })
    return merged


# ── Core ──────────────────────────────────────────────────────────────────────

def process_usd(bak_path: Path, out_path: Path) -> bool:
    # Start from clean .bak original
    shutil.copy2(str(bak_path), str(out_path))
    stage = Usd.Stage.Open(str(out_path))
    if stage is None:
        print(f"  ERROR: cannot open"); return False

    mesh_prim = next((p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)), None)
    if mesh_prim is None:
        print("  ERROR: no mesh prim"); return False

    mesh = UsdGeom.Mesh(mesh_prim)
    pts  = np.array([[v[0], v[1], v[2]] for v in mesh.GetPointsAttr().Get()], dtype=np.float64)
    fc   = list(mesh.GetFaceVertexCountsAttr().Get())
    fi   = list(mesh.GetFaceVertexIndicesAttr().Get())
    print(f"  source    verts={len(pts):6d}  faces={len(fc):5d}")

    # Detect tread faces by normal direction
    tread_faces  = get_tread_faces(pts, fc, fi)
    tread_groups = group_tread_faces(tread_faces)
    print(f"  horizontal faces: {len(tread_faces)}  →  tread levels: {len(tread_groups)}")
    for i, g in enumerate(tread_groups):
        print(f"    G{i:02d} z={g['z']:7.1f} z_max={g['z_max']:7.1f}  "
              f"cx={g['cx']:7.1f} cy={g['cy']:7.1f}  "
              f"x=[{g['x0']:7.1f},{g['x1']:7.1f}]  y=[{g['y0']:7.1f},{g['y1']:7.1f}]")

    # Prepend a virtual floor group so the bottom riser gets generated
    if tread_groups:
        first = tread_groups[0]
        floor_z = float(pts[:, 2].min())
        tread_groups = [{
            "z": floor_z, "z_max": floor_z,
            "x0": first["x0"], "x1": first["x1"],
            "y0": first["y0"], "y1": first["y1"],
            "cx": first["cx"] + 1.0, "cy": first["cy"],
        }] + tread_groups

    new_pts = pts.tolist()
    riser_count = 0
    skipped = 0

    for i in range(1, len(tread_groups)):
        lo, hi = tread_groups[i - 1], tread_groups[i]
        z_lo, z_hi = lo["z"], hi["z"]
        dz = z_hi - z_lo

        # Skip structural noise (too close) and landing transitions (too tall)
        if dz < MIN_RISER_HEIGHT or dz > MAX_RISER_HEIGHT:
            skipped += 1
            continue

        dx = hi["cx"] - lo["cx"]
        dy = hi["cy"] - lo["cy"]

        # Use tread top surfaces for Z
        z_lo = lo["z_max"]
        z_hi = hi["z_max"]

        if abs(dx) >= abs(dy):
            # Riser at the front face of the upper tread
            x_r  = hi["x1"] if dx < 0 else hi["x0"]
            y_lo = max(lo["y0"], hi["y0"])
            y_hi = min(lo["y1"], hi["y1"])
            if (y_hi - y_lo) < MIN_RISER_SPAN:
                skipped += 1
                continue
            print(f"    riser {riser_count+1:2d}: z=[{z_lo:.1f},{z_hi:.1f}] x={x_r:.1f} y=[{y_lo:.1f},{y_hi:.1f}]")
            b = len(new_pts)
            new_pts += [
                [x_r, y_lo, z_lo], [x_r, y_hi, z_lo],
                [x_r, y_hi, z_hi], [x_r, y_lo, z_hi],
            ]
        else:
            # Riser at the front face of the upper tread
            y_r  = hi["y1"] if dy < 0 else hi["y0"]
            x_lo = max(lo["x0"], hi["x0"])
            x_hi = min(lo["x1"], hi["x1"])
            if (x_hi - x_lo) < MIN_RISER_SPAN:
                skipped += 1
                continue
            print(f"    riser {riser_count+1:2d}: z=[{z_lo:.1f},{z_hi:.1f}] y={y_r:.1f} x=[{x_lo:.1f},{x_hi:.1f}]")
            b = len(new_pts)
            new_pts += [
                [x_lo, y_r, z_lo], [x_hi, y_r, z_lo],
                [x_hi, y_r, z_hi], [x_lo, y_r, z_hi],
            ]

        fc += [3, 3]
        fi += [b, b + 1, b + 2, b, b + 2, b + 3]
        riser_count += 1

    print(f"  skipped: {skipped}  risers added: {riser_count}")
    if riser_count == 0:
        print("  WARN: no risers generated")
        return False

    mesh.GetPointsAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(p[0], p[1], p[2]) for p in new_pts])
    )
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(fc))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(fi))
    print(f"  updated   verts={len(new_pts):6d}  faces={len(fc):5d}")

    # Ensure exact-mesh collision
    if not mesh_prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(mesh_prim)
    if not mesh_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        mc = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        mc.CreateApproximationAttr().Set("none")
    else:
        UsdPhysics.MeshCollisionAPI(mesh_prim).GetApproximationAttr().Set("none")

    stage.GetRootLayer().Save()
    print(f"  saved  → {out_path.name}")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not SUBUSD_DIR.exists():
        print(f"WARN: SubUSDs dir not found at {SUBUSD_DIR}")

    success = 0
    for name in TARGETS:
        bak = THIS_DIR / (name + ".bak")
        out = THIS_DIR / name
        print(f"\n=== {name} ===")
        if not bak.exists():
            print(f"  SKIP — .bak not found"); continue
        if process_usd(bak, out):
            if SUBUSD_DIR.exists():
                dst = SUBUSD_DIR / name
                shutil.copy2(str(out), str(dst))
                print(f"  copied → {dst}")
            success += 1

    print(f"\n{'='*50}")
    print(f"Done: {success}/{len(TARGETS)} USDs updated")


if __name__ == "__main__":
    main()
