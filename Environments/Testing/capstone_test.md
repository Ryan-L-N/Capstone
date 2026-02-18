# Capstone Comparative Policy Evaluation Test Plan

**Project:** AI2C Tech Capstone - Immersive Modeling and Simulation for Autonomy
**Date:** February 17, 2026
**Version:** 1.0
**Platform:** NVIDIA Isaac Sim 5.1.0 / Isaac Lab + RSL-RL (PPO)
**Hardware:** NVIDIA H100 NVL 80GB (headless execution)

---

## 1. Overview and Objectives

This document defines a rigorous comparative evaluation between two Boston Dynamics Spot locomotion policies across four progressively challenging environments. The goal is to quantify the performance gap between the NVIDIA pre-trained flat terrain baseline and our custom rough terrain policy trained for 48 hours on the H100 cluster.

**Research Question:** Does the rough terrain policy (trained with height scan perception and terrain curriculum) significantly outperform the flat terrain baseline across diverse real-world-analog terrain challenges?

**Test Scale:**
- **Headless:** 4 environments x 2 policies x 1,000 episodes = **8,000 statistical episodes** (512 parallel envs, ~1-1.5 hours)
- **Rendered:** 4 environments x 2 policies x 10 episodes = **80 visualization episodes** (5 parallel envs with video capture, ~3.5-4 hours)
- **Total:** 8,080 episodes, estimated wall-clock time: **~5-5.5 hours**

---

## 2. Policy Specifications

### 2.1 Policy A: NVIDIA Flat Terrain Baseline (`SpotFlatTerrainPolicy`)

| Parameter | Value |
|-----------|-------|
| Source | NVIDIA Isaac Lab pre-trained checkpoint |
| Observation dim | 48 (proprioceptive only) |
| Observation breakdown | lin_vel(3) + ang_vel(3) + gravity(3) + cmd(3) + joint_pos(12) + joint_vel(12) + last_action(12) |
| Network | MLP [512, 256, 128] ELU activation |
| Action dim | 12 (joint position targets) |
| Action scale | 0.25 |
| Control rate | 50 Hz (decimation=10 at 500 Hz physics) |
| Training config | `SpotFlatPPORunnerCfg` — 20,000 iter, lr=1e-3, entropy=0.0025 |
| Terrain perception | **None** |
| Config file | `isaac_lab_spot_configs/agents/rsl_rl_ppo_cfg.py` |

### 2.2 Policy B: Custom Rough Terrain Policy (`SpotRoughTerrainPolicy`)

| Parameter | Value |
|-----------|-------|
| Source | Custom 48h H100 training, checkpoint `model_29999.pt` (6.6 MB) |
| Observation dim | 208 (48 proprioceptive + 160 height scan) |
| Height scanner | RayCaster 16x10 grid, 0.1m resolution, 1.6m x 1.0m area |
| Network | MLP [512, 256, 128] ELU activation |
| Action dim | 12 (joint position targets) |
| Action scale | 0.25 |
| Control rate | 50 Hz (decimation=10 at 500 Hz physics) |
| Training config | `spot_rough_48h_cfg.py` — 30,000 iter, lr=3e-4, entropy=0.008, 8192 envs |
| Terrain perception | 160-dim height scan with noise (Unoise -0.1 to 0.1) |
| Final training reward | +143.74, terrain level 4.42/5.0 |
| Config file | `spot_rough_48h_cfg.py` |

### 2.3 Key Architectural Differences

| Feature | Flat Policy | Rough Policy |
|---------|------------|--------------|
| Height scan | No | Yes (160-dim) |
| Observation noise | Yes | Yes (wider) |
| Terrain curriculum | No | Yes (5 levels) |
| Friction randomization | 0.5-1.25 static | 0.5-1.25 static |
| Foot clearance weight | 0.5 | 2.0 (4x higher) |
| Base orientation penalty | -3.0 | -5.0 (stronger upright) |
| Value loss coefficient | 0.5 | 1.0 |
| Entropy coefficient | 0.0025 | 0.005-0.008 |

---

## 3. Test Environment Specifications

All four environments share a common arena format:
- **Arena dimensions:** 30m wide (Y-axis) x 50m long (X-axis)
- **Zone structure:** 5 zones of 10m each along the X-axis (direction of travel)
- **Difficulty progression:** Each zone increases the target variable
- **Spawn position:** x=0m, y=15m (center), z=robot standing height
- **Goal:** Traverse from x=0m to x=50m

### Common Physics Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Physics timestep | 0.002s (500 Hz) | `rough_env_cfg.py:398` |
| Control decimation | 10 (50 Hz control) | `rough_env_cfg.py:396` |
| Friction combine mode | Multiply | `rough_env_cfg.py:402` |
| Restitution combine mode | Multiply | `rough_env_cfg.py:403` |
| GPU PhysX pipeline | Enabled | Required for policy accuracy |
| Gravity | -9.81 m/s² (Z-down) | Isaac Sim default |

---

### 3.1 Environment 1: Friction Surface

**Purpose:** Test locomotion stability across decreasing friction coefficients, starting from 60-grit sandpaper and degrading to oil-slicked steel. The robot begins on familiar high-friction terrain and faces progressively harder traction challenges. Isolates the robot's ability to maintain traction without terrain obstacles.

**Terrain geometry:** Flat ground plane (no elevation changes)

#### Zone Specifications

| Zone | X Range | Surface Type | mu_static | mu_dynamic | Real-World Analog |
|------|---------|-------------|-----------|-----------|-------------------|
| 1 | 0-10m | Very high friction | 0.90 | 0.80 | 60-grit sandpaper |
| 2 | 10-20m | High friction | 0.60 | 0.50 | Dry rubber on concrete |
| 3 | 20-30m | Medium friction | 0.35 | 0.25 | Wet concrete |
| 4 | 30-40m | Low friction | 0.15 | 0.08 | Wet ice |
| 5 | 40-50m | Ultra-low friction | 0.05 | 0.02 | Oil on polished steel |

#### Isaac Sim Implementation

Each zone is a separate `UsdGeom.Cube` ground segment (30m x 10m x 0.1m thick) with its own `UsdPhysics.MaterialAPI`:

```python
# Per-zone material creation pattern (from spot_obstacle_course.py)
from pxr import UsdPhysics, PhysxSchema, UsdShade

def create_friction_zone(stage, zone_idx, x_start, mu_s, mu_d):
    """Create a 30m x 10m ground segment with specified friction."""
    zone_path = f"/World/FrictionSurface/Zone_{zone_idx}"

    # Create ground geometry
    cube = UsdGeom.Cube.Define(stage, f"{zone_path}/Ground")
    cube.GetSizeAttr().Set(1.0)
    # Scale: 10m long (X), 30m wide (Y), 0.1m thick (Z)
    UsdGeom.XformCommonAPI(cube).SetScale(Gf.Vec3f(10.0, 30.0, 0.1))
    UsdGeom.XformCommonAPI(cube).SetTranslate(
        Gf.Vec3d(x_start + 5.0, 15.0, -0.05)
    )

    # Apply collision
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    # Create physics material
    mat_path = f"{zone_path}/Material"
    material = UsdShade.Material.Define(stage, mat_path)
    physics_mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_mat.CreateStaticFrictionAttr(mu_s)
    physics_mat.CreateDynamicFrictionAttr(mu_d)
    physics_mat.CreateRestitutionAttr(0.01)

    # Set combine mode to multiply (match training config)
    physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_mat.CreateFrictionCombineModeAttr().Set("multiply")
    physx_mat.CreateRestitutionCombineModeAttr().Set("multiply")

    # Bind material to ground
    binding = UsdShade.MaterialBindingAPI.Apply(cube.GetPrim())
    binding.Bind(material)
```

#### Expected Challenges

- **Zone 1 (sandpaper):** Both policies trained in this friction range (0.5-1.25), should perform well. Provides a warm-up zone.
- **Zone 2 (dry rubber):** Still within training distribution. Both policies should handle comfortably.
- **Zone 3 (wet concrete):** Moderate challenge. mu=0.35 is below training range (0.5-1.25), so both policies encounter out-of-distribution friction.
- **Zone 4 (wet ice):** Flat policy likely fails here. Rough policy may adapt via conservative stepping.
- **Zone 5 (oil):** Both policies expected to struggle. mu=0.05 provides almost no traction — any lateral force causes sliding. The robot's feet will slip on nearly every step.

#### Effective Friction Calculation

Since friction combine mode = `multiply`, the effective friction between robot foot and ground is:

```
mu_effective = mu_robot_foot × mu_ground_surface
```

Robot foot friction (from training randomization): mu_static = 0.5-1.25, mu_dynamic = 0.4-1.0

| Zone | mu_ground | mu_robot (avg=0.875) | mu_effective |
|------|-----------|---------------------|-------------|
| 1 | 0.90 | 0.875 | 0.788 |
| 2 | 0.60 | 0.875 | 0.525 |
| 3 | 0.35 | 0.875 | 0.306 |
| 4 | 0.15 | 0.875 | 0.131 |
| 5 | 0.05 | 0.875 | 0.044 |

---

### 3.2 Environment 2: Grass / Fluid Resistance

**Purpose:** Test the robot's ability to push through increasing resistance forces, simulating terrain from a light fluid (alcohol viscosity) to dense grass/brush. This tests motor torque capacity, gait adaptation, and energy efficiency.

**Terrain geometry:** Flat ground with proxy stalk obstacles + scripted drag forces

#### Zone Specifications

| Zone | X Range | Stalk Density | Stalk Height | Drag Coeff (c_d) | mu_ground | Real-World Analog |
|------|---------|--------------|-------------|-------------------|-----------|-------------------|
| 1 | 0-10m | 0 stalks/m² | N/A | 0.5 N-s/m | 0.80 | Light fluid (ethanol) |
| 2 | 10-20m | 2 stalks/m² | 0.15-0.25m | 2.0 N-s/m | 0.75 | Thin grass / light brush |
| 3 | 20-30m | 5 stalks/m² | 0.25-0.35m | 5.0 N-s/m | 0.70 | Medium lawn grass |
| 4 | 30-40m | 10 stalks/m² | 0.30-0.45m | 10.0 N-s/m | 0.65 | Thick overgrown grass |
| 5 | 40-50m | 20 stalks/m² | 0.35-0.50m | 20.0 N-s/m | 0.60 | Dense brush / undergrowth |

#### Dual Resistance Model

This environment combines two resistance mechanisms from existing project code:

**Mechanism 1: Proxy Stalks** (from `grass_physics_config.py`)
- Kinematic cylinder colliders placed on the ground within each zone
- Increasing density per zone creates physical resistance
- Stalks use `PROXY_STALK_CONFIG` settings: kinematic=True, self_collision=False
- Collision group filtering prevents stalk-stalk collision (performance optimization)

```python
# Per-zone stalk generation (extending grass_physics_config.py patterns)
def create_grass_zone(stage, zone_idx, x_start, density, height_range):
    """Create proxy stalks for a 30m x 10m zone."""
    zone_area_m2 = 30.0 * 10.0  # 300 m²
    num_stalks = int(zone_area_m2 * density)

    for i in range(num_stalks):
        x = x_start + np.random.uniform(0, 10.0)
        y = np.random.uniform(0, 30.0)
        h = np.random.uniform(height_range[0], height_range[1])
        r = np.random.uniform(0.005, 0.015)  # 5-15mm radius

        stalk_path = f"/World/GrassEnv/Zone_{zone_idx}/stalk_{i}"
        cylinder = UsdGeom.Cylinder.Define(stage, stalk_path)
        cylinder.GetHeightAttr().Set(h)
        cylinder.GetRadiusAttr().Set(r)

        xform = UsdGeom.XformCommonAPI(cylinder)
        xform.SetTranslate(Gf.Vec3d(x, y, h / 2.0 - 0.02))  # sink 2cm

        # Kinematic rigid body (no physics solve)
        rb = UsdPhysics.RigidBodyAPI.Apply(cylinder.GetPrim())
        rb.CreateKinematicEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
```

**Mechanism 2: Scripted Drag Force**
- Applied via Isaac Lab's `apply_external_force_torque` event system (from `events.py:175-183`)
- Drag force: `F_drag = -c_drag * v_base` (proportional to base velocity, opposing motion)
- Calculated per physics step using robot root state velocity
- Zone-specific drag coefficient determined by robot's current X position

```python
# Drag force callback (runs each control step)
def apply_zone_drag(env, asset_cfg):
    """Apply position-dependent drag force to robot base."""
    root_pos = env.scene["robot"].data.root_pos_w       # (N, 3)
    root_vel = env.scene["robot"].data.root_lin_vel_w    # (N, 3)

    # Determine zone index from x-position
    zone_idx = torch.clamp((root_pos[:, 0] / 10.0).long(), 0, 4)

    # Drag coefficients per zone
    c_drag = torch.tensor([0.5, 2.0, 5.0, 10.0, 20.0], device=root_vel.device)
    drag = c_drag[zone_idx].unsqueeze(1)  # (N, 1)

    # Apply opposing force: F = -c_d * v
    force = -drag * root_vel  # (N, 3)
    torque = torch.zeros_like(force)

    env.scene["robot"].set_external_force_and_torque(force, torque, body_ids=[0])
```

#### Resistance Budget

| Zone | Drag Force at 1 m/s | Stalk Collisions/m | Total Resistance | Spot Max Forward Force |
|------|---------------------|--------------------|-----------------|----------------------|
| 1 | 0.5 N | 0 | ~0.5 N | ~200 N |
| 2 | 2.0 N | ~0.6 | ~4 N | ~200 N |
| 3 | 5.0 N | ~1.5 | ~12 N | ~200 N |
| 4 | 10.0 N | ~3.0 | ~25 N | ~200 N |
| 5 | 20.0 N | ~6.0 | ~50 N | ~200 N |

Even Zone 5 total resistance (~50 N) is well within Spot's ~200 N forward thrust capacity, so the challenge is gait stability through physical collisions, not raw force.

---

### 3.3 Environment 3: Unstructured Boulder Field

**Purpose:** Test traversal over unstructured terrain with irregularly shaped polyhedral obstacles. Uses a mixed distribution of four dice-like polyhedra (D8, D10, D12, D20) to create unstable, unpredictable gravel surfaces. The shape variety prevents the robot from learning a single stepping strategy and tests the rough policy's height scan advantage for obstacle detection and foot placement.

**Terrain geometry:** Flat ground with mixed polyhedral mesh obstacles of increasing size

#### Shape Distribution

Each zone contains an equal 25% mix of four polyhedral shapes:

| Shape | Polyhedron | Faces | Vertices | Edges | Face Type | Rolling Behavior |
|-------|-----------|-------|----------|-------|-----------|-----------------|
| D8 | Octahedron | 8 | 6 | 12 | Equilateral triangles | Sharp peaks, tips easily — most unstable underfoot |
| D10 | Pentagonal trapezohedron | 10 | 12 | 20 | Kite-shaped quads | Asymmetric, unpredictable roll direction |
| D12 | Dodecahedron | 12 | 20 | 30 | Regular pentagons | Moderate stability, tends to settle on flat faces |
| D20 | Icosahedron | 20 | 12 | 30 | Equilateral triangles | Near-spherical, rolls freely — hardest to stand on |

The combination creates terrain where no two footfalls encounter the same geometry, forcing the policy to generalize across contact surfaces.

#### Zone Specifications

| Zone | X Range | Edge Length | Height (approx) | Density | Count (per zone) | Real-World Analog |
|------|---------|-----------|-----------------|---------|-----------------|-------------------|
| 1 | 0-10m | 3-5 cm | 2-5 cm | 15/m² | ~4,500 | Gravel / pebbles |
| 2 | 10-20m | 10-15 cm | 8-15 cm | 8/m² | ~2,400 | River rocks |
| 3 | 20-30m | 25-35 cm | 20-35 cm | 4/m² | ~1,200 | Large rocks |
| 4 | 30-40m | 50-70 cm | 40-70 cm | 2/m² | ~600 | Small boulders |
| 5 | 40-50m | 80-120 cm | 65-120 cm | 1/m² | ~300 | Large boulders |

#### Polyhedron Geometry Definitions

```python
import numpy as np

# ─── Golden ratio (used by D12 and D20) ───
PHI = (1.0 + np.sqrt(5.0)) / 2.0

def _unit_verts(verts):
    """Normalize vertices so the longest axis spans 1.0, then scale by edge_length."""
    v = np.array(verts, dtype=np.float32)
    v -= v.mean(axis=0)  # center at origin
    v /= np.abs(v).max()  # normalize to [-1, 1]
    return v


# ─── D8: Octahedron (8 triangular faces, 6 vertices) ───
def octahedron_mesh(edge_length):
    """Regular octahedron scaled to given edge length."""
    verts = _unit_verts([
        [ 1,  0,  0], [-1,  0,  0],
        [ 0,  1,  0], [ 0, -1,  0],
        [ 0,  0,  1], [ 0,  0, -1],
    ]) * edge_length * 0.707  # circumradius = a/sqrt(2)

    faces = [
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],  # top 4
        [0, 3, 5], [3, 1, 5], [1, 2, 5], [2, 0, 5],  # bottom 4
    ]
    return verts, faces


# ─── D10: Pentagonal Trapezohedron (10 kite faces, 12 vertices) ───
def trapezohedron_mesh(edge_length):
    """Pentagonal trapezohedron (standard D10 shape)."""
    # Top and bottom apex
    top = [0, 0,  1.0]
    bot = [0, 0, -1.0]

    # Two rings of 5 vertices, offset by 36 degrees
    upper_ring, lower_ring = [], []
    for i in range(5):
        angle_u = 2 * np.pi * i / 5
        angle_l = 2 * np.pi * (i + 0.5) / 5
        upper_ring.append([np.cos(angle_u) * 0.85, np.sin(angle_u) * 0.85,  0.30])
        lower_ring.append([np.cos(angle_l) * 0.85, np.sin(angle_l) * 0.85, -0.30])

    verts = _unit_verts([top] + upper_ring + lower_ring + [bot]) * edge_length * 0.6
    # top=0, upper=1-5, lower=6-10, bot=11

    faces = []
    for i in range(5):
        u0 = 1 + i
        u1 = 1 + (i + 1) % 5
        l0 = 6 + i
        l1 = 6 + (i + 1) % 5
        faces.append([0, u0, l0])       # top kite upper
        faces.append([0, l0, u1])       # top kite lower
        faces.append([11, l0, u0])      # bot kite upper
        faces.append([11, u1, l0])      # bot kite lower
    # Simplified to triangulated kites (20 tris from 10 kite faces)
    return verts, faces


# ─── D12: Dodecahedron (12 pentagonal faces, 20 vertices) ───
def dodecahedron_mesh(edge_length):
    """Regular dodecahedron scaled to given edge length."""
    # 20 vertices from cube + rectangle coordinates
    raw = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            for s3 in (-1, 1):
                raw.append([s1, s2, s3])
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            raw.append([0, s1 * PHI, s2 / PHI])
            raw.append([s1 / PHI, 0, s2 * PHI])
            raw.append([s1 * PHI, s2 / PHI, 0])

    verts = _unit_verts(raw) * edge_length * 0.75

    # Pentagonal faces triangulated (3 triangles per pentagon = 36 tris)
    # Use convexHull approximation in PhysX instead of explicit face winding
    # (see collision setup below)
    faces = _convex_hull_faces(verts)
    return verts, faces


# ─── D20: Icosahedron (20 triangular faces, 12 vertices) ───
def icosahedron_mesh(edge_length):
    """Regular icosahedron scaled to given edge length."""
    raw = []
    for s in (-1, 1):
        raw.append([0,  s,  PHI])
        raw.append([0,  s, -PHI])
        raw.append([ PHI, 0,  s])
        raw.append([-PHI, 0,  s])
        raw.append([ s,  PHI, 0])
        raw.append([ s, -PHI, 0])

    verts = _unit_verts(raw) * edge_length * 0.525  # circumradius ≈ 0.951a

    faces = [
        [0, 4, 8], [0, 8, 6], [0, 6, 10], [0, 10, 2], [0, 2, 4],
        [3, 4, 2], [3, 8, 4], [3, 6, 8], [3, 10, 6], [3, 2, 10],
        [1, 5, 9], [1, 9, 7], [1, 7, 11], [1, 11, 5], [1, 5, 9],
        [5, 4, 9], [9, 8, 7], [7, 6, 11], [11, 10, 5], [5, 2, 4],
    ]
    return verts, faces


def _convex_hull_faces(verts):
    """Compute triangulated convex hull faces via scipy."""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(verts)
    return hull.simplices.tolist()


# ─── Shape registry ───
SHAPE_GENERATORS = {
    "D8":  octahedron_mesh,
    "D10": trapezohedron_mesh,
    "D12": dodecahedron_mesh,
    "D20": icosahedron_mesh,
}
```

#### USD Mesh Creation

```python
def create_polyhedron(stage, path, shape_name, edge_length, position, rotation):
    """Create a polyhedral trimesh obstacle at the given position."""
    gen_fn = SHAPE_GENERATORS[shape_name]
    verts, faces = gen_fn(edge_length)

    # Flatten face indices for USD
    face_indices = []
    face_counts = []
    for f in faces:
        face_indices.extend(f)
        face_counts.append(len(f))

    # Create USD mesh
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set(verts.tolist())
    mesh.GetFaceVertexIndicesAttr().Set(face_indices)
    mesh.GetFaceVertexCountsAttr().Set(face_counts)

    # Position and rotation
    xform = UsdGeom.XformCommonAPI(mesh)
    xform.SetTranslate(Gf.Vec3d(*position))
    xform.SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), rotation[0]) *
                     Gf.Rotation(Gf.Vec3d(0, 1, 0), rotation[1]) *
                     Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation[2]))

    # Physics: static rigid body with convex hull collision
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision.CreateApproximationAttr("convexHull")

    # Kinematic = immovable
    rb = UsdPhysics.RigidBodyAPI.Apply(mesh.GetPrim())
    rb.CreateKinematicEnabledAttr(True)
```

#### Zone Generation

```python
SHAPE_NAMES = ["D8", "D10", "D12", "D20"]  # 25% each

def populate_boulder_zone(stage, zone_idx, x_start, edge_range, density, seed):
    """Scatter mixed polyhedra across a 30m x 10m zone."""
    np.random.seed(seed + zone_idx)
    zone_area = 30.0 * 10.0  # m²
    count = int(zone_area * density)

    for i in range(count):
        # Select shape: 25% each of D8, D10, D12, D20
        shape = SHAPE_NAMES[i % 4]

        edge = np.random.uniform(edge_range[0], edge_range[1])
        x = x_start + np.random.uniform(0, 10.0)
        y = np.random.uniform(0, 30.0)
        z = 0.0  # sitting on ground

        # Random rotation for natural appearance (uniform SO(3))
        rot = (np.random.uniform(0, 360),
               np.random.uniform(0, 360),
               np.random.uniform(0, 360))

        path = f"/World/BoulderField/Zone_{zone_idx}/{shape}_{i}"
        create_polyhedron(stage, path, shape, edge, (x, y, z), rot)
```

#### Performance Optimization

- **Vertex counts per shape:** D8 (6 verts), D10 (12 verts), D12 (20 verts), D20 (12 verts) — avg ~12.5 verts/shape
- **Zone 1 (~4,500 shapes):** Use collision group batching. `convexHull` approximation keeps solver cost low regardless of shape complexity.
- **Zones 3-5 (~2,100 total):** Larger but fewer objects, standard collision handling.
- **Total mesh count:** ~9,000 across all zones. For 512 parallel envs, each env gets its own instance. GPU PhysX handles this via instanced collision.
- **Memory estimate:** ~9,000 meshes x ~12.5 avg vertices x 12 bytes = ~1.4 MB geometry per env. At 512 envs: ~700 MB (fits in H100 80GB).

#### Expected Challenges

- **Zone 1:** Mixed gravel provides rough but traversable surface. D20 pebbles roll underfoot causing micro-instability. Both policies should manage.
- **Zone 2:** River rocks at 10-15cm start interfering with foot placement. D8 octahedra create sharp peaks that destabilize contact. Rough policy's height scan should help.
- **Zone 3:** Rocks at 25-35cm are near Spot's 12cm foot clearance target. D10 trapezohedra create unpredictable asymmetric contacts. Rough policy's higher clearance weight (2.0 vs 0.5) is critical.
- **Zone 4:** Boulders at 50-70cm exceed Spot's leg reach. D12 dodecahedra settle on flat pentagonal faces but create cliff-like edges between them. Must step over or navigate between.
- **Zone 5:** 80-120cm boulders are at or above Spot's hip height (~50cm). The mix of D20 (near-spherical) and D8 (sharp-peaked) creates maximally unstructured terrain. Expected to be impassable for both policies — this zone tests graceful failure.

---

### 3.4 Environment 4: Staircase

**Purpose:** Test stair climbing capability with increasing step height. Directly tests the rough policy's enhanced foot clearance reward and terrain curriculum training.

**Terrain geometry:** Ascending staircases with 2m flat platforms between zones

#### Zone Specifications

| Zone | X Range | Step Height | Step Depth (Tread) | Steps per Zone | Total Rise | Stair Width | Real-World Analog |
|------|---------|------------|-------------------|---------------|-----------|------------|-------------------|
| 1 | 0-10m | 3 cm | 30 cm | 33 | 0.99 m | 30 m | Shallow access ramp |
| 2 | 10-20m | 8 cm | 30 cm | 33 | 2.64 m | 30 m | Low residential stairs |
| 3 | 20-30m | 13 cm | 30 cm | 33 | 4.29 m | 30 m | Standard residential (7" rise) |
| 4 | 30-40m | 18 cm | 30 cm | 33 | 5.94 m | 30 m | Steep commercial stairs |
| 5 | 40-50m | 23 cm | 30 cm | 33 | 7.59 m | 30 m | Maximum challenge |

**Note:** Zone 5 step height (23cm) matches the upper bound of training terrain (`ROUGH_TERRAINS_CFG` stairs: 0.05-0.23m).

#### Layout Detail

Each zone is 10m long with stairs occupying 9.9m (33 steps x 0.30m tread) and a 2m flat platform connecting to the next zone. The platform allows the robot to stabilize before entering the next difficulty level.

```
Zone 1          Platform  Zone 2          Platform  Zone 3 ...
|_____|_____|... |======| |_____|_____|... |======| |_____|...
 3cm steps        2m flat   8cm steps       2m flat  13cm steps
```

**Cumulative elevation** at the end of each zone (above spawn):

| After Zone | Cumulative Height |
|------------|------------------|
| 1 | 0.99 m |
| 2 | 3.63 m |
| 3 | 7.92 m |
| 4 | 13.86 m |
| 5 | 21.45 m |

#### Isaac Sim Implementation

```python
# Stair generation (reusing ROUGH_TERRAINS_CFG patterns)
def create_stair_zone(stage, zone_idx, x_start, step_height, step_depth,
                      num_steps, width, base_elevation):
    """Create ascending stairs as stacked box prims."""
    zone_path = f"/World/Staircase/Zone_{zone_idx}"
    current_z = base_elevation

    for s in range(num_steps):
        step_path = f"{zone_path}/step_{s}"
        cube = UsdGeom.Cube.Define(stage, step_path)
        cube.GetSizeAttr().Set(1.0)

        # Each step is a box: depth x width x (cumulative height)
        step_x = x_start + s * step_depth + step_depth / 2.0
        current_z += step_height
        step_z = current_z / 2.0  # center of cumulative box

        UsdGeom.XformCommonAPI(cube).SetScale(
            Gf.Vec3f(step_depth, width, current_z)
        )
        UsdGeom.XformCommonAPI(cube).SetTranslate(
            Gf.Vec3d(step_x, width / 2.0, step_z)
        )

        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    # Create flat platform after stairs
    platform_path = f"{zone_path}/platform"
    platform = UsdGeom.Cube.Define(stage, platform_path)
    platform.GetSizeAttr().Set(1.0)
    UsdGeom.XformCommonAPI(platform).SetScale(
        Gf.Vec3f(2.0, width, current_z)
    )
    UsdGeom.XformCommonAPI(platform).SetTranslate(
        Gf.Vec3d(x_start + num_steps * step_depth + 1.0, width / 2.0, current_z / 2.0)
    )
    UsdPhysics.CollisionAPI.Apply(platform.GetPrim())

    return current_z  # Return final elevation for next zone
```

#### Expected Challenges

- **Zone 1 (3cm):** Effectively a textured ramp. Both policies should handle easily.
- **Zone 2 (8cm):** Within training distribution. Rough policy should excel with foot clearance reward.
- **Zone 3 (13cm):** Standard residential stair height. This is the key battleground — rough policy was specifically trained with 0.12m foot clearance target.
- **Zone 4 (18cm):** Exceeds the 0.12m clearance target. Requires aggressive leg lifting. Rough policy's 2.0x foot clearance weight should help.
- **Zone 5 (23cm):** At the extreme of training distribution (0.05-0.23m from `ROUGH_TERRAINS_CFG`). Expected high failure rate for both policies but rough policy should last longer.

---

## 4. Waypoint Navigation System

### 4.1 Waypoint Configuration

The robot follows a series of waypoints placed at zone boundaries along the centerline:

| Waypoint | Position (x, y) | Purpose |
|----------|-----------------|---------|
| WP0 (spawn) | (0.0, 15.0) | Starting position |
| WP1 | (10.0, 15.0) | Zone 1 → Zone 2 boundary |
| WP2 | (20.0, 15.0) | Zone 2 → Zone 3 boundary |
| WP3 | (30.0, 15.0) | Zone 3 → Zone 4 boundary |
| WP4 | (40.0, 15.0) | Zone 4 → Zone 5 boundary |
| WP5 (goal) | (50.0, 15.0) | End of course |

### 4.2 Heading Controller

The waypoint follower generates velocity commands that override the standard `UniformVelocityCommandCfg` random sampling:

```python
class WaypointFollower:
    """Generates velocity commands to follow waypoints."""

    def __init__(self, waypoints, num_envs, device):
        self.waypoints = torch.tensor(waypoints, device=device)  # (6, 2)
        self.current_wp = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.Kp_yaw = 2.0          # Proportional gain for heading correction
        self.vx_target = 1.0       # Target forward velocity (m/s)
        self.wp_threshold = 0.5    # Waypoint reached threshold (m)
        self.lateral_band = 2.0    # Max lateral deviation before correction

    def compute_commands(self, root_pos, root_yaw):
        """
        Compute velocity commands for all environments.

        Args:
            root_pos: (N, 3) robot root position in world frame
            root_yaw: (N,) robot yaw angle in radians

        Returns:
            commands: (N, 3) [vx, vy, omega_z] velocity commands
        """
        N = root_pos.shape[0]

        # Get current target waypoint for each env
        wp_idx = self.current_wp.clamp(0, len(self.waypoints) - 1)
        target = self.waypoints[wp_idx]  # (N, 2)

        # Compute heading to waypoint
        dx = target[:, 0] - root_pos[:, 0]
        dy = target[:, 1] - root_pos[:, 1]
        desired_yaw = torch.atan2(dy, dx)

        # Heading error (wrapped to [-pi, pi])
        yaw_err = desired_yaw - root_yaw
        yaw_err = torch.atan2(torch.sin(yaw_err), torch.cos(yaw_err))

        # Velocity commands
        vx = torch.full((N,), self.vx_target, device=root_pos.device)
        vy = torch.zeros(N, device=root_pos.device)
        omega_z = self.Kp_yaw * yaw_err

        # Clamp angular velocity to policy training range
        omega_z = omega_z.clamp(-2.0, 2.0)

        # Advance waypoint when robot passes waypoint x-position
        at_wp = root_pos[:, 0] >= target[:, 0] - self.wp_threshold
        self.current_wp = torch.where(
            at_wp & (self.current_wp < len(self.waypoints) - 1),
            self.current_wp + 1,
            self.current_wp
        )

        return torch.stack([vx, vy, omega_z], dim=1)
```

### 4.3 Command Integration

The waypoint follower replaces the standard velocity command sampling during evaluation. Commands are injected into the observation buffer at indices 9:12 (matching `velocity_commands` observation term in `rough_env_cfg.py:77-79`):

```python
# In evaluation loop, override commands each control step:
commands = waypoint_follower.compute_commands(root_pos, root_yaw)
env.command_manager.set_command("base_velocity", commands)
```

---

## 5. Metrics and Data Collection

### 5.1 Per-Episode Metrics

| Metric | Type | Description | Computation |
|--------|------|-------------|-------------|
| `episode_id` | int | Unique episode identifier | Sequential |
| `policy` | str | "flat" or "rough" | Config parameter |
| `environment` | str | "friction", "grass", "boulder", "stairs" | Config parameter |
| `completion` | bool | Robot reached x >= 49m within timeout | `max(root_pos_x) >= 49.0` |
| `progress` | float | Maximum x-distance achieved (0-50m) | `max(root_pos_x)` over episode |
| `zone_reached` | int | Highest zone fully traversed (0-5) | `floor(progress / 10)` |
| `time_to_complete` | float | Seconds to reach x=50m (NaN if timeout) | First timestep where x >= 49m |
| `stability_score` | float | Composite stability metric (lower = better) | See Section 5.2 |
| `mean_roll` | float | Mean absolute roll angle (rad) | `mean(abs(roll))` |
| `mean_pitch` | float | Mean absolute pitch angle (rad) | `mean(abs(pitch))` |
| `height_variance` | float | Variance of base height (m^2) | `var(base_height)` |
| `mean_ang_vel` | float | Mean angular velocity magnitude (rad/s) | `mean(norm(ang_vel))` |
| `fall_detected` | bool | Base height dropped below 0.15m | `min(base_height) < 0.15` |
| `fall_location` | float | X-position of fall (NaN if no fall) | X at first `base_height < 0.15` |
| `fall_zone` | int | Zone where fall occurred (0 if no fall) | `floor(fall_location / 10)` |
| `mean_velocity` | float | Average forward velocity achieved (m/s) | `mean(root_vel_x)` |
| `total_energy` | float | Sum of absolute joint torques over episode | `sum(abs(joint_torques))` |
| `episode_length` | float | Episode duration in seconds | Timestep count x dt |

### 5.2 Stability Score Computation

```python
def compute_stability_score(roll_history, pitch_history, height_history, ang_vel_history):
    """
    Composite stability metric. Lower = more stable.

    Components (weighted):
    - Mean absolute roll:   weight 1.0 (rad)
    - Mean absolute pitch:  weight 1.0 (rad)
    - Height variance:      weight 10.0 (m^2 -> amplified for sensitivity)
    - Mean angular velocity: weight 0.5 (rad/s)
    """
    score = (
        1.0 * torch.mean(torch.abs(roll_history)) +
        1.0 * torch.mean(torch.abs(pitch_history)) +
        10.0 * torch.var(height_history) +
        0.5 * torch.mean(torch.norm(ang_vel_history, dim=-1))
    )
    return score.item()
```

### 5.3 Metric Extraction from Isaac Lab

All metrics are derived from the robot's root state tensor, available at each control step:

```python
# Root state access (from rough_env_cfg.py observation terms)
root_pos = env.scene["robot"].data.root_pos_w          # (N, 3) world position
root_quat = env.scene["robot"].data.root_quat_w        # (N, 4) quaternion
root_lin_vel = env.scene["robot"].data.root_lin_vel_w   # (N, 3) linear velocity
root_ang_vel = env.scene["robot"].data.root_ang_vel_w   # (N, 3) angular velocity
joint_torques = env.scene["robot"].data.applied_torque  # (N, 12) joint torques

# Euler angles from quaternion
roll, pitch, yaw = quat_to_euler(root_quat)            # (N,) each
base_height = root_pos[:, 2]                            # (N,)
```

### 5.4 Output Format

**Per-episode JSON log** (one file per policy/environment combination):

```
results/
├── friction_flat_episodes.jsonl
├── friction_rough_episodes.jsonl
├── grass_flat_episodes.jsonl
├── grass_rough_episodes.jsonl
├── boulder_flat_episodes.jsonl
├── boulder_rough_episodes.jsonl
├── stairs_flat_episodes.jsonl
├── stairs_rough_episodes.jsonl
└── summary.csv
```

Each `.jsonl` file contains one JSON object per line (one per episode):

```json
{
  "episode_id": 0,
  "policy": "rough",
  "environment": "friction",
  "completion": false,
  "progress": 23.47,
  "zone_reached": 2,
  "time_to_complete": null,
  "stability_score": 0.342,
  "mean_roll": 0.05,
  "mean_pitch": 0.08,
  "height_variance": 0.002,
  "mean_ang_vel": 0.31,
  "fall_detected": true,
  "fall_location": 23.47,
  "fall_zone": 3,
  "mean_velocity": 0.67,
  "total_energy": 145230.5,
  "episode_length": 35.2
}
```

---

## 6. Headless H100 Execution Plan

### 6.1 Batch Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Parallel envs per batch | 512 | Fits in H100 80GB VRAM with margin |
| Episodes per policy/env | 1,000 | Statistically significant sample size |
| Batches per policy/env | ceil(1000/512) = 2 | 2 batches: 512 + 488 episodes |
| Total batches | 2 policies x 4 envs x 2 batches = **16** | |
| Episode timeout | 120 seconds sim time | ~2.4x time needed at 1.0 m/s for 50m |
| Control steps per episode | 120s / 0.02s = 6,000 max | At 50 Hz control rate |
| Physics steps per episode | 120s / 0.002s = 60,000 max | At 500 Hz |

### 6.2 GPU Memory Budget

| Component | Memory (per 512 envs) |
|-----------|----------------------|
| Robot state (12 joints + root) | ~50 MB |
| Policy network (512-256-128 MLP) | ~5 MB |
| Height scanner (160 rays x 512 envs) | ~10 MB |
| Environment geometry (varies) | 200 MB - 2 GB |
| PhysX solver buffers | ~2 GB |
| PyTorch inference cache | ~1 GB |
| **Total estimate** | **~4-6 GB** |
| **H100 available** | **80 GB** |

Memory is not a bottleneck. Could potentially scale to 2048+ parallel envs if faster throughput is needed.

### 6.3 Execution Commands

```bash
# SSH into H100 cluster
ssh user@172.24.254.24

# Navigate to Isaac Lab
cd /path/to/IsaacLab

# Run evaluation for each policy/environment combination
# Each command runs 1000 episodes in 2 batches of 512

# Environment 1: Friction
./isaaclab.sh -p run_capstone_eval.py --headless \
    --num_envs 512 --policy flat --env friction \
    --num_episodes 1000 --output_dir results/

./isaaclab.sh -p run_capstone_eval.py --headless \
    --num_envs 512 --policy rough --env friction \
    --num_episodes 1000 --output_dir results/

# Environment 2: Grass
./isaaclab.sh -p run_capstone_eval.py --headless \
    --num_envs 512 --policy flat --env grass \
    --num_episodes 1000 --output_dir results/

./isaaclab.sh -p run_capstone_eval.py --headless \
    --num_envs 512 --policy rough --env grass \
    --num_episodes 1000 --output_dir results/

# Environment 3: Boulder
./isaaclab.sh -p run_capstone_eval.py --headless \
    --num_envs 512 --policy flat --env boulder \
    --num_episodes 1000 --output_dir results/

./isaaclab.sh -p run_capstone_eval.py --headless \
    --num_envs 512 --policy rough --env boulder \
    --num_episodes 1000 --output_dir results/

# Environment 4: Stairs
./isaaclab.sh -p run_capstone_eval.py --headless \
    --num_envs 512 --policy flat --env stairs \
    --num_episodes 1000 --output_dir results/

./isaaclab.sh -p run_capstone_eval.py --headless \
    --num_envs 512 --policy rough --env stairs \
    --num_episodes 1000 --output_dir results/
```

#### Automated Batch Script

```bash
#!/bin/bash
# run_all_evaluations.sh — Execute full 8000-episode evaluation suite

POLICIES=("flat" "rough")
ENVS=("friction" "grass" "boulder" "stairs")
OUTPUT_DIR="results/capstone_eval_$(date +%Y%m%d_%H%M%S)"
NUM_ENVS=512
NUM_EPISODES=1000

mkdir -p $OUTPUT_DIR

for policy in "${POLICIES[@]}"; do
    for env in "${ENVS[@]}"; do
        echo "=== Running: policy=$policy env=$env ==="
        ./isaaclab.sh -p run_capstone_eval.py --headless \
            --num_envs $NUM_ENVS \
            --policy $policy \
            --env $env \
            --num_episodes $NUM_EPISODES \
            --output_dir $OUTPUT_DIR \
            2>&1 | tee "$OUTPUT_DIR/${env}_${policy}.log"
        echo "=== Completed: policy=$policy env=$env ==="
    done
done

echo "=== All evaluations complete. Results in $OUTPUT_DIR ==="
```

### 6.4 Timing Estimates

| Phase | Wall-Clock Time |
|-------|----------------|
| **Headless Statistical Runs** | |
| Environment build (per env) | ~30-60 seconds |
| Policy loading | ~5 seconds |
| 512-env batch (120s sim) | ~3-5 minutes at 30-50x real-time |
| Per policy/env (2 batches) | ~8-12 minutes |
| Headless subtotal (8 combinations) | ~60-90 minutes |
| Post-processing & summary | ~5 minutes |
| **Rendered Visualization Runs** | |
| Per rendered episode (5 envs, 120s sim) | ~2-3 minutes at ~1x real-time |
| Per policy/env (10 episodes) | ~25-30 minutes |
| Rendered subtotal (8 combinations) | ~3.5-4 hours |
| **Grand total (headless + rendered)** | **~5-5.5 hours** |

### 6.5 Rendered Visualization Runs

In addition to the 8,000 headless statistical episodes, run **10 rendered iterations with 5 parallel environments** per policy/environment combination to capture visual footage of robot behavior.

#### Visualization Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Parallel envs | 5 | Small enough for clear visual, all visible at once |
| Episodes (iterations) | 10 | Enough to capture representative successes and failures |
| Rendering | **Enabled** (not headless) | Full Isaac Sim viewport rendering |
| Resolution | 1920x1080 | Standard HD for documentation/presentation |
| Capture format | MP4 video + PNG key frames | Video for review, stills for reports |
| Camera | Follow-cam + fixed overhead | Two perspectives per run |
| Total rendered runs | 2 policies x 4 envs x 10 episodes = **80 rendered episodes** |

#### Camera Setup

Two simultaneous camera views per run:

1. **Follow Camera:** Tracks the center robot (env index 0), positioned at 45-degree rear-quarter angle, 3m distance. Shows gait detail, foot placement, and obstacle interaction up close.
2. **Overhead Camera:** Fixed top-down view of the full 30m x 50m arena. Shows all 5 robots progressing through zones simultaneously. Useful for comparing traversal patterns.

```python
# Camera configuration for rendered runs
from isaaclab.envs import ViewerCfg

# Follow cam (rear-quarter, 3m back and 2m up)
viewer_follow = ViewerCfg(
    eye=(−3.0, 2.0, 2.0),    # relative offset behind robot
    origin_type="env",         # track robot env
    env_index=0,
    asset_name="robot",
)

# Overhead cam (fixed, looking down at arena center)
overhead_cam_pos = (25.0, 15.0, 40.0)   # centered at x=25m, y=15m, 40m up
overhead_cam_target = (25.0, 15.0, 0.0)  # looking straight down
```

#### Video Capture

```python
# Isaac Sim video recording (runs within rendered mode)
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.kit.capture")

import omni.kit.capture
capture = omni.kit.capture.Capture()
capture.start(
    output_path=f"results/video/{env}_{policy}_ep{episode:02d}.mp4",
    fps=30,
    resolution=(1920, 1080),
)
# ... run episode ...
capture.stop()
```

#### Key Frame Extraction

Automatically capture PNG snapshots at zone boundaries (x = 0, 10, 20, 30, 40, 50m) and at fall events:

```python
def capture_keyframes(root_pos, episode_id, env_name, policy_name, step):
    """Save PNG at zone transitions and falls."""
    x = root_pos[0, 0].item()  # env 0 x-position
    zone = int(x / 10.0)

    # Capture at zone entry (within 0.5m of boundary)
    zone_boundary = zone * 10.0
    if abs(x - zone_boundary) < 0.5 and zone not in captured_zones:
        save_viewport_png(f"results/frames/{env_name}_{policy_name}_ep{episode_id}_zone{zone}.png")
        captured_zones.add(zone)

    # Capture at fall event
    if root_pos[0, 2].item() < 0.15:
        save_viewport_png(f"results/frames/{env_name}_{policy_name}_ep{episode_id}_fall_x{x:.1f}.png")
```

#### Execution Commands

```bash
# Rendered visualization runs (NOT headless — requires display or virtual framebuffer)

POLICIES=("flat" "rough")
ENVS=("friction" "grass" "boulder" "stairs")
OUTPUT_DIR="results/capstone_visual_$(date +%Y%m%d_%H%M%S)"

mkdir -p $OUTPUT_DIR/video $OUTPUT_DIR/frames

for policy in "${POLICIES[@]}"; do
    for env in "${ENVS[@]}"; do
        echo "=== Rendered: policy=$policy env=$env ==="
        ./isaaclab.sh -p run_capstone_eval.py \
            --num_envs 5 \
            --policy $policy \
            --env $env \
            --num_episodes 10 \
            --output_dir $OUTPUT_DIR \
            --render \
            --capture_video \
            --capture_keyframes
        echo "=== Completed visual: policy=$policy env=$env ==="
    done
done
```

#### Timing Estimate

| Phase | Wall-Clock Time |
|-------|----------------|
| Per rendered episode (120s sim, ~1x real-time with rendering) | ~2-3 minutes |
| Per policy/env (10 episodes) | ~25-30 minutes |
| **Total (8 combinations)** | **~3.5-4 hours** |

#### Output Structure

```
results/
├── capstone_eval_*/          # Headless statistical results (8,000 episodes)
│   ├── *.jsonl
│   └── summary.csv
└── capstone_visual_*/        # Rendered visualization results (80 episodes)
    ├── video/
    │   ├── friction_flat_ep00.mp4
    │   ├── friction_flat_ep01.mp4
    │   ├── ...
    │   └── stairs_rough_ep09.mp4
    └── frames/
        ├── friction_flat_ep00_zone0.png
        ├── friction_flat_ep00_zone1.png
        ├── friction_flat_ep00_fall_x14.3.png
        ├── ...
        └── stairs_rough_ep09_zone4.png
```

### 6.6 Checkpointing and Fault Tolerance

- Results saved after each batch (512 episodes) to prevent data loss
- Each JSONL file is append-mode — can resume if a batch fails
- GPU utilization logged via `nvidia-smi` every 10 seconds to CSV
- If an env crashes, the batch script continues to the next combination
- Rendered runs are independent — a crash in one does not affect others

---

## 7. Evaluation Script Architecture

### 7.1 File Structure

```
Capstone/Experiments/capstone_eval/
├── run_capstone_eval.py          # Main entry point (Isaac Lab launcher)
├── envs/
│   ├── __init__.py
│   ├── friction_env.py           # Env 1: Friction surface builder
│   ├── grass_env.py              # Env 2: Grass/fluid resistance builder
│   ├── boulder_env.py            # Env 3: Boulder field builder
│   └── stairs_env.py             # Env 4: Staircase builder
├── navigation/
│   ├── __init__.py
│   └── waypoint_follower.py      # Waypoint-based velocity command generator
├── metrics/
│   ├── __init__.py
│   ├── collector.py              # Per-episode metric collection
│   └── reporter.py               # Summary statistics & CSV output
└── configs/
    ├── eval_cfg.py               # Evaluation-specific env config
    └── zone_params.py            # Zone specifications for all 4 envs
```

### 7.2 Main Evaluation Loop

```python
# run_capstone_eval.py (pseudocode)
def main():
    # 1. Parse args (policy, env, num_episodes, num_envs)
    # 2. Initialize Isaac Lab AppLauncher (headless)
    # 3. Build environment from zone_params
    # 4. Load policy checkpoint
    # 5. Create waypoint follower
    # 6. Create metrics collector

    episodes_completed = 0
    batch_idx = 0

    while episodes_completed < args.num_episodes:
        batch_size = min(args.num_envs, args.num_episodes - episodes_completed)

        # Reset all envs to spawn position
        env.reset()
        waypoint_follower.reset()

        # Run episode
        for step in range(MAX_STEPS):
            # Get velocity commands from waypoint follower
            commands = waypoint_follower.compute_commands(root_pos, root_yaw)

            # Override env commands
            env.command_manager.set_command("base_velocity", commands)

            # Get observations
            obs = env.get_observations()

            # Policy inference (no gradient)
            with torch.no_grad():
                actions = policy(obs)

            # Step environment
            env.step(actions)

            # Collect metrics
            metrics_collector.step(root_pos, root_quat, root_vel, joint_torques)

            # Check termination (all envs either completed, fallen, or timed out)
            if metrics_collector.all_done():
                break

        # Save batch results
        metrics_collector.save_batch(batch_idx)
        episodes_completed += batch_size
        batch_idx += 1

    # Generate summary report
    reporter.generate_summary(args.output_dir)
```

---

## 8. Results and Analysis

### 8.1 Summary Statistics Tables

After all 8,000 episodes complete, generate the following comparison tables:

#### Table A: Completion Rate (%)

| Environment | Flat Policy | Rough Policy | Delta |
|-------------|------------|-------------|-------|
| Friction | — | — | — |
| Grass | — | — | — |
| Boulder | — | — | — |
| Stairs | — | — | — |

#### Table B: Mean Progress (meters, out of 50m)

| Environment | Flat Policy (mean +/- std) | Rough Policy (mean +/- std) | p-value |
|-------------|---------------------------|----------------------------|---------|
| Friction | — | — | — |
| Grass | — | — | — |
| Boulder | — | — | — |
| Stairs | — | — | — |

#### Table C: Zone Completion Distribution (% of 1000 runs reaching each zone)

| Environment | Policy | Zone 1 | Zone 2 | Zone 3 | Zone 4 | Zone 5 |
|-------------|--------|--------|--------|--------|--------|--------|
| Friction | Flat | — | — | — | — | — |
| Friction | Rough | — | — | — | — | — |
| Grass | Flat | — | — | — | — | — |
| Grass | Rough | — | — | — | — | — |
| Boulder | Flat | — | — | — | — | — |
| Boulder | Rough | — | — | — | — | — |
| Stairs | Flat | — | — | — | — | — |
| Stairs | Rough | — | — | — | — | — |

#### Table D: Stability Score (lower = better)

| Environment | Flat Policy (mean +/- std) | Rough Policy (mean +/- std) |
|-------------|---------------------------|----------------------------|
| Friction | — | — |
| Grass | — | — |
| Boulder | — | — |
| Stairs | — | — |

#### Table E: Fall Rate and Location

| Environment | Policy | Fall Rate (%) | Mean Fall Location (m) | Most Common Fall Zone |
|-------------|--------|--------------|----------------------|---------------------|
| Friction | Flat | — | — | — |
| Friction | Rough | — | — | — |
| Grass | Flat | — | — | — |
| Grass | Rough | — | — | — |
| Boulder | Flat | — | — | — |
| Boulder | Rough | — | — | — |
| Stairs | Flat | — | — | — |
| Stairs | Rough | — | — | — |

### 8.2 Statistical Analysis

- **Completion rate comparison:** Two-proportion z-test per environment (alpha = 0.05)
- **Progress comparison:** Welch's t-test for mean progress per environment
- **Effect size:** Cohen's d for progress differences
- **Zone survival curves:** Kaplan-Meier-style analysis of zone-by-zone attrition

### 8.3 Visualization Recommendations

1. **Bar chart:** Completion rate by environment (grouped by policy)
2. **Box plot:** Progress distribution by environment and policy
3. **Heatmap:** Zone completion matrix (environment x zone x policy)
4. **Histogram:** Fall location distribution per environment (overlaid flat vs rough)
5. **Line plot:** Stability score vs zone for each environment
6. **Survival curve:** Fraction of runs still active at each x-position

---

## 9. Implementation Dependencies

### 9.1 Existing Code to Reuse

| File | What to Reuse | In Which Component |
|------|-------------- |-------------------|
| `rough_env_cfg.py` | Env config structure, observation groups, height scanner setup | `eval_cfg.py` |
| `rsl_rl_ppo_cfg.py` | Policy network architecture, checkpoint loading | `run_capstone_eval.py` |
| `grass_physics_config.py` | Proxy stalk creation, material setup, density management | `grass_env.py` |
| `rewards.py` | Stability-related reward terms (reuse as metric functions) | `collector.py` |
| `events.py` | External force application pattern | `grass_env.py` (drag forces) |
| `spot_rough_48h_cfg.py` | AppLauncher setup, headless execution, TF32 config | `run_capstone_eval.py` |
| `spot_obstacle_course.py` | `create_physics_material()`, `create_steps()`, USD prim patterns | All env builders |
| `world_factory.py` | `WorldConfig` for physics params | `eval_cfg.py` |
| `data_collector.py` | JSON logging pattern | `collector.py` |

### 9.2 Required Packages

All already present in `requirements.txt`:
- `isaacSim==5.1.0.0` (simulation engine)
- `torch==2.7.0` (policy inference)
- `numpy==1.26.0` (numerical computation)
- `scipy==1.15.3` (statistical tests)
- `matplotlib==3.10.3` (visualization)

---

## 10. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Zone 1 friction too low for any traversal | High | Low | Document as expected result; provides baseline failure data |
| Boulder Zone 5 impassable for both policies | High | Low | Expected — tests graceful failure behavior |
| Proxy stalks cause PhysX instability at high density | Medium | Medium | Cap Zone 5 at 20 stalks/m²; use kinematic bodies |
| 9,000 tetrahedra per env consume too much GPU memory | Medium | High | Reduce Zone 1 density (use heightfield instead of individual meshes for gravel) |
| Waypoint follower commands outside policy training range | Low | High | Clamp commands to training ranges: vx[-2,3], vy[-1.5,1.5], wz[-2,2] |
| GPU PhysX vs CPU PhysX mismatch | Low | Critical | Always use GPU PhysX pipeline (lesson learned from `ROUGH_POLICY_DEBUG_HANDOFF.md`) |
| Height scan fill value bug | Low | Critical | Use `SCAN_FILL_VAL = 0.0` (not 1.0) for flat terrain observations |

---

## 11. Success Criteria

The evaluation is considered successful if:

1. All 8,000 episodes complete without simulation crashes
2. Results show statistically significant differences (p < 0.05) between policies on at least 2 of 4 environments
3. The rough terrain policy demonstrates higher completion rate on stairs and boulders (environments where height scan provides advantage)
4. Data is sufficient to generate all tables and visualizations in Section 8
5. Total wall-clock execution time is under 2 hours on H100

---

## Appendix A: Physics Reference Values

### Friction Coefficients (Real-World Reference)

| Surface Pair | mu_static | mu_dynamic | Source |
|-------------|-----------|-----------|--------|
| Steel on steel (oiled) | 0.03-0.06 | 0.02-0.04 | Engineering Toolbox |
| Rubber on wet ice | 0.10-0.20 | 0.05-0.10 | NHTSA |
| Rubber on wet concrete | 0.30-0.45 | 0.25-0.35 | ASCE |
| Rubber on dry concrete | 0.60-0.80 | 0.50-0.70 | ASTM C1028 |
| Rubber on coarse sandpaper | 0.80-1.00 | 0.70-0.90 | Tribology handbook |

### Spot Physical Parameters

| Parameter | Value |
|-----------|-------|
| Standing height | ~0.50 m |
| Body length | ~1.10 m |
| Body width | ~0.50 m |
| Mass (nominal) | ~32 kg |
| Mass (with randomization) | 27-37 kg |
| Max joint torque (hip) | 45 N-m |
| Max joint torque (knee) | 30.6-113.2 N-m (angle-dependent) |
| Foot clearance target | 0.12 m |
| Default stance height | ~0.45 m |
| Fall threshold | base_height < 0.15 m |

### Stair Building Code Reference (IBC 2021)

| Stair Type | Min Tread Depth | Max Riser Height |
|-----------|----------------|-----------------|
| Residential | 254 mm (10") | 196 mm (7-3/4") |
| Commercial | 279 mm (11") | 178 mm (7") |
| ADA accessible | 279 mm (11") | 178 mm (7") |

Our Zone 3 (13cm = 5.1") and Zone 4 (18cm = 7.1") bracket the standard residential range.

---

## Appendix B: Evaluation Script CLI Reference

```
usage: run_capstone_eval.py [-h] --policy {flat,rough} --env {friction,grass,boulder,stairs}
                            [--num_envs NUM_ENVS] [--num_episodes NUM_EPISODES]
                            [--output_dir OUTPUT_DIR] [--seed SEED]
                            [--timeout TIMEOUT] [--headless]

Capstone Policy Comparative Evaluation

required arguments:
  --policy {flat,rough}       Policy to evaluate
  --env {friction,grass,boulder,stairs}  Environment to test

optional arguments:
  --num_envs NUM_ENVS         Parallel environments (default: 512)
  --num_episodes NUM_EPISODES Total episodes to run (default: 1000)
  --output_dir OUTPUT_DIR     Results output directory (default: results/)
  --seed SEED                 Random seed (default: 42)
  --timeout TIMEOUT           Episode timeout in seconds (default: 120)
  --headless                  Run without rendering
```
