"""Grid-based A* planner for Cole-arena obstacle fields.

Swaps reactive APF for a global plan through known obstacles. APF still runs as a
local deviator on top of the A* sub-waypoint stream for dynamic/moveable cubes.
"""

import heapq
import math
import numpy as np


def _inflate_grid(obstacles, bounds, grid_res, inflate):
    lo, hi = bounds
    n = int(math.ceil((hi - lo) / grid_res))
    occ = np.zeros((n, n), dtype=bool)
    if not obstacles:
        return occ, lo, grid_res, n

    inflate2 = inflate * inflate
    # Each obstacle = (cx, cy, size) where size is full extent (diameter/side).
    for (cx, cy, size) in obstacles:
        half = size * 0.5
        pad = half + inflate
        imin = max(0, int(math.floor((cx - pad - lo) / grid_res)))
        imax = min(n - 1, int(math.ceil((cx + pad - lo) / grid_res)))
        jmin = max(0, int(math.floor((cy - pad - lo) / grid_res)))
        jmax = min(n - 1, int(math.ceil((cy + pad - lo) / grid_res)))
        for i in range(imin, imax + 1):
            x = lo + (i + 0.5) * grid_res
            for j in range(jmin, jmax + 1):
                y = lo + (j + 0.5) * grid_res
                # Closest point on the axis-aligned square to cell center
                dx = max(abs(x - cx) - half, 0.0)
                dy = max(abs(y - cy) - half, 0.0)
                if dx * dx + dy * dy <= inflate2:
                    occ[i, j] = True
    return occ, lo, grid_res, n


def _world_to_cell(pt, lo, res, n):
    i = int((pt[0] - lo) / res)
    j = int((pt[1] - lo) / res)
    i = max(0, min(n - 1, i))
    j = max(0, min(n - 1, j))
    return i, j


def _cell_to_world(ij, lo, res):
    return (lo + (ij[0] + 0.5) * res, lo + (ij[1] + 0.5) * res)


def _nearest_free(occ, ij, n):
    i0, j0 = ij
    if not occ[i0, j0]:
        return ij
    # BFS outward up to 8 cells (4m at res=0.5)
    for r in range(1, 9):
        for di in range(-r, r + 1):
            for dj in (-r, r):
                i, j = i0 + di, j0 + dj
                if 0 <= i < n and 0 <= j < n and not occ[i, j]:
                    return (i, j)
        for di in (-r, r):
            for dj in range(-r + 1, r):
                i, j = i0 + di, j0 + dj
                if 0 <= i < n and 0 <= j < n and not occ[i, j]:
                    return (i, j)
    return ij


def _astar(occ, start_ij, goal_ij, n):
    if start_ij == goal_ij:
        return [start_ij]
    gscore = {start_ij: 0.0}
    came = {}
    heap = [(0.0, start_ij)]
    NEIGH = [(-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
             (0, -1, 1.0),                          (0, 1, 1.0),
             (1, -1, math.sqrt(2)),  (1, 0, 1.0),  (1, 1, math.sqrt(2))]
    while heap:
        f, cur = heapq.heappop(heap)
        if cur == goal_ij:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path
        g_cur = gscore[cur]
        ci, cj = cur
        for di, dj, cost in NEIGH:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < n and 0 <= nj < n):
                continue
            if occ[ni, nj]:
                continue
            # Prevent diagonal corner-cutting through obstacle corners
            if di != 0 and dj != 0:
                if occ[ci + di, cj] or occ[ci, cj + dj]:
                    continue
            tentative = g_cur + cost
            nxt = (ni, nj)
            if tentative < gscore.get(nxt, float("inf")):
                came[nxt] = cur
                gscore[nxt] = tentative
                h = math.hypot(goal_ij[0] - ni, goal_ij[1] - nj)
                heapq.heappush(heap, (tentative + h, nxt))
    return None


def _smooth_line_of_sight(path_cells, occ, n):
    """Pull-string smoothing: drop cells where start→end has free LOS."""
    if len(path_cells) <= 2:
        return path_cells
    out = [path_cells[0]]
    i = 0
    while i < len(path_cells) - 1:
        j = len(path_cells) - 1
        while j > i + 1:
            if _line_free(path_cells[i], path_cells[j], occ, n):
                break
            j -= 1
        out.append(path_cells[j])
        i = j
    return out


def _line_free(a, b, occ, n):
    ai, aj = a
    bi, bj = b
    di = abs(bi - ai)
    dj = abs(bj - aj)
    si = 1 if ai < bi else -1
    sj = 1 if aj < bj else -1
    err = di - dj
    ci, cj = ai, aj
    while True:
        if not (0 <= ci < n and 0 <= cj < n) or occ[ci, cj]:
            return False
        if (ci, cj) == (bi, bj):
            return True
        e2 = 2 * err
        if e2 > -dj:
            err -= dj
            ci += si
        if e2 < di:
            err += di
            cj += sj


def _densify(waypoints_xy, step_m):
    """Resample a polyline to max segment length step_m."""
    if len(waypoints_xy) < 2:
        return list(waypoints_xy)
    out = [waypoints_xy[0]]
    for prev, cur in zip(waypoints_xy[:-1], waypoints_xy[1:]):
        dx = cur[0] - prev[0]
        dy = cur[1] - prev[1]
        d = math.hypot(dx, dy)
        if d <= step_m:
            out.append(cur)
            continue
        k = int(math.ceil(d / step_m))
        for i in range(1, k + 1):
            t = i / k
            out.append((prev[0] + t * dx, prev[1] + t * dy))
    return out


def plan_path(start_xy, goal_xy, obstacles, bounds=(-25.0, 25.0),
              grid_res=0.5, inflate=0.75, subwp_step=2.0):
    """Return a list of (x,y) sub-waypoints from start to goal, or None on failure.

    obstacles: list of (cx, cy, size) where size is full side length (axis-aligned square).
    bounds: (lo, hi) shared for x and y (square arena).
    grid_res: cell size in m.
    inflate: robot_radius + safety margin in m.
    subwp_step: max spacing between output sub-waypoints.
    """
    occ, lo, res, n = _inflate_grid(obstacles, bounds, grid_res, inflate)
    sij = _nearest_free(occ, _world_to_cell(start_xy, lo, res, n), n)
    gij = _nearest_free(occ, _world_to_cell(goal_xy, lo, res, n), n)
    cells = _astar(occ, sij, gij, n)
    if cells is None:
        return None
    cells = _smooth_line_of_sight(cells, occ, n)
    xy = [_cell_to_world(c, lo, res) for c in cells]
    # Force exact start and goal so APF reach-check hits the real target.
    xy[0] = (float(start_xy[0]), float(start_xy[1]))
    xy[-1] = (float(goal_xy[0]), float(goal_xy[1]))
    xy = _densify(xy, subwp_step)
    return xy


def expand_waypoints(original_waypoints, robot_xy, obstacles,
                     bounds=(-25.0, 25.0), grid_res=0.5, inflate=0.75,
                     subwp_step=2.0):
    """Expand a list of {label, pos} waypoints into a dense list with A* sub-waypoints.

    The original waypoint labels/positions are preserved at the end of each segment;
    interior sub-waypoints get a transparent label like "A.1", "A.2", ...
    """
    out = []
    prev_xy = (float(robot_xy[0]), float(robot_xy[1]))
    n_planned = 0
    n_fallback = 0
    for wp in original_waypoints:
        goal = (float(wp["pos"][0]), float(wp["pos"][1]))
        path = plan_path(prev_xy, goal, obstacles, bounds, grid_res, inflate, subwp_step)
        if path is None or len(path) < 2:
            out.append({"label": wp["label"], "pos": np.array(goal)})
            n_fallback += 1
        else:
            for k, pt in enumerate(path[1:-1], start=1):
                out.append({"label": f"{wp['label']}.{k}", "pos": np.array(pt)})
            out.append({"label": wp["label"], "pos": np.array(goal)})
            n_planned += 1
        prev_xy = goal
    return out, n_planned, n_fallback
