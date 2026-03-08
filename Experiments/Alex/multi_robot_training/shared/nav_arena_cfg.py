"""Procedural arena generator for Phase C navigation training.

Generates 30x30m arenas with random obstacles (cylinders, boxes, walls)
for obstacle-avoidance training. 4 existing arenas held out for evaluation.

Arena types:
    - sparse: Few large obstacles (easy pathfinding)
    - dense: Many small obstacles (tight maneuvering)
    - corridor: Narrow passages between walls
    - mixed: Random combination of all obstacle types

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ObstacleCfg:
    """Configuration for a single obstacle type."""
    count_range: tuple[int, int] = (5, 15)
    height_range: tuple[float, float] = (0.3, 1.5)
    # For cylinders: radius range; for boxes: width/depth range
    size_range: tuple[float, float] = (0.2, 1.0)
    shape: str = "cylinder"  # "cylinder", "box", "wall"


@dataclass
class NavArenaCfg:
    """Configuration for procedural navigation arena generation."""

    # Arena dimensions
    arena_size: tuple[float, float] = (30.0, 30.0)
    border_height: float = 2.0
    border_thickness: float = 0.2

    # Goal placement
    goal_min_dist: float = 5.0
    goal_max_dist: float = 25.0

    # Obstacle configs per arena type
    obstacle_types: dict[str, list[ObstacleCfg]] = field(default_factory=lambda: {
        "sparse": [
            ObstacleCfg(count_range=(3, 8), size_range=(0.5, 1.5), shape="cylinder"),
            ObstacleCfg(count_range=(2, 5), size_range=(0.5, 2.0), shape="box"),
        ],
        "dense": [
            ObstacleCfg(count_range=(15, 30), size_range=(0.2, 0.6), shape="cylinder"),
            ObstacleCfg(count_range=(5, 10), size_range=(0.3, 0.8), shape="box"),
        ],
        "corridor": [
            ObstacleCfg(count_range=(4, 8), size_range=(3.0, 8.0), height_range=(1.0, 2.0), shape="wall"),
            ObstacleCfg(count_range=(3, 6), size_range=(0.3, 0.5), shape="cylinder"),
        ],
        "mixed": [
            ObstacleCfg(count_range=(5, 12), size_range=(0.3, 1.0), shape="cylinder"),
            ObstacleCfg(count_range=(3, 8), size_range=(0.4, 1.2), shape="box"),
            ObstacleCfg(count_range=(2, 4), size_range=(2.0, 5.0), height_range=(1.0, 2.0), shape="wall"),
        ],
    })

    # Probability of each arena type during training
    type_weights: dict[str, float] = field(default_factory=lambda: {
        "sparse": 0.25,
        "dense": 0.30,
        "corridor": 0.20,
        "mixed": 0.25,
    })

    # Safety: minimum clearance around spawn and goal
    spawn_clearance: float = 2.0
    goal_clearance: float = 1.5


def sample_arena_type(cfg: NavArenaCfg, rng: torch.Generator | None = None) -> str:
    """Sample an arena type based on configured weights."""
    types = list(cfg.type_weights.keys())
    weights = torch.tensor([cfg.type_weights[t] for t in types])
    weights = weights / weights.sum()
    idx = torch.multinomial(weights, 1, generator=rng).item()
    return types[idx]


def generate_obstacle_positions(
    cfg: NavArenaCfg,
    arena_type: str,
    spawn_pos: tuple[float, float] = (0.0, 0.0),
    goal_pos: tuple[float, float] = (15.0, 0.0),
    device: str = "cpu",
) -> list[dict]:
    """Generate random obstacle positions for a given arena type.

    Returns list of dicts with keys: shape, position (x,y), size, height, rotation.
    Ensures no obstacles overlap with spawn or goal clearance zones.
    """
    obstacles = []
    half_x, half_y = cfg.arena_size[0] / 2, cfg.arena_size[1] / 2
    spawn_t = torch.tensor(spawn_pos, device=device)
    goal_t = torch.tensor(goal_pos, device=device)

    for obs_cfg in cfg.obstacle_types.get(arena_type, []):
        count = torch.randint(obs_cfg.count_range[0], obs_cfg.count_range[1] + 1, (1,)).item()

        for _ in range(count):
            # Try up to 20 times to find valid placement
            for _attempt in range(20):
                x = (torch.rand(1).item() - 0.5) * 2 * (half_x - 1.0)
                y = (torch.rand(1).item() - 0.5) * 2 * (half_y - 1.0)
                pos = torch.tensor([x, y], device=device)

                # Check clearance from spawn and goal
                if torch.linalg.norm(pos - spawn_t) < cfg.spawn_clearance:
                    continue
                if torch.linalg.norm(pos - goal_t) < cfg.goal_clearance:
                    continue

                size = obs_cfg.size_range[0] + torch.rand(1).item() * (
                    obs_cfg.size_range[1] - obs_cfg.size_range[0]
                )
                height = obs_cfg.height_range[0] + torch.rand(1).item() * (
                    obs_cfg.height_range[1] - obs_cfg.height_range[0]
                )
                rotation = torch.rand(1).item() * 3.14159  # random yaw

                obstacles.append({
                    "shape": obs_cfg.shape,
                    "position": (x, y),
                    "size": size,
                    "height": height,
                    "rotation": rotation,
                })
                break

    return obstacles


def sample_goal_position(
    cfg: NavArenaCfg,
    spawn_pos: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    """Sample a random goal position within arena bounds, at valid distance from spawn."""
    half_x, half_y = cfg.arena_size[0] / 2, cfg.arena_size[1] / 2
    spawn_t = torch.tensor(spawn_pos)

    for _ in range(100):
        x = (torch.rand(1).item() - 0.5) * 2 * (half_x - 2.0)
        y = (torch.rand(1).item() - 0.5) * 2 * (half_y - 2.0)
        goal_t = torch.tensor([x, y])
        dist = torch.linalg.norm(goal_t - spawn_t).item()
        if cfg.goal_min_dist <= dist <= cfg.goal_max_dist:
            return (x, y)

    # Fallback: place goal at max distance along +X
    return (cfg.goal_max_dist * 0.8, 0.0)
