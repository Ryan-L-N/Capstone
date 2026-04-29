"""Loco_Policy_5_Final_Capstone_Policy custom sub-terrains."""

from .open_riser_stairs import MeshOpenRiserStairsTerrainCfg, open_riser_stairs_terrain
from .open_riser_straight import MeshOpenRiserStraightFlightCfg, open_riser_straight_terrain
from .open_riser_switchback import MeshOpenRiserSwitchbackCfg, open_riser_switchback_terrain

__all__ = [
    "MeshOpenRiserStairsTerrainCfg", "open_riser_stairs_terrain",
    "MeshOpenRiserStraightFlightCfg", "open_riser_straight_terrain",
    "MeshOpenRiserSwitchbackCfg", "open_riser_switchback_terrain",
]
