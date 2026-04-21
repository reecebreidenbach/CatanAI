"""
AI ASSISTED
board.py - Catan board data model.

Defines hex types, the standard tile/token distribution, and the Board class
which holds the 19 HexTile objects in a 3-4-5-4-3 row layout.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Hex types
class HexType(Enum):
    FOREST = "forest"
    PASTURE = "pasture"
    FIELD = "field"
    HILL = "hill"
    MOUNTAIN = "mountain"
    DESERT = "desert"


# Resource names
RESOURCE_NAME: dict[HexType, Optional[str]] = {
    HexType.FOREST: "Wood",
    HexType.PASTURE: "Sheep",
    HexType.FIELD: "Wheat",
    HexType.HILL: "Brick",
    HexType.MOUNTAIN: "Ore",
    HexType.DESERT: None,
}

# Standard Catan tile distribution (19 total)
STANDARD_HEX_DISTRIBUTION: list[HexType] = [
    *([HexType.FOREST] * 4),
    *([HexType.PASTURE] * 4),
    *([HexType.FIELD] * 4),
    *([HexType.HILL] * 3),
    *([HexType.MOUNTAIN] * 3),
    HexType.DESERT,
]

# Standard number tokens A-R (18 tokens, placed in alphabetical clockwise spiral order)
STANDARD_TOKENS: list[int] = [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11]

# Number of hexes per row (top to bottom)
ROW_SIZES: list[int] = [3, 4, 5, 4, 3]


# ── Ports ──────────────────────────────────────────────────────────────────────

class PortType(Enum):
    GENERIC = "generic"   # 3:1 any resource
    LUMBER  = "lumber"    # 2:1 lumber
    WOOL    = "wool"      # 2:1 wool
    GRAIN   = "grain"     # 2:1 grain
    BRICK   = "brick"     # 2:1 brick
    ORE     = "ore"       # 2:1 ore


@dataclass
class Port:
    """A coastal trading port occupying one edge (two vertices) of a hex."""
    port_type: PortType
    hex_row:   int
    hex_col:   int
    edge_idx:  int   # 0-5; port sits on edge between hex_vertices[edge_idx] and [(edge_idx+1)%6]


# Standard port positions (row, col, edge_idx, PortType) — counter-clockwise
# starting from the upper-left edge of (0,0), with a repeating 3-3-4 gap pattern
# across the 30 coastal edges.  Port types are shuffled each game; positions are fixed.
# edge_idx: 0=upper-right side, 1=lower-right side, 2=lower-left side,
#           3=left side, 4=upper-left side, 5=upper-right side (top→upper-right)
_STANDARD_PORT_POSITIONS: list[tuple] = [
    (0, 0, 4),   # upper-left  (coast idx  0)
    (1, 0, 3),   # left        (coast idx  3)
    (3, 0, 3),   # lower-left  (coast idx  7)
    (4, 0, 2),   # bottom-left (coast idx 10)
    (4, 1, 1),   # bottom      (coast idx 13)
    (3, 3, 1),   # bottom-right(coast idx 17)
    (2, 4, 0),   # right       (coast idx 20)
    (1, 3, 5),   # upper-right (coast idx 23)
    (0, 1, 5),   # top-right   (coast idx 27)
]

# 5 specific 2:1 ports + 4 generic 3:1 ports — shuffled into positions each game
_STANDARD_PORT_TYPES: list[PortType] = [
    PortType.LUMBER,
    PortType.WOOL,
    PortType.GRAIN,
    PortType.BRICK,
    PortType.ORE,
    PortType.GENERIC,
    PortType.GENERIC,
    PortType.GENERIC,
    PortType.GENERIC,
]

# Kept for backwards compatibility; types are re-shuffled in Board.build()
STANDARD_PORTS_TEMPLATE: list[tuple] = [
    (*pos, pt) for pos, pt in zip(_STANDARD_PORT_POSITIONS, _STANDARD_PORT_TYPES)
]


@dataclass
class HexTile:
    hex_type: HexType
    token: Optional[int]  # None for the desert tile
    row: int
    col: int

    @property
    def resource(self) -> Optional[str]:
        return RESOURCE_NAME[self.hex_type]

    def __repr__(self) -> str:
        tok = str(self.token) if self.token else "--"
        return f"HexTile({self.hex_type.value}, token={tok}, pos=({self.row},{self.col}))"


class Board:
    """Represents a full Catan board with 19 hex tiles."""

    def __init__(self) -> None:
        self.hexes: list[HexTile] = []
        self.ports: list[Port]    = []
        self.build()

    def build(self) -> None:
        """Lay out all 19 hex tiles, optionally shuffled."""
        types = STANDARD_HEX_DISTRIBUTION[:]
        tokens = STANDARD_TOKENS[:]

        #Create a random board by shuffling tiles
        random.shuffle(types)

        self.hexes = []
        tile_idx = 0

        # Phase 1: place tile types in row-major order
        for row, size in enumerate(ROW_SIZES):
            for col in range(size):
                t = types[tile_idx]
                self.hexes.append(HexTile(hex_type=t, token=None, row=row, col=col))
                tile_idx += 1

        # Phase 2: assign tokens in clockwise spiral order (A–R), skipping desert
        spiral_order = [
            (0, 0), (0, 1), (0, 2),
            (1, 3), (2, 4), (3, 3), (4, 2), (4, 1), (4, 0),
            (3, 0), (2, 0), (1, 0),
            (1, 1), (1, 2), (2, 3), (3, 2), (3, 1), (2, 1),
            (2, 2),
        ]
        token_idx = 0
        for row, col in spiral_order:
            h = self.get(row, col)
            if h.hex_type != HexType.DESERT:
                h.token = tokens[token_idx]
                token_idx += 1

        self.topology = BoardTopology(self.hexes)
        # Keep positions fixed; shuffle the canonical set of port types each game.
        port_types = _STANDARD_PORT_TYPES[:]
        random.shuffle(port_types)
        self.ports = [Port(pt, r, c, e) for (r, c, e), pt in zip(_STANDARD_PORT_POSITIONS, port_types)]

    def randomize(self) -> None:
        """Shuffle tiles and tokens into a new random arrangement."""
        self.build()

    def get(self, row: int, col: int) -> Optional[HexTile]:
        """Return the HexTile at (row, col), or None if not found."""
        for h in self.hexes:
            if h.row == row and h.col == col:
                return h
        return None

    def __repr__(self) -> str:
        lines = []
        for row, size in enumerate(ROW_SIZES):
            row_tiles = [self.get(row, col) for col in range(size)]
            lines.append("  ".join(repr(t) for t in row_tiles if t))
        return "\n".join(lines)


class BoardTopology:
    """
    Precomputed graph structure of the Catan board.

    Vertices are the 54 intersection points where 2-3 hexes meet.
    Edges are the 72 hex borders.

    All IDs are stable integers assigned in the order they are first
    encountered walking the board in row-major order.  This gives a
    consistent mapping that doesn't change between games (tile types are
    randomized, but the geometry never changes).

    Key attributes
    --------------
    num_vertices        int              54
    num_edges           int              72
    hex_vertices        list[list[int]]  [hex_idx]    -> [v0..v5] (CW from top)
    hex_edges           list[list[int]]  [hex_idx]    -> [e0..e5]
    edge_vertices       list[tuple]      [edge_id]    -> (v1, v2)
    edge_hexes          list[list[int]]  [edge_id]    -> [hex_idx, ...] (1 or 2)
    vertex_hexes        list[list[int]]  [vertex_id]  -> [hex_idx, ...] (2 or 3)
    vertex_neighbors    list[list[int]]  [vertex_id]  -> adjacent vertex ids
    vertex_edges        list[list[int]]  [vertex_id]  -> adjacent edge ids
    """

    def __init__(self, hexes: list[HexTile]) -> None:
        self._build(hexes)

    def _build(self, hexes: list[HexTile]) -> None:
        R  = 1.0                  # unit circumradius — independent of screen size
        hw = math.sqrt(3) * R    # hex width

        # ── Step 1: assign integer vertex IDs by position ─────────────────────
        raw_to_vid: dict[tuple[float, float], int] = {}
        hex_vertex_ids: list[list[int]] = []
        next_vid = 0

        for h in hexes:
            # Flat centre of this hex in unit coordinates
            cx = hw * (h.col - (ROW_SIZES[h.row] - 1) / 2)
            cy = h.row * 1.5 * R
            vids: list[int] = []
            for i in range(6):
                angle = math.radians(60 * i - 30)   # pointy-top
                key = (round(cx + R * math.cos(angle), 4),
                       round(cy + R * math.sin(angle), 4))
                if key not in raw_to_vid:
                    raw_to_vid[key] = next_vid
                    next_vid += 1
                vids.append(raw_to_vid[key])
            hex_vertex_ids.append(vids)

        num_vertices = next_vid  # 54 for a standard board

        # ── Step 2: assign integer edge IDs ───────────────────────────────────
        raw_to_eid: dict[tuple[int, int], int] = {}
        hex_edge_ids: list[list[int]] = []
        next_eid = 0

        for vids in hex_vertex_ids:
            eids: list[int] = []
            for i in range(6):
                v1, v2 = vids[i], vids[(i + 1) % 6]
                key = (min(v1, v2), max(v1, v2))
                if key not in raw_to_eid:
                    raw_to_eid[key] = next_eid
                    next_eid += 1
                eids.append(raw_to_eid[key])
            hex_edge_ids.append(eids)

        num_edges = next_eid  # 72 for a standard board

        # ── Step 3: build all adjacency arrays from the edge map ───────────────
        edge_verts: list[tuple[int, int]] = [(-1, -1)] * num_edges
        for (v1, v2), eid in raw_to_eid.items():
            edge_verts[eid] = (v1, v2)

        vertex_edges: list[list[int]] = [[] for _ in range(num_vertices)]
        vertex_neighbors: list[list[int]] = [[] for _ in range(num_vertices)]
        for eid, (v1, v2) in enumerate(edge_verts):
            vertex_edges[v1].append(eid)
            vertex_edges[v2].append(eid)
            if v2 not in vertex_neighbors[v1]:
                vertex_neighbors[v1].append(v2)
            if v1 not in vertex_neighbors[v2]:
                vertex_neighbors[v2].append(v1)

        vertex_hexes: list[list[int]] = [[] for _ in range(num_vertices)]
        for hidx, vids in enumerate(hex_vertex_ids):
            for v in vids:
                vertex_hexes[v].append(hidx)

        edge_hexes: list[list[int]] = [[] for _ in range(num_edges)]
        for hidx, eids in enumerate(hex_edge_ids):
            for e in eids:
                if hidx not in edge_hexes[e]:
                    edge_hexes[e].append(hidx)

        # ── Expose everything as plain list attributes ─────────────────────────
        self.num_vertices       = num_vertices           # 54
        self.num_edges          = num_edges              # 72
        self.hex_vertices       = hex_vertex_ids         # [hidx]  -> [v0..v5]
        self.hex_edges          = hex_edge_ids           # [hidx]  -> [e0..e5]
        self.edge_vertices      = edge_verts             # [eid]   -> (v1, v2)
        self.edge_hexes         = edge_hexes             # [eid]   -> [hidx, ...]
        self.vertex_hexes       = vertex_hexes           # [vid]   -> [hidx, ...]
        self.vertex_neighbors   = vertex_neighbors       # [vid]   -> [vid, ...]
        self.vertex_edges       = vertex_edges           # [vid]   -> [eid, ...]
