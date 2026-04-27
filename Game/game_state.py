"""
AI ASSISTED
game_state.py  —  Pure data: what IS the game right now?

Think of this file as a save file.  It contains every variable that
describes a single snapshot of a Catan game:
  - The board (which tile is where, which number tokens)
  - Each player's resources and pieces
  - What phase the game is in (setup, roll, main, etc.)
  - Where every settlement, city, and road sits on the board
  - Who holds the Longest Road / Largest Army cards

There is NO game logic here.  Rules and move generation live in
game_engine.py.  Keeping them separate means the AI can read/copy state
without pulling in any logic, and the GUI can display state without
knowing the rules.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from board import Board, BoardTopology, HexType, PortType


# ── Development cards ──────────────────────────────────────────────────────────

class DevCard(Enum):
    """The five development card types."""
    KNIGHT          = "knight"
    ROAD_BUILDING   = "road_building"
    YEAR_OF_PLENTY  = "year_of_plenty"
    MONOPOLY        = "monopoly"
    VICTORY_POINT   = "victory_point"


# Standard Catan development card distribution (25 total)
STANDARD_DEV_DECK: list[DevCard] = [
    *([DevCard.KNIGHT]         * 14),
    *([DevCard.ROAD_BUILDING]  * 2),
    *([DevCard.YEAR_OF_PLENTY] * 2),
    *([DevCard.MONOPOLY]       * 2),
    *([DevCard.VICTORY_POINT]  * 5),
]

DEV_CARD_LABEL: dict[DevCard, str] = {
    DevCard.KNIGHT:         "Knight",
    DevCard.ROAD_BUILDING:  "Road Building",
    DevCard.YEAR_OF_PLENTY: "Year of Plenty",
    DevCard.MONOPOLY:       "Monopoly",
    DevCard.VICTORY_POINT:  "Victory Point",
}


# ── Resources ──────────────────────────────────────────────────────────────────

class Resource(Enum):
    """The five tradeable resources."""
    LUMBER = "lumber"
    WOOL   = "wool"
    GRAIN  = "grain"
    BRICK  = "brick"
    ORE    = "ore"


# Maps each hex type to the resource it produces (None for the desert).
HEX_RESOURCE: dict[HexType, Optional[Resource]] = {
    HexType.FOREST:   Resource.LUMBER,
    HexType.PASTURE:  Resource.WOOL,
    HexType.FIELD:    Resource.GRAIN,
    HexType.HILL:     Resource.BRICK,
    HexType.MOUNTAIN: Resource.ORE,
    HexType.DESERT:   None,
}

# Maps each port type to the specific resource it trades (None = any resource).
PORT_RESOURCE: dict[PortType, Optional[Resource]] = {
    PortType.LUMBER:  Resource.LUMBER,
    PortType.WOOL:    Resource.WOOL,
    PortType.GRAIN:   Resource.GRAIN,
    PortType.BRICK:   Resource.BRICK,
    PortType.ORE:     Resource.ORE,
    PortType.GENERIC: None,
}


# ── Turn phases ─────────────────────────────────────────────────────────────────

class Phase(Enum):
    """
    What stage of the turn we're in.  The engine uses this to gate which
    actions are legal at any point.

    SETUP   – The two opening rounds: each player places two settlements
              and two roads before resources start flowing.
    ROLL    – The active player must roll the dice (or play a Knight first).
    MAIN    – After the roll: the player may build, buy a dev card, trade,
              or end their turn in any order.
    DISCARD – A 7 was rolled.  Every player holding more than 7 cards must
              discard exactly half (rounded down) before the robber moves.
    ROBBER  – The active player must move the robber to a new hex and may
              steal one card from an adjacent opponent.
    DONE    – A player has reached 10 VP.  The game is over.
    """
    SETUP   = "setup"
    ROLL    = "roll"
    MAIN    = "main"
    DISCARD = "discard"
    ROBBER  = "robber"
    DONE    = "done"


# ── Player state ────────────────────────────────────────────────────────────────

@dataclass
class PlayerState:
    """
    Everything that belongs to one player.

    resources       – Cards currently in hand, keyed by Resource enum.
                      Initialised to zero for every resource type.
    dev_cards       – Development cards in hand.  Stored as strings for now
                      ("knight", "road_building", etc.); not yet enforced.
    *_left          – Pieces remaining in the player's supply.
    knights_played  – Running count of Knight cards played; used for
                      Largest Army.
    """
    player_id:        int
    resources:        dict[Resource, int] = field(
        default_factory=lambda: {r: 0 for r in Resource}
    )
    dev_cards:        list[DevCard] = field(default_factory=list)
    # Cards bought this turn cannot be played until next turn
    dev_cards_new:    list[DevCard] = field(default_factory=list)
    settlements_left: int = 5
    cities_left:      int = 4
    roads_left:       int = 15
    knights_played:   int = 0

    # ── convenience helpers ────────────────────────────────────────────────

    def resource_count(self) -> int:
        """Total number of resource cards in hand."""
        return sum(self.resources.values())

    def can_afford(self, cost: dict[Resource, int]) -> bool:
        """Return True if hand contains enough of each resource."""
        return all(self.resources.get(r, 0) >= amt for r, amt in cost.items())

    def spend(self, cost: dict[Resource, int]) -> None:
        """Deduct resources.  Call can_afford() first."""
        for r, amt in cost.items():
            self.resources[r] -= amt

    def gain(self, gains: dict[Resource, int]) -> None:
        """Add resources to hand."""
        for r, amt in gains.items():
            self.resources[r] = self.resources.get(r, 0) + amt


# ── Full game state ─────────────────────────────────────────────────────────────

@dataclass
class GameState:
    """
    Complete snapshot of a game in progress.

    Board occupancy (settlements, cities, roads) is stored as flat integer
    lists indexed by vertex_id or edge_id from BoardTopology:

        vertex_owner[v]     – player index (0..n-1) or -1 (empty)
        vertex_building[v]  – 0 = empty, 1 = settlement, 2 = city
        edge_owner[e]       – player index (0..n-1) or -1 (no road)

    These lists are trivially convertible to numpy arrays for RL observation
    vectors (see catan_env.py, to be added later).

    Setup bookkeeping
    -----------------
    setup_turn             : 0 .. 2*num_players-1  (overall placement index)
    setup_step             : 0 = place settlement, 1 = place road
    last_placed_settlement : vertex_id of the settlement placed this half-turn;
                             used to restrict where the road may go.

    Post-7 bookkeeping
    ------------------
    pending_discards       : list of player_ids who still need to discard.
                             The engine sets current_player to pending_discards[0]
                             during the DISCARD phase.
    rolling_player         : player_id who rolled the 7; restored to
                             current_player after all discards and robber move.
    """

    board:            Board
    topology:         BoardTopology
    players:          list[PlayerState]

    # Turn / phase
    current_player:   int
    phase:            Phase
    turn_number:      int = 0
    last_roll:        Optional[int] = None

    # Setup tracking
    setup_turn:               int           = 0
    setup_step:               int           = 0
    last_placed_settlement:   Optional[int] = None

    # Board occupation
    vertex_owner:    list[int] = field(default_factory=list)
    vertex_building: list[int] = field(default_factory=list)
    edge_owner:      list[int] = field(default_factory=list)

    # Robber
    robber_hex: int = 0

    # Post-7 bookkeeping
    pending_discards: list[int] = field(default_factory=list)
    rolling_player:   Optional[int] = None   # player who rolled the 7

    # Dev card sub-state
    dev_card_played_this_turn: bool = False   # only one dev card per turn
    free_roads_remaining:      int  = 0       # roads left from Road Building card
    robber_from_knight:        bool = False   # True if robber triggered by Knight (not a 7)

    # Special development cards (None until first holder qualifies)
    longest_road_owner:  Optional[int] = None
    largest_army_owner:  Optional[int] = None

    # Development card deck (shared, face-down)
    dev_deck:  list[DevCard] = field(default_factory=list)

    # Resource bank — 19 of each resource
    bank: dict[Resource, int] = field(
        default_factory=lambda: {r: 19 for r in Resource}
    )

    # Terminal
    winner: Optional[int] = None

    # ── utility ───────────────────────────────────────────────────────────

    def copy(self) -> "GameState":
        """
        Deep-copy the entire state.

        Used by game_engine.step() so every call produces an independent
        snapshot with no aliasing — safe for MCTS tree search and rollouts.

        For high-throughput RL training you may want to replace this with a
        faster manual copy (numpy arrays, etc.), but deepcopy is correct and
        easy to reason about.
        """
        return copy.deepcopy(self)
