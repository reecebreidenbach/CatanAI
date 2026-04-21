"""
AI ASSISTED
game_engine.py  —  Pure logic: what CAN happen and what DOES happen.

Think of this file as the rulebook.  It never holds state itself — it
reads a GameState, decides what moves are legal, and produces the next
GameState when a move is made.

The two primary methods for AI / RL use:

    engine.legal_actions(state)         -> list[Action]
        Returns every move the current player can legally make right now.
        During RL training, this list becomes an action-mask so the agent
        never wastes gradient on illegal moves.

    engine.step(state, action)          -> (new_state, reward, done)
        Applies the action to a COPY of state (original is untouched).
        reward  = 1.0 if the acting player just won, otherwise 0.0.
        done    = True once any player reaches WIN_VP points.

Typical inner training loop:

    engine = GameEngine()
    state  = engine.new_game()
    while True:
        actions = engine.legal_actions(state)
        action  = agent.choose(state, actions)   # your AI here
        state, reward, done = engine.step(state, action)
        if done:
            break
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from board import Board, HexType, PortType
from game_state import GameState, PlayerState, Phase, Resource, HEX_RESOURCE, PORT_RESOURCE, \
    DevCard, STANDARD_DEV_DECK


# ── Action representation ───────────────────────────────────────────────────────

class ActionType(Enum):
    """
    Every distinct kind of action a player can take.

    ROLL_DICE        – Must be the first action every non-setup turn.
    PLACE_SETTLEMENT – Used in both the setup phase (free) and the main
                       game (costs Lumber + Brick + Wool + Grain).
    PLACE_ROAD       – Used in setup (free) and main game (Lumber + Brick).
    UPGRADE_CITY     – Upgrade your own settlement to a city (2 Grain + 3 Ore).
    MOVE_ROBBER      – Place the robber on a new hex after rolling 7 or
                       playing a Knight.  May steal from an adjacent player.
    DISCARD          – Discard half your cards after a 7 is rolled (if you
                       hold more than 7).  The 'discard' dict on the Action
                       specifies exactly which cards to remove.
    END_TURN         – Voluntarily end the main phase and pass to the next player.
    """
    ROLL_DICE        = auto()
    PLACE_SETTLEMENT = auto()
    PLACE_ROAD       = auto()
    UPGRADE_CITY     = auto()
    MOVE_ROBBER      = auto()
    DISCARD          = auto()
    END_TURN         = auto()
    MARITIME_TRADE   = auto()
    PLAYER_TRADE     = auto()
    BUY_DEV_CARD     = auto()


@dataclass
class Action:
    """
    A single game move.

    Only the fields relevant to the action type need to be set.
    The engine validates all populated fields in step().

    vertex_id   – target vertex (PLACE_SETTLEMENT, UPGRADE_CITY)
    edge_id     – target edge   (PLACE_ROAD)
    hex_id      – target hex    (MOVE_ROBBER)
    steal_from  – player index to steal from (MOVE_ROBBER, or None)
    discard     – {Resource: count} of cards to discard (DISCARD)
    """
    type:       ActionType
    vertex_id:  Optional[int]      = None
    edge_id:    Optional[int]      = None
    hex_id:     Optional[int]      = None
    steal_from: Optional[int]      = None
    discard:    Optional[dict]     = None
    give:            Optional[Resource] = None   # MARITIME_TRADE: resource to give
    receive:         Optional[Resource] = None   # MARITIME_TRADE: resource to receive
    give_amounts:    Optional[dict]     = None   # PLAYER_TRADE: {Resource: int} to give
    receive_amounts: Optional[dict]     = None   # PLAYER_TRADE: {Resource: int} to receive
    trade_with:      Optional[int]      = None   # PLAYER_TRADE: opponent player index

    def __repr__(self) -> str:
        parts = [self.type.name]
        if self.vertex_id      is not None: parts.append(f"v={self.vertex_id}")
        if self.edge_id        is not None: parts.append(f"e={self.edge_id}")
        if self.hex_id         is not None: parts.append(f"h={self.hex_id}")
        if self.steal_from     is not None: parts.append(f"steal={self.steal_from}")
        if self.discard        is not None: parts.append(f"discard={self.discard}")
        if self.give           is not None: parts.append(f"give={self.give.value}")
        if self.receive        is not None: parts.append(f"recv={self.receive.value}")
        if self.give_amounts   is not None: parts.append(f"give={self.give_amounts}")
        if self.receive_amounts is not None: parts.append(f"recv={self.receive_amounts}")
        if self.trade_with     is not None: parts.append(f"with={self.trade_with}")
        return f"Action({', '.join(parts)})"


# ── Game engine ─────────────────────────────────────────────────────────────────

class GameEngine:
    """
    Stateless rules engine.  Create once and reuse across many games.

    Build costs (for reference):
      Road        : 1 Lumber + 1 Brick
      Settlement  : 1 Lumber + 1 Brick + 1 Wool + 1 Grain
      City        : 2 Grain  + 3 Ore
    """

    WIN_VP: int = 10

    BUILD_COSTS: dict[str, dict[Resource, int]] = {
        "road":       {Resource.LUMBER: 1, Resource.BRICK: 1},
        "settlement": {Resource.LUMBER: 1, Resource.BRICK: 1,
                       Resource.WOOL:   1, Resource.GRAIN: 1},
        "city":       {Resource.GRAIN: 2,  Resource.ORE:   3},
        "dev_card":   {Resource.ORE: 1,    Resource.WOOL:  1, Resource.GRAIN: 1},
    }

    # ── public API ─────────────────────────────────────────────────────────────

    def new_game(self, num_players: int = 4, randomize_board: bool = True) -> GameState:
        """
        Create and return a fresh GameState ready to play.

        The board is randomly laid out (or fixed if randomize_board=False).
        All players start with empty hands.  Phase is SETUP, current player
        is 0, and the robber starts on the desert hex.
        """
        board    = Board()
        topology = board.topology
        players  = [PlayerState(player_id=i) for i in range(num_players)]

        desert_idx = next(
            i for i, h in enumerate(board.hexes) if h.hex_type == HexType.DESERT
        )

        deck = STANDARD_DEV_DECK[:]
        random.shuffle(deck)

        return GameState(
            board            = board,
            topology         = topology,
            players          = players,
            current_player   = 0,
            phase            = Phase.SETUP,
            vertex_owner     = [-1] * topology.num_vertices,
            vertex_building  = [0]  * topology.num_vertices,
            edge_owner       = [-1] * topology.num_edges,
            robber_hex       = desert_idx,
            dev_deck         = deck,
            bank             = {r: 19 for r in Resource},
        )

    def legal_actions(self, state: GameState) -> list[Action]:
        """
        Return every Action the current player may legally make.

        The returned list is never empty except in Phase.DONE.
        During DISCARD, the single sentinel Action has type=DISCARD and
        discard=None; the caller must fill in the 'discard' dict before
        passing it to step().
        """
        pid = state.current_player

        if state.phase == Phase.SETUP:
            return self._legal_setup(state, pid)
        if state.phase == Phase.ROLL:
            return [Action(ActionType.ROLL_DICE)]
        if state.phase == Phase.DISCARD:
            return [Action(ActionType.DISCARD)]   # caller fills .discard
        if state.phase == Phase.ROBBER:
            return self._legal_robber(state, pid)
        if state.phase == Phase.MAIN:
            return self._legal_main(state, pid)
        return []  # DONE

    def step(self, state: GameState, action: Action) -> tuple[GameState, float, bool]:
        """
        Apply action to a copy of state.  Returns (new_state, reward, done).

        reward  – 1.0 if state.current_player just won, else 0.0
        done    – True if the game has ended

        The original state is never modified, making this safe for tree
        search (MCTS) and parallel RL rollouts.
        """
        new_state = state.copy()
        self._apply(new_state, action)

        winner = self._check_win(new_state)
        if winner is not None:
            new_state.phase  = Phase.DONE
            new_state.winner = winner
            reward = 1.0 if winner == state.current_player else 0.0
            return new_state, reward, True

        return new_state, 0.0, False

    def compute_public_vp(self, state: GameState, pid: int) -> int:
        """
        Return the *publicly visible* VP for player pid.
        VP dev cards are secret and are intentionally excluded here;
        use compute_vp() only for win detection.
        """
        vp = sum(
            state.vertex_building[v]
            for v in range(state.topology.num_vertices)
            if state.vertex_owner[v] == pid
        )
        if state.longest_road_owner == pid:
            vp += 2
        if state.largest_army_owner == pid:
            vp += 2
        return vp

    def compute_vp(self, state: GameState, pid: int) -> int:
        """Return *total* VP including secret VP dev cards (use only for win detection)."""
        return (self.compute_public_vp(state, pid) +
                sum(1 for c in state.players[pid].dev_cards
                    if c == DevCard.VICTORY_POINT))

    # ── legal-action helpers ───────────────────────────────────────────────────

    def _legal_setup(self, state: GameState, pid: int) -> list[Action]:
        topo = state.topology
        if state.setup_step == 0:
            # Settlement: empty vertex, distance rule (no neighbour has a building)
            return [
                Action(ActionType.PLACE_SETTLEMENT, vertex_id=v)
                for v in range(topo.num_vertices)
                if self._ok_setup_vertex(state, v)
            ]
        else:
            # Road: must touch the settlement just placed this half-turn
            last = state.last_placed_settlement
            return [
                Action(ActionType.PLACE_ROAD, edge_id=e)
                for e in topo.vertex_edges[last]
                if state.edge_owner[e] == -1
            ]

    def _legal_robber(self, state: GameState, pid: int) -> list[Action]:
        """
        Robber may move to any hex except the one it's already on.
        If opponents have pieces on the target hex, one must be chosen
        to steal from; otherwise steal_from is None.
        """
        actions: list[Action] = []
        for hidx in range(len(state.board.hexes)):
            if hidx == state.robber_hex:
                continue
            victims = {
                state.vertex_owner[v]
                for v in state.topology.hex_vertices[hidx]
                if state.vertex_owner[v] not in (-1, pid)
            }
            if victims:
                for victim in victims:
                    actions.append(Action(ActionType.MOVE_ROBBER,
                                         hex_id=hidx, steal_from=victim))
            else:
                actions.append(Action(ActionType.MOVE_ROBBER,
                                      hex_id=hidx, steal_from=None))
        return actions

    def _legal_main(self, state: GameState, pid: int) -> list[Action]:
        """
        During the main phase the player may (in any order and any number
        of times, subject to resources and piece supply):
          - Build a road
          - Build a settlement
          - Upgrade a settlement to a city
          - End their turn (always available)
        Dev cards and trading are stubs for a future iteration.
        """
        p    = state.players[pid]
        topo = state.topology
        actions: list[Action] = [Action(ActionType.END_TURN)]

        if p.roads_left > 0 and p.can_afford(self.BUILD_COSTS["road"]):
            for e in range(topo.num_edges):
                if self._ok_road_edge(state, e, pid):
                    actions.append(Action(ActionType.PLACE_ROAD, edge_id=e))

        if p.settlements_left > 0 and p.can_afford(self.BUILD_COSTS["settlement"]):
            for v in range(topo.num_vertices):
                if self._ok_main_vertex(state, v, pid):
                    actions.append(Action(ActionType.PLACE_SETTLEMENT, vertex_id=v))

        if p.cities_left > 0 and p.can_afford(self.BUILD_COSTS["city"]):
            for v in range(topo.num_vertices):
                if self._ok_city_vertex(state, v, pid):
                    actions.append(Action(ActionType.UPGRADE_CITY, vertex_id=v))

        # Maritime trades — port rates or 4:1 bank rate (always available)
        for give_r in Resource:
            ratio = self._get_trade_ratio(state, pid, give_r)
            if p.resources.get(give_r, 0) >= ratio:
                for recv_r in Resource:
                    if recv_r != give_r and state.bank.get(recv_r, 0) > 0:
                        actions.append(Action(ActionType.MARITIME_TRADE,
                                              give=give_r, receive=recv_r))

        # Buy a development card
        if state.dev_deck and p.can_afford(self.BUILD_COSTS["dev_card"]):
            actions.append(Action(ActionType.BUY_DEV_CARD))

        return actions

    # ── placement validators ───────────────────────────────────────────────────

    def _ok_setup_vertex(self, state: GameState, v: int) -> bool:
        """
        Distance rule: vertex must be empty AND no adjacent vertex may have
        any building.  Applies in both setup rounds.
        """
        if state.vertex_building[v] != 0:
            return False
        return all(
            state.vertex_building[nb] == 0
            for nb in state.topology.vertex_neighbors[v]
        )

    def _ok_main_vertex(self, state: GameState, v: int, pid: int) -> bool:
        """
        Main-game settlement: distance rule PLUS must be connected to an
        existing road owned by this player.
        """
        if not self._ok_setup_vertex(state, v):
            return False
        return any(
            state.edge_owner[e] == pid
            for e in state.topology.vertex_edges[v]
        )

    def _ok_road_edge(self, state: GameState, e: int, pid: int,
                      is_setup: bool = False) -> bool:
        """
        A road may be placed on edge e if:
          1. The edge is empty.
          2. (Setup)  it touches the settlement placed this half-turn.
          3. (Main)   at least one endpoint connects to this player's network
                      without crossing an opponent's settlement.
        """
        if state.edge_owner[e] != -1:
            return False
        if is_setup:
            v1, v2 = state.topology.edge_vertices[e]
            return v1 == state.last_placed_settlement or \
                   v2 == state.last_placed_settlement
        v1, v2 = state.topology.edge_vertices[e]
        return (self._connects_at(state, v1, e, pid) or
                self._connects_at(state, v2, e, pid))

    def _connects_at(self, state: GameState, v: int, new_edge: int,
                     pid: int) -> bool:
        """
        Can player pid extend their road network through vertex v?
          - YES if they own the settlement/city there.
          - YES if the vertex is empty and they already have an adjacent road
                (other than the edge being placed).
          - NO  if an opponent's building blocks the vertex.
        """
        owner = state.vertex_owner[v]
        if owner == pid:
            return True
        if owner != -1:
            return False   # blocked by opponent piece
        return any(
            state.edge_owner[adj] == pid
            for adj in state.topology.vertex_edges[v]
            if adj != new_edge
        )

    def _ok_city_vertex(self, state: GameState, v: int, pid: int) -> bool:
        """Vertex must hold this player's settlement (not already a city)."""
        return state.vertex_owner[v] == pid and state.vertex_building[v] == 1

    # ── action application ─────────────────────────────────────────────────────

    def _apply(self, state: GameState, action: Action) -> None:
        """Dispatch to the appropriate handler (state is mutated in place)."""
        dispatch = {
            ActionType.PLACE_SETTLEMENT: self._do_settlement,
            ActionType.PLACE_ROAD:       self._do_road,
            ActionType.UPGRADE_CITY:     self._do_city,
            ActionType.ROLL_DICE:        self._do_roll,
            ActionType.MOVE_ROBBER:      self._do_robber,
            ActionType.DISCARD:          self._do_discard,
            ActionType.END_TURN:         self._do_end_turn,
            ActionType.MARITIME_TRADE:   self._do_trade,
            ActionType.PLAYER_TRADE:     self._do_player_trade,
            ActionType.BUY_DEV_CARD:     self._do_buy_dev_card,
        }
        dispatch[action.type](state, action)

    def _do_settlement(self, state: GameState, action: Action) -> None:
        v   = action.vertex_id
        pid = state.current_player

        # Charge resources in the main game only (setup is free)
        if state.phase == Phase.MAIN:
            cost = self.BUILD_COSTS["settlement"]
            state.players[pid].spend(cost)
            for r, amt in cost.items():
                state.bank[r] = state.bank.get(r, 0) + amt

        state.vertex_owner[v]    = pid
        state.vertex_building[v] = 1
        state.players[pid].settlements_left -= 1

        if state.phase == Phase.SETUP:
            state.last_placed_settlement = v
            # Second setup round: grant one card from each adjacent hex
            if state.setup_turn >= len(state.players):
                self._collect_setup_resources(state, pid, v)
            state.setup_step = 1   # next action for this turn is road
        else:
            # A new settlement can split opponent longest roads
            self._update_longest_road(state)

    def _do_road(self, state: GameState, action: Action) -> None:
        e   = action.edge_id
        pid = state.current_player

        if state.phase == Phase.MAIN:
            cost = self.BUILD_COSTS["road"]
            state.players[pid].spend(cost)
            for r, amt in cost.items():
                state.bank[r] = state.bank.get(r, 0) + amt

        state.edge_owner[e] = pid
        state.players[pid].roads_left -= 1

        if state.phase == Phase.SETUP:
            self._advance_setup(state)
        else:
            self._update_longest_road(state)

    def _do_city(self, state: GameState, action: Action) -> None:
        v   = action.vertex_id
        pid = state.current_player
        cost = self.BUILD_COSTS["city"]
        state.players[pid].spend(cost)
        for r, amt in cost.items():
            state.bank[r] = state.bank.get(r, 0) + amt
        state.vertex_building[v]            = 2
        state.players[pid].settlements_left += 1  # settlement goes back to supply
        state.players[pid].cities_left      -= 1

    def _do_roll(self, state: GameState, action: Action) -> None:
        """
        Roll two dice.

        7  →  check who must discard → move robber
        else → distribute resources to all players with pieces on matching hexes
        """
        roll = random.randint(1, 6) + random.randint(1, 6)
        state.last_roll     = roll
        state.rolling_player = state.current_player

        if roll == 7:
            must_discard = [
                i for i, p in enumerate(state.players) if p.resource_count() > 7
            ]
            if must_discard:
                state.pending_discards = must_discard
                state.current_player   = must_discard[0]
                state.phase            = Phase.DISCARD
            else:
                state.phase = Phase.ROBBER
        else:
            self._distribute_resources(state, roll)
            state.phase = Phase.MAIN

    def _do_robber(self, state: GameState, action: Action) -> None:
        """Move the robber; optionally steal one random card from a victim."""
        state.robber_hex = action.hex_id
        if action.steal_from is not None:
            victim = state.players[action.steal_from]
            thief  = state.players[state.current_player]
            pool   = [r for r, cnt in victim.resources.items() if cnt > 0]
            if pool:
                stolen = random.choice(pool)
                victim.resources[stolen] -= 1
                thief.resources[stolen]  += 1
        state.phase = Phase.MAIN

    def _do_discard(self, state: GameState, action: Action) -> None:
        """
        Remove the specified cards from the current discarder's hand.

        The engine validates the count at step() time.  After the last
        discard, transition to ROBBER and restore the original rolling player.
        """
        discarder = state.players[state.current_player]
        if action.discard:
            for r, amt in action.discard.items():
                actual = min(amt, discarder.resources[r])
                discarder.resources[r] -= actual
                state.bank[r] = state.bank.get(r, 0) + actual
        state.pending_discards.pop(0)

        if state.pending_discards:
            state.current_player = state.pending_discards[0]
        else:
            state.current_player = state.rolling_player  # type: ignore[assignment]
            state.phase = Phase.ROBBER

    def _do_end_turn(self, state: GameState, _: Action) -> None:
        pid = state.current_player
        p   = state.players[pid]
        # Cards bought this turn become playable next turn
        p.dev_cards.extend(p.dev_cards_new)
        p.dev_cards_new.clear()

        state.turn_number    += 1
        state.current_player  = (state.current_player + 1) % len(state.players)
        state.phase           = Phase.ROLL
        state.last_roll       = None

    def _do_trade(self, state: GameState, action: Action) -> None:
        """Execute a maritime trade: give N of one resource, receive 1 of another."""
        pid   = state.current_player
        ratio = self._get_trade_ratio(state, pid, action.give)
        state.players[pid].resources[action.give]    -= ratio
        state.bank[action.give]                      += ratio
        state.players[pid].resources[action.receive]  = (
            state.players[pid].resources.get(action.receive, 0) + 1
        )
        state.bank[action.receive] -= 1

    def _do_buy_dev_card(self, state: GameState, action: Action) -> None:
        """Draw the top card from the deck; charge Ore + Wool + Grain."""
        pid = state.current_player
        cost = self.BUILD_COSTS["dev_card"]
        state.players[pid].spend(cost)
        # Return build cost to bank
        for r, amt in cost.items():
            state.bank[r] = state.bank.get(r, 0) + amt
        card = state.dev_deck.pop(0)
        # VP cards go straight into the playable hand (they are always secret);
        # all other cards are held in dev_cards_new and become playable next turn.
        if card == DevCard.VICTORY_POINT:
            state.players[pid].dev_cards.append(card)
        else:
            state.players[pid].dev_cards_new.append(card)

    def _do_player_trade(self, state: GameState, action: Action) -> None:
        """Execute a player-to-player trade agreed upon through the GUI."""
        pid        = state.current_player
        target_pid = action.trade_with
        give       = action.give_amounts    or {}
        receive    = action.receive_amounts or {}
        for r, amt in give.items():
            state.players[pid].resources[r]        -= amt
            state.players[target_pid].resources[r] = (
                state.players[target_pid].resources.get(r, 0) + amt
            )
        for r, amt in receive.items():
            state.players[target_pid].resources[r] -= amt
            state.players[pid].resources[r]         = (
                state.players[pid].resources.get(r, 0) + amt
            )

    # ── trade helpers ────────────────────────────────────────────────────────────

    def _get_trade_ratio(self, state: GameState, pid: int, resource: Resource) -> int:
        """Return the best trade ratio for this player and resource (2, 3, or 4)."""
        ratio = 4  # default bank rate
        hex_idx = {(h.row, h.col): i for i, h in enumerate(state.board.hexes)}
        for port in state.board.ports:
            hidx = hex_idx.get((port.hex_row, port.hex_col))
            if hidx is None:
                continue
            vids = state.topology.hex_vertices[hidx]
            va   = vids[port.edge_idx]
            vb   = vids[(port.edge_idx + 1) % 6]
            if state.vertex_owner[va] == pid or state.vertex_owner[vb] == pid:
                port_res = PORT_RESOURCE.get(port.port_type)
                if port_res is None:             # generic 3:1
                    ratio = min(ratio, 3)
                elif port_res == resource:       # specific 2:1
                    ratio = min(ratio, 2)
        return ratio

    # ── setup progression ──────────────────────────────────────────────────────

    def _advance_setup(self, state: GameState) -> None:
        """
        Move to the next setup half-turn or transition to the main game.

        Setup order for 4 players (snake draft):
            Round 1 (setup_turn 0-3): players  0, 1, 2, 3
            Round 2 (setup_turn 4-7): players  3, 2, 1, 0
        """
        n = len(state.players)
        state.setup_turn += 1
        state.setup_step  = 0
        state.last_placed_settlement = None

        if state.setup_turn >= 2 * n:
            # All setup done — begin main game with player 0
            state.current_player = 0
            state.phase          = Phase.ROLL
        else:
            t = state.setup_turn
            state.current_player = t if t < n else (2 * n - 1 - t)
            state.phase          = Phase.SETUP

    # ── resource helpers ───────────────────────────────────────────────────────

    def _collect_setup_resources(self, state: GameState, pid: int,
                                 v: int) -> None:
        """
        At the end of the second setup round, grant the placing player one
        card of each resource type adjacent to their new settlement.
        Bank limits apply: if a resource is depleted, it is skipped.
        """
        for hidx in state.topology.vertex_hexes[v]:
            res = HEX_RESOURCE[state.board.hexes[hidx].hex_type]
            if res is not None and state.bank.get(res, 0) > 0:
                state.players[pid].resources[res] += 1
                state.bank[res] -= 1

    def _distribute_resources(self, state: GameState, roll: int) -> None:
        """
        For every hex whose number token matches the roll (and the robber is
        not present), give each adjacent player 1 card per settlement and
        2 cards per city of that hex's resource type.

        Bank rule: if the total demand for a resource from a single hex
        exceeds the bank supply, *no one* receives that resource from
        that hex (standard Catan scarcity rule).
        """
        for hidx, h in enumerate(state.board.hexes):
            if h.token != roll or hidx == state.robber_hex:
                continue
            res = HEX_RESOURCE[h.hex_type]
            if res is None:
                continue
            # Calculate each player's demand from this hex
            demand: dict[int, int] = {}
            for v in state.topology.hex_vertices[hidx]:
                owner = state.vertex_owner[v]
                if owner == -1:
                    continue
                amount = 2 if state.vertex_building[v] == 2 else 1
                demand[owner] = demand.get(owner, 0) + amount
            total_demand = sum(demand.values())
            if total_demand == 0:
                continue
            # If total demand exceeds bank supply, nobody gets anything
            if total_demand > state.bank.get(res, 0):
                continue
            for owner, amount in demand.items():
                state.players[owner].resources[res] += amount
                state.bank[res] -= amount

    # ── special cards ──────────────────────────────────────────────────────────

    def _update_longest_road(self, state: GameState) -> None:
        """
        Recompute Longest Road after any road is placed or a settlement
        splits an opponent's network.

        Rules:
          - Minimum 5 roads to claim the card.
          - You must strictly exceed the current holder to take it.
            Ties do NOT change the holder.
        """
        MIN_ROAD     = 5
        holder       = state.longest_road_owner
        holder_len   = (self._longest_road_dfs(state, holder)
                        if holder is not None else 0)

        for pid in range(len(state.players)):
            length = self._longest_road_dfs(state, pid)
            if length >= MIN_ROAD and length > holder_len:
                state.longest_road_owner = pid
                holder_len = length

    def _longest_road_dfs(self, state: GameState, pid: int) -> int:
        """
        Find the longest continuous road for player pid using DFS.

        Each edge is visited at most once per path.  A vertex occupied by
        an opponent's settlement or city breaks the continuity (the road
        cannot pass through it), but the player's own settlements do not.

        Time complexity: O(E!) worst-case in theory, but E ≤ 72 and the
        topology is highly constrained, so it runs in microseconds in practice.
        """
        topo         = state.topology
        player_edges = {e for e in range(topo.num_edges)
                        if state.edge_owner[e] == pid}
        if not player_edges:
            return 0

        best = 0

        def dfs(v: int, visited_edges: set[int]) -> None:
            nonlocal best
            for e in topo.vertex_edges[v]:
                if e not in player_edges or e in visited_edges:
                    continue
                v1, v2 = topo.edge_vertices[e]
                nxt    = v2 if v1 == v else v1
                # Opponent building blocks the road
                if state.vertex_owner[nxt] not in (-1, pid):
                    continue
                visited_edges.add(e)
                best = max(best, len(visited_edges))
                dfs(nxt, visited_edges)
                visited_edges.remove(e)

        # Start DFS from every vertex that borders this player's roads
        start_verts: set[int] = set()
        for e in player_edges:
            v1, v2 = topo.edge_vertices[e]
            start_verts.add(v1)
            start_verts.add(v2)
        for sv in start_verts:
            dfs(sv, set())

        return best

    def _update_largest_army(self, state: GameState) -> None:
        """
        Award Largest Army to any player who has played ≥ 3 knights and
        strictly exceeds the current holder's count.
        (Called after a Knight dev card is played — stub for now.)
        """
        MIN_ARMY = 3
        holder   = state.largest_army_owner
        for pid, p in enumerate(state.players):
            if p.knights_played < MIN_ARMY:
                continue
            if (holder is None or
                    p.knights_played > state.players[holder].knights_played):
                state.largest_army_owner = pid
                holder = pid

    # ── win condition ──────────────────────────────────────────────────────────

    def _check_win(self, state: GameState) -> Optional[int]:
        """Return the player_id who won, or None if the game continues."""
        for pid in range(len(state.players)):
            if self.compute_vp(state, pid) >= self.WIN_VP:
                return pid
        return None
