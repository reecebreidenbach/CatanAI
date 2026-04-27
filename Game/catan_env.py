"""
catan_env.py  —  Gym-style multi-agent environment wrapper for the Catan engine.

Quick-start
-----------
    from catan_env import CatanEnv, RandomAgent, run_episode

    env    = CatanEnv(num_players=4)
    agents = [RandomAgent() for _ in range(4)]
    winner, history = run_episode(env, agents, verbose=True)

Observation vector
------------------
The observation is always from the perspective of the CURRENT player.
Players are rotated so "slot 0" always means *me*, slot 1 means the next
player clockwise, etc.  This makes the policy invariant to seat position.

Hidden-information rule:
    - The current player sees their own full hand (resources + dev cards).
    - Opponents' hands show only the TOTAL card count, not per-resource
      breakdown.  (Standard Catan hidden-information rule.)
    - VP dev cards are kept secret in the player's own observation only.

Observation size for N players:  ~955 floats  (N=4, which is the default).

Action space (flat integer index, size=298)
-------------------------------------------
    0            ROLL_DICE
    1..54        PLACE_SETTLEMENT  vertex 0-53
    55..126      PLACE_ROAD        edge 0-71
    127..180     UPGRADE_CITY      vertex 0-53
    181..275     MOVE_ROBBER       hex*5 + steal_player (4 = nobody to steal)
    276          END_TURN
    277..296     MARITIME_TRADE    (give_res * 4 + recv_offset) — 20 combos
    297          BUY_DEV_CARD

PLAYER_TRADE is not in the RL action space (too complex for flat indexing).
DISCARD is handled automatically by a greedy heuristic so the agent never
has to deal with it.

Reward shaping (optional)
--------------------------
Pass reward_shaping=True to add configurable shaping rewards in the main game.
By default this gives a small reward for public VP gains, roads, and buying
dev cards. Terminal win/loss rewards are also configurable.

Plugging in a neural network
-----------------------------
    obs_size   = env.obs_size()      # input dimension
    act_size   = env.action_size()   # output dimension (logits)

    # In training loop:
    obs, mask  = env.observe()
    logits     = network(obs)
    logits[~mask] = -1e9             # mask illegal actions before softmax
    action_idx = sample(softmax(logits))
    rewards, done = env.step(action_idx)
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Optional

import numpy as np

from game_engine import GameEngine, Action, ActionType
from game_state  import (
    GameState, Phase, Resource, DevCard,
    HEX_RESOURCE,
)
from board import HexType


# ── Stable orderings for one-hot / index lookups ────────────────────────────────
_RESOURCES: list[Resource] = list(Resource)          # 5 resources, fixed order
_DEV_TYPES: list[DevCard]  = list(DevCard)           # 5 dev-card types
_PHASES: list[Phase] = [
    Phase.SETUP, Phase.ROLL, Phase.MAIN,
    Phase.ROBBER, Phase.DISCARD, Phase.DONE,
]
_HEX_TYPES: list[HexType] = [
    HexType.FOREST, HexType.PASTURE, HexType.FIELD,
    HexType.HILL, HexType.MOUNTAIN, HexType.DESERT,
]

NUM_VERTICES  = 54
NUM_EDGES     = 72
NUM_HEXES     = 19
NUM_RESOURCES = len(_RESOURCES)    # 5
NUM_DEV_TYPES = len(_DEV_TYPES)    # 5
NUM_HEX_TYPES = len(_HEX_TYPES)    # 6
NUM_PHASES    = len(_PHASES)       # 6


# ── Action index layout ─────────────────────────────────────────────────────────
ACTION_SIZE   = 320   # total number of distinct RL actions

_ACT_ROLL         = 0
_ACT_SETTLE       = 1          # + vertex_id  →  1..54
_ACT_ROAD         = 55         # + edge_id    → 55..126
_ACT_CITY         = 127        # + vertex_id  → 127..180
_ACT_ROBBER       = 181        # + hex_id*5 + steal_slot → 181..275
_ACT_END          = 276
_ACT_TRADE        = 277        # + trade_sub_idx → 277..296   (20 combos)
_ACT_BUY          = 297
_ACT_KNIGHT       = 298        # 1 action
_ACT_MONOPOLY     = 299        # + resource_idx → 299..303  (5 actions)
_ACT_YOP          = 304        # + combo_idx   → 304..318  (15 combos)
_ACT_ROAD_BUILDING = 319       # 1 action


def _trade_sub(give: Resource, recv: Resource) -> int:
    """Encode a (give, recv) pair as 0..19. give != recv required."""
    g = _RESOURCES.index(give)
    r = _RESOURCES.index(recv)
    rr = r if r < g else r - 1   # skip the 'same resource' slot
    return g * 4 + rr


def _trade_sub_inv(sub: int) -> tuple[Resource, Resource]:
    """Inverse of _trade_sub."""
    g, rr = divmod(sub, 4)
    r     = rr if rr < g else rr + 1
    return _RESOURCES[g], _RESOURCES[r]


# Year of Plenty: 15 (r1, r2) combos where r1_idx <= r2_idx (with repetition allowed)
_YOP_COMBOS: list[tuple[Resource, Resource]] = [
    (r1, r2)
    for i, r1 in enumerate(_RESOURCES)
    for r2 in _RESOURCES[i:]
]
_YOP_COMBO_INDEX: dict[tuple[Resource, Resource], int] = {
    combo: idx for idx, combo in enumerate(_YOP_COMBOS)
}


def _yop_sub(r1: Resource, r2: Resource) -> int:
    """Encode a Year of Plenty (r1, r2) pair as 0..14.  r1_idx must be <= r2_idx."""
    key = (r1, r2) if _RESOURCES.index(r1) <= _RESOURCES.index(r2) else (r2, r1)
    return _YOP_COMBO_INDEX[key]


def _yop_sub_inv(sub: int) -> tuple[Resource, Resource]:
    """Inverse of _yop_sub."""
    return _YOP_COMBOS[sub]


# ── Observation size ────────────────────────────────────────────────────────────

def _obs_size(num_players: int) -> int:
    N = num_players
    return (
        NUM_HEXES * (NUM_HEX_TYPES + 1)  # hex type one-hot + token normalised
        + NUM_HEXES                        # robber one-hot
        + NUM_VERTICES * (N + 1 + 2)      # owner one-hot (N+1) + settlement + city
        + NUM_EDGES * (N + 1)             # edge owner one-hot (N+1)
        + NUM_RESOURCES                    # my resources (normalised)
        + NUM_DEV_TYPES                    # my playable dev cards
        + NUM_DEV_TYPES                    # my dev_cards_new (unplayable this turn)
        + 3                                # my pieces: roads/15, settlements/5, cities/4
        + N * 5                            # per player: total_cards, public_vp, knights,
                                           #             settlements_placed, cities_placed
        + (N + 1) * 2                      # longest_road_owner, largest_army_owner (one-hot)
        + NUM_RESOURCES                    # bank (normalised /19)
        + 1                                # dev deck remaining (normalised /25)
        + NUM_PHASES                       # phase one-hot
    )


# ── Observation encoding ────────────────────────────────────────────────────────

def encode_obs(state: GameState, agent_pid: int, engine: GameEngine) -> np.ndarray:
    """
    Build the flat float32 observation vector for *agent_pid*.

    Players are re-indexed relative to agent_pid:
        slot 0 = agent_pid  (me)
        slot 1 = (agent_pid+1) % N
        ...
    This makes the representation seat-invariant.
    """
    N    = len(state.players)
    topo = state.topology
    obs: list[float] = []

    # ── Hex features ─────────────────────────────────────────────────────────
    for hidx, h in enumerate(state.board.hexes):
        # Hex type one-hot (NUM_HEX_TYPES values)
        obs.extend(1.0 if h.hex_type == t else 0.0 for t in _HEX_TYPES)
        # Number token, normalised to [0, 1]  (desert token = 0)
        obs.append((h.token or 0) / 12.0)

    # Robber position one-hot
    obs.extend(1.0 if i == state.robber_hex else 0.0 for i in range(NUM_HEXES))

    # ── Vertex features ───────────────────────────────────────────────────────
    # (N+1) owner slots: 0 = nobody, 1..N = relative player slots
    for v in range(topo.num_vertices):
        raw_owner = state.vertex_owner[v]
        slot      = _abs_to_rel(raw_owner, agent_pid, N)
        owner_oh  = [0.0] * (N + 1)
        owner_oh[slot] = 1.0
        obs.extend(owner_oh)
        bldg = state.vertex_building[v]
        obs.append(1.0 if bldg == 1 else 0.0)   # settlement flag
        obs.append(1.0 if bldg == 2 else 0.0)   # city flag

    # ── Edge features ─────────────────────────────────────────────────────────
    for e in range(topo.num_edges):
        raw_owner = state.edge_owner[e]
        slot      = _abs_to_rel(raw_owner, agent_pid, N)
        owner_oh  = [0.0] * (N + 1)
        owner_oh[slot] = 1.0
        obs.extend(owner_oh)

    # ── My private hand ───────────────────────────────────────────────────────
    me = state.players[agent_pid]
    for r in _RESOURCES:
        obs.append(me.resources.get(r, 0) / 19.0)

    dc      = Counter(me.dev_cards)
    dc_new  = Counter(me.dev_cards_new)
    for d in _DEV_TYPES:
        obs.append(float(dc.get(d, 0)))
    for d in _DEV_TYPES:
        obs.append(float(dc_new.get(d, 0)))

    obs.append(me.roads_left       / 15.0)
    obs.append(me.settlements_left /  5.0)
    obs.append(me.cities_left      /  4.0)

    # ── Per-player public info (relative ordering) ────────────────────────────
    for slot in range(N):
        pid = _rel_to_abs(slot, agent_pid, N)
        p   = state.players[pid]
        obs.append(p.resource_count()                          / 95.0)
        obs.append(engine.compute_public_vp(state, pid)        / 10.0)
        obs.append(p.knights_played                            / 14.0)
        obs.append((5 - p.settlements_left)                    /  5.0)   # placed settlements
        obs.append((4 - p.cities_left)                         /  4.0)   # placed cities

    # ── Special cards (Longest Road / Largest Army) ───────────────────────────
    for holder in (state.longest_road_owner, state.largest_army_owner):
        h_oh = [0.0] * (N + 1)    # indices 0..N-1 = relative players, N = nobody
        if holder is None:
            h_oh[N] = 1.0
        else:
            h_oh[(holder - agent_pid) % N] = 1.0
        obs.extend(h_oh)

    # ── Bank & deck ───────────────────────────────────────────────────────────
    for r in _RESOURCES:
        obs.append(state.bank.get(r, 0) / 19.0)
    obs.append(len(state.dev_deck) / 25.0)

    # ── Phase one-hot ─────────────────────────────────────────────────────────
    obs.extend(1.0 if state.phase == p else 0.0 for p in _PHASES)

    result = np.array(obs, dtype=np.float32)
    assert len(result) == _obs_size(N), (
        f"encode_obs produced {len(result)} values, expected {_obs_size(N)}"
    )
    return result


def _abs_to_rel(pid: int, agent_pid: int, N: int) -> int:
    """Convert absolute player id to relative slot (0=nobody, 1=me, 2..N=others)."""
    if pid == -1:
        return 0
    return (pid - agent_pid) % N + 1


def _rel_to_abs(slot: int, agent_pid: int, N: int) -> int:
    """Convert relative slot (0=me, 1=next, ...) back to absolute player id."""
    return (agent_pid + slot) % N


# ── Action encode / decode ──────────────────────────────────────────────────────

def encode_action(action: Action, state: GameState) -> int:
    """Map a legal Action to its flat integer index (0..ACTION_SIZE-1)."""
    t = action.type
    if t == ActionType.ROLL_DICE:
        return _ACT_ROLL
    if t == ActionType.PLACE_SETTLEMENT:
        return _ACT_SETTLE + action.vertex_id
    if t == ActionType.PLACE_ROAD:
        return _ACT_ROAD + action.edge_id
    if t == ActionType.UPGRADE_CITY:
        return _ACT_CITY + action.vertex_id
    if t == ActionType.MOVE_ROBBER:
        steal_slot = action.steal_from if action.steal_from is not None else 4
        return _ACT_ROBBER + action.hex_id * 5 + steal_slot
    if t == ActionType.END_TURN:
        return _ACT_END
    if t == ActionType.MARITIME_TRADE:
        return _ACT_TRADE + _trade_sub(action.give, action.receive)
    if t == ActionType.BUY_DEV_CARD:
        return _ACT_BUY
    if t == ActionType.PLAY_KNIGHT:
        return _ACT_KNIGHT
    if t == ActionType.PLAY_MONOPOLY:
        return _ACT_MONOPOLY + _RESOURCES.index(action.receive)
    if t == ActionType.PLAY_YEAR_OF_PLENTY:
        return _ACT_YOP + _yop_sub(action.give, action.receive)
    if t == ActionType.PLAY_ROAD_BUILDING:
        return _ACT_ROAD_BUILDING
    raise ValueError(
        f"Cannot encode {t.name} into flat RL action space "
        "(PLAYER_TRADE and DISCARD are handled outside the action index)"
    )


def decode_action(idx: int, state: GameState) -> Action:
    """Map a flat integer index back to an Action dataclass."""
    if idx == _ACT_ROLL:
        return Action(ActionType.ROLL_DICE)
    if _ACT_SETTLE <= idx < _ACT_ROAD:
        return Action(ActionType.PLACE_SETTLEMENT, vertex_id=idx - _ACT_SETTLE)
    if _ACT_ROAD <= idx < _ACT_CITY:
        return Action(ActionType.PLACE_ROAD, edge_id=idx - _ACT_ROAD)
    if _ACT_CITY <= idx < _ACT_ROBBER:
        return Action(ActionType.UPGRADE_CITY, vertex_id=idx - _ACT_CITY)
    if _ACT_ROBBER <= idx < _ACT_END:
        offset     = idx - _ACT_ROBBER
        hex_id     = offset // 5
        steal_slot = offset %  5
        steal_from = steal_slot if steal_slot < 4 else None
        return Action(ActionType.MOVE_ROBBER, hex_id=hex_id, steal_from=steal_from)
    if idx == _ACT_END:
        return Action(ActionType.END_TURN)
    if _ACT_TRADE <= idx < _ACT_BUY:
        give, recv = _trade_sub_inv(idx - _ACT_TRADE)
        return Action(ActionType.MARITIME_TRADE, give=give, receive=recv)
    if idx == _ACT_BUY:
        return Action(ActionType.BUY_DEV_CARD)
    if idx == _ACT_KNIGHT:
        return Action(ActionType.PLAY_KNIGHT)
    if _ACT_MONOPOLY <= idx < _ACT_YOP:
        return Action(ActionType.PLAY_MONOPOLY, receive=_RESOURCES[idx - _ACT_MONOPOLY])
    if _ACT_YOP <= idx < _ACT_ROAD_BUILDING:
        r1, r2 = _yop_sub_inv(idx - _ACT_YOP)
        return Action(ActionType.PLAY_YEAR_OF_PLENTY, give=r1, receive=r2)
    if idx == _ACT_ROAD_BUILDING:
        return Action(ActionType.PLAY_ROAD_BUILDING)
    raise ValueError(f"Unknown action index {idx} (valid range 0..{ACTION_SIZE-1})")


def legal_action_mask(state: GameState, engine: GameEngine) -> np.ndarray:
    """
    Return a boolean array of shape (ACTION_SIZE,) where True = legal action.

    DISCARD and PLAYER_TRADE are handled outside this mask and will never
    appear as True here.
    """
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    for a in engine.legal_actions(state):
        try:
            mask[encode_action(a, state)] = True
        except ValueError:
            pass    # DISCARD / PLAYER_TRADE
    return mask


# ── Reward-shaping helpers ─────────────────────────────────────────────────────

# Expected number of pips produced per roll for each number token.
_PIP_TABLE: dict[int, int] = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}


def _vertex_pip_score(vertex_id: int, state: "GameState") -> int:
    """
    Sum of pip counts for all non-desert hexes adjacent to vertex_id.
    A corner touching 6-8-5 scores 5+5+4 = 14.
    """
    score = 0
    for hidx in state.topology.vertex_hexes[vertex_id]:
        token = state.board.hexes[hidx].token
        if token:
            score += _PIP_TABLE.get(token, 0)
    return score


def _vertex_resource_diversity(vertex_id: int, state: "GameState") -> int:
    """Number of distinct resources produceable from hexes adjacent to vertex_id."""
    resources: set = set()
    for hidx in state.topology.vertex_hexes[vertex_id]:
        res = HEX_RESOURCE.get(state.board.hexes[hidx].hex_type)
        if res is not None:
            resources.add(res)
    return len(resources)


def _reachable_buildable_count(state: "GameState", pid: int) -> int:
    """
    Count empty, buildable vertices reachable by pid along their road network.

    A vertex is buildable if it is unoccupied and no adjacent vertex is occupied
    (standard Catan distance rule).
    """
    topo = state.topology
    owned = {v for v in range(topo.num_vertices) if state.vertex_owner[v] == pid}

    # BFS along pid's edges from pid's settlements/cities.
    reachable: set[int] = set(owned)
    frontier: list[int] = list(owned)
    while frontier:
        v = frontier.pop()
        for eid in topo.vertex_edges[v]:
            if state.edge_owner[eid] == pid:
                v1, v2 = topo.edge_vertices[eid]
                nv = v2 if v1 == v else v1
                if nv not in reachable:
                    reachable.add(nv)
                    frontier.append(nv)

    def buildable(v: int) -> bool:
        return (
            state.vertex_owner[v] == -1
            and all(state.vertex_owner[nv] == -1 for nv in topo.vertex_neighbors[v])
        )

    return sum(1 for v in reachable if v not in owned and buildable(v))


# ── Greedy discard helper ───────────────────────────────────────────────────────

def _auto_discard(state: GameState, pid: int) -> Action:
    """
    Build a DISCARD action using a greedy heuristic: discard the most-
    abundant resources first until exactly half the hand (rounded down)
    has been removed.  This removes strategic discard choices from the
    agent's action space.
    """
    hand   = {r: state.players[pid].resources.get(r, 0) for r in Resource}
    total  = sum(hand.values())
    needed = total // 2
    result: dict[Resource, int] = {}
    for r in sorted(hand, key=lambda r: hand[r], reverse=True):
        drop = min(hand[r], needed)
        if drop:
            result[r] = drop
            needed   -= drop
        if needed == 0:
            break
    return Action(ActionType.DISCARD, discard=result if result else {})


# ── CatanEnv ────────────────────────────────────────────────────────────────────

class CatanEnv:
    """
    Multi-agent Catan environment.

    Typical self-play loop
    ----------------------
        env = CatanEnv(num_players=4)
        obs, mask = env.reset()

        while not env.done:
            obs, mask = env.observe()          # current player's perspective
            action_idx = my_agent.choose(obs, mask)
            rewards, done = env.step(action_idx)

        print("Winner:", env.winner)

    Notes
    -----
    * Observations are always from the perspective of env.current_player.
    * DISCARD is resolved automatically (greedy heuristic) inside step().
    * step() returns a reward dict keyed by absolute player id.
      By default only the winner gets +1 at game end (sparse).
      Pass reward_shaping=True in the constructor for per-step VP-delta rewards.
    """

    def __init__(
        self,
        num_players:    int  = 4,
        randomize_board: bool = True,
        reward_shaping:  bool = False,
        public_vp_reward: float = 0.3,
        road_reward: float = 0.05,
        buy_dev_reward: float = 0.10,
        win_reward: float = 5.0,
        loss_penalty: float = 5.0,
        setup_settle_reward: float = 0.5,
        robber_block_reward: float = 0.1,
        monopoly_reward: float = 0.3,
        yop_build_reward: float = 0.15,
    ) -> None:
        if not (2 <= num_players <= 4):
            raise ValueError("num_players must be 2–4")
        self.num_players     = num_players
        self.randomize_board = randomize_board
        self.reward_shaping  = reward_shaping
        self.public_vp_reward = public_vp_reward
        self.road_reward      = road_reward
        self.buy_dev_reward   = buy_dev_reward
        self.win_reward           = win_reward
        self.loss_penalty         = loss_penalty
        self.setup_settle_reward  = setup_settle_reward
        self.robber_block_reward  = robber_block_reward
        self.monopoly_reward      = monopoly_reward
        self.yop_build_reward     = yop_build_reward
        self._engine              = GameEngine()
        self._state: Optional[GameState] = None
        self._prev_vp: list[int] = []

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def current_player(self) -> int:
        """Absolute index of the player whose turn it currently is."""
        return self._state.current_player if self._state else 0

    @property
    def done(self) -> bool:
        return self._state is not None and self._state.phase == Phase.DONE

    @property
    def winner(self) -> Optional[int]:
        return self._state.winner if self._state else None

    # ── Core API ──────────────────────────────────────────────────────────────

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """Start a new game. Returns (obs, legal_mask) for the first player."""
        self._state    = self._engine.new_game(
            num_players=self.num_players,
            randomize_board=self.randomize_board,
        )
        self._prev_vp  = [
            self._engine.compute_public_vp(self._state, i)
            for i in range(self.num_players)
        ]
        return self.observe()

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (observation, legal_action_mask) for the current player.

        During DISCARD the mask is all-False (handled automatically in step).
        """
        state = self._state
        pid   = state.current_player
        obs   = encode_obs(state, pid, self._engine)
        if state.phase == Phase.DISCARD:
            mask = np.zeros(ACTION_SIZE, dtype=bool)
        else:
            mask = legal_action_mask(state, self._engine)
        return obs, mask

    def step(
        self, action_idx: int
    ) -> tuple[dict[int, float], bool]:
        """
        Apply the chosen action.

        Parameters
        ----------
        action_idx : integer index from the action space (0..ACTION_SIZE-1).
                     Ignored when the current phase is DISCARD.

        Returns
        -------
        rewards : dict[player_id → float]
            With reward_shaping=True, players get configured shaping rewards
            during the main game.  Terminal win/loss rewards are applied when
            the game ends.
        done : bool
        """
        state   = self._state
        rewards = {i: 0.0 for i in range(self.num_players)}

        # Execute the chosen action
        if state.phase == Phase.DISCARD:
            action = _auto_discard(state, state.current_player)
        else:
            action = decode_action(action_idx, state)

        acting_pid = state.current_player
        old_public_vp = [
            self._engine.compute_public_vp(state, i)
            for i in range(self.num_players)
        ]
        old_dev_total = (
            len(state.players[acting_pid].dev_cards)
            + len(state.players[acting_pid].dev_cards_new)
        )
        new_state, _reward, done = self._engine.step(state, action)
        self._state = new_state

        # Auto-resolve all pending discards (there may be multiple players)
        while not done and self._state.phase == Phase.DISCARD:
            pid     = self._state.current_player
            d_act   = _auto_discard(self._state, pid)
            self._state, _reward, done = self._engine.step(self._state, d_act)

        new_public_vp = [
            self._engine.compute_public_vp(self._state, i)
            for i in range(self.num_players)
        ]
        new_dev_total = (
            len(self._state.players[acting_pid].dev_cards)
            + len(self._state.players[acting_pid].dev_cards_new)
        )

        if self.reward_shaping:
            # Setup settlement placement: reward proportional to pip productivity
            # and resource diversity of the chosen vertex.  Setup roads are free
            # forced moves, so we don't reward them.
            if state.phase == Phase.SETUP and action.type == ActionType.PLACE_SETTLEMENT:
                pip = _vertex_pip_score(action.vertex_id, state)
                div = _vertex_resource_diversity(action.vertex_id, state)
                # Normalise: 15 pips is the theoretical ceiling; 3 = full diversity.
                rewards[acting_pid] += self.setup_settle_reward * (pip / 15.0 + div / 3.0) / 2.0

            if state.phase != Phase.SETUP:
                for i in range(self.num_players):
                    vp_delta = new_public_vp[i] - old_public_vp[i]
                    rewards[i] += self.public_vp_reward * vp_delta

                if action.type == ActionType.PLACE_ROAD:
                    # Reward roads that open new reachable buildable vertices;
                    # ignore roads that lead only to occupied or already-reachable spots.
                    old_reach = _reachable_buildable_count(state, acting_pid)
                    new_reach = _reachable_buildable_count(self._state, acting_pid)
                    if new_reach > old_reach:
                        rewards[acting_pid] += self.road_reward

                if action.type == ActionType.BUY_DEV_CARD and new_dev_total > old_dev_total:
                    rewards[acting_pid] += self.buy_dev_reward

                # ── Dev card play rewards ──────────────────────────────────
                if action.type == ActionType.MOVE_ROBBER:
                    # Reward placing the robber on a productive hex that hurts an opponent.
                    # Applies whether triggered by a 7 or a Knight card.
                    target_hex = action.hex_id
                    has_opponent = any(
                        state.vertex_owner[v] not in (-1, acting_pid)
                        for v in state.topology.hex_vertices[target_hex]
                    )
                    pip = sum(
                        _PIP_TABLE.get(state.board.hexes[target_hex].token or 0, 0)
                        for _ in [1]   # single-element loop for expression
                    )
                    if has_opponent and pip >= 3:
                        rewards[acting_pid] += self.robber_block_reward * (pip / 5.0)

                if action.type == ActionType.PLAY_MONOPOLY:
                    # Reward proportional to how many cards were stolen.
                    resource = action.receive
                    cards_before = sum(
                        state.players[i].resources.get(resource, 0)
                        for i in range(self.num_players) if i != acting_pid
                    )
                    cards_after = sum(
                        self._state.players[i].resources.get(resource, 0)
                        for i in range(self.num_players) if i != acting_pid
                    )
                    stolen = cards_before - cards_after
                    if stolen > 0:
                        rewards[acting_pid] += self.monopoly_reward * min(stolen / 8.0, 1.0)

                if action.type == ActionType.PLAY_YEAR_OF_PLENTY:
                    # Reward if the gained resources come from outside normal production
                    # AND now enable the player to afford something they couldn't before.
                    r1, r2 = action.give, action.receive
                    produces: set[Resource] = set()
                    for v in range(state.topology.num_vertices):
                        if state.vertex_owner[v] == acting_pid:
                            for hidx in state.topology.vertex_hexes[v]:
                                res = HEX_RESOURCE[state.board.hexes[hidx].hex_type]
                                if res is not None:
                                    produces.add(res)
                    gained_outside = r1 not in produces or r2 not in produces
                    p_new = self._state.players[acting_pid]
                    can_build = (
                        p_new.can_afford(self._engine.BUILD_COSTS["road"]) or
                        p_new.can_afford(self._engine.BUILD_COSTS["settlement"]) or
                        p_new.can_afford(self._engine.BUILD_COSTS["city"])
                    )
                    if can_build and gained_outside:
                        rewards[acting_pid] += self.yop_build_reward

        if done and self._state.winner is not None:
            winner = self._state.winner
            rewards[winner] += self.win_reward
            for pid in range(self.num_players):
                if pid != winner:
                    rewards[pid] -= self.loss_penalty

        self._prev_vp = new_public_vp

        return rewards, done

    # ── Info helpers ─────────────────────────────────────────────────────────

    def obs_size(self) -> int:
        """Total length of the observation vector."""
        return _obs_size(self.num_players)

    def action_size(self) -> int:
        """Total number of possible action indices."""
        return ACTION_SIZE

    def state(self) -> Optional[GameState]:
        """Direct access to the raw GameState (useful for MCTS / debugging)."""
        return self._state

    def scoreboard(self) -> dict[int, int]:
        """Public VP for every player (excludes secret VP dev cards)."""
        if self._state is None:
            return {}
        return {
            i: self._engine.compute_public_vp(self._state, i)
            for i in range(self.num_players)
        }


# ── Agents ───────────────────────────────────────────────────────────────────────

class RandomAgent:
    """
    Picks uniformly at random from all legal actions.
    Useful as a baseline and for smoke-testing the environment.
    """

    def choose(self, obs: np.ndarray, mask: np.ndarray) -> int:
        legal = np.where(mask)[0]
        if len(legal) == 0:
            return 0    # should only happen during auto-DISCARD steps
        return int(random.choice(legal))


class GreedyVPAgent:
    """
    One-step lookahead: always picks the action that maximises the immediate
    public VP gain.  Falls back to random when no action changes VP.

    Not a strong player but significantly beats RandomAgent.
    """

    def __init__(self, env: CatanEnv) -> None:
        self._env = env

    def choose(self, obs: np.ndarray, mask: np.ndarray) -> int:
        state  = self._env.state()
        engine = self._env._engine
        pid    = state.current_player
        legal  = np.where(mask)[0]
        if len(legal) == 0:
            return 0

        best_idx   = int(legal[0])
        best_vp    = -1

        for idx in legal:
            action    = decode_action(int(idx), state)
            new_state, _, _ = engine.step(state, action)
            vp = engine.compute_vp(new_state, pid)
            if vp > best_vp:
                best_vp  = vp
                best_idx = int(idx)

        return best_idx


# ── Self-play runner ──────────────────────────────────────────────────────────────

def run_episode(
    env:     CatanEnv,
    agents:  list,           # list of objects with .choose(obs, mask) -> int
    verbose: bool = False,
    max_steps: int = 5000,
) -> tuple[Optional[int], list[dict]]:
    """
    Run one complete game episode.

    Parameters
    ----------
    env     : a CatanEnv instance (will be reset at the start)
    agents  : one agent per player; each must have .choose(obs, mask) -> int
    verbose : print a brief log each step
    max_steps : safety limit to prevent infinite games

    Returns
    -------
    winner  : winning player index, or None if max_steps reached
    history : list of step dicts with keys (player, action_idx, rewards, done)
    """
    assert len(agents) == env.num_players, \
        f"Need {env.num_players} agents, got {len(agents)}"

    obs, mask = env.reset()
    history: list[dict] = []
    steps = 0

    while not env.done and steps < max_steps:
        pid        = env.current_player
        action_idx = agents[pid].choose(obs, mask)

        if verbose:
            state = env.state()
            print(
                f"Step {steps:4d} | Player {pid} | "
                f"Phase {state.phase.value:<8} | "
                f"Action {action_idx:3d} ({decode_action(action_idx, state)!r})"
            )

        rewards, done = env.step(action_idx)
        history.append({"player": pid, "action": action_idx,
                         "rewards": rewards, "done": done})
        steps += 1

        if not done:
            obs, mask = env.observe()

    if verbose:
        if env.winner is not None:
            print(f"\nGame over in {steps} steps — Player {env.winner} wins!")
            print("Scores:", env.scoreboard())
        else:
            print(f"\nGame truncated after {max_steps} steps.")

    return env.winner, history


# ── CLI smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("Running 10 self-play games with 4 RandomAgents …")
    wins    = [0] * 4
    lengths = []
    t0      = time.time()

    for game in range(10):
        env    = CatanEnv(num_players=4, reward_shaping=False)
        agents = [RandomAgent() for _ in range(4)]
        winner, history = run_episode(env, agents, verbose=False)
        if winner is not None:
            wins[winner] += 1
        lengths.append(len(history))
        print(f"  Game {game+1:2d}: {len(history):5d} steps, winner = {winner}")

    elapsed = time.time() - t0
    print(f"\nWin counts: {wins}")
    print(f"Avg steps : {sum(lengths) / len(lengths):.0f}")
    print(f"Wall time : {elapsed:.2f}s  ({elapsed/10:.2f}s/game)")
    print(f"Obs size  : {CatanEnv(4).obs_size()}  floats")
    print(f"Act size  : {ACTION_SIZE}")
