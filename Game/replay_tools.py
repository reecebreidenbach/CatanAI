"""
Record and replay Catan episodes with full GameState snapshots.

Examples
--------
Record one policy-vs-random game:
    python replay_tools.py record --policy phase1_policy.pt --output replay.pkl

Record with a random learner seat and greedy opponents:
    python replay_tools.py record --policy phase1_policy.pt --learner-seat random --opponent greedy --output replay.pkl

Record current policy against a random frozen league snapshot:
    python replay_tools.py record --league-pool league_pool.pt --policies phase3_policy.pt league random random --output replay.pkl

Record the latest frozen snapshot currently saved in the league pool:
    python replay_tools.py record --league-pool league_pool.pt --policies league_latest random random random --output replay.pkl

Record the latest snapshot against three random league-pool opponents:
    python replay_tools.py record --league-pool league_pool.pt --policies league_latest --fill-missing league --output replay.pkl

View a saved replay:
    python replay_tools.py view replay.pkl
"""

from __future__ import annotations

import argparse
import pickle
import random
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch

from board_builder import BoardBuilderApp, PLAYER_COLORS
from catan_env import CatanEnv, GreedyVPAgent, RandomAgent, decode_action
from game_engine import GameEngine
from game_state import DevCard
from policy import CatanPolicy, masked_sample
from ppo_utils import HIDDEN_SIZE


@dataclass
class ReplayEvent:
    step: int
    player: int
    action_idx: int
    action_repr: str
    rewards: dict[int, float]
    done: bool


@dataclass
class ReplayData:
    states: list
    events: list[ReplayEvent]
    winner: Optional[int]
    num_players: int
    metadata: dict[str, object]


class _ReplayUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "__main__" and name == "ReplayData":
            return ReplayData
        if module == "__main__" and name == "ReplayEvent":
            return ReplayEvent
        return super().find_class(module, name)


class PolicyAgent:
    """Policy-backed agent for replay capture."""

    def __init__(self, policy_path: str | Path, env: CatanEnv) -> None:
        self._policy = CatanPolicy(
            obs_size=env.obs_size(),
            action_size=env.action_size(),
            hidden=HIDDEN_SIZE,
        )
        self._policy.load_state_dict(torch.load(policy_path, map_location="cpu"))
        self._policy.eval()

    def choose(self, obs: np.ndarray, mask: np.ndarray) -> int:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        mask_t = torch.tensor(mask, dtype=torch.bool)
        with torch.no_grad():
            logits, _ = self._policy(obs_t)
        return masked_sample(logits, mask_t)


class SnapshotPolicyAgent:
    """Policy-backed agent created from an in-memory state_dict snapshot."""

    def __init__(self, state_dict: dict, env: CatanEnv) -> None:
        self._policy = CatanPolicy(
            obs_size=env.obs_size(),
            action_size=env.action_size(),
            hidden=HIDDEN_SIZE,
        )
        self._policy.load_state_dict(state_dict)
        self._policy.eval()

    def choose(self, obs: np.ndarray, mask: np.ndarray) -> int:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        mask_t = torch.tensor(mask, dtype=torch.bool)
        with torch.no_grad():
            logits, _ = self._policy(obs_t)
        return masked_sample(logits, mask_t)


_LEAGUE_TOKEN = "__LEAGUE_RANDOM__"
_LEAGUE_LATEST_TOKEN = "__LEAGUE_LATEST__"


def record_episode_replay(
    policy_path: str | Path | None,
    output_path: str | Path,
    *,
    learner_seat: int | Literal["random"] = 0,
    opponent: Literal["random", "greedy"] = "random",
    num_players: int = 4,
    max_steps: int = 5000,
    reward_shaping: bool = False,
    per_seat_policies: list[str | Path | None] | None = None,
    league_pool_path: str | Path | None = None,
    fill_missing: Literal["random", "league"] = "random",
) -> Path:
    env = CatanEnv(num_players=num_players, reward_shaping=reward_shaping)

    league_pool: list[dict] = []
    if league_pool_path is not None:
        league_pool = torch.load(league_pool_path, map_location="cpu", weights_only=False)

    if per_seat_policies is not None:
        # Multi-policy mode: each seat gets its own agent.
        # None entries fall back to RandomAgent.
        # If fewer paths than seats are given, remaining seats are random.
        agents = []
        chosen_league_indices: list[int | None] = []
        for path in per_seat_policies:
            if path is None:
                agents.append(RandomAgent())
                chosen_league_indices.append(None)
            elif path == _LEAGUE_TOKEN:
                if not league_pool:
                    raise ValueError("--policies includes 'league' but no --league-pool was provided")
                snap_idx = random.randrange(len(league_pool))
                agents.append(SnapshotPolicyAgent(league_pool[snap_idx], env))
                chosen_league_indices.append(snap_idx)
            elif path == _LEAGUE_LATEST_TOKEN:
                if not league_pool:
                    raise ValueError("--policies includes 'league_latest' but no --league-pool was provided")
                snap_idx = len(league_pool) - 1
                agents.append(SnapshotPolicyAgent(league_pool[snap_idx], env))
                chosen_league_indices.append(snap_idx)
            else:
                agents.append(PolicyAgent(path, env))
                chosen_league_indices.append(None)
        while len(agents) < num_players:
            if fill_missing == "league":
                if not league_pool:
                    raise ValueError("--fill-missing league requires a non-empty --league-pool")
                snap_idx = random.randrange(len(league_pool))
                agents.append(SnapshotPolicyAgent(league_pool[snap_idx], env))
                chosen_league_indices.append(snap_idx)
            else:
                agents.append(RandomAgent())
                chosen_league_indices.append(None)
        chosen_seat = 0  # metadata only — all seats are policy-controlled
        opp_label = "multi-policy"
    else:
        chosen_seat = random.randrange(num_players) if learner_seat == "random" else learner_seat
        policy_agent = PolicyAgent(policy_path, env)
        if opponent == "greedy":
            opponents = [GreedyVPAgent(env) for _ in range(num_players)]
        else:
            opponents = [RandomAgent() for _ in range(num_players)]
        agents = opponents[:]
        agents[chosen_seat] = policy_agent
        opp_label = opponent
        chosen_league_indices = [None] * num_players

    obs, mask = env.reset()
    states = [env.state().copy()]
    events: list[ReplayEvent] = []
    steps = 0

    while not env.done and steps < max_steps:
        pid = env.current_player
        action_idx = agents[pid].choose(obs, mask)
        action_repr = repr(decode_action(action_idx, env.state()))
        rewards, done = env.step(action_idx)
        events.append(
            ReplayEvent(
                step=steps,
                player=pid,
                action_idx=action_idx,
                action_repr=action_repr,
                rewards=dict(rewards),
                done=done,
            )
        )
        states.append(env.state().copy())
        steps += 1
        if not done:
            obs, mask = env.observe()

    replay = ReplayData(
        states=states,
        events=events,
        winner=env.winner,
        num_players=num_players,
        metadata={
            "policy_path": str(policy_path) if policy_path else None,
            "per_seat_policies": [str(p) if p else None for p in (per_seat_policies or [])],
            "league_pool_path": str(league_pool_path) if league_pool_path else None,
            "league_snapshot_indices": chosen_league_indices,
            "fill_missing": fill_missing,
            "learner_seat": chosen_seat,
            "opponent": opp_label,
            "max_steps": max_steps,
            "reward_shaping": reward_shaping,
        },
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(replay, handle)
    return output


def load_replay(path: str | Path) -> ReplayData:
    with Path(path).open("rb") as handle:
        return _ReplayUnpickler(handle).load()


class ReplayViewerApp(BoardBuilderApp):
    def __init__(self, replay: ReplayData) -> None:
        self._replay = replay
        self._replay_index = 0
        self._autoplay_job: Optional[str] = None
        super().__init__()
        self._event_var = tk.StringVar(master=self, value="")
        self.title("Catan Replay Viewer")
        self._build_replay_controls()
        self._build_stats_panel()
        self.bind("<Left>", lambda _event: self._prev_step())
        self.bind("<Right>", lambda _event: self._next_step())
        self._load_snapshot(0)

    def _build_replay_controls(self) -> None:
        controls = tk.Frame(self, bg="#efe6d3", padx=10, pady=8)
        controls.pack(side=tk.BOTTOM, fill=tk.X)

        tk.Button(controls, text="|< Start", command=lambda: self._load_snapshot(0)).pack(side=tk.LEFT, padx=4)
        tk.Button(controls, text="< Prev", command=self._prev_step).pack(side=tk.LEFT, padx=4)
        tk.Button(controls, text="Next >", command=self._next_step).pack(side=tk.LEFT, padx=4)
        tk.Button(controls, text=">| End", command=lambda: self._load_snapshot(len(self._replay.states) - 1)).pack(side=tk.LEFT, padx=4)
        tk.Button(controls, text="Play", command=self._toggle_autoplay).pack(side=tk.LEFT, padx=10)

        meta = self._replay.metadata
        summary = (
            f"learner seat: {meta.get('learner_seat')}  |  opponent: {meta.get('opponent')}  |  "
            f"winner: {self._replay.winner}"
        )
        tk.Label(controls, text=summary, bg="#efe6d3", fg="#3b342c", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=16)
        tk.Label(controls, textvariable=self._event_var, bg="#efe6d3", fg="#3b342c", font=("Arial", 10), anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

    def _build_stats_panel(self) -> None:
        """Build a persistent per-player stats bar above the controls."""
        n = self._replay.num_players
        panel = tk.Frame(self, bg="#2a2a2a", padx=6, pady=4)
        panel.pack(side=tk.BOTTOM, fill=tk.X)

        self._stat_vars: list[dict[str, tk.StringVar]] = []
        for pid in range(n):
            color_info = PLAYER_COLORS[pid]
            col = tk.Frame(panel, bg=color_info["fill"], bd=1, relief=tk.RAISED, padx=6, pady=3)
            col.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=3)

            tk.Label(col, text=color_info["name"], font=("Arial", 9, "bold"),
                     bg=color_info["fill"], fg=color_info["text"]).pack(anchor="w")

            svars: dict[str, tk.StringVar] = {}
            for key, label in [
                ("vp",       "VP (est.)"),
                ("knights",  "Knights ⚔"),
                ("hand",     "K in hand"),
                ("settle",   "Settlements"),
                ("cities",   "Cities"),
                ("roads",    "Roads"),
                ("army",     "Largest Army"),
            ]:
                row = tk.Frame(col, bg=color_info["fill"])
                row.pack(fill=tk.X)
                tk.Label(row, text=f"{label}:", font=("Arial", 8),
                         bg=color_info["fill"], fg=color_info["text"],
                         width=13, anchor="w").pack(side=tk.LEFT)
                sv = tk.StringVar(value="—")
                svars[key] = sv
                tk.Label(row, textvariable=sv, font=("Arial", 8, "bold"),
                         bg=color_info["fill"], fg=color_info["text"],
                         anchor="w").pack(side=tk.LEFT)
            self._stat_vars.append(svars)

    def _update_stats_panel(self) -> None:
        """Refresh per-player knight/VP stats from the current snapshot."""
        if not hasattr(self, "_stat_vars"):
            return
        state = self._game_state
        if state is None:
            return
        n = len(state.players)
        for pid in range(min(n, len(self._stat_vars))):
            p = state.players[pid]
            svars = self._stat_vars[pid]

            knights_played = p.knights_played
            # Knights in hand = playable knights (in dev_cards, not dev_cards_new)
            knights_in_hand = p.dev_cards.count(DevCard.KNIGHT)

            built_settle = 5 - p.settlements_left
            built_cities = 4 - p.cities_left
            built_roads  = 15 - p.roads_left

            # Estimate VP: settlements + 2*cities + largest army (2) + longest road (2)
            vp = built_settle + 2 * built_cities
            if state.largest_army_owner == pid:
                vp += 2
            if state.longest_road_owner == pid:
                vp += 2
            # VP dev cards in hand
            vp += p.dev_cards.count(DevCard.VICTORY_POINT)

            svars["vp"].set(str(vp))
            svars["knights"].set(str(knights_played))
            svars["hand"].set(str(knights_in_hand))
            svars["settle"].set(str(built_settle))
            svars["cities"].set(str(built_cities))
            svars["roads"].set(str(built_roads))
            svars["army"].set("YES" if state.largest_army_owner == pid else "no")

    def _load_snapshot(self, index: int) -> None:
        index = max(0, min(index, len(self._replay.states) - 1))
        self._replay_index = index
        self._game_engine = GameEngine()
        self._game_state = self._replay.states[index].copy()
        self.board = self._game_state.board
        self.settlements.clear()
        self.cities.clear()
        self.roads.clear()
        self._draw_board()
        self._sync_pieces_from_state()
        self._update_status()
        self._update_event_label()
        self._update_stats_panel()

    def _update_event_label(self) -> None:
        if self._replay_index == 0:
            self._event_var.set(f"State 0/{len(self._replay.states) - 1} | initial position")
            return
        event = self._replay.events[self._replay_index - 1]
        reward_str = ", ".join(f"p{pid}:{value:.2f}" for pid, value in sorted(event.rewards.items()))
        self._event_var.set(
            f"State {self._replay_index}/{len(self._replay.states) - 1} | "
            f"step {event.step} | p{event.player} | {event.action_repr} | rewards [{reward_str}]"
        )

    def _prev_step(self) -> None:
        self._stop_autoplay()
        self._load_snapshot(self._replay_index - 1)

    def _next_step(self) -> None:
        self._stop_autoplay()
        self._load_snapshot(self._replay_index + 1)

    def _toggle_autoplay(self) -> None:
        if self._autoplay_job is not None:
            self._stop_autoplay()
            return
        self._autoplay_step()

    def _autoplay_step(self) -> None:
        if self._replay_index >= len(self._replay.states) - 1:
            self._stop_autoplay()
            return
        self._load_snapshot(self._replay_index + 1)
        self._autoplay_job = self.after(700, self._autoplay_step)

    def _stop_autoplay(self) -> None:
        if self._autoplay_job is not None:
            self.after_cancel(self._autoplay_job)
            self._autoplay_job = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record and replay Catan games.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser("record", help="Record one episode to a replay file.")
    record_parser.add_argument("--policy", default=None, help="Path to a .pt policy checkpoint (single learner seat).")
    record_parser.add_argument("--policies", nargs="+", default=None,
                             help="Per-seat policy paths, one per seat (use 'random' for a random agent, 'league' for a random snapshot from --league-pool, or 'league_latest' for the newest snapshot). "
                                 "Overrides --policy. Example: --policies league_latest league random p3.pt")
    record_parser.add_argument("--league-pool", default=None,
                               help="Path to league_pool.pt when using 'league' entries in --policies.")
    record_parser.add_argument("--fill-missing", choices=["random", "league"], default="random",
                               help="How to fill any seats not specified by --policies: plain random agents or random snapshots from --league-pool.")
    record_parser.add_argument("--output", default="replays/replay.pkl", help="Replay output path.")
    record_parser.add_argument("--learner-seat", default="0", help="Seat index 0-3 or 'random' (single-policy mode only).")
    record_parser.add_argument("--opponent", choices=["random", "greedy"], default="random")
    record_parser.add_argument("--num-players", type=int, default=4)
    record_parser.add_argument("--max-steps", type=int, default=5000)
    record_parser.add_argument("--reward-shaping", action="store_true")

    view_parser = subparsers.add_parser("view", help="Open the replay viewer for a saved replay.")
    view_parser.add_argument("replay", help="Path to a saved replay .pkl file.")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "record":
        learner_seat: int | Literal["random"]
        learner_seat = "random" if args.learner_seat == "random" else int(args.learner_seat)

        per_seat: list[str | Path | None] | None = None
        if args.policies is not None:
            per_seat = [
                None if p.lower() == "random"
                else _LEAGUE_TOKEN if p.lower() == "league"
                else _LEAGUE_LATEST_TOKEN if p.lower() == "league_latest"
                else p
                for p in args.policies
            ]

        output = record_episode_replay(
            policy_path=args.policy,
            output_path=args.output,
            learner_seat=learner_seat,
            opponent=args.opponent,
            num_players=args.num_players,
            max_steps=args.max_steps,
            reward_shaping=args.reward_shaping,
            per_seat_policies=per_seat,
            league_pool_path=args.league_pool,
            fill_missing=args.fill_missing,
        )
        print(f"Saved replay to {output}")
        return

    replay = load_replay(args.replay)
    app = ReplayViewerApp(replay)
    app.mainloop()


if __name__ == "__main__":
    main()