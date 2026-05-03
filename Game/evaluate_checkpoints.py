"""Evaluate saved policy checkpoints against mixed opponents and report behavior metrics."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import torch

from catan_env import CatanEnv, GreedyVPAgent, RandomAgent, _opening_strategy_bias, decode_action
from game_engine import ActionType
from game_state import DevCard
from policy import CatanPolicy, masked_sample
from ppo_utils import CKPT_PHASE2, CKPT_PHASE3, HIDDEN_SIZE


class PolicyAgent:
    def __init__(self, policy_path: str | Path, env: CatanEnv) -> None:
        self._policy = CatanPolicy(
            obs_size=env.obs_size(),
            action_size=env.action_size(),
            hidden=HIDDEN_SIZE,
        )
        self._policy.load_state_dict(torch.load(policy_path, map_location="cpu", weights_only=True))
        self._policy.eval()

    def choose(self, obs, mask) -> int:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        mask_t = torch.tensor(mask, dtype=torch.bool)
        with torch.no_grad():
            logits, _ = self._policy(obs_t)
        return masked_sample(logits, mask_t)


def _latest_checkpoint_paths() -> list[Path]:
    game_dir = Path(__file__).resolve().parent
    found: list[Path] = []
    for base in (CKPT_PHASE2, CKPT_PHASE3):
        stem = Path(base).stem
        suffix = Path(base).suffix or ".pt"
        matches = sorted(game_dir.glob(f"{stem}*{suffix}"), key=lambda p: p.stat().st_mtime)
        if matches:
            found.append(matches[-1])
    return found


def _available_checkpoints() -> list[Path]:
    game_dir = Path(__file__).resolve().parent
    return sorted(game_dir.glob("phase*_policy*.pt"), key=lambda p: p.stat().st_mtime)


def _resolve_checkpoint_paths(checkpoints: list[Path]) -> list[Path]:
    game_dir = Path(__file__).resolve().parent
    resolved: list[Path] = []
    missing: list[Path] = []

    for checkpoint in checkpoints:
        path = checkpoint if checkpoint.is_absolute() else game_dir / checkpoint
        if path.exists():
            resolved.append(path)
        else:
            missing.append(checkpoint)

    if missing:
        available = _available_checkpoints()
        available_text = "\n".join(f"  - {path.name}" for path in available) or "  (none found)"
        missing_text = ", ".join(str(path) for path in missing)
        raise SystemExit(
            "Checkpoint file(s) not found: "
            f"{missing_text}\n"
            "Available checkpoints in Game/:\n"
            f"{available_text}"
        )

    return resolved


def _total_vp(state, pid: int) -> int:
    vp = 0
    for owner, building in zip(state.vertex_owner, state.vertex_building):
        if owner == pid:
            vp += 1 if building == 1 else 2 if building == 2 else 0
    if state.longest_road_owner == pid:
        vp += 2
    if state.largest_army_owner == pid:
        vp += 2
    vp += sum(1 for c in state.players[pid].dev_cards + state.players[pid].dev_cards_new if c == DevCard.VICTORY_POINT)
    return vp


def _owned_vertex_count(state, pid: int) -> int:
    return sum(1 for owner, building in zip(state.vertex_owner, state.vertex_building) if owner == pid and building in (1, 2))


def _opening_bucket(state, pid: int) -> tuple[str, float]:
    bias = _opening_strategy_bias(pid, state)
    label = "road" if bias >= 0.0 else "dev"
    return label, bias


def _mean(values: list[int | float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _share(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _format_percent(value: float) -> str:
    return f"{100.0 * value:5.1f}%"


def _format_counter(counter: dict[object, int]) -> str:
    if not counter:
        return "none"
    parts = [f"P{key}:{value}" for key, value in sorted(counter.items())]
    return "  ".join(parts)


def _format_action_mix(counter: dict[str, int]) -> str:
    if not counter:
        return "none"
    parts = [f"{name.lower()}={count}" for name, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]
    return ", ".join(parts)


def _bucket_summary(bucket_games: list[dict[str, float | int]]) -> str:
    if not bucket_games:
        return "n=0"

    road_total = sum(int(game["early_roads"]) for game in bucket_games)
    dev_total = sum(int(game["early_dev_buys"]) for game in bucket_games)
    third_turns = [float(game["third_turn"]) for game in bucket_games if game["third_turn"] > 0]
    by_100 = sum(int(game["third_by_100"]) for game in bucket_games)
    road_share = _share(road_total, road_total + dev_total)
    return (
        f"n={len(bucket_games)}  avg_bias={_mean([float(game['opening_bias']) for game in bucket_games]):+.2f}  "
        f"early_road_share={_format_percent(road_share)}  "
        f"avg_early_roads={_mean([int(game['early_roads']) for game in bucket_games]):.2f}  "
        f"avg_early_dev_buys={_mean([int(game['early_dev_buys']) for game in bucket_games]):.2f}  "
        f"3rd_settle_by_t100={_format_percent(_share(by_100, len(bucket_games)))}  "
        f"avg_3rd_settle_turn={_mean(third_turns):.1f}"
    )


def _print_result(result: dict[str, object]) -> None:
    print(result["policy"])
    print(
        "  Outcome  "
        f"win_rate={_format_percent(result['win_rate'])}  "
        f"avg_vp={result['avg_vp']:.2f}  "
        f"games={result['games']}"
    )
    print(
        "  Board    "
        f"avg_roads={result['avg_roads']:.2f}  "
        f"avg_cities={result['avg_cities']:.2f}  "
        f"avg_maritime_trades={result['avg_maritime_trades']:.2f}"
    )
    print(
        "  Opening  "
        f"3rd_settle_rate={_format_percent(result['third_settlement_rate'])}  "
        f"3rd_settle_by_t100={_format_percent(result['third_settlement_by_100_rate'])}  "
        f"avg_3rd_settle_turn={result['avg_third_settlement_turn']:.1f}"
    )
    print(
        "  Opening  "
        f"avg_early_roads={result['avg_early_roads']:.2f}  "
        f"avg_early_dev_buys={result['avg_early_dev_buys']:.2f}  "
        f"early_road_share={_format_percent(result['early_road_share'])}"
    )
    print(f"  Winners  {_format_counter(result['winner_counts'])}")
    print(f"  Action   {_format_action_mix(result['action_mix'])}")
    print(f"  Opening Split (road-friendly)  {_bucket_summary(result['opening_buckets']['road'])}")
    print(f"  Opening Split (dev-friendly)   {_bucket_summary(result['opening_buckets']['dev'])}")
    print()


def evaluate_policy(policy_path: Path, games: int, seed: int) -> dict[str, object]:
    random.seed(seed)
    env = CatanEnv(num_players=4, reward_shaping=False)
    policy_agent = PolicyAgent(policy_path, env)
    greedy_agent = GreedyVPAgent(env)
    random_agent = RandomAgent()

    learner_wins = 0
    winner_counts = Counter()
    action_counts = Counter()
    roads_built: list[int] = []
    cities_built: list[int] = []
    maritime_trades: list[int] = []
    final_vp: list[int] = []
    third_settlement_turns: list[int] = []
    early_road_counts: list[int] = []
    early_dev_buy_counts: list[int] = []
    opening_buckets: dict[str, list[dict[str, float | int]]] = {"road": [], "dev": []}

    for _ in range(games):
        learner_pid = random.randrange(env.num_players)
        opponents = [greedy_agent, greedy_agent, random_agent, random_agent]
        random.shuffle(opponents)
        opponents[learner_pid] = policy_agent

        obs, mask = env.reset()
        steps = 0
        game_trade_count = 0
        game_early_roads = 0
        game_early_dev_buys = 0
        third_settlement_turn: int | None = None
        opening_bucket: str | None = None
        opening_bias = 0.0
        while not env.done and steps < 5000:
            pid = env.current_player
            action_idx = opponents[pid].choose(obs, mask)
            if pid == learner_pid:
                state = env.state()
                if opening_bucket is None and _owned_vertex_count(state, learner_pid) >= 2:
                    opening_bucket, opening_bias = _opening_bucket(state, learner_pid)
                action = decode_action(action_idx, state)
                action_counts[action.type.name] += 1
                if action.type == ActionType.MARITIME_TRADE:
                    game_trade_count += 1
                if third_settlement_turn is None:
                    if action.type == ActionType.PLACE_ROAD:
                        game_early_roads += 1
                    elif action.type == ActionType.BUY_DEV_CARD:
                        game_early_dev_buys += 1
                    if action.type == ActionType.PLACE_SETTLEMENT and _owned_vertex_count(state, learner_pid) == 2:
                        third_settlement_turn = state.turn_number
            _, done = env.step(action_idx)
            steps += 1
            if not done:
                obs, mask = env.observe()

        final_state = env.state()
        winner = env.winner
        winner_counts[winner] += 1
        learner_wins += int(winner == learner_pid)
        roads_built.append(sum(1 for owner in final_state.edge_owner if owner == learner_pid))
        cities_built.append(sum(1 for owner, building in zip(final_state.vertex_owner, final_state.vertex_building) if owner == learner_pid and building == 2))
        maritime_trades.append(game_trade_count)
        final_vp.append(_total_vp(final_state, learner_pid))
        early_road_counts.append(game_early_roads)
        early_dev_buy_counts.append(game_early_dev_buys)
        if third_settlement_turn is not None:
            third_settlement_turns.append(third_settlement_turn)

        if opening_bucket is None and _owned_vertex_count(final_state, learner_pid) >= 2:
            opening_bucket, opening_bias = _opening_bucket(final_state, learner_pid)
        opening_buckets[opening_bucket or "road"].append(
            {
                "opening_bias": opening_bias,
                "early_roads": game_early_roads,
                "early_dev_buys": game_early_dev_buys,
                "third_turn": third_settlement_turn or 0,
                "third_by_100": int(third_settlement_turn is not None and third_settlement_turn <= 100),
            }
        )

    total_early_roads = sum(early_road_counts)
    total_early_dev_buys = sum(early_dev_buy_counts)

    return {
        "policy": policy_path.name,
        "games": games,
        "win_rate": learner_wins / games,
        "winner_counts": dict(winner_counts),
        "avg_roads": sum(roads_built) / games,
        "avg_cities": sum(cities_built) / games,
        "avg_maritime_trades": sum(maritime_trades) / games,
        "avg_vp": sum(final_vp) / games,
        "action_mix": dict(action_counts),
        "third_settlement_rate": len(third_settlement_turns) / games,
        "third_settlement_by_100_rate": sum(1 for turn in third_settlement_turns if turn <= 100) / games,
        "avg_third_settlement_turn": _mean(third_settlement_turns),
        "avg_early_roads": _mean(early_road_counts),
        "avg_early_dev_buys": _mean(early_dev_buy_counts),
        "early_road_share": _share(total_early_roads, total_early_roads + total_early_dev_buys),
        "opening_buckets": opening_buckets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate policy checkpoints against mixed opponents.")
    parser.add_argument("--checkpoints", nargs="*", help="Checkpoint filenames or absolute paths to evaluate.")
    parser.add_argument("--games", type=int, default=20, help="Games per checkpoint.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    checkpoints = [Path(p) for p in args.checkpoints] if args.checkpoints else _latest_checkpoint_paths()
    if not checkpoints:
        raise SystemExit("No checkpoints found to evaluate.")

    for path in _resolve_checkpoint_paths(checkpoints):
        result = evaluate_policy(path, games=args.games, seed=args.seed)
        _print_result(result)


if __name__ == "__main__":
    main()