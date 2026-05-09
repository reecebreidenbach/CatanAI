# AI-ASSISTED
"""
analyze_human_replay.py — Full game breakdown for human vs AI replays.

Usage
-----
    python analyze_human_replay.py replays/human_vs_ai_20260507_143022.pkl
    python analyze_human_replay.py replays/human_vs_ai_20260507_143022.pkl --human-seat 0

Sections printed
----------------
  1. Game summary (winner, VP, turns)
  2. Setup quality — pip scores and diversity for every starting settlement
  3. Build timeline — turn-by-turn VP progression per player
  4. Road efficiency — roads built vs settlements/cities placed
  5. Resource economy — total resources collected, spent, wasted (discarded)
  6. Dev cards — what each player bought and played
  7. Trade activity — maritime and player trades per player
  8. Robber impact — times robbed / times placed robber on opponent hex
  9. Head-to-head summary — where the human beat the bots and where they lost
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_tools import ReplayData, ReplayEvent
from game_state import Phase, Resource, DevCard


# ── load ──────────────────────────────────────────────────────────────────────

def load(path: str) -> ReplayData:
    with open(path, "rb") as f:
        class _U(pickle.Unpickler):
            def find_class(self, m, n):
                if n == "ReplayData":  return ReplayData
                if n == "ReplayEvent": return ReplayEvent
                return super().find_class(m, n)
        return _U(f).load()


# ── helpers ───────────────────────────────────────────────────────────────────

PIP_TABLE = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

PLAYER_NAMES = ["Red", "Blue", "Orange", "White"]


def pip_name(pid: int) -> str:
    return PLAYER_NAMES[pid] if pid < len(PLAYER_NAMES) else f"P{pid}"


def vertex_pip_info(board, topo, vid: int):
    """Return (total_pips, diversity, description) for a vertex."""
    pips, resources, details = 0, set(), []
    for hid in topo.vertex_hexes[vid]:
        h = board.hexes[hid]
        if h.hex_type.value != "desert":
            pip = PIP_TABLE.get(h.token, 0)
            pips += pip
            resources.add(h.hex_type.value)
            details.append(f"{h.hex_type.value[0].upper()}({h.token}={pip}p)")
        else:
            details.append("Desert")
    return pips, len(resources), " + ".join(details)


def compute_public_vp(state, pid: int) -> int:
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


def hr(title: str, width: int = 60) -> None:
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def label(pid: int, human_seat: int) -> str:
    name = pip_name(pid)
    tag  = " (YOU)" if pid == human_seat else "  (AI)"
    return f"P{pid} {name}{tag}"


# ── main analysis ─────────────────────────────────────────────────────────────

def analyze(data: ReplayData, human_seat: int) -> None:
    states = data.states
    events = data.events
    n      = data.num_players
    board  = states[0].board
    topo   = states[0].topology
    meta   = data.metadata or {}

    # ── 1. Game summary ───────────────────────────────────────────────────────
    hr("GAME SUMMARY")
    policy = Path(meta.get("policy_path", "?")).name
    recorded = meta.get("recorded_at", "?")
    print(f"  Recorded : {recorded}")
    print(f"  Policy   : {policy}")
    print(f"  Players  : {n}  |  human seat: {human_seat}")
    final = states[-1]
    winner = data.winner
    total_turns = final.turn_number
    print(f"  Winner   : {label(winner, human_seat)}")
    print(f"  Turns    : {total_turns}")
    print()
    for pid in range(n):
        p   = final.players[pid]
        vp  = compute_public_vp(final, pid)
        sett = 5 - p.settlements_left
        city = 4 - p.cities_left
        road = 15 - p.roads_left
        print(f"  {label(pid, human_seat):24s}  VP={vp:2d}  "
              f"sett={sett}  city={city}  road={road:2d}")

    # ── 2. Setup quality ──────────────────────────────────────────────────────
    hr("SETUP QUALITY  (starting settlements)")
    setup_events = [e for e in events if "PLACE_SETTLEMENT" in e.action_repr][:2 * n]
    pip_scores: dict[int, list[int]] = defaultdict(list)
    best_available_pips = None

    # rank all vertices once
    all_verts = sorted(
        [(vid, *vertex_pip_info(board, topo, vid)) for vid in range(topo.num_vertices)],
        key=lambda x: (x[1], x[2]), reverse=True
    )

    for e in setup_events:
        vid = int(e.action_repr.split("v=")[1].rstrip(")"))
        pips, div, detail = vertex_pip_info(board, topo, vid)
        pip_scores[e.player].append(pips)
        rank = next(i for i, (v, p, d, _) in enumerate(all_verts) if v == vid) + 1
        print(f"  {label(e.player, human_seat):24s}  v={vid:3d}  "
              f"pips={pips:2d}  div={div}  rank=#{rank:2d}  [{detail}]")

    print()
    print("  Setup pip totals:")
    for pid in range(n):
        total = sum(pip_scores.get(pid, []))
        print(f"    {label(pid, human_seat):24s}  total={total:2d}")

    # ── 3. VP progression ─────────────────────────────────────────────────────
    hr("VP PROGRESSION  (every 5 turns)")
    # sample VP at regular intervals from states that fall on turn boundaries
    sampled_states: list[tuple[int, object]] = []
    last_turn = -1
    for st in states:
        if st.phase in (Phase.MAIN, Phase.DONE) and st.turn_number != last_turn:
            sampled_states.append((st.turn_number, st))
            last_turn = st.turn_number

    sample_interval = max(1, total_turns // 10)
    checkpoints = sorted({0, *range(sample_interval, total_turns + sample_interval, sample_interval)})
    header = "  Turn  |" + "".join(f" {pip_name(p):>8}" for p in range(n))
    print(header)
    print("  " + "-" * (len(header) - 2))
    shown = set()
    for turn, st in sampled_states:
        bucket = (turn // sample_interval) * sample_interval
        if bucket in shown:
            continue
        shown.add(bucket)
        vps = [compute_public_vp(st, p) for p in range(n)]
        row = f"  {turn:5d}  |" + "".join(f" {v:>8}" for v in vps)
        print(row)

    # ── 4. Road efficiency ────────────────────────────────────────────────────
    hr("ROAD EFFICIENCY")
    build_counts: dict[int, Counter] = {pid: Counter() for pid in range(n)}
    for e in events:
        r = e.action_repr
        if   "PLACE_ROAD"        in r: build_counts[e.player]["road"]       += 1
        elif "PLACE_SETTLEMENT"  in r: build_counts[e.player]["settlement"] += 1
        elif "UPGRADE_CITY"      in r: build_counts[e.player]["city"]       += 1
        elif "BUY_DEV_CARD"      in r: build_counts[e.player]["dev"]        += 1

    print(f"  {'Player':24s}  roads  sett  city  dev   roads/prod")
    for pid in range(n):
        c    = build_counts[pid]
        prod = c["settlement"] + c["city"]   # productive buildings
        ratio = f"{c['road']/prod:.1f}" if prod else "N/A"
        print(f"  {label(pid, human_seat):24s}  "
              f"{c['road']:5d}  {c['settlement']:4d}  {c['city']:4d}  "
              f"{c['dev']:3d}   {ratio:>10}")

    # ── 5. Resource economy ───────────────────────────────────────────────────
    hr("RESOURCE ECONOMY")
    # Track resources gained and lost by watching state diffs around ROLL events
    gained:   dict[int, Counter] = {pid: Counter() for pid in range(n)}
    discarded:dict[int, Counter] = {pid: Counter() for pid in range(n)}
    traded_away: dict[int, int]  = {pid: 0 for pid in range(n)}
    traded_got:  dict[int, int]  = {pid: 0 for pid in range(n)}

    for i, e in enumerate(events):
        r = e.action_repr
        # states[0] = manually-added initial snapshot; states[k] = state saved
        # at the start of step() for events[k-1].  So the correct pair is:
        #   before events[i]  → states[i+1]
        #   after  events[i]  → states[i+2]
        st_before = states[i + 1] if i + 1 < len(states) else states[-1]
        st_after  = states[i + 2] if i + 2 < len(states) else states[-1]

        if "ROLL_DICE" in r:
            # Resources distributed after roll
            for pid in range(n):
                for res in Resource:
                    delta = (st_after.players[pid].resources.get(res, 0) -
                             st_before.players[pid].resources.get(res, 0))
                    if delta > 0:
                        gained[pid][res] += delta

        elif "DISCARD" in r and "PLACE" not in r:
            pid = e.player
            for res in Resource:
                delta = (st_before.players[pid].resources.get(res, 0) -
                         st_after.players[pid].resources.get(res, 0))
                if delta > 0:
                    discarded[pid][res] += delta

        elif "MARITIME_TRADE" in r:
            pid = e.player
            # parse give/recv from repr e.g. "Action(MARITIME_TRADE, give=lumber, recv=ore)"
            try:
                give_r  = Resource(r.split("give=")[1].split(",")[0].rstrip(")"))
                recv_r  = Resource(r.split("recv=")[1].split(",")[0].rstrip(")"))
                ratio   = st_before.players[pid].resources.get(give_r, 0) - \
                          st_after.players[pid].resources.get(give_r, 0)
                traded_away[pid] += max(ratio, 0)
                recv_delta = (st_after.players[pid].resources.get(recv_r, 0) -
                              st_before.players[pid].resources.get(recv_r, 0))
                traded_got[pid]  += max(recv_delta, 0)
            except Exception:
                pass

        elif "PLAY_MONOPOLY" in r:
            pid = e.player
            for res in Resource:
                delta = (st_after.players[pid].resources.get(res, 0) -
                         st_before.players[pid].resources.get(res, 0))
                if delta > 0:
                    gained[pid][res] += delta

        elif "PLAY_YEAR_OF_PLENTY" in r:
            pid = e.player
            for res in Resource:
                delta = (st_after.players[pid].resources.get(res, 0) -
                         st_before.players[pid].resources.get(res, 0))
                if delta > 0:
                    gained[pid][res] += delta

    print(f"  {'Player':24s}  gained  discarded  trades(give/get)  efficiency")
    for pid in range(n):
        g   = sum(gained[pid].values())
        d   = sum(discarded[pid].values())
        eff = f"{(g - d) / g * 100:.0f}%" if g else "N/A"
        tw  = traded_away[pid]
        tg  = traded_got[pid]
        print(f"  {label(pid, human_seat):24s}  "
              f"{g:6d}  {d:9d}  {tw:5d}/{tg:<5d}        {eff}")

    print()
    # Per-resource gain breakdown
    print("  Resources gained per player:")
    res_order = list(Resource)
    header2 = f"  {'Player':24s}  " + "  ".join(f"{r.value[:4]:>4}" for r in res_order)
    print(header2)
    for pid in range(n):
        row = f"  {label(pid, human_seat):24s}  " + \
              "  ".join(f"{gained[pid].get(r, 0):4d}" for r in res_order)
        print(row)

    # ── 6. Dev cards ──────────────────────────────────────────────────────────
    hr("DEV CARDS")
    dev_bought: dict[int, int]     = {pid: 0 for pid in range(n)}
    dev_played: dict[int, Counter] = {pid: Counter() for pid in range(n)}
    for e in events:
        if "BUY_DEV_CARD"      in e.action_repr: dev_bought[e.player] += 1
        elif "PLAY_KNIGHT"     in e.action_repr: dev_played[e.player]["knight"]       += 1
        elif "PLAY_MONOPOLY"   in e.action_repr: dev_played[e.player]["monopoly"]     += 1
        elif "PLAY_YEAR"       in e.action_repr: dev_played[e.player]["year_plenty"]  += 1
        elif "PLAY_ROAD_BUILD" in e.action_repr: dev_played[e.player]["road_build"]   += 1

    print(f"  {'Player':24s}  bought  knight  mono  yop  road_build")
    for pid in range(n):
        dp = dev_played[pid]
        print(f"  {label(pid, human_seat):24s}  "
              f"{dev_bought[pid]:6d}  "
              f"{dp['knight']:6d}  "
              f"{dp['monopoly']:4d}  "
              f"{dp['year_plenty']:3d}  "
              f"{dp['road_build']:10d}")

    # ── 7. Trade activity ─────────────────────────────────────────────────────
    hr("TRADE ACTIVITY")
    maritime_trades: dict[int, int] = {pid: 0 for pid in range(n)}
    player_trades:   dict[int, int] = {pid: 0 for pid in range(n)}
    for e in events:
        if "MARITIME_TRADE" in e.action_repr: maritime_trades[e.player] += 1
        elif "PLAYER_TRADE" in e.action_repr: player_trades[e.player]   += 1

    print(f"  {'Player':24s}  maritime  player-trade")
    for pid in range(n):
        print(f"  {label(pid, human_seat):24s}  "
              f"{maritime_trades[pid]:8d}  {player_trades[pid]:12d}")

    # ── 8. Robber impact ──────────────────────────────────────────────────────
    hr("ROBBER IMPACT")
    robber_placed_on_opp: dict[int, int] = {pid: 0 for pid in range(n)}
    times_robbed:         dict[int, int] = {pid: 0 for pid in range(n)}

    for i, e in enumerate(events):
        if "MOVE_ROBBER" not in e.action_repr:
            continue
        pid = e.player
        # steal_from in repr: "steal=X"
        if "steal=" in e.action_repr:
            try:
                victim = int(e.action_repr.split("steal=")[1].rstrip(")"))
                if victim != pid:
                    robber_placed_on_opp[pid]  += 1
                    times_robbed[victim]        += 1
            except Exception:
                pass
        else:
            # no steal but still placed on an occupied hex?
            robber_placed_on_opp[pid] += 1

    print(f"  {'Player':24s}  robbed-others  times-robbed")
    for pid in range(n):
        print(f"  {label(pid, human_seat):24s}  "
              f"{robber_placed_on_opp[pid]:13d}  {times_robbed[pid]:12d}")

    # ── 9. Head-to-head summary ───────────────────────────────────────────────
    hr("HEAD-TO-HEAD: YOU vs AVG(AI)")
    if n < 2:
        print("  Not enough players for comparison.")
        return

    ai_seats = [pid for pid in range(n) if pid != human_seat]
    f = states[-1]

    def avg(vals):
        return sum(vals) / len(vals) if vals else 0

    def diff_line(metric: str, human_val, ai_vals, higher_better: bool = True):
        ai_avg = avg(ai_vals)
        if isinstance(human_val, float):
            h_str  = f"{human_val:.1f}"
            a_str  = f"{ai_avg:.1f}"
            delta  = human_val - ai_avg
            d_str  = f"{delta:+.1f}"
        else:
            h_str  = str(human_val)
            a_str  = f"{ai_avg:.1f}"
            delta  = human_val - ai_avg
            d_str  = f"{delta:+.1f}"
        better = (delta > 0) == higher_better
        tag = "✓ BETTER" if better else "✗ WORSE " if delta != 0 else "  EQUAL "
        print(f"  {metric:28s}  you={h_str:>6}  ai_avg={a_str:>6}  {d_str:>6}  {tag}")

    h = human_seat
    pip_h  = sum(pip_scores.get(h, []))
    pip_ai = [sum(pip_scores.get(pid, [])) for pid in ai_seats]
    diff_line("Setup pip total",      pip_h,  pip_ai)

    vp_h   = compute_public_vp(f, h)
    vp_ai  = [compute_public_vp(f, pid) for pid in ai_seats]
    diff_line("Final VP",             vp_h,   vp_ai)

    bc_h   = build_counts[h]
    diff_line("Settlements built",    bc_h["settlement"], [build_counts[pid]["settlement"] for pid in ai_seats])
    diff_line("Cities built",         bc_h["city"],       [build_counts[pid]["city"]       for pid in ai_seats])
    diff_line("Roads built",          bc_h["road"],       [build_counts[pid]["road"]       for pid in ai_seats], higher_better=False)
    diff_line("Dev cards bought",     bc_h["dev"],        [build_counts[pid]["dev"]        for pid in ai_seats])

    g_h = sum(gained[h].values())
    g_ai = [sum(gained[pid].values()) for pid in ai_seats]
    diff_line("Resources gained",     g_h,    g_ai)

    d_h = sum(discarded[h].values())
    d_ai = [sum(discarded[pid].values()) for pid in ai_seats]
    diff_line("Cards discarded",      d_h,    d_ai, higher_better=False)

    mt_h = maritime_trades[h]
    mt_ai = [maritime_trades[pid] for pid in ai_seats]
    diff_line("Maritime trades",      mt_h,   mt_ai)

    kr_h = dev_played[h]["knight"]
    kr_ai = [dev_played[pid]["knight"] for pid in ai_seats]
    diff_line("Knights played",       kr_h,   kr_ai)

    rob_h = robber_placed_on_opp[h]
    rob_ai = [robber_placed_on_opp[pid] for pid in ai_seats]
    diff_line("Robber vs opponents",  rob_h,  rob_ai)

    rib_h = times_robbed[h]
    rib_ai = [times_robbed[pid] for pid in ai_seats]
    diff_line("Times robbed",         rib_h,  rib_ai, higher_better=False)

    print()
    if winner == human_seat:
        print("  ★ YOU WON! ★")
    else:
        print(f"  Winner was {label(winner, human_seat)}.")
    print()


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a human vs AI Catan replay."
    )
    parser.add_argument("replay", help="Path to .pkl replay file")
    parser.add_argument(
        "--human-seat", type=int, default=None,
        help="Which seat is the human (0-3). Auto-detected from metadata if omitted.",
    )
    args = parser.parse_args()

    data = load(args.replay)
    meta = data.metadata or {}

    human_seat = args.human_seat
    if human_seat is None:
        human_seat = meta.get("human_seat", 0)

    print(f"\nAnalyzing: {args.replay}")
    analyze(data, int(human_seat))


if __name__ == "__main__":
    main()
