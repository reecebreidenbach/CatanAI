# AI-ASSISTED
"""
play_vs_ai.py — Interactive human vs AI game.

Subclasses board_builder.py and adds:
  - AI policy running all non-human seats
  - Playable dev cards (Knight, Monopoly, Year of Plenty, Road Building)
  - Startup dialog to choose policy and seat color
  - Randomized seat assignment option

Usage
-----
    python play_vs_ai.py          # opens setup dialog
    python play_vs_ai.py --policy phase3_policy.pt --human-seat 2

Run from the Game/ directory.
"""

from __future__ import annotations

import argparse
import random
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Optional

import datetime
import pickle
import copy

import torch

from board_builder import BoardBuilderApp, PLAYER_COLORS
from catan_env import CatanEnv, encode_obs, legal_action_mask, decode_action, _auto_discard
from game_engine import Action, ActionType
from game_state import Phase, Resource, DevCard
from policy import CatanPolicy, masked_sample
from ppo_utils import HIDDEN_SIZE
from replay_tools import ReplayData, ReplayEvent


# ── Recording engine wrapper ─────────────────────────────────────────────────

class _RecordingEngine:
    """
    Thin proxy around GameEngine that records every step() call.
    All other attribute accesses are forwarded to the real engine.
    """

    def __init__(self, engine) -> None:
        self._engine      = engine
        self.states: list  = []
        self.events: list[ReplayEvent] = []
        self._step_count   = 0

    def __getattr__(self, name: str):
        return getattr(self._engine, name)

    def step(self, state, action):
        self.states.append(state.copy())
        new_state, reward, done = self._engine.step(state, action)
        pid = state.current_player
        self.events.append(ReplayEvent(
            step=self._step_count,
            player=pid,
            action_idx=-1,          # not applicable for human/mixed play
            action_repr=repr(action),
            rewards={pid: reward},
            done=done,
        ))
        self._step_count += 1
        return new_state, reward, done


# ── AI agent ──────────────────────────────────────────────────────────────────

class _AIAgent:
    def __init__(self, policy_path: Path, env: CatanEnv) -> None:
        self._policy = CatanPolicy(
            obs_size=env.obs_size(),
            action_size=env.action_size(),
            hidden=HIDDEN_SIZE,
        )
        self._policy.load_state_dict(
            torch.load(policy_path, map_location="cpu", weights_only=True)
        )
        self._policy.eval()

    def choose(self, obs, mask) -> int:
        obs_t  = torch.tensor(obs,  dtype=torch.float32)
        mask_t = torch.tensor(mask, dtype=torch.bool)
        with torch.no_grad():
            logits, _ = self._policy(obs_t)
        return masked_sample(logits, mask_t)


# ── Setup dialog ──────────────────────────────────────────────────────────────

class _SetupDialog(tk.Toplevel):
    """
    Modal startup dialog: pick policy file and seat color.
    Sets .result = (policy_path, human_seat) or None if cancelled.
    """

    def __init__(self, parent: tk.Tk) -> None:
        super().__init__(parent)
        self.title("Catan vs AI — Setup")
        self.resizable(False, False)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.result: Optional[tuple[Path, int]] = None

        game_dir = Path(__file__).resolve().parent
        policies = _available_policies(game_dir)

        BG   = "#1a2e4a"
        FG   = "#ffffff"
        BOLD = ("Arial", 11, "bold")
        NORM = ("Arial", 10)
        self.configure(bg=BG)

        tk.Label(self, text="Catan vs AI", font=("Arial", 16, "bold"),
                 bg=BG, fg="#f0c040", pady=12).pack()

        # ── Policy selection ────────────────────────────────────────────────
        tk.Label(self, text="Choose AI policy:", font=BOLD,
                 bg=BG, fg="#aad4f5", anchor="w", padx=20).pack(fill=tk.X)

        self._policy_var = tk.StringVar()
        policy_frame = tk.Frame(self, bg=BG, padx=20, pady=4)
        policy_frame.pack(fill=tk.X)

        if policies:
            self._policy_var.set(str(policies[0]))
            for p in policies:
                tk.Radiobutton(
                    policy_frame, text=p.name,
                    variable=self._policy_var, value=str(p),
                    font=NORM, bg=BG, fg=FG,
                    selectcolor="#2a4e7a", activebackground=BG,
                ).pack(anchor="w")
        else:
            tk.Label(policy_frame, text="No policies found in Game/",
                     font=NORM, bg=BG, fg="#ff6b6b").pack(anchor="w")

        browse_row = tk.Frame(self, bg=BG, padx=20, pady=2)
        browse_row.pack(fill=tk.X)
        tk.Button(browse_row, text="Browse…", font=NORM,
                  command=self._browse).pack(side=tk.LEFT)
        self._browse_lbl = tk.Label(browse_row, text="", font=("Arial", 9),
                                    bg=BG, fg="#aaaaaa")
        self._browse_lbl.pack(side=tk.LEFT, padx=8)

        # ── Seat selection ──────────────────────────────────────────────────
        tk.Label(self, text="Choose your color:", font=BOLD,
                 bg=BG, fg="#aad4f5", anchor="w", padx=20).pack(fill=tk.X, pady=(8, 0))

        seat_frame = tk.Frame(self, bg=BG, padx=20, pady=4)
        seat_frame.pack(fill=tk.X)

        self._seat_var = tk.IntVar(value=-1)   # -1 = random
        tk.Radiobutton(
            seat_frame, text="Random seat",
            variable=self._seat_var, value=-1,
            font=NORM, bg=BG, fg=FG,
            selectcolor="#2a4e7a", activebackground=BG,
        ).pack(anchor="w")
        for i, color in enumerate(PLAYER_COLORS):
            tk.Radiobutton(
                seat_frame,
                text=f"{color['name']} (seat {i})",
                variable=self._seat_var, value=i,
                font=NORM, bg=color["fill"], fg=color["text"],
                selectcolor=color["outline"],
                activebackground=color["fill"],
            ).pack(anchor="w", pady=1, fill=tk.X)

        # ── Buttons ─────────────────────────────────────────────────────────
        btn_row = tk.Frame(self, bg=BG, pady=12)
        btn_row.pack()
        tk.Button(btn_row, text="Start Game", font=BOLD,
                  bg="#2a7a2a", fg="white", padx=20, pady=6,
                  command=self._confirm).pack(side=tk.LEFT, padx=8)
        tk.Button(btn_row, text="Cancel", font=NORM,
                  bg="#7a2a2a", fg="white", padx=12, pady=6,
                  command=self._cancel).pack(side=tk.LEFT, padx=8)

        self.update_idletasks()
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select policy checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
            initialdir=str(Path(__file__).resolve().parent),
        )
        if path:
            self._policy_var.set(path)
            self._browse_lbl.config(text=Path(path).name)

    def _confirm(self) -> None:
        policy_str = self._policy_var.get()
        if not policy_str:
            return
        policy_path = Path(policy_str)
        if not policy_path.exists():
            tk.messagebox.showerror("File not found", str(policy_path), parent=self)
            return
        seat = self._seat_var.get()
        if seat == -1:
            seat = random.randrange(4)
        self.result = (policy_path, seat)
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()


# ── Main app ──────────────────────────────────────────────────────────────────

class PlayVsAIApp(BoardBuilderApp):
    """
    BoardBuilderApp extended with:
      - AI policy auto-playing all non-human seats
      - Dev card play buttons (Knight, Monopoly, YoP, Road Building)
      - Randomized seat selection at startup
    """

    _AI_STEP_DELAY_MS = 350

    def __init__(self, policy_path: Path, human_seat: int) -> None:
        self._policy_path = policy_path
        self._human_seat  = human_seat
        self._ai_agent:   Optional[_AIAgent] = None
        self._ai_pending: bool = False
        # Dev card button references (created in _build_dev_card_buttons)
        self._dev_btns: dict[DevCard, tk.Button] = {}
        super().__init__()
        color = PLAYER_COLORS[human_seat]
        self.title(
            f"Catan vs AI  —  You are {color['name']} (seat {human_seat})  |  "
            f"policy: {policy_path.name}"
        )

    # ── Sidebar extension — dev card play buttons ─────────────────────────────
    # board_builder._build_ui() calls _section(sidebar, "Game") and then packs
    # the buttons we need to appear after. We override _build_ui to inject our
    # extra buttons right after the existing Game section ones.

    def _build_ui(self) -> None:
        super()._build_ui()
        # Find the sidebar scrollable frame (it's the parent of the Buy Dev Card button)
        sidebar = self._buy_dev_btn.master
        self._build_dev_card_buttons(sidebar)

    def _build_dev_card_buttons(self, sidebar: tk.Frame) -> None:
        """Add a 'Play Dev Card' section to the sidebar."""
        from board_builder import BoardBuilderApp
        # Reuse the _section helper from BoardBuilderApp
        self._section(sidebar, "Play Dev Card")

        specs = [
            (DevCard.KNIGHT,         "⚔  Knight  (move robber)"),
            (DevCard.MONOPOLY,       "💰  Monopoly"),
            (DevCard.YEAR_OF_PLENTY, "🌾  Year of Plenty"),
            (DevCard.ROAD_BUILDING,  "🛤  Road Building (2 free roads)"),
        ]
        for card, label in specs:
            btn = tk.Button(
                sidebar, text=label, font=("Arial", 10), pady=3,
                state=tk.DISABLED,
                command=lambda c=card: self._play_dev_card(c),
            )
            btn.pack(fill=tk.X, pady=(0, 3))
            self._dev_btns[card] = btn

    # ── Dev card play ─────────────────────────────────────────────────────────

    def _play_dev_card(self, card: DevCard) -> None:
        state = self._game_state
        if state is None or state.phase != Phase.MAIN:
            return
        if state.current_player != self._human_seat:
            return
        if card == DevCard.KNIGHT:
            self._game_state, _, _ = self._game_engine.step(
                state, Action(ActionType.PLAY_KNIGHT))
            self._sync_pieces_from_state()
            self._update_status()
        elif card == DevCard.MONOPOLY:
            self._show_monopoly_dialog()
        elif card == DevCard.YEAR_OF_PLENTY:
            self._show_yop_dialog()
        elif card == DevCard.ROAD_BUILDING:
            self._game_state, _, _ = self._game_engine.step(
                state, Action(ActionType.PLAY_ROAD_BUILDING))
            self._sync_pieces_from_state()
            self._update_status()

    def _show_monopoly_dialog(self) -> None:
        state = self._game_state
        if state is None:
            return
        BG, FG = "#1a2e4a", "#ffffff"
        BOLD, NORM = ("Arial", 12, "bold"), ("Arial", 11)
        dlg = tk.Toplevel(self)
        dlg.title("Monopoly")
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.grab_set()
        tk.Label(dlg, text="Choose a resource to monopolize:",
                 font=BOLD, bg=BG, fg="#f0c040", padx=16, pady=10).pack()
        res_var = tk.StringVar(value="lumber")
        for r in Resource:
            tk.Radiobutton(dlg, text=r.value.capitalize(),
                           variable=res_var, value=r.value,
                           font=NORM, bg=BG, fg=FG,
                           selectcolor="#2a4e7a", activebackground=BG,
                           ).pack(anchor="w", padx=24)
        def _confirm():
            dlg.destroy()
            act = Action(ActionType.PLAY_MONOPOLY, receive=Resource(res_var.get()))
            self._game_state, _, _ = self._game_engine.step(state, act)
            self._sync_pieces_from_state()
            self._update_status()
        tk.Button(dlg, text="Confirm", font=BOLD, bg="#2a7a2a", fg="white",
                  padx=16, pady=5, command=_confirm).pack(pady=(8, 12))
        dlg.update_idletasks()
        w, h = dlg.winfo_reqwidth(), dlg.winfo_reqheight()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        dlg.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def _show_yop_dialog(self) -> None:
        state = self._game_state
        if state is None:
            return
        BG, FG = "#1a2e4a", "#ffffff"
        BOLD, NORM = ("Arial", 12, "bold"), ("Arial", 11)
        dlg = tk.Toplevel(self)
        dlg.title("Year of Plenty")
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.grab_set()
        tk.Label(dlg, text="Take 2 resources from the bank:",
                 font=BOLD, bg=BG, fg="#f0c040", padx=16, pady=10).pack()
        res_list = list(Resource)
        res_names = [r.value.capitalize() for r in res_list]
        r1_var = tk.StringVar(value=res_list[0].value)
        r2_var = tk.StringVar(value=res_list[0].value)
        for label, var in (("First resource:", r1_var), ("Second resource:", r2_var)):
            tk.Label(dlg, text=label, font=BOLD, bg=BG, fg="#aad4f5",
                     padx=16, anchor="w").pack(fill=tk.X, pady=(6, 0))
            row = tk.Frame(dlg, bg=BG, padx=16)
            row.pack(fill=tk.X)
            for r in res_list:
                tk.Radiobutton(row, text=r.value.capitalize(),
                               variable=var, value=r.value,
                               font=NORM, bg=BG, fg=FG,
                               selectcolor="#2a4e7a", activebackground=BG,
                               ).pack(side=tk.LEFT, padx=4)
        err_var = tk.StringVar()
        tk.Label(dlg, textvariable=err_var, font=("Arial", 10),
                 bg=BG, fg="#ff6b6b").pack()
        def _confirm():
            r1 = Resource(r1_var.get())
            r2 = Resource(r2_var.get())
            # Validate bank has enough
            bank = state.bank
            need = {r1: 0, r2: 0}
            need[r1] += 1
            need[r2] += 1
            for r, amt in need.items():
                if bank.get(r, 0) < amt:
                    err_var.set(f"Bank has no {r.value}.")
                    return
            dlg.destroy()
            act = Action(ActionType.PLAY_YEAR_OF_PLENTY, give=r1, receive=r2)
            self._game_state, _, _ = self._game_engine.step(state, act)
            self._sync_pieces_from_state()
            self._update_status()
        tk.Button(dlg, text="Confirm", font=BOLD, bg="#2a7a2a", fg="white",
                  padx=16, pady=5, command=_confirm).pack(pady=(8, 12))
        dlg.update_idletasks()
        w, h = dlg.winfo_reqwidth(), dlg.winfo_reqheight()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        dlg.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    # ── Refresh dev-card button states ────────────────────────────────────────

    def _update_dev_card_buttons(self) -> None:
        if not self._dev_btns:
            return
        state = self._game_state
        is_human_main = (
            state is not None
            and state.phase == Phase.MAIN
            and state.current_player == self._human_seat
            and not state.dev_card_played_this_turn
        )
        if not is_human_main:
            for btn in self._dev_btns.values():
                btn.config(state=tk.DISABLED)
            return
        from collections import Counter
        hand = Counter(state.players[self._human_seat].dev_cards)
        playable = {
            DevCard.KNIGHT:         hand[DevCard.KNIGHT] > 0,
            DevCard.MONOPOLY:       hand[DevCard.MONOPOLY] > 0,
            DevCard.YEAR_OF_PLENTY: hand[DevCard.YEAR_OF_PLENTY] > 0,
            DevCard.ROAD_BUILDING:  hand[DevCard.ROAD_BUILDING] > 0 and state.players[self._human_seat].roads_left > 0,
        }
        for card, btn in self._dev_btns.items():
            btn.config(state=tk.NORMAL if playable[card] else tk.DISABLED)

    # ── Game start ────────────────────────────────────────────────────────────

    def _start_game(self) -> None:
        super()._start_game()
        # Wrap the engine with a recorder immediately after super() creates it
        self._game_engine = _RecordingEngine(self._game_engine)
        # Snapshot initial state
        self._game_engine.states.append(self._game_state.copy())
        n = self._num_players.get()
        env = CatanEnv(num_players=n)
        self._ai_agent   = _AIAgent(self._policy_path, env)
        self._ai_pending = False
        self._schedule_ai_if_needed()

    # ── AI scheduling ─────────────────────────────────────────────────────────

    def _schedule_ai_if_needed(self) -> None:
        if self._ai_pending:
            return
        state = self._game_state
        if state is None or state.phase == Phase.DONE:
            return
        if state.current_player == self._human_seat:
            return
        if state.phase == Phase.DISCARD:
            return
        self._ai_pending = True
        self.after(self._AI_STEP_DELAY_MS, self._advance_ai_turn)

    def _advance_ai_turn(self) -> None:
        self._ai_pending = False
        state = self._game_state
        if state is None or state.phase == Phase.DONE or self._ai_agent is None:
            return
        if state.current_player == self._human_seat:
            return
        if state.phase == Phase.DISCARD:
            return
        pid = state.current_player
        obs  = encode_obs(state, pid, self._game_engine)
        mask = legal_action_mask(state, self._game_engine)
        action_idx = self._ai_agent.choose(obs, mask)
        action     = decode_action(action_idx, state)
        old_robber = state.robber_hex
        self._game_state, _, _ = self._game_engine.step(state, action)
        self._sync_pieces_from_state()
        # If the robber moved, animate it (the AI bypasses the click handler
        # which normally triggers _animate_robber, so we do it here).
        new_robber = self._game_state.robber_hex
        if new_robber != old_robber:
            self._animate_robber(old_robber, new_robber)
        self._update_status()

    # ── Override _update_status ───────────────────────────────────────────────

    def _update_status(self) -> None:
        super()._update_status()
        self._update_dev_card_buttons()
        self._schedule_ai_if_needed()
        # Offer to save replay when the game ends
        state = self._game_state
        if state is not None and state.phase == Phase.DONE:
            self.after(200, self._offer_save_replay)

    # ── Replay saving ─────────────────────────────────────────────────────────

    def _offer_save_replay(self) -> None:
        """Ask the user if they want to save the replay, then do so."""
        if not isinstance(self._game_engine, _RecordingEngine):
            return
        if not tk.messagebox.askyesno(
            "Game over",
            "Save replay for analysis?\n"
            "(Analyze later with: python analyze_replay.py replays/<file>)",
            parent=self,
        ):
            return
        self._save_replay()

    def _save_replay(self) -> None:
        if not isinstance(self._game_engine, _RecordingEngine):
            return
        rec = self._game_engine
        # Append the final state
        if self._game_state is not None:
            rec.states.append(self._game_state.copy())
        n = self._num_players.get()
        replay = ReplayData(
            states=rec.states,
            events=rec.events,
            winner=self._game_state.winner if self._game_state else None,
            num_players=n,
            metadata={
                "policy_path": str(self._policy_path),
                "human_seat":  self._human_seat,
                "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "type": "human_vs_ai",
            },
        )
        game_dir  = Path(__file__).resolve().parent
        replay_dir = game_dir / "replays"
        replay_dir.mkdir(exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"human_vs_ai_{ts}.pkl"
        path = filedialog.asksaveasfilename(
            title="Save replay",
            initialdir=str(replay_dir),
            initialfile=default_name,
            defaultextension=".pkl",
            filetypes=[("Pickle replay", "*.pkl"), ("All files", "*.*")],
            parent=self,
        )
        if not path:
            return
        with open(path, "wb") as f:
            pickle.dump(replay, f)
        print(f"Replay saved: {path}")
        tk.messagebox.showinfo(
            "Saved",
            f"Replay saved to:\n{path}\n\n"
            f"Analyze with:\n  python analyze_replay.py {Path(path).name}",
            parent=self,
        )

    # ── Auto-discard for AI seats ─────────────────────────────────────────────

    def _show_discard_dialog(self) -> None:
        state = self._game_state
        if state is None or state.phase != Phase.DISCARD:
            return
        pid = state.current_player
        if pid != self._human_seat:
            action = _auto_discard(state, pid)
            self._game_state, _, _ = self._game_engine.step(state, action)
            self._sync_pieces_from_state()
            self._update_status()
            return
        super()._show_discard_dialog()

    # ── Block human input during AI turns ─────────────────────────────────────

    def _on_game_click(self, event: tk.Event) -> None:
        state = self._game_state
        if state is not None and state.current_player != self._human_seat:
            return
        super()._on_game_click(event)

    def _update_action_buttons(self) -> None:
        super()._update_action_buttons()
        state = self._game_state
        if state is None or state.phase == Phase.DONE:
            return
        if state.current_player != self._human_seat:
            for btn in (
                self._roll_btn, self._end_turn_btn,
                self._trade_btn, self._player_trade_btn, self._buy_dev_btn,
            ):
                if btn is not None:
                    btn.config(state=tk.DISABLED)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _available_policies(game_dir: Path) -> list[Path]:
    """Return all phase policy checkpoints, newest first."""
    found: list[Path] = []
    for glob in ("phase3_policy*.pt", "phase2_policy*.pt", "phase1_policy*.pt"):
        found.extend(sorted(game_dir.glob(glob), key=lambda p: p.stat().st_mtime, reverse=True))
    return found


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Play Catan against the AI policy.")
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--human-seat", type=str, default=None)
    args = parser.parse_args()

    game_dir = Path(__file__).resolve().parent

    # If both args provided, skip the dialog.
    if args.policy and args.human_seat is not None:
        policy_path = Path(args.policy)
        if not policy_path.is_absolute():
            policy_path = game_dir / policy_path
        if not policy_path.exists():
            raise SystemExit(f"Policy file not found: {policy_path}")
        if args.human_seat == "random":
            human_seat = random.randrange(4)
        else:
            try:
                human_seat = int(args.human_seat)
            except ValueError:
                raise SystemExit("--human-seat must be 0-3 or 'random'.")
            if not 0 <= human_seat <= 3:
                raise SystemExit("--human-seat must be 0-3.")
    else:
        # Show the setup dialog
        _root = tk.Tk()
        _root.withdraw()
        dlg = _SetupDialog(_root)
        _root.wait_window(dlg)
        if dlg.result is None:
            _root.destroy()
            return
        policy_path, human_seat = dlg.result
        _root.destroy()

    print(f"Policy:    {policy_path.name}")
    print(f"Your seat: {human_seat} ({PLAYER_COLORS[human_seat]['name']})")
    print()

    app = PlayVsAIApp(policy_path, human_seat)
    app.mainloop()


if __name__ == "__main__":
    main()
