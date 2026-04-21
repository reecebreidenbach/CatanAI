"""
AI ASSISTED
board_builder.py - Interactive Catan board builder GUI.

Run:  python board_builder.py

Click on the board to place settlements, cities, and roads.
  - Left-click  : place selected piece at nearest valid snap point
  - Right-click : remove piece at nearest snap point
Buttons let you switch mode/player, randomize the layout, and export to PNG.
"""

from __future__ import annotations

import math
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

from board import Board, HexType, ROW_SIZES, PortType

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from game_engine import GameEngine, Action, ActionType
    from game_state import Phase, Resource, PORT_RESOURCE, DevCard, DEV_CARD_LABEL
    GAME_ENGINE_AVAILABLE = True
except ImportError:
    GAME_ENGINE_AVAILABLE = False

# ── paths ──────────────────────────────────────────────────────────────────────
IMAGES_DIR  = Path(__file__).parent.parent / "Images"
HEX_IMG_DIR = IMAGES_DIR / "hexes" / "vector"

# ── hex colours (fallback when images unavailable) ─────────────────────────────
HEX_COLORS: dict[HexType, str] = {
    HexType.FOREST:   "#2d6a4f",
    HexType.PASTURE:  "#74c69d",
    HexType.FIELD:    "#ffd166",
    HexType.HILL:     "#c1531e",
    HexType.MOUNTAIN: "#8d99ae",
    HexType.DESERT:   "#e9d8a6",
}

LEGEND_LABELS: dict[HexType, str] = {
    HexType.FOREST:   "Forest (Wood)",
    HexType.PASTURE:  "Pasture (Sheep)",
    HexType.FIELD:    "Field (Wheat)",
    HexType.HILL:     "Hill (Brick)",
    HexType.MOUNTAIN: "Mountain (Ore)",
    HexType.DESERT:   "Desert",
}

# Port colours and labels
_PORT_COLOR: dict[str, str] = {
    "generic": "#ffffff",
    "lumber":  "#2d6a4f",
    "wool":    "#74c69d",
    "grain":   "#ffd166",
    "brick":   "#c1531e",
    "ore":     "#8d99ae",
}
_PORT_LABEL: dict[str, str] = {
    "generic": "3:1",
    "lumber":  "2:1\nWood",
    "wool":    "2:1\nSheep",
    "grain":   "2:1\nWheat",
    "brick":   "2:1\nBrick",
    "ore":     "2:1\nOre",
}

# ── player colours ─────────────────────────────────────────────────────────────
PLAYER_COLORS: list[dict[str, str]] = [
    {"name": "Red",    "fill": "#e63946", "outline": "#9b0000", "text": "white"},
    {"name": "Blue",   "fill": "#4361ee", "outline": "#1a2a8c", "text": "white"},
    {"name": "Orange", "fill": "#fb8500", "outline": "#a65200", "text": "white"},
    {"name": "White",  "fill": "#f0f0f0", "outline": "#555555", "text": "black"},
]

# ── layout/sizing ──────────────────────────────────────────────────────────────
SIDEBAR_W     = 185
OCEAN_COLOR   = "#1a6fb0"
_FALLBACK_R   = 62.0  # used when images are unavailable


class BoardBuilderApp(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Catan Board Builder")
        self.resizable(True, True)

        self.board = Board()
        self._photo_refs: list = []
        self._hex_photos: dict[HexType, Optional[object]] = {}
        self._raw_images: dict[HexType, object] = {}   # original PIL images for rescaling
        self._board_radius: float = _FALLBACK_R
        self._zoom: float = 1.0
        self._raw_robber: object = None      # raw PIL image for the robber sprite
        self._robber_photo: object = None    # current PhotoImage for robber
        self._robber_canvas_id: int = 0      # canvas item id for the robber sprite
        self._robber_anim_job: object = None # pending after() job

        # ── piece state  (key → player index) ─────────────────────────────────
        self.settlements: dict[tuple, int] = {}
        self.cities:      dict[tuple, int] = {}
        self.roads:       dict[tuple, int] = {}

        # ── snap-point geometry (populated in _compute_snap_points) ───────────
        self._vertices: dict[tuple, tuple[float, float]] = {}
        self._edges:    dict[tuple, dict] = {}

        # ── UI state ───────────────────────────────────────────────────────────
        self.mode           = tk.StringVar(value="settlement")
        self.current_player = tk.IntVar(value=0)
        self._hover_key: Optional[tuple] = None

        # ── game mode state ────────────────────────────────────────────────────
        self._game_engine: Optional[object] = None
        self._game_state:  Optional[object] = None
        self._vkey_to_vid: dict = {}
        self._vid_to_vkey: dict = {}
        self._ekey_to_eid: dict = {}
        self._eid_to_ekey: dict = {}
        self._num_players  = tk.IntVar(value=4)
        self._status_var   = tk.StringVar(value="Press 'Start Game'\nto begin setup.")
        self._roll_btn:          Optional[object] = None
        self._end_turn_btn:      Optional[object] = None
        self._trade_btn:         Optional[object] = None
        self._player_trade_btn:  Optional[object] = None
        self._buy_dev_btn:        Optional[object] = None
        self._trade_overlay: Optional[object] = None   # in-canvas trade panel
        self._res_labels:   dict = {}   # resource name string → tk.Label

        self._build_ui()
        self._load_hex_images()
        # Size window to fit board on screen
        vw, vh = self._board_dimensions()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        win_w = min(vw + SIDEBAR_W + 20, sw - 80)
        win_h = min(vh + 20, sh - 80)
        self.geometry(f"{win_w}x{win_h}")
        self._draw_board()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Canvas + scrollbars in a sub-frame
        canvas_frame = tk.Frame(self, bg=OCEAN_COLOR)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        vbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        hbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        vbar.pack(side=tk.RIGHT,  fill=tk.Y)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas = tk.Canvas(
            canvas_frame, bg=OCEAN_COLOR,
            xscrollcommand=hbar.set, yscrollcommand=vbar.set,
            highlightthickness=0, cursor="crosshair",
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        hbar.config(command=self.canvas.xview)
        vbar.config(command=self.canvas.yview)

        self.canvas.bind("<Button-1>",  self._on_click)
        self.canvas.bind("<Button-3>",  self._on_right_click)
        self.canvas.bind("<Motion>",    self._on_hover)
        self.canvas.bind("<Leave>",     self._on_leave)
        # Mouse-wheel scroll
        self.canvas.bind("<MouseWheel>",       self._on_wheel_y)
        self.canvas.bind("<Shift-MouseWheel>", self._on_wheel_x)

        # ── Scrollable sidebar shell ────────────────────────────────────────────
        _sb_outer = tk.Frame(self, width=SIDEBAR_W, bg="#f0f0f0")
        _sb_outer.pack(side=tk.RIGHT, fill=tk.Y)
        _sb_outer.pack_propagate(False)

        _sb_vscroll = tk.Scrollbar(_sb_outer, orient=tk.VERTICAL)
        _sb_vscroll.pack(side=tk.RIGHT, fill=tk.Y)

        _sb_canvas = tk.Canvas(_sb_outer, bg="#f0f0f0", highlightthickness=0,
                               yscrollcommand=_sb_vscroll.set)
        _sb_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _sb_vscroll.config(command=_sb_canvas.yview)

        sidebar = tk.Frame(_sb_canvas, padx=10, pady=10, bg="#f0f0f0")
        _sb_win = _sb_canvas.create_window((0, 0), window=sidebar, anchor="nw")

        def _sb_on_configure(e):
            _sb_canvas.configure(scrollregion=_sb_canvas.bbox("all"))
        def _sb_on_canvas_resize(e):
            _sb_canvas.itemconfig(_sb_win, width=e.width)
        sidebar.bind("<Configure>", _sb_on_configure)
        _sb_canvas.bind("<Configure>", _sb_on_canvas_resize)

        def _sb_on_wheel(e):
            _sb_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        sidebar.bind_all("<MouseWheel>", lambda e: (
            _sb_on_wheel(e) if e.widget.winfo_toplevel() == self and
            str(e.widget).startswith(str(_sb_outer)) else None
        ))

        tk.Label(sidebar, text="Catan Board Builder",
                 font=("Arial", 13, "bold"), bg="#f0f0f0").pack(pady=(0, 10))

        # ── Place pieces ───────────────────────────────────────────────────────
        self._section(sidebar, "Place Pieces")

        for label, value in [("Settlement", "settlement"), ("Road", "road"), ("City", "city")]:
            tk.Radiobutton(sidebar, text=label, variable=self.mode, value=value,
                           bg="#f0f0f0", font=("Arial", 10)).pack(anchor="w")

        tk.Label(sidebar, text="Player", font=("Arial", 10, "bold"),
                 bg="#f0f0f0").pack(anchor="w", pady=(8, 2))

        pf = tk.Frame(sidebar, bg="#f0f0f0")
        pf.pack(fill=tk.X)
        pf.columnconfigure(0, weight=1)
        pf.columnconfigure(1, weight=1)
        self._player_btns: list[tk.Button] = []
        for i, p in enumerate(PLAYER_COLORS):
            btn = tk.Button(
                pf, text=p["name"], bg=p["fill"], fg=p["text"],
                font=("Arial", 9, "bold"), pady=2,
                relief=tk.SUNKEN if i == 0 else tk.RAISED,
                command=lambda idx=i: self._select_player(idx),
            )
            btn.grid(row=i // 2, column=i % 2, padx=2, pady=2, sticky="ew")
            self._player_btns.append(btn)

        tk.Label(sidebar, text="Right-click to remove",
                 font=("Arial", 8), fg="#777", bg="#f0f0f0").pack(anchor="w", pady=(4, 0))
        tk.Button(sidebar, text="🗑  Clear Pieces",
                  command=self._clear_pieces, font=("Arial", 10), pady=3
                  ).pack(fill=tk.X, pady=(4, 10))

        # ── Game setup ─────────────────────────────────────────────────────────
        self._section(sidebar, "Game")
        pf2 = tk.Frame(sidebar, bg="#f0f0f0")
        pf2.pack(fill=tk.X, pady=(0, 4))
        tk.Label(pf2, text="Players:", font=("Arial", 9),
                 bg="#f0f0f0").pack(side=tk.LEFT)
        tk.Spinbox(pf2, from_=2, to=4, textvariable=self._num_players,
                   width=3, font=("Arial", 9)).pack(side=tk.LEFT, padx=(4, 0))
        tk.Button(sidebar, text="▶  Start Game",
                  command=self._start_game, font=("Arial", 10), pady=3
                  ).pack(fill=tk.X, pady=(0, 4))
        tk.Label(sidebar, textvariable=self._status_var,
                 font=("Arial", 8), bg="#e8f4f8", fg="#1a3a5c",
                 relief=tk.SOLID, bd=1, padx=4, pady=4,
                 wraplength=155, justify=tk.LEFT).pack(fill=tk.X, pady=(0, 6))
        self._roll_btn = tk.Button(sidebar, text="🎲  Roll Dice",
                  command=self._do_roll_action, font=("Arial", 10), pady=3,
                  state=tk.DISABLED)
        self._roll_btn.pack(fill=tk.X, pady=(0, 4))
        self._end_turn_btn = tk.Button(sidebar, text="⏭  End Turn",
                  command=self._do_end_turn_action, font=("Arial", 10), pady=3,
                  state=tk.DISABLED)
        self._end_turn_btn.pack(fill=tk.X, pady=(0, 4))
        self._trade_btn = tk.Button(sidebar, text="🔄  Maritime Trade",
                  command=self._show_trade_dialog, font=("Arial", 10), pady=3,
                  state=tk.DISABLED)
        self._trade_btn.pack(fill=tk.X, pady=(0, 4))
        self._player_trade_btn = tk.Button(sidebar, text="🤝  Player Trade",
                  command=self._show_player_trade_dialog, font=("Arial", 10), pady=3,
                  state=tk.DISABLED)
        self._player_trade_btn.pack(fill=tk.X, pady=(0, 4))
        self._buy_dev_btn = tk.Button(sidebar, text="🃏  Buy Dev Card  (Ore+Wool+Grain)",
                  command=self._do_buy_dev_card, font=("Arial", 10), pady=3,
                  state=tk.DISABLED)
        self._buy_dev_btn.pack(fill=tk.X, pady=(0, 8))

        # ── Hand ───────────────────────────────────────────────────────────────
        self._section(sidebar, "Hand")
        for _rname in ["Lumber", "Brick", "Wool", "Grain", "Ore"]:
            _rf = tk.Frame(sidebar, bg="#f0f0f0")
            _rf.pack(fill=tk.X, pady=1)
            tk.Label(_rf, text=f"{_rname}:", font=("Arial", 9),
                     bg="#f0f0f0", width=7, anchor="w").pack(side=tk.LEFT)
            _lbl = tk.Label(_rf, text="0", font=("Arial", 9, "bold"),
                            bg="#f0f0f0", anchor="w")
            _lbl.pack(side=tk.LEFT)
            self._res_labels[_rname.lower()] = _lbl

        # ── Dev Cards in Hand ──────────────────────────────────────────────
        self._section(sidebar, "Dev Cards")
        self._dev_cards_frame = tk.Frame(sidebar, bg="#f0f0f0")
        self._dev_cards_frame.pack(fill=tk.X, pady=(0, 6))
        # Deck count label
        _dcf = tk.Frame(sidebar, bg="#f0f0f0")
        _dcf.pack(fill=tk.X, pady=(0, 6))
        tk.Label(_dcf, text="Deck:", font=("Arial", 9), bg="#f0f0f0",
                 width=7, anchor="w").pack(side=tk.LEFT)
        self._deck_count_lbl = tk.Label(_dcf, text="25", font=("Arial", 9, "bold"),
                                        bg="#f0f0f0", anchor="w")
        self._deck_count_lbl.pack(side=tk.LEFT)

        # ── Board ──────────────────────────────────────────────────────────────
        self._section(sidebar, "Board")
        tk.Button(sidebar, text="🎲  Randomize Board",
                  command=self._randomize, font=("Arial", 10), pady=3
                  ).pack(fill=tk.X, pady=(0, 4))
        tk.Button(sidebar, text="💾  Export as PNG",
                  command=self._export, font=("Arial", 10), pady=3
                  ).pack(fill=tk.X, pady=(0, 6))

        # ── Zoom ───────────────────────────────────────────────────────────────
        self._section(sidebar, "Zoom")
        zf = tk.Frame(sidebar, bg="#f0f0f0")
        zf.pack(fill=tk.X, pady=(0, 8))
        tk.Button(zf, text="−", width=3, font=("Arial", 12, "bold"),
                  command=lambda: self._set_zoom(self._zoom - 0.25)
                  ).pack(side=tk.LEFT)
        self._zoom_label = tk.Label(zf, text="100%", font=("Arial", 10),
                                    bg="#f0f0f0", width=5)
        self._zoom_label.pack(side=tk.LEFT, expand=True)
        tk.Button(zf, text="+", width=3, font=("Arial", 12, "bold"),
                  command=lambda: self._set_zoom(self._zoom + 0.25)
                  ).pack(side=tk.LEFT)

        # ── Legend ─────────────────────────────────────────────────────────────
        self._section(sidebar, "Resources")
        for htype, color in HEX_COLORS.items():
            rf = tk.Frame(sidebar, bg="#f0f0f0")
            rf.pack(anchor="w", pady=1)
            tk.Canvas(rf, width=13, height=13, bg=color,
                      highlightthickness=1, highlightbackground="#555"
                      ).pack(side=tk.LEFT, padx=(0, 5))
            tk.Label(rf, text=LEGEND_LABELS[htype],
                     font=("Arial", 8), bg="#f0f0f0").pack(side=tk.LEFT)

        if not PIL_AVAILABLE:
            tk.Label(sidebar, text="⚠ pip install Pillow\nfor image rendering",
                     font=("Arial", 8), bg="#fff3cd", fg="#856404",
                     relief=tk.SOLID, bd=1, padx=4, pady=3).pack(pady=(10, 0))

    @staticmethod
    def _section(parent: tk.Frame, text: str) -> None:
        tk.Label(parent, text=text, font=("Arial", 10, "bold"),
                 bg="#f0f0f0").pack(anchor="w")
        tk.Frame(parent, bg="#aaaaaa", height=1).pack(fill=tk.X, pady=(1, 6))

    # ── image loading ──────────────────────────────────────────────────────────

    def _load_hex_images(self) -> None:
        self._board_radius = _FALLBACK_R
        self._raw_images   = {}
        if not PIL_AVAILABLE:
            self._hex_photos = {t: None for t in HexType}
            return
        for htype in HexType:
            path = HEX_IMG_DIR / f"{htype.value}.png"
            if path.exists():
                self._raw_images[htype] = Image.open(path).convert("RGBA")
            else:
                self._raw_images[htype] = None
        # load robber sprite
        robber_path = IMAGES_DIR / "robber" / "vector" / "robber.png"
        if robber_path.exists():
            self._raw_robber = Image.open(robber_path).convert("RGBA")
        self._rescale_images()

    def _rescale_images(self) -> None:
        """Rebuild PhotoImage objects at the current zoomed radius."""
        if not PIL_AVAILABLE:
            return
        R     = _FALLBACK_R * self._zoom
        img_w = max(1, int(math.sqrt(3) * R))
        img_h = max(1, int(2 * R))
        self._photo_refs = []
        for htype in HexType:
            raw = self._raw_images.get(htype)
            if raw is not None:
                photo = ImageTk.PhotoImage(
                    raw.resize((img_w, img_h), Image.LANCZOS))
                self._hex_photos[htype] = photo
                self._photo_refs.append(photo)
            else:
                self._hex_photos[htype] = None
        # rescale robber sprite to ~40% of hex height
        if self._raw_robber is not None:
            rh = max(8, int(img_h * 0.385))
            raw_w, raw_h = self._raw_robber.size
            rw = max(1, int(rh * raw_w / raw_h))
            self._robber_photo = ImageTk.PhotoImage(
                self._raw_robber.resize((rw, rh), Image.LANCZOS))
            self._photo_refs.append(self._robber_photo)

    # ── geometry ───────────────────────────────────────────────────────────────

    def _board_dimensions(self) -> tuple[int, int]:
        """Virtual canvas size that fits the board with padding."""
        R      = self._board_radius
        margin = R / 2
        vw = int(5 * math.sqrt(3) * R + 2 * margin)
        vh = int(8 * R + 2 * margin)
        return vw, vh

    def _current_radius(self) -> float:
        return _FALLBACK_R * self._zoom

    def _hex_center(self, row: int, col: int) -> tuple[float, float]:
        R  = self._board_radius
        hw = math.sqrt(3) * R
        vw, vh = self._board_dimensions()
        vcx = vw / 2
        vcy = vh / 2
        n   = ROW_SIZES[row]
        return vcx + hw * (col - (n - 1) / 2), vcy + (row - 2) * 1.5 * R

    def _hex_flat_pts(self, cx: float, cy: float) -> list[float]:
        R = self._current_radius()
        pts: list[float] = []
        for i in range(6):
            a = math.radians(60 * i - 30)
            pts.extend([cx + R * math.cos(a), cy + R * math.sin(a)])
        return pts

    def _hex_vertices(self, cx: float, cy: float) -> list[tuple[float, float]]:
        R = self._current_radius()
        return [(cx + R * math.cos(math.radians(60 * i - 30)),
                 cy + R * math.sin(math.radians(60 * i - 30))) for i in range(6)]

    @staticmethod
    def _vkey(x: float, y: float) -> tuple[int, int]:
        return (round(x), round(y))

    def _compute_snap_points(self) -> None:
        verts: dict[tuple, tuple[float, float]] = {}
        edges: dict[tuple, dict]               = {}
        for h in self.board.hexes:
            cx, cy  = self._hex_center(h.row, h.col)
            hverts  = self._hex_vertices(cx, cy)
            vkeys   = [self._vkey(*v) for v in hverts]
            for key, pt in zip(vkeys, hverts):
                verts[key] = pt
            for i in range(6):
                k1, k2   = vkeys[i], vkeys[(i + 1) % 6]
                ekey     = tuple(sorted([k1, k2]))
                if ekey not in edges:
                    x1, y1 = verts[k1]
                    x2, y2 = verts[k2]
                    edges[ekey] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                   "mx": (x1 + x2) / 2, "my": (y1 + y2) / 2}
        self._vertices = verts
        self._edges    = edges

        # ── topology ID mappings (pixel key ↔ board topology ID) ──────────────
        vkey_to_vid: dict = {}
        vid_counter = 0
        for h in self.board.hexes:
            cx, cy = self._hex_center(h.row, h.col)
            for pt in self._hex_vertices(cx, cy):
                key = self._vkey(*pt)
                if key not in vkey_to_vid:
                    vkey_to_vid[key] = vid_counter
                    vid_counter += 1
        self._vkey_to_vid = vkey_to_vid
        self._vid_to_vkey = {v: k for k, v in vkey_to_vid.items()}

        ekey_to_eid: dict = {}
        eid_to_ekey: dict = {}
        seen_topo:   dict = {}
        eid_counter = 0
        for h in self.board.hexes:
            cx, cy = self._hex_center(h.row, h.col)
            vkeys  = [self._vkey(*pt) for pt in self._hex_vertices(cx, cy)]
            vids   = [vkey_to_vid[k] for k in vkeys]
            for i in range(6):
                v1, v2   = vids[i], vids[(i + 1) % 6]
                k1, k2   = vkeys[i], vkeys[(i + 1) % 6]
                topo_key = (min(v1, v2), max(v1, v2))
                pkey     = tuple(sorted([k1, k2]))
                if topo_key not in seen_topo:
                    seen_topo[topo_key]    = eid_counter
                    ekey_to_eid[pkey]      = eid_counter
                    eid_to_ekey[eid_counter] = pkey
                    eid_counter += 1
        self._ekey_to_eid = ekey_to_eid
        self._eid_to_ekey = eid_to_ekey

    # ── snap hit-testing ───────────────────────────────────────────────────────

    def _nearest_vertex(self, x: float, y: float) -> Optional[tuple]:
        snap = self._current_radius() * 0.30
        best_key, best_d = None, snap
        for key, (vx, vy) in self._vertices.items():
            d = math.hypot(x - vx, y - vy)
            if d < best_d:
                best_d, best_key = d, key
        return best_key

    def _nearest_edge(self, x: float, y: float) -> Optional[tuple]:
        snap = self._current_radius() * 0.30
        best_key, best_d = None, snap
        for key, info in self._edges.items():
            d = math.hypot(x - info["mx"], y - info["my"])
            if d < best_d:
                best_d, best_key = d, key
        return best_key

    # ── drawing ────────────────────────────────────────────────────────────────

    def _draw_board(self) -> None:
        vw, vh = self._board_dimensions()
        self.canvas.config(scrollregion=(0, 0, vw, vh))
        self.canvas.delete("all")
        for h in self.board.hexes:
            cx, cy = self._hex_center(h.row, h.col)
            self.canvas.create_polygon(
                self._hex_flat_pts(cx, cy),
                fill=HEX_COLORS[h.hex_type], outline="#3a1f00", width=2, tags="hex",
            )
            photo = self._hex_photos.get(h.hex_type)
            if photo:
                self.canvas.create_image(cx, cy, image=photo, anchor=tk.CENTER, tags="hex")
            if h.token is not None:
                self._draw_token(cx, cy, h.token)
        self._compute_snap_points()
        self._draw_ports()
        self._draw_pieces()
        self._draw_robber()

    def _draw_ports(self) -> None:
        """Draw port indicators on coastal edges."""
        R = self._current_radius()
        for port in self.board.ports:
            cx, cy = self._hex_center(port.hex_row, port.hex_col)
            verts  = self._hex_vertices(cx, cy)
            v1     = verts[port.edge_idx]
            v2     = verts[(port.edge_idx + 1) % 6]
            mx     = (v1[0] + v2[0]) / 2
            my     = (v1[1] + v2[1]) / 2
            # unit vector pointing outward from hex center
            dx, dy = mx - cx, my - cy
            dist   = math.hypot(dx, dy) or 1
            dx /= dist; dy /= dist
            offset = R * 0.40
            px = mx + dx * offset
            py = my + dy * offset
            ptype = port.port_type.value
            color = _PORT_COLOR.get(ptype, "#eee")
            label = _PORT_LABEL.get(ptype, "?")
            # lines from indicator to edge endpoints
            self.canvas.create_line(px, py, v1[0], v1[1],
                                    fill="#333", width=1, tags="hex")
            self.canvas.create_line(px, py, v2[0], v2[1],
                                    fill="#333", width=1, tags="hex")
            # indicator box
            bw = R * 0.26
            self.canvas.create_rectangle(
                px - bw, py - bw, px + bw, py + bw,
                fill=color, outline="#333", width=1, tags="hex"
            )
            fs = max(5, int(R * 0.14))
            self.canvas.create_text(
                px, py, text=label,
                font=("Arial", fs, "bold"), fill="black", tags="hex"
            )
            # small rings on the two port vertices so players know where to settle
            vr = R * 0.13
            for vx, vy in (v1, v2):
                self.canvas.create_oval(
                    vx - vr, vy - vr, vx + vr, vy + vr,
                    outline=color, fill="", width=2, tags="hex"
                )

    def _draw_robber(self, cx: Optional[float] = None, cy: Optional[float] = None) -> None:
        """Place/move the robber canvas item to (cx, cy). Uses current game state if not given."""
        if self._game_state is None:
            return
        if cx is None or cy is None:
            hidx = self._game_state.robber_hex
            cx, cy = self._hex_center(self.board.hexes[hidx].row,
                                       self.board.hexes[hidx].col)
            # offset slightly so it doesn't cover the token circle
            cy += self._current_radius() * 0.28
        if self._robber_photo is not None:
            if self._robber_canvas_id and self.canvas.find_withtag("robber"):
                self.canvas.coords("robber", cx, cy)
            else:
                self._robber_canvas_id = self.canvas.create_image(
                    cx, cy, image=self._robber_photo, anchor=tk.CENTER, tags="robber")
        else:
            # Fallback: a dark circle
            r = self._current_radius() * 0.18
            if self._robber_canvas_id and self.canvas.find_withtag("robber"):
                self.canvas.coords("robber", cx - r, cy - r, cx + r, cy + r)
            else:
                self._robber_canvas_id = self.canvas.create_oval(
                    cx - r, cy - r, cx + r, cy + r,
                    fill="#1a1a1a", outline="#555", width=2, tags="robber")

    def _animate_robber(self, from_hidx: int, to_hidx: int,
                        on_done: Optional[object] = None) -> None:
        """Slide the robber sprite from one hex center to another over ~300 ms."""
        STEPS = 15
        DELAY = 20  # ms per frame

        fh = self.board.hexes[from_hidx]
        th = self.board.hexes[to_hidx]
        fx, fy = self._hex_center(fh.row, fh.col)
        tx, ty = self._hex_center(th.row, th.col)
        off_y  = self._current_radius() * 0.28
        fy += off_y
        ty += off_y

        step_container = [0]

        def _tick():
            t = step_container[0] / STEPS
            # ease-in-out
            t_ease = t * t * (3 - 2 * t)
            cx_ = fx + (tx - fx) * t_ease
            cy_ = fy + (ty - fy) * t_ease
            self._draw_robber(cx_, cy_)
            step_container[0] += 1
            if step_container[0] <= STEPS:
                self._robber_anim_job = self.after(DELAY, _tick)
            else:
                self._robber_anim_job = None
                self._draw_robber()   # snap to final position
                if on_done:
                    on_done()

        # cancel any in-progress animation
        if self._robber_anim_job:
            self.after_cancel(self._robber_anim_job)
            self._robber_anim_job = None
        _tick()

    def _draw_token(self, cx: float, cy: float, token: int) -> None:
        tr = self._current_radius() * 0.28
        self.canvas.create_oval(cx - tr, cy - tr, cx + tr, cy + tr,
                                fill="#f5e6c8", outline="#888888", width=1, tags="hex")
        self.canvas.create_text(cx, cy, text=str(token),
                                font=("Arial", max(8, int(tr * 0.95)), "bold"),
                                fill="red" if token in (6, 8) else "black", tags="hex")

    def _draw_pieces(self) -> None:
        """Redraw all roads, settlements and cities (always on top of tiles)."""
        self.canvas.delete("piece")
        # Roads first so settlements sit on top at intersections
        for ekey, pidx in self.roads.items():
            info = self._edges.get(ekey)
            if info:
                self._paint_road(info["x1"], info["y1"], info["x2"], info["y2"], pidx, "piece")
        for vkey, pidx in self.settlements.items():
            vx, vy = self._vertices.get(vkey, (0.0, 0.0))
            self._paint_settlement(vx, vy, pidx, "piece")
        for vkey, pidx in self.cities.items():
            vx, vy = self._vertices.get(vkey, (0.0, 0.0))
            self._paint_city(vx, vy, pidx, "piece")

    # ── piece painters ─────────────────────────────────────────────────────────

    def _paint_settlement(self, cx: float, cy: float, pidx: int, tag: str) -> None:
        p, s = PLAYER_COLORS[pidx], self._current_radius() * 0.21
        # Pentagon house: square base + triangle roof
        pts = [
            cx - s, cy + s,       # bottom-left
            cx + s, cy + s,       # bottom-right
            cx + s, cy,           # right shoulder
            cx,     cy - s * 1.1, # roof peak
            cx - s, cy,           # left shoulder
        ]
        self.canvas.create_polygon(pts, fill=p["fill"], outline=p["outline"],
                                   width=2, tags=tag)

    def _paint_city(self, cx: float, cy: float, pidx: int, tag: str) -> None:
        p, s = PLAYER_COLORS[pidx], self._current_radius() * 0.21 * 1.45
        # Blocky two-tower city silhouette
        pts = [
            cx - s,         cy + s,
            cx + s,         cy + s,
            cx + s,         cy + s * 0.1,
            cx + s * 0.35,  cy + s * 0.1,
            cx + s * 0.35,  cy - s * 0.95,
            cx - s * 0.35,  cy - s * 0.95,
            cx - s * 0.35,  cy + s * 0.1,
            cx - s,         cy + s * 0.1,
        ]
        self.canvas.create_polygon(pts, fill=p["fill"], outline=p["outline"],
                                   width=2, tags=tag)

    def _paint_road(self, x1: float, y1: float, x2: float, y2: float,
                    pidx: int, tag: str) -> None:
        p  = PLAYER_COLORS[pidx]
        rw = max(2, int(self._current_radius() * 0.13))
        self.canvas.create_line(x1, y1, x2, y2, fill=p["outline"],
                                width=rw + 3, capstyle=tk.ROUND, tags=tag)
        self.canvas.create_line(x1, y1, x2, y2, fill=p["fill"],
                                width=rw,     capstyle=tk.ROUND, tags=tag)

    # ── hover ghost ────────────────────────────────────────────────────────────

    def _on_hover(self, event: tk.Event) -> None:
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        mode = self.mode.get()
        key  = (self._nearest_vertex(cx, cy)
                if mode in ("settlement", "city")
                else self._nearest_edge(cx, cy))
        if key == self._hover_key:
            return
        self._hover_key = key
        self.canvas.delete("hover")
        if key is None:
            return
        pidx = self.current_player.get()
        p    = PLAYER_COLORS[pidx]
        if mode == "settlement":
            vx, vy = self._vertices[key]
            s = self._current_radius() * 0.21
            pts = [vx - s, vy + s, vx + s, vy + s, vx + s, vy,
                   vx, vy - s * 1.1, vx - s, vy]
            self.canvas.create_polygon(pts, fill="", outline=p["fill"],
                                       width=2, dash=(5, 3), tags="hover")
        elif mode == "city":
            vx, vy = self._vertices[key]
            s = self._current_radius() * 0.21 * 1.45
            pts = [vx - s, vy + s, vx + s, vy + s, vx + s, vy + s * 0.1,
                   vx + s * 0.35, vy + s * 0.1, vx + s * 0.35, vy - s * 0.95,
                   vx - s * 0.35, vy - s * 0.95, vx - s * 0.35, vy + s * 0.1,
                   vx - s, vy + s * 0.1]
            self.canvas.create_polygon(pts, fill="", outline=p["fill"],
                                       width=2, dash=(5, 3), tags="hover")
        else:
            info = self._edges[key]
            rw   = max(2, int(self._current_radius() * 0.13))
            self.canvas.create_line(info["x1"], info["y1"], info["x2"], info["y2"],
                                    fill=p["fill"], width=rw + 1,
                                    dash=(6, 4), capstyle=tk.ROUND, tags="hover")

    def _on_leave(self, _: tk.Event) -> None:
        self._hover_key = None
        self.canvas.delete("hover")

    def _on_wheel_y(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def _on_wheel_x(self, event: tk.Event) -> None:
        self.canvas.xview_scroll(int(-event.delta / 120), "units")

    # ── click handlers ─────────────────────────────────────────────────────────

    def _on_click(self, event: tk.Event) -> None:
        if self._game_state is not None:
            self._on_game_click(event)
            return
        mode  = self.mode.get()
        pidx  = self.current_player.get()
        cx    = self.canvas.canvasx(event.x)
        cy    = self.canvas.canvasy(event.y)
        if mode in ("settlement", "city"):
            key = self._nearest_vertex(cx, cy)
            if key is None:
                return
            if mode == "settlement":
                # Don't overwrite a city
                if key not in self.cities:
                    self.settlements[key] = pidx
            else:
                # City can upgrade an existing settlement or be placed freely
                self.settlements.pop(key, None)
                self.cities[key] = pidx
        else:
            key = self._nearest_edge(cx, cy)
            if key is None:
                return
            self.roads[key] = pidx
        self._draw_pieces()

    def _on_right_click(self, event: tk.Event) -> None:
        cx  = self.canvas.canvasx(event.x)
        cy  = self.canvas.canvasy(event.y)
        key = self._nearest_vertex(cx, cy)
        if key and (key in self.settlements or key in self.cities):
            self.settlements.pop(key, None)
            self.cities.pop(key, None)
            self._draw_pieces()
            return
        key = self._nearest_edge(cx, cy)
        if key and key in self.roads:
            del self.roads[key]
            self._draw_pieces()

    # ── game mode ──────────────────────────────────────────────────────────────

    def _start_game(self) -> None:
        """Initialise a GameState and begin the snake-draft setup phase."""
        if not GAME_ENGINE_AVAILABLE:
            messagebox.showerror("Error",
                "game_engine.py or game_state.py not found next to board_builder.py.")
            return
        n = self._num_players.get()
        self._game_engine = GameEngine()
        self._game_state  = self._game_engine.new_game(num_players=n)
        # Sync the rendered board to the game-state board so hex indices match
        self.board = self._game_state.board
        self.settlements.clear()
        self.cities.clear()
        self.roads.clear()
        self._draw_board()   # also rebuilds snap-point topology maps
        self._update_status()

    def _on_game_click(self, event: tk.Event) -> None:
        """Handle a canvas click while a game is active."""
        state  = self._game_state
        engine = self._game_engine
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        action = None

        if state.phase == Phase.SETUP:
            if state.setup_step == 0:           # place settlement
                key = self._nearest_vertex(cx, cy)
                if key is None:
                    return
                vid = self._vkey_to_vid.get(key)
                if vid is None:
                    return
                legal_vids = {
                    a.vertex_id for a in engine.legal_actions(state)
                    if a.type == ActionType.PLACE_SETTLEMENT
                }
                if vid not in legal_vids:
                    return
                action = Action(ActionType.PLACE_SETTLEMENT, vertex_id=vid)
            else:                               # place road
                key = self._nearest_edge(cx, cy)
                if key is None:
                    return
                eid = self._ekey_to_eid.get(key)
                if eid is None:
                    return
                legal_eids = {
                    a.edge_id for a in engine.legal_actions(state)
                    if a.type == ActionType.PLACE_ROAD
                }
                if eid not in legal_eids:
                    return
                action = Action(ActionType.PLACE_ROAD, edge_id=eid)

        elif state.phase == Phase.MAIN:
            mode = self.mode.get()
            pid  = state.current_player
            if mode == "road":
                key = self._nearest_edge(cx, cy)
                if key is None:
                    return
                eid = self._ekey_to_eid.get(key)
                if eid is None:
                    return
                legal_eids = {
                    a.edge_id for a in engine.legal_actions(state)
                    if a.type == ActionType.PLACE_ROAD
                }
                if eid not in legal_eids:
                    return
                action = Action(ActionType.PLACE_ROAD, edge_id=eid)
            elif mode == "settlement":
                key = self._nearest_vertex(cx, cy)
                if key is None:
                    return
                vid = self._vkey_to_vid.get(key)
                if vid is None:
                    return
                legal_vids = {
                    a.vertex_id for a in engine.legal_actions(state)
                    if a.type == ActionType.PLACE_SETTLEMENT
                }
                if vid not in legal_vids:
                    return
                action = Action(ActionType.PLACE_SETTLEMENT, vertex_id=vid)
            elif mode == "city":
                key = self._nearest_vertex(cx, cy)
                if key is None:
                    return
                vid = self._vkey_to_vid.get(key)
                if vid is None:
                    return
                legal_vids = {
                    a.vertex_id for a in engine.legal_actions(state)
                    if a.type == ActionType.UPGRADE_CITY
                }
                if vid not in legal_vids:
                    return
                action = Action(ActionType.UPGRADE_CITY, vertex_id=vid)

        elif state.phase == Phase.ROBBER:
            hidx = self._nearest_hex(cx, cy)
            if hidx is None:
                return
            legal = [a for a in engine.legal_actions(state)
                     if a.type == ActionType.MOVE_ROBBER and a.hex_id == hidx]
            if not legal:
                return
            old_hidx = state.robber_hex
            victims = [a.steal_from for a in legal if a.steal_from is not None]

            def _apply_robber(steal_from):
                act = Action(ActionType.MOVE_ROBBER, hex_id=hidx, steal_from=steal_from)
                self._game_state, _, _ = engine.step(state, act)
                self._sync_pieces_from_state()
                self._update_status()
                self._animate_robber(old_hidx, hidx)

            if not victims:
                _apply_robber(None)
            elif len(victims) == 1:
                _apply_robber(victims[0])
            else:
                self._show_steal_dialog(victims, _apply_robber)
            return

        if action is None:
            return
        self._game_state, _, _ = engine.step(state, action)
        self._sync_pieces_from_state()
        self._update_status()

    def _sync_pieces_from_state(self) -> None:
        """Rebuild the piece dicts from the current GameState."""
        state = self._game_state
        self.settlements.clear()
        self.cities.clear()
        self.roads.clear()
        for vid in range(state.topology.num_vertices):
            owner = state.vertex_owner[vid]
            if owner == -1:
                continue
            key = self._vid_to_vkey.get(vid)
            if key is None:
                continue
            if state.vertex_building[vid] == 1:
                self.settlements[key] = owner
            else:
                self.cities[key] = owner
        for eid in range(state.topology.num_edges):
            owner = state.edge_owner[eid]
            if owner == -1:
                continue
            key = self._eid_to_ekey.get(eid)
            if key is None:
                continue
            self.roads[key] = owner
        self._draw_pieces()

    def _update_status(self) -> None:
        """Refresh the status label, player highlight, buttons, and resource display."""
        state = self._game_state
        if state is None:
            self._status_var.set("Press 'Start Game'\nto begin setup.")
            self._update_action_buttons()
            self._update_resources_display()
            return
        pid   = state.current_player
        pname = PLAYER_COLORS[pid]["name"]
        self._select_player(pid)
        if state.phase == Phase.SETUP:
            n      = len(state.players)
            turn   = state.setup_turn + 1
            action = "place settlement" if state.setup_step == 0 else "place road"
            self.mode.set("settlement" if state.setup_step == 0 else "road")
            self._status_var.set(f"Setup {turn}/{2 * n}\n{pname}: {action}")
        elif state.phase == Phase.ROLL:
            self._status_var.set(f"{pname}: roll the dice!")
        elif state.phase == Phase.MAIN:
            roll = state.last_roll
            self._status_var.set(f"Rolled {roll}\n{pname}: build or end turn")
        elif state.phase == Phase.ROBBER:
            self._status_var.set(f"Rolled 7!\n{pname}: click a hex\nto move robber")
        elif state.phase == Phase.DISCARD:
            needed = state.players[state.current_player].resource_count() // 2
            self._status_var.set(f"{pname}: discard {needed} card(s)")
            self.after(50, self._show_discard_dialog)
        elif state.phase == Phase.DONE:
            winner = PLAYER_COLORS[state.winner]["name"]
            self._status_var.set(f"Game over!\n{winner} wins!")
        else:
            self._status_var.set(f"{pname}'s turn\n({state.phase.value})")
        self._update_action_buttons()
        self._update_resources_display()

    # ── sidebar callbacks ──────────────────────────────────────────────────────

    def _do_roll_action(self) -> None:
        if self._game_state is None or self._game_engine is None:
            return
        if self._game_state.phase != Phase.ROLL:
            return
        old_robber = self._game_state.robber_hex
        self._game_state, _, _ = self._game_engine.step(
            self._game_state, Action(ActionType.ROLL_DICE))
        self._sync_pieces_from_state()
        self._update_status()
        # if the phase is now ROBBER we got a 7 — robber hasn't moved yet, stay put
        # if auto-moved (not the case here) we'd animate; nothing to animate on 7-roll itself

    def _show_trade_dialog(self) -> None:
        """Show an in-canvas overlay panel for maritime trading."""
        state  = self._game_state
        engine = self._game_engine
        if state is None or state.phase != Phase.MAIN or engine is None:
            return

        # Dismiss any existing overlay first
        self._hide_trade_overlay()

        pid   = state.current_player
        pname = PLAYER_COLORS[pid]["name"]
        hand  = state.players[pid].resources
        res_order = ["lumber", "brick", "wool", "grain", "ore"]

        tradeable = []
        for rname in res_order:
            r     = Resource(rname)
            ratio = engine._get_trade_ratio(state, pid, r)
            count = hand.get(r, 0)
            if count >= ratio:
                tradeable.append((rname, ratio, count))

        BG   = "#1a2e4a"
        FG   = "#ffffff"
        FONT = ("Arial", 12)
        BOLD = ("Arial", 12, "bold")
        HDR  = ("Arial", 14, "bold")

        panel = tk.Frame(self, bg=BG, bd=3, relief=tk.RIDGE)
        self._trade_overlay = panel

        tk.Label(panel, text=f"Maritime Trade — {pname}",
                 font=HDR, bg=BG, fg="#f0c040", pady=8).pack(fill=tk.X)

        if not tradeable:
            tk.Label(panel, text="Not enough resources to trade\n(need 4 of any, 3 with a 3:1 port,\nor 2 with a 2:1 port).",
                     font=FONT, bg=BG, fg=FG, padx=16, pady=8).pack()
            tk.Button(panel, text="Close", font=BOLD, padx=20, pady=6,
                      command=self._hide_trade_overlay).pack(pady=(0, 12))
            self._place_trade_overlay(panel)
            return

        # ── Give ─────────────────────────────────────────────────────────────
        tk.Label(panel, text="Give:", font=BOLD, bg=BG, fg="#aad4f5",
                 padx=16, anchor="w").pack(fill=tk.X)

        give_var = tk.StringVar(value=tradeable[0][0])
        for rname, ratio, count in tradeable:
            row = tk.Frame(panel, bg=BG)
            row.pack(fill=tk.X, padx=16, pady=2)
            tk.Radiobutton(row, text=f"{rname.capitalize()}",
                           variable=give_var, value=rname,
                           font=FONT, bg=BG, fg=FG,
                           selectcolor="#2a4e7a", activebackground=BG,
                           command=lambda: _refresh_receive()).pack(side=tk.LEFT)
            rate_label = f"{ratio}:1 port" if ratio < 4 else f"{ratio}:1 bank"
            tk.Label(row, text=f"  have {count}  ·  {rate_label}",
                     font=("Arial", 11), bg=BG, fg="#aaaaaa").pack(side=tk.LEFT)

        # ── Receive ───────────────────────────────────────────────────────────
        tk.Label(panel, text="Receive 1 of:", font=BOLD, bg=BG, fg="#aad4f5",
             padx=16, anchor="w").pack(fill=tk.X, pady=(8, 0))
        receive_var = tk.StringVar()
        recv_frame  = tk.Frame(panel, bg=BG, padx=16)
        recv_frame.pack(fill=tk.X, pady=(2, 8))
        recv_btns: dict[str, tk.Radiobutton] = {}
        for rname in res_order:
            btn = tk.Radiobutton(recv_frame, text=rname.capitalize(),
                                 variable=receive_var, value=rname,
                                 font=FONT, bg=BG, fg=FG,
                                 selectcolor="#2a4e7a", activebackground=BG)
            btn.pack(side=tk.LEFT, padx=6)
            recv_btns[rname] = btn

        def _refresh_receive():
            g = give_var.get()
            for rname, btn in recv_btns.items():
                btn.config(state=tk.DISABLED if rname == g else tk.NORMAL)
            if receive_var.get() == g:
                receive_var.set(next(r for r in res_order if r != g))

        _refresh_receive()

        err_var = tk.StringVar()
        tk.Label(panel, textvariable=err_var, fg="#ff6b6b", bg=BG,
                 font=("Arial", 11), pady=2).pack()

        def _confirm():
            give_res = Resource(give_var.get())
            recv_res = Resource(receive_var.get())
            ratio    = engine._get_trade_ratio(state, pid, give_res)
            if hand.get(give_res, 0) < ratio:
                err_var.set(f"Not enough {give_res.value} (need {ratio}).")
                return
            self._hide_trade_overlay()
            act = Action(ActionType.MARITIME_TRADE, give=give_res, receive=recv_res)
            self._game_state, _, _ = engine.step(state, act)
            self._sync_pieces_from_state()
            self._update_status()
            self._update_resources_display()
            self._update_action_buttons()

        btn_row = tk.Frame(panel, bg=BG, pady=8)
        btn_row.pack()
        tk.Button(btn_row, text="Trade", command=_confirm,
                  font=BOLD, bg="#2a7a2a", fg="white",
                  activebackground="#3a9a3a", padx=20, pady=6,
                  relief=tk.RAISED).pack(side=tk.LEFT, padx=8)
        tk.Button(btn_row, text="Cancel", command=self._hide_trade_overlay,
                  font=FONT, bg="#7a2a2a", fg="white",
                  activebackground="#9a3a3a", padx=16, pady=6,
                  relief=tk.RAISED).pack(side=tk.LEFT, padx=8)

        self._place_trade_overlay(panel)

    def _place_trade_overlay(self, panel: tk.Frame) -> None:
        """Place the overlay panel centered over the main window."""
        panel.place(in_=self, relx=0.5, rely=0.5, anchor="center")
        panel.lift()

    def _hide_trade_overlay(self) -> None:
        """Remove the trade overlay panel if it exists."""
        if self._trade_overlay is not None:
            self._trade_overlay.destroy()
            self._trade_overlay = None

    def _show_player_trade_dialog(self) -> None:
        """Overlay for the active player to build a trade offer for an opponent."""
        state  = self._game_state
        engine = self._game_engine
        if state is None or state.phase != Phase.MAIN or engine is None:
            return

        self._hide_trade_overlay()

        pid       = state.current_player
        pname     = PLAYER_COLORS[pid]["name"]
        hand      = state.players[pid].resources
        res_order = ["lumber", "brick", "wool", "grain", "ore"]
        opponents = [i for i in range(len(state.players)) if i != pid]

        BG   = "#1a2e4a"
        FG   = "#ffffff"
        FONT = ("Arial", 11)
        BOLD = ("Arial", 11, "bold")
        HDR  = ("Arial", 13, "bold")

        panel = tk.Frame(self, bg=BG, bd=3, relief=tk.RIDGE)
        self._trade_overlay = panel

        tk.Label(panel, text=f"Player Trade — {pname}",
                 font=HDR, bg=BG, fg="#f0c040", pady=6).pack(fill=tk.X)

        # ── helper: +/− row for a resource ────────────────────────────────────
        def _resource_row(parent, rname, var, max_val):
            row = tk.Frame(parent, bg=BG)
            row.pack(fill=tk.X, padx=12, pady=1)
            tk.Label(row, text=f"{rname.capitalize()}", font=FONT, bg=BG, fg=FG,
                     width=8, anchor="w").pack(side=tk.LEFT)
            tk.Button(row, text="−", font=FONT, bg="#2a4e7a", fg=FG, width=2,
                      command=lambda v=var: v.set(max(0, v.get() - 1))
                      ).pack(side=tk.LEFT)
            tk.Label(row, textvariable=var, font=BOLD, bg=BG, fg="#f0c040",
                     width=3, anchor="center").pack(side=tk.LEFT)
            tk.Button(row, text="+", font=FONT, bg="#2a4e7a", fg=FG, width=2,
                      command=lambda v=var, m=max_val: v.set(
                          v.get() + 1 if m is None else min(m, v.get() + 1))
                      ).pack(side=tk.LEFT)
            if max_val is not None:
                tk.Label(row, text=f"  (have {max_val})", font=("Arial", 9),
                         bg=BG, fg="#888").pack(side=tk.LEFT)

        # ── You Give ──────────────────────────────────────────────────────────
        tk.Label(panel, text="You give:", font=BOLD, bg=BG, fg="#aad4f5",
                 padx=12, anchor="w").pack(fill=tk.X, pady=(4, 0))
        give_vars: dict[str, tk.IntVar] = {}
        for rname in res_order:
            var = tk.IntVar(value=0)
            give_vars[rname] = var
            _resource_row(panel, rname, var, hand.get(Resource(rname), 0))

        # ── You Receive ───────────────────────────────────────────────────────
        tk.Label(panel, text="You receive:", font=BOLD, bg=BG, fg="#aad4f5",
                 padx=12, anchor="w").pack(fill=tk.X, pady=(8, 0))
        recv_vars: dict[str, tk.IntVar] = {}
        for rname in res_order:
            var = tk.IntVar(value=0)
            recv_vars[rname] = var
            _resource_row(panel, rname, var, None)   # no cap; opponent may have enough

        err_var = tk.StringVar()
        tk.Label(panel, textvariable=err_var, fg="#ff6b6b", bg=BG,
                 font=("Arial", 10), pady=2).pack()

        def _confirm_offer():
            give_dict = {Resource(r): v.get()
                         for r, v in give_vars.items() if v.get() > 0}
            recv_dict = {Resource(r): v.get()
                         for r, v in recv_vars.items() if v.get() > 0}
            if not give_dict:
                err_var.set("You must give at least 1 resource.")
                return
            if not recv_dict:
                err_var.set("You must receive at least 1 resource.")
                return
            for r, amt in give_dict.items():
                if hand.get(r, 0) < amt:
                    err_var.set(
                        f"Not enough {r.value} (have {hand.get(r,0)}, need {amt}).")
                    return
            # Send offer to all opponents in sequence
            self._show_trade_offer_response(state, pid, list(opponents),
                                            give_dict, recv_dict)

        btn_row = tk.Frame(panel, bg=BG, pady=8)
        btn_row.pack()
        tk.Button(btn_row, text="Send Offer", command=_confirm_offer,
                  font=BOLD, bg="#2a7a2a", fg="white",
                  activebackground="#3a9a3a", padx=14, pady=5,
                  relief=tk.RAISED).pack(side=tk.LEFT, padx=8)
        tk.Button(btn_row, text="Cancel", command=self._hide_trade_overlay,
                  font=FONT, bg="#7a2a2a", fg="white",
                  activebackground="#9a3a3a", padx=10, pady=5,
                  relief=tk.RAISED).pack(side=tk.LEFT, padx=8)

        self._place_trade_overlay(panel)

    def _show_trade_offer_response(
        self,
        state,
        offerer_pid:  int,
        remaining:    list,   # list of opponent pids yet to be asked
        give_dict:    dict,
        recv_dict:    dict,
    ) -> None:
        """
        Show the trade offer to remaining[0].  On decline, recurse to
        remaining[1:].  On accept (or all decline), close everything.
        """
        if not remaining:
            # All opponents declined — notify and leave overlay open
            dlg = tk.Toplevel(self)
            dlg.title("Trade Offer")
            dlg.configure(bg="#1a2e4a")
            dlg.resizable(False, False)
            dlg.grab_set()
            tk.Label(dlg, text="All players declined the offer.",
                     font=("Arial", 11), bg="#1a2e4a", fg="#ff6b6b",
                     padx=20, pady=16).pack()
            tk.Button(dlg, text="OK", font=("Arial", 11, "bold"),
                      command=dlg.destroy, padx=16, pady=4).pack(pady=(0, 12))
            dlg.update_idletasks()
            w, h = dlg.winfo_reqwidth(), dlg.winfo_reqheight()
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            dlg.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
            return

        target_pid = remaining[0]
        oname = PLAYER_COLORS[offerer_pid]["name"]
        tname = PLAYER_COLORS[target_pid]["name"]
        total = len([i for i in range(len(state.players)) if i != offerer_pid])
        current_n = total - len(remaining) + 1   # e.g. "1 of 3"

        def _fmt(d: dict) -> str:
            return ", ".join(
                f"{amt} {r.value.capitalize()}" for r, amt in d.items()
            )

        BG   = "#1a2e4a"
        FG   = "#ffffff"
        FONT = ("Arial", 11)
        BOLD = ("Arial", 11, "bold")

        dlg = tk.Toplevel(self)
        dlg.title(f"Trade Offer ({current_n}/{total})")
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.protocol("WM_DELETE_WINDOW", lambda: None)

        tk.Label(dlg, text=f"Trade Offer — {tname}", font=("Arial", 13, "bold"),
                 bg=BG, fg="#f0c040", pady=8, padx=16).pack(fill=tk.X)

        tk.Label(dlg, text=f"{oname} gives you:", font=BOLD,
                 bg=BG, fg="#aad4f5", padx=16, anchor="w").pack(fill=tk.X)
        tk.Label(dlg, text=f"  {_fmt(give_dict)}", font=FONT,
                 bg=BG, fg=FG, padx=16, anchor="w").pack(fill=tk.X)

        tk.Label(dlg, text=f"{oname} wants from you:", font=BOLD,
                 bg=BG, fg="#aad4f5", padx=16, anchor="w").pack(fill=tk.X, pady=(6, 0))
        tk.Label(dlg, text=f"  {_fmt(recv_dict)}", font=FONT,
                 bg=BG, fg=FG, padx=16, anchor="w").pack(fill=tk.X)

        target_hand = state.players[target_pid].resources
        can_afford  = all(target_hand.get(r, 0) >= amt
                          for r, amt in recv_dict.items())
        missing_str = ", ".join(
            f"{amt - target_hand.get(r, 0)} {r.value}"
            for r, amt in recv_dict.items()
            if target_hand.get(r, 0) < amt
        )

        tk.Label(dlg, text=f"{tname}: do you accept?", font=BOLD,
                 bg=BG, fg="#aad4f5", padx=16, anchor="w").pack(fill=tk.X, pady=(8, 0))

        if not can_afford:
            tk.Label(dlg, text=f"  (missing: {missing_str})",
                     font=("Arial", 10), bg=BG, fg="#ff6b6b",
                     padx=16, anchor="w").pack(fill=tk.X)

        def _accept():
            dlg.destroy()
            self._hide_trade_overlay()
            act = Action(
                ActionType.PLAYER_TRADE,
                give_amounts    = give_dict,
                receive_amounts = recv_dict,
                trade_with      = target_pid,
            )
            self._game_state, _, _ = self._game_engine.step(state, act)
            self._sync_pieces_from_state()
            self._update_status()
            self._update_resources_display()
            self._update_action_buttons()

        def _decline():
            dlg.destroy()
            # Ask the next opponent
            self._show_trade_offer_response(
                state, offerer_pid, remaining[1:], give_dict, recv_dict)

        btn_row = tk.Frame(dlg, bg=BG, pady=10)
        btn_row.pack()
        tk.Button(btn_row, text="Accept", command=_accept,
                  font=BOLD, bg="#2a7a2a", fg="white", padx=14, pady=5,
                  state=tk.NORMAL if can_afford else tk.DISABLED,
                  relief=tk.RAISED).pack(side=tk.LEFT, padx=8)
        tk.Button(btn_row, text="Decline", command=_decline,
                  font=FONT, bg="#7a2a2a", fg="white", padx=10, pady=5,
                  relief=tk.RAISED).pack(side=tk.LEFT, padx=8)

        dlg.update_idletasks()
        w, h = dlg.winfo_reqwidth(), dlg.winfo_reqheight()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        dlg.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def _show_steal_dialog(self, victims: list, on_choice) -> None:
        """Modal dialog: current player picks which opponent to steal one card from."""
        state  = self._game_state
        pid    = state.current_player
        pname  = PLAYER_COLORS[pid]["name"]

        dlg = tk.Toplevel(self)
        dlg.title("Steal a Card")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.protocol("WM_DELETE_WINDOW", lambda: None)

        tk.Label(dlg, text=f"{pname}: choose who to steal from.",
                 font=("Arial", 10), padx=12, pady=8).pack()

        chosen = tk.IntVar(value=victims[0])
        for vpid in victims:
            vname = PLAYER_COLORS[vpid]["name"]
            hand_total = sum(state.players[vpid].resources.values())
            tk.Radiobutton(
                dlg, text=f"{vname}  ({hand_total} card{'s' if hand_total != 1 else ''})",
                variable=chosen, value=vpid,
                font=("Arial", 10), padx=12,
            ).pack(anchor="w")

        def _confirm():
            dlg.destroy()
            on_choice(chosen.get())

        tk.Button(dlg, text="Steal", command=_confirm,
                  font=("Arial", 10, "bold"), pady=4).pack(pady=(8, 10))

        self.update_idletasks()
        x = self.winfo_x() + (self.winfo_width()  - dlg.winfo_reqwidth())  // 2
        y = self.winfo_y() + (self.winfo_height() - dlg.winfo_reqheight()) // 2
        dlg.geometry(f"+{x}+{y}")

    def _show_discard_dialog(self) -> None:
        """Modal dialog letting the current player choose which cards to discard."""
        state = self._game_state
        if state is None or state.phase != Phase.DISCARD:
            return
        pid    = state.current_player
        pname  = PLAYER_COLORS[pid]["name"]
        hand   = state.players[pid].resources
        total  = sum(hand.values())
        needed = total // 2

        dlg = tk.Toplevel(self)
        dlg.title(f"{pname} – Discard {needed} card(s)")
        dlg.resizable(False, False)
        dlg.grab_set()   # modal
        dlg.protocol("WM_DELETE_WINDOW", lambda: None)  # block close button

        tk.Label(dlg, text=f"{pname} has {total} cards and must discard {needed}.",
                 font=("Arial", 10), padx=12, pady=8).pack()

        res_order = ["lumber", "brick", "wool", "grain", "ore"]
        spinvars: dict[str, tk.IntVar] = {}
        for rname in res_order:
            count = hand.get(Resource(rname), 0)
            if count == 0:
                continue
            row = tk.Frame(dlg, padx=12, pady=2)
            row.pack(fill=tk.X)
            tk.Label(row, text=f"{rname.capitalize()}  (have {count}):",
                     font=("Arial", 9), width=22, anchor="w").pack(side=tk.LEFT)
            var = tk.IntVar(value=0)
            spinvars[rname] = var
            sb = tk.Spinbox(row, from_=0, to=count, textvariable=var,
                            width=4, font=("Arial", 9))
            sb.pack(side=tk.LEFT)

        err_var = tk.StringVar()
        tk.Label(dlg, textvariable=err_var, font=("Arial", 9),
                 fg="red", pady=4).pack()

        def _confirm():
            chosen = {Resource(r): v.get() for r, v in spinvars.items()}
            chosen_total = sum(chosen.values())
            if chosen_total != needed:
                err_var.set(f"Must discard exactly {needed} (currently {chosen_total}).")
                return
            dlg.destroy()
            action = Action(ActionType.DISCARD, discard=chosen)
            self._game_state, _, _ = self._game_engine.step(state, action)
            self._sync_pieces_from_state()
            self._update_status()

        tk.Button(dlg, text="Confirm Discard", command=_confirm,
                  font=("Arial", 10, "bold"), pady=4).pack(pady=(4, 10))

        # Center over main window
        self.update_idletasks()
        x = self.winfo_x() + (self.winfo_width()  - dlg.winfo_reqwidth())  // 2
        y = self.winfo_y() + (self.winfo_height() - dlg.winfo_reqheight()) // 2
        dlg.geometry(f"+{x}+{y}")

    def _do_buy_dev_card(self) -> None:
        if self._game_state is None or self._game_engine is None:
            return
        if self._game_state.phase != Phase.MAIN:
            return
        self._game_state, _, _ = self._game_engine.step(
            self._game_state, Action(ActionType.BUY_DEV_CARD))
        self._sync_pieces_from_state()
        self._update_status()
        self._update_resources_display()
        self._update_action_buttons()

    def _do_end_turn_action(self) -> None:
        if self._game_state is None or self._game_engine is None:
            return
        if self._game_state.phase != Phase.MAIN:
            return
        self._game_state, _, _ = self._game_engine.step(
            self._game_state, Action(ActionType.END_TURN))
        self._sync_pieces_from_state()
        self._update_status()

    def _update_action_buttons(self) -> None:
        if self._roll_btn is None or self._end_turn_btn is None:
            return
        state = self._game_state
        is_main = state is not None and state.phase == Phase.MAIN
        if state is None:
            self._roll_btn.config(state=tk.DISABLED)
            self._end_turn_btn.config(state=tk.DISABLED)
            if self._trade_btn:
                self._trade_btn.config(state=tk.DISABLED)
            if self._player_trade_btn:
                self._player_trade_btn.config(state=tk.DISABLED)
            if self._buy_dev_btn:
                self._buy_dev_btn.config(state=tk.DISABLED)
            return
        self._roll_btn.config(
            state=tk.NORMAL if state.phase == Phase.ROLL else tk.DISABLED)
        self._end_turn_btn.config(
            state=tk.NORMAL if is_main else tk.DISABLED)
        if self._trade_btn:
            can_trade = False
            if is_main and self._game_engine is not None:
                pid  = state.current_player
                hand = state.players[pid].resources
                for r in Resource:
                    ratio = self._game_engine._get_trade_ratio(state, pid, r)
                    if hand.get(r, 0) >= ratio:
                        can_trade = True
                        break
            if can_trade:
                self._trade_btn.config(state=tk.NORMAL,
                                       text="🔄  Trade")
            elif is_main:
                self._trade_btn.config(state=tk.DISABLED,
                                       text="🔄  Trade (need 4+ resources)")
            else:
                self._trade_btn.config(state=tk.DISABLED,
                                       text="🔄  Trade")
        if self._player_trade_btn:
            # Available in MAIN phase whenever the active player has at least 1 resource
            # and there is at least one opponent
            can_player_trade = False
            if is_main and state is not None:
                pid  = state.current_player
                hand = state.players[pid].resources
                has_res = any(v > 0 for v in hand.values())
                has_opp = len(state.players) > 1
                can_player_trade = has_res and has_opp
            self._player_trade_btn.config(
                state=tk.NORMAL if can_player_trade else tk.DISABLED)
        if self._buy_dev_btn:
            can_buy = False
            if is_main and self._game_engine is not None and state.dev_deck:
                pid = state.current_player
                can_buy = state.players[pid].can_afford(
                    self._game_engine.BUILD_COSTS["dev_card"])
            remaining = len(state.dev_deck) if state is not None else 0
            self._buy_dev_btn.config(
                state=tk.NORMAL if can_buy else tk.DISABLED,
                text=f"🃏  Buy Dev Card  ({remaining} left)")

    def _update_resources_display(self) -> None:
        if not self._res_labels:
            return
        state = self._game_state
        if state is None or not GAME_ENGINE_AVAILABLE:
            for lbl in self._res_labels.values():
                lbl.config(text="0")
            return
        pid  = state.current_player
        hand = state.players[pid].resources
        for res_name, lbl in self._res_labels.items():
            count = hand.get(Resource(res_name), 0)
            lbl.config(text=str(count),
                       fg="black" if count > 0 else "#aaa")

        # Refresh dev card hand display
        if hasattr(self, "_dev_cards_frame"):
            for widget in self._dev_cards_frame.winfo_children():
                widget.destroy()
            p = state.players[pid]
            all_cards = p.dev_cards + p.dev_cards_new
            if not all_cards:
                tk.Label(self._dev_cards_frame, text="(none)",
                         font=("Arial", 9), bg="#f0f0f0", fg="#aaa"
                         ).pack(anchor="w")
            else:
                from collections import Counter
                counts = Counter(all_cards)
                for card, cnt in counts.items():
                    is_new = card == DevCard.VICTORY_POINT and card in p.dev_cards_new
                    is_vp  = (card == DevCard.VICTORY_POINT)
                    label  = DEV_CARD_LABEL.get(card, card.value)
                    suffix = " (new, unplayable)" if (card in p.dev_cards_new and not is_vp) else ""
                    secret = "  🔒 secret" if is_vp else ""
                    tk.Label(self._dev_cards_frame,
                             text=f"{label}{secret}: {cnt}{suffix}",
                             font=("Arial", 9), bg="#f0f0f0",
                             fg="#888" if (card in p.dev_cards_new) else ("#7a5c00" if is_vp else "black")
                             ).pack(anchor="w")
                if any(card in p.dev_cards_new for card in counts if card != DevCard.VICTORY_POINT):
                    tk.Label(self._dev_cards_frame,
                             text="* can't play until next turn",
                             font=("Arial", 8), bg="#f0f0f0", fg="#888"
                             ).pack(anchor="w")

        # Refresh deck count
        if hasattr(self, "_deck_count_lbl"):
            self._deck_count_lbl.config(text=str(len(state.dev_deck)))

    def _nearest_hex(self, x: float, y: float) -> Optional[int]:
        """Return the board.hexes index of the hex whose center is nearest (x, y)."""
        R = self._current_radius()
        best_idx, best_d = None, float("inf")
        for i, h in enumerate(self.board.hexes):
            hx, hy = self._hex_center(h.row, h.col)
            d = math.hypot(x - hx, y - hy)
            if d < best_d:
                best_d, best_idx = d, i
        return best_idx if best_d < R else None

    def _select_player(self, idx: int) -> None:
        self.current_player.set(idx)
        for i, btn in enumerate(self._player_btns):
            btn.config(relief=tk.SUNKEN if i == idx else tk.RAISED)

    def _clear_pieces(self) -> None:
        self.settlements.clear()
        self.cities.clear()
        self.roads.clear()
        self._draw_pieces()

    def _set_zoom(self, new_zoom: float) -> None:
        self._zoom = max(0.25, min(4.0, new_zoom))
        self._zoom_label.config(text=f"{int(self._zoom * 100)}%")
        self._board_radius = _FALLBACK_R * self._zoom
        self._rescale_images()
        self._draw_board()

    def _randomize(self) -> None:
        self._game_state  = None
        self._game_engine = None
        self._status_var.set("Press 'Start Game'\nto begin setup.")
        self.board.build()
        self.settlements.clear()
        self.cities.clear()
        self.roads.clear()
        self._draw_board()

    def _export(self) -> None:
        try:
            from renderer import BoardRenderer
        except ImportError:
            messagebox.showerror("Error", "renderer.py not found next to board_builder.py.")
            return
        if not PIL_AVAILABLE:
            messagebox.showerror("Error", "Pillow is required.\nRun:  pip install Pillow")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            title="Save board image",
        )
        if not path:
            return
        try:
            img = BoardRenderer().render(self.board)
            img.save(path)
            messagebox.showinfo("Saved", f"Board saved to:\n{path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export failed", str(exc))


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = BoardBuilderApp()
    app.mainloop()
