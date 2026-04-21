"""
renderer.py - High-resolution Catan board image composer.

Uses Pillow to paste the scanned hex tile images from the Images folder onto a
canvas at the correct positions, then overlays number tokens.  The output PNG
is suitable for printing a physical Catan board.

Usage (standalone):
    python renderer.py [output.png]

Usage (from board_builder.py export button):
    from renderer import BoardRenderer
    img = BoardRenderer().render(board)
    img.save("my_board.png")
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from board import Board, HexType, ROW_SIZES

# ── paths ──────────────────────────────────────────────────────────────────────
IMAGES_DIR   = Path(__file__).parent.parent / "Images"
HEX_IMG_DIR  = IMAGES_DIR / "hexes" / "vector"

# ── fallback fill colours (RGBA) ───────────────────────────────────────────────
HEX_FILL: dict[HexType, tuple[int, int, int, int]] = {
    HexType.FOREST:   (45,  106,  79, 255),
    HexType.PASTURE:  (116, 198, 157, 255),
    HexType.FIELD:    (255, 209, 102, 255),
    HexType.HILL:     (193,  83,  30, 255),
    HexType.MOUNTAIN: (141, 153, 174, 255),
    HexType.DESERT:   (233, 216, 166, 255),
}

OCEAN_FILL = (26, 111, 176, 255)


class BoardRenderer:
    """Composes a full-board PNG from the scanned hex images."""

    # Tune these for the desired output resolution.
    # At HEX_RADIUS=150 the board is ~1750 × 1600 px (≈ 12 × 11 in at 150 dpi).
    HEX_RADIUS: int = 150

    @property
    def _canvas_w(self) -> int:
        return int(math.sqrt(3) * self.HEX_RADIUS * 5 + self.HEX_RADIUS * 2)

    @property
    def _canvas_h(self) -> int:
        return int(1.5 * self.HEX_RADIUS * 4 + self.HEX_RADIUS * 2 + self.HEX_RADIUS)

    # ── public API ─────────────────────────────────────────────────────────────

    def render(self, board: Board) -> Image.Image:
        """Return a PIL Image of the complete board."""
        canvas = Image.new("RGBA", (self._canvas_w, self._canvas_h), OCEAN_FILL)
        draw   = ImageDraw.Draw(canvas)

        hex_imgs = self._load_hex_images()

        for h in board.hexes:
            cx, cy = self._hex_center(h.row, h.col)
            self._draw_hex(canvas, draw, cx, cy, h.hex_type, hex_imgs)
            if h.token is not None:
                self._draw_token(draw, cx, cy, h.token)

        return canvas.convert("RGB")

    # ── image loading ──────────────────────────────────────────────────────────

    def _load_hex_images(self) -> dict[HexType, Optional[Image.Image]]:
        imgs: dict[HexType, Optional[Image.Image]] = {}
        img_w = int(math.sqrt(3) * self.HEX_RADIUS)
        img_h = int(2 * self.HEX_RADIUS)

        for htype in HexType:
            path = HEX_IMG_DIR / f"{htype.value}.png"
            if path.exists():
                imgs[htype] = (
                    Image.open(path)
                    .convert("RGBA")
                    .resize((img_w, img_h), Image.LANCZOS)
                )
            else:
                imgs[htype] = None

        return imgs

    # ── geometry ───────────────────────────────────────────────────────────────

    def _hex_center(self, row: int, col: int) -> tuple[float, float]:
        R  = self.HEX_RADIUS
        hw = math.sqrt(3) * R
        cx = self._canvas_w / 2
        cy = R + R * 0.5          # top margin

        n = ROW_SIZES[row]
        x = cx + hw * (col - (n - 1) / 2)
        y = cy + row * 1.5 * R
        return x, y

    def _hex_polygon(self, cx: float, cy: float) -> list[tuple[float, float]]:
        R = self.HEX_RADIUS
        pts = []
        for i in range(6):
            angle = math.radians(60 * i - 30)
            pts.append((cx + R * math.cos(angle), cy + R * math.sin(angle)))
        return pts

    # ── drawing helpers ────────────────────────────────────────────────────────

    def _draw_hex(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        cx: float,
        cy: float,
        htype: HexType,
        hex_imgs: dict[HexType, Optional[Image.Image]],
    ) -> None:
        pts   = self._hex_polygon(cx, cy)
        color = HEX_FILL[htype]

        # Filled polygon (fallback colour, also covers any sub-pixel gaps)
        draw.polygon(pts, fill=color, outline=(58, 31, 0, 255))

        # Paste scanned image if available
        img = hex_imgs.get(htype)
        if img:
            ox = int(cx - img.width  / 2)
            oy = int(cy - img.height / 2)
            canvas.paste(img, (ox, oy), mask=img)

    def _draw_token(
        self,
        draw: ImageDraw.ImageDraw,
        cx: float,
        cy: float,
        token: int,
    ) -> None:
        R  = self.HEX_RADIUS
        tr = R * 0.26

        draw.ellipse(
            [(cx - tr, cy - tr), (cx + tr, cy + tr)],
            fill=(245, 230, 200, 255),
            outline=(100, 100, 100, 255),
            width=max(1, int(R * 0.02)),
        )

        text_color = (180, 0, 0) if token in (6, 8) else (0, 0, 0)
        font_size  = max(12, int(tr * 1.1))
        font       = self._get_font(font_size)
        draw.text((cx, cy), str(token), font=font, fill=text_color, anchor="mm")

        # Probability dots (5 for 6/8, 4 for 5/9, etc.)
        pips = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
        n_pips = pips.get(token, 0)
        if n_pips:
            pip_r  = max(2, int(R * 0.025))
            spacing = pip_r * 2.6
            start_x = cx - spacing * (n_pips - 1) / 2
            dot_y   = cy + tr * 0.62
            for k in range(n_pips):
                px = start_x + k * spacing
                draw.ellipse(
                    [(px - pip_r, dot_y - pip_r), (px + pip_r, dot_y + pip_r)],
                    fill=text_color,
                )

    @staticmethod
    def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()


# ── standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    output_path = sys.argv[1] if len(sys.argv) > 1 else "board_output.png"
    board  = Board(randomize=True)
    img    = BoardRenderer().render(board)
    img.save(output_path)
    print(f"Board image saved to: {output_path}")
    print(f"Size: {img.size[0]} × {img.size[1]} px")
