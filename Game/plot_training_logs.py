"""
Plot training progress from pasted log text.

Usage examples
--------------
1. Paste directly into RAW_LOG_TEXT below, then run:
       python plot_training_logs.py

2. Or save logs to a text file and run:
       python plot_training_logs.py --input-file selfplay_logs.txt

3. Or pipe text on stdin:
       Get-Content selfplay_logs.txt | python plot_training_logs.py

Outputs
-------
Creates two PNG files in ./training_plots by default:
    - games_over_time.png
    - winrates_over_time.png
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RAW_LOG_TEXT = """
Update   10/700 | loss    2.2496 | games  60 | p0 0.37 | p1 0.23 | p2 0.25 | p3 0.15
Update   20/700 | loss    0.5837 | games  68 | p0 0.32 | p1 0.19 | p2 0.32 | p3 0.16
Update   30/700 | loss    4.4629 | games  65 | p0 0.28 | p1 0.26 | p2 0.17 | p3 0.29
Update   40/700 | loss    1.0354 | games  65 | p0 0.26 | p1 0.18 | p2 0.29 | p3 0.26
Update   50/700 | loss    1.3559 | games  64 | p0 0.20 | p1 0.25 | p2 0.28 | p3 0.27
Update   60/700 | loss    4.6939 | games  64 | p0 0.31 | p1 0.25 | p2 0.23 | p3 0.20
Update   70/700 | loss    0.9433 | games  62 | p0 0.37 | p1 0.21 | p2 0.27 | p3 0.15
Update   80/700 | loss    1.5381 | games  62 | p0 0.27 | p1 0.31 | p2 0.23 | p3 0.19
Update   90/700 | loss    0.6021 | games  57 | p0 0.37 | p1 0.33 | p2 0.19 | p3 0.11
Update  100/700 | loss    3.2186 | games  57 | p0 0.25 | p1 0.32 | p2 0.26 | p3 0.18
Update  110/700 | loss    0.1201 | games  61 | p0 0.34 | p1 0.18 | p2 0.34 | p3 0.13
Update  120/700 | loss    5.2106 | games  57 | p0 0.28 | p1 0.23 | p2 0.19 | p3 0.30
Update  130/700 | loss    1.0880 | games  51 | p0 0.31 | p1 0.18 | p2 0.27 | p3 0.24
Update  140/700 | loss    1.0888 | games  56 | p0 0.38 | p1 0.14 | p2 0.43 | p3 0.05
Update  150/700 | loss    4.2124 | games  55 | p0 0.33 | p1 0.15 | p2 0.33 | p3 0.20
Update  160/700 | loss    2.5755 | games  58 | p0 0.41 | p1 0.12 | p2 0.38 | p3 0.09
Update  170/700 | loss   -0.0989 | games  52 | p0 0.29 | p1 0.27 | p2 0.29 | p3 0.15
Update  180/700 | loss    1.0685 | games  55 | p0 0.44 | p1 0.24 | p2 0.16 | p3 0.16
Update  190/700 | loss    4.9998 | games  52 | p0 0.23 | p1 0.29 | p2 0.33 | p3 0.15
Update  200/700 | loss    3.1254 | games  54 | p0 0.24 | p1 0.20 | p2 0.33 | p3 0.22
Update  210/700 | loss    0.6388 | games  54 | p0 0.31 | p1 0.19 | p2 0.24 | p3 0.26
Update  220/700 | loss    3.6069 | games  52 | p0 0.37 | p1 0.25 | p2 0.25 | p3 0.13
Update  230/700 | loss    0.6789 | games  56 | p0 0.20 | p1 0.25 | p2 0.41 | p3 0.14
Update  240/700 | loss    2.2984 | games  52 | p0 0.27 | p1 0.23 | p2 0.44 | p3 0.06
Update  250/700 | loss    3.1850 | games  56 | p0 0.23 | p1 0.29 | p2 0.46 | p3 0.02
Update  260/700 | loss    1.3501 | games  55 | p0 0.22 | p1 0.29 | p2 0.42 | p3 0.07
Update  270/700 | loss    6.3112 | games  57 | p0 0.37 | p1 0.30 | p2 0.26 | p3 0.07
Update  280/700 | loss    0.8614 | games  58 | p0 0.40 | p1 0.22 | p2 0.28 | p3 0.10
Update  290/700 | loss    2.4536 | games  57 | p0 0.14 | p1 0.39 | p2 0.40 | p3 0.07
Update  300/700 | loss    0.4912 | games  59 | p0 0.27 | p1 0.27 | p2 0.36 | p3 0.10
Update  310/700 | loss    0.2084 | games  56 | p0 0.30 | p1 0.25 | p2 0.43 | p3 0.02
Update  320/700 | loss    3.6675 | games  58 | p0 0.22 | p1 0.17 | p2 0.48 | p3 0.12
Update  330/700 | loss    2.9191 | games  61 | p0 0.30 | p1 0.18 | p2 0.49 | p3 0.03
Update  340/700 | loss    3.5763 | games  61 | p0 0.23 | p1 0.34 | p2 0.34 | p3 0.08
Update  350/700 | loss    1.3465 | games  61 | p0 0.23 | p1 0.23 | p2 0.49 | p3 0.05
Update  360/700 | loss    2.7256 | games  64 | p0 0.16 | p1 0.30 | p2 0.52 | p3 0.03
Update  370/700 | loss    2.2987 | games  65 | p0 0.31 | p1 0.15 | p2 0.48 | p3 0.06
Update  380/700 | loss    1.3139 | games  67 | p0 0.25 | p1 0.12 | p2 0.61 | p3 0.01
Update  390/700 | loss    2.5073 | games  69 | p0 0.22 | p1 0.17 | p2 0.58 | p3 0.03
Update  400/700 | loss    1.4184 | games  67 | p0 0.18 | p1 0.16 | p2 0.66 | p3 0.00
Update  410/700 | loss    3.3827 | games  66 | p0 0.18 | p1 0.09 | p2 0.70 | p3 0.03
Update  420/700 | loss    1.4666 | games  64 | p0 0.25 | p1 0.20 | p2 0.53 | p3 0.02
Update  430/700 | loss    1.3759 | games  74 | p0 0.22 | p1 0.12 | p2 0.59 | p3 0.07
Update  440/700 | loss    1.4107 | games  73 | p0 0.18 | p1 0.12 | p2 0.67 | p3 0.03
Update  450/700 | loss    2.5938 | games  66 | p0 0.32 | p1 0.09 | p2 0.52 | p3 0.08
Update  460/700 | loss    1.9390 | games  68 | p0 0.22 | p1 0.15 | p2 0.59 | p3 0.04
Update  470/700 | loss    1.1832 | games  72 | p0 0.22 | p1 0.14 | p2 0.60 | p3 0.04
Update  480/700 | loss    2.3092 | games  63 | p0 0.13 | p1 0.17 | p2 0.67 | p3 0.03
Update  490/700 | loss    3.3599 | games  66 | p0 0.14 | p1 0.17 | p2 0.68 | p3 0.02
"""


UPDATE_RE = re.compile(
    r"Update\s+(?P<update>\d+)\s*/\s*(?P<total>\d+)\s*\|\s*"
    r"loss\s+(?P<loss>-?\d+(?:\.\d+)?)\s*\|\s*"
    r"games\s+(?P<games>\d+)\s*\|\s*"
    r"p0\s+(?P<p0>\d+(?:\.\d+)?)\s*\|\s*"
    r"p1\s+(?P<p1>\d+(?:\.\d+)?)\s*\|\s*"
    r"p2\s+(?P<p2>\d+(?:\.\d+)?)\s*\|\s*"
    r"p3\s+(?P<p3>\d+(?:\.\d+)?)"
)

PLOT_BG = "#fffaf2"
AXIS = "#3b342c"
GRID = "#d8cfc3"
TEXT = "#2a241d"
COLORS = {
    "games": "#0f766e",
    "p0": "#b91c1c",
    "p1": "#2563eb",
    "p2": "#15803d",
    "p3": "#a16207",
}


@dataclass
class TrainingPoint:
    update: int
    total_updates: int
    loss: float
    games: int
    p0: float
    p1: float
    p2: float
    p3: float


def parse_training_log(text: str) -> list[TrainingPoint]:
    points: list[TrainingPoint] = []
    for match in UPDATE_RE.finditer(text):
        points.append(
            TrainingPoint(
                update=int(match.group("update")),
                total_updates=int(match.group("total")),
                loss=float(match.group("loss")),
                games=int(match.group("games")),
                p0=float(match.group("p0")),
                p1=float(match.group("p1")),
                p2=float(match.group("p2")),
                p3=float(match.group("p3")),
            )
        )
    return points


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _linear_regression(points: list[tuple[int, float]]) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    if len(points) == 1:
        return 0.0, points[0][1]

    n = len(points)
    sum_x = sum(x for x, _ in points)
    sum_y = sum(y for _, y in points)
    mean_x = sum_x / n
    mean_y = sum_y / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0:
        return 0.0, mean_y
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    return slope, intercept


def _draw_dotted_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: str,
    width: int = 2,
    dash: int = 8,
    gap: int = 6,
) -> None:
    if len(points) < 2:
        return

    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        dx = x2 - x1
        dy = y2 - y1
        segment_len = (dx * dx + dy * dy) ** 0.5
        if segment_len == 0:
            continue
        drawn = 0.0
        while drawn < segment_len:
            start_ratio = drawn / segment_len
            end_ratio = min(drawn + dash, segment_len) / segment_len
            sx = x1 + dx * start_ratio
            sy = y1 + dy * start_ratio
            ex = x1 + dx * end_ratio
            ey = y1 + dy * end_ratio
            draw.line((sx, sy, ex, ey), fill=color, width=width)
            drawn += dash + gap


def _zoom_bounds(values: list[float], min_floor: float | None = None, max_ceiling: float | None = None) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if low == high:
        padding = max(1.0, abs(low) * 0.1)
    else:
        padding = (high - low) * 0.15
    low -= padding
    high += padding
    if min_floor is not None:
        low = max(min_floor, low)
    if max_ceiling is not None:
        high = min(max_ceiling, high)
    if low == high:
        high = low + 1.0
    return low, high


def _draw_line_chart(
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, list[tuple[int, float]], str]],
    output_path: Path,
    y_min: float,
    y_max: float,
    add_regression: bool = False,
) -> None:
    width, height = 1200, 700
    margin_left, margin_right = 100, 40
    margin_top, margin_bottom = 80, 90

    image = Image.new("RGB", (width, height), PLOT_BG)
    draw = ImageDraw.Draw(image)
    title_font = _font(28)
    label_font = _font(20)
    tick_font = _font(16)
    legend_font = _font(18)

    chart_left = margin_left
    chart_top = margin_top
    chart_right = width - margin_right
    chart_bottom = height - margin_bottom
    chart_width = chart_right - chart_left
    chart_height = chart_bottom - chart_top

    all_x = [x for _, points, _ in series for x, _ in points]
    x_min = min(all_x)
    x_max = max(all_x)
    if x_min == x_max:
        x_max += 1
    if y_min == y_max:
        y_max += 1.0

    def scale_x(value: int) -> float:
        return chart_left + ((value - x_min) / (x_max - x_min)) * chart_width

    def scale_y(value: float) -> float:
        return chart_bottom - ((value - y_min) / (y_max - y_min)) * chart_height

    draw.text((margin_left, 24), title, fill=TEXT, font=title_font)

    for i in range(6):
        y_value = y_min + (i / 5) * (y_max - y_min)
        y = scale_y(y_value)
        draw.line((chart_left, y, chart_right, y), fill=GRID, width=1)
        draw.text((20, y - 8), f"{y_value:.2f}" if y_max <= 1.0 else f"{y_value:.0f}", fill=TEXT, font=tick_font)

    x_ticks = sorted(set(all_x))
    if len(x_ticks) > 10:
        step = max(1, len(x_ticks) // 10)
        x_ticks = x_ticks[::step]
        if x_ticks[-1] != max(all_x):
            x_ticks.append(max(all_x))

    for x_value in x_ticks:
        x = scale_x(x_value)
        draw.line((x, chart_top, x, chart_bottom), fill=GRID, width=1)
        draw.text((x - 12, chart_bottom + 10), str(x_value), fill=TEXT, font=tick_font)

    draw.line((chart_left, chart_top, chart_left, chart_bottom), fill=AXIS, width=2)
    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill=AXIS, width=2)

    legend_x = chart_right - 220
    legend_y = 28
    for index, (name, points, color) in enumerate(series):
        ly = legend_y + index * 28
        draw.line((legend_x, ly + 10, legend_x + 24, ly + 10), fill=color, width=4)
        draw.text((legend_x + 34, ly), name, fill=TEXT, font=legend_font)

        if len(points) == 1:
            x, y = scale_x(points[0][0]), scale_y(points[0][1])
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            continue

        scaled_points = [(scale_x(x), scale_y(y)) for x, y in points]
        draw.line(scaled_points, fill=color, width=4)
        for x, y in scaled_points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)

        if add_regression:
            slope, intercept = _linear_regression(points)
            x_start = min(x for x, _ in points)
            x_end = max(x for x, _ in points)
            regression_points = [
                (scale_x(x_start), scale_y(slope * x_start + intercept)),
                (scale_x(x_end), scale_y(slope * x_end + intercept)),
            ]
            _draw_dotted_line(draw, regression_points, color=color, width=2)

    draw.text((chart_left + chart_width / 2 - 50, height - 45), x_label, fill=TEXT, font=label_font)
    draw.text((20, chart_top - 40), y_label, fill=TEXT, font=label_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def plot_training_points(points: list[TrainingPoint], output_dir: Path) -> tuple[Path, Path]:
    games_series = [(point.update, float(point.games)) for point in points]
    games_path = output_dir / "games_over_time.png"
    winrates_path = output_dir / "winrates_over_time.png"
    games_y_min, games_y_max = _zoom_bounds([point.games for point in points], min_floor=0.0)

    _draw_line_chart(
        title="Completed Games Over Time",
        x_label="Update",
        y_label="Games",
        series=[("games", games_series, COLORS["games"])],
        output_path=games_path,
        y_min=games_y_min,
        y_max=games_y_max,
    )

    _draw_line_chart(
        title="Seat Win Rates Over Time",
        x_label="Update",
        y_label="Win Rate",
        series=[
            ("p0", [(point.update, point.p0) for point in points], COLORS["p0"]),
            ("p1", [(point.update, point.p1) for point in points], COLORS["p1"]),
            ("p2", [(point.update, point.p2) for point in points], COLORS["p2"]),
            ("p3", [(point.update, point.p3) for point in points], COLORS["p3"]),
        ],
        output_path=winrates_path,
        y_min=0.0,
        y_max=1.0,
        add_regression=True,
    )

    return games_path, winrates_path


def _load_input_text(input_file: str | None) -> str:
    if input_file:
        return Path(input_file).read_text(encoding="utf-8")

    if not sys.stdin.isatty():
        piped = sys.stdin.read()
        if piped.strip():
            return piped

    text = RAW_LOG_TEXT.strip()
    if text and text != "PASTE LOG LINES HERE":
        return text

    raise ValueError(
        "No training log text provided. Paste logs into RAW_LOG_TEXT, pipe them on stdin, "
        "or use --input-file."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training logs from pasted update lines.")
    parser.add_argument("--input-file", help="Path to a text file containing Update log lines.")
    parser.add_argument(
        "--output-dir",
        default="training_plots",
        help="Directory to write PNG plots into. Default: training_plots",
    )
    args = parser.parse_args()

    text = _load_input_text(args.input_file)
    points = parse_training_log(text)
    if not points:
        raise ValueError("No valid Update lines were found in the provided text.")

    games_path, winrates_path = plot_training_points(points, Path(args.output_dir))
    print(f"Parsed {len(points)} update lines.")
    print(f"Saved {games_path}")
    print(f"Saved {winrates_path}")


if __name__ == "__main__":
    main()