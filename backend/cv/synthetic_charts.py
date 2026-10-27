"""
Synthetic candlestick chart generator.

Renders candlestick charts from *known* OHLC data so the digitiser can be
tested against exact ground truth. Several visual themes imitate common
trading apps (light, dark, classic red/green), with gridlines and a price axis.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

THEMES = {
    # name: (background, grid, text, up colour, down colour)
    "light": ("#ffffff", "#e6e8ec", "#4a4f5c", "#26a69a", "#ef5350"),
    "dark": ("#131722", "#2a2e39", "#b2b5be", "#26a69a", "#ef5350"),
    "classic": ("#ffffff", "#dddddd", "#222222", "#00b050", "#ff0000"),
    "midnight": ("#0b0e14", "#1c212b", "#9aa0aa", "#22c55e", "#f43f5e"),
}


def random_ohlc(n: int = 80, start: float = 150.0, vol: float = 0.02,
                drift: float = 0.0005, seed: int | None = None) -> pd.DataFrame:
    """Geometric-Brownian-motion OHLC series."""
    rng = np.random.default_rng(seed)
    closes = [start]
    for _ in range(n - 1):
        closes.append(closes[-1] * np.exp(drift + vol * rng.standard_normal()))
    closes = np.array(closes)
    opens = np.concatenate([[start * (1 + vol * rng.standard_normal() * 0.3)], closes[:-1]])
    spread = np.abs(rng.standard_normal(n)) * vol * closes * 0.6
    highs = np.maximum(opens, closes) + spread * rng.uniform(0.2, 1.0, n)
    lows = np.minimum(opens, closes) - spread * rng.uniform(0.2, 1.0, n)
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes})


def render_chart(df: pd.DataFrame, theme: str = "light", width_px: int = 1200,
                 height_px: int = 700, axis_side: str = "right", title: str | None = "DEMO · 1D",
                 dpi: int = 100) -> np.ndarray:
    """Render OHLC as a candlestick chart and return it as a BGR uint8 image."""
    bg, grid, text, up, down = THEMES[theme]
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    n = len(df)
    body_w = 0.62
    for i, row in enumerate(df.itertuples(index=False)):
        colour = up if row.Close >= row.Open else down
        ax.vlines(i, row.Low, row.High, color=colour, linewidth=1.1, zorder=3)
        lo, hi = sorted((row.Open, row.Close))
        h = max(hi - lo, (df.High.max() - df.Low.min()) * 0.0015)
        ax.add_patch(plt.Rectangle((i - body_w / 2, lo), body_w, h, facecolor=colour,
                                   edgecolor=colour, linewidth=0.6, zorder=4))

    pad = (df.High.max() - df.Low.min()) * 0.06
    ax.set_xlim(-1, n + 1)
    ax.set_ylim(df.Low.min() - pad, df.High.max() + pad)
    ax.grid(True, color=grid, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(grid)
    if axis_side == "right":
        ax.yaxis.tick_right()
    ax.tick_params(colors=text, labelsize=11, length=0)
    ax.set_xticks(np.arange(0, n, max(1, n // 6)))
    ax.set_xticklabels([f"D{i}" for i in range(0, n, max(1, n // 6))])
    if title:
        ax.set_title(title, color=text, loc="left", fontsize=12)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=bg)
    plt.close(fig)
    buf.seek(0)
    import cv2  # local import keeps matplotlib-only users light

    img = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
    return img


@dataclass
class SyntheticCase:
    image: np.ndarray
    truth: pd.DataFrame
    theme: str


def make_case(n: int = 80, theme: str = "light", seed: int = 0, **kw) -> SyntheticCase:
    rng = np.random.default_rng(seed)
    df = random_ohlc(n=n, start=float(rng.uniform(20, 900)), vol=float(rng.uniform(0.01, 0.035)),
                     seed=seed)
    return SyntheticCase(render_chart(df, theme=theme, **kw), df, theme)
