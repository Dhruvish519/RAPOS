"""
RAPOS Chart Vision: candlestick chart digitiser (OpenCV).

Turns a screenshot of a candlestick chart into OHLC price data.

Pipeline
--------
1. Theme detection   - estimate background colour from the image border.
2. Colour masks      - HSV thresholds isolate bullish (green) and bearish (red)
                       pixels; tiny connected components (anti-aliasing,
                       stray text) are removed.
3. Grid / plot area  - morphological opening with long line kernels finds
                       horizontal and vertical gridlines; their extent gives
                       the plot rectangle, which excludes axis badges/labels.
4. Candle detection  - column-occupancy runs per colour give each candle's
                       x-extent; merged neighbours are split at profile minima;
                       glyph-like blobs (legend text) are rejected.
5. Body / wick split - row-width profile inside each candle separates the wide
                       body from the thin wick -> open/close vs high/low pixels.
6. Y calibration     - the price axis is OCR'd with Tesseract; a robust
                       (RANSAC-style) linear fit maps pixel row -> price. Users
                       can instead supply the visible high/low prices.
7. Output            - OHLC DataFrame + annotated overlay image.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from itertools import combinations

import cv2
import numpy as np
import pandas as pd

try:  # OCR is optional: manual calibration works without it
    import pytesseract

    pytesseract.get_tesseract_version()
    OCR_AVAILABLE = True
except Exception:  # pragma: no cover - depends on host
    OCR_AVAILABLE = False


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------
@dataclass
class Candle:
    x0: int
    x1: int
    high_y: int
    low_y: int
    body_top_y: int
    body_bottom_y: int
    bullish: bool

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def open_y(self) -> int:
        return self.body_bottom_y if self.bullish else self.body_top_y

    @property
    def close_y(self) -> int:
        return self.body_top_y if self.bullish else self.body_bottom_y


@dataclass
class DigitizeResult:
    ohlc: pd.DataFrame
    candles: list[Candle]
    theme: str
    plot_area: tuple[int, int, int, int]          # x0, y0, x1, y1
    h_gridlines: list[int]
    calibration: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    mode: str = "green-red"
    overlay: np.ndarray | None = None
    masks: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "n_candles": len(self.candles),
            "theme": self.theme,
            "colour_mode": self.mode,
            "plot_area": list(map(int, self.plot_area)),
            "n_gridlines": len(self.h_gridlines),
            "calibration": self.calibration,
            "warnings": self.warnings,
        }


# ----------------------------------------------------------------------------
# Parameters (HSV in OpenCV units: H 0-180, S/V 0-255)
# ----------------------------------------------------------------------------
GREEN_H = (35, 100)
RED_H_LOW, RED_H_HIGH = 12, 160
MIN_SAT, MIN_VAL = 70, 70


def _load(image) -> np.ndarray:
    if isinstance(image, np.ndarray):
        img = image
    elif isinstance(image, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    # Very large screenshots: downscale for speed, keeps geometry proportional
    h, w = img.shape[:2]
    if max(h, w) > 2600:
        s = 2600 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


# ----------------------------------------------------------------------------
# Step 1: theme / background
# ----------------------------------------------------------------------------
def detect_background(img: np.ndarray) -> tuple[np.ndarray, str]:
    b = max(3, min(img.shape[:2]) // 100)
    border = np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
                             img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    lum = 0.114 * bg[0] + 0.587 * bg[1] + 0.299 * bg[2]
    return bg.astype(np.uint8), ("dark" if lum < 110 else "light")


# ----------------------------------------------------------------------------
# Step 2: colour segmentation
# ----------------------------------------------------------------------------
def _remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return (keep[labels] * 255).astype(np.uint8)


def colour_masks(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    sat = (s >= MIN_SAT) & (v >= MIN_VAL)
    green = (sat & (h >= GREEN_H[0]) & (h <= GREEN_H[1])).astype(np.uint8) * 255
    red = (sat & ((h <= RED_H_LOW) | (h >= RED_H_HIGH))).astype(np.uint8) * 255
    min_area = max(3, img.shape[0] * img.shape[1] // 400_000)
    green, red = _remove_small(green, min_area), _remove_small(red, min_area)
    green, red = remove_horizontal_lines(green), remove_horizontal_lines(red)
    return remove_volume_histogram(green, red)


def remove_volume_histogram(green: np.ndarray, red: np.ndarray):
    """
    Volume bars all stand on one baseline row, so that row is almost fully
    coloured across the chart width - something candles never do. Find the
    lowest such row and erase every coloured column-run that touches it.
    """
    colour = (green > 0) | (red > 0)
    h, w = colour.shape
    cols = np.where(colour.any(axis=0))[0]
    if len(cols) < 20:
        return green, red
    xa, xb = cols.min(), cols.max()
    cover = colour[:, xa:xb + 1].mean(axis=1)
    rows = np.where(cover > 0.5)[0]
    rows = rows[rows > h * 0.3]
    if len(rows) == 0:
        return green, red
    base = int(rows.max())
    g, r = green.copy(), red.copy()
    # Erase every connected blob that stands on the baseline (whole bars,
    # including columns where a gridline broke the baseline row)
    for m in (g, r):
        n, labels = cv2.connectedComponents(m, connectivity=8)
        if n <= 1:
            continue
        band = labels[max(0, base - 2):base + 3]
        ids = np.unique(band[band > 0])
        m[np.isin(labels, ids)] = 0
    return g, r


def remove_horizontal_lines(mask: np.ndarray) -> np.ndarray:
    """
    Remove thin coloured horizontal lines (e.g. dotted 'last price' lines).
    Only pixels that are vertically thin (< 3 px) are considered, so candle
    bodies - which are tall - are never touched.
    """
    w = mask.shape[1]
    tall = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    thin = cv2.subtract(mask, tall)
    joined = cv2.morphologyEx(thin, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)))
    lines = cv2.morphologyEx(joined, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 6), 1)))
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    out = mask.copy()
    out[(lines > 0) & (thin > 0)] = 0
    return out


# ----------------------------------------------------------------------------
# Step 3: gridlines and plot area
# ----------------------------------------------------------------------------
def _cluster_1d(idx: np.ndarray, gap: int = 2) -> list[int]:
    if len(idx) == 0:
        return []
    groups, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] <= gap:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return [int(round(np.mean(g))) for g in groups]


def detect_grid(img: np.ndarray, bg: np.ndarray, colour: np.ndarray):
    diff = cv2.absdiff(img, np.full_like(img, bg)).max(axis=2)
    nonbg = ((diff > 6) & (colour == 0)).astype(np.uint8) * 255
    h, w = nonbg.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 4), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 4)))
    # Candles occlude gridlines; closing the colour gaps first keeps lines whole
    bridged = cv2.bitwise_or(nonbg, colour)
    hl = cv2.morphologyEx(bridged, cv2.MORPH_OPEN, hk)
    vl = cv2.morphologyEx(bridged, cv2.MORPH_OPEN, vk)
    hl[colour > 0] = 0
    vl[colour > 0] = 0

    # A real gridline/spine spans a large part of the chart; a long wick or a
    # short rule does not. Only such lines define the plot rectangle.
    rows = np.where(hl.sum(axis=1) / 255 > w * 0.4)[0]
    cols = np.where(vl.sum(axis=0) / 255 > h * 0.45)[0]
    hlines = _cluster_1d(rows)
    vlines = _cluster_1d(cols)

    x0, y0, x1, y1 = 0, 0, w - 1, h - 1
    if len(rows):
        xs = np.where(hl[rows].any(axis=0))[0]
        if len(xs):
            x0, x1 = int(xs.min()), int(xs.max())
    if len(cols):
        ys = np.where(vl[:, cols].any(axis=1))[0]
        if len(ys):
            y0, y1 = int(ys.min()), int(ys.max())
    keep = np.zeros_like(hl)
    keep[rows] = hl[rows]
    keep[:, cols] |= vl[:, cols]
    return hlines, vlines, (x0, y0, x1, y1), keep


# ----------------------------------------------------------------------------
# Step 4/5: candles
# ----------------------------------------------------------------------------
def _runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask_1d.astype(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _segments_per_column(col: np.ndarray, max_gap: int = 1) -> int:
    ys = np.where(col)[0]
    if len(ys) == 0:
        return 0
    return 1 + int(np.sum(np.diff(ys) > max_gap + 1))


def _split_wide(run, profile, width):
    x0, x1 = run
    k = int(round((x1 - x0 + 1) / width))
    if k <= 1:
        return [run]
    cuts = [x0]
    for j in range(1, k):
        guess = x0 + int(round(j * (x1 - x0 + 1) / k))
        lo, hi = max(x0 + 1, guess - width // 3), min(x1 - 1, guess + width // 3)
        seg = profile[lo:hi + 1]
        cuts.append(lo + int(np.argmin(seg)) if len(seg) else guess)
    cuts.append(x1 + 1)
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - 1 >= a:
            out.append((a, b - 1))
    return out


def _measure(mask: np.ndarray, x0: int, x1: int, bullish: bool) -> Candle | None:
    sub = mask[:, x0:x1 + 1] > 0
    rows = np.where(sub.any(axis=1))[0]
    if len(rows) == 0:
        return None
    width = x1 - x0 + 1
    rw = sub.sum(axis=1)
    if width >= 3:
        body = np.where(rw >= max(2, 0.55 * width))[0]
    else:
        body = rows
    if len(body) == 0:
        body = np.array([rows[np.argmax(rw[rows])]])
    return Candle(x0, x1, int(rows.min()), int(rows.max()), int(body.min()), int(body.max()), bullish)


def _ink_threshold(diff: np.ndarray, gridmask: np.ndarray | None, base: float = 40) -> float:
    """
    Contrast threshold that sits above the faint gridlines. Dark frame spines
    are excluded from the estimate (they would push it far too high).
    """
    if gridmask is None or not gridmask.any():
        return base
    vals = diff[gridmask > 0]
    faint = vals[vals < 100]
    grid_c = float(np.median(faint)) if len(faint) > 0.1 * len(vals) else 0.0
    return float(min(100.0, max(base, grid_c + 15)))


def _ink(img: np.ndarray, bg: np.ndarray, gridmask: np.ndarray | None, plot_area) -> np.ndarray:
    """
    Pixels that clearly stand out from the background (more than the faint
    gridlines do; median is used because frame spines are dark), minus long
    vertical lines (gridlines/spines) that wicks must never be traced along.
    """
    diff = cv2.absdiff(img, np.full_like(img, bg)).max(axis=2)
    thr = _ink_threshold(diff, gridmask, 45)
    ink = (diff > thr).astype(np.uint8) * 255
    _, y0p, _, y1p = plot_area
    joined = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)))
    vlines = cv2.morphologyEx(joined, cv2.MORPH_OPEN, cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(40, int((y1p - y0p) * 0.6)))))
    return (ink > 0) & (vlines == 0)


def extend_wicks(img: np.ndarray, bg: np.ndarray, candles: list[Candle], plot_area,
                 gridmask: np.ndarray | None = None) -> None:
    """
    Some apps draw wicks in grey/black instead of the candle colour. From the
    body, follow 'ink' pixels (clearly different from the background) straight
    up and down the candle's centre columns to recover the true high and low.
    """
    ink = _ink(img, bg, gridmask, plot_area)
    _, y0p, _, y1p = plot_area
    for c in candles:
        xs = range(max(c.x0, int(c.xc) - 1), min(c.x1, int(c.xc) + 1) + 1)
        best_hi, best_lo = c.high_y, c.low_y
        for x in xs:
            y = c.body_top_y
            while y - 1 >= y0p and ink[y - 1, x]:
                y -= 1
            best_hi = min(best_hi, y)
            y = c.body_bottom_y
            while y + 1 <= y1p and ink[y + 1, x]:
                y += 1
            best_lo = max(best_lo, y)
        c.high_y, c.low_y = best_hi, best_lo


def detect_candles(green, red, plot_area, warnings: list[str],
                   layers: list | None = None) -> list[Candle]:
    x0p, y0p, x1p, y1p = plot_area
    candidates = []  # (x0, x1, mask, bullish)
    for mask, bull in (layers or ((green, True), (red, False))):
        m = np.zeros_like(mask)
        m[y0p:y1p + 1, x0p:x1p + 1] = mask[y0p:y1p + 1, x0p:x1p + 1]
        # rows spanning most of the plot are frame lines / rules, never candles
        m[(m > 0).sum(axis=1) > 0.6 * (x1p - x0p + 1)] = 0
        prof = (m > 0).sum(axis=0)
        for r in _runs(prof > 0):
            if prof[r[0]:r[1] + 1].sum() >= 4:
                candidates.append((r, m, bull, prof))
    if not candidates:
        return []

    # Typical candle width = median run width of "normal" runs
    widths = np.array([r[1] - r[0] + 1 for r, *_ in candidates])
    w_med = int(np.median(widths))
    w_typ = int(np.median(widths[widths <= max(w_med * 1.5, 2)])) if len(widths) else w_med
    w_typ = max(w_typ, 1)

    candles: list[Candle] = []
    rejected_text = 0
    for r, m, bull, prof in candidates:
        pieces = _split_wide(r, prof, w_typ) if (r[1] - r[0] + 1) > 1.7 * w_typ else [r]
        for a, b in pieces:
            sub = m[:, a:b + 1] > 0
            segs = [_segments_per_column(sub[:, j]) for j in range(sub.shape[1])]
            multi = np.mean([s > 1 for s in segs]) if segs else 0
            if multi > 0.35:
                rejected_text += 1
                continue
            c = _measure(m, a, b, bull)
            if c is not None:
                candles.append(c)

    if rejected_text:
        warnings.append(f"Ignored {rejected_text} coloured text/legend blobs.")
    candles.sort(key=lambda c: c.xc)

    candles = reject_volume_bars(candles, warnings)

    # Remove blobs far narrower than a candle (stray marks) when dense enough
    if len(candles) > 5:
        ws = np.array([c.x1 - c.x0 + 1 for c in candles])
        keep = ws >= max(1, 0.35 * np.median(ws))
        candles = [c for c, k in zip(candles, keep) if k]

    # Wicks/bodies that touch vertically can be split into two blobs at the
    # same x; merge blobs that overlap horizontally and share a colour.
    merged: list[Candle] = []
    for c in candles:
        if merged:
            p = merged[-1]
            overlap = min(p.x1, c.x1) - max(p.x0, c.x0) + 1
            if overlap > 0.5 * min(p.x1 - p.x0 + 1, c.x1 - c.x0 + 1) and p.bullish == c.bullish:
                merged[-1] = Candle(min(p.x0, c.x0), max(p.x1, c.x1), min(p.high_y, c.high_y),
                                    max(p.low_y, c.low_y), min(p.body_top_y, c.body_top_y),
                                    max(p.body_bottom_y, c.body_bottom_y), p.bullish)
                continue
        merged.append(c)
    return merged


# ----------------------------------------------------------------------------
# Colour-agnostic fallback (white/blue, hollow/filled, ... schemes)
# ----------------------------------------------------------------------------
def contrast_mask(img: np.ndarray, bg: np.ndarray, gridmask: np.ndarray, plot_area) -> np.ndarray:
    """Candle pixels by contrast with the background, with hollow bodies filled."""
    x0p, y0p, x1p, y1p = plot_area
    diff = cv2.absdiff(img, np.full_like(img, bg)).max(axis=2)
    thr = _ink_threshold(diff, gridmask, 40)
    m = np.zeros(diff.shape, np.uint8)
    m[y0p:y1p + 1, x0p:x1p + 1] = (diff[y0p:y1p + 1, x0p:x1p + 1] > thr).astype(np.uint8) * 255
    h, w = m.shape
    # remove frame spines and (dashed) gridlines before filling holes, or the
    # whole plot rectangle would count as one giant "hollow candle"
    for k_close, k_open, k_thick in (((15, 1), (max(40, (x1p - x0p) // 3), 1), (1, 3)),
                                     ((1, 15), (1, max(40, int((y1p - y0p) * 0.6))), (3, 1))):
        joined = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, k_close))
        lines = cv2.morphologyEx(joined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k_open))
        # only erase line pixels that are thin across the line, so a gridline
        # crossing a candle body does not cut the body in two
        thick = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k_thick))
        m[(lines > 0) & (thick == 0)] = 0
    m = remove_horizontal_lines(m)
    # seal outlines broken by resampling, then fill enclosed background
    # (hollow candle interiors)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    ff = m.copy()
    pad = cv2.copyMakeBorder(ff, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    mask = np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8)
    cv2.floodFill(pad, mask, (0, 0), 255)
    holes = cv2.bitwise_not(pad)[1:-1, 1:-1]
    m = cv2.bitwise_or(m, holes)
    return _remove_small(m, max(3, h * w // 400_000))


def classify_two_colour(img: np.ndarray, candles: list[Candle], warnings: list[str]) -> str:
    """
    Split candles into two colour groups (k-means in Lab space on body colour)
    and decide which group is 'up' using price continuity: a candle normally
    opens close to where the previous one closed.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    feats = []
    for c in candles:
        ya, yb = c.body_top_y, c.body_bottom_y
        xa, xb = c.x0, c.x1
        if yb - ya >= 3 and xb - xa >= 3:
            ya, yb, xa, xb = ya + 1, yb - 1, xa + 1, xb - 1
        feats.append(lab[ya:yb + 1, xa:xb + 1].reshape(-1, 3).mean(axis=0))
    feats = np.array(feats, np.float32)
    if len(candles) < 4:
        return "two-colour (too few candles)"
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, labels, centres = cv2.kmeans(feats, 2, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    if np.linalg.norm(centres[0] - centres[1]) < 12:
        warnings.append("Candles are all one colour; up/down direction could not be read.")
        for c in candles:
            c.bullish = True
        return "single-colour"

    def cost(up_label):
        opens, closes = [], []
        for c, l in zip(candles, labels):
            bull = l == up_label
            opens.append(c.body_bottom_y if bull else c.body_top_y)
            closes.append(c.body_top_y if bull else c.body_bottom_y)
        return float(np.median(np.abs(np.array(opens[1:]) - np.array(closes[:-1]))))

    up = 0 if cost(0) <= cost(1) else 1
    for c, l in zip(candles, labels):
        c.bullish = bool(l == up)
    return "two-colour"


def recover_missing(img: np.ndarray, bg: np.ndarray, candles: list[Candle], plot_area,
                    gridmask: np.ndarray, warnings: list[str]) -> list[Candle]:
    """
    Candles are evenly spaced, so a gap of ~2 spacings means a candle was
    missed - almost always a doji whose 1-px body is too faint to pass the
    colour threshold. Look for its wick at the expected x and rebuild it.
    """
    if len(candles) < 8:
        return candles
    xs = np.array([c.xc for c in candles])
    step = float(np.median(np.diff(xs)))
    if step < 2:
        return candles
    ink = _ink(img, bg, gridmask, plot_area)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, y0p, _, y1p = plot_area
    w_typ = int(np.median([c.x1 - c.x0 + 1 for c in candles]))
    out, added = [candles[0]], 0
    for prev, nxt in zip(candles[:-1], candles[1:]):
        gap = nxt.xc - prev.xc
        k = int(round(gap / step)) - 1
        for j in range(1, k + 1) if 0 < k <= 3 else []:
            xc = int(round(prev.xc + j * gap / (k + 1)))
            # search only near the neighbours' price band (not e.g. a volume bar)
            span = max(max(prev.low_y, nxt.low_y) - min(prev.high_y, nxt.high_y),
                       0.08 * (y1p - y0p))
            ya = max(y0p, int(min(prev.high_y, nxt.high_y) - 1.5 * span - 5))
            yb = min(y1p, int(max(prev.low_y, nxt.low_y) + 1.5 * span + 5))
            best = None
            for x in range(xc - 1, xc + 2):
                col = ink[ya:yb + 1, x]
                for a, b in _runs(col):
                    if a == 0 or b == yb - ya:   # touches the band edge: not a doji
                        continue
                    if best is None or b - a > best[1] - best[0]:
                        best = (a + ya, b + ya, x)
            if best is None:
                continue
            top, bot, x = best
            if bot - top < 3:
                # a very short mark only counts if it carries candle colour
                patch = hsv[top:bot + 1, max(0, xc - w_typ // 2):xc + w_typ // 2 + 1]
                if patch.size == 0 or patch[..., 1].max() < 40:
                    continue
            # doji body = the most saturated row of the wick
            sat = hsv[top:bot + 1, max(0, xc - w_typ // 2):xc + w_typ // 2 + 1, 1].max(axis=1)
            by = top + int(np.argmax(sat))
            hue = int(hsv[by, max(0, xc - w_typ // 2):xc + w_typ // 2 + 1, 0][
                np.argmax(hsv[by, max(0, xc - w_typ // 2):xc + w_typ // 2 + 1, 1])])
            bull = GREEN_H[0] <= hue <= GREEN_H[1]
            if sat.max() < 25:
                by = (top + bot) // 2
            half = max(1, w_typ // 2)
            out.append(Candle(xc - half, xc + half, top, bot, by, by, bull))
            added += 1
        out.append(nxt)
    if added:
        warnings.append(f"Recovered {added} faint doji candle(s) from their wicks.")
    return out


def reject_volume_bars(candles: list[Candle], warnings: list[str]) -> list[Candle]:
    """
    Volume histograms are coloured like candles but are wickless bars that all
    stand on one shared baseline. Find that baseline and drop the bars.
    """
    if len(candles) < 8:
        return candles
    lows = np.array([c.low_y for c in candles])
    vals, counts = np.unique(lows, return_counts=True)
    # Baseline = the most common bottom row (allowing +-2 px)
    best_y, best_n = None, 0
    for v in vals:
        n = int(np.sum(np.abs(lows - v) <= 2))
        if n > best_n:
            best_y, best_n = v, n
    if best_n < 0.3 * len(candles) or best_n < 6:
        return candles
    is_vol = [abs(c.low_y - best_y) <= 2 and (c.body_top_y - c.high_y) <= 1
              and (c.low_y - c.body_bottom_y) <= 1 for c in candles]
    if sum(is_vol) < 6:
        return candles
    others = [c for c, v in zip(candles, is_vol) if not v]
    # Only treat them as volume if real candles remain above the baseline
    if len(others) < 0.3 * len(candles) or np.median([c.low_y for c in others]) >= best_y:
        return candles
    warnings.append(f"Ignored {sum(is_vol)} volume bars.")
    return others


# ----------------------------------------------------------------------------
# Step 6: price-axis calibration
# ----------------------------------------------------------------------------
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _text_lines(strip: np.ndarray, y_range: tuple[float, float]):
    """
    Locate individual axis labels in a strip: binarise against the strip's own
    background, drop long ruler lines (spines, gridline stubs), then split the
    horizontal projection into text lines.
    """
    h, w = strip.shape[:2]
    bg = np.median(strip.reshape(-1, 3), axis=0)
    ink = (cv2.absdiff(strip, np.full_like(strip, bg.astype(np.uint8))).max(axis=2) > 50)
    ink[:, ink.mean(axis=0) > 0.5] = False       # vertical spines
    ink[ink.mean(axis=1) > 0.6, :] = False       # horizontal rules
    rows = ink.sum(axis=1) > 0
    lines = []
    for y0, y1 in _runs(rows):
        if not (5 <= y1 - y0 + 1 <= 45):
            continue
        yc = (y0 + y1) / 2
        if not (y_range[0] <= yc <= y_range[1]):
            continue
        xs = np.where(ink[y0:y1 + 1].any(axis=0))[0]
        if len(xs) == 0:
            continue
        # Split the line into ink clusters separated by gaps wider than a
        # character gap; tick marks and rotated axis titles become separate
        # clusters. Keep the widest cluster - that is the number.
        gap = max(5, 0.5 * min(y1 - y0 + 1, 14))
        cuts = np.where(np.diff(xs) > gap)[0]
        bounds = np.split(xs, cuts + 1)
        groups = [(g[0], g[-1]) for g in bounds if g[-1] - g[0] >= 4]
        if not groups:
            continue
        xa, xb = max(groups, key=lambda g: g[1] - g[0])
        # trim axis tick marks: leading/trailing columns whose ink is only a
        # 1-2 px horizontal stroke (digits are taller than that)
        line_ink = ink[y0:y1 + 1]
        hcol = line_ink.sum(axis=0)
        tall = max(3, 0.45 * (y1 - y0 + 1))
        while xa < xb and hcol[xa] < tall and hcol[xa:xa + 3].max() < tall:
            xa += 1
        while xb > xa and hcol[xb] < tall and hcol[xb - 2:xb + 1].max() < tall:
            xb -= 1
        if xb - xa < 4 or xb >= w - 1 or xa == 0:   # too small / cut off by the strip edge
            continue
        rr = np.where(ink[y0:y1 + 1, xa:xb + 1].any(axis=1))[0]
        ya, yb = y0 + rr[0], y0 + rr[-1]
        if not (5 <= yb - ya + 1 <= 30):
            continue
        lines.append((ya, yb, xa, xb))
    return lines


def _ocr_strip(strip: np.ndarray, dark: bool, x_off: int,
               y_range: tuple[float, float] = (0, 1e9), scale: int = 4):
    if strip.size == 0 or strip.shape[1] < 8:
        return []
    g = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    if dark or np.median(g) < 110:
        g = 255 - g
    def read(crop, text_h, target):
        s = target / max(text_h, 1)
        c = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        c = cv2.copyMakeBorder(c, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
        t = pytesseract.image_to_string(
            c, config="--psm 7 -c tessedit_char_whitelist=0123456789.,-").strip().replace(" ", "")
        return t if _NUM.fullmatch(t) else None

    pts = []
    for y0, y1, xa, xb in _text_lines(strip, y_range):
        crop = g[max(0, y0 - 3):y1 + 4, max(0, xa - 3):xb + 4]
        text_h = y1 - y0 + 1
        # Tesseract is most reliable at ~20-32 px text height; over-upscaling
        # makes it confuse similar digits (5/9, 3/8). Read at two sizes and
        # use a third as a tie-breaker.
        a, b = read(crop, text_h, 22), read(crop, text_h, 30)
        txt = a if a == b else (lambda c: c if c in (a, b) else None)(read(crop, text_h, 26))
        if not txt:
            continue
        try:
            val = float(txt.replace(",", ""))
        except ValueError:
            continue
        pts.append({"y": (y0 + y1) / 2, "value": val, "x": x_off + xa, "text": txt})
    # Axis tick marks touching a label read as a leading minus sign. A price
    # axis never mixes positive and negative stock prices, so when any label
    # is positive a dash is a tick, not a sign.
    if pts and any(p["value"] > 0 for p in pts):
        for p in pts:
            p["value"] = abs(p["value"])
    return pts


def _robust_fit(pts: list[dict], hlines: list[int]):
    if len(pts) < 2:
        return None
    ys = np.array([p["y"] for p in pts], float)
    # Snap label centres to nearby gridlines: labels are drawn centred on them
    if hlines:
        hl = np.array(hlines)
        for i, y in enumerate(ys):
            j = np.argmin(np.abs(hl - y))
            if abs(hl[j] - y) <= 4:
                ys[i] = hl[j]
    vals = np.array([p["value"] for p in pts], float)
    span = max(vals.max() - vals.min(), 1e-9)
    best = None
    for i, j in combinations(range(len(pts)), 2):
        if abs(ys[i] - ys[j]) < 5 or vals[i] == vals[j]:
            continue
        a = (vals[j] - vals[i]) / (ys[j] - ys[i])
        if a >= 0:  # price must decrease downwards
            continue
        b = vals[i] - a * ys[i]
        res = np.abs(a * ys + b - vals)
        inl = res < 0.02 * span
        if best is None or inl.sum() > best[0].sum():
            best = (inl, a, b)
    if best is None or best[0].sum() < 2:
        return None
    inl = best[0]
    a, b = np.polyfit(ys[inl], vals[inl], 1)
    resid = np.abs(a * ys[inl] + b - vals[inl])
    return {"a": float(a), "b": float(b), "inliers": int(inl.sum()),
            "points": [{"y": float(y), "value": float(v)} for y, v in zip(ys[inl], vals[inl])],
            "max_residual": float(resid.max()) if len(resid) else 0.0}


def calibrate_ocr(img, plot_area, candles, hlines, dark):
    if not OCR_AVAILABLE:
        return None
    h, w = img.shape[:2]
    x0p, _, x1p, _ = plot_area
    first = min(c.x0 for c in candles)
    last = max(c.x1 for c in candles)
    # Candles always sit between the two axes, so strips start just beyond the
    # outermost candles. (Plot-edge estimates can drift a few px into the
    # labels on compressed images and would clip their first digit.)
    right_x = last + 3
    left_x = first - 3
    # Only labels level with the candles belong to the price axis (this drops
    # e.g. a volume pane's axis numbers underneath)
    # robust range: a stray blob must not stretch the window into another pane
    top = float(np.percentile([c.high_y for c in candles], 2))
    bottom = float(np.percentile([c.low_y for c in candles], 98))
    margin = 0.12 * (bottom - top) + 15
    best = None
    for name, (a, b) in (("right", (right_x, w)), ("left", (0, max(0, left_x)))):
        strip = img[:, a:b]
        pts = _ocr_strip(strip, dark, a, (top - margin, bottom + margin))
        fit = _robust_fit(pts, hlines)
        if fit and (best is None or fit["inliers"] > best["inliers"]):
            best = {**fit, "side": name}
    if best is None or best["inliers"] < 3:
        return None
    return best


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------
def digitize(image, price_high: float | None = None, price_low: float | None = None,
             use_ocr: bool = True, draw_overlay: bool = True, keep_masks: bool = False
             ) -> DigitizeResult:
    """
    Extract OHLC data from a candlestick chart image.

    Calibration priority: manual (price_high & price_low = the highest wick and
    lowest wick visible) -> OCR of the price axis -> relative scale (0-100).
    """
    img = _load(image)
    warnings: list[str] = []
    bg, theme = detect_background(img)
    green, red = colour_masks(img)
    colour = cv2.bitwise_or(green, red)
    hlines, vlines, plot_area, gridmask = detect_grid(img, bg, colour)
    candles = detect_candles(green, red, plot_area, warnings)
    mode = "green-red"
    cm = contrast_mask(img, bg, gridmask, plot_area)
    gr_share = (colour > 0).sum() / max(1, (cm > 0).sum())
    if len(candles) < 5 or gr_share < 0.35:
        # Not a green/red chart (few or stray green/red pixels compared with
        # all candle-like ink): fall back to contrast + colour clustering
        gen_warn: list[str] = []
        gen = detect_candles(None, None, plot_area, gen_warn, layers=[(cm, True)])
        if len(gen) > len(candles):
            candles, warnings = gen, warnings + gen_warn
            colour = cm
            mode = classify_two_colour(img, candles, warnings)
    if len(candles) >= 2:
        extend_wicks(img, bg, candles, plot_area, gridmask)
        candles = recover_missing(img, bg, candles, plot_area, gridmask, warnings)

    if len(candles) < 2:
        raise ValueError("No candlesticks found. Use a green/red candlestick chart screenshot.")

    widths = [c.x1 - c.x0 + 1 for c in candles]
    step = float(np.median(np.diff([c.xc for c in candles]))) if len(candles) > 2 else 99
    if np.median(widths) < 3 or step < 8:
        warnings.append("Candles are densely packed; a few may be missed or merged. "
                        "Zoom in (fewer candles on screen) for best accuracy.")

    # --- calibration -------------------------------------------------------
    calib: dict
    if price_high is not None and price_low is not None and price_high > price_low:
        y_hi = min(c.high_y for c in candles)
        y_lo = max(c.low_y for c in candles)
        a = (price_low - price_high) / max(y_lo - y_hi, 1)
        b = price_high - a * y_hi
        calib = {"method": "manual", "a": a, "b": b}
    else:
        fit = calibrate_ocr(img, plot_area, candles, hlines, theme == "dark") if use_ocr else None
        if fit:
            calib = {"method": "ocr", **fit}
        else:
            y_hi = min(c.high_y for c in candles)
            y_lo = max(c.low_y for c in candles)
            a = -100.0 / max(y_lo - y_hi, 1)
            b = -a * y_lo
            calib = {"method": "relative", "a": a, "b": b}
            warnings.append("Price axis not readable: prices are on a relative 0-100 scale. "
                            "Enter the chart's high and low prices for real values.")

    a, b = calib["a"], calib["b"]
    to_price = lambda y: a * (y + 0.5) + b  # noqa: E731  (+0.5 = pixel centre)
    rows = []
    for c in candles:
        rows.append({
            "Open": to_price(c.open_y), "High": to_price(c.high_y - 0.5),
            "Low": to_price(c.low_y + 0.5), "Close": to_price(c.close_y),
            "Bullish": c.bullish, "x": c.xc,
        })
    # Body edges: use the outer edge of the body pixel row
    for r, c in zip(rows, candles):
        top = to_price(c.body_top_y - 0.5)
        bot = to_price(c.body_bottom_y + 0.5)
        r["Open"], r["Close"] = (bot, top) if c.bullish else (top, bot)
    ohlc = pd.DataFrame(rows)

    res = DigitizeResult(ohlc=ohlc, candles=candles, theme=theme, plot_area=plot_area,
                         h_gridlines=hlines, calibration=calib, warnings=warnings, mode=mode)
    if draw_overlay:
        res.overlay = draw_debug(img, res)
    if keep_masks:
        res.masks = {"green": green, "red": red, "grid": gridmask, "candles": colour}
    return res


def draw_debug(img: np.ndarray, res: DigitizeResult) -> np.ndarray:
    out = img.copy()
    dark = res.theme == "dark"
    accent = (255, 200, 0)
    x0, y0, x1, y1 = res.plot_area
    cv2.rectangle(out, (x0, y0), (x1, y1), (200, 120, 255), 1)
    for y in res.h_gridlines:
        cv2.line(out, (x0, y), (x1, y), (160, 160, 60) if dark else (200, 200, 120), 1, cv2.LINE_AA)
    for i, c in enumerate(res.candles):
        col = (80, 255, 80) if c.bullish else (80, 80, 255)
        cv2.rectangle(out, (c.x0 - 1, c.body_top_y - 1), (c.x1 + 1, c.body_bottom_y + 1), accent, 1)
        cv2.line(out, (int(c.xc), c.high_y), (int(c.xc), c.low_y), col, 1)
        cv2.circle(out, (int(c.xc), c.high_y), 2, accent, -1)
        cv2.circle(out, (int(c.xc), c.low_y), 2, accent, -1)
    for p in res.calibration.get("points", []):
        # mark OCR'd label positions just inside the plot so labels stay readable
        cv2.drawMarker(out, (x1 - 6 if res.calibration.get("side") == "right" else x0 + 6,
                             int(p["y"])), (0, 165, 255), cv2.MARKER_DIAMOND, 8, 2)
    label = f"{len(res.candles)} candles | calib: {res.calibration.get('method')}"
    cv2.putText(out, label, (x0 + 8, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, label, (x0 + 8, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                accent, 1, cv2.LINE_AA)
    return out
