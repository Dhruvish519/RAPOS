"""
Forecasting and risk analytics for a price series (e.g. one read from a chart
image by Chart Vision).

The LSTM was trained on one simulated ticker in absolute price units, so a
digitised series (any price level) is mapped into the model's input range
window-by-window ("relative-scale transfer"), forecast, and mapped back.
Inference runs in NumPy (lstm_numpy.py); PyTorch is only needed for training.
"""
from __future__ import annotations

import os

import numpy as np

import lstm_numpy

SEQ_LEN = 60

# periods per year for annualising volatility, by candle interval
PERIODS_PER_YEAR = {"1H": 252 * 6.5, "4H": 252 * 6.5 / 4, "1D": 252, "1W": 52, "1M": 12}


def load_model():
    global SEQ_LEN
    m = lstm_numpy.load()
    if m is not None:
        SEQ_LEN = m.sequence_length
    return m


def lstm_next(closes: np.ndarray) -> tuple[float | None, str]:
    """Next-step forecast. Short series are left-padded with their first value."""
    model = load_model()
    if model is None or len(closes) < 10:
        return None, "unavailable"
    window = np.asarray(closes[-SEQ_LEN:], float)
    note = "full window"
    if len(window) < SEQ_LEN:
        window = np.concatenate([np.full(SEQ_LEN - len(window), window[0]), window])
        note = f"padded ({len(closes)} of {SEQ_LEN} candles)"
    lo, hi = window.min(), window.max()
    span = max(hi - lo, 1e-9)
    scaled = (window - lo) / span * 0.7 + 0.15   # into the model's familiar range
    y = float(model.predict(scaled.reshape(1, SEQ_LEN, 1))[0, 0])
    return float((y - 0.15) / 0.7 * span + lo), note


def analyse(closes, interval: str = "1D") -> dict:
    """Risk metrics + LSTM forecast for a close-price series."""
    c = np.asarray(closes, float)
    c = c[np.isfinite(c) & (c > 0)]
    if len(c) < 5:
        raise ValueError("Need at least 5 positive prices")
    rets = np.diff(np.log(c))
    ppy = PERIODS_PER_YEAR.get(interval, 252)
    vol = float(rets.std(ddof=1) * np.sqrt(ppy) * 100)
    peak = np.maximum.accumulate(c)
    max_dd = float(((c - peak) / peak).min() * 100)
    # trend: slope of a straight-line fit to log price, per candle
    slope = float(np.polyfit(np.arange(len(c)), np.log(c), 1)[0] * 100)
    downside = rets[rets < 0]
    sortino_den = downside.std(ddof=1) * np.sqrt(ppy) if len(downside) > 1 else np.nan
    ann_ret = float(rets.mean() * ppy)

    pred, note = lstm_next(c)
    risk_score = float(min(max(vol / 3.0, 1.0), 10.0))   # same scale as /api/predict-risk
    out = {
        "n_points": int(len(c)),
        "interval": interval,
        "last_close": round(float(c[-1]), 4),
        "period_return_pct": round(float((c[-1] / c[0] - 1) * 100), 2),
        "annualised_volatility_pct": round(vol, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "trend_pct_per_candle": round(slope, 3),
        "trend": "Uptrend" if slope > 0.05 else ("Downtrend" if slope < -0.05 else "Sideways"),
        "sortino": (round(float(ann_ret / sortino_den), 2)
                    if sortino_den and np.isfinite(sortino_den) and sortino_den > 0 else None),
        "ai_risk_score": round(risk_score, 1),
        "risk_band": "Low" if risk_score < 4 else ("Medium" if risk_score < 7 else "High"),
        "lstm": None,
    }
    if pred is not None:
        out["lstm"] = {
            "predicted_next_close": round(pred, 4),
            "predicted_change_pct": round(float((pred / c[-1] - 1) * 100), 3),
            "direction": "Up" if pred > c[-1] else "Down",
            "window": note,
            # On random-walk-like prices the best simple forecast is "no
            # change"; report it so users can judge the model fairly.
            "naive_baseline": round(float(c[-1]), 4),
            "caveat": "Experimental. Short-horizon price moves are close to random; "
                      "treat this as a signal, not a prediction.",
        }
    return out
