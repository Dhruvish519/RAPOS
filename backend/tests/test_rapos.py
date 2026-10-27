"""
Automated tests for Chart Vision, the optimiser and the API.
Run from backend/:  python -m pytest -q
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optimizer  # noqa: E402
from cv.chart_digitizer import digitize, OCR_AVAILABLE  # noqa: E402
from cv.synthetic_charts import THEMES, random_ohlc, render_chart  # noqa: E402


def _mape(a, b):
    return float(np.mean(np.abs(a - b) / b) * 100)


# ----------------------------------------------------------------- Chart Vision
@pytest.mark.parametrize("theme", list(THEMES))
def test_counts_candles_in_every_theme(theme):
    df = random_ohlc(n=50, seed=4)
    res = digitize(render_chart(df, theme=theme), use_ocr=False)
    assert len(res.candles) == 50


def test_up_down_direction_matches_truth():
    df = random_ohlc(n=60, seed=8)
    res = digitize(render_chart(df, theme="dark"), use_ocr=False)
    assert (res.ohlc.Bullish.values == (df.Close >= df.Open).values).all()


def test_manual_calibration_gives_real_prices():
    df = random_ohlc(n=40, seed=9)
    res = digitize(render_chart(df), price_high=df.High.max(), price_low=df.Low.min(), use_ocr=False)
    assert res.calibration["method"] == "manual"
    assert _mape(res.ohlc.Close.values, df.Close.values) < 0.5


@pytest.mark.skipif(not OCR_AVAILABLE, reason="Tesseract not installed")
def test_ocr_reads_price_axis():
    df = random_ohlc(n=60, start=487.0, seed=12)
    res = digitize(render_chart(df, theme="light"))
    assert res.calibration["method"] == "ocr"
    assert _mape(res.ohlc.Close.values, df.Close.values) < 0.5


def test_survives_jpeg_compression():
    df = random_ohlc(n=45, seed=3)
    ok, enc = cv2.imencode(".jpg", render_chart(df, theme="dark"), [cv2.IMWRITE_JPEG_QUALITY, 60])
    res = digitize(enc.tobytes(), price_high=df.High.max(), price_low=df.Low.min(), use_ocr=False)
    assert len(res.candles) == 45


def test_uncalibrated_falls_back_to_relative_scale():
    df = random_ohlc(n=30, seed=1)
    res = digitize(render_chart(df), use_ocr=False)
    assert res.calibration["method"] == "relative"
    assert res.warnings


def test_rejects_non_chart_image():
    blank = np.full((400, 600, 3), 255, np.uint8)
    with pytest.raises(ValueError):
        digitize(blank)
    with pytest.raises(ValueError):
        digitize(b"not an image")


# -------------------------------------------------------------------- Optimiser
@pytest.fixture(scope="module")
def universe():
    return optimizer.load_universe()


def test_weights_valid_and_profiles_ordered(universe):
    vols = []
    for p in ["Conservative", "Moderate", "Aggressive"]:
        r = optimizer.optimise(universe, profile=p, max_weight=0.6)
        w = np.array(list(r["recommended"]["weights"].values()))
        assert abs(w.sum() - 1) < 1e-3 and (w >= -1e-9).all() and (w <= 0.6 + 1e-6).all()
        assert r["recommended"]["volatility"] <= r["target_volatility"] + 1e-3
        vols.append(r["recommended"]["volatility"])
    assert vols[0] < vols[1] < vols[2]


def test_max_sharpe_beats_frontier_points(universe):
    r = optimizer.optimise(universe, include_cash=False)
    best = max(f["sharpe"] for f in r["frontier"])
    assert r["max_sharpe"]["sharpe"] >= best - 1e-3


def test_min_variance_is_lowest_risk(universe):
    r = optimizer.optimise(universe, include_cash=False)
    assert r["min_variance"]["volatility"] <= min(f["volatility"] for f in r["frontier"]) + 1e-4


def test_risk_contributions_sum_to_one(universe):
    r = optimizer.optimise(universe)
    assert abs(sum(r["recommended"]["risk_contribution"].values()) - 1) < 1e-3


def test_rebalance_trades():
    t = optimizer.rebalance_trades({"A": 600, "B": 400}, {"A": 0.5, "B": 0.5})
    assert {x["asset"]: x["action"] for x in t} == {"A": "SELL", "B": "BUY"}
    assert all(x["amount"] == 100 for x in t)


# -------------------------------------------------------------------------- API
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    import main
    with TestClient(main.app) as c:
        yield c


def test_api_rebalance_keeps_v2_format(client):
    j = client.get("/api/rebalance", params={"total_value": 10000, "profile": "Moderate"}).json()
    groups = sorted(r["asset_type"] for r in j["optimization_report"])
    assert groups == ["Bonds", "Commodities", "Crypto", "Stocks"]   # same as the js/api.js fallback
    assert abs(sum(r["target_value"] for r in j["optimization_report"]) - 10000) < 5


def test_api_digitize_and_optimise_custom(client):
    df = random_ohlc(n=40, seed=5)
    ok, enc = cv2.imencode(".png", render_chart(df, theme="dark"))
    r = client.post("/api/digitize-chart", files={"image": ("c.png", enc.tobytes(), "image/png")},
                    data={"price_high": str(df.High.max()), "price_low": str(df.Low.min()), "use_ocr": "false"})
    assert r.status_code == 200
    j = r.json()
    assert j["summary"]["n_candles"] == 40 and j["analysis"]["ai_risk_score"] >= 1
    assert j["overlay"].startswith("data:image/jpeg;base64,")
    closes = [c["close"] for c in j["candles"]]
    r = client.post("/api/optimize", json={"profile": "Moderate", "custom_assets": [{"name": "Mine", "closes": closes}]})
    assert r.status_code == 200 and "MINE" in r.json()["recommended"]["weights"]


def test_api_validation(client):
    assert client.post("/api/digitize-chart", files={"image": ("x.png", b"junk", "image/png")}).status_code == 422
    df = random_ohlc(n=30, seed=2)
    ok, enc = cv2.imencode(".png", render_chart(df))
    r = client.post("/api/digitize-chart", files={"image": ("c.png", enc.tobytes(), "image/png")},
                    data={"price_high": "100"})
    assert r.status_code == 400
    assert client.get("/api/optimize", params={"max_weight": 5}).status_code == 400
