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


# ------------------------------------------------------------------ LSTM serving
def test_numpy_lstm_matches_pytorch():
    torch = pytest.importorskip("torch")   # training-only dependency
    import lstm_numpy
    from export_lstm_weights import HERE
    ck = torch.load(os.path.join(HERE, "lstm_model.pth"), map_location="cpu")
    lstm = torch.nn.LSTM(ck["input_size"], ck["hidden_size"], ck["num_layers"], batch_first=True)
    fc = torch.nn.Linear(ck["hidden_size"], ck["output_size"])
    sd = ck["model_state_dict"]
    lstm.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("lstm.")})
    fc.load_state_dict({k[3:]: v for k, v in sd.items() if k.startswith("fc.")})
    x = np.random.default_rng(1).uniform(0, 1, (8, 60, 1))
    with torch.no_grad():
        out, _ = lstm(torch.tensor(x, dtype=torch.float32))
        ref = fc(out[:, -1]).numpy()
    assert np.abs(lstm_numpy.NumpyLSTM().predict(x) - ref).max() < 1e-5


def test_api_predict_risk_uses_model(client):
    j = client.get("/api/predict-risk", params={"ticker": "SPXX"}).json()
    assert j["is_simulated"] is False and j["model_type"] == "LSTM Neural Network"


def test_api_predict_risk_unknown_ticker_gives_no_fake_score(client):
    scores = {client.get("/api/predict-risk", params={"ticker": "AAPL"}).json().get("ai_risk_score")
              for _ in range(3)}
    j = client.get("/api/predict-risk", params={"ticker": "AAPL"}).json()
    assert scores == {None} and "SPXX" in j["available_tickers"] and j["error"]


def test_cors_allows_site_and_previews_only(client):
    def preflight(origin):
        return client.options("/api/health", headers={
            "Origin": origin, "Access-Control-Request-Method": "GET"})
    for ok in ("https://rapos-sable.vercel.app", "http://localhost:5500",
               "https://rapos-git-feature-dhruvish519s-projects.vercel.app"):
        r = preflight(ok)
        assert r.headers.get("access-control-allow-origin") == ok
        assert r.headers.get("access-control-allow-credentials") == "true"
    r = preflight("https://evil.example.com")
    assert "access-control-allow-origin" not in r.headers


# ------------------------------------------------------------- real portfolio
def _fake_prices(seed=1, n=300):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    vols = {"AAA": 0.035, "BBB": 0.015, "CCC": 0.006}       # risky, medium, calm
    return pd.DataFrame({t: 100 * np.exp(np.cumsum(rng.normal(0.0004, v, n)))
                         for t, v in vols.items()}, index=idx)


def test_portfolio_analysis_risk_and_trades():
    import portfolio
    prices = _fake_prices()
    holdings = [{"ticker": "AAA", "quantity": 10}, {"ticker": "BBB", "quantity": 5},
                {"ticker": "aaa", "quantity": 5}]                       # same ticker summed
    live = {"AAA": {"price": float(prices.AAA.iloc[-1]), "percent_change": 1.0, "source": "test"}}
    out = portfolio.analyse(holdings, prices, live, profile="Conservative")
    assert {h["ticker"] for h in out["holdings"]} == {"AAA", "BBB"}       # CCC not held: ignored
    assert abs(sum(h["weight"] for h in out["holdings"]) - 1) < 1e-3
    cur = out["current"]
    assert cur["volatility"] > out["target_volatility"]                   # concentrated in AAA
    assert out["recommended"]["volatility"] <= out["target_volatility"] + 1e-3
    assert 0 < cur["var_95_1d_pct"] < 0.2 and cur["max_drawdown"] < 0
    assert abs(sum(cur["risk_contribution"].values()) - 1) < 1e-3
    sells = {t["asset"]: t for t in out["trades"] if t["action"] == "SELL"}
    assert "AAA" in sells and sells["AAA"]["shares"] > 0                  # de-risk the volatile stock
    total_after = out["total_value"] + sum(t["amount"] * (1 if t["action"] == "BUY" else -1)
                                           for t in out["trades"])
    assert abs(total_after - out["total_value"]) < 0.02 * out["total_value"]   # trades are self-funding


def test_api_portfolio_analyze(client, monkeypatch):
    import market_data
    prices = _fake_prices(2)
    monkeypatch.setattr(market_data, "histories",
                        lambda syms, years=2: (prices[[s for s in syms if s in prices]],
                                               {s: "no data" for s in syms if s not in prices}))
    monkeypatch.setattr(market_data, "quotes", lambda syms: {s: {"ticker": s, "error": "offline"} for s in syms})
    r = client.post("/api/portfolio/analyze", json={
        "holdings": [{"ticker": "AAA", "quantity": 3}, {"ticker": "CCC", "quantity": 20},
                     {"ticker": "ZZZ", "quantity": 1}], "profile": "Aggressive"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["missing"] == {"ZZZ": "no data"} and j["profile"] == "Aggressive"
    assert set(j["recommended"]["weights"]) == {"AAA", "CCC", "CASH"}
    assert client.post("/api/portfolio/analyze", json={"holdings": [{"ticker": "ZZZ", "quantity": 1}]}).status_code == 422
    assert client.post("/api/portfolio/analyze", json={"holdings": [{"ticker": "AAA", "quantity": 1}],
                                                       "profile": "YOLO"}).status_code == 400


def test_symbol_normalisation():
    import market_data as md
    assert md.normalise(" btc-usd ") == "BTC" and md.normalise("ETHUSDT") == "ETH"
    assert md.normalise("aapl") == "AAPL" and md.normalise("USDC") == "USDC"


def test_portfolio_cash_and_value_holdings():
    import portfolio
    prices = _fake_prices(3)
    out = portfolio.analyse([{"ticker": "AAA", "value": 1000}, {"ticker": "BBB", "value": 1000},
                             {"ticker": "CASH", "quantity": 2000}], prices, {}, profile="Moderate")
    w = {h["ticker"]: h["weight"] for h in out["holdings"]}
    assert abs(w["AAA"] - 0.25) < 1e-3 and abs(w["CASH"] - 0.5) < 1e-3
    assert out["current"]["volatility"] < 0.5 * out["assets"][0]["volatility"] + 0.2


def test_chart_analysis_reports_range_not_direction():
    import forecast
    rng = np.random.default_rng(5)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400)))
    a = forecast.analyse(closes)
    r = a["next_range"]
    assert r["low_95"] < r["low_68"] < a["last_close"] < r["high_68"] < r["high_95"]
    assert 92 <= r["past_candles_inside_95_pct"] <= 98        # band is calibrated
    assert "lstm" not in a and a["var_95_pct"] > 0
