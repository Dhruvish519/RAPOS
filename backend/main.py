from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import os
import base64
import functools
import time
import finnhub
from typing import List, Optional

import cv2
import optimizer
import market_data
import portfolio
import forecast
import lstm_numpy
from cv.chart_digitizer import digitize, OCR_AVAILABLE

# Initialize FastAPI app
app = FastAPI(title="RAPOS AI Prediction Service", version="3.0.0")

# --- MARKET DATA CLIENT ---
# The key is read ONLY from the environment (set FINNHUB_API_KEY on Render).
# Never commit API keys to the repository.
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
finnhub_client = finnhub.Client(api_key=FINNHUB_KEY) if FINNHUB_KEY else None
if not FINNHUB_KEY:
    print("RAPOS WARNING: FINNHUB_API_KEY not set - live market endpoints will return errors.")

market_data.set_finnhub(finnhub_client)

# Popular symbols whose history is fetched at startup, so the first portfolio
# analysis doesn't wait on the network for them
PREWARM = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "SPY",
           "QQQ", "TLT", "GLD", "BND", "BTC", "ETH", "INR=X", "RELIANCE.NS", "TCS.NS",
           "HDFCBANK.NS", "INFY.NS", "NIFTYBEES.NS"]


def _finnhub():
    if finnhub_client is None:
        raise RuntimeError("Market data unavailable: FINNHUB_API_KEY is not configured")
    return finnhub_client


# --- CORS SETUP ---
# Browsers reject allow_origins=["*"] together with credentials, so list the
# real origins. ALLOWED_ORIGINS (comma-separated) is set on Render; Vercel
# preview deployments (rapos-<hash>.vercel.app) are allowed by pattern.
DEFAULT_ORIGINS = "https://rapos-sable.vercel.app,http://localhost:5500,http://127.0.0.1:5500"
ALLOWED_ORIGINS = [o.strip().rstrip("/")
                   for o in os.getenv("ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()]
ALLOWED_ORIGIN_REGEX = r"https://rapos-.*\.vercel\.app"

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL LOADING ---
# The LSTM is trained with PyTorch (train_lstm.py) and served with NumPy
# (lstm_numpy.py): same outputs, ~300 MB less RAM, fits Render's free plan.
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'data', 'scaler_params.npy')
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'simulated_stocks.csv')

loaded_model = None
scaler_scale = None
scaler_min = None
SEQUENCE_LENGTH = 60

@app.on_event("startup")
def load_assets():
    global loaded_model, scaler_scale, scaler_min, SEQUENCE_LENGTH
    try:
        loaded_model = lstm_numpy.load()
        if loaded_model is not None and os.path.exists(SCALER_PATH):
            SEQUENCE_LENGTH = loaded_model.sequence_length
            scaler_scale, scaler_min = np.load(SCALER_PATH)[:2]
            print("RAPOS AI: LSTM weights loaded (NumPy inference).")
        else:
            loaded_model = None
            print("RAPOS AI WARNING: Model files not found. Running in simulation fallback mode.")
    except Exception as e:
        loaded_model = None
        print(f"RAPOS AI Error loading model: {e}")
    market_data.prewarm(PREWARM)

# --- AI ENDPOINTS ---

@app.get("/")
async def root():
    return {"message": "RAPOS AI Service is online", "integration": "Supabase Auth Hybrid"}

@app.get("/api/predict-risk")
def predict_risk(ticker: str = 'SPXX'):
    """
    LSTM next-close forecast + volatility-based risk score (1-10) for one of
    the simulated assets the model was trained on (see generate_stock_data.py).
    Unknown tickers get an explanatory error instead of made-up numbers.
    """
    global loaded_model, scaler_scale, scaler_min
    ticker = ticker.upper()
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        return {"ticker": ticker, "error": f"Price history unavailable: {e}", "ai_risk_score": None}
    available = sorted(df['Ticker'].unique().tolist())

    if loaded_model is None:
        return {"ticker": ticker, "error": "LSTM model is not loaded on the server.",
                "ai_risk_score": None, "available_tickers": available}

    df_ticker = df[df['Ticker'] == ticker]
    if len(df_ticker) < SEQUENCE_LENGTH:
        return {"ticker": ticker, "ai_risk_score": None, "available_tickers": available,
                "error": f"No price history for {ticker}. The LSTM covers the simulated assets "
                         f"{', '.join(available)}."}

    try:
        recent_data = df_ticker['Close'].values[-SEQUENCE_LENGTH:]
        scaled_inputs = recent_data * scaler_scale + scaler_min
        scaled_prediction = float(loaded_model.predict(scaled_inputs.reshape(1, SEQUENCE_LENGTH, 1))[0, 0])

        predicted_price = (scaled_prediction - scaler_min) / scaler_scale
        recent_returns = pd.Series(recent_data).pct_change().dropna()
        annual_volatility = recent_returns.std() * np.sqrt(252) * 100
        last_close = recent_data[-1]
        predicted_return = ((predicted_price - last_close) / last_close) * 100
        risk_score = min(max((annual_volatility / 3.0), 1.0), 10.0)

        return {
            "ticker": ticker,
            "last_close_price": round(float(last_close), 2),
            "predicted_next_close": round(float(predicted_price), 2),
            "predicted_daily_return_pct": round(float(predicted_return), 2),
            "historical_volatility_pct": round(float(annual_volatility), 2),
            "ai_risk_score": round(float(risk_score), 1),
            "risk_score_method": "annualised volatility / 3, clipped to 1-10",
            "model_type": "LSTM Neural Network",
            "is_simulated": False,      # a real model output (on simulated price data)
            "data_source": "simulated",
        }
    except Exception as e:
        return {"ticker": ticker, "error": f"Prediction failed: {e}", "ai_risk_score": None}

# --- REAL-TIME MARKET DATA ---

@app.get("/api/live-price")
def get_live_price(ticker: str):
    """Current price and day change (Finnhub, Yahoo fallback; cached 30 s)."""
    try:
        q = market_data.quote(ticker)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "ticker": q["ticker"],
        "current_price": round(q["price"], 2),
        "change": round(q["change"], 2),
        "percent_change": round(q["percent_change"], 2),
        "high": round(q["high"], 2),
        "low": round(q["low"], 2),
        "previous_close": round(q["previous_close"], 2),
        "source": q["source"],
    }


@app.get("/api/history")
def get_history(ticker: str, days: int = 365):
    """Daily closes in the listing's own currency, for the Market Hub chart
    (served from the same cache the optimiser uses, so it is usually instant)."""
    days = max(20, min(days, 730))
    try:
        ser = market_data.history(ticker, 2)
    except LookupError as e:
        raise HTTPException(404, str(e))
    ser = ser[ser.index >= ser.index.max() - pd.Timedelta(days=days)]
    rets = np.diff(np.log(ser.values))
    peak = np.maximum.accumulate(ser.values)
    ppy = 365 if market_data.is_crypto(ticker) else 252
    return {
        "ticker": market_data.normalise(ticker),
        "currency": market_data.currency(ticker),
        "dates": [d.strftime("%Y-%m-%d") for d in ser.index],
        "closes": [round(float(x), 4) for x in ser.values],
        "stats": {
            "return_pct": round(float((ser.iloc[-1] / ser.iloc[0] - 1) * 100), 2),
            "volatility_pct": round(float(rets.std(ddof=1) * np.sqrt(ppy) * 100), 2) if len(rets) > 1 else None,
            "max_drawdown_pct": round(float(((ser.values - peak) / peak).min() * 100), 2),
            "high": round(float(ser.max()), 4), "low": round(float(ser.min()), 4),
        },
    }


@app.get("/api/quotes")
def get_quotes(tickers: str):
    """Batch live prices: /api/quotes?tickers=AAPL,MSFT,BTC (fetched in parallel)."""
    syms = [t for t in tickers.split(",") if t.strip()][:40]
    if not syms:
        raise HTTPException(400, "Pass tickers=AAPL,MSFT,...")
    return {"quotes": market_data.quotes(syms)}


@app.get("/api/search-stocks")
def search_stocks(q: str):
    """Search US stocks, ETFs and major crypto by company name or ticker (Yahoo, Finnhub fallback)."""
    if not q.strip():
        return {"query": q, "results": []}
    return {"query": q, "results": market_data.search(q[:40])}

# --- PORTFOLIO OPTIMIZATION ---

# Asset classes used by the dashboard (same four as its offline fallback in
# js/api.js), and which optimiser assets feed each one
DASHBOARD_GROUPS = {"Stocks": ["EQUITY", "TECH"], "Bonds": ["BONDS"],
                    "Crypto": ["CRYPTO"], "Commodities": ["GOLD"]}


@functools.lru_cache(maxsize=16)
def _cached_optimise(profile: str, risk_free: float, max_weight: float, dashboard: bool):
    prices = optimizer.load_universe()
    # The dashboard has no cash class, so its portfolios are fully invested
    return optimizer.optimise(prices, profile=profile, risk_free=risk_free, max_weight=max_weight,
                              include_cash=not dashboard)


@app.get("/api/rebalance")
def rebalance_portfolio(total_value: float, profile: str = "Moderate"):
    """
    Target dollar amount per asset class for a risk profile, computed with
    mean-variance optimisation (max return at the profile's target volatility).
    Response format is unchanged from v2 so the dashboard keeps working.
    """
    if profile not in optimizer.PROFILE_TARGET_VOL:
        profile = "Moderate"
    res = _cached_optimise(profile, 0.04, 0.6, True)
    weights = res["recommended"]["weights"]
    report = []
    for group, members in DASHBOARD_GROUPS.items():
        w = sum(weights.get(m, 0.0) for m in members)
        report.append({  # largest allocation first
            "asset_type": group,
            "target_weight": f"{round(w * 100)}%",
            "target_value": round(total_value * w, 2),
            "description": f"Targeting {w:.0%} allocation for {group}.",
        })
    report.sort(key=lambda r: -r["target_value"])
    rec = res["recommended"]
    return {
        "profile": profile,
        "total_value": total_value,
        "optimization_report": report,
        "expected_return": rec["expected_return"],
        "volatility": rec["volatility"],
        "sharpe": rec["sharpe"],
        "method": "Mean-variance (Markowitz), Ledoit-Wolf covariance",
        "advice": (f"Optimised for {profile} profile: expected return {rec['expected_return']:.1%}, "
                   f"volatility {rec['volatility']:.1%}. Adjust holdings to match target values."),
    }


# --- PORTFOLIO OPTIMISER (mean-variance) ---

@app.get("/api/optimize")
def optimize_default(profile: str = "Moderate", risk_free: float = 0.04, max_weight: float = 0.6):
    """Optimise the built-in multi-asset universe for a risk profile."""
    if not 0 <= risk_free < 0.5 or not 0.1 <= max_weight <= 1:
        raise HTTPException(400, "risk_free must be in [0, 0.5) and max_weight in [0.1, 1]")
    return _cached_optimise(profile, round(risk_free, 4), round(max_weight, 3), False)


class CustomAsset(BaseModel):
    name: str = Field(..., min_length=1, max_length=20)
    closes: List[float] = Field(..., min_length=20)


class OptimizeRequest(BaseModel):
    profile: str = "Moderate"
    risk_free: float = 0.04
    max_weight: float = 0.6
    custom_assets: List[CustomAsset] = []
    include_universe: bool = True


@app.post("/api/optimize")
def optimize_custom(req: OptimizeRequest):
    """
    Optimise with extra assets supplied as daily close prices, e.g. a series
    read from a chart screenshot by Chart Vision. The universe is aligned to
    the most recent N days, where N is the shortest custom series.
    """
    if not req.custom_assets and req.include_universe:
        return optimize_default(req.profile, req.risk_free, req.max_weight)
    if not 0 <= req.risk_free < 0.5 or not 0.1 <= req.max_weight <= 1:
        raise HTTPException(400, "risk_free must be in [0, 0.5) and max_weight in [0.1, 1]")
    n = min(len(a.closes) for a in req.custom_assets) if req.custom_assets else None
    frames = {}
    if req.include_universe:
        uni = optimizer.load_universe()
        uni = uni.tail(n) if n else uni
        frames.update({c: uni[c].values for c in uni.columns})
    for a in req.custom_assets:
        name = a.name.upper().replace(" ", "_")
        if name in frames or name == "CASH":
            name += "_X"
        c = np.asarray(a.closes[-n:], float)
        if np.any(~np.isfinite(c)) or np.any(c <= 0):
            raise HTTPException(400, f"{a.name}: prices must be positive numbers")
        frames[name] = c
    if len(frames) < 2:
        raise HTTPException(400, "Need at least two assets to optimise")
    prices = pd.DataFrame(frames)
    try:
        return optimizer.optimise(prices, profile=req.profile, risk_free=req.risk_free,
                                  max_weight=req.max_weight)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))


# --- REAL PORTFOLIO: risk analysis + optimisation ---

class Holding(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=15)
    quantity: Optional[float] = Field(None, gt=0)   # shares (or dollars for CASH)
    value: Optional[float] = Field(None, gt=0)      # ...or a dollar amount


class PortfolioRequest(BaseModel):
    holdings: List[Holding] = Field(..., min_length=1, max_length=30)
    profile: str = "Moderate"
    risk_free: float = 0.04
    max_weight: float = 0.6
    years: int = Field(2, ge=1, le=5)


@app.post("/api/portfolio/analyze")
def analyze_portfolio(req: PortfolioRequest):
    """
    Risk-aware optimisation of the user's real holdings: current risk
    (volatility, VaR, drawdown, risk contributions), the optimal mix for the
    risk profile over the same assets plus cash, and the trades to get there.
    Uses real daily prices (Yahoo/Stooq) and live quotes (Finnhub/Yahoo).
    """
    if not 0 <= req.risk_free < 0.5 or not 0.1 <= req.max_weight <= 1:
        raise HTTPException(400, "risk_free must be in [0, 0.5) and max_weight in [0.1, 1]")
    if req.profile not in optimizer.PROFILE_TARGET_VOL:
        raise HTTPException(400, f"profile must be one of {list(optimizer.PROFILE_TARGET_VOL)}")
    if any(h.quantity is None and h.value is None for h in req.holdings):
        raise HTTPException(400, "Each holding needs a quantity or a value")
    holdings = [{"ticker": market_data.normalise(h.ticker), "quantity": h.quantity, "value": h.value}
                for h in req.holdings]
    syms = [h["ticker"] for h in holdings if h["ticker"] != "CASH"]
    if not syms:
        raise HTTPException(422, "Add at least one stock, ETF or crypto holding to analyse.")
    hist_f = market_data._pool.submit(market_data.histories, syms, req.years)
    live = market_data.quotes(syms)           # runs while history downloads
    prices, missing = hist_f.result()
    if not any(h["ticker"] in prices.columns for h in holdings):
        raise HTTPException(422, {"message": "Couldn't get price history for any holding.",
                                  "missing": missing})
    try:
        out = portfolio.analyse(holdings, prices, {k: v for k, v in live.items() if "error" not in v},
                                profile=req.profile, risk_free=req.risk_free, max_weight=req.max_weight)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(422, str(e))
    out["missing"] = missing
    return out


# --- CHART VISION (OpenCV) ---

MAX_UPLOAD_BYTES = 8 * 1024 * 1024


def _png_b64(img: np.ndarray, max_w: int = 1400) -> str:
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode()


@app.post("/api/digitize-chart")
async def digitize_chart(
    image: UploadFile = File(..., description="Screenshot of a candlestick chart"),
    price_high: Optional[float] = Form(None, description="Highest wick price visible (optional)"),
    price_low: Optional[float] = Form(None, description="Lowest wick price visible (optional)"),
    interval: str = Form("1D", description="Candle interval: 1H, 4H, 1D, 1W, 1M"),
    use_ocr: bool = Form(True),
):
    """
    Read OHLC candles from a chart screenshot with OpenCV, then run RAPOS risk
    analytics and the LSTM forecast on the extracted closing prices.
    """
    data = await image.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large (max 8 MB)")
    if (price_high is None) != (price_low is None):
        raise HTTPException(400, "Give both price_high and price_low, or neither")
    if price_high is not None and price_high <= price_low:
        raise HTTPException(400, "price_high must be greater than price_low")
    try:
        res = digitize(data, price_high=price_high, price_low=price_low, use_ocr=use_ocr)
    except ValueError as e:
        raise HTTPException(422, str(e))

    ohlc = res.ohlc
    candles = [{"index": i, "open": round(float(r.Open), 4), "high": round(float(r.High), 4),
                "low": round(float(r.Low), 4), "close": round(float(r.Close), 4),
                "bullish": bool(r.Bullish)} for i, r in enumerate(ohlc.itertuples())]
    # Returns (and so risk) depend on the real price level, so analytics run
    # only when the price axis was calibrated (OCR or user-entered high/low).
    analysis = None
    if res.calibration.get("method") != "relative" and len(ohlc) >= 5:
        analysis = forecast.analyse(ohlc.Close.values, interval)
    calib = {k: v for k, v in res.calibration.items() if k in ("method", "inliers", "side", "max_residual")}
    return {
        "summary": {**res.summary(), "calibration": calib, "ocr_available": OCR_AVAILABLE},
        "candles": candles,
        "analysis": analysis,
        "overlay": _png_b64(res.overlay),
    }


@app.post("/api/analyze-series")
def analyze_series(closes: List[float], interval: str = "1D"):
    """Risk metrics + LSTM forecast for any list of closing prices."""
    try:
        return forecast.analyse(closes, interval)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/health")
def health():
    return {"status": "ok", "ocr": OCR_AVAILABLE, "lstm": forecast.load_model() is not None,
            "market_data": finnhub_client is not None,
            "history_cached": len(market_data._hist_cache),
            "history_sources": sorted(set(market_data.last_source.values()))}


# --- WEALTH HISTORY & NEWS ---

@app.get("/api/market-discover")
def market_discover():
    """Today's biggest movers among a fixed watch-list (live quotes, fetched in parallel)."""
    watch = ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMD", "AMZN", "META", "BTC", "ETH"]
    qs = [q for q in market_data.quotes(watch).values() if "error" not in q]
    movers = sorted(qs, key=lambda q: -q["percent_change"])[:4]
    return {"discovery": [{"ticker": q["ticker"], "price": round(q["price"], 2),
                           "change": round(q["percent_change"], 2),
                           "sentiment": "Up today" if q["percent_change"] > 0 else "Down today"}
                          for q in movers],
            "intelligence_note": "Largest percentage moves today in the RAPOS watch-list."}

_news_cache: dict = {}


@app.get("/api/market-news")
def get_market_news(category: str = "general"):
    """Latest market headlines from Finnhub (cached 5 minutes)."""
    hit = _news_cache.get(category)
    if hit and time.time() - hit[0] < 300:
        return hit[1]
    try:
        news = _finnhub().general_news(category, min_id=0)
        out = {"news": [{k: n.get(k) for k in ("headline", "source", "datetime", "url", "summary")}
                        for n in news[:15]]}
        _news_cache[category] = (time.time(), out)
        return out
    except Exception as e:
        return hit[1] if hit else {"news": [], "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
