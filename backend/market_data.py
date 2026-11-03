"""
Real market data for RAPOS: daily price history and live quotes.

* History (for risk metrics and optimisation): Yahoo Finance's public chart
  API, with Stooq's free CSV download as a fallback. Finnhub's free plan has
  no historical candles, so it can't be used here.
* Quotes (current price, day change): Finnhub when a key is set, otherwise
  Yahoo's chart API.

Everything is fetched in parallel and cached in memory (history for 6 h,
quotes for 30 s), so after the first request a portfolio refresh is
near-instant. Ticker symbols are plain US symbols (AAPL, MSFT); common crypto
symbols (BTC, ETH, ...) map to their USD pairs.
"""
from __future__ import annotations

import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

HISTORY_TTL = 6 * 3600
QUOTE_TTL = 30
TIMEOUT = 8
UA = {"User-Agent": "Mozilla/5.0 (RAPOS portfolio app; student project)"}

CRYPTO = {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "BNB", "AVAX", "DOT", "LTC", "MATIC"}

_pool = ThreadPoolExecutor(max_workers=12)
_session = requests.Session()
_session.headers.update(UA)
_lock = threading.Lock()
_hist_cache: dict[str, tuple[float, pd.Series]] = {}
_quote_cache: dict[str, tuple[float, dict]] = {}
_finnhub_client = None
last_source: dict[str, str] = {}       # symbol -> where its history came from (for /api/health)


def set_finnhub(client):
    global _finnhub_client
    _finnhub_client = client


def normalise(symbol: str) -> str:
    s = symbol.strip().upper().replace(" ", "")
    for suffix in ("-USD", "/USD", "USDT", "USD"):           # BTC-USD, BTCUSDT -> BTC
        if s.endswith(suffix) and s[: -len(suffix)] in CRYPTO:
            return s[: -len(suffix)]
    return s


def is_crypto(symbol: str) -> bool:
    return normalise(symbol) in CRYPTO


def _yahoo_symbol(s: str) -> str:
    return f"{s}-USD" if s in CRYPTO else s.replace(".", "-")   # BRK.B -> BRK-B


# --------------------------------------------------------------------------- history
def _yahoo_history(s: str, years: int) -> pd.Series:
    r = _session.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{_yahoo_symbol(s)}",
                     params={"range": f"{years}y", "interval": "1d", "includeAdjustedClose": "true"},
                     timeout=TIMEOUT)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]
    closes = (q.get("adjclose") or [{}])[0].get("adjclose") or q["quote"][0]["close"]
    ser = pd.Series(closes, index=pd.to_datetime(ts, unit="s").normalize(), dtype=float)
    return ser


def _stooq_history(s: str, years: int) -> pd.Series:
    sym = f"{s.lower()}usd" if s in CRYPTO else f"{s.lower().replace('.', '-')}.us"
    r = _session.get("https://stooq.com/q/d/l/", params={"s": sym, "i": "d"}, timeout=TIMEOUT)
    r.raise_for_status()
    if not r.text.startswith("Date"):
        raise ValueError("no data")
    df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"]).set_index("Date")
    ser = df["Close"].astype(float)
    return ser[ser.index >= ser.index.max() - pd.Timedelta(days=365 * years)]


def history(symbol: str, years: int = 2) -> pd.Series:
    """Daily closes (adjusted where available) for one symbol, cached."""
    s = normalise(symbol)
    now = time.time()
    with _lock:
        hit = _hist_cache.get(s)
    if hit and now - hit[0] < HISTORY_TTL:
        return hit[1]
    errors = []
    for name, fn in (("yahoo", _yahoo_history), ("stooq", _stooq_history)):
        try:
            ser = fn(s, years).dropna()
            ser = ser[~ser.index.duplicated(keep="last")].sort_index()
            if len(ser) >= 30:
                with _lock:
                    _hist_cache[s] = (now, ser)
                    last_source[s] = name
                return ser
            errors.append(f"{name}: only {len(ser)} days")
        except Exception as e:  # network error, unknown symbol, rate limit...
            errors.append(f"{name}: {type(e).__name__}")
    if hit:                      # stale data beats no data
        return hit[1]
    raise LookupError(f"No price history for {s} ({'; '.join(errors)})")


def histories(symbols: list[str], years: int = 2) -> tuple[pd.DataFrame, dict[str, str]]:
    """Aligned daily closes for several symbols (fetched in parallel).
    Returns (prices, errors) - symbols that failed are left out and reported."""
    syms = list(dict.fromkeys(normalise(s) for s in symbols))
    futs = {s: _pool.submit(history, s, years) for s in syms}
    series, errors = {}, {}
    for s, f in futs.items():
        try:
            series[s] = f.result()
        except Exception as e:
            errors[s] = str(e)
    if not series:
        return pd.DataFrame(), errors
    df = pd.DataFrame(series)
    # Crypto trades every day, stocks on weekdays: keep the days where every
    # stock traded (business days) and carry crypto prices across them.
    if any(not is_crypto(s) for s in series):
        df = df[df.index.dayofweek < 5]
    df = df.sort_index().ffill().dropna()
    return df, errors


# --------------------------------------------------------------------------- quotes
def _finnhub_quote(s: str) -> dict:
    sym = f"BINANCE:{s}USDT" if s in CRYPTO else s
    q = _finnhub_client.quote(sym)
    if not q or not q.get("c"):
        raise ValueError("no quote")
    return {"price": float(q["c"]), "change": float(q.get("d") or 0),
            "percent_change": float(q.get("dp") or 0), "previous_close": float(q.get("pc") or 0),
            "high": float(q.get("h") or 0), "low": float(q.get("l") or 0), "source": "finnhub"}


def _yahoo_quote(s: str) -> dict:
    r = _session.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{_yahoo_symbol(s)}",
                     params={"range": "5d", "interval": "1d"}, timeout=TIMEOUT)
    r.raise_for_status()
    m = r.json()["chart"]["result"][0]["meta"]
    price = float(m["regularMarketPrice"])
    prev = float(m.get("chartPreviousClose") or m.get("previousClose") or price)
    return {"price": price, "change": price - prev,
            "percent_change": (price / prev - 1) * 100 if prev else 0.0, "previous_close": prev,
            "high": float(m.get("regularMarketDayHigh") or 0), "low": float(m.get("regularMarketDayLow") or 0),
            "source": "yahoo"}


def quote(symbol: str) -> dict:
    s = normalise(symbol)
    now = time.time()
    with _lock:
        hit = _quote_cache.get(s)
    if hit and now - hit[0] < QUOTE_TTL:
        return hit[1]
    sources = ([_finnhub_quote] if _finnhub_client else []) + [_yahoo_quote]
    last_err = None
    for fn in sources:
        try:
            q = {"ticker": s, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in fn(s).items()}}
            with _lock:
                _quote_cache[s] = (now, q)
            return q
        except Exception as e:
            last_err = e
    if hit:
        return {**hit[1], "stale": True}
    raise LookupError(f"No live price for {s} ({type(last_err).__name__})")


def quotes(symbols: list[str]) -> dict[str, dict]:
    syms = list(dict.fromkeys(normalise(s) for s in symbols if s.strip()))
    futs = {s: _pool.submit(quote, s) for s in syms}
    out = {}
    for s, f in futs.items():
        try:
            out[s] = f.result()
        except Exception as e:
            out[s] = {"ticker": s, "error": str(e)}
    return out


def prewarm(symbols: list[str]):
    """Fill the history cache in the background (called at startup)."""
    for s in symbols:
        _pool.submit(lambda x=s: _safe(history, x))


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception:
        return None
