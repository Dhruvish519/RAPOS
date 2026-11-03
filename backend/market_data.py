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


# Indian listings use Yahoo's exchange suffixes: RELIANCE.NS (NSE), RELIANCE.BO (BSE)
INDIA_SUFFIXES = (".NS", ".BO")
FX_INR = "INR=X"          # Yahoo: rupees per US dollar


def currency(symbol: str) -> str:
    return "INR" if normalise(symbol).endswith(INDIA_SUFFIXES) else "USD"


def _yahoo_symbol(s: str) -> str:
    if s in CRYPTO:
        return f"{s}-USD"
    if s.endswith(INDIA_SUFFIXES) or s == FX_INR:
        return s
    return s.replace(".", "-")   # BRK.B -> BRK-B


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
    if s.endswith(INDIA_SUFFIXES) or s == FX_INR:
        raise ValueError("not on Stooq")
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
    inr = [s for s in series if currency(s) == "INR"]
    if inr:
        # Portfolio maths is in US dollars: convert rupee prices with the
        # daily USD/INR rate, so FX moves count as part of the asset's risk.
        try:
            fx = history(FX_INR, years)
            for s in inr:
                series[s] = (series[s] / fx.reindex(series[s].index, method="ffill")).dropna()
        except Exception as e:
            for s in inr:
                errors[s] = f"No USD/INR exchange rate ({type(e).__name__})"
                series.pop(s)
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
    """Live quote. 'price' is in US dollars; 'native_price' in the listing's
    own currency (INR for .NS/.BO), with 'currency' saying which."""
    s = normalise(symbol)
    now = time.time()
    with _lock:
        hit = _quote_cache.get(s)
    if hit and now - hit[0] < QUOTE_TTL:
        return hit[1]
    if currency(s) == "INR":
        try:
            q = _native_quote(s, [_yahoo_quote], None)
            fx = _native_quote(FX_INR, [_yahoo_quote], None)["price"]
        except LookupError:
            if hit:                      # cached quote is already in USD
                return {**hit[1], "stale": True}
            raise
        out = {**q, "currency": "INR", "native_price": q["price"], "fx_rate": round(fx, 4),
               "price": round(q["price"] / fx, 4), "change": round(q["change"] / fx, 4),
               "previous_close": round(q["previous_close"] / fx, 4),
               "high": round(q["high"] / fx, 4), "low": round(q["low"] / fx, 4)}
        with _lock:
            _quote_cache[s] = (now, out)
        return out
    q = _native_quote(s, ([_finnhub_quote] if _finnhub_client else []) + [_yahoo_quote], hit)
    q = {**q, "currency": "USD", "native_price": q["price"]}
    with _lock:
        _quote_cache[s] = (now, q)
    return q


def _native_quote(s: str, sources, hit) -> dict:
    now = time.time()
    if s == FX_INR:
        with _lock:
            fx_hit = _quote_cache.get(s)
        if fx_hit and now - fx_hit[0] < QUOTE_TTL:
            return fx_hit[1]
    last_err = None
    for fn in sources:
        try:
            q = {"ticker": s, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in fn(s).items()}}
            if s == FX_INR:
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
    out, failed = {}, []
    for s, f in futs.items():
        try:
            out[s] = f.result()
        except Exception:
            failed.append(s)
    # Yahoo throttles bursts of parallel requests: retry failures one by one
    for s in failed:
        try:
            out[s] = quote(s)
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


# --------------------------------------------------------------------------- search
_search_cache: dict[str, tuple[float, list]] = {}
US_EXCHANGES = {"NMS", "NGM", "NCM", "NYQ", "ASE", "PCX", "BTS", "NAS", "NYS", "NSQ", "PNK"}
CRYPTO_NAMES = {"BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "XRP": "XRP", "ADA": "Cardano",
                "DOGE": "Dogecoin", "BNB": "BNB", "AVAX": "Avalanche", "DOT": "Polkadot",
                "LTC": "Litecoin", "MATIC": "Polygon"}


def _yahoo_search(q: str) -> list[dict]:
    r = _session.get("https://query1.finance.yahoo.com/v1/finance/search",
                     params={"q": q, "quotesCount": 15, "newsCount": 0}, timeout=TIMEOUT)
    r.raise_for_status()
    out = []
    for it in r.json().get("quotes", []):
        sym, kind = it.get("symbol", ""), it.get("quoteType", "")
        name = it.get("longname") or it.get("shortname") or sym
        if kind == "CRYPTOCURRENCY" and sym.endswith("-USD") and sym[:-4] in CRYPTO:
            out.append({"symbol": sym[:-4], "name": CRYPTO_NAMES[sym[:-4]], "type": "Crypto"})
        elif kind in ("EQUITY", "ETF", "MUTUALFUND") and it.get("exchange") in US_EXCHANGES and "." not in sym:
            out.append({"symbol": sym, "name": name, "type": "ETF" if kind != "EQUITY" else "Stock",
                        "exchange": it.get("exchDisp") or it.get("exchange"), "currency": "USD"})
        elif kind in ("EQUITY", "ETF") and sym.endswith(INDIA_SUFFIXES):
            out.append({"symbol": sym, "name": name, "type": "ETF" if kind == "ETF" else "Stock",
                        "exchange": "NSE" if sym.endswith(".NS") else "BSE", "currency": "INR"})
    return out


def _finnhub_search(q: str) -> list[dict]:
    res = _finnhub_client.symbol_lookup(q).get("result", [])
    return [{"symbol": it["symbol"], "name": it["description"].title(),
             "type": "ETF" if "ETP" in it.get("type", "") or "ETF" in it.get("type", "") else "Stock"}
            for it in res if "." not in it["symbol"] and ":" not in it["symbol"]][:12]


def search(q: str) -> list[dict]:
    """Company / ETF / crypto search by name or ticker, US listings only (cached 1 h)."""
    key = q.strip().lower()
    if not key:
        return []
    now = time.time()
    hit = _search_cache.get(key)
    if hit and now - hit[0] < 3600:
        return hit[1]
    crypto = [{"symbol": s, "name": n, "type": "Crypto"} for s, n in CRYPTO_NAMES.items()
              if key in s.lower() or key in n.lower()]
    found = []
    for fn in [_yahoo_search] + ([_finnhub_search] if _finnhub_client else []):
        try:
            found = fn(q)
            break
        except Exception:
            continue
    seen, out = set(), []
    for it in crypto + found:
        if it["symbol"] not in seen:
            seen.add(it["symbol"]); out.append(it)
    _search_cache[key] = (now, out[:12])
    return out[:12]
