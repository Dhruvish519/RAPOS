# RAPOS: Risk-Aware Portfolio Optimisation System

**Live app:** https://rapos-sable.vercel.app  ·  **Chart Vision:** https://rapos-sable.vercel.app/chart-vision  ·  **Optimiser:** https://rapos-sable.vercel.app/optimizer

**Backend API:** https://rapos-api.onrender.com ([health check](https://rapos-api.onrender.com/api/health); the free instance can take up to a minute to wake after being idle)

RAPOS is a portfolio web app with an AI backend. Version 3 adds two modules:

| Module | What it does | Where |
|---|---|---|
| **Chart Vision** (OpenCV) | Reads a screenshot of any candlestick chart and rebuilds the OHLC price data, then scores its risk | `backend/cv/`, `chart-vision.html` |
| **Portfolio Optimiser** | Mean-variance (Markowitz) optimisation with an efficient frontier, replacing the old fixed allocation table | `backend/optimizer.py`, `optimizer.html` |

## Chart Vision pipeline

1. **Background**: the median colour of the image border gives the theme (light/dark).
2. **Colour masks**: HSV thresholds isolate rising (green) and falling (red) candles; tiny connected components are removed. Thin coloured horizontal lines (e.g. a dotted "last price" line) and volume histograms (bars standing on one shared baseline) are erased.
3. **Grid and plot area**: morphological opening with long line kernels finds gridlines; lines spanning most of the chart define the plot rectangle.
4. **Candles**: column-occupancy runs give each candle's x-extent; merged neighbours are split at profile minima; text-like blobs are rejected. The row-width profile separates the wide body from the thin wick. Grey wicks are traced through contrast, and evenly spaced gaps are checked for faint doji candles.
5. **Other colour schemes**: if the chart isn't green/red, candles are found by contrast, hollow bodies are filled, body colours are clustered (k-means in Lab), and which cluster means "up" is decided by price continuity (a candle usually opens near the previous close).
6. **Price axis**: each axis label is located, scaled to a fixed text height, and all labels are stacked into one image read by a single Tesseract call (falling back to per-label reading with size voting if the lines don't line up); the right-hand axis is tried first and the left only if needed; a RANSAC-style linear fit maps pixel rows to prices. If OCR isn't available, the user enters the chart's high and low prices.
7. **Analysis**: volatility, drawdown, trend, 95 % Value at Risk and the likely range of the next candle (from the chart's own volatility) on the extracted closes.

`python -m cv.evaluate` (from `backend/`) benchmarks the digitiser against ground truth across 13 chart styles (including the independent `mplfinance` library), 3 densities and JPEG/resize degradation. Results go to `backend/cv/results/`.

**Benchmark (351 images, 13 styles, 3 densities, clean/JPEG/resized):**

| Metric | Result |
|---|---|
| Images processed without error | 351 / 351 |
| Price axis read by OCR | 99.7% |
| Mean close-price error (when candle count exact) | 0.064% (95th percentile 0.19%, max 0.82%) |
| Up/down direction correct | 99.7% |
| Correlation of extracted vs true returns | 0.997 |
| Exact candle count, 40 / 90 / 160 candles, clean image | 100% / 100% / 95% |
| Exact candle count, 40 / 90 / 160 candles, JPEG | 95% / 87% / 41% |
| Mean time per image | 2.1 s |

## LSTM serving

The LSTM is trained with PyTorch (`train_lstm.py`) and exported to `lstm_weights.npz` (`export_lstm_weights.py`). The API runs it with NumPy (`lstm_numpy.py`), which matches PyTorch to within 1e-7 and saves about 300 MB of RAM. PyTorch is only needed to retrain.

## Optimiser

- Expected returns: annualised mean daily returns, shrunk toward the cross-asset average by 252/(n+252) for n observations, so short histories are trusted less.
- Covariance: Ledoit-Wolf shrinkage.
- Portfolios (long-only, optional per-asset cap): minimum variance, maximum Sharpe, and the risk-profile portfolio (highest return at the profile's target volatility: Conservative 5%, Moderate 10%, Aggressive 18%).
- Assets read by Chart Vision can be added to the optimisation ("Add to optimiser").
- **Your real portfolio** (dashboard → Rebalancer, and the Optimiser page's "Real tickers" mode): 2 years of real daily prices for your holdings, plus a cash asset earning the risk-free rate. RAPOS reports the portfolio's current volatility, expected return, Sharpe ratio, maximum drawdown, 1-day 95 % historical VaR and each holding's share of risk, then the optimal mix for your profile and the exact trades (dollars and shares) to reach it. "Apply" updates the quantities RAPOS tracks; it never places real orders.
- Speed: analytic gradients for the SLSQP solver and a warm-started frontier (about 8× faster than finite differences); market data is fetched in parallel and cached (history 6 h, quotes 30 s), and popular symbols are pre-loaded at startup.
- The built-in demo universe (`backend/data/asset_universe.csv`, made by `generate_asset_universe.py`) is **simulated** data calibrated to typical long-run asset-class returns, volatilities and correlations.

## API (v3)

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/digitize-chart` | multipart `image` (+ optional `price_high`, `price_low`, `interval`) → candles, overlay image, risk analysis |
| GET | `/api/optimize?profile=&risk_free=&max_weight=` | optimise the built-in universe |
| POST | `/api/optimize` | same, with `custom_assets: [{name, closes}]` |
| POST | `/api/analyze-series` | risk metrics + LSTM signal for a list of closes |
| POST | `/api/portfolio/analyze` | `{holdings: [{ticker, quantity} or {ticker, value}], profile}` → current risk, optimal mix, trades (real market data) |
| GET | `/api/quotes?tickers=AAPL,MSFT,BTC` | batched live prices (Finnhub, Yahoo fallback) |
| GET | `/api/live-price?ticker=` | one live price |
| GET | `/api/rebalance` | older demo rebalancer on the simulated universe (kept for compatibility) |
| GET | `/api/health` | service, OCR, model and market-data status |

## Deploy (Render backend + Vercel frontend)

1. **Backend on Render**: Render dashboard → **New → Blueprint** → choose this repo. `render.yaml` builds `backend/Dockerfile` (FastAPI, OpenCV, Tesseract for axis OCR; the LSTM runs in NumPy, so the service uses ~230 MB and fits the free plan). Enter `FINNHUB_API_KEY` when asked (optional; live prices and news). When it's live, open `https://<your-service>.onrender.com/api/health`.
2. **Point the site at it**: put that URL in `PROD_API` in `js/rapos-config.js` and push. It is the only place the backend URL is set.
3. **Frontend on Vercel**: Vercel → **Add New → Project** → import this repo → Framework "Other", no build command, root `./`. Every push to `main` redeploys; other branches get preview URLs.
4. **Supabase login**: in the Supabase project, Authentication → URL Configuration, add the Vercel URL as Site URL / redirect URL so sign-up confirmation links work.

Render's free plan sleeps after ~15 minutes idle; the first request then takes ~30-60 s, and the pages say so.

## Run locally

```bash
cd backend
pip install -r requirements-dev.txt    # server deps + PyTorch (training), pytest; plus the Tesseract binary for axis OCR (apt install tesseract-ocr / brew install tesseract)
export FINNHUB_API_KEY=your_key         # optional, for live prices and news
uvicorn main:app --reload               # http://localhost:8000/docs
python -m pytest -q                     # 20 tests
```

Serve the frontend from the repo root (e.g. `python -m http.server 5500`) and open `chart-vision.html` or `optimizer.html`. Pages served from localhost use the local backend automatically; add `?api=https://...` to point elsewhere.

**Deploying with OCR**: Render's native Python environment has no Tesseract. Deploy `backend/` with the included `Dockerfile` (Render → Docker runtime) to enable axis OCR; otherwise Chart Vision asks for the high/low prices.

**API key**: set `FINNHUB_API_KEY` as an environment variable. Never commit keys.

## Known limitations

- Very dense charts (more than ~120 candles across a normal screenshot) combined with JPEG compression can miss or merge a few candles; the app warns and suggests zooming in.
- Line charts, Heikin-Ashi and bar (OHLC tick) charts are not supported.
- Market data: price history comes from Yahoo Finance's public chart API (Stooq as fallback); these are free, unofficial sources and can rate-limit. Quotes come from Finnhub (free key). US tickers and major crypto (BTC, ETH, ...) are supported.
- The LSTM next-close forecast was tested on 640 points of random-walk prices: its error was about twice that of a "no change" guess (2.4 % vs 1.2 %) and it called direction 48 % of the time. The app therefore no longer shows LSTM predictions; Chart Vision reports the volatility-based range of the next candle instead (68 % / 95 %), and the dashboard shows the risk of your actual portfolio. The model and `/api/predict-risk` remain for reference.


## Project layout

| Path | What it is |
|---|---|
| `index.html`, `login.html`, `signup.html`, `forgot-password.html`, `reset-password.html` | Landing page and Supabase auth |
| `dashboard.html` | Holdings (live prices), portfolio risk check, Rebalancer, PDF plan export |
| `chart-vision.html` | Chart screenshot → OHLC data → risk analysis |
| `optimizer.html` | Mean-variance optimiser (your portfolio, any tickers, or the demo universe) |
| `js/` | `rapos-config.js` (backend URL), `api.js`, `ui.js`, `live-ticker.js`, `symbols.js` |
| `backend/` | FastAPI service: `main.py`, `market_data.py`, `portfolio.py`, `optimizer.py`, `cv/` (Chart Vision), `forecast.py`, LSTM files, `tests/` |
| `docs/coursework/` | Lab, testing and poster documents submitted for the course |
| `Project_Report/` | Final project report |
