"""
Risk-aware analysis and optimisation of a user's real portfolio.

Input: holdings (ticker + quantity), live prices, and daily price history.
Output:
  * the current portfolio's value, weights and risk: annualised volatility,
    expected return, Sharpe ratio, maximum drawdown, 1-day 95 % Value at Risk,
    and each holding's share of total risk
  * the mean-variance optimal portfolio for the chosen risk profile (plus the
    minimum-variance and maximum-Sharpe portfolios and the efficient frontier),
    over the SAME assets plus cash
  * the trades (in dollars and shares) that move the current portfolio to the
    recommended one

Pure functions: market data is passed in, so this is testable offline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import optimizer

TRADING_DAYS = optimizer.TRADING_DAYS


def _drawdown(values: pd.Series) -> float:
    peak = values.cummax()
    return float(((values - peak) / peak).min())


def current_risk(prices: pd.DataFrame, weights: dict[str, float], mu, cov, names,
                 risk_free: float, total_value: float) -> dict:
    """Risk of the portfolio as it is now (historical + model-based)."""
    w = np.array([weights.get(n, 0.0) for n in names])
    ret, vol, sharpe = optimizer._stats(w, mu, cov, risk_free)
    rc = optimizer.risk_contributions(w, cov)

    risky = [n for n in names if n != "CASH"]
    daily = prices[risky].pct_change().dropna()
    wr = np.array([weights.get(n, 0.0) for n in risky])
    port = daily.values @ wr + weights.get("CASH", 0.0) * risk_free / TRADING_DAYS
    port = pd.Series(port, index=daily.index)
    growth = (1 + port).cumprod()
    var95 = float(-np.percentile(port, 5)) if len(port) else 0.0
    return {
        "expected_return": round(ret, 4),
        "volatility": round(vol, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(_drawdown(growth), 4) if len(growth) else 0.0,
        "var_95_1d_pct": round(var95, 4),
        "var_95_1d_value": round(var95 * total_value, 2),
        "risk_contribution": {n: round(float(x), 4) for n, x in zip(names, rc)},
        "weights": {n: round(float(weights.get(n, 0.0)), 4) for n in names},
        "period_return": round(float(growth.iloc[-1] - 1), 4) if len(growth) else 0.0,
    }


def analyse(holdings: list[dict], prices: pd.DataFrame, live: dict[str, dict],
            profile: str = "Moderate", risk_free: float = 0.04, max_weight: float = 0.6) -> dict:
    """
    holdings: [{"ticker": "AAPL", "quantity": 10}, {"ticker": "MSFT", "value": 1000},
               {"ticker": "CASH", "quantity": 500}, ...]  (same ticker summed; "value"
               is a dollar amount, converted to shares at the current price)
    prices:   aligned daily closes, one column per ticker
    live:     {ticker: {"price": ...}} current prices (falls back to last close)
    """
    def price_of(t):
        q = live.get(t) or {}
        return float(q.get("price") or prices[t].iloc[-1])

    qty: dict[str, float] = {}
    cash = 0.0
    for h in holdings:
        t = str(h["ticker"]).upper()
        if t == "CASH":
            cash += float(h.get("quantity") or h.get("value") or 0)
            continue
        if t not in prices.columns:
            continue
        q = float(h["quantity"]) if h.get("quantity") else float(h.get("value") or 0) / price_of(t)
        qty[t] = qty.get(t, 0.0) + q
    tickers = [t for t in qty if qty[t] > 0]
    if not tickers:
        raise ValueError("None of the holdings have price history.")
    prices = prices[tickers]

    rows, values = [], {}
    for t in tickers:
        q = live.get(t) or {}
        price = price_of(t)
        values[t] = qty[t] * price
        rows.append({"ticker": t, "quantity": qty[t], "price": round(price, 4),
                     "value": round(values[t], 2),
                     "day_change_pct": round(float(q.get("percent_change", 0.0)), 2),
                     "price_source": q.get("source", "last close"),
                     "currency": q.get("currency", "INR" if t.endswith((".NS", ".BO")) else "USD"),
                     "native_price": round(float(q.get("native_price") or price), 4)})
    if cash > 0:
        values["CASH"] = cash
        rows.append({"ticker": "CASH", "quantity": round(cash, 2), "price": 1.0, "value": round(cash, 2),
                     "day_change_pct": 0.0, "price_source": "cash"})
    total = sum(values.values())
    if total <= 0:
        raise ValueError("Portfolio value is zero.")
    weights = {t: v / total for t, v in values.items()}
    for r in rows:
        r["weight"] = round(weights[r["ticker"]], 4)

    # Optimise over the user's own assets plus cash (so a cautious profile can
    # hold cash instead of being forced into risky stocks)
    res = optimizer.optimise(prices, profile=profile, risk_free=risk_free,
                             max_weight=max_weight, include_cash=True, frontier_points=20)
    names, mu, cov, _, _ = optimizer.estimate(prices, risk_free, include_cash=True)
    now = current_risk(prices, weights, mu, cov, names, risk_free, total)

    target = res["recommended"]["weights"]
    px = {r["ticker"]: r["price"] for r in rows}
    trades = []
    for tr in optimizer.rebalance_trades(values, target):
        p = px.get(tr["asset"], 1.0 if tr["asset"] == "CASH" else None)
        tr["shares"] = round(tr["amount"] / p, 4) if p else None
        if tr["action"] == "SELL" and target.get(tr["asset"], 0) < 1e-4 and tr["asset"] in qty:
            tr["shares"] = round(qty[tr["asset"]], 6)          # sell the whole position
            tr["sell_all"] = True
        trades.append(tr)
    trades.sort(key=lambda t: -t["amount"])

    tv = res["target_volatility"]
    gap = now["volatility"] - tv
    if abs(gap) <= 0.02:
        verdict = f"Your portfolio's risk ({now['volatility']:.0%} volatility) already fits the {profile} profile."
    elif gap > 0:
        verdict = (f"Your portfolio is riskier than the {profile} profile allows: "
                   f"{now['volatility']:.0%} volatility vs a {tv:.0%} target.")
    else:
        verdict = (f"Your portfolio takes less risk than the {profile} profile allows "
                   f"({now['volatility']:.0%} vs {tv:.0%} volatility), so it may be giving up return.")

    return {
        "profile": res["profile"],
        "total_value": round(total, 2),
        "holdings": rows,
        "current": now,
        "recommended": res["recommended"],
        "min_variance": res["min_variance"],
        "max_sharpe": res["max_sharpe"],
        "frontier": res["frontier"],
        "observations": res["observations"],
        "return_shrinkage": res["return_shrinkage"],
        "risk_free": res["risk_free"],
        "max_weight": res["max_weight"],
        "assets": res["assets"],
        "correlation": res["correlation"],
        "target_volatility": tv,
        "trades": trades,
        "verdict": verdict,
        "history": {"start": str(prices.index[0].date()), "end": str(prices.index[-1].date()),
                    "days": int(len(prices))},
        "method": "Mean-variance (Markowitz) on daily returns; Ledoit-Wolf covariance; "
                  "expected returns shrunk toward the cross-asset mean",
    }
