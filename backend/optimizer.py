"""
RAPOS mean-variance portfolio optimiser (Markowitz).

Given price histories for N assets:
  * expected returns  mu  = annualised mean daily return, shrunk toward the
                            cross-asset mean by 252/(n+252) for n observations
  * covariance        S   = Ledoit-Wolf shrunk sample covariance (annualised);
                            shrinkage keeps S well-conditioned with short or
                            noisy histories (e.g. digitised chart data)
Portfolios (long-only, weights sum to 1, optional per-asset cap):
  * minimum variance       min  w'Sw
  * maximum Sharpe         max  (w'mu - rf) / sqrt(w'Sw)
  * risk-profile target    max  w'mu   s.t.  sqrt(w'Sw) <= target volatility
  * efficient frontier     min variance for a grid of target returns
A risk-free CASH asset can be included (zero variance, return = rf).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "asset_universe.csv")

# Target annual volatility for each investor risk profile
PROFILE_TARGET_VOL = {"Conservative": 0.05, "Moderate": 0.10, "Aggressive": 0.18}


def load_universe(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    return df.astype(float)


def estimate(prices: pd.DataFrame, risk_free: float = 0.04, include_cash: bool = True):
    """Annualised expected returns and shrunk covariance from price columns."""
    rets = prices.pct_change().dropna(how="all").fillna(0.0)
    if len(rets) < 10:
        raise ValueError("Need at least 10 price points per asset.")
    mu = rets.mean().values * TRADING_DAYS
    # Mean returns are very noisy over short windows (a 60-day rally looks
    # like a 50%/year asset). Shrink them toward the cross-asset average,
    # trusting the sample more as history grows (James-Stein-style; w = n/(n+252)).
    n_obs = len(rets)
    w_mu = n_obs / (n_obs + TRADING_DAYS)
    mu = w_mu * mu + (1 - w_mu) * mu.mean()
    lw = LedoitWolf().fit(rets.values)
    cov = lw.covariance_ * TRADING_DAYS
    names = list(prices.columns)
    if include_cash:
        names.append("CASH")
        mu = np.append(mu, risk_free)
        n = len(names)
        c = np.zeros((n, n))
        c[:-1, :-1] = cov
        cov = c
    return names, mu, cov, float(lw.shrinkage_), float(1 - w_mu)


# ---------------------------------------------------------------------------
def _stats(w, mu, cov, rf):
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    sharpe = (ret - rf) / vol if vol > 1e-9 else 0.0
    return ret, vol, sharpe


def _solve(objective, n, cons=(), max_weight=1.0, x0=None, multistart=False):
    bounds = [(0.0, max_weight)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}, *cons]
    x0 = np.full(n, 1.0 / n) if x0 is None else x0
    best = None
    # a few starting points guard against local minima on flat objectives
    starts = [x0, *(np.eye(n)[: min(n, 4)] * 0.5 + 0.5 / n)] if multistart else [x0]
    for start in starts:
        r = minimize(objective, start, method="SLSQP", bounds=bounds, constraints=cons,
                     options={"maxiter": 500, "ftol": 1e-12})
        if r.success and (best is None or r.fun < best.fun):
            best = r
    if best is None:
        raise RuntimeError("Optimisation did not converge")
    w = np.clip(best.x, 0, None)
    return w / w.sum()


def min_variance(mu, cov, max_weight=1.0):
    return _solve(lambda w: w @ cov @ w, len(mu), max_weight=max_weight)


def max_sharpe(mu, cov, rf, names, max_weight=1.0):
    # Tangency portfolio over risky assets only (cash has no risk)
    risky = [i for i, n in enumerate(names) if n != "CASH"]
    m, c = mu[risky], cov[np.ix_(risky, risky)]

    def neg_sharpe(w):
        v = np.sqrt(max(w @ c @ w, 1e-12))
        return -(w @ m - rf) / v

    wr = _solve(neg_sharpe, len(risky), max_weight=max_weight, multistart=True)
    w = np.zeros(len(mu))
    w[risky] = wr
    return w


def target_volatility(mu, cov, target, max_weight=1.0):
    n = len(mu)
    cons = [{"type": "ineq", "fun": lambda w: target ** 2 - w @ cov @ w}]
    return _solve(lambda w: -(w @ mu), n, cons=cons, max_weight=max_weight, multistart=True)


def efficient_frontier(mu, cov, rf, points: int = 30, max_weight=1.0):
    w_min = min_variance(mu, cov, max_weight)
    r_lo = float(w_min @ mu)
    # highest achievable return under the weight cap
    r_hi = float(_solve(lambda w: -(w @ mu), len(mu), max_weight=max_weight) @ mu)
    out = []
    for target in np.linspace(r_lo, r_hi, points):
        cons = [{"type": "ineq", "fun": lambda w, t=target: w @ mu - t}]
        try:
            w = _solve(lambda w: w @ cov @ w, len(mu), cons=cons, max_weight=max_weight, x0=w_min)
        except RuntimeError:
            continue
        ret, vol, sh = _stats(w, mu, cov, rf)
        out.append({"return": ret, "volatility": vol, "sharpe": sh})
    return out


def risk_contributions(w, cov):
    """Share of total portfolio variance contributed by each asset."""
    total = w @ cov @ w
    if total <= 1e-12:
        return np.zeros_like(w)
    return w * (cov @ w) / total


# ---------------------------------------------------------------------------
def optimise(prices: pd.DataFrame | None = None, profile: str = "Moderate",
             risk_free: float = 0.04, max_weight: float = 0.6, include_cash: bool = True,
             frontier_points: int = 30) -> dict:
    prices = load_universe() if prices is None else prices
    names, mu, cov, shrink, mu_shrink = estimate(prices, risk_free, include_cash)
    # the cap must allow a feasible portfolio
    max_weight = max(max_weight, 1.0 / len(names) + 1e-6)

    def pack(w, label):
        ret, vol, sh = _stats(w, mu, cov, risk_free)
        rc = risk_contributions(w, cov)
        return {
            "label": label,
            "weights": {n: round(float(x), 4) for n, x in zip(names, w)},
            "risk_contribution": {n: round(float(x), 4) for n, x in zip(names, rc)},
            "expected_return": round(ret, 4),
            "volatility": round(vol, 4),
            "sharpe": round(sh, 3),
        }

    w_mv = min_variance(mu, cov, max_weight)
    w_ms = max_sharpe(mu, cov, risk_free, names, max_weight)
    vol_floor = _stats(w_mv, mu, cov, risk_free)[1]
    target = max(PROFILE_TARGET_VOL.get(profile, 0.10), vol_floor + 1e-4)
    w_pf = target_volatility(mu, cov, target, max_weight)

    assets = []
    for i, n in enumerate(names):
        assets.append({"asset": n, "expected_return": round(float(mu[i]), 4),
                       "volatility": round(float(np.sqrt(cov[i, i])), 4)})
    corr = None
    sd = np.sqrt(np.diag(cov))
    with np.errstate(invalid="ignore", divide="ignore"):
        cm = cov / np.outer(sd, sd)
    cm = np.nan_to_num(cm)
    np.fill_diagonal(cm, 1.0)
    corr = {"assets": names, "matrix": np.round(cm, 3).tolist()}

    return {
        "profile": profile if profile in PROFILE_TARGET_VOL else "Moderate",
        "target_volatility": round(target, 4),
        "risk_free": risk_free,
        "max_weight": round(max_weight, 4),
        "covariance_shrinkage": round(shrink, 4),
        "return_shrinkage": round(mu_shrink, 4),
        "observations": int(len(prices)),
        "assets": assets,
        "correlation": corr,
        "recommended": pack(w_pf, f"{profile} (target vol {target:.0%})"),
        "min_variance": pack(w_mv, "Minimum variance"),
        "max_sharpe": pack(w_ms, "Maximum Sharpe"),
        "frontier": efficient_frontier(mu, cov, risk_free, frontier_points, max_weight),
    }


def rebalance_trades(current: dict[str, float], target_weights: dict[str, float]) -> list[dict]:
    """Dollar trades that move current holdings (value per asset) to target weights."""
    total = sum(current.values())
    trades = []
    for a in sorted(set(current) | set(target_weights)):
        cur = current.get(a, 0.0)
        tgt = target_weights.get(a, 0.0) * total
        if abs(tgt - cur) >= 0.005 * max(total, 1):
            trades.append({"asset": a, "action": "BUY" if tgt > cur else "SELL",
                           "amount": round(abs(tgt - cur), 2),
                           "current_value": round(cur, 2), "target_value": round(tgt, 2)})
    return trades
