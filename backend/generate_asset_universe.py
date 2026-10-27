"""
Generate a simulated multi-asset price history for the portfolio optimiser.

The numbers are SIMULATED (multivariate geometric Brownian motion), but the
annual return, volatility and correlation assumptions are set to typical
long-run ranges for each asset class so the optimiser behaves realistically.
Output: data/asset_universe.csv  (Date + one close-price column per asset)
"""
import os

import numpy as np
import pandas as pd

ASSETS = {
    #           annual return, annual volatility, start price
    "EQUITY": (0.08, 0.16, 100.0),   # broad stock index
    "TECH":   (0.11, 0.24, 100.0),   # technology stocks
    "BONDS":  (0.035, 0.06, 100.0),  # government bonds
    "GOLD":   (0.05, 0.15, 100.0),
    "CRYPTO": (0.20, 0.65, 100.0),
}
CORR = np.array([
    # EQ    TECH  BOND  GOLD  CRYPTO
    [1.00, 0.85, -0.20, 0.05, 0.35],
    [0.85, 1.00, -0.15, 0.00, 0.40],
    [-0.20, -0.15, 1.00, 0.30, -0.05],
    [0.05, 0.00, 0.30, 1.00, 0.10],
    [0.35, 0.40, -0.05, 0.10, 1.00],
])


def generate(days: int = 1260, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(ASSETS)
    mu = np.array([ASSETS[n][0] for n in names])
    vol = np.array([ASSETS[n][1] for n in names])
    cov_d = np.outer(vol, vol) * CORR / 252
    drift = mu / 252 - 0.5 * np.diag(cov_d)          # log-drift so E[return] = mu
    shocks = rng.multivariate_normal(drift, cov_d, size=days)
    prices = np.array([ASSETS[n][2] for n in names]) * np.exp(np.cumsum(shocks, axis=0))
    dates = pd.bdate_range("2021-01-04", periods=days)
    df = pd.DataFrame(prices.round(4), columns=names)
    df.insert(0, "Date", dates.strftime("%Y-%m-%d"))
    return df


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "data", "asset_universe.csv")
    df = generate()
    df.to_csv(out, index=False)
    print(f"Wrote {out}: {df.shape}")
