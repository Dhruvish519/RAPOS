"""
Benchmark for the RAPOS chart digitiser.

Renders candlestick charts from known OHLC data with two independent renderers
(our own matplotlib renderer and the mplfinance library's built-in styles),
applies screenshot-like degradations (JPEG compression, resizing), digitises
each image and compares the result against the ground truth.

Run:  python -m cv.evaluate          (from the backend/ folder)
Outputs: cv/results/benchmark.csv, cv/results/summary.json, sample overlays.
"""
from __future__ import annotations

import io
import json
import os
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplfinance as mpf  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cv.chart_digitizer import digitize  # noqa: E402
from cv.synthetic_charts import THEMES, random_ohlc, render_chart  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results")

MPF_STYLES = ["yahoo", "charles", "binance", "binancedark", "nightclouds", "tradingview", "classic"]


def render_mpf(df: pd.DataFrame, style: str, volume: bool, size=(12, 7)) -> np.ndarray:
    d = df.copy()
    d.index = pd.bdate_range("2025-01-01", periods=len(d))
    if volume:
        d["Volume"] = np.random.default_rng(len(d)).integers(1e5, 1e6, len(d))
    fig, _ = mpf.plot(d, type="candle", style=style, volume=volume, returnfig=True,
                      figsize=size, tight_layout=True, warn_too_much_data=10_000)
    buf = io.BytesIO()
    # bbox_inches="tight" keeps the right-hand price labels inside the image,
    # as they are in a real app screenshot
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)


def degrade(img: np.ndarray, kind: str) -> np.ndarray:
    if kind == "jpeg":
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if kind == "resize":
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * 0.75), int(h * 0.75)), interpolation=cv2.INTER_AREA)
    return img


def score(truth: pd.DataFrame, res) -> dict:
    o = res.ohlc
    row = {"n_true": len(truth), "n_found": len(o), "count_ok": len(o) == len(truth),
           "calib": res.calibration.get("method")}
    row["colour_mode"] = res.mode
    if len(o) == len(truth):
        span = truth.High.max() - truth.Low.min()
        # Absolute price error only means something when real prices were read
        for k in ([] if row["calib"] == "relative" else ["Open", "High", "Low", "Close"]):
            err = np.abs(o[k].values - truth[k].values)
            row[f"mape_{k.lower()}"] = float(np.mean(err / truth[k].values) * 100)
            row[f"err_{k.lower()}_pct_range"] = float(np.mean(err) / span * 100)
        true_dir = (truth.Close >= truth.Open).values
        row["direction_acc"] = float(np.mean(o.Bullish.values == true_dir) * 100)
        tr = np.diff(np.log(truth.Close.values))
        pr = np.diff(np.log(np.clip(o.Close.values, 1e-6, None)))
        row["return_corr"] = float(np.corrcoef(tr, pr)[0, 1]) if len(tr) > 2 else np.nan
        tv, pv = tr.std() * np.sqrt(252) * 100, pr.std() * np.sqrt(252) * 100
        row["vol_err_pts"] = float(abs(tv - pv))
    return row


def run(n_seeds: int = 3, save_samples: bool = True) -> pd.DataFrame:
    os.makedirs(OUT, exist_ok=True)
    rows = []
    sample_saved = set()
    for seed in range(n_seeds):
        for n in (40, 90, 160):
            rng = np.random.default_rng(seed * 100 + n)
            df = random_ohlc(n=n, start=float(rng.uniform(20, 900)),
                             vol=float(rng.uniform(0.01, 0.035)), seed=seed * 100 + n)
            jobs = [(f"rapos-{t}", lambda t=t: render_chart(df, theme=t)) for t in THEMES]
            jobs += [(f"mpf-{s}", lambda s=s: render_mpf(df, s, volume=False)) for s in MPF_STYLES]
            jobs += [(f"mpf-{s}+vol", lambda s=s: render_mpf(df, s, volume=True))
                     for s in ("yahoo", "binancedark")]
            for name, fn in jobs:
                base = fn()
                for deg in ("none", "jpeg", "resize"):
                    img = degrade(base, deg)
                    t0 = time.perf_counter()
                    try:
                        res = digitize(img)
                        r = score(df, res)
                        r["ok"] = True
                    except Exception as e:  # count failures honestly
                        res, r = None, {"ok": False, "error": str(e), "n_true": n, "count_ok": False}
                    r.update({"renderer": name, "degradation": deg, "n": n, "seed": seed,
                              "ms": (time.perf_counter() - t0) * 1000})
                    rows.append(r)
                    key = (name, n)
                    if save_samples and res is not None and deg == "none" and seed == 0 \
                            and key not in sample_saved and n == 90:
                        sample_saved.add(key)
                        cv2.imwrite(os.path.join(OUT, f"sample_{name}.png"), res.overlay)
    df_res = pd.DataFrame(rows)
    df_res.to_csv(os.path.join(OUT, "benchmark.csv"), index=False)
    return df_res


def summarise(df: pd.DataFrame) -> dict:
    ok = df[df.count_ok == True]  # noqa: E712
    summary = {
        "images": int(len(df)),
        "candle_count_exact_pct": round(float(df.count_ok.mean() * 100), 1),
        "ocr_calibrated_pct": round(float((df.calib == "ocr").mean() * 100), 1),
        "mean_close_mape_pct": round(float(ok.mape_close.mean()), 3),
        "median_close_mape_pct": round(float(ok.mape_close.median()), 3),
        "mean_err_pct_of_range": round(float(ok[[c for c in ok if c.startswith("err_")]].mean().mean()), 3),
        "failed_images": int((~df.ok).sum()),
        "direction_acc_pct": round(float(ok.direction_acc.mean()), 2),
        "mean_return_corr": round(float(ok.return_corr.mean()), 4),
        "mean_ms_per_image": round(float(df.ms.mean()), 1),
    }
    by = lambda col: (df.groupby(col).agg(  # noqa: E731
        count_exact=("count_ok", "mean"),
        ocr=("calib", lambda s: (s == "ocr").mean()),
        close_mape=("mape_close", "mean"),
        direction=("direction_acc", "mean"),
    ).round(4).reset_index().to_dict(orient="records"))
    summary["by_renderer"] = by("renderer")
    summary["by_degradation"] = by("degradation")
    summary["by_n_candles"] = by("n")
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    res = run()
    s = summarise(res)
    print(json.dumps({k: v for k, v in s.items() if not k.startswith("by_")}, indent=2))
    print(pd.DataFrame(s["by_renderer"]).to_string(index=False))
    print(pd.DataFrame(s["by_degradation"]).to_string(index=False))
    print(pd.DataFrame(s["by_n_candles"]).to_string(index=False))
