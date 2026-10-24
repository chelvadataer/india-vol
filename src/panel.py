"""Daily realized-variance panel + the causal baselines (RV20, HAR) for IndiaVol (docs/2026-10-24-design-and-prereg.md).

rv(day) = Σ squared 5-min log returns inside 09:15–15:30 IST + the squared overnight return (previous close → open).
Panel: ~/india-vol-data/panel.parquet with columns series, date, rv, close.
Run: <venv>/python src/panel.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "india-vol-data"
INDEX_DIR = Path.home() / "mtf-data"
INDICES = ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX")
MIN_DAYS = 500
HAR_MIN_ORIGINS = 250


def daily_rv(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    mod = t.dt.hour * 60 + t.dt.minute
    keep = (mod >= 555) & (mod < 930)
    d = df.loc[keep, ["o", "c"]].copy()
    d.index = t[keep]
    c5 = d["c"].resample("5min").last().dropna()
    c5 = c5[c5 > 0]
    r = np.log(c5).groupby(c5.index.date).diff().dropna()
    intraday = (r ** 2).groupby(r.index.date).sum()
    opens = d["o"].groupby(d.index.date).first()
    closes = d["c"].groupby(d.index.date).last()
    on = np.log(opens / closes.shift(1)).dropna()
    rv = intraday.add(on ** 2, fill_value=0.0)
    out = pd.DataFrame({"rv": rv, "close": closes}).dropna()
    return out[out.rv > 0]


def build_panel() -> pd.DataFrame:
    files = [(n, INDEX_DIR / f"{n}.parquet") for n in INDICES]
    files += [(p.stem, p) for p in sorted((ROOT / "1m").glob("*.parquet"))]
    parts = []
    for name, path in files:
        if not path.exists():
            continue
        d = daily_rv(pd.read_parquet(path))
        if len(d) < MIN_DAYS:
            continue
        d = d.reset_index(names="date")
        d["series"] = name
        parts.append(d[["series", "date", "rv", "close"]])
    panel = pd.concat(parts, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.date
    return panel


# ---------------------------------------------------------------- causal forecasts (σ annualised)

def realized_sigma(rv: np.ndarray, i: int, h: int) -> float:
    seg = rv[i + 1:i + 1 + h]
    return math.sqrt(252 * seg.mean()) if len(seg) == h else float("nan")


def rv20_sigma(close: np.ndarray, i: int, window: int = 20) -> float:
    if i < window:
        return float("nan")
    r = np.diff(np.log(close[i - window:i + 1]))
    return float(r.std(ddof=1) * math.sqrt(252))


def har_features(rv: np.ndarray) -> np.ndarray:
    s = pd.Series(rv)
    return np.column_stack([np.ones(len(rv)), np.log(rv), np.log(s.rolling(5).mean().values),
                            np.log(s.rolling(22).mean().values)])


def har_sigma(rv: np.ndarray, X: np.ndarray, i: int, h: int) -> float:
    """Direct h-step log-HAR at origin i, OLS on origins j with j + h <= i (targets realized by i's close)."""
    js = np.arange(21, i - h + 1)
    if len(js) < HAR_MIN_ORIGINS:
        return float("nan")
    csum = np.r_[0.0, np.cumsum(rv)]
    y = np.log((csum[js + 1 + h] - csum[js + 1]) / h)
    b, *_ = np.linalg.lstsq(X[js], y, rcond=None)
    res = y - X[js] @ b
    return math.sqrt(252 * math.exp(X[i] @ b + 0.5 * res.var(ddof=X.shape[1])))


def qlike(f, r):
    q = (np.asarray(f) ** 2) / (np.asarray(r) ** 2)
    return q - np.log(q) - 1


if __name__ == "__main__":
    p = build_panel()
    p.to_parquet(ROOT / "panel.parquet")
    print(f"panel: {p.series.nunique()} series, {len(p)} series-days, {p.date.min()} → {p.date.max()}")
