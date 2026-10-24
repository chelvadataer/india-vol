"""Write the Chronos inputs: one log-RV series per instrument + the validation / holdout origin lists (prereg addendum).
Run: <venv>/python src/export.py
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "india-vol-data"
INDICES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"}
VAL = (dt.date(2024, 7, 1), dt.date(2024, 12, 31))
HOLD_FROM = dt.date(2025, 1, 1)
H_MAX = 10


def origins(panel: pd.DataFrame, lo: dt.date, hi: dt.date | None) -> pd.DataFrame:
    rows = []
    for s, g in panel.groupby("series", sort=True):
        dates = list(g.date)
        idx = [i for i, d in enumerate(dates) if d >= lo and (hi is None or d <= hi) and i + H_MAX < len(dates)]
        if s not in INDICES:
            idx = idx[::5]                          # stocks: every 5th trading day (prereg addendum)
        rows += [{"series": s, "origin": dates[i].isoformat(), "i": i} for i in idx]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    panel = pd.read_parquet(ROOT / "panel.parquet").sort_values(["series", "date"])
    sdir = ROOT / "series"
    sdir.mkdir(exist_ok=True)
    for s, g in panel.groupby("series"):
        pd.DataFrame({"date": [d.isoformat() for d in g.date], "log_rv": np.log(g.rv.values)}).to_csv(sdir / f"{s}.csv", index=False)
    v, h = origins(panel, *VAL), origins(panel, HOLD_FROM, None)
    v.to_csv(ROOT / "origins_val.csv", index=False)
    h.to_csv(ROOT / "origins_holdout.csv", index=False)
    print(f"{panel.series.nunique()} series; validation origins {len(v)}, holdout origins {len(h)}")
