"""Fetch 1m history (Upstox v3 public historical-candle, no token) for every current NSE F&O stock underlying.

Writes ~/india-vol-data/1m/<SYMBOL>.parquet (ts epoch s, o, h, l, c, v); month responses are cached as JSON under
~/india-vol-data/raw so a rerun resumes. API errors are never cached. Raw data stays local (not redistributed).
Run: <venv>/python src/fetch_1m.py [--limit N]
"""
from __future__ import annotations

import datetime as dt
import gzip
import io
import json
import sys
import time
import urllib.parse as up
from pathlib import Path

import pandas as pd
import requests

ROOT = Path.home() / "india-vol-data"
RAW, OUT = ROOT / "raw", ROOT / "1m"
START, END = dt.date(2022, 1, 1), dt.date.today()
NSE_MASTER = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
COLS = ["ts", "o", "h", "l", "c", "v"]


def fno_underlyings() -> dict[str, str]:
    """{symbol: NSE_EQ instrument key} for every stock with a live NSE future."""
    raw = json.loads(gzip.decompress(requests.get(NSE_MASTER, timeout=60).content))
    now_ms = time.time() * 1000
    out = {}
    for x in raw:
        if x.get("segment") == "NSE_FO" and x.get("instrument_type") == "FUT" and x.get("expiry", 0) > now_ms:
            k, s = x.get("underlying_key", ""), x.get("underlying_symbol")
            if k.startswith("NSE_EQ|") and s:
                out[s] = k
    return dict(sorted(out.items()))


def months(a: dt.date, b: dt.date):
    cur = a.replace(day=1)
    while cur <= b:
        nxt = (cur.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        yield cur, min(nxt - dt.timedelta(days=1), b)
        cur = nxt


def fetch_month(key: str, a: dt.date, b: dt.date) -> list:
    url = f"https://api.upstox.com/v3/historical-candle/{up.quote(key, safe='')}/minutes/1/{b}/{a}"
    for i in range(6):
        r = requests.get(url, timeout=60)
        if r.status_code == 429:
            time.sleep(2 * (i + 1))
            continue
        body = r.json()
        if body.get("status") != "success":
            raise RuntimeError(f"{key} {a}: {body.get('status')} {str(body.get('errors'))[:100]}")
        return body.get("data", {}).get("candles", [])
    raise RuntimeError(f"{key} {a}: rate-limited")


def fetch_symbol(sym: str, key: str) -> int:
    rows = []
    this_month = dt.date.today().replace(day=1)
    for a, b in months(START, END):
        cache = RAW / f"{sym}_{a:%Y-%m}.json"
        if cache.exists() and a < this_month:          # the current month is always re-fetched
            c = json.loads(cache.read_text())
        else:
            c = fetch_month(key, a, b)
            cache.write_text(json.dumps(c))
            time.sleep(0.25)
        rows += c
    if not rows:
        return 0
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v", "oi"])
    t = pd.to_datetime(df["t"], utc=True)
    df["ts"] = (t - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    df = df[COLS].drop_duplicates("ts").sort_values("ts")
    df.to_parquet(OUT / f"{sym}.parquet")
    return len(df)


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    syms = fno_underlyings()
    if "--limit" in sys.argv:
        syms = dict(list(syms.items())[: int(sys.argv[sys.argv.index("--limit") + 1])])
    print(f"{len(syms)} F&O underlyings", flush=True)
    for i, (s, k) in enumerate(syms.items(), 1):
        try:
            n = fetch_symbol(s, k)
            print(f"[{i}/{len(syms)}] {s}: {n} bars", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(syms)}] {s}: FAILED {e}", flush=True)
