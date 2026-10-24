"""IndiaVol evaluation (docs/2026-10-24-design-and-prereg.md + addendum).

  val      score B0/B1/B2 + every M1 candidate on validation origins, pick M1 (h=5 QLIKE), compute each Chronos model's
           bias at h=1/5/10, and FREEZE them to ~/india-vol-data/selection.json (trials -> ledger 'vol-forecast')
  holdout  refuses without selection.json; scores B0/B1/B2/M1 on the holdout once and applies the publish rule

Chronos predictions come from run_chronos.sh predict ... -> ~/india-vol-data/pred_<model>_<val|holdout>.csv
Run: <venv>/python src/evaluate.py val|holdout
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel as P  # noqa: E402

ROOT = Path.home() / "india-vol-data"
HS = (1, 5, 10)
INDICES = ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX")
M1_CANDIDATES = ("m1_lora_1000", "m1_lora_3000", "m1_full_1000")
B = 2000


def baselines(origins: pd.DataFrame) -> pd.DataFrame:
    panel = pd.read_parquet(ROOT / "panel.parquet").sort_values(["series", "date"])
    out = []
    for s, g in panel.groupby("series"):
        o = origins[origins.series == s]
        if o.empty:
            continue
        rv, close = g.rv.values, g.close.values
        X = P.har_features(rv)
        for r in o.itertuples():
            row = {"series": s, "origin": r.origin, "B0_rv20": P.rv20_sigma(close, r.i)}
            for h in HS:
                row[f"real{h}"] = P.realized_sigma(rv, r.i, h)
                row[f"B1_har{h}"] = P.har_sigma(rv, X, r.i, h)
            out.append(row)
    return pd.DataFrame(out)


def chronos(name: str, split: str) -> pd.DataFrame | None:
    p = ROOT / f"pred_{name}_{split}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p).rename(columns={f"s{h}": f"{name}{h}" for h in HS})


def frame(split: str, names) -> pd.DataFrame:
    o = pd.read_csv(ROOT / f"origins_{split}.csv")
    f = baselines(o)
    for n in names:
        c = chronos(n, split)
        if c is not None:
            f = f.merge(c, on=["series", "origin"], how="left")
    return f


def ql(f: pd.DataFrame, col: str, h: int) -> float:
    m = f[col].notna() & f[f"real{h}"].notna()
    return float(P.qlike(f.loc[m, col], f.loc[m, f"real{h}"]).mean())


def cols(f, name, h):
    return {"B0_rv20": "B0_rv20", "B1_har": f"B1_har{h}"}.get(name, f"{name}{h}")


def val() -> int:
    f = frame("val", ("b2_zeroshot",) + M1_CANDIDATES)
    rows, bias = [], {}
    for name in ("B0_rv20", "B1_har", "b2_zeroshot") + M1_CANDIDATES:
        if cols(f, name, 5) not in f:
            continue
        r = {"model": name}
        for h in HS:
            c = cols(f, name, h)
            if name.startswith(("b2", "m1")):
                m = f[c].notna() & f[f"real{h}"].notna()
                b = float(np.log(f.loc[m, c] / f.loc[m, f"real{h}"]).mean())
                bias.setdefault(name, {})[h] = b
                f[c + "_bc"] = f[c] * math.exp(-b)
                r[f"qlike{h}"] = ql(f, c + "_bc", h)
                r[f"bias{h}"] = b
            else:
                r[f"qlike{h}"] = ql(f, c, h)
        rows.append(r)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("=== validation (2024-07 → 2024-12), QLIKE after bias correction (lower = better) ===")
    print(t.round(4).to_string(index=False))
    cands = t[t.model.isin(M1_CANDIDATES)]
    if cands.empty:
        print("no M1 candidate predictions yet")
        return 1
    best = cands.sort_values("qlike5").iloc[0]["model"]
    sel = {"m1": best, "bias": bias, "val_table": rows}
    (ROOT / "selection.json").write_text(json.dumps(sel, indent=2))
    print(f"\nselected M1 = {best} (validation h=5 QLIKE) -> selection.json FROZEN")
    sys.path.insert(0, str(Path.home() / "server_setup" / "upstox-trade-bot"))
    import trial_ledger
    for r in rows:
        trial_ledger.record("vol-forecast", "indiavol-2026-10", {"model": r["model"]}, n_obs=len(f), val_qlike5=r["qlike5"])
    return 0


def day_bootstrap_diff(f: pd.DataFrame, a: str, b: str, h: int, rng) -> tuple[float, float, float]:
    m = f[a].notna() & f[b].notna() & f[f"real{h}"].notna()
    g = f[m].copy()
    g["d"] = P.qlike(g[a], g[f"real{h}"]) - P.qlike(g[b], g[f"real{h}"])
    by = {k: v["d"].values for k, v in g.groupby("origin")}
    days = list(by)
    est = float(g["d"].mean())
    vals = []
    for _ in range(B):
        s = np.concatenate([by[d] for d in (rng.choice(days) for _ in days)])
        vals.append(s.mean())
    vals.sort()
    return est, vals[int(0.025 * B)], vals[int(0.975 * B) - 1]


def dm(l0, l1, h):
    d = np.asarray(l1) - np.asarray(l0)
    n = len(d)
    dm_ = d - d.mean()
    v = dm_ @ dm_ / n + sum(2 * (1 - k / h) * (dm_[k:] @ dm_[:-k]) / n for k in range(1, h))
    from statistics import NormalDist
    t = d.mean() / math.sqrt(v / n)
    return t, NormalDist().cdf(t)


def holdout() -> int:
    sp = ROOT / "selection.json"
    if not sp.exists():
        print("refusing: no selection.json — run `evaluate.py val` first (the holdout opens once, after selection)")
        return 1
    sel = json.loads(sp.read_text())
    m1 = sel["m1"]
    f = frame("holdout", ("b2_zeroshot", m1))
    for name in ("b2_zeroshot", m1):
        for h in HS:
            c = f"{name}{h}"
            if c in f:
                f[c + "_bc"] = f[c] * math.exp(-sel["bias"][name][str(h)])
    f.to_csv(ROOT / "holdout_scored.csv", index=False)
    rng = random.Random(20261024)
    print(f"=== HOLDOUT (2025-01 →), {f.series.nunique()} series, {len(f)} origins; QLIKE (lower = better) ===")
    names = {"B0 RV20": lambda h: "B0_rv20", "B1 HAR": lambda h: f"B1_har{h}",
             "B2 Chronos-2 zero-shot (bc)": lambda h: f"b2_zeroshot{h}_bc", f"M1 {m1} (bc)": lambda h: f"{m1}{h}_bc"}
    for h in HS:
        print(f"  h={h}: " + " | ".join(f"{k} {ql(f, c(h), h):.4f}" for k, c in names.items() if c(h) in f))
    est, lo, hi = day_bootstrap_diff(f, f"{m1}5_bc", "B1_har5", 5, rng)
    print(f"\npooled h=5 QLIKE(M1) − QLIKE(HAR) = {est:+.4f}  day-bootstrap CI [{lo:+.4f}, {hi:+.4f}]")
    wins = 0
    for s in INDICES:
        g = f[(f.series == s) & f[f"{m1}5_bc"].notna() & f["B1_har5"].notna()]
        if g.empty:
            continue
        t, p = dm(P.qlike(g["B1_har5"], g["real5"]), P.qlike(g[f"{m1}5_bc"], g["real5"]), 5)
        better = ql(g, f"{m1}5_bc", 5) < ql(g, "B1_har5", 5)
        wins += better
        print(f"  {s:11} M1 {ql(g, f'{m1}5_bc', 5):.4f} vs HAR {ql(g, 'B1_har5', 5):.4f}  DM t {t:+.2f} p {p:.3f}  {'M1 better' if better else 'HAR better'}")
    passed = hi < 0 and wins >= 4
    print(f"\nPUBLISH RULE: pooled CI < 0: {hi < 0}; M1 beats HAR on >= 4/5 indices: {wins}/5  =>  "
          f"{'PUBLISH M1 as IndiaVol' if passed else 'publish the honest comparison (M1 did not beat HAR)'}")
    (ROOT / "holdout_result.json").write_text(json.dumps({"m1": m1, "pooled_h5_diff": [est, lo, hi], "index_wins": wins,
                                                          "publish_m1": passed}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit({"val": val, "holdout": holdout}[sys.argv[1]]())
