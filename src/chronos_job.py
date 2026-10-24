"""Chronos-2 work for IndiaVol, run inside the yug-base container with the cached Chronos venv (no network):

  predict  <model: zeroshot | /data/models/NAME/final> <origins.csv> <out.csv>
  finetune <NAME> <lora|full> <lr> <steps>        -> /data/models/NAME/final (save_pretrained)

/data = ~/india-vol-data (rw). Contexts never extend past the origin (causal). σ̂_h is UNcorrected here; evaluate.py
applies the validation bias correction.
"""
import datetime as dt
import os
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

DATA = Path("/data")
CTX, PRED, BATCH = 512, 10, 128
TRAIN_END = dt.date(2024, 6, 30)
VAL_END = dt.date(2024, 12, 31)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def series():
    out = {}
    for p in sorted((DATA / "series").glob("*.csv")):
        d = pd.read_csv(p)
        out[p.stem] = (np.array([dt.date.fromisoformat(x) for x in d.date]), d.log_rv.values.astype(np.float32))
    return out


def load(model):
    src = "amazon/chronos-2" if model == "zeroshot" else model
    return Chronos2Pipeline.from_pretrained(src, device_map=DEV)


def predict(model, req_path, out_path):
    S, pipe = series(), load(model)
    req = pd.read_csv(req_path)
    rows = []
    for k in range(0, len(req), BATCH):
        chunk = req.iloc[k:k + BATCH]
        ctxs = [torch.tensor(S[r.series][1][max(0, r.i + 1 - CTX):r.i + 1]) for r in chunk.itertuples()]
        q, _ = pipe.predict_quantiles(ctxs, prediction_length=PRED, quantile_levels=[0.5])
        for r, qq in zip(chunk.itertuples(), q):
            med = (qq[0][:, 0] if qq.dim() == 3 else qq[:, 0]).float().cpu().numpy()
            rows.append({"series": r.series, "origin": r.origin,
                         **{f"s{h}": math.sqrt(252 * float(np.exp(med[:h]).mean())) for h in (1, 5, 10)}})
        if k % (BATCH * 20) == 0:
            print(f"  {k}/{len(req)}", file=sys.stderr, flush=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"predicted {len(rows)} -> {out_path}")


def finetune(name, mode, lr, steps):
    S = series()
    train = [v[d <= TRAIN_END] for d, v in S.values() if (d <= TRAIN_END).sum() >= 200]
    val = [v[d <= VAL_END] for d, v in S.values() if (d <= VAL_END).sum() >= 300]
    pipe = load("zeroshot")
    out = DATA / "models" / name
    ft = pipe.fit(train, prediction_length=PRED, validation_inputs=val, finetune_mode=mode,
                  learning_rate=float(lr), num_steps=int(steps), batch_size=int(os.getenv("FT_BATCH", "64")), context_length=CTX,
                  output_dir=str(out / "trainer"))
    ft.save_pretrained(str(out / "final"))
    print(f"fine-tuned {name}: {len(train)} train series, {len(val)} validation series -> {out / 'final'}")


if __name__ == "__main__":
    print(f"device {DEV}", file=sys.stderr)
    {"predict": lambda: predict(*sys.argv[2:5]), "finetune": lambda: finetune(*sys.argv[2:6])}[sys.argv[1]]()
