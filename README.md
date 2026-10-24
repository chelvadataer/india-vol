# IndiaVol — realized-volatility forecasting for Indian markets

IndiaVol forecasts how volatile NSE indices and F&O stocks will be over the next **1, 5 and 10 trading days**. It
does **not** predict price direction.

Status: **done.** The holdout was opened once (2026-10-24) and the fine-tuned model **passed** the pre-registered
publish rule. Model: **[chelvaser/indiavol-chronos2](https://huggingface.co/chelvaser/indiavol-chronos2)**.

## Results (holdout 2025-01 → 2026-10, 211 series, 20,323 origins; QLIKE, lower = better)
| model | h = 1 | h = 5 | h = 10 |
|---|---|---|---|
| B0 RV20 | 0.562 | 0.351 | 0.306 |
| B1 HAR | 0.317 | 0.190 | 0.170 |
| B2 Chronos-2 zero-shot (bias-corrected) | 0.212 | 0.134 | 0.122 |
| **M1 IndiaVol (full fine-tune, bias-corrected)** | **0.205** | **0.130** | **0.117** |

- Publish rule (h = 5): pooled M1 − HAR = −0.060, day-bootstrap 95 % CI [−0.077, −0.044]; M1 beats HAR on **5 / 5**
  indices (DM p ≤ 0.004 each). **Passed.**
- **Honest caveat:** most of the gain over HAR comes from Chronos-2 itself. Fine-tuning adds a small extra gain on the
  holdout (h = 5: −0.004, CI [−0.009, −0.0002], secondary / not pre-registered), and zero-shot was slightly better on
  validation. Full table and caveats: [`hf/README.md`](hf/README.md) (the model card).

## Why volatility, not direction
Short-horizon price direction failed every honest out-of-sample test we ran: rule searches, "smart money" setups,
momentum, RL, and LLM agents and judges. Volatility *is* forecastable and useful for option selling, position sizing
and stops. In earlier work a classic HAR model, and Chronos-2 even zero-shot, beat trailing 20-day volatility on all
five NSE indices.

## What it is
- **Target:** daily realized variance = Σ squared 5-minute log returns (09:15–15:30 IST) + the squared overnight
  return. Forecasts are annualised σ over the next h days.
- **Universe:** NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, SENSEX and all current NSE F&O stock underlyings (~213), with
  1-minute data from 2022.
- **Candidates:**

  | id | model |
  |---|---|
  | B0 | RV20, trailing 20-day close-to-close vol |
  | B1 | log-HAR, direct h-step, expanding-window OLS |
  | B2 | [Chronos-2](https://huggingface.co/amazon/chronos-2) zero-shot on daily log RV, bias-corrected on validation |
  | **M1** | **Chronos-2 fine-tuned** on the training period (LoRA / full; the variant is picked on validation) |

- **Splits:** train 2022-01 → 2024-06, validation 2024-07 → 2024-12, **holdout 2025-01 → present (opened once)**.

## Pre-registration and publish rule
Everything above was frozen in [`docs/2026-10-24-design-and-prereg.md`](docs/2026-10-24-design-and-prereg.md)
**before any model was run**. M1 is published as "IndiaVol" only if, on the holdout, its h = 5 QLIKE beats HAR
(pooled day-block bootstrap CI excludes 0) **and** it beats HAR on at least 4 of the 5 indices. Otherwise the
comparison itself is published, with M1 shown losing where it loses.

## Reproduce
```bash
python src/fetch_1m.py            # 1m history from the public Upstox v3 historical-candle API (no token)
python src/panel.py && python src/export.py
./run_chronos.sh predict zeroshot /data/origins_val.csv /data/pred_b2_zeroshot_val.csv
./run_chronos.sh finetune m1_lora_1000 lora 1e-5 1000
python src/evaluate.py val        # freezes the model selection
python src/evaluate.py holdout    # refuses without the frozen selection; run once
```
`pipeline.sh` runs the whole sequence. `run_chronos.sh` assumes a local container with `chronos-forecasting` 2.3
installed; adapt the paths to your setup.

## Data
The raw market data is **not** included or redistributed. It comes from Upstox's public historical-candle API, and
you must fetch it yourself under their terms. Only code, model weights (if published) and aggregate metrics are
shared.

## Caveats
- **Survivorship:** the stock universe is today's F&O list. This matters less for volatility than for returns, but
  it is not point-in-time.
- **Not trading advice.** A better volatility forecast is not a trading strategy by itself.

## License
Apache-2.0 (see `LICENSE`). Chronos-2 is © Amazon, Apache-2.0.
