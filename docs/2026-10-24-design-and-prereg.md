# IndiaVol — a realized-volatility forecaster for Indian markets: design + pre-registration (2026-10-24)

Owner request: "create a new model and train it for market prediction and put it on Hugging Face". The owner skipped
the questions; every choice below is my call, stated plainly.

## What it predicts, and why not price direction
- **Target:** forecast **realized volatility** over the next **1, 5 and 10 trading days** for NSE indices and liquid
  F&O stocks.
- **Why not direction:** every honest direction test in this project failed: 700-config search, SMC, KN, momentum,
  RL, LLM agents, and the shadow-judge comparison 2026-10-24. Volatility *is* forecastable, and it is useful (option
  selling, sizing, stops):
  - VRP M8: HAR beat RV20 on all 5 indices, and Chronos-2 zero-shot was the best forecaster, though biased low.
  - arXiv 2511.18578: finance-specific training is what makes foundation models work on financial series.
- **Anti-goal:** a model card that implies trading profit. The card reports forecast accuracy vs simple baselines on
  a sealed holdout, nothing else.

## Data
- **Instruments:**
  - 5 indices: NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, SENSEX (already in `~/mtf-data`);
  - all current NSE F&O stock underlyings (≈ 200).
  - Survivorship note: today's F&O list. This matters less for volatility than for returns; stated on the card.
- **Raw data:** 1m bars from the public Upstox v3 historical-candle API, 2022-01 → 2026-10.
- **Daily realized variance:** Σ squared 5-min log returns 09:15–15:30 + squared overnight return (as in VRP M8).
- **Licensing:** the raw market data is **not** redistributed. Only model weights, code and aggregate metrics are
  published.

## Splits (by date, all instruments together)
- **Train:** 2022-01 → 2024-06.
- **Validation:** 2024-07 → 2024-12 (model / epoch selection only).
- **Holdout:** 2025-01-01 → end. Opened **once**, for the final comparison. Nothing is tuned after that.

## Models (the only candidates)
| id | model |
|---|---|
| B0 | RV20: 20-day close-to-close vol (naive) |
| B1 | HAR: log-HAR direct h-step, expanding-window OLS (as in M8) |
| B2 | Chronos-2 zero-shot on daily log RV, median, **bias-corrected** by the mean log error on train + validation |
| **M1** | **Chronos-2 fine-tuned** on train-period log-RV series of all instruments (context ≤ 512), checkpoint chosen on validation QLIKE; same bias correction |

## Metrics
- **QLIKE on variance** (primary) and MSE of log σ, at h = 1, 5, 10.
- **Diebold-Mariano (HAC)** vs B1 HAR, per series.
- **Pooled DM** over all series with a day-block bootstrap (series are cross-correlated).

## Publish rule (decided now)
- **M1 is published as "IndiaVol"** only if, on the holdout, its h = 5 QLIKE beats **HAR** in a pooled day-block
  bootstrap (CI excludes 0) **and** it beats HAR on ≥ 4 of 5 indices.
- **Otherwise:** the honest result is published instead (repo with code, HAR + bias-corrected Chronos baselines and
  the comparison), and the card says the fine-tune did not beat HAR.
- **Either way** the card shows the full holdout table, including where it loses.

## How it's built
- **Repo:** `~/server_setup/india-vol` (private GitHub mirror later).
- **Python:** reuse the upstox-trade-bot venv for data and the `yug-base` container + Chronos venv for Chronos
  work; GPU if free (≈ 4 GB needed), else CPU.
- **Trial ledger:** family `vol-forecast` in `~/trial-ledger`.
- **Hugging Face:** needs the owner's login (`hf auth login`). The repo name, public/private and licence (Apache-2.0,
  as Chronos-2) are confirmed with the owner **before** any upload.
