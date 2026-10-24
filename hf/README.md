---
license: apache-2.0
base_model: amazon/chronos-2
base_model_relation: finetune
pipeline_tag: time-series-forecasting
library_name: chronos-forecasting
tags:
- time-series-forecasting
- chronos-forecasting
- chronos-2
- finance
- volatility
- realized-volatility
- india
- nse
---

# IndiaVol — Chronos-2 fine-tuned for NSE realized volatility

IndiaVol forecasts how volatile Indian stocks and indices will be over the next **1, 5 and 10 trading days**. It is
[amazon/chronos-2](https://huggingface.co/amazon/chronos-2), fully fine-tuned on daily realized variance of
5 NSE/BSE indices and ~206 NSE F&O stocks (2022-01 → 2024-06).

**It does not predict price direction, and it is not a trading strategy.** It is a volatility forecast, useful as an
input for option pricing, position sizing and stops.

Code, pre-registration and full evaluation: **https://github.com/chelvadataer/india-vol**

## Headline result (sealed holdout, scored once)
The evaluation plan and publish rule were written down **before any model was run**
([pre-registration](https://github.com/chelvadataer/india-vol/blob/master/docs/2026-10-24-design-and-prereg.md)).
The holdout (2025-01 → 2026-10, 211 series, 20,323 forecast origins) was opened once, after the model variant and
its bias correction had been frozen on the validation period.

QLIKE on variance (lower is better):

| model | h = 1 | h = 5 | h = 10 |
|---|---|---|---|
| RV20 (trailing 20-day close-to-close vol) | 0.562 | 0.351 | 0.306 |
| HAR (log-HAR, direct h-step, expanding OLS) | 0.317 | 0.190 | 0.170 |
| Chronos-2 zero-shot (bias-corrected) | 0.212 | 0.134 | 0.122 |
| **IndiaVol (this model, bias-corrected)** | **0.205** | **0.130** | **0.117** |

**Publish rule** (h = 5): beat HAR in a pooled day-block bootstrap **and** on ≥ 4 of 5 indices.
- Pooled QLIKE(IndiaVol) − QLIKE(HAR) = **−0.060**, 95 % CI [−0.077, −0.044].
- Per index (Diebold-Mariano, HAC, one-sided):

  | index | IndiaVol | HAR | DM t | p |
  |---|---|---|---|---|
  | NIFTY | 0.176 | 0.246 | −2.75 | 0.003 |
  | BANKNIFTY | 0.166 | 0.239 | −2.63 | 0.004 |
  | FINNIFTY | 0.165 | 0.232 | −2.82 | 0.002 |
  | MIDCPNIFTY | 0.158 | 0.193 | −2.70 | 0.004 |
  | SENSEX | 0.178 | 0.247 | −2.68 | 0.004 |

Rule passed: 5 / 5.

### Read this before you use it
- **Most of the gain over HAR comes from Chronos-2 itself, not from the fine-tune.** On the holdout, fine-tuning adds
  a small further improvement over zero-shot Chronos-2 (h = 5: −0.004, 95 % CI [−0.009, −0.0002]; a secondary,
  not pre-registered comparison). On the validation period zero-shot was slightly *better* (0.1287 vs 0.1301).
  If you only need "better than HAR", bias-corrected zero-shot Chronos-2 gets you almost all of the way.
- **Bias correction is part of the model.** Raw medians are biased; apply the frozen factors below.
- **Pretraining overlap:** we cannot rule out that Chronos-2's own pretraining corpus contains series overlapping
  the 2025 holdout period.
- **Survivorship:** the stock universe is today's F&O list, not point-in-time.
- **Not trading advice.** A better volatility forecast does not by itself make money.

## Training details
| | |
|---|---|
| Target series | daily log realized variance: Σ squared 5-min log returns 09:15–15:30 IST + squared overnight return |
| Data | 1-minute bars, public Upstox v3 historical-candle API (not redistributed) |
| Train / val / holdout | ≤ 2024-06-30 / 2024-07 → 2024-12 / 2025-01 → 2026-10 |
| Candidates (picked on validation h = 5 QLIKE) | LoRA lr 1e-5 × 1,000 and × 3,000 steps; **full fine-tune lr 1e-6 × 1,000 steps (selected)** |
| Context / horizon | 512 days / 10 days, batch 64, CPU |
| Forecast origins | indices every trading day, stocks every 5th trading day |

## Usage
```python
import math, numpy as np, torch
from chronos import Chronos2Pipeline   # pip install "chronos-forecasting>=2.3"

pipe = Chronos2Pipeline.from_pretrained("chelvaser/indiavol-chronos2", device_map="cpu")
BIAS = {1: 0.016230, 5: -0.079483, 10: -0.103922}   # frozen on validation (see indiavol_config.json)

# log_rv: 1-D array of daily log realized variance up to and including today (most recent last)
q, _ = pipe.predict_quantiles([torch.tensor(log_rv[-512:], dtype=torch.float32)],
                              prediction_length=10, quantile_levels=[0.5])
med = q[0][0, :, 0].numpy() if q[0].dim() == 3 else q[0][:, 0].numpy()
sigma = {h: math.sqrt(252 * np.exp(med[:h]).mean()) * math.exp(-BIAS[h]) for h in (1, 5, 10)}
print(sigma)   # annualised σ forecast over the next 1 / 5 / 10 trading days
```
Computing daily realized variance from 1-minute bars: `src/panel.py` in the GitHub repo.

## License
Apache-2.0, same as the base model. Chronos-2 is © Amazon.
