import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import panel as P  # noqa: E402


def test_forecasts_are_prefix_invariant_and_the_target_is_not():
    """Leaky-oracle check (trial-ledger rule): σ̂ at origin i must not move when data after i is scrambled."""
    rng = np.random.default_rng(3)
    rv = np.exp(rng.normal(-9, 0.6, 700))
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 700)))
    i, h = 600, 5
    rv2, c2 = rv.copy(), close.copy()
    rv2[i + 1:] = np.exp(rng.normal(-6, 1, len(rv) - i - 1))
    c2[i + 1:] = close[i] * np.exp(np.cumsum(rng.normal(0, 0.05, len(close) - i - 1)))
    assert math.isclose(P.har_sigma(rv, P.har_features(rv), i, h), P.har_sigma(rv2, P.har_features(rv2), i, h))
    assert math.isclose(P.rv20_sigma(close, i), P.rv20_sigma(c2, i))
    assert not math.isclose(P.realized_sigma(rv, i, h), P.realized_sigma(rv2, i, h))


def test_qlike_zero_at_truth():
    assert P.qlike([0.2], [0.2])[0] == 0 and P.qlike([0.1], [0.2])[0] > 0
