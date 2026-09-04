"""Paired statistics shared by the depth sweeps.

One home for the row-level bootstrap so the slot sweep (`core_depth_sweep.py`), the
token sweep and any arm-vs-arm readout compute the SAME interval the same way.
"""
from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(sum_a: np.ndarray, sum_b: np.ndarray, count: np.ndarray,
                        n_boot: int = 2000, seed: int = 0,
                        level: float = 0.95) -> dict[str, float]:
    """Token-weighted paired bootstrap of ``mean_a - mean_b`` over resampled units.

    ``sum_a`` / ``sum_b`` are per-unit (row or batch) SUMS of the two paired
    measurements and ``count`` the per-unit number of scored items, so a resample's
    point estimate is ``Σsum_a/Σcount − Σsum_b/Σcount`` — the same token-weighted mean
    the sweeps print, not a mean of per-row means. Returns the point estimate on the
    full data plus the percentile interval at ``level``. Deterministic in ``seed``.
    """
    sum_a = np.asarray(sum_a, dtype=np.float64)
    sum_b = np.asarray(sum_b, dtype=np.float64)
    count = np.asarray(count, dtype=np.float64)
    if not (sum_a.shape == sum_b.shape == count.shape) or sum_a.ndim != 1:
        raise ValueError("sum_a, sum_b and count must be 1-D arrays of the same length")
    n = sum_a.shape[0]
    if n == 0 or float(count.sum()) <= 0.0:
        raise ValueError("bootstrap needs at least one unit with a positive count")
    point = float(sum_a.sum() / count.sum() - sum_b.sum() / count.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    ca = sum_a[idx].sum(axis=1)
    cb = sum_b[idx].sum(axis=1)
    cn = count[idx].sum(axis=1)
    diffs = ca / cn - cb / cn
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(diffs, [alpha, 1.0 - alpha])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_units": int(n),
            "n_boot": int(n_boot), "level": float(level)}
