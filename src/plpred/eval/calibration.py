"""Calibration diagnostics: are the stated probabilities honest?"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reliability_curve(
    p: np.ndarray,
    observed: np.ndarray,
    bins: int = 10,
    edges: np.ndarray | None = None,
) -> pd.DataFrame:
    """Bin forecasts by predicted probability and compare with the hit rate."""
    p = np.asarray(p, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if edges is None:
        edges = np.linspace(0.0, 1.0, bins + 1)
    edges = np.asarray(edges, dtype=float)
    n_bins = len(edges) - 1
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        mean_pred = float(p[mask].mean())
        obs = float(observed[mask].mean())
        se = float(np.sqrt(max(obs * (1 - obs), 1e-9) / n))
        rows.append({
            "bin_lo": float(edges[b]),
            "bin_hi": float(edges[b + 1]),
            "n": n,
            "mean_pred": mean_pred,
            "observed": obs,
            "se": se,
        })
    return pd.DataFrame(rows)


def expected_calibration_error(curve: pd.DataFrame) -> float:
    """Sample-weighted mean absolute gap between forecast and outcome rate."""
    if curve.empty:
        return float("nan")
    weights = curve["n"] / curve["n"].sum()
    return float(np.sum(weights * (curve["mean_pred"] - curve["observed"]).abs()))
