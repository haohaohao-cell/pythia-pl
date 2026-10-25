"""Murphy decomposition of the multiclass Brier score.

Brier = reliability - resolution + uncertainty.

* **reliability** — how far stated probabilities sit from observed frequencies
  (calibration error; lower is better).
* **resolution** — how much the forecast varies across bins, i.e. how much the
  model actually knows (higher is better).
* **uncertainty** — the irreducible variance of the outcome itself.

The point of the decomposition here: if two forecasts have similar reliability
but different resolution, the weaker one does not lack calibration, it lacks
information.  That is the honest way to describe a gap to the closing line.
"""

from __future__ import annotations

import numpy as np


def _binary_parts(p: np.ndarray, o: np.ndarray, bins: int) -> tuple[float, float, float]:
    """Reliability, resolution and uncertainty for one class, binned by p."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    n = len(p)
    base = float(o.mean())

    reliability = 0.0
    resolution = 0.0
    for b in range(bins):
        mask = idx == b
        k = int(mask.sum())
        if k == 0:
            continue
        weight = k / n
        observed = float(o[mask].mean())
        predicted = float(p[mask].mean())
        reliability += weight * (predicted - observed) ** 2
        resolution += weight * (observed - base) ** 2

    uncertainty = base * (1.0 - base)
    return reliability, resolution, uncertainty


def brier_decomposition(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> dict[str, float]:
    """Sum the per-class binary decomposition (the standard multiclass extension)."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0

    rel = res = unc = 0.0
    for k in range(probs.shape[1]):
        r, s, u = _binary_parts(probs[:, k], onehot[:, k], bins)
        rel += r
        res += s
        unc += u

    return {
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(unc),
        "brier": float(rel - res + unc),
    }
