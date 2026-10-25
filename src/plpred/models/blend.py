"""Linear opinion pool of the structural and the tree model.

The weight is *not* fitted on the season being predicted.  It is fitted on the
previous season's out-of-fold forecasts, i.e. exactly the information a real
user would have had at the start of the season.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base import Model


def log_loss(probs: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0)
    return float(-np.mean(np.log(p)))


class Blend:
    def __init__(self, models: list[Model], weights: list[float] | None = None, name: str = "blend") -> None:
        self.models = models
        self.weights = weights
        self.name = name

    def fit(self, train: pd.DataFrame) -> "Blend":
        for model in self.models:
            model.fit(train)
        if self.weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.weights is not None
        stacked = np.zeros((len(frame), 3))
        for weight, model in zip(self.weights, self.models):
            stacked += weight * model.predict_proba(frame)
        return stacked


def tune_weights(prob_dict: dict[str, np.ndarray], y: np.ndarray) -> dict[str, float]:
    """Find simplex weights minimising log loss on held-out forecasts."""
    names = list(prob_dict)
    stack = np.stack([prob_dict[n] for n in names])  # (k, n, 3)

    def objective(w):
        w = np.clip(w, 0, None)
        w = w / w.sum()
        return log_loss(np.tensordot(w, stack, axes=1), y)

    k = len(names)
    res = minimize(
        objective,
        np.full(k, 1.0 / k),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
    )
    w = np.clip(res.x, 0, None)
    w = w / w.sum()
    return {n: float(wi) for n, wi in zip(names, w)}
