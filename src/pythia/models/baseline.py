"""Baseline: the empirical home/draw/away frequencies of the training window.

Anything that cannot beat this is not a model, it is noise.  Reported first in
the leaderboard on purpose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class BaseRates:
    name = "base_rates"

    def __init__(self) -> None:
        self.probs = np.array([0.46, 0.24, 0.30])

    def fit(self, train: pd.DataFrame) -> "BaseRates":
        rates = train["ftr"].value_counts(normalize=True)
        self.probs = np.array([rates.get("H", 0.0), rates.get("D", 0.0), rates.get("A", 0.0)])
        self.probs = self.probs / self.probs.sum()
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.tile(self.probs, (len(frame), 1))
