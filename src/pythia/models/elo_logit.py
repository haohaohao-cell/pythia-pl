"""Elo difference -> outcome probabilities via a multinomial logistic fit.

Elo alone is a strong baseline for football; letting a one-feature softmax learn
the mapping is more honest than hand-tuned rating-to-probability formulas,
because it is fitted on the same data the other models see.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .base import outcome_index, reorder_classes


class EloLogit:
    name = "elo_logit"

    def __init__(self, c: float = 1.0) -> None:
        self.c = c
        self.model: LogisticRegression | None = None

    @staticmethod
    def _features(frame: pd.DataFrame) -> np.ndarray:
        diff = frame["elo_diff"].to_numpy(dtype=float)
        return diff.reshape(-1, 1)

    def fit(self, train: pd.DataFrame) -> "EloLogit":
        self.model = LogisticRegression(C=self.c, max_iter=2000)
        self.model.fit(self._features(train), outcome_index(train))
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "fit() first"
        return reorder_classes(self.model, self.model.predict_proba(self._features(frame)))
