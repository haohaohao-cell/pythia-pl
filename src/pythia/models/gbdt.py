"""Gradient-boosted trees on the lagged form features.

HistGradientBoosting is used because it needs no external native library, handles
missing values natively and is fast on ~5k rows.  Samples are weighted by the
same exponential time decay as the Dixon-Coles fit, which matters more than
extra trees on a dataset this small.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ..config import DC_HALF_LIFE_DAYS, RANDOM_STATE
from ..features.rolling import FEATURE_COLUMNS, FEATURE_COLUMNS_XG
from .base import outcome_index, reorder_classes


class GradientBoost:
    name = "gbdt"

    def __init__(
        self,
        max_iter: int = 120,
        learning_rate: float = 0.04,
        max_leaf_nodes: int = 3,
        min_samples_leaf: int = 80,
        l2_regularization: float = 1.0,
        half_life_days: float = DC_HALF_LIFE_DAYS,
        use_xg: bool = False,
    ) -> None:
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.half_life_days = half_life_days
        self.use_xg = use_xg
        self.model: HistGradientBoostingClassifier | None = None

    @property
    def feature_names(self) -> list[str]:
        return list(FEATURE_COLUMNS_XG) if self.use_xg else list(FEATURE_COLUMNS)

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self.feature_names if c in frame.columns]
        if not cols:
            raise ValueError("no feature columns found in frame")
        return frame[cols].to_numpy(dtype=float)

    def fit(self, train: pd.DataFrame) -> "GradientBoost":
        X = self._matrix(train)
        y = outcome_index(train)
        decay = np.log(2.0) / self.half_life_days
        age = (train["date"].max() - train["date"]).dt.days.clip(lower=0).to_numpy(dtype=float)
        weights = np.exp(-decay * age)

        self.model = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=False,
            random_state=RANDOM_STATE,
        )
        self.model.fit(X, y, sample_weight=weights)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "fit() first"
        return reorder_classes(self.model, self.model.predict_proba(self._matrix(frame)))
