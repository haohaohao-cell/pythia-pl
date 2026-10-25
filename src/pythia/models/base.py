"""Model interface shared by every forecaster in the project."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class Model(Protocol):
    """A three-way (home / draw / away) probabilistic classifier."""

    name: str

    def fit(self, train: pd.DataFrame) -> "Model": ...

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return an ``(n, 3)`` array ordered home, draw, away."""
        ...


def outcome_index(frame: pd.DataFrame) -> np.ndarray:
    return frame["ftr"].map({"H": 0, "D": 1, "A": 2}).to_numpy(dtype=int)


def reorder_classes(model, proba: np.ndarray) -> np.ndarray:
    """Map a classifier's class order to the canonical (H, D, A) columns.

    Classifiers trained on integer-encoded labels expose ``classes_ = [0, 1, 2]``
    which already matches; string labels arrive alphabetically as ``['A','D','H']``.
    """
    mapping = {"H": 0, "D": 1, "A": 2}
    classes = list(model.classes_)
    if all(isinstance(c, (int, np.integer)) for c in classes):
        order = [int(c) for c in classes]
    else:
        order = [mapping[c] for c in classes]
    out = np.empty_like(proba)
    for src, dst in enumerate(order):
        out[:, dst] = proba[:, src]
    return out
