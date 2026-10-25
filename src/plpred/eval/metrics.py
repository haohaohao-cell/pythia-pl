"""Scoring rules for probabilistic forecasts.

Accuracy is reported but never optimised: it throws away confidence and rewards
hedging on the most likely class.  Log loss is the primary metric, with the
Ranked Probability Score (which respects the ordinal nature of H > D > A) as a
secondary.
"""

from __future__ import annotations

import numpy as np


def _check(probs: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError("probs must have shape (n, 3)")
    if len(probs) != len(y):
        raise ValueError("probs and y length mismatch")
    return probs, y


def log_loss(probs: np.ndarray, y: np.ndarray, eps: float = 1e-15) -> float:
    probs, y = _check(probs, y)
    p = np.clip(probs[np.arange(len(y)), y], eps, 1.0)
    return float(-np.mean(np.log(p)))


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    probs, y = _check(probs, y)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    probs, y = _check(probs, y)
    return float(np.mean(np.argmax(probs, axis=1) == y))


def rps(probs: np.ndarray, y: np.ndarray) -> float:
    probs, y = _check(probs, y)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    cum_p = np.cumsum(probs, axis=1)[:, :-1]
    cum_o = np.cumsum(onehot, axis=1)[:, :-1]
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / (probs.shape[1] - 1)))


def mean_confidence(probs: np.ndarray) -> float:
    return float(np.mean(np.max(np.asarray(probs, dtype=float), axis=1)))


def evaluate(probs: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "log_loss": log_loss(probs, y),
        "brier": brier(probs, y),
        "rps": rps(probs, y),
        "accuracy": accuracy(probs, y),
    }


def per_class_recall(probs: np.ndarray, y: np.ndarray, names=("H", "D", "A")) -> dict[str, float]:
    probs, y = _check(probs, y)
    pred = np.argmax(probs, axis=1)
    out = {}
    for i, name in enumerate(names):
        mask = y == i
        out[name] = float(np.mean(pred[mask] == i)) if mask.any() else float("nan")
    return out
