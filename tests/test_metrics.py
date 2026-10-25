"""Scoring rules must agree with the reference implementations."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import log_loss as sklearn_log_loss

from plpred.eval.metrics import accuracy, brier, evaluate, log_loss, rps


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([3, 2, 3], size=500)
    y = np.array([rng.choice(3, p=p) for p in probs])
    return probs, y


def test_log_loss_matches_sklearn(sample) -> None:
    probs, y = sample
    assert log_loss(probs, y) == pytest.approx(sklearn_log_loss(y, probs), abs=1e-10)


def test_perfect_forecast_scores_zero() -> None:
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    y = np.array([0, 2])
    assert log_loss(probs, y) < 1e-12
    assert brier(probs, y) < 1e-12
    assert accuracy(probs, y) == 1.0
    assert rps(probs, y) < 1e-12


def test_uniform_forecast_is_worse_than_sharp_correct_forecast() -> None:
    y = np.array([0, 0, 0, 0])
    uniform = np.full((4, 3), 1 / 3)
    sharp = np.array([[0.7, 0.2, 0.1]] * 4)
    assert log_loss(sharp, y) < log_loss(uniform, y)
    assert rps(sharp, y) < rps(uniform, y)


def test_probabilities_must_be_normalised_like() -> None:
    with pytest.raises(ValueError):
        log_loss(np.array([[0.5, 0.2]]), np.array([0]))


def test_evaluate_returns_all_metrics(sample) -> None:
    probs, y = sample
    out = evaluate(probs, y)
    assert set(out) == {"log_loss", "brier", "rps", "accuracy"}
    assert 0.0 <= out["accuracy"] <= 1.0
    assert 0.0 <= out["rps"] <= 1.0
