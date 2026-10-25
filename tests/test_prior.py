"""Market-prior study: does the model add anything on top of the opening line?"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pythia.eval.metrics import log_loss
from pythia.eval.prior import _mix, best_weight, opening_probabilities


def test_mix_endpoints_are_identity() -> None:
    model = np.array([[0.5, 0.3, 0.2]])
    prior = np.array([[0.2, 0.3, 0.5]])
    assert np.allclose(_mix(model, prior, 0.0), model)
    assert np.allclose(_mix(model, prior, 1.0), prior)


def test_mix_stays_a_distribution() -> None:
    rng = np.random.default_rng(0)
    model = rng.dirichlet([2, 2, 2], size=50)
    prior = rng.dirichlet([2, 2, 2], size=50)
    mixed = _mix(model, prior, 0.3)
    assert np.allclose(mixed.sum(axis=1), 1.0)


def test_weight_prefers_a_useful_prior() -> None:
    y = np.array([0, 0, 1, 2, 2, 0])
    prior = np.eye(3)[y] * 0.9 + 0.0333  # almost perfect
    prior = prior / prior.sum(axis=1, keepdims=True)
    model = np.full((len(y), 3), 1 / 3)
    assert best_weight(model, prior, y) > 0.9


def test_weight_ignores_a_useless_prior() -> None:
    y = np.array([0, 0, 1, 2, 2, 0])
    model = np.eye(3)[y] * 0.9 + 0.0333
    model = model / model.sum(axis=1, keepdims=True)
    prior = np.full((len(y), 3), 1 / 3)
    assert best_weight(model, prior, y) < 0.1


def test_opening_probabilities_are_normalised() -> None:
    features = pd.DataFrame({
        "odds_open_h": [2.0, np.nan],
        "odds_open_d": [3.4, 3.4],
        "odds_open_a": [3.8, 3.8],
    })
    probs = opening_probabilities(features)
    assert probs.iloc[0].sum() == pytest.approx(1.0)
    assert probs.iloc[1].isna().all()


def test_mixing_with_the_prior_never_loses_to_the_better_input() -> None:
    rng = np.random.default_rng(7)
    y = rng.integers(0, 3, 400)
    model = rng.dirichlet([3, 2, 3], size=400)
    prior = rng.dirichlet([3, 2, 3], size=400)
    w = best_weight(model, prior, y)
    best_single = min(log_loss(model, y), log_loss(prior, y))
    assert log_loss(_mix(model, prior, w), y) <= best_single + 0.02
