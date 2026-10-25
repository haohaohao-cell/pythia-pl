"""Temperature scaling and the Brier decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pythia.eval.decomposition import brier_decomposition
from pythia.eval.metrics import brier
from pythia.models.calibration import apply_temperature, calibrate_by_season, fit_temperature


def test_temperature_one_is_identity() -> None:
    probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    assert np.allclose(apply_temperature(probs, 1.0), probs)


def test_high_temperature_flattens() -> None:
    probs = np.array([[0.8, 0.15, 0.05]])
    flat = apply_temperature(probs, 2.0)
    assert flat[0, 0] < probs[0, 0]
    assert np.isclose(flat.sum(), 1.0)


def test_low_temperature_sharpens() -> None:
    probs = np.array([[0.45, 0.30, 0.25]])
    sharp = apply_temperature(probs, 0.6)
    assert sharp[0, 0] > probs[0, 0]
    assert np.isclose(sharp.sum(), 1.0)


def test_fit_recovers_overconfidence() -> None:
    """Probabilities that overstate the truth should want T > 1."""
    rng = np.random.default_rng(0)
    truth = np.array([0, 0, 1, 2, 2, 0, 1, 1])
    probs = np.array([
        [0.85, 0.10, 0.05], [0.80, 0.12, 0.08], [0.05, 0.10, 0.85], [0.05, 0.10, 0.85],
        [0.05, 0.10, 0.85], [0.85, 0.10, 0.05], [0.05, 0.10, 0.85], [0.05, 0.10, 0.85],
    ])
    y = truth.copy()
    y[6] = 2  # one confident draw miss to make the fit non-trivial
    t = fit_temperature(probs, y)
    assert t > 1.0


def test_fit_is_bounded() -> None:
    probs = np.array([[0.4, 0.3, 0.3], [0.2, 0.5, 0.3]])
    y = np.array([0, 1])
    assert 0.5 <= fit_temperature(probs, y) <= 3.0


def test_calibration_is_causal() -> None:
    """The first season has no history, so it must stay at T = 1."""
    preds = pd.DataFrame({
        "match_id": [f"m{i}" for i in range(300)],
        "season": ["2020-21"] * 150 + ["2021-22"] * 150,
        "model": ["blend"] * 300,
        "ftr": (["H", "D", "A"] * 100),
        "p_home": 0.45, "p_draw": 0.30, "p_away": 0.25,
    })
    aug, lambdas = calibrate_by_season(preds, models=("blend",))
    assert lambdas[("blend", "2020-21")] == 1.0
    assert ("blend", "2021-22") in lambdas
    assert set(aug["model"]) == {"blend", "blend+T"}


def test_decomposition_reconstructs_brier() -> None:
    rng = np.random.default_rng(3)
    probs = rng.dirichlet([4, 3, 4], size=400)
    y = np.array([rng.choice(3, p=p) for p in probs])
    parts = brier_decomposition(probs, y, bins=10)
    assert parts["brier"] == pytest.approx(brier(probs, y), abs=0.02)
    assert parts["reliability"] >= 0
    assert parts["resolution"] >= 0


def test_resolution_is_higher_for_sharper_forecasts() -> None:
    rng = np.random.default_rng(5)
    probs = rng.dirichlet([8, 4, 8], size=500)
    y = np.array([rng.choice(3, p=p) for p in probs])
    diffuse = np.tile([1 / 3, 1 / 3, 1 / 3], (len(y), 1))
    assert brier_decomposition(probs, y)["resolution"] > brier_decomposition(diffuse, y)["resolution"]
