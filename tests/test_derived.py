"""Over/under 2.5 evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pythia.eval.derived import binary_log_loss, devig_two_way, over_probability
from pythia.models.dixon_coles import DixonColes

from conftest import make_matches


def test_devig_sums_to_one() -> None:
    p = devig_two_way(np.array([1.8, 2.1]), np.array([2.2, 2.05]))
    assert np.allclose(p + (1 - p), 1.0)
    assert p[0] > 0.5  # the shorter over price implies a higher over probability


def test_devig_prefers_the_shorter_price() -> None:
    p = devig_two_way(np.array([1.5]), np.array([2.6]))
    assert 0.6 < p[0] < 0.65


def test_devig_handles_missing_prices() -> None:
    p = devig_two_way(np.array([np.nan]), np.array([2.0]))
    assert np.isnan(p[0])


def test_over_probability_is_a_distribution() -> None:
    matches = make_matches(n_seasons=2, seed=3)
    dc = DixonColes().fit(matches)
    p = over_probability(dc, "alpha", "bravo")
    assert 0.0 < p < 1.0


def test_scaling_rates_moves_the_total_market() -> None:
    matches = make_matches(n_seasons=2, seed=3)
    dc = DixonColes().fit(matches)
    base = over_probability(dc, "alpha", "bravo")
    high = 1.0 - sum(
        dc.score_matrix("alpha", "bravo", scale=1.25)[i, j]
        for i in range(dc.max_goals) for j in range(dc.max_goals) if i + j <= 2
    )
    assert high > base


def test_binary_log_loss_bounds() -> None:
    perfect = binary_log_loss(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    assert perfect < 1e-12
    assert binary_log_loss(np.array([0.5]), np.array([1.0])) == pytest.approx(np.log(2))


def test_binary_log_loss_penalises_confident_errors() -> None:
    y = np.array([1.0])
    assert binary_log_loss(np.array([0.95]), y) < binary_log_loss(np.array([0.6]), y)
    assert binary_log_loss(np.array([0.05]), y) > binary_log_loss(np.array([0.6]), y)


def test_feature_table_carries_over_under_columns(matches) -> None:
    """add_form_features must not drop columns it does not understand."""
    from pythia.features.elo import add_elo
    from pythia.features.rolling import add_form_features

    frame = matches.copy()
    frame["ou_over"] = 1.9
    frame["ou_under"] = 1.9
    out = add_form_features(add_elo(frame))
    assert {"ou_over", "ou_under"}.issubset(out.columns)
