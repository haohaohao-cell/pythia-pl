"""Dixon-Coles properties: normalisation, home advantage, recovered strengths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pythia.models.dixon_coles import DixonColes

from conftest import make_matches


@pytest.fixture(scope="module")
def fitted():
    strengths = {
        "alpha": 0.8, "bravo": 0.4, "charlie": 0.0,
        "delta": -0.3, "echo": -0.6, "foxtrot": -0.9,
    }
    matches = make_matches(n_seasons=4, goals=strengths, seed=11)
    return DixonColes(half_life_days=400).fit(matches), strengths, matches


def test_probs_are_a_distribution(fitted) -> None:
    model, _, matches = fitted
    probs = model.predict_proba(matches.head(20))
    assert probs.shape == (20, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0).all()


def test_score_matrix_is_normalised(fitted) -> None:
    model, _, _ = fitted
    matrix = model.score_matrix("alpha", "foxtrot")
    assert matrix.shape == (model.max_goals, model.max_goals)
    assert matrix.sum() == pytest.approx(1.0)


def test_home_advantage_is_positive(fitted) -> None:
    model, _, _ = fitted
    assert model.home_adv_ > 0


def test_stronger_teams_get_higher_attack(fitted) -> None:
    model, strengths, _ = fitted
    table = model.strength_table().set_index("team")["attack"]
    ordered = [table[t] for t in sorted(strengths, key=lambda t: -strengths[t])]
    assert ordered == sorted(ordered, reverse=True)


def test_strong_team_beats_weak_team_more_often(fitted) -> None:
    model, _, _ = fitted
    strong = model.outcome_probs("alpha", "foxtrot")
    weak = model.outcome_probs("foxtrot", "alpha")
    assert strong[0] > weak[0]


def test_unknown_teams_do_not_crash(fitted) -> None:
    model, _, _ = fitted
    probs = model.outcome_probs("brand_new_club", "alpha")
    assert probs.sum() == pytest.approx(1.0)


def test_time_decay_prefers_recent_form() -> None:
    """A club that only wins lately should out-rank one that only won long ago."""
    base = make_matches(n_seasons=3, seed=5)
    shifted = base.copy()
    early = shifted["date"] < shifted["date"].quantile(0.4)
    late = shifted["date"] > shifted["date"].quantile(0.6)
    target_home, target_away = "alpha", "bravo"
    mask_early = early & (shifted["home"] == target_home)
    mask_late = late & (shifted["home"] == target_home)
    shifted.loc[mask_early, ["fthg", "ftag"]] = [0, 3]
    shifted.loc[mask_late, ["fthg", "ftag"]] = [3, 0]

    fresh = DixonColes(half_life_days=120).fit(shifted)
    stale = DixonColes(half_life_days=100_000).fit(shifted)
    assert fresh._rates(target_home, target_away)[0] > stale._rates(target_home, target_away)[0]
