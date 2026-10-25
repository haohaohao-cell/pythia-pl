"""De-vigging and the market benchmark fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plpred.odds import implied_probs, market_probs, overround


def test_implied_probs_sum_to_one() -> None:
    odds = np.array([[1.9, 3.5, 4.2], [2.5, 3.2, 2.9]])
    probs = implied_probs(odds)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_overround_is_positive_for_a_real_book() -> None:
    assert overround(np.array([[1.9, 3.5, 4.2]]))[0] > 0.0


def test_missing_or_invalid_odds_become_nan() -> None:
    probs = implied_probs(np.array([[0.0, 3.5, 4.2], [1.9, np.nan, 4.2]]))
    assert np.isnan(probs).all()


def test_market_probs_falls_back_to_opening_prices() -> None:
    frame = pd.DataFrame({
        "odds_close_h": [np.nan], "odds_close_d": [np.nan], "odds_close_a": [np.nan],
        "odds_open_h": [2.0], "odds_open_d": [3.4], "odds_open_a": [3.8],
    })
    out = market_probs(frame)
    assert out.loc[0, "market_src"] == "open"
    assert out[["mkt_h", "mkt_d", "mkt_a"]].to_numpy().sum() == pytest.approx(1.0)


def test_market_probs_prefers_closing_prices() -> None:
    frame = pd.DataFrame({
        "odds_close_h": [2.0], "odds_close_d": [3.4], "odds_close_a": [3.8],
        "odds_open_h": [3.0], "odds_open_d": [3.4], "odds_open_a": [2.4],
    })
    out = market_probs(frame)
    assert out.loc[0, "market_src"] == "close"
    assert out.loc[0, "mkt_h"] > out.loc[0, "mkt_a"]
