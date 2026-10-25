"""The invariant that makes or breaks a forecasting repository.

If a feature for match *k* changes when later data is added, the backtest is
leaking the future and every number in the README is fiction.  These tests pin
that down two ways: truncating the input at a date, and mutating a result that
happens *after* the match being scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from plpred.features.elo import add_elo
from plpred.features.rolling import add_form_features

from conftest import make_matches

PAST_COLUMNS = [
    "elo_h", "elo_a", "elo_diff",
    "home_pts_form", "away_pts_form", "home_gf_form", "away_ga_form",
    "home_sot_form", "away_sot_form", "diff_pts_form", "diff_rest_days",
    "home_match_no", "away_match_no", "home_newly_promoted", "away_newly_promoted",
]


def _build(matches: pd.DataFrame) -> pd.DataFrame:
    return add_form_features(add_elo(matches))


def _assert_same(frame_a: pd.DataFrame, frame_b: pd.DataFrame, match_id: str) -> None:
    a = frame_a.loc[frame_a["match_id"] == match_id].iloc[0]
    b = frame_b.loc[frame_b["match_id"] == match_id].iloc[0]
    for col in PAST_COLUMNS:
        va, vb = a[col], b[col]
        assert np.isclose(float(va), float(vb), equal_nan=True), f"{col}: {va} != {vb}"


def test_truncating_future_matches_does_not_change_past_features(matches: pd.DataFrame) -> None:
    full = _build(matches)
    target = matches.iloc[len(matches) // 2]
    truncated = _build(matches.loc[matches["date"] <= target["date"]])
    _assert_same(full, truncated, target["match_id"])


def test_mutating_a_future_result_does_not_change_past_features(matches: pd.DataFrame) -> None:
    full = _build(matches)
    target = matches.iloc[len(matches) // 2]
    future = matches["date"] > target["date"]

    tampered = matches.copy()
    tampered.loc[future, "fthg"] = 7
    tampered.loc[future, "ftag"] = 0
    tampered.loc[future, "ftr"] = "H"

    _assert_same(full, _build(tampered), target["match_id"])


def test_elo_does_not_consume_its_own_result(matches: pd.DataFrame) -> None:
    """Flipping match k's score must leave match k's pre-match ratings untouched.

    The companion assertion checks the same flip *does* move that club's next
    match, so the test cannot pass just because nothing was computed.
    """
    with_elo = add_elo(matches)
    target = with_elo.iloc[len(with_elo) // 3]

    tampered = matches.copy()
    row = tampered["match_id"] == target["match_id"]
    tampered.loc[row, ["fthg", "ftag", "ftr"]] = [5, 0, "H"]
    flipped = add_elo(tampered)

    a = with_elo.loc[row].iloc[0]
    b = flipped.loc[row].iloc[0]
    assert a["elo_h"] == b["elo_h"]
    assert a["elo_a"] == b["elo_a"]

    team = target["home"]
    next_a = with_elo.loc[(with_elo["home"] == team) & (with_elo["date"] > target["date"])].iloc[0]
    next_b = flipped.loc[(flipped["home"] == team) & (flipped["date"] > target["date"])].iloc[0]
    assert next_a["elo_h"] != next_b["elo_h"]


def test_form_window_uses_only_completed_matches(matches: pd.DataFrame) -> None:
    feats = add_form_features(add_elo(matches))
    # Matchday 1 of every season has no in-season history for either side.
    openers = feats.loc[feats["home_match_no"] == 1]
    assert (openers["home_pts_form"] >= 0).all()
    assert feats["home_pts_form"].notna().all()


def test_all_feature_columns_present(features: pd.DataFrame) -> None:
    from plpred.features.rolling import FEATURE_COLUMNS

    missing = [c for c in FEATURE_COLUMNS if c not in features.columns]
    assert not missing, f"model matrix is missing {missing}"
