"""Understat ingestion and the xG ablation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pythia.eval.ablation import has_xg
from pythia.ingest import understat
from pythia.ingest.understat import _build_frame, _season_candidates


def test_season_ids_cover_both_formats() -> None:
    assert _season_candidates(2021) == ["2122", 2021]
    assert _season_candidates(2026) == ["2627", 2026]


def _raw() -> pd.DataFrame:
    return pd.DataFrame({
        "date": ["2023-08-11 19:00:00", "2023-08-12 11:30:00", None],
        "home_team": ["Burnley", "Arsenal", "Everton"],
        "away_team": ["Manchester City", "Nottingham Forest", "Fulham"],
        "home_xg": [0.31, 0.84, None],
        "away_xg": [2.40, 0.97, 1.10],
        "home_goals": [0, 2, None],
        "away_goals": [3, 1, None],
    })


def test_build_frame_maps_names_and_drops_missing_xg() -> None:
    frame = _build_frame(_raw(), 2023)
    assert len(frame) == 2
    assert set(frame["home"]) == {"burnley", "arsenal"}
    assert set(frame["away"]) == {"man_city", "nottm_forest"}
    assert frame["season"].iat[0] == "2023-24"


def test_merge_uses_season_club_club_key() -> None:
    matches = pd.DataFrame({
        "match_id": ["a", "b"],
        "season": ["2023-24", "2023-24"],
        "home": ["arsenal", "burnley"],
        "away": ["nottm_forest", "man_city"],
    })
    xg = pd.DataFrame({
        "season": ["2023-24", "2023-24"],
        "home": ["arsenal", "burnley"],
        "away": ["nottm_forest", "man_city"],
        "home_xg": [0.84, 0.31],
        "away_xg": [0.97, 2.40],
    })
    merged = understat.merge_into_matches(matches, xg)
    assert len(merged) == 2
    assert list(merged["match_id"]) == ["a", "b"]
    assert merged["hxg"].tolist() == [0.84, 0.31]


def test_merge_leaves_unmatched_rows_missing() -> None:
    matches = pd.DataFrame({
        "match_id": ["a"], "season": ["2023-24"], "home": ["arsenal"], "away": ["fulham"],
    })
    xg = pd.DataFrame(columns=["season", "home", "away", "home_xg", "away_xg"])
    merged = understat.merge_into_matches(matches, xg)
    assert np.isnan(merged["hxg"].iat[0])


def test_has_xg_detects_missing_columns(features) -> None:
    assert not has_xg(features)


def test_has_xg_accepts_populated_columns(features) -> None:
    frame = features.copy()
    for col in ("home_xg_form", "away_xg_form", "home_xg_ag_form", "away_xg_ag_form"):
        frame[col] = 1.3
    assert has_xg(frame)
