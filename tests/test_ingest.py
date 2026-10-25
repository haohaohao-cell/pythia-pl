"""Parsing the raw CSVs, including the 2025-26 Pinnacle column rename."""

from __future__ import annotations

import pandas as pd

from pythia.ingest.football_data import normalise
from pythia.teams import team_key


def _raw(**overrides) -> pd.DataFrame:
    base = {
        "Div": ["E0", "E0"],
        "Date": ["16/08/2024", "17/08/24"],
        "HomeTeam": ["Man United", "Ipswich"],
        "AwayTeam": ["Fulham", "Liverpool"],
        "FTHG": ["1", "0"],
        "FTAG": ["0", "2"],
        "HS": ["14", "7"], "AS": ["10", "18"],
        "HST": ["5", "2"], "AST": ["2", "5"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_dates_parse_both_formats() -> None:
    out = normalise(_raw(), "2024-25")
    assert out["date"].iloc[0] == pd.Timestamp("2024-08-16")
    assert out["date"].iloc[1] == pd.Timestamp("2024-08-17")


def test_team_names_map_to_canonical_keys() -> None:
    out = normalise(_raw(), "2024-25")
    assert out["home"].tolist() == ["man_united", "ipswich"]
    assert out["away"].tolist() == ["fulham", "liverpool"]
    assert team_key("Nott'm Forest") == "nottm_forest"
    assert team_key("Nottingham Forest") == "nottm_forest"


def test_pinnacle_legacy_columns_resolve() -> None:
    raw = _raw(PSCH=["1.6", "8.5"], PSCD=["4.2", "5.5"], PSCA=["5.2", "1.33"])
    out = normalise(raw, "2024-25")
    assert out["odds_close_src"].iloc[0] == "PSCH+PSCD+PSCA"
    assert out["odds_close_h"].iloc[0] == 1.6


def test_pinnacle_renamed_columns_resolve() -> None:
    """2025-26 onwards ships PP* instead of PS* — the regression that broke the benchmark."""
    raw = _raw(PPCH=["1.6", "8.5"], PPCD=["4.2", "5.5"], PPCA=["5.2", "1.33"])
    out = normalise(raw, "2025-26")
    assert out["odds_close_src"].iloc[0] == "PPCH+PPCD+PPCA"
    assert out["odds_close_a"].iloc[1] == 1.33


def test_falls_back_to_bet365_when_pinnacle_is_sparse() -> None:
    raw = _raw(
        PSCD=["4.2", "5.5"],  # incomplete family
        B365CH=["1.7", "8.0"], B365CD=["4.0", "5.5"], B365CA=["5.0", "1.36"],
    )
    out = normalise(raw, "2024-25")
    assert out["odds_close_src"].iloc[0] == "B365CH+B365CD+B365CA"


def test_rows_without_a_score_are_dropped() -> None:
    raw = _raw(FTHG=["1", ""], FTAG=["0", ""])
    out = normalise(raw, "2024-25")
    assert len(out) == 1


def test_match_ids_are_unique_per_season() -> None:
    raw = pd.concat([_raw()] * 3, ignore_index=True)
    out = normalise(raw, "2024-25")
    assert out["match_id"].is_unique
