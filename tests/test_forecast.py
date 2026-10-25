"""The weekly artefacts: fixture parsing and the round table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pythia.forecast import round_markdown
from pythia.ingest.fixtures_feed import parse_fixtures


def _raw() -> pd.DataFrame:
    return pd.DataFrame({
        "Div": ["E0", "E0", "B1"],
        "Date": ["24/10/2026", "25/10/2026", "24/10/2026"],
        "Time": ["12:30", "", "19:45"],
        "HomeTeam": ["Aston Villa", "Man United", "Kortrijk"],
        "AwayTeam": ["Man City", "Bournemouth", "Mechelen"],
        "PPH": ["4.75", "1.9", "2.55"], "PPD": ["4.33", "3.5", "3.4"], "PPA": ["1.57", "4.2", "2.45"],
        "B365H": ["5", "2", "2.6"], "B365D": ["4.33", "3.5", "3.4"], "B365A": ["1.62", "4", "2.5"],
        "B365>2.5": ["1.53", "1.8", "1.9"], "B365<2.5": ["2.5", "2.0", "1.9"],
    })


def test_only_premier_league_rows_are_kept() -> None:
    out = parse_fixtures(_raw())
    assert len(out) == 2
    assert "kortrijk" not in set(out["home"])


def test_kickoff_combines_date_and_time() -> None:
    out = parse_fixtures(_raw())
    kickoff = out["kickoff"].iloc[0]
    assert kickoff.hour == 12 and kickoff.minute == 30
    assert kickoff.date() == pd.Timestamp("2026-10-24").date()


def test_kickoff_is_uk_local_time() -> None:
    """BST on 24 Oct 2026, GMT from the 25th — the DST switch that broke the clock check."""
    out = parse_fixtures(_raw())
    bst = out["kickoff"].iloc[0]
    gmt = out["kickoff"].iloc[1]
    assert bst.utcoffset() == pd.Timedelta(hours=1)
    assert gmt.utcoffset() == pd.Timedelta(0)


def test_kickoff_compares_correctly_from_another_timezone() -> None:
    """The same instant must be judged 'past' regardless of the machine's zone."""
    out = parse_fixtures(_raw())
    kickoff = out["kickoff"].iloc[0]
    later = kickoff.tz_convert("Asia/Shanghai") + pd.Timedelta(minutes=5)
    assert kickoff < later
    assert (kickoff + pd.Timedelta(minutes=5)) > kickoff


def test_missing_time_defaults_to_afternoon() -> None:
    out = parse_fixtures(_raw())
    assert out["kickoff"].iloc[1].hour == 15


def test_opening_prices_prefer_pinnacle() -> None:
    out = parse_fixtures(_raw())
    assert out["odds_src"].iloc[0] == "PPH+PPD+PPA"
    assert out["odds_h_h"].iloc[0] == 4.75


def test_team_names_are_canonical() -> None:
    out = parse_fixtures(_raw())
    assert list(out["home"]) == ["aston_villa", "man_united"]
    assert list(out["away"]) == ["man_city", "bournemouth"]


def _forecast(status: str = "upcoming", hit=None) -> pd.DataFrame:
    return pd.DataFrame([{
        "kickoff": pd.Timestamp("2026-10-24 12:30"),
        "home": "aston_villa", "away": "man_city",
        "p_home": 0.21, "p_draw": 0.26, "p_away": 0.53,
        "mkt_home": 0.20, "mkt_draw": 0.21, "mkt_away": 0.59,
        "pick": "A", "pick_prob": 0.53, "status": status,
        "result": "A" if status == "played" else "",
        "score": "1-2" if status == "played" else "",
        "hit": hit,
    }])


def test_round_markdown_omits_result_column_when_nothing_is_played() -> None:
    table = round_markdown(_forecast())
    assert "Result" not in table and "结果" not in table
    assert "Aston Villa v Man City" in table


def test_round_markdown_shows_settled_rows() -> None:
    table = round_markdown(_forecast("played", hit=True))
    assert "1-2" in table
    assert "✅" in table


def test_round_markdown_is_bilingual() -> None:
    assert "Kick-off" in round_markdown(_forecast(), lang="en")
    assert "开球" in round_markdown(_forecast(), lang="zh")


def test_empty_forecast_does_not_crash() -> None:
    assert round_markdown(pd.DataFrame())


def test_probabilities_are_finite() -> None:
    row = _forecast().iloc[0]
    assert np.isfinite([row.p_home, row.p_draw, row.p_away]).all()
