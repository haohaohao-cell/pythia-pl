"""Upcoming round: FPL calendar plus opportunistic football-data prices."""

from __future__ import annotations

import pandas as pd
import pytest

from pythia.ingest import fpl, upcoming
from pythia.ingest.fixtures_feed import UK_TZ


def _fixtures() -> pd.DataFrame:
    return pd.DataFrame({
        "event": [12, 12, 13],
        "kickoff": pd.to_datetime(
            ["2026-11-27T20:00:00Z", "2026-11-28T15:00:00Z", "2026-12-01T20:00:00Z"], utc=True
        ).tz_convert(UK_TZ),
        "home": ["nottm_forest", "leeds", "chelsea"],
        "away": ["chelsea", "coventry", "fulham"],
        "finished": [False, False, False],
        "started": [False, False, False],
    })


def test_next_round_takes_the_earliest_pending_gameweek() -> None:
    block = fpl.next_round_fixtures(_fixtures())
    assert len(block) == 2
    assert set(block["event"]) == {12}


def test_finished_matches_are_excluded_from_the_next_round() -> None:
    frame = _fixtures()
    frame.loc[0, "finished"] = True
    block = fpl.next_round_fixtures(frame)
    assert len(block) == 1
    assert block["home"].iat[0] == "leeds"


def test_no_pending_matches_returns_empty() -> None:
    frame = _fixtures()
    frame["finished"] = True
    assert fpl.next_round_fixtures(frame).empty


def test_prices_attach_when_dates_agree(monkeypatch) -> None:
    monkeypatch.setattr(fpl, "fetch_fixtures", lambda *a, **k: _fixtures())
    monkeypatch.setattr(upcoming, "load_fixtures", lambda *a, **k: pd.DataFrame({
        "home": ["nottm_forest", "leeds"],
        "away": ["chelsea", "coventry"],
        "kickoff": pd.to_datetime(
            ["2026-11-27T20:00:00Z", "2026-11-28T15:00:00Z"], utc=True
        ).tz_convert(UK_TZ),
        "odds_h_h": [2.9, 2.1], "odds_h_d": [3.4, 3.5], "odds_h_a": [2.4, 3.3],
        "odds_src": ["PPH+PPD+PPA", "PPH+PPD+PPA"],
    }))

    block = upcoming.load_upcoming()
    assert len(block) == 2
    assert block["odds_h_h"].tolist() == [2.9, 2.1]


def test_stale_prices_are_dropped(monkeypatch) -> None:
    monkeypatch.setattr(fpl, "fetch_fixtures", lambda *a, **k: _fixtures())
    monkeypatch.setattr(upcoming, "load_fixtures", lambda *a, **k: pd.DataFrame({
        "home": ["nottm_forest"], "away": ["chelsea"],
        "kickoff": pd.to_datetime(["2026-10-24T15:00:00Z"], utc=True).tz_convert(UK_TZ),
        "odds_h_h": [3.0], "odds_h_d": [3.3], "odds_h_a": [2.25],
        "odds_src": ["PPH+PPD+PPA"],
    }))

    block = upcoming.load_upcoming()
    assert block["odds_h_h"].isna().all()
    assert (block["odds_src"] == "").all()


def test_falls_back_to_the_flat_feed_when_fpl_fails(monkeypatch) -> None:
    def boom(*a, **k):
        raise RuntimeError("fpl down")

    monkeypatch.setattr(fpl, "fetch_fixtures", boom)
    monkeypatch.setattr(upcoming, "load_fixtures", lambda *a, **k: pd.DataFrame({
        "home": ["arsenal"], "away": ["hull"],
        "kickoff": pd.to_datetime(["2026-11-28T15:00:00Z"], utc=True).tz_convert(UK_TZ),
        "odds_h_h": [1.4], "odds_h_d": [4.5], "odds_h_a": [7.0], "odds_src": ["PPH+PPD+PPA"],
    }))

    block = upcoming.load_upcoming()
    assert len(block) == 1
    assert block["home"].iat[0] == "arsenal"


def test_no_sources_returns_empty_frame(monkeypatch) -> None:
    def boom(*a, **k):
        raise RuntimeError("fpl down")

    monkeypatch.setattr(fpl, "fetch_fixtures", boom)
    monkeypatch.setattr(upcoming, "load_fixtures", lambda *a, **k: pd.DataFrame())

    block = upcoming.load_upcoming()
    assert block.empty
    assert list(block.columns) == upcoming.OUT_COLUMNS


def test_kickoff_is_uk_local(monkeypatch) -> None:
    block = fpl.next_round_fixtures(_fixtures())
    assert block["kickoff"].dt.tz is not None
    assert str(block["kickoff"].dt.tz) == "Europe/London"
    # 20:00 UTC on 27 Nov is 20:00 GMT in London (winter).
    assert block["kickoff"].iat[0].hour == 20


@pytest.mark.parametrize("drift_days", [0, 3, 6])
def test_prices_within_a_week_are_kept(monkeypatch, drift_days) -> None:
    shifted = pd.Timestamp("2026-11-27T20:00:00Z") + pd.Timedelta(days=drift_days)
    monkeypatch.setattr(fpl, "fetch_fixtures", lambda *a, **k: _fixtures())
    monkeypatch.setattr(upcoming, "load_fixtures", lambda *a, **k: pd.DataFrame({
        "home": ["nottm_forest"], "away": ["chelsea"],
        "kickoff": pd.DatetimeIndex([shifted]).tz_convert(UK_TZ),
        "odds_h_h": [2.9], "odds_h_d": [3.4], "odds_h_a": [2.4], "odds_src": ["PPH+PPD+PPA"],
    }))
    block = upcoming.load_upcoming().set_index(["home", "away"])
    assert pd.notna(block.loc[("nottm_forest", "chelsea"), "odds_h_h"])
