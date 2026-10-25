"""FPL availability: parsing, weighting and snapshot storage."""

from __future__ import annotations

import pandas as pd
import pytest

from pythia.ingest import fpl


def _bootstrap() -> dict:
    return {
        "events": [{"id": 7, "is_current": False}, {"id": 8, "is_current": True}],
        "teams": [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Man Utd"}],
        "elements": [
            {"web_name": "Fully fit", "team": 1, "element_type": 4, "status": "a",
             "chance_of_playing_next_round": None, "minutes": 900, "now_cost": 55, "news": ""},
            {"web_name": "Out", "team": 1, "element_type": 3, "status": "i",
             "chance_of_playing_next_round": 0, "minutes": 100, "now_cost": 50, "news": "Knee"},
            {"web_name": "Maybe", "team": 2, "element_type": 3, "status": "d",
             "chance_of_playing_next_round": 75, "minutes": 400, "now_cost": 60, "news": "Illness"},
            {"web_name": "Unlikely", "team": 2, "element_type": 2, "status": "d",
             "chance_of_playing_next_round": 50, "minutes": 200, "now_cost": 45, "news": ""},
        ],
    }


def test_team_names_are_canonical() -> None:
    players = fpl.players_frame(_bootstrap())
    assert set(players["team"]) == {"arsenal", "man_united"}


def test_current_gameweek_prefers_is_current() -> None:
    assert fpl.current_gameweek(_bootstrap()) == 8


def test_availability_is_minutes_weighted() -> None:
    table = fpl.team_availability(_bootstrap()).set_index("team")
    # Arsenal: 900 of 1000 minutes still selectable.
    assert table.loc["arsenal", "avail_score"] == pytest.approx(0.9)
    # Man Utd: the 75% doubtful defender counts, the 50% one does not.
    assert table.loc["man_united", "avail_score"] == pytest.approx(400 / 600)


def test_status_counts_are_reported() -> None:
    table = fpl.team_availability(_bootstrap()).set_index("team")
    assert table.loc["arsenal", "n_injured"] == 1
    assert table.loc["man_united", "n_doubtful"] == 2
    assert table.loc["arsenal", "n_suspended"] == 0


def test_players_frame_marks_status_labels() -> None:
    players = fpl.players_frame(_bootstrap())
    labels = dict(zip(players["player"], players["status_label"]))
    assert labels["Out"] == "injured"
    assert labels["Maybe"] == "doubtful"
    assert labels["Fully fit"] == "available"


def test_snapshot_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fpl, "SNAPSHOT_DIR", tmp_path)
    path = fpl.save_snapshot(_bootstrap(), when=pd.Timestamp("2026-10-25"))
    assert path.name == "availability_2026-10-25.parquet"

    history = fpl.load_history()
    assert len(history) == 2
    assert set(history["gameweek"]) == {8}
    assert history["date"].nunique() == 1


def test_empty_history_is_safe(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fpl, "SNAPSHOT_DIR", tmp_path / "missing")
    assert fpl.load_history().empty
