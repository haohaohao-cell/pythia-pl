"""Build a model row for a fixture that has not been played yet.

The feature table is indexed by played matches, so forecasting an upcoming game
means asking "what did each club look like after its most recent match?" and
reassembling a row in exactly the layout the models expect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .teams import en_name

SIDE_COLUMNS = (
    "pts_form", "gf_form", "ga_form", "sot_form", "sot_ag_form", "shots_form",
    "goal_diff_form", "pts_ewm", "rest_days", "match_no", "newly_promoted",
)
DIFF_COLUMNS = (
    "pts_form", "gf_form", "ga_form", "sot_form", "sot_ag_form",
    "goal_diff_form", "pts_ewm", "rest_days",
)


def team_state(features: pd.DataFrame, team: str) -> dict[str, float]:
    """Latest known pre-match state for a club, from whichever side it last played."""
    block = features.loc[(features["home"] == team) | (features["away"] == team)]
    if block.empty:
        raise KeyError(f"no matches for {team!r}")
    last = block.sort_values("date").iloc[-1]
    is_home = last["home"] == team
    prefix = "home" if is_home else "away"

    state = {
        col[len(prefix) + 1:]: float(last[col])
        for col in features.columns
        if col.startswith(f"{prefix}_") and col[len(prefix) + 1:] in SIDE_COLUMNS
    }
    state["elo"] = float(last["elo_h"] if is_home else last["elo_a"])
    state["date"] = last["date"]
    state["season"] = last["season"]
    return state


def fixture_row(features: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """One-row frame that any model in this project can score."""
    home_state = team_state(features, home)
    away_state = team_state(features, away)

    row: dict[str, float] = {
        "home": home,
        "away": away,
        "season": max(home_state["season"], away_state["season"]),
        "date": features["date"].max(),
        "ftr": "D",
        "fthg": 0,
        "ftag": 0,
    }
    for key, value in home_state.items():
        if key in SIDE_COLUMNS:
            row[f"home_{key}"] = value
    for key, value in away_state.items():
        if key in SIDE_COLUMNS:
            row[f"away_{key}"] = value

    row["elo_h"] = home_state["elo"]
    row["elo_a"] = away_state["elo"]
    row["elo_diff"] = row["elo_h"] - row["elo_a"]

    for col in DIFF_COLUMNS:
        row[f"diff_{col}"] = row.get(f"home_{col}", np.nan) - row.get(f"away_{col}", np.nan)

    row["season_progress"] = row.get("home_match_no", 1.0) / 38.0
    return pd.DataFrame([row])


def describe(features: pd.DataFrame, home: str, away: str) -> str:
    home_state = team_state(features, home)
    away_state = team_state(features, away)
    return (
        f"{en_name(home)} (Elo {home_state['elo']:.0f}, last played {home_state['date']:%Y-%m-%d}) "
        f"vs {en_name(away)} (Elo {away_state['elo']:.0f}, last played {away_state['date']:%Y-%m-%d})"
    )
