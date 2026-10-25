"""Elo ratings with home advantage, season regression and goal-margin weighting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import (
    ELO_HOME_ADVANTAGE,
    ELO_K,
    ELO_NEW_TEAM,
    ELO_SEASON_REGRESSION,
    ELO_START,
)


def expected_score(rating_a: float, rating_b: float) -> float:
    """Standard Elo expectation for side A (0..1)."""
    return 1.0 / (1.0 + 10.0 ** (-(rating_a - rating_b) / 400.0))


def _margin_factor(goal_diff: int) -> float:
    """Bigger wins move the rating more (a light version of the ClubElo rule)."""
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return 1.75


def add_elo(matches: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``matches`` with pre-match Elo columns.

    The values in ``elo_h``/``elo_a`` are the ratings *before kick-off*, which is
    what a forecast is allowed to see.  Processing is strictly chronological.
    """
    df = matches.sort_values(["date", "home"]).reset_index(drop=True).copy()
    ratings: dict[str, float] = {}
    current_season: str | None = None

    elo_h = np.empty(len(df))
    elo_a = np.empty(len(df))

    for i, row in enumerate(df.itertuples(index=False)):
        if row.season != current_season:
            if current_season is not None:
                for team in ratings:
                    ratings[team] = ELO_START + (ratings[team] - ELO_START) * (1.0 - ELO_SEASON_REGRESSION)
            current_season = row.season

        rh = ratings.get(row.home, ELO_NEW_TEAM)
        ra = ratings.get(row.away, ELO_NEW_TEAM)
        elo_h[i], elo_a[i] = rh, ra

        exp_h = expected_score(rh + ELO_HOME_ADVANTAGE, ra)
        gd = int(row.fthg) - int(row.ftag)
        score_h = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        delta = ELO_K * _margin_factor(abs(gd)) * (score_h - exp_h)

        ratings[row.home] = rh + delta
        ratings[row.away] = ra - delta

    df["elo_h"] = elo_h
    df["elo_a"] = elo_a
    df["elo_diff"] = df["elo_h"] - df["elo_a"]
    return df


def elo_table(matches_with_elo: pd.DataFrame) -> pd.DataFrame:
    """Latest rating per club, useful for plots and the dashboard."""
    rows = []
    for row in matches_with_elo.itertuples(index=False):
        rows.append((row.date, row.home, row.elo_h))
        rows.append((row.date, row.away, row.elo_a))
    long = pd.DataFrame(rows, columns=["date", "team", "elo"])
    long = long.sort_values("date").groupby("team", as_index=False).tail(1)
    return long.sort_values("elo", ascending=False).reset_index(drop=True)
