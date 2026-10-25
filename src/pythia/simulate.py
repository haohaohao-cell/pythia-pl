"""Monte Carlo simulation of the rest of the season.

The remaining fixture list is recovered by subtracting the pairs already played
from the full double round-robin, which works because every club plays every
other club home and away exactly once.  Postponed games therefore stay in the
schedule automatically.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .config import RANDOM_STATE


def full_round_robin(teams: list[str]) -> pd.DataFrame:
    pairs = [(h, a) for h, a in itertools.permutations(sorted(teams), 2)]
    return pd.DataFrame(pairs, columns=["home", "away"])


def remaining_fixtures(matches: pd.DataFrame, season: str) -> pd.DataFrame:
    """Double round-robin schedule minus the fixtures already played."""
    block = matches.loc[matches["season"] == season]
    teams = sorted(set(block["home"]) | set(block["away"]))
    played = set(zip(block["home"], block["away"]))
    schedule = full_round_robin(teams)
    mask = [pair not in played for pair in zip(schedule["home"], schedule["away"])]
    return schedule.loc[mask].reset_index(drop=True)


def current_table(matches: pd.DataFrame, season: str) -> pd.DataFrame:
    block = matches.loc[matches["season"] == season]
    rows: dict[str, dict[str, int]] = {}
    for row in block.itertuples(index=False):
        for team, gf, ga in ((row.home, row.fthg, row.ftag), (row.away, row.ftag, row.fthg)):
            entry = rows.setdefault(str(team), {"played": 0, "points": 0, "gf": 0, "ga": 0})
            entry["played"] += 1
            entry["gf"] += int(gf)
            entry["ga"] += int(ga)
            entry["points"] += 3 if gf > ga else (1 if gf == ga else 0)
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "team"
    table["gd"] = table["gf"] - table["ga"]
    return table.reset_index().sort_values(["points", "gd"], ascending=False).reset_index(drop=True)


def _sample_scorelines(dc, fixtures: pd.DataFrame, n_sims: int, rng: np.random.Generator):
    """Draw (home_goals, away_goals) for every fixture and simulation."""
    home_goals = np.empty((len(fixtures), n_sims), dtype=np.int8)
    away_goals = np.empty((len(fixtures), n_sims), dtype=np.int8)
    size = dc.max_goals

    for i, row in enumerate(fixtures.itertuples(index=False)):
        matrix = dc.score_matrix(row.home, row.away)
        flat = matrix.ravel()
        cdf = np.cumsum(flat)
        cdf[-1] = 1.0
        draws = rng.random(n_sims)
        idx = np.searchsorted(cdf, draws, side="right")
        home_goals[i] = idx // size
        away_goals[i] = idx % size
    return home_goals, away_goals


def simulate_remaining(
    matches: pd.DataFrame,
    dc,
    season: str,
    *,
    n_sims: int = 20_000,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (summary, point distribution sample)."""
    fixtures = remaining_fixtures(matches, season)
    table = current_table(matches, season)
    teams = table["team"].tolist()
    team_index = {t: i for i, t in enumerate(teams)}

    rng = np.random.default_rng(seed)
    points = np.zeros((n_sims, len(teams)), dtype=np.int16)
    gd = np.zeros((n_sims, len(teams)), dtype=np.int16)

    if not fixtures.empty:
        home_goals, away_goals = _sample_scorelines(dc, fixtures, n_sims, rng)
        for i, row in enumerate(fixtures.itertuples(index=False)):
            h, a = team_index[row.home], team_index[row.away]
            gh, ga = home_goals[i], away_goals[i]
            home_pts = np.where(gh > ga, 3, np.where(gh == ga, 1, 0)).astype(np.int16)
            away_pts = np.where(gh < ga, 3, np.where(gh == ga, 1, 0)).astype(np.int16)
            points[:, h] += home_pts
            points[:, a] += away_pts
            gd[:, h] += (gh - ga).astype(np.int16)
            gd[:, a] += (ga - gh).astype(np.int16)

    base_points = table.set_index("team")["points"]
    base_gd = table.set_index("team")["gd"]
    final_points = points + base_points.to_numpy()[None, :]
    final_gd = gd + base_gd.to_numpy()[None, :]

    # Rank within each simulation; random jitter breaks ties consistently.
    jitter = rng.random(final_points.shape) * 1e-3
    order = np.argsort(-(final_points + jitter), axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, len(teams) + 1)[None, :].repeat(n_sims, 0), axis=1)
    title = (ranks == 1).mean(axis=0)
    top4 = (ranks <= 4).mean(axis=0)
    relegation = (ranks >= len(teams) - 2).mean(axis=0)

    summary = pd.DataFrame({
        "team": teams,
        "played": table["played"].to_numpy(),
        "points_now": base_points.to_numpy(),
        "gd_now": base_gd.to_numpy(),
        "projected": final_points.mean(axis=0),
        "p10": np.percentile(final_points, 10, axis=0),
        "p90": np.percentile(final_points, 90, axis=0),
        "p_title": title,
        "p_top4": top4,
        "p_relegation": relegation,
    }).sort_values("projected", ascending=False).reset_index(drop=True)

    keep = min(n_sims, 2000)
    sample = pd.DataFrame({
        "team": np.repeat(teams, keep),
        "points": final_points[:keep].T.ravel(),
    })
    return summary, sample
