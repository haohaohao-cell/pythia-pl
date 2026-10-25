"""Pre-match form features built from each team's own match history.

Every statistic here is lagged by one match: a fixture row only ever sees the
results of games that finished earlier.  This is the invariant that
``tests/test_leakage.py`` enforces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FORM_EWM_HALFLIFE, FORM_WINDOW

#: Priors used when a team has no usable history yet (promoted sides, matchday 1).
PRIOR_PTS = 1.40
PRIOR_GF = 1.45
PRIOR_GA = 1.45
PRIOR_SOT = 4.50
PRIOR_SHOTS = 10.0
PRIOR_XG = 1.40
PRIOR_XG_AG = 1.40


def team_long_frame(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, match) with the team's own perspective."""
    home = pd.DataFrame(
        {
            "match_id": matches["match_id"], "date": matches["date"], "season": matches["season"],
            "team": matches["home"], "opponent": matches["away"], "is_home": True,
            "gf": matches["fthg"], "ga": matches["ftag"],
            "sot_for": matches["hst"], "sot_ag": matches["ast"],
            "shots_for": matches["hs"], "shots_ag": matches["as_"],
            "xg_for": matches.get("hxg", np.nan), "xg_ag": matches.get("axg", np.nan),
        }
    )
    away = pd.DataFrame(
        {
            "match_id": matches["match_id"], "date": matches["date"], "season": matches["season"],
            "team": matches["away"], "opponent": matches["home"], "is_home": False,
            "gf": matches["ftag"], "ga": matches["fthg"],
            "sot_for": matches["ast"], "sot_ag": matches["hst"],
            "shots_for": matches["as_"], "shots_ag": matches["hs"],
            "xg_for": matches.get("axg", np.nan), "xg_ag": matches.get("hxg", np.nan),
        }
    )
    long = pd.concat([home, away], ignore_index=True)
    long["date"] = pd.to_datetime(long["date"])
    long = long.sort_values(["team", "date", "is_home"]).reset_index(drop=True)
    long["points"] = np.where(long.gf > long.ga, 3, np.where(long.gf == long.ga, 1, 0))
    return long


def _roll(series: pd.Series, window: int) -> pd.Series:
    """Mean of the previous ``window`` values (current match excluded)."""
    return series.shift(1).rolling(window, min_periods=1).mean()


def add_form_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Attach per-team rolling form to each fixture and return a model matrix.

    The result has one row per match with ``home_*``/``away_*`` columns plus
    differences, and is aligned to ``matches`` by index order (sorted by date).
    """
    long = team_long_frame(matches)
    w = FORM_WINDOW

    long["pts_form"] = _roll(long["points"], w)
    long["gf_form"] = _roll(long["gf"], w)
    long["ga_form"] = _roll(long["ga"], w)
    long["sot_form"] = _roll(long["sot_for"], w)
    long["sot_ag_form"] = _roll(long["sot_ag"], w)
    long["shots_form"] = _roll(long["shots_for"], w)
    long["goal_diff_form"] = long["gf_form"] - long["ga_form"]

    # xG: how many chances a side created and conceded, lagged like everything else.
    has_xg = long["xg_for"].notna().any()
    if has_xg:
        long["xg_form"] = _roll(long["xg_for"], w)
        long["xg_ag_form"] = _roll(long["xg_ag"], w)
        long["xg_diff_form"] = long["xg_form"] - long["xg_ag_form"]
        long["finishing_form"] = _roll(long["gf"] - long["xg_for"], w)

    long["pts_ewm"] = long["points"].shift(1).ewm(halflife=FORM_EWM_HALFLIFE, min_periods=1).mean()

    long["rest_days"] = long.groupby("team")["date"].diff().dt.days

    long["match_no"] = long.groupby(["team", "season"]).cumcount() + 1

    # A club is "new to the league" when its previous appearance was not last season.
    prev_season: dict[str, int] = {}
    promoted = np.zeros(len(long), dtype=float)
    for i, row in enumerate(long.itertuples(index=False)):
        year = int(str(row.season)[:4])
        last = prev_season.get(row.team)
        if last is None or year - last > 1:
            promoted[i] = 1.0
        prev_season[row.team] = year
    long["newly_promoted"] = promoted

    def side(frame: pd.DataFrame, prefix: str, is_home: bool) -> pd.DataFrame:
        cols = [
            "match_id", "pts_form", "gf_form", "ga_form", "sot_form", "sot_ag_form",
            "shots_form", "goal_diff_form", "pts_ewm", "rest_days", "match_no", "newly_promoted",
        ]
        cols += [c for c in ("xg_form", "xg_ag_form", "xg_diff_form", "finishing_form") if c in frame.columns]
        part = frame.loc[frame["is_home"] == is_home, cols]
        return part.rename(columns={c: f"{prefix}_{c}" for c in part.columns if c != "match_id"})

    home = side(long, "home", True)
    away = side(long, "away", False)
    feats = home.merge(away, on="match_id", how="inner")

    drop = [c for c in ("home_raw", "away_raw") if c in matches.columns]
    out = matches.drop(columns=drop).copy()
    out = out.merge(feats, on="match_id", how="left")

    fill = {
        "pts_form": PRIOR_PTS, "gf_form": PRIOR_GF, "ga_form": PRIOR_GA,
        "goal_diff_form": PRIOR_GF - PRIOR_GA, "sot_form": PRIOR_SOT,
        "sot_ag_form": PRIOR_SOT, "shots_form": PRIOR_SHOTS, "pts_ewm": PRIOR_PTS,
    }
    for col in list(fill):
        for prefix in ("home", "away"):
            out[f"{prefix}_{col}"] = out[f"{prefix}_{col}"].fillna(fill[col])
    for prefix in ("home", "away"):
        out[f"{prefix}_rest_days"] = out[f"{prefix}_rest_days"].fillna(7.0)
        out[f"{prefix}_match_no"] = out[f"{prefix}_match_no"].fillna(1.0)
        out[f"{prefix}_newly_promoted"] = out[f"{prefix}_newly_promoted"].fillna(1.0)
        for col in ("xg_form", "xg_ag_form", "xg_diff_form", "finishing_form"):
            name = f"{prefix}_{col}"
            if name in out.columns:
                prior = PRIOR_XG_AG if col in ("xg_ag_form",) else (0.0 if col in ("xg_diff_form", "finishing_form") else PRIOR_XG)
                out[name] = out[name].fillna(prior)

    # Differences carry most of the signal for tree models.
    diff_cols = ["pts_form", "gf_form", "ga_form", "sot_form", "sot_ag_form", "goal_diff_form", "pts_ewm", "rest_days"]
    diff_cols += [c for c in ("xg_form", "xg_ag_form", "xg_diff_form", "finishing_form")
                  if f"home_{c}" in out.columns]
    for col in diff_cols:
        out[f"diff_{col}"] = out[f"home_{col}"] - out[f"away_{col}"]

    out["season_progress"] = out["home_match_no"] / 38.0
    return out


FEATURE_COLUMNS = [
    "elo_diff", "elo_h", "elo_a",
    "diff_pts_form", "diff_gf_form", "diff_ga_form", "diff_goal_diff_form",
    "diff_sot_form", "diff_sot_ag_form", "diff_pts_ewm", "diff_rest_days",
    "home_sot_form", "away_sot_form", "home_sot_ag_form", "away_sot_ag_form",
    "home_pts_ewm", "away_pts_ewm", "season_progress",
    "home_newly_promoted", "away_newly_promoted", "home_match_no", "away_match_no",
]

#: Everything above plus the xG block, used only when Understat data is present.
XG_FEATURE_COLUMNS = [
    "home_xg_form", "away_xg_form", "home_xg_ag_form", "away_xg_ag_form",
    "diff_xg_form", "diff_xg_ag_form", "diff_xg_diff_form", "diff_finishing_form",
]

FEATURE_COLUMNS_XG = FEATURE_COLUMNS + XG_FEATURE_COLUMNS
