"""Shared fixtures: a small synthetic league so tests never touch the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

TEAMS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]


def make_matches(
    n_seasons: int = 3,
    teams: list[str] | None = None,
    seed: int = 7,
    goals: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Generate a deterministic double round-robin over synthetic seasons."""
    teams = teams or TEAMS
    rng = np.random.default_rng(seed)
    strength = {t: rng.normal(0.0, 0.35) for t in teams}
    if goals:
        strength.update(goals)

    rows = []
    match_id = 0
    for s in range(n_seasons):
        start = pd.Timestamp(f"{2020 + s}-08-10")
        pairs = [(h, a) for h in teams for a in teams if h != a]
        rng.shuffle(pairs)
        for round_no, (home, away) in enumerate(pairs):
            date = start + pd.Timedelta(days=7 * (round_no // (len(teams) // 2)) + round_no % 3)
            lam = np.exp(0.25 + strength[home] - strength[away])
            mu = np.exp(strength[away] - strength[home])
            hg, ag = rng.poisson(lam), rng.poisson(mu)
            rows.append({
                "match_id": f"s{s}-{match_id:03d}",
                "season": f"{2020 + s}-{(2021 + s) % 100:02d}",
                "date": date,
                "home": home,
                "away": away,
                "fthg": int(hg),
                "ftag": int(ag),
                "ftr": "H" if hg > ag else ("A" if hg < ag else "D"),
                "hs": int(rng.poisson(11)),
                "as_": int(rng.poisson(10)),
                "hst": int(min(int(hg) + rng.poisson(3), 12)),
                "ast": int(min(int(ag) + rng.poisson(3), 12)),
                "odds_open_h": 2.0, "odds_open_d": 3.4, "odds_open_a": 3.8,
                "odds_close_h": 1.95, "odds_close_d": 3.5, "odds_close_a": 3.9,
            })
            match_id += 1
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


@pytest.fixture
def matches() -> pd.DataFrame:
    return make_matches()


@pytest.fixture
def features(matches: pd.DataFrame) -> pd.DataFrame:
    from plpred.features.elo import add_elo
    from plpred.features.rolling import add_form_features

    return add_form_features(add_elo(matches))
