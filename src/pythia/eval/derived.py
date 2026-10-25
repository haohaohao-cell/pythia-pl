"""Derived markets: over/under 2.5 goals.

This is the one place where the structural model has a natural advantage: a
Dixon-Coles fit produces the whole scoreline distribution, so P(over 2.5) falls
out of it for free, while a classifier over 1X2 labels cannot price a goals
market at all.

football-data only publishes over/under prices from 2019-20, so this is a
seven-season study rather than nine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import season_boundaries
from ..models.dixon_coles import DixonColes

LINE = 2.5
FIRST_SEASON = "2019-20"


def devig_two_way(over_odds: np.ndarray, under_odds: np.ndarray) -> np.ndarray:
    """Proportional de-vig for a two-outcome market."""
    over_odds = np.asarray(over_odds, dtype=float)
    under_odds = np.asarray(under_odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_over = 1.0 / over_odds
        raw_under = 1.0 / under_odds
        total = raw_over + raw_under
        return np.where(np.isfinite(total) & (total > 0), raw_over / total, np.nan)


def over_probability(dc: DixonColes, home: str, away: str, line: float = LINE) -> float:
    """P(total goals > line) from the fitted scoreline distribution."""
    matrix = dc.score_matrix(home, away)
    max_under = int(np.floor(line))
    under = 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i + j <= max_under:
                under += matrix[i, j]
    return float(1.0 - under)


def binary_log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def evaluate_over_under(features: pd.DataFrame, seasons: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Walk-forward evaluation of the 2.5-goal market."""
    df = features.sort_values(["date", "home"]).reset_index(drop=True)
    usable = df.loc[df["ou_over"].notna() & df["ou_under"].notna()]
    seasons = seasons or [s for s in sorted(usable["season"].unique()) if s >= FIRST_SEASON]

    rows = []
    model_all, market_all, base_all, y_all = [], [], [], []

    for season in seasons:
        start, end = season_boundaries(season)
        train = df.loc[df["date"] < start]
        test = usable.loc[(usable["season"] == season) & (usable["date"] >= start)]
        if test.empty or train.empty:
            continue

        dc = DixonColes().fit(train)
        p_model = np.array([
            over_probability(dc, h, a) for h, a in zip(test["home"], test["away"])
        ])
        p_market = devig_two_way(test["ou_over"].to_numpy(), test["ou_under"].to_numpy())
        over_rate = float((train["fthg"].astype(float) + train["ftag"].astype(float) > LINE).mean())
        p_base = np.full(len(test), over_rate)
        y = ((test["fthg"].astype(float) + test["ftag"].astype(float)) > LINE).to_numpy(dtype=float)

        keep = np.isfinite(p_market)
        p_model, p_market, p_base, y = p_model[keep], p_market[keep], p_base[keep], y[keep]
        if not len(y):
            continue

        rows.append({
            "season": season,
            "n": int(len(y)),
            "over_rate": float(y.mean()),
            "model": binary_log_loss(p_model, y),
            "market": binary_log_loss(p_market, y),
            "base": binary_log_loss(p_base, y),
            "delta": binary_log_loss(p_model, y) - binary_log_loss(p_market, y),
        })
        model_all.append(p_model)
        market_all.append(p_market)
        base_all.append(p_base)
        y_all.append(y)

    table = pd.DataFrame(rows)
    if not rows:
        return table, {}

    y_cat = np.concatenate(y_all)
    p_m = np.concatenate(model_all)
    p_k = np.concatenate(market_all)
    p_b = np.concatenate(base_all)

    rng = np.random.default_rng(42)
    deltas = np.empty(2000)
    for i in range(len(deltas)):
        idx = rng.integers(0, len(y_cat), len(y_cat))
        deltas[i] = binary_log_loss(p_m[idx], y_cat[idx]) - binary_log_loss(p_k[idx], y_cat[idx])

    summary = {
        "n": int(len(y_cat)),
        "over_rate": float(y_cat.mean()),
        "model": binary_log_loss(p_m, y_cat),
        "market": binary_log_loss(p_k, y_cat),
        "base": binary_log_loss(p_b, y_cat),
        "delta": float(binary_log_loss(p_m, y_cat) - binary_log_loss(p_k, y_cat)),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_better": float(np.mean(deltas < 0)),
    }
    return table, summary
