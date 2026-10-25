"""Appendix analysis: does the model find positive expected value?

This is deliberately *not* a betting strategy.  It exists to answer one research
question — when the model disagrees with the pre-match price by a wide margin,
is it right more often than the price implies?  Flat stakes, no staking plan, no
bankroll management, no compounding.  Treat it as a market-efficiency probe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models.base import outcome_index


def value_bets(
    preds: pd.DataFrame,
    features: pd.DataFrame,
    *,
    model: str = "blend",
    min_edge: float = 0.05,
) -> pd.DataFrame:
    """One row per match where the model's edge over the opening price clears a threshold."""
    cols = ["match_id", "odds_open_h", "odds_open_d", "odds_open_a",
            "odds_close_h", "odds_close_d", "odds_close_a"]
    merged = preds.loc[preds["model"] == model].merge(features[cols], on="match_id", how="inner")
    merged = merged.dropna(subset=["odds_open_h", "odds_open_d", "odds_open_a"])

    probs = merged[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    open_odds = merged[["odds_open_h", "odds_open_d", "odds_open_a"]].to_numpy(dtype=float)
    close_odds = merged[["odds_close_h", "odds_close_d", "odds_close_a"]].to_numpy(dtype=float)

    edges = probs * open_odds - 1.0
    pick = edges.argmax(axis=1)
    rows = np.arange(len(merged))
    edge = edges[rows, pick]
    price = open_odds[rows, pick]
    outcome = outcome_index(merged)
    won = outcome == pick

    merged = merged.assign(
        pick=pick,
        pick_label=np.array(["H", "D", "A"])[pick],
        edge=edge,
        price=price,
        won=won,
        pnl=np.where(won, price - 1.0, -1.0),
        close_price=close_odds[rows, pick],
    )
    # Beat-the-close: implied probability of the closing line minus the price taken.
    with np.errstate(divide="ignore", invalid="ignore"):
        merged["clv"] = 1.0 / merged["close_price"] - 1.0 / merged["price"]
    return merged.loc[merged["edge"] > min_edge].reset_index(drop=True)


def summarise(bets: pd.DataFrame, n_matches: int) -> dict[str, float]:
    if bets.empty:
        return {"bets": 0, "staked": 0.0, "profit": 0.0, "roi": float("nan"),
                "hit_rate": float("nan"), "avg_edge": float("nan"), "clv": float("nan"),
                "share": 0.0}
    staked = float(len(bets))
    profit = float(bets["pnl"].sum())
    return {
        "bets": len(bets),
        "staked": staked,
        "profit": profit,
        "roi": profit / staked,
        "hit_rate": float(bets["won"].mean()),
        "avg_edge": float(bets["edge"].mean()),
        "clv": float(bets["clv"].mean()),
        "share": len(bets) / max(n_matches, 1),
    }


def season_summary(bets: pd.DataFrame, preds: pd.DataFrame, model: str = "blend") -> pd.DataFrame:
    totals = preds.loc[preds["model"] == model].groupby("season").size()
    rows = []
    for season, block in bets.groupby("season"):
        row = {"season": season}
        row.update(summarise(block, int(totals.get(season, 0))))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)
