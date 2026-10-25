"""How much of the market can a model replicate?

The whole project so far asks "can the model beat the closing line" — answer: no.
The sharper question is how much of the gap is *reachable*.  Anchoring the model
on the de-vigged **opening** price answers it, because the opening line is
legitimate pre-match information (it exists hours before kick-off, and it is not
the benchmark being scored).

That gives a clean ladder:

    model alone  >  model + opening line  >  closing line

and the distance between the middle and the right is the part of the market's
knowledge that no public feature set reproduces — roughly the information that
arrives between the opening and closing prices (line-ups, team news, money).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.base import outcome_index
from ..odds import implied_probs
from .metrics import log_loss

WEIGHT_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def opening_probabilities(features: pd.DataFrame) -> pd.DataFrame:
    odds = features[["odds_open_h", "odds_open_d", "odds_open_a"]].to_numpy(dtype=float)
    probs = implied_probs(odds)
    return pd.DataFrame(probs, columns=["open_h", "open_d", "open_a"], index=features.index)


def _mix(model: np.ndarray, prior: np.ndarray, weight: float) -> np.ndarray:
    mixed = (1.0 - weight) * model + weight * prior
    return mixed / mixed.sum(axis=1, keepdims=True)


def best_weight(model: np.ndarray, prior: np.ndarray, y: np.ndarray) -> float:
    """Grid search over the mix weight; one parameter, so a grid is enough."""
    scores = [log_loss(_mix(model, prior, w), y) for w in WEIGHT_GRID]
    return float(WEIGHT_GRID[int(np.argmin(scores))])


def run_prior_study(features: pd.DataFrame, preds: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Walk-forward: the weight is tuned on the previous season, never the current one."""
    opens = opening_probabilities(features).reset_index().rename(columns={"index": "row"})
    opens["match_id"] = features["match_id"].to_numpy()

    model = preds.loc[preds["model"] == "blend", ["match_id", "season", "ftr", "p_home", "p_draw", "p_away"]]
    closing = preds.loc[preds["model"] == "market", ["match_id", "p_home", "p_draw", "p_away"]]
    closing = closing.rename(columns={"p_home": "c_h", "p_draw": "c_d", "p_away": "c_a"})

    data = model.merge(closing, on="match_id", how="inner").merge(
        opens[["match_id", "open_h", "open_d", "open_a"]], on="match_id", how="inner"
    )
    data = data.dropna(subset=["open_h", "c_h"])

    rows = []
    model_probs, open_probs, close_probs, hybrid_probs, ys = [], [], [], [], []
    weight_used: list[float] = []

    for season in sorted(data["season"].unique()):
        block = data.loc[data["season"] == season]
        p_model = block[["p_home", "p_draw", "p_away"]].to_numpy()
        p_open = block[["open_h", "open_d", "open_a"]].to_numpy()
        p_close = block[["c_h", "c_d", "c_a"]].to_numpy()
        y = outcome_index(block)

        history = data.loc[data["season"] < season]
        weight = 0.5
        if len(history) >= 300:
            weight = best_weight(
                history[["p_home", "p_draw", "p_away"]].to_numpy(),
                history[["open_h", "open_d", "open_a"]].to_numpy(),
                outcome_index(history),
            )
        p_hybrid = _mix(p_model, p_open, weight)

        rows.append({
            "season": season,
            "n": len(block),
            "weight": weight,
            "model": log_loss(p_model, y),
            "opening": log_loss(p_open, y),
            "hybrid": log_loss(p_hybrid, y),
            "closing": log_loss(p_close, y),
        })
        model_probs.append(p_model)
        open_probs.append(p_open)
        close_probs.append(p_close)
        hybrid_probs.append(p_hybrid)
        ys.append(y)
        weight_used.append(weight)

    table = pd.DataFrame(rows)
    if table.empty:
        return table, {}

    from .walk_forward import bootstrap_delta

    y_all = np.concatenate(ys)
    p_model = np.vstack(model_probs)
    p_open = np.vstack(open_probs)
    p_close = np.vstack(close_probs)
    p_hybrid = np.vstack(hybrid_probs)

    summary = {
        "n": int(len(y_all)),
        "model": log_loss(p_model, y_all),
        "opening": log_loss(p_open, y_all),
        "hybrid": log_loss(p_hybrid, y_all),
        "closing": log_loss(p_close, y_all),
        "median_weight": float(np.median(weight_used)),
        "hybrid_vs_model": bootstrap_delta(p_hybrid, p_model, y_all),
        "hybrid_vs_closing": bootstrap_delta(p_hybrid, p_close, y_all),
        "opening_vs_closing": bootstrap_delta(p_open, p_close, y_all),
    }

    gap = summary["model"] - summary["closing"]
    recovered = (summary["model"] - summary["hybrid"]) / gap if gap > 0 else float("nan")
    summary["gap_closed"] = float(recovered)
    return table, summary
