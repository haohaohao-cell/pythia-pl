"""Does xG actually buy anything?

One question, one protocol: same training window, same test matches, same
hyper-parameters — the only difference is whether the tree model sees the xG
block.  The blend comparison holds the other two components fixed, so the
number measures the xG contribution and nothing else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import season_boundaries
from ..features.rolling import XG_FEATURE_COLUMNS
from ..models.base import outcome_index
from ..models.gbdt import GradientBoost
from .metrics import log_loss


def has_xg(features: pd.DataFrame) -> bool:
    return all(col in features.columns and features[col].notna().mean() > 0.9 for col in XG_FEATURE_COLUMNS[:4])


def _blend_of(elo: np.ndarray, dc: np.ndarray, gbdt: np.ndarray) -> np.ndarray:
    mixed = (elo + dc + gbdt) / 3.0
    return mixed / mixed.sum(axis=1, keepdims=True)


def run_ablation(features: pd.DataFrame, preds: pd.DataFrame, seasons: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (per-season table, overall summary)."""
    df = features.sort_values(["date", "home"]).reset_index(drop=True)
    seasons = seasons or sorted(preds["season"].unique())

    records: dict[str, list[np.ndarray]] = {
        "gbdt": [], "gbdt_xg": [], "blend": [], "blend_xg": [],
    }
    outcomes: list[np.ndarray] = []
    rows = []

    for season in seasons:
        start, end = season_boundaries(season)
        train = df.loc[df["date"] < start]
        test = df.loc[(df["season"] == season) & (df["date"] >= start) & (df["date"] < end)]
        if test.empty or train.empty:
            continue

        base = GradientBoost().fit(train)
        with_xg = GradientBoost(use_xg=True).fit(train)
        p_base = np.asarray(base.predict_proba(test), dtype=float)
        p_xg = np.asarray(with_xg.predict_proba(test), dtype=float)

        # The other two blend components are taken from the backtest, so the
        # only moving part is the tree model.
        elo = preds.loc[(preds["season"] == season) & (preds["model"] == "elo_logit")].set_index("match_id")
        dc = preds.loc[(preds["season"] == season) & (preds["model"] == "dixon_coles")].set_index("match_id")
        ids = test["match_id"].to_numpy()
        try:
            elo_p = elo.loc[ids, ["p_home", "p_draw", "p_away"]].to_numpy()
            dc_p = dc.loc[ids, ["p_home", "p_draw", "p_away"]].to_numpy()
        except KeyError:  # pragma: no cover - backtest and features must align
            continue

        blend = _blend_of(elo_p, dc_p, p_base)
        blend_xg = _blend_of(elo_p, dc_p, p_xg)
        y = outcome_index(test)

        for name, probs in (("gbdt", p_base), ("gbdt_xg", p_xg), ("blend", blend), ("blend_xg", blend_xg)):
            records[name].append(probs)

        rows.append({
            "season": season,
            "n": len(test),
            "gbdt": log_loss(p_base, y),
            "gbdt_xg": log_loss(p_xg, y),
            "blend": log_loss(blend, y),
            "blend_xg": log_loss(blend_xg, y),
            "d_gbdt": log_loss(p_xg, y) - log_loss(p_base, y),
            "d_blend": log_loss(blend_xg, y) - log_loss(blend, y),
        })
        outcomes.append(y)

    table = pd.DataFrame(rows)
    y_all = np.concatenate(outcomes) if outcomes else np.array([], dtype=int)
    summary = {"n": int(len(y_all))}
    for name, chunks in records.items():
        if chunks:
            probs = np.vstack(chunks)
            summary[name] = log_loss(probs, y_all)

    if "blend" in summary and "blend_xg" in summary:
        from .walk_forward import bootstrap_delta

        summary["delta_blend"] = bootstrap_delta(
            np.vstack(records["blend_xg"]), np.vstack(records["blend"]), y_all
        )
    return table, summary
