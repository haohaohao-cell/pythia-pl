"""Walk-forward backtesting.

One rule: a match is forecast by models fitted only on matches that kicked off
earlier.  Concretely, every test season is predicted by refitting on the entire
history that ended before that season's opening day.  This is slower than random
K-fold cross-validation and it is the only honest option for time series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FIRST_TEST_SEASON, season_boundaries
from ..odds import market_probs
from ..models.base import outcome_index
from ..models.baseline import BaseRates
from ..models.blend import tune_weights
from ..models.dixon_coles import DixonColes
from ..models.elo_logit import EloLogit
from ..models.gbdt import GradientBoost
from .metrics import log_loss

MODEL_ORDER = ("base_rates", "elo_logit", "dixon_coles", "gbdt", "blend", "market")


def _predict_frame(model, frame: pd.DataFrame) -> np.ndarray:
    probs = np.asarray(model.predict_proba(frame), dtype=float)
    probs = np.clip(probs, 1e-9, 1.0)
    return probs / probs.sum(axis=1, keepdims=True)


def test_seasons_from(features: pd.DataFrame, first: str = FIRST_TEST_SEASON) -> list[str]:
    seasons = sorted(features["season"].unique())
    return [s for s in seasons if s >= first]


def run_backtest(
    features: pd.DataFrame,
    seasons: list[str] | None = None,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return tidy predictions: one row per (match, model)."""
    df = features.sort_values(["date", "home"]).reset_index(drop=True)
    seasons = seasons or test_seasons_from(df)

    records: list[pd.DataFrame] = []

    for season in seasons:
        start, end = season_boundaries(season)
        train = df.loc[df["date"] < start]
        test = df.loc[(df["season"] == season) & (df["date"] >= start) & (df["date"] < end)]
        if test.empty:
            continue
        if len(train) < 200:
            raise ValueError(f"not enough history before {season}: {len(train)} matches")

        if verbose:
            print(f"  [{season}] train={len(train):4d} test={len(test):3d}", flush=True)

        fitted: dict[str, object] = {}
        for name, model in (
            ("elo_logit", EloLogit()),
            ("dixon_coles", DixonColes()),
            ("gbdt", GradientBoost()),
        ):
            fitted[name] = model.fit(train)

        fitted["base_rates"] = BaseRates().fit(train)

        for name, model in fitted.items():
            probs = _predict_frame(model, test)
            part = test[["match_id", "season", "date", "home", "away", "ftr"]].copy()
            part["model"] = name
            part[["p_home", "p_draw", "p_away"]] = probs
            records.append(part)

        # Market benchmark: de-vigged closing line (opening as fallback).
        mkt = market_probs(test)
        part = test[["match_id", "season", "date", "home", "away", "ftr"]].copy()
        part["model"] = "market"
        part[["p_home", "p_draw", "p_away"]] = mkt[["mkt_h", "mkt_d", "mkt_a"]].to_numpy()
        part["market_src"] = mkt["market_src"].to_numpy()
        records.append(part)

    preds = pd.concat(records, ignore_index=True)
    preds = preds.dropna(subset=["p_home", "p_draw", "p_away"]).reset_index(drop=True)
    preds = _add_blend(preds)
    return preds


def _add_blend(
    preds: pd.DataFrame,
    components: tuple[str, ...] = ("elo_logit", "dixon_coles", "gbdt"),
    *,
    window: int = 2,
    shrink: float = 0.5,
) -> pd.DataFrame:
    """Blend the three fitted models with weights estimated on past seasons only.

    Three design choices, each made *before* looking at the leaderboard:

    * the weights come from the two preceding seasons (760 matches) rather than
      one, because one season of 380 matches estimates a 3-simplex too noisily;
    * they are shrunk halfway toward equal weights, a standard bias-variance
      trade-off for unstable stacking estimates;
    * the first season falls back to equal weights, which is disclosed.

    ``reports/README`` shows the ablation: equal weights score 0.9770 and the
    tuned variants 0.9767-0.9803, i.e. the choice is inside the noise.
    """
    wide = (
        preds.pivot_table(index="match_id", columns="model", values=["p_home", "p_draw", "p_away"])
        .sort_index()
    )
    meta = preds.drop_duplicates("match_id").set_index("match_id")[["season", "date", "home", "away", "ftr"]]
    wide = pd.concat([wide, meta], axis=1)

    def matrix(block: pd.DataFrame, name: str) -> np.ndarray:
        return np.column_stack([
            block[("p_home", name)].to_numpy(),
            block[("p_draw", name)].to_numpy(),
            block[("p_away", name)].to_numpy(),
        ])

    equal = {n: 1.0 / len(components) for n in components}
    rows = []
    seasons = sorted(wide["season"].unique())

    for i, season in enumerate(seasons):
        block = wide.loc[wide["season"] == season]
        weights = dict(equal)
        history = seasons[max(0, i - window) : i]
        if history:
            prev = wide.loc[wide["season"].isin(history)]
            if len(prev) >= 200:
                tuned = tune_weights({n: matrix(prev, n) for n in components}, outcome_index(prev))
                weights = {n: shrink * equal[n] + (1 - shrink) * tuned[n] for n in components}
                total = sum(weights.values())
                weights = {n: w / total for n, w in weights.items()}

        blended = np.zeros((len(block), 3))
        for name in components:
            blended += weights[name] * matrix(block, name)
        blended = blended / blended.sum(axis=1, keepdims=True)

        part = block[["season", "date", "home", "away", "ftr"]].reset_index()
        part["model"] = "blend"
        part[["p_home", "p_draw", "p_away"]] = blended
        part["weights"] = ",".join(f"{n}={weights[n]:.2f}" for n in components)
        rows.append(part)

    blend = pd.concat(rows, ignore_index=True)[
        ["match_id", "season", "date", "home", "away", "ftr", "model", "p_home", "p_draw", "p_away", "weights"]
    ]
    blend = blend.drop_duplicates("match_id")
    return pd.concat([preds, blend], ignore_index=True)


def leaderboard(preds: pd.DataFrame) -> pd.DataFrame:
    """Overall metrics per model, ordered by log loss."""
    from .metrics import evaluate

    rows = []
    for model, block in preds.groupby("model"):
        y = outcome_index(block)
        probs = block[["p_home", "p_draw", "p_away"]].to_numpy()
        row = {"model": model, "n": len(block)}
        row.update(evaluate(probs, y))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    return out


def season_table(preds: pd.DataFrame) -> pd.DataFrame:
    """Per-season log loss, the table that reveals whether a model is stable."""
    rows = []
    for (season, model), block in preds.groupby(["season", "model"]):
        rows.append({
            "season": season,
            "model": model,
            "n": len(block),
            "log_loss": log_loss(block[["p_home", "p_draw", "p_away"]].to_numpy(), outcome_index(block)),
        })
    return pd.DataFrame(rows).pivot(index="season", columns="model", values="log_loss")


def bootstrap_delta(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap of log-loss difference (A minus B): mean and 95% CI.

    Negative means A is better.  Overlapping zero means the gap is not
    distinguishable from luck at this sample size — which is most of them.
    """
    rng = np.random.default_rng(seed)
    probs_a = np.asarray(probs_a, float)
    probs_b = np.asarray(probs_b, float)
    y = np.asarray(y, int)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = log_loss(probs_a[idx], y[idx]) - log_loss(probs_b[idx], y[idx])
    return {
        "delta": float(log_loss(probs_a, y) - log_loss(probs_b, y)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_better": float(np.mean(diffs < 0)),
    }
