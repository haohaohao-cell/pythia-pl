"""Post-hoc probability calibration.

The models output three-way probabilities fitted by maximum likelihood, which
does not guarantee that a stated 30% actually happens 30% of the time.  A single
temperature parameter ``T`` rescales the probabilities:

    p_i  ->  p_i ** (1/T) / sum_j p_j ** (1/T)

``T = 1`` is the identity, ``T > 1`` flattens an over-confident model and
``T < 1`` sharpens an under-confident one.  One parameter cannot overfit 380
matches, which is the point: it is a bias correction, not a second model.

Crucially, T is fitted on the *previous* season's out-of-sample forecasts, so it
is causal with respect to the season being scored.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from ..eval.metrics import log_loss

MIN_T = 0.5
MAX_T = 3.0


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    scaled = probs ** (1.0 / temperature)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    """Return the T that minimises log loss on held-out forecasts."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(probs) == 0:
        return 1.0

    def objective(t: float) -> float:
        return log_loss(apply_temperature(probs, t), y)

    res = minimize_scalar(objective, bounds=(MIN_T, MAX_T), method="bounded")
    return float(res.x) if res.success else 1.0


def calibrate_by_season(
    preds,
    models: tuple[str, ...] = ("base_rates", "elo_logit", "dixon_coles", "gbdt", "blend"),
):
    """Attach ``<model>+T`` rows whose temperature was fitted on the prior season.

    Returns ``(augmented_predictions, lambdas)`` where ``lambdas`` maps
    ``(model, season) -> T`` for transparency in the report.
    """
    import pandas as pd

    from ..models.base import outcome_index

    rows = []
    lambdas: dict[tuple[str, str], float] = {}

    for model in models:
        block = preds.loc[preds["model"] == model]
        if block.empty:
            continue
        seasons = sorted(block["season"].unique())
        for i, season in enumerate(seasons):
            current = block.loc[block["season"] == season]
            temperature = 1.0
            if i > 0:
                previous = block.loc[block["season"] == seasons[i - 1]]
                if len(previous) >= 100:
                    temperature = fit_temperature(
                        previous[["p_home", "p_draw", "p_away"]].to_numpy(),
                        outcome_index(previous),
                    )
            lambdas[(model, season)] = temperature

            part = current.copy()
            part["model"] = f"{model}+T"
            part[["p_home", "p_draw", "p_away"]] = apply_temperature(
                current[["p_home", "p_draw", "p_away"]].to_numpy(), temperature
            )
            rows.append(part)

    if not rows:
        return preds, lambdas
    return pd.concat([preds] + rows, ignore_index=True), lambdas
