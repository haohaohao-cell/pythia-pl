"""Odds helpers: de-vigging and the market benchmark.

A bookmaker's decimal odds embed a margin (overround).  For a fair comparison
against a model we must remove it; proportional normalisation is the standard,
transparent choice and is what this project reports by default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def implied_probs(odds: np.ndarray) -> np.ndarray:
    """Convert decimal odds to normalised probabilities.

    ``odds`` has shape ``(..., 3)`` ordered home/draw/away.  Rows containing
    non-positive or missing prices become ``nan`` so callers can drop them.
    """
    odds = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 1.0 / odds
        total = raw.sum(axis=-1, keepdims=True)
        probs = raw / total
    probs = np.where(np.isfinite(probs) & (total > 0) & (odds > 1.0).all(axis=-1, keepdims=True), probs, np.nan)
    return probs


def overround(odds: np.ndarray) -> np.ndarray:
    """Bookmaker margin: sum of implied probabilities minus one."""
    odds = np.asarray(odds, dtype=float)
    return (1.0 / odds).sum(axis=-1) - 1.0


def market_probs(df: pd.DataFrame, *, column: str = "close") -> pd.DataFrame:
    """Attach de-vigged market probabilities to a match frame.

    Falls back from closing to opening prices when a closing line is missing,
    and records which one was used in ``market_src``.
    """
    close = df[[f"odds_{column}_h", f"odds_{column}_d", f"odds_{column}_a"]].to_numpy(float)
    probs = implied_probs(close)
    used = np.full(len(df), column, dtype=object)

    missing = ~np.isfinite(probs).all(axis=1)
    if missing.any():
        other = "open" if column == "close" else "close"
        alt = implied_probs(df[[f"odds_{other}_h", f"odds_{other}_d", f"odds_{other}_a"]].to_numpy(float))
        probs[missing] = alt[missing]
        used[missing] = other

    out = pd.DataFrame(probs, columns=["mkt_h", "mkt_d", "mkt_a"], index=df.index)
    out["market_src"] = used
    return out
