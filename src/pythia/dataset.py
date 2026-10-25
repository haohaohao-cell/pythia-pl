"""Cached dataset assembly: matches -> (xG) -> Elo -> form features."""

from __future__ import annotations

import pandas as pd

from .config import FEATURES_PARQUET, MATCHES_PARQUET
from .features.elo import add_elo
from .features.rolling import add_form_features
from .ingest.football_data import build_matches


def load_matches(*, force: bool = False) -> pd.DataFrame:
    """Canonical match table, cached as parquet."""
    if MATCHES_PARQUET.exists() and not force:
        return pd.read_parquet(MATCHES_PARQUET)
    matches = build_matches(force=force)
    matches.to_parquet(MATCHES_PARQUET, index=False)
    return matches


def attach_xg(matches: pd.DataFrame) -> pd.DataFrame:
    """Merge Understat xG when the local cache exists; otherwise leave it missing."""
    from .ingest import understat

    if not understat.CACHE_PATH.exists():
        return matches
    return understat.merge_into_matches(matches, understat.load())


def build_dataset(*, force: bool = False) -> pd.DataFrame:
    """Full feature table used by every model.

    xG is optional: when ``data/raw/understat_xg.parquet`` is absent the columns
    are simply missing, the xG features fall back to NaN, and every non-xG model
    behaves exactly as before.
    """
    matches = load_matches(force=force)
    matches = attach_xg(matches)
    with_elo = add_elo(matches)
    features = add_form_features(with_elo)
    features.to_parquet(FEATURES_PARQUET, index=False)
    return features


def load_dataset() -> pd.DataFrame:
    if FEATURES_PARQUET.exists():
        return pd.read_parquet(FEATURES_PARQUET)
    return build_dataset()
