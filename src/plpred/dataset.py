"""Cached dataset assembly: matches -> Elo -> form features."""

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


def build_dataset(*, force: bool = False) -> pd.DataFrame:
    """Full feature table used by every model."""
    matches = load_matches(force=force)
    with_elo = add_elo(matches)
    features = add_form_features(with_elo)
    features.to_parquet(FEATURES_PARQUET, index=False)
    return features


def load_dataset() -> pd.DataFrame:
    if FEATURES_PARQUET.exists():
        return pd.read_parquet(FEATURES_PARQUET)
    return build_dataset()
