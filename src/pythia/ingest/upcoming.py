"""The upcoming round: FPL for the calendar, football-data for the prices.

Neither source is sufficient alone.  football-data's ``fixtures.csv`` carries
Pinnacle opening prices but frequently lists no Premier League matches at all
(it only contains the divisions playing in the next few days).  The FPL endpoint
always knows the schedule but has no odds.  So the calendar comes from FPL and
the prices are joined on where they exist — and only when the two feeds agree
on roughly the same date, so a stale price list cannot attach last month's
prices to this week's fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import fpl
from .fixtures_feed import load_fixtures

PRICE_COLUMNS = ["odds_h_h", "odds_h_d", "odds_h_a"]
OUT_COLUMNS = ["kickoff", "home", "away", *PRICE_COLUMNS, "odds_src"]
MAX_DATE_DRIFT_DAYS = 7


def _prices_from_football_data() -> pd.DataFrame:
    try:
        priced = load_fixtures()
    except Exception:  # noqa: BLE001 - the price feed is optional here
        return pd.DataFrame(columns=["home", "away", "price_kickoff", *PRICE_COLUMNS, "odds_src"])
    if priced.empty:
        return pd.DataFrame(columns=["home", "away", "price_kickoff", *PRICE_COLUMNS, "odds_src"])
    return priced[["home", "away", "kickoff", *PRICE_COLUMNS, "odds_src"]].rename(
        columns={"kickoff": "price_kickoff"}
    )


def _blank() -> pd.DataFrame:
    return pd.DataFrame(columns=OUT_COLUMNS)


def load_upcoming(*, prefer: str = "fpl") -> pd.DataFrame:
    """Return the next round with opening prices attached where available."""
    prices = _prices_from_football_data()

    if prefer == "fpl":
        try:
            block = fpl.next_round_fixtures(fpl.fetch_fixtures())
        except Exception:  # noqa: BLE001 - fall back to the flat feed
            block = pd.DataFrame()

        if not block.empty:
            merged = block.merge(prices, on=["home", "away"], how="left")
            if "price_kickoff" in merged.columns:
                # Coerce both sides: an all-NaN price column arrives as float.
                kickoff = pd.to_datetime(merged["kickoff"], errors="coerce", utc=True)
                price_kickoff = pd.to_datetime(merged["price_kickoff"], errors="coerce", utc=True)
                drift = (kickoff - price_kickoff).abs().dt.total_seconds() / 86400.0
                stale = ~(drift <= MAX_DATE_DRIFT_DAYS)
                stale = stale.fillna(True)
                merged.loc[stale, PRICE_COLUMNS] = np.nan
                merged.loc[stale, "odds_src"] = ""
            else:
                for col in PRICE_COLUMNS:
                    merged[col] = np.nan
                merged["odds_src"] = ""
            merged["odds_src"] = merged["odds_src"].fillna("")
            return merged[OUT_COLUMNS].sort_values("kickoff").reset_index(drop=True)

    if not prices.empty:
        fallback = prices.rename(columns={"price_kickoff": "kickoff"})
        return fallback[OUT_COLUMNS].sort_values("kickoff").reset_index(drop=True)
    return _blank()
