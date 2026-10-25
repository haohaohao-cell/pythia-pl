"""Project-wide configuration: paths, seasons and modelling constants."""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
DOCS_DIR = ROOT / "docs"

for _d in (RAW_DIR, PROCESSED_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MATCHES_PARQUET = PROCESSED_DIR / "matches.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"
PREDICTIONS_PARQUET = PROCESSED_DIR / "backtest_predictions.parquet"
LEADERBOARD_CSV = REPORTS_DIR / "leaderboard.csv"

# --------------------------------------------------------------------------- #
# Data source
# --------------------------------------------------------------------------- #
BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
DIVISION = "E0"

# Seasons available from football-data.co.uk, oldest first (start year -> code).
FIRST_SEASON_START = 2014
LAST_SEASON_START = 2026

#: Outcomes in the canonical column order used everywhere in this project.
OUTCOMES = ("H", "D", "A")
OUTCOME_INDEX = {o: i for i, o in enumerate(OUTCOMES)}

#: The first season used for evaluation. Earlier seasons are training-only.
FIRST_TEST_SEASON = "2018-19"

#: A match needs at least one bookmaker line to serve in the market benchmark.
MIN_ODDS_COVERAGE = 0.90

# --------------------------------------------------------------------------- #
# Model constants (documented in README, tune with care)
# --------------------------------------------------------------------------- #
ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 60.0
ELO_SEASON_REGRESSION = 1.0 / 3.0  # shrink 1/3 of the way back to 1500 each summer
ELO_NEW_TEAM = 1450.0

#: Half-life (days) of the Dixon-Coles time-decay weight.
DC_HALF_LIFE_DAYS = 540.0
DC_MAX_GOALS = 10

#: Rolling window length (matches) for form features.
FORM_WINDOW = 5
FORM_EWM_HALFLIFE = 4.0

RANDOM_STATE = 42


def season_code(start_year: int) -> str:
    """2014 -> '1415' (the football-data.co.uk URL fragment)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """2014 -> '2014-15'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def season_start_year(label: str) -> int:
    """'2014-15' -> 2014."""
    return int(str(label).split("-")[0])


ALL_SEASONS = [
    season_label(y) for y in range(FIRST_SEASON_START, LAST_SEASON_START + 1)
]
SEASON_CODES = {season_label(y): season_code(y) for y in range(FIRST_SEASON_START, LAST_SEASON_START + 1)}


def season_boundaries(season: str) -> tuple["object", "object"]:
    """Return (start, end) timestamps delimiting a season label."""
    import pandas as pd

    start_year = season_start_year(season)
    start = pd.Timestamp(f"{start_year}-07-01")
    end = pd.Timestamp(f"{start_year + 1}-07-01")
    return start, end
