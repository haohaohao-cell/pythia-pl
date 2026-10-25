"""Expected goals from Understat (2014-15 onwards).

xG is the strongest public signal for "how well did a team actually play",
independent of whether the ball went in. It is the obvious feature this project
was missing, and the ablation in ``eval/ablation.py`` asks whether it buys
anything.

``soccerdata`` is an optional dependency: Understat sits behind Cloudflare, and
the package handles the TLS fingerprinting that a plain request cannot. Install
it with ``pip install -e ".[xg]"``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import RAW_DIR, season_label
from ..teams import team_key

CACHE_PATH = RAW_DIR / "understat_xg.parquet"
FIRST_SEASON = 2014
LAST_SEASON = 2026

COLUMNS = ["season", "date", "home", "away", "home_xg", "away_xg", "home_goals", "away_goals"]


def _season_candidates(start_year: int) -> list[object]:
    """Understat ids changed shape over the years: 2014..2020, then '2122' style."""
    code = f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"
    return [code, start_year]


def _build_frame(raw: pd.DataFrame, start_year: int) -> pd.DataFrame:
    frame = pd.DataFrame({
        "season": season_label(start_year),
        "date": pd.to_datetime(raw["date"], errors="coerce").dt.normalize(),
        "home": raw["home_team"].map(team_key),
        "away": raw["away_team"].map(team_key),
        "home_xg": pd.to_numeric(raw["home_xg"], errors="coerce"),
        "away_xg": pd.to_numeric(raw["away_xg"], errors="coerce"),
        "home_goals": pd.to_numeric(raw["home_goals"], errors="coerce"),
        "away_goals": pd.to_numeric(raw["away_goals"], errors="coerce"),
    })
    return frame.loc[frame["home_xg"].notna() & frame["away_xg"].notna()]


def _fetch_one(start_year: int) -> pd.DataFrame:
    try:
        import soccerdata as sd
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            'xG features need the optional dependency: pip install -e ".[xg]"'
        ) from exc

    # The id is not reliable on its own: Understat used the start year until
    # 2020 and four-digit codes afterwards, and both 2020 and 2021 resolve to
    # 2020-21. So the identifier is a candidate and the kick-off dates decide.
    window_start = pd.Timestamp(f"{start_year}-07-15")
    window_end = pd.Timestamp(f"{start_year + 1}-08-01")
    last_err: Exception | None = None

    for candidate in _season_candidates(start_year):
        try:
            understat = sd.Understat(leagues="ENG-Premier League", seasons=[candidate])
            raw = understat.read_schedule().reset_index()
        except Exception as exc:  # noqa: BLE001 - try the next id
            last_err = exc
            continue

        dates = pd.to_datetime(raw["date"], errors="coerce")
        if dates.notna().any() and window_start <= dates.min() <= window_end:
            return _build_frame(raw, start_year)
        last_err = ValueError(f"season id {candidate!r} covered {dates.min()} to {dates.max()}")

    raise RuntimeError(f"no Understat data for {season_label(start_year)}: {last_err}")


def fetch(*, force: bool = False, seasons: list[int] | None = None, verbose: bool = False) -> pd.DataFrame:
    """Download (or reuse) per-match xG for every season, cached as parquet."""
    if CACHE_PATH.exists() and not force:
        return pd.read_parquet(CACHE_PATH)

    years = seasons or list(range(FIRST_SEASON, LAST_SEASON + 1))
    frames = []
    for year in years:
        try:
            part = _fetch_one(year)
        except Exception as exc:  # noqa: BLE001 - one bad season should not sink the run
            if verbose:
                print(f"  {season_label(year)} failed: {exc}")
            continue
        if verbose:
            print(f"  {season_label(year)}: {len(part)} matches")
        frames.append(part)

    if not frames:  # pragma: no cover - network failure path
        raise RuntimeError("no Understat seasons could be downloaded")

    data = pd.concat(frames, ignore_index=True).drop_duplicates(["season", "home", "away"])
    data = data.sort_values(["date", "home"]).reset_index(drop=True)
    data.to_parquet(CACHE_PATH, index=False)
    return data


def load() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        return fetch()
    return pd.read_parquet(CACHE_PATH)


def merge_into_matches(matches: pd.DataFrame, xg: pd.DataFrame) -> pd.DataFrame:
    """Attach ``hxg``/``axg`` to the canonical match table.

    The join key is ``(season, home, away)``: each ordered pairing happens exactly
    once per season, so the join cannot fan out.  Coverage is reported by the
    caller rather than silently filled.
    """
    keys = ["season", "home", "away"]
    left = matches.drop(columns=[c for c in ("hxg", "axg") if c in matches.columns])
    merged = left.merge(xg[keys + ["home_xg", "away_xg"]], on=keys, how="left")
    merged = merged.rename(columns={"home_xg": "hxg", "away_xg": "axg"})

    # Fall back to the existing columns (football-data ships them from 2026-27).
    if "hxg" in matches.columns and "match_id" in matches.columns:
        fallback = matches.set_index("match_id")[["hxg", "axg"]]
        merged = merged.join(fallback, on="match_id", rsuffix="_fb")
        merged["hxg"] = merged["hxg"].fillna(merged.pop("hxg_fb"))
        merged["axg"] = merged["axg"].fillna(merged.pop("axg_fb"))
    return merged
