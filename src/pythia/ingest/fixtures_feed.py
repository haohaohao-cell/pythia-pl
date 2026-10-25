"""Upcoming fixtures from football-data.co.uk.

The results files only contain matches that have already been played, so the
weekly forecast needs a second feed (``fixtures.csv``).  It carries a single
round of every division, opening prices from several books, and the 2.5-goal and
Asian-handicap lines — no closing prices, because those exist only after kick-off.
"""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from ..config import DIVISION, RAW_DIR
from ..teams import team_key
from .football_data import _parse_dates, _pick_families

FIXTURE_URL = "https://www.football-data.co.uk/fixtures.csv"
FIXTURE_PATH = RAW_DIR / "fixtures.csv"

#: Kick-off times are published in UK local time, DST included.
UK_TZ = ZoneInfo("Europe/London")

#: Opening prices for a match that has not kicked off yet.
FIXTURE_FAMILIES: tuple[tuple[str, str, str], ...] = (
    ("PPH", "PPD", "PPA"),
    ("PSH", "PSD", "PSA"),
    ("B365H", "B365D", "B365A"),
    ("AvgH", "AvgD", "AvgA"),
)


def download_fixtures(*, force: bool = False, timeout: int = 60) -> Path:
    if FIXTURE_PATH.exists() and not force and FIXTURE_PATH.stat().st_size > 5_000:
        return FIXTURE_PATH
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.get(FIXTURE_URL, timeout=timeout, headers={"User-Agent": "pythia-pl/0.1"})
            resp.raise_for_status()
            text = resp.content.decode("utf-8-sig")
            FIXTURE_PATH.write_text(text, encoding="utf-8")
            return FIXTURE_PATH
        except Exception as exc:  # noqa: BLE001 - retried below
            last_err = exc
    raise RuntimeError(f"could not download fixtures: {last_err}")


def parse_fixtures(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise one round of fixtures onto the project's schema."""
    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.loc[raw["Div"].astype(str).str.strip() == DIVISION]
    if raw.empty:
        return pd.DataFrame(columns=["home", "away", "kickoff"])

    out = pd.DataFrame(index=raw.index)
    out["date"] = _parse_dates(raw["Date"])
    out["time"] = raw["Time"].astype(str).str.strip() if "Time" in raw.columns else "15:00"
    out["home_raw"] = raw["HomeTeam"].astype(str).str.strip()
    out["away_raw"] = raw["AwayTeam"].astype(str).str.strip()
    out["home"] = out["home_raw"].map(team_key)
    out["away"] = out["away_raw"].map(team_key)

    clock = pd.to_datetime(out["time"], format="%H:%M", errors="coerce").dt.time
    clock = clock.fillna(pd.to_datetime("15:00", format="%H:%M").time())
    naive = pd.to_datetime([
        pd.Timestamp(day) + pd.Timedelta(hours=t.hour, minutes=t.minute)
        for day, t in zip(out["date"], clock)
    ])
    # football-data publishes UK local time.  Localising here (rather than
    # comparing against the machine clock) keeps "has this kicked off?" correct
    # on any runner, including the BST/GMT switch at the end of October.
    out["kickoff"] = naive.tz_localize(
        UK_TZ, ambiguous="raise", nonexistent="shift_forward"
    )

    picked = _pick_families(raw, FIXTURE_FAMILIES, 0.9)
    for slot, side in enumerate(("h", "d", "a")):
        out[f"odds_h_{side}"] = pd.to_numeric(raw[picked[slot]], errors="coerce") if picked else float("nan")
    out["odds_src"] = "+".join(picked) if picked else ""

    for src, dst in (("B365>2.5", "ou_over"), ("B365<2.5", "ou_under")):
        out[dst] = pd.to_numeric(raw[src], errors="coerce") if src in raw.columns else float("nan")

    out = out.dropna(subset=["date", "home", "away"]).dropna(subset=["kickoff"])
    return out.sort_values("kickoff").reset_index(drop=True)


def load_fixtures(*, force: bool = False) -> pd.DataFrame:
    path = download_fixtures(force=force)
    raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig", on_bad_lines="skip")
    return parse_fixtures(raw)
